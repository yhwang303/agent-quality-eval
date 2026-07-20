"""
cot_otel_enricher
=================

把已经提取好的 ``SessionCoT`` 按 OpenTelemetry GenAI 语义约定补全字段，**不**改写
原有任何字段——所有 OTel 视图相关的数据全部以可选字段形式新增，前端按需消费。

v0.11.2 重要变更：自动检测 model
---------------------------------
此前需要用户在 ``.env`` 里手动配 ``COT_DEFAULT_MODEL`` 才能补齐 model / provider /
cost——这本质是硬编码，违反「自动观测」初衷。本版本起改为从 ``cot-stream.js`` 实时
hook 写下的 ``events.jsonl`` 自动检测：

  Cursor 在所有 hook payload 顶层都注入了 ``model`` / ``generation_id`` /
  ``cursor_version`` / ``user_email``；``afterAgentResponse`` 还会带上真实的
  ``input_tokens`` / ``output_tokens`` / ``cache_read_tokens`` /
  ``cache_write_tokens``。``cot-stream.js`` 已经把整段 payload 落到
  ``<COT_ROOT>/output/events/<sid>/events.jsonl``，本模块只要扫这个文件就能拿到。

新优先级：``events.jsonl`` > ``COT_DEFAULT_MODEL`` env > transcript metadata > ``unknown``。
``COT_DEFAULT_MODEL`` 仅在 events.jsonl 不存在（用户没装 cot-stream hook）时作为兜底。

补全维度（对照 OTel `gen_ai.*` / OpenInference 约定）
---------------------------------------------------

§1 Trace Context
    - 每个 session 合成稳定的 ``trace_id``（128-bit hex）
    - 每个 turn 合成 ``root_span_id``（64-bit hex），父 span 为 session 假根
    - 每个 step 合成 ``span_id`` + ``parent_span_id``（指向所属 turn 的 root span）
    所有 ID 都是从 ``session_id`` + index 推导的稳定 hash，不依赖外部 tracer，
    便于前端绘制依赖树。**这只是 client-side 合成**，不能跨进程传播——但能补
    "step 之间的父子关系"这个 OTel trace 的核心可视化信号。

§2 Token & Cost
    - 复用 transcript 已有的 ``usage.input_tokens`` / ``output_tokens`` 当作权威数；
    - transcript 没给的（Cursor 大多数情况）用 char/4 启发式做 client-side 估算；
    - 按 model pricing table 算 cost（USD），找不到 model 则 cost = None；
    - 在 step / turn / session 三个粒度同步暴露。

§3 Structured Messages
    - 把 ``metadata.prompt_preview`` 重打包成 OTel ``gen_ai.input.messages`` 风格：
      ``[{role, parts:[{type:'text', content:'...'}, ...]}, ...]``。
    - 把 ``observed_output``/``content`` 拍成 ``gen_ai.output.messages``：
      ``[{role:'assistant', parts:[{type:'text', content:'...'}], finish_reason:'...'}]``。

§4 Retrieval Documents
    - 把 ``recall_preview`` 切成 ``documents[]``。如果原文里看起来是结构化（json
      array / 编号列表 / 分隔行），按结构化拆；否则按段落切，每段一个 doc。
    - 每条 doc 含 ``id``、``content``、``score=null``、``metadata``。

§5 finish_reason 归一化
    - transcript 真正给到的 ``stop_reason`` → 直接映射；
    - Cursor 推断出来的 ``inferred_final.reason`` → ``stop``；
    - 错误 step → ``error``；
    - 工具调用决策 step → ``tool_calls``；
    - 默认 ``stop``。

§6 Resource Attributes
    - 顶层 ``resource_attributes``：``service.name`` / ``service.version`` /
      ``deployment.environment`` / ``host.name`` / ``telemetry.sdk.name`` /
      ``telemetry.sdk.language``。

§7 Eval (response-verifier)
    - session_cot 已有 ``response_score``（来自 response-verifier 报告，外面注入）
      时，把它写入 ``otel_view.eval``，并尝试同步到每个 turn 的 ``eval``。
    - 没有报告时全部留 None / 空。

§8 Provider / Model
    - 尝试从 transcript 中扫到 ``message.model`` 字段；扫不到（Cursor 不带）
      则填 ``unknown``。同步推断 provider（claude → anthropic, gpt → openai）。

刻意留空（OTel 能观察、本项目暂无能力获取）
------------------------------------------
- 真实 LLM 调用参数：``temperature`` / ``top_p`` / ``max_tokens`` / ``seed`` /
  ``stop_sequences``。Cursor 不暴露给 transcript。**字段保留 None。**
- ``response.id``、``system_instructions``、``tool.definitions``、原生 reasoning
  signature。同上保留 None。
- 跨进程 ``traceparent`` 注入：本模块只做合成，不接 OTLP exporter。

后续如要接真正的 OTLP，可以在这层之上再包一个 exporter 把这些字段直接
``set_attribute`` 到真 span 上，schema 已经对齐 OTel。
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import time
from typing import Any, Dict, List, Optional, Tuple

# ─── trace context 合成 ────────────────────────────────────

_OTEL_VERSION = "0.1.0-clientside"


def _hex_id(parts: List[Any], length: int = 16) -> str:
    """从输入拍出一个稳定 hex id（默认 16 hex chars = 64-bit span id）。"""
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    h = hashlib.sha256(raw).hexdigest()
    return h[:length]


def _trace_id_for(session_id: str) -> str:
    """OTel trace id 是 32 hex chars (128-bit)。"""
    return _hex_id(["trace", session_id], length=32)


def _span_id_for(session_id: str, *suffix: Any) -> str:
    """span id 是 16 hex chars (64-bit)。"""
    return _hex_id(["span", session_id, *suffix], length=16)


# ─── token 估算 ────────────────────────────────────────────

# 简化 pricing，单位：USD per 1K tokens (input, output)
# 没有 model 命中时返回 None，由前端展示 "—"
_PRICING_USD_PER_1K: Dict[str, Tuple[float, float]] = {
    # Anthropic Claude
    "claude-opus-4": (0.015, 0.075),
    "claude-sonnet-4": (0.003, 0.015),
    "claude-3-5-sonnet": (0.003, 0.015),
    "claude-3-7-sonnet": (0.003, 0.015),
    "claude-3-haiku": (0.00025, 0.00125),
    "claude-4.6-sonnet": (0.003, 0.015),
    # Cursor 在 events.jsonl 里报的实际 model 名（v0.11.2 起自动检测）
    "claude-opus-4-7": (0.015, 0.075),
    "claude-opus-4-6": (0.015, 0.075),
    # OpenAI / Azure OpenAI
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4.1": (0.005, 0.015),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "o1": (0.015, 0.06),
    "o1-mini": (0.003, 0.012),
    # Cursor 自有模型映射（粗略对齐，仅作前端展示参考）
    "composer-2-fast": (0.0005, 0.0015),
    "gpt-5.5-medium": (0.005, 0.015),
}


def _normalize_model_key(model: Optional[str]) -> Optional[str]:
    if not model:
        return None
    m = model.lower().strip()
    # 去掉日期 / 版本 suffix：claude-3-5-sonnet-20240620 → claude-3-5-sonnet
    m = re.sub(r"-\d{8}.*$", "", m)
    m = re.sub(r"-\d{4}-\d{2}-\d{2}.*$", "", m)
    m = re.sub(r"-latest$", "", m)
    if m in _PRICING_USD_PER_1K:
        return m
    # 前缀匹配：claude-3-5-sonnet-* → claude-3-5-sonnet
    for k in _PRICING_USD_PER_1K:
        if m.startswith(k):
            return k
    return None


def _provider_from_model(model: Optional[str]) -> str:
    """Map model name → vendor / provider.

    覆盖：anthropic / openai / google / xai / moonshot / zhipu / tencent /
    deepseek / mistral / cohere / cursor。命中不了就 ``unknown``。
    """
    if not model:
        return "unknown"
    m = model.lower()
    if "claude" in m:
        return "anthropic"
    # OpenAI 家：gpt-*、o1/o3/o4 推理、codex、gpt-oss
    if (
        "gpt" in m
        or m.startswith(("o1", "o3", "o4"))
        or "codex" in m
    ):
        return "openai"
    if "gemini" in m:
        return "google"
    if "grok" in m:
        return "xai"
    if "kimi" in m or "moonshot" in m:
        return "moonshot"
    if "glm" in m or "chatglm" in m or "zhipu" in m:
        return "zhipu"
    if "hunyuan" in m or m.startswith("hy-"):
        return "tencent"
    if "deepseek" in m or m.startswith("ds-"):
        return "deepseek"
    if "mistral" in m or "mixtral" in m or "magistral" in m:
        return "mistral"
    if "command" in m or "cohere" in m:
        return "cohere"
    if "qwen" in m or "tongyi" in m:
        return "alibaba"
    if "doubao" in m:
        return "bytedance"
    if "yi-" in m or m.startswith("yi-") or "01ai" in m:
        return "01-ai"
    # Cursor 自有：composer-*, cursor-*, "default"（auto-pick）
    if "composer" in m or m.startswith("cursor") or m == "default":
        return "cursor"
    return "unknown"


# ─── 默认 model 解析（修 unknown 的核心入口） ───────────────
#
# Cursor 不在 transcript 暴露 LLM model / provider，所以 enricher 端只能：
#   1. 优先 env：用户在 .env 里写 ``COT_DEFAULT_MODEL=claude-4.6-sonnet``，立刻生效
#   2. 其次扫 transcript metadata（万一某个 hook 把 model 写到 tool_input 里）
#   3. 都没拿到就标 ``unknown`` 但保留 reason，供前端显示「需要在 .env 里配置」
#
# 这里同时支持显式覆盖 provider / agent_name，方便用户标注「我现在用的是 cursor + claude」。

ENV_DEFAULT_MODEL = "COT_DEFAULT_MODEL"
ENV_DEFAULT_PROVIDER = "COT_DEFAULT_PROVIDER"
ENV_AGENT_NAME = "COT_AGENT_NAME"
ENV_COT_ROOT = "COT_EXTRACTOR_ROOT"


# ─── 信号源 #2：Cursor renderer.log ─────────────────────────
#
# 关键发现（v0.13.x）：当用户在 Cursor 里切换到非 Claude 模型（GPT-5.x /
# Codex / GLM / Hunyuan / Kimi / Grok / Composer 等），Cursor 自身的 hook
# 调用链对部分模型并不稳定地写出 ``payload.model``——events.jsonl 里那
# 个字段会变成 None 甚至 "default"，导致下游 OTel 视图全部 unknown。
#
# 但 Cursor 的 renderer.log 里有一行决定性日志：
#   2026-04-28 17:58:42.830 [info] [buildRequestedModel] composerId=<sid>
#       catalogModelId=gpt-5.4 idSource=selectedModels[0] ...
#
# composerId 就是我们的 ``session_id``，catalogModelId 就是真实生效的
# model。同一个 session 的多次 buildRequestedModel 时间序列正好就是
# "用户在该 session 内切换了几次模型"的真值——直接把它做成时间轴：
#   [(t_ms, model), ...]
#
# 用法：
#   - session 级 dominant model：取出现频次最高的
#   - turn 级 model：用 turn_start_time 二分（或线性扫）找时间轴上 ≤t 的
#     最后一项
#   - session 出现过的 model 集合：set(timeline 里所有 model)
#
# 跨平台路径：Windows = %APPDATA%/Cursor/logs；macOS =
# ~/Library/Application Support/Cursor/logs；Linux = ~/.config/Cursor/logs。
# 找不到就静默返回空（不抛错），让其他信号源接管。

ENV_CURSOR_LOGS_ROOT = "CURSOR_LOGS_ROOT"  # 可选 override

_RENDERER_MODEL_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
    r".*?\[buildRequestedModel\].*?composerId=(\S+).*?catalogModelId=(\S+)"
)


def _cursor_logs_root() -> Optional[str]:
    """跨平台定位 Cursor logs 根目录；未找到返回 None。"""
    override = (os.environ.get(ENV_CURSOR_LOGS_ROOT) or "").strip()
    if override and os.path.isdir(override):
        return override
    candidates: List[str] = []
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(os.path.join(appdata, "Cursor", "logs"))
    else:
        home = os.path.expanduser("~")
        candidates.extend([
            os.path.join(home, "Library", "Application Support", "Cursor", "logs"),
            os.path.join(home, ".config", "Cursor", "logs"),
        ])
    for c in candidates:
        try:
            if c and os.path.isdir(c):
                return c
        except OSError:
            continue
    return None


def _parse_renderer_ts(ts: str) -> float:
    """'2026-04-28 17:58:42.830' / 带 T 的 ISO → epoch ms。"""
    from datetime import datetime
    s = ts.replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).timestamp() * 1000.0
        except ValueError:
            continue
    return 0.0


def _load_renderer_models(session_id: str) -> List[Tuple[float, str]]:
    """扫所有 Cursor renderer.log，返回该 session_id 的 model 时间轴。

    返回 ``[(t_ms, model), ...]``，按时间升序，已去重相邻同值条目。

    实现细节：
    - 直接遍历 ``<logs_root>/*/window*/renderer.log``。Cursor 每次启动会建一
      个新的 ``YYYYMMDDTHHMMSS`` 子目录，所以一个长 session 可能横跨多个 run，
      全部扫完才能拿全时间轴。
    - 用早期短路：``buildRequestedModel`` 字符串和 session_id 都不在某行就
      跳过，避免每行跑 regex 拖累 IO。
    - 解析失败 / 文件不可读静默忽略；找不到就返回空列表。
    """
    if not session_id:
        return []
    root = _cursor_logs_root()
    if not root:
        return []

    timeline: List[Tuple[float, str]] = []
    try:
        run_dirs = os.listdir(root)
    except OSError:
        return []

    for run in run_dirs:
        run_path = os.path.join(root, run)
        if not os.path.isdir(run_path):
            continue
        try:
            entries = os.listdir(run_path)
        except OSError:
            continue
        for entry in entries:
            if not entry.startswith("window"):
                continue
            rfile = os.path.join(run_path, entry, "renderer.log")
            if not os.path.isfile(rfile):
                continue
            try:
                with open(rfile, "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        if "buildRequestedModel" not in line:
                            continue
                        if session_id not in line:
                            continue
                        m = _RENDERER_MODEL_RE.search(line)
                        if not m:
                            continue
                        ts_str, sid, model = m.group(1), m.group(2), m.group(3)
                        if sid != session_id:
                            continue
                        t_ms = _parse_renderer_ts(ts_str)
                        timeline.append((t_ms, model.strip()))
            except OSError:
                continue

    if not timeline:
        return []

    # 按时间排序，再去重 "相邻同 model" 条目（buildRequestedModel 在同一次
    # 用户提交里会被打多次，没必要全留下）
    timeline.sort(key=lambda x: x[0])
    deduped: List[Tuple[float, str]] = []
    last_model: Optional[str] = None
    for t, m in timeline:
        if m != last_model:
            deduped.append((t, m))
            last_model = m
    return deduped


def _model_at_ts(timeline: List[Tuple[float, str]], t_ms: float) -> Optional[str]:
    """Return the last selected model at or before ``t_ms``."""
    if not timeline:
        return None
    chosen: Optional[str] = None
    for ts, m in timeline:
        if ts <= t_ms:
            chosen = m
        else:
            break
    # 没有任何 entry 早于 t_ms（用户问题先于第一次 buildRequestedModel 记录），
    # fallback 到第一项——比 unknown 强
    return chosen or timeline[0][1]


def _events_jsonl_path(session_id: str) -> Optional[str]:
    """返回 cot-stream.js 写下的 ``events.jsonl`` 真实磁盘路径。

    搜索顺序（v0.18.7 起跟 ``cot_extractor._load_cursor_events`` 完全对齐
    —— 0.18.5 之前 enricher 漏迁移路径，导致 wheel 安装态用户的
    events.jsonl 永远找不到 → ``events_meta=None`` → ``_resolve_session_model``
    返回 ``"unknown"`` → 前端 model / cost / OTel KPI 全部塌成 unknown，
    哪怕 ``cot-extractor.py`` 本身已经能正确读出 events.jsonl）：

      1. ``$AGENT_COT_DATA_ROOT/events/<sid>/events.jsonl``  —— ``agent-cot start``
         注入；wheel 安装态默认值。
      2. ``~/.agent-cot/data/events/<sid>/events.jsonl``  —— v0.18.5 起统一默认。
      3. ``$COT_EXTRACTOR_ROOT/output/events/<sid>/events.jsonl``  —— v0.18.4 及之前老路径。
      4. ``<enricher_file>/../output/events/<sid>/events.jsonl``  —— 源码态 dev。
      5. ``./output/events/<sid>/events.jsonl``  —— cwd 兜底。

    多 IDE 接入（CodeBuddy / Claude）也尝试 provider 前缀目录
    （``codebuddy-<sid>`` / ``claude-<sid>``），跟
    ``_load_cursor_events`` 的 ``_EVENT_PROVIDER_PREFIXES`` 同款。
    """
    bases: List[str] = []

    # 1) 新优先：AGENT_COT_DATA_ROOT/events/
    env_data_root = (os.environ.get("AGENT_COT_DATA_ROOT") or "").strip()
    if env_data_root:
        bases.append(os.path.join(os.path.expanduser(env_data_root), "events"))

    # 2) 用户级默认（v0.18.5+）
    bases.append(os.path.join(os.path.expanduser("~"), ".agent-cot", "data", "events"))

    # 3) 老路径：COT_EXTRACTOR_ROOT/output/events/
    env_root = (os.environ.get(ENV_COT_ROOT) or "").strip()
    if env_root:
        bases.append(os.path.join(env_root, "output", "events"))

    # 4) 源码态 dev：enricher 装在 cot-extractor/src/，所以 .. 就是 cot-extractor/
    here = os.path.dirname(os.path.abspath(__file__))
    bases.append(os.path.join(here, "..", "output", "events"))

    # 5) cwd 兜底
    bases.append(os.path.join(os.getcwd(), "output", "events"))

    # 多 IDE 前缀（跟 cot_extractor._EVENT_PROVIDER_PREFIXES 对齐；这里硬
    # 编码而不是 import，是为了让 enricher 完全独立、不引入 cot_extractor 的
    # 大模块依赖 —— enricher 在 backend / langfuse exporter 里也会被单独 import。）
    PROVIDER_PREFIXES = ("codebuddy-", "claude-")

    candidates: List[str] = []
    seen: set = set()
    for base in bases:
        for sid_dir in (session_id,):
            p = os.path.join(base, sid_dir, "events.jsonl")
            if p not in seen:
                seen.add(p)
                candidates.append(p)
        # 如果调用方传的就是裸 sid（没带 provider 前缀），也尝试加前缀
        if not any(session_id.startswith(prefix) for prefix in PROVIDER_PREFIXES):
            for prefix in PROVIDER_PREFIXES:
                p = os.path.join(base, f"{prefix}{session_id}", "events.jsonl")
                if p not in seen:
                    seen.add(p)
                    candidates.append(p)

    for p in candidates:
        try:
            if p and os.path.isfile(p):
                return os.path.abspath(p)
        except OSError:
            continue
    return None


def _load_events_meta(session_id: str) -> Optional[Dict[str, Any]]:
    """扫 ``events.jsonl`` 抽出 session 级 metadata + 真实 token 用量。

    返回结构：
        {
            "model": "claude-opus-4-7",
            "cursor_version": "3.1.17",
            "user_email": "...",
            "events_count": 951,
            "events_path": "...",
            "actual_token_usage": {
                "input_tokens": ...,
                "output_tokens": ...,
                "cache_read_tokens": ...,
                "cache_write_tokens": ...,
                "agent_response_count": 18,
            },
            # generation_id → token usage（每次 LLM 调用一份）
            "by_generation": {gid: {input_tokens, output_tokens, ...}},
            # session 内出现过的所有 model（多 model 场景）
            "model_distribution": {"claude-opus-4-7": 951},
        }

    扫不到（events 目录不存在 / 文件为空 / 全是脏数据）返回 None。
    """
    path = _events_jsonl_path(session_id)
    if not path:
        return None

    model_counts: Dict[str, int] = {}
    cursor_version = ""
    user_email = ""
    by_gen: Dict[str, Dict[str, int]] = {}
    in_t = out_t = cache_r = cache_w = 0
    agent_response_count = 0
    events_count = 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                events_count += 1
                payload = obj.get("payload") or {}
                if not isinstance(payload, dict):
                    continue
                m = payload.get("model")
                if isinstance(m, str) and m:
                    model_counts[m] = model_counts.get(m, 0) + 1
                if not cursor_version and isinstance(payload.get("cursor_version"), str):
                    cursor_version = payload["cursor_version"]
                if not user_email and isinstance(payload.get("user_email"), str):
                    user_email = payload["user_email"]

                if obj.get("event") == "afterAgentResponse":
                    agent_response_count += 1
                    pi = int(payload.get("input_tokens") or 0)
                    po = int(payload.get("output_tokens") or 0)
                    pcr = int(payload.get("cache_read_tokens") or 0)
                    pcw = int(payload.get("cache_write_tokens") or 0)
                    in_t += pi
                    out_t += po
                    cache_r += pcr
                    cache_w += pcw
                    gid = payload.get("generation_id")
                    if isinstance(gid, str) and gid:
                        by_gen[gid] = {
                            "input_tokens": pi,
                            "output_tokens": po,
                            "cache_read_tokens": pcr,
                            "cache_write_tokens": pcw,
                            "model": m if isinstance(m, str) else "",
                        }
    except OSError:
        return None

    if events_count == 0:
        # events.jsonl 啥都没扫到也不立即放弃——renderer.log 可能仍能补 model
        renderer_timeline = _load_renderer_models(session_id)
        if not renderer_timeline:
            return None
        # 走 renderer-only 分支
        rcounts: Dict[str, int] = {}
        for _, m in renderer_timeline:
            rcounts[m] = rcounts.get(m, 0) + 1
        dominant_model = max(rcounts.items(), key=lambda kv: kv[1])[0]
        return {
            "model": dominant_model,
            "cursor_version": None,
            "user_email": None,
            "events_count": 0,
            "events_path": path,
            "actual_token_usage": None,
            "by_generation": {},
            "model_distribution": rcounts,
            "model_timeline": [
                {"t_ms": t, "model": m} for t, m in renderer_timeline
            ],
            "model_source": "renderer_log",
            "models_seen": sorted(rcounts.keys()),
        }

    # ── events.jsonl 有数据：用它当主信号源，再把 renderer.log 的时间轴
    #    叠加进来作为 per-turn 模型识别的辅助。
    # 注意：events.jsonl 可能有数据但 model_counts 为空（比如该 session 只
    # 被 transcript_watcher 写过 agentToolCall/Result，Cursor 自家 hook 没
    # 给 GPT/GLM 这类模型触发——cf9201f9 就是这种），这时 renderer.log 是
    # 唯一可用信号源。
    renderer_timeline = _load_renderer_models(session_id)
    if renderer_timeline:
        # 把 renderer 里的 model 也算进 distribution（events 可能漏掉 GPT/GLM 等）
        for _, m in renderer_timeline:
            model_counts[m] = model_counts.get(m, 0) + 1

    # dominant 选 events 与 renderer 合并后频次最高的
    dominant_model = ""
    if model_counts:
        dominant_model = max(model_counts.items(), key=lambda kv: kv[1])[0]
    # events.jsonl 报的是 "default"（Cursor auto-pick）但 renderer 知道
    # 真名 → 优先采纳 renderer 的真名
    if dominant_model == "default" and renderer_timeline:
        # 用 renderer 时间轴里出现最多的非 default model 取代
        rcounts: Dict[str, int] = {}
        for _, m in renderer_timeline:
            if m != "default":
                rcounts[m] = rcounts.get(m, 0) + 1
        if rcounts:
            dominant_model = max(rcounts.items(), key=lambda kv: kv[1])[0]

    # 把所有信号源里出现过的 model 名（去重排序）单独保留，让前端展示
    # "本 session 用过 N 个不同模型"
    models_seen = sorted({
        *(m for m in model_counts.keys() if m and m != "default"),
        *(m for _, m in renderer_timeline if m and m != "default"),
    })

    return {
        "model": dominant_model or None,
        "cursor_version": cursor_version or None,
        "user_email": user_email or None,
        "events_count": events_count,
        "events_path": path,
        "actual_token_usage": {
            "input_tokens": in_t,
            "output_tokens": out_t,
            "cache_read_tokens": cache_r,
            "cache_write_tokens": cache_w,
            "agent_response_count": agent_response_count,
        } if agent_response_count else None,
        "by_generation": by_gen,
        "model_distribution": model_counts,
        "model_timeline": [
            {"t_ms": t, "model": m} for t, m in renderer_timeline
        ] if renderer_timeline else [],
        "model_source": "events_jsonl" + ("+renderer_log" if renderer_timeline else ""),
        "models_seen": models_seen,
    }


def _resolve_session_model(session_cot) -> Tuple[str, str, str, str, Optional[Dict[str, Any]]]:
    """决定 session 级别的 model / provider / agent_name / source / events_meta。

    新优先级（v0.11.2）：
      1. ``events.jsonl`` → source = ``"events"``（cot-stream.js 实时抓的真值）
      2. ``COT_DEFAULT_MODEL`` env → source = ``"env"``（兜底）
      3. transcript metadata → source = ``"transcript"``
      4. 都没有 → source = ``"unknown"``

    ``events_meta`` 是 ``_load_events_meta`` 的返回值（即使没用它做 model 来源也可能有，
    比如用户既在 .env 配了 model 又装了 cot-stream，我们仍然把真实 token 用量回传）。
    """
    session_id = getattr(session_cot, "session_id", None) or ""
    env_model = (os.environ.get(ENV_DEFAULT_MODEL) or "").strip()
    env_provider = (os.environ.get(ENV_DEFAULT_PROVIDER) or "").strip()
    env_agent = (os.environ.get(ENV_AGENT_NAME) or "").strip()

    events_meta = _load_events_meta(session_id) if session_id else None

    # 1. events.jsonl 优先（_load_events_meta 内部已合并 renderer.log 信号）
    if events_meta and events_meta.get("model"):
        m = events_meta["model"]
        provider = env_provider or _provider_from_model(m)
        agent_name = env_agent or "cursor-agent"
        # source 标识：让前端知道走的是 events / renderer / 二者合并
        src_label = events_meta.get("model_source") or "events"
        return m, provider, agent_name, src_label, events_meta

    # 1.5. events.jsonl 没扫到（完全不存在 / 或扫到了但 model 为空）：
    # 直接试 renderer.log 兜底——这是 GPT-5.x / Codex / GLM 等非 Claude
    # 模型唯一稳定的 model 真值来源。
    if session_id:
        rtl = _load_renderer_models(session_id)
        if rtl:
            rcounts: Dict[str, int] = {}
            for _, m in rtl:
                rcounts[m] = rcounts.get(m, 0) + 1
            dominant = max(rcounts.items(), key=lambda kv: kv[1])[0]
            provider = env_provider or _provider_from_model(dominant)
            agent_name = env_agent or "cursor-agent"
            # 即使 events_meta=None，也回填一个最小 stub 让下游 enrichment
            # 拿到 renderer 时间轴
            stub_meta = events_meta or {
                "model": dominant,
                "cursor_version": None,
                "user_email": None,
                "events_count": 0,
                "events_path": None,
                "actual_token_usage": None,
                "by_generation": {},
                "model_distribution": rcounts,
            }
            stub_meta["model_timeline"] = [
                {"t_ms": t, "model": m} for t, m in rtl
            ]
            stub_meta["model_source"] = "renderer_log"
            stub_meta["models_seen"] = sorted(rcounts.keys())
            return dominant, provider, agent_name, "renderer_log", stub_meta

    # 2. env 兜底
    if env_model:
        provider = env_provider or _provider_from_model(env_model)
        agent_name = env_agent or "cursor-agent"
        return env_model, provider, agent_name, "env", events_meta

    # 3. transcript 扫描
    scanned_model = ""
    for turn in getattr(session_cot, "turns", []) or []:
        for s in getattr(turn, "steps", []) or []:
            md = s.metadata or {}
            ti = md.get("decision_tool_input") or md.get("tool_input")
            if isinstance(ti, dict):
                m = ti.get("model")
                if isinstance(m, str) and m.strip():
                    scanned_model = m.strip()
                    break
        if scanned_model:
            break

    if scanned_model:
        provider = env_provider or _provider_from_model(scanned_model)
        agent_name = env_agent or "cursor-agent"
        return scanned_model, provider, agent_name, "transcript", events_meta

    # 4. v0.15.0：transcript-first 兜底（主要给 Claude 用）。
    #
    # 上面那次扫描只看 ``step.metadata.tool_input.model``——这是 Cursor 的
    # ``CallLLM`` 工具自报的 model。Claude 不通过工具发起 LLM 调用，model 直
    # 接挂在每条 assistant 消息的 ``message.model`` 上，且 usage 含
    # ``cache_creation_input_tokens`` / ``cache_read_input_tokens`` 这两个
    # Cursor 不会出的字段。读 transcript 是从原始字节直接拿真值，不做任何
    # 推断，符合 §12.1 transcript-first 原则。
    transcript_path = getattr(session_cot, "transcript_path", None) or ""
    if transcript_path and os.path.isfile(transcript_path):
        try:
            from collections import Counter
            mdl_counts: Counter = Counter()
            usage_total = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            }
            assistant_seen = 0
            with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except Exception:
                        continue
                    if rec.get("type") != "assistant":
                        continue
                    msg = rec.get("message") or {}
                    m = msg.get("model")
                    if isinstance(m, str) and m.strip():
                        mdl_counts[m.strip()] += 1
                    u = msg.get("usage") or {}
                    if isinstance(u, dict):
                        for k in usage_total:
                            v = u.get(k)
                            if isinstance(v, (int, float)):
                                usage_total[k] += int(v)
                    assistant_seen += 1
            if mdl_counts:
                dominant_model = mdl_counts.most_common(1)[0][0]
                stub_meta: Dict[str, Any] = {
                    "model": dominant_model,
                    "model_source": "transcript",
                    "models_seen": sorted(mdl_counts.keys()),
                    "model_distribution": dict(mdl_counts),
                    "actual_token_usage": {
                        "input_tokens": usage_total["input_tokens"],
                        "output_tokens": usage_total["output_tokens"],
                        # cache_read = LLM 端缓存命中读量
                        # cache_write = LLM 端首次缓存写量（Anthropic 命名 cache_creation）
                        "cache_read_tokens": usage_total["cache_read_input_tokens"],
                        "cache_write_tokens": usage_total["cache_creation_input_tokens"],
                        "source": "transcript",
                        "assistant_messages_counted": assistant_seen,
                    },
                    "events_count": 0,
                    "events_path": None,
                    "by_generation": {},
                    "user_email": None,
                    "cursor_version": None,
                }
                provider = env_provider or _provider_from_model(dominant_model)
                # agent_name：Claude 模型默认 claude-code，其余沿用 cursor-agent
                if dominant_model.startswith("claude-"):
                    agent_name = env_agent or "claude-code"
                else:
                    agent_name = env_agent or "cursor-agent"
                return dominant_model, provider, agent_name, "transcript", stub_meta
        except Exception:
            pass

    return "unknown", env_provider or "unknown", env_agent or "cursor-or-claude", "unknown", events_meta


# ─── step kind 分类（修 host-tool 错标 unknown 的核心） ───
#
# OTel GenAI 的 ``gen_ai.*`` 命名空间只对「真正经过 LLM」的 span 有意义。
# 对 ``Tool Execution → Shell`` 这类 host runtime 行为，强塞 ``gen_ai.request.model``
# 反而是误导。我们把 step 分四档：
#   * llm_call    → 走 session.model（thinking/decision/final_response/...）
#   * host_tool   → model = "host:cursor"，cost = N/A，operation_name = execute_tool
#   * user_input  → 客户端输入（非 LLM、非 tool），model = "n/a (user)"
#   * agent_event → strategy_shift / error_recovery / mode_transition / plan
#                   等合成事件，model = "n/a (synthetic)"

LLM_STEP_TYPES = {
    "thinking_inter",
    "thinking_intermediate",
    "thinking_explicit",
    "pre_tool_reasoning",
    "tool_decision",
    "final_response",
}

HOST_TOOL_STEP_TYPES = {
    "tool_execution",
}

USER_STEP_TYPES = {
    "user_input",
}

AGENT_EVENT_STEP_TYPES = {
    "strategy_shift",
    "error_recovery",
    "mode_transition",
    "plan_update",
    "todo_progress",
    "plan",
}


def _classify_step_kind(step) -> str:
    st = getattr(step, "step_type", "") or ""
    if st in LLM_STEP_TYPES:
        return "llm_call"
    if st in HOST_TOOL_STEP_TYPES:
        return "host_tool"
    if st in USER_STEP_TYPES:
        return "user_input"
    if st in AGENT_EVENT_STEP_TYPES:
        return "agent_event"
    # 兜底：未知类型按 LLM 处理（保守）
    return "llm_call"


def _estimate_tokens_from_chars(text: str) -> int:
    """char/4 简易估算（OpenAI / Anthropic 英文/混合文本通用近似）。"""
    if not text:
        return 0
    # 中文字符比英文 token 密度高约 2x，所以 CJK 用 /1.7
    cjk = sum(1 for ch in text if "\u3000" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, round(cjk / 1.7 + other / 4))


def _compute_cache_aware_cost_usd(
    model_key: Optional[str],
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
) -> Tuple[Optional[float], Dict[str, float]]:
    """Anthropic / OpenAI prompt-cache 感知的 cost 计算。

    Anthropic 标准定价（``claude-opus-4`` 系列）：
      * non-cache input  : 1.0x   ($15 / 1M)
      * cache write      : 1.25x  ($18.75 / 1M)
      * cache read       : 0.1x   ($1.50 / 1M)
      * output           : 1.0x   ($75 / 1M)

    Cursor hook 报上来的 ``input_tokens`` 实际含 cache，按以下口径分摊：
        non_cache_input = max(0, input_tokens - cache_read - cache_write)
        cost = non_cache_input * in_p
             + cache_write     * in_p * 1.25
             + cache_read      * in_p * 0.10
             + output_tokens   * out_p
    返回 ``(total_cost, breakdown_dict)``。
    """
    if not model_key:
        return None, {}
    p = _PRICING_USD_PER_1K.get(model_key)
    if not p:
        return None, {}
    in_p, out_p = p
    non_cache_input = max(0, int(input_tokens) - int(cache_read_tokens) - int(cache_write_tokens))
    cost_in = non_cache_input / 1000.0 * in_p
    cost_cw = int(cache_write_tokens) / 1000.0 * in_p * 1.25
    cost_cr = int(cache_read_tokens) / 1000.0 * in_p * 0.10
    cost_out = int(output_tokens) / 1000.0 * out_p
    total = cost_in + cost_cw + cost_cr + cost_out
    breakdown = {
        "non_cache_input_usd": round(cost_in, 6),
        "cache_write_usd": round(cost_cw, 6),
        "cache_read_usd": round(cost_cr, 6),
        "output_usd": round(cost_out, 6),
        "non_cache_input_tokens": non_cache_input,
    }
    return round(total, 6), breakdown


def _compute_cost_usd(
    model_key: Optional[str], input_tokens: int, output_tokens: int
) -> Tuple[Optional[float], str]:
    """返回 ``(cost_usd, cost_reason)``。

    ``cost_reason`` 用于前端显示 None 值的解释：
      * ``"ok"``          - 正常算出
      * ``"unknown_model"`` - model 是 unknown，没法查 pricing
      * ``"no_pricing"`` - model 已知但不在 pricing 表里
      * ``"non_llm_step"`` - 这个 step 不经过 LLM，按 OTel 规范不计 cost
    """
    if not model_key:
        return None, "unknown_model"
    p = _PRICING_USD_PER_1K.get(model_key)
    if not p:
        return None, "no_pricing"
    in_p, out_p = p
    cost = (input_tokens / 1000.0) * in_p + (output_tokens / 1000.0) * out_p
    return round(cost, 6), "ok"


def _build_token_usage(
    model_key: Optional[str],
    in_t: int,
    out_t: int,
    is_estimate: bool,
    *,
    non_llm: bool = False,
) -> Dict[str, Any]:
    """合成 token usage block。``non_llm=True`` 时强制 cost=None / reason=non_llm_step。"""
    if non_llm:
        cost: Optional[float] = None
        reason = "non_llm_step"
    else:
        cost, reason = _compute_cost_usd(model_key, in_t, out_t)
    return {
        "input_tokens": int(in_t),
        "output_tokens": int(out_t),
        "total_tokens": int(in_t) + int(out_t),
        "cost_usd": cost,
        "cost_reason": reason,
        "currency": "USD",
        "is_estimate": is_estimate,
        "model_key": model_key,
    }


# ─── messages 重打包 ───────────────────────────────────────


def _pack_message(role: str, text: str) -> Dict[str, Any]:
    """单条 OTel-style message：``{role, parts:[{type,content}]}``。"""
    return {
        "role": role,
        "parts": [{"type": "text", "content": text or ""}],
    }


def _try_parse_json_array(text: str) -> Optional[List[Any]]:
    if not text:
        return None
    s = text.strip()
    if not s.startswith("["):
        return None
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else None
    except Exception:
        return None


def _structured_input_messages(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把 step.metadata 里 LLM/RAG 的入参拍成 input.messages。

    优先级：
    1. ``decision_tool_input.messages`` / ``tool_input.messages`` 这种结构化已有
    2. ``prompt_preview`` 当 user-text fallback
    3. 没东西就 []
    """
    if not isinstance(meta, dict):
        return []
    for parent_key in ("decision_tool_input", "tool_input"):
        ti = meta.get(parent_key)
        if isinstance(ti, dict):
            msgs = ti.get("messages")
            if isinstance(msgs, list):
                out = []
                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    role = m.get("role") or m.get("type") or "user"
                    content = m.get("content")
                    if isinstance(content, str):
                        out.append(_pack_message(role, content))
                    elif isinstance(content, list):
                        parts: List[Dict[str, Any]] = []
                        for c in content:
                            if isinstance(c, dict):
                                parts.append({
                                    "type": c.get("type") or "text",
                                    "content": str(
                                        c.get("text")
                                        or c.get("content")
                                        or c.get("input")
                                        or ""
                                    ),
                                })
                            else:
                                parts.append({"type": "text", "content": str(c)})
                        out.append({"role": role, "parts": parts})
                    else:
                        out.append(_pack_message(role, str(content or "")))
                if out:
                    return out
            # query/prompt 一类的简单字段
            for k in ("prompt", "query", "question", "objective"):
                v = ti.get(k)
                if isinstance(v, str) and v.strip():
                    return [_pack_message("user", v)]
    pp = meta.get("prompt_preview")
    if isinstance(pp, str) and pp.strip():
        return [_pack_message("user", pp)]
    return []


def _structured_output_messages(
    content: str, meta: Dict[str, Any], finish_reason: str
) -> List[Dict[str, Any]]:
    if not content and not meta.get("recall_preview"):
        return []
    text = content or ""
    # RAG 召回内容也作为 assistant output 的 retrieval evidence 给前端看
    if meta.get("recall_preview") and not text:
        text = str(meta["recall_preview"])
    msg = _pack_message("assistant", text)
    msg["finish_reason"] = finish_reason
    return [msg]


# ─── retrieval documents 拆分 ─────────────────────────────


def _split_retrieval_documents(recall_preview: str) -> List[Dict[str, Any]]:
    """把 recall_preview 字符串切成 documents[]。

    优先级：
    1. JSON list of dicts/strings → 直接转
    2. ``\\n---\\n`` 分隔行 → 每段一篇
    3. 编号开头（``1. ...`` / ``# 1`` / ``## Doc``）→ 按编号切
    4. 兜底：整段当一篇
    """
    if not recall_preview:
        return []
    arr = _try_parse_json_array(recall_preview)
    if arr is not None:
        out: List[Dict[str, Any]] = []
        for i, item in enumerate(arr):
            if isinstance(item, dict):
                out.append({
                    "id": str(item.get("id") or item.get("doc_id") or f"doc-{i+1}"),
                    "content": str(
                        item.get("content")
                        or item.get("text")
                        or item.get("body")
                        or json.dumps(item, ensure_ascii=False)
                    )[:4000],
                    "score": item.get("score") or item.get("similarity"),
                    "metadata": {
                        k: v for k, v in item.items()
                        if k not in ("content", "text", "body", "score", "similarity")
                    },
                })
            else:
                out.append({
                    "id": f"doc-{i+1}",
                    "content": str(item)[:4000],
                    "score": None,
                    "metadata": {},
                })
        return out

    # 分隔行
    parts: List[str] = []
    if "\n---\n" in recall_preview:
        parts = [p.strip() for p in recall_preview.split("\n---\n") if p.strip()]
    elif re.search(r"\n\s*##\s+", recall_preview):
        parts = re.split(r"\n\s*##\s+", recall_preview)
        parts = [p.strip() for p in parts if p.strip()]
    elif re.search(r"\n\s*\d+\.\s+", recall_preview):
        parts = re.split(r"\n(?=\s*\d+\.\s+)", recall_preview)
        parts = [p.strip() for p in parts if p.strip()]

    if len(parts) >= 2:
        return [{
            "id": f"doc-{i+1}",
            "content": p[:4000],
            "score": None,
            "metadata": {},
        } for i, p in enumerate(parts)]

    return [{
        "id": "doc-1",
        "content": recall_preview[:4000],
        "score": None,
        "metadata": {"split": "single"},
    }]


# ─── finish reason ─────────────────────────────────────────

_FINISH_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
    "error": "error",
}


def _normalize_finish_reason(meta: Dict[str, Any], step_type: str) -> str:
    """把任意 step 的 finish_reason 归一化到 OTel ``gen_ai.response.finish_reasons``。"""
    if not isinstance(meta, dict):
        meta = {}
    sr = meta.get("stop_reason")
    if isinstance(sr, str) and sr in _FINISH_REASON_MAP:
        return _FINISH_REASON_MAP[sr]
    if meta.get("inferred_final"):
        return "stop"
    if meta.get("is_error"):
        return "error"
    if step_type == "tool_decision":
        return "tool_calls"
    if step_type == "final_response":
        return "stop"
    return "stop"


# ─── Resource attributes ──────────────────────────────────

def _build_resource_attributes() -> Dict[str, Any]:
    return {
        "service.name": "cot-extractor",
        "service.version": "v0.12.0",
        "service.namespace": "ai-ide-langfuse",
        "deployment.environment": os.environ.get("COT_ENV", "local-dev"),
        "host.name": socket.gethostname(),
        "host.arch": platform.machine(),
        "host.os.type": platform.system().lower(),
        "host.os.version": platform.release(),
        "telemetry.sdk.name": "cot-otel-enricher",
        "telemetry.sdk.language": "python",
        "telemetry.sdk.version": _OTEL_VERSION,
        "process.runtime.name": "cpython",
        "process.runtime.version": platform.python_version(),
    }


# ─── 主入口 ────────────────────────────────────────────────


def enrich_session_with_otel(
    session_cot,
    *,
    response_report: Optional[Dict[str, Any]] = None,
) -> None:
    """就地把 OTel 视图字段注入到 SessionCoT 实例。

    - ``session_cot``: cot_extractor.extract_session_cot() 返回的 SessionCoT 对象
    - ``response_report``: 可选，response-verifier 的 report dict（含 scores）
    """
    if session_cot is None:
        return

    session_id = getattr(session_cot, "session_id", None) or "unknown"
    trace_id = _trace_id_for(session_id)
    session_root_span = _span_id_for(session_id, "root")

    # v0.11.2：自动检测 model（events.jsonl > renderer.log > env > transcript > unknown）
    model, provider, agent_name, model_source, events_meta = _resolve_session_model(session_cot)
    model_key = _normalize_model_key(model) if model and model != "unknown" else None

    # v0.13.x：抽出 per-turn model 时间轴。renderer.log 的
    # ``buildRequestedModel`` 是 session 内"用户切了几次模型"的真值流，
    # 我们用它给每个 turn 找当时生效的 model（用 turn_start_time 落到时间轴
    # 上 ≤t 的最后一项），同 session 跨模型场景就能各 turn 标对了。
    _turn_timeline: List[Tuple[float, str]] = []
    if events_meta and isinstance(events_meta.get("model_timeline"), list):
        for e in events_meta["model_timeline"]:
            try:
                t_ms = float(e.get("t_ms") or 0)
                m = str(e.get("model") or "").strip()
                if t_ms > 0 and m:
                    _turn_timeline.append((t_ms, m))
            except (TypeError, ValueError):
                continue

    def _ts_iso_to_ms(ts: Optional[str]) -> Optional[float]:
        """ISO 字符串（包括带 Z 的 UTC）转 epoch ms。"""
        if not ts:
            return None
        try:
            from datetime import datetime
            s = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
            return datetime.fromisoformat(s).timestamp() * 1000.0
        except (ValueError, TypeError):
            return None

    # 预先建一份 turn 内 ``tool_use_id → tool_decision step`` 的索引，
    # 给 tool_execution 反查它对应的 LLM 入参字符数（修 input_tokens=0 的核心）
    decision_index: Dict[Tuple[int, str], "Any"] = {}
    for turn in session_cot.turns:
        for s in turn.steps:
            if s.step_type == "tool_decision" and s.tool_use_id:
                decision_index[(turn.turn_index, s.tool_use_id)] = s

    session_input_tokens = 0
    session_output_tokens = 0
    session_cost = 0.0
    has_real_tokens = False

    # ── 给每个 turn / step 注入 OTel 字段 ────────────────
    for turn in session_cot.turns:
        turn_root_span = _span_id_for(session_id, "turn", turn.turn_index)

        # v0.13.x: 该 turn 实际生效的 model。用 renderer.log 时间轴 + 该
        # turn 的开始时间查表；查不到就退回 session dominant model。
        turn_model = model
        turn_provider = provider
        turn_model_source = model_source
        if _turn_timeline:
            t_start_ms = _ts_iso_to_ms(getattr(turn, "turn_start_time", None))
            if t_start_ms is not None:
                m_at = _model_at_ts(_turn_timeline, t_start_ms)
                if m_at:
                    turn_model = m_at
                    turn_provider = _provider_from_model(m_at)
                    turn_model_source = "renderer_log_timeline"
        turn_model_key = (
            _normalize_model_key(turn_model)
            if turn_model and turn_model != "unknown" else None
        )

        # turn-level usage（transcript 有给）
        usage = turn.usage or {}
        in_t = int(usage.get("input_tokens") or 0)
        out_t = int(usage.get("output_tokens") or 0)
        is_est = False
        if in_t == 0 and out_t == 0:
            # 启发式估算：把 user_query + final_response 字符数推一次
            in_t = _estimate_tokens_from_chars(turn.user_query or "")
            out_t = _estimate_tokens_from_chars(turn.final_response or "")
            is_est = True
        else:
            has_real_tokens = True

        # 用 turn-specific model_key 算 cost（不同模型 pricing 不同）
        turn_token_usage = _build_token_usage(turn_model_key, in_t, out_t, is_est)
        turn_cost = turn_token_usage["cost_usd"]
        session_input_tokens += in_t
        session_output_tokens += out_t
        if isinstance(turn_cost, (int, float)):
            session_cost += float(turn_cost)

        # finish_reason for turn = 末尾 step 的 finish_reason
        finish_reason = "stop"
        if turn.steps:
            last_step = turn.steps[-1]
            finish_reason = _normalize_finish_reason(
                last_step.metadata or {}, last_step.step_type
            )

        # OTel GenAI 1.27+ standard span attributes for the turn root span
        # 这一块是给 OTLP exporter / Langfuse / Phoenix 直接消费的（不仅给前端）
        turn_attrs: Dict[str, Any] = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.conversation.id": session_id,
            "gen_ai.agent.name": agent_name,
            "gen_ai.provider.name": turn_provider or "unknown",
            "gen_ai.request.model": turn_model or "unknown",
            "gen_ai.response.model": turn_model or "unknown",
            "gen_ai.response.finish_reasons": [finish_reason],
            "gen_ai.usage.input_tokens": int(turn_token_usage["input_tokens"]),
            "gen_ai.usage.output_tokens": int(turn_token_usage["output_tokens"]),
        }
        # turn 总耗时（ms）→ gen_ai.client.operation.duration
        if getattr(turn, "turn_duration_ms", None) is not None:
            try:
                turn_attrs["gen_ai.client.operation.duration"] = float(turn.turn_duration_ms)
            except (TypeError, ValueError):
                pass

        turn_otel: Dict[str, Any] = {
            "trace_id": trace_id,
            "span_id": turn_root_span,
            "parent_span_id": session_root_span,
            "operation_name": "invoke_agent",
            "agent_name": agent_name,
            "model": turn_model,
            "provider": turn_provider,
            "model_source": turn_model_source,
            "finish_reasons": [finish_reason],
            "token_usage": turn_token_usage,
            "conversation_id": session_id,
            "turn_index": turn.turn_index,
            "duration_ms": getattr(turn, "turn_duration_ms", None),
            "request_params": {
                "temperature": None,
                "top_p": None,
                "max_tokens": None,
                "seed": None,
                "stop_sequences": None,
            },
            "response": {
                "id": None,
                "model": turn_model,
            },
            "attributes": turn_attrs,
        }
        # 把它挂到 dataclass 上（cot_extractor 已加 otel 字段）
        if hasattr(turn, "otel"):
            turn.otel = turn_otel

        # eval（如有 response_report）
        eval_obj: Optional[Dict[str, Any]] = None
        if response_report:
            score = (response_report.get("scores") or {}).get("overall")
            if score is None:
                # 用 turn_quality_score fallback
                score = turn.turn_quality_score
            label = "ok" if (score is not None and score >= 0.7) else (
                "warn" if (score is not None and score >= 0.4) else "fail"
            )
            eval_obj = {
                "metric_name": "response_quality",
                "score": score,
                "label": label,
                "details": response_report.get("details") or {},
                "summary": response_report.get("summary") or "",
            }
        elif turn.turn_quality_score is not None:
            eval_obj = {
                "metric_name": "turn_quality_signal",
                "score": turn.turn_quality_score,
                "label": "ok" if turn.turn_quality_score >= 0.7 else (
                    "warn" if turn.turn_quality_score >= 0.4 else "fail"
                ),
                "details": dict(turn.quality_signals or {}),
                "summary": "",
            }
        if hasattr(turn, "eval") and eval_obj is not None:
            turn.eval = eval_obj

        # ── 给每个 step 注入 OTel 字段 ──────────────────
        prev_span_id: Optional[str] = None
        for idx, step in enumerate(turn.steps):
            step_span_id = _span_id_for(session_id, "step", step.step_index)
            md = step.metadata if isinstance(step.metadata, dict) else {}
            kind = _classify_step_kind(step)

            # ── token 估算 ────────────────────────────
            step_in_chars = 0
            step_out_chars = 0
            if step.step_type == "tool_decision":
                # decision = LLM 决定调工具 → 入参就是 prompt_preview / input_summary
                tool_in = md.get("input_summary") or md.get("prompt_preview") or ""
                step_in_chars = len(str(tool_in))
                # decision 本身往往没有 LLM 输出文本，但若有 thinking 摘要就当 output
                step_out_chars = len(step.content or "")
            elif step.step_type in ("final_response", "thinking_inter",
                                    "thinking_intermediate", "thinking_explicit",
                                    "pre_tool_reasoning"):
                # 纯 LLM 输出
                step_in_chars = len(str(md.get("prompt_preview") or ""))
                step_out_chars = len(step.content or "")
            elif step.step_type == "tool_execution":
                # host runtime 执行：input = 配对 decision 的 tool_input；output = 执行结果
                step_out_chars = len(step.content or "")
                paired = decision_index.get((turn.turn_index, step.tool_use_id))
                if paired is not None:
                    pmd = paired.metadata if isinstance(paired.metadata, dict) else {}
                    step_in_chars = len(
                        str(pmd.get("input_summary") or pmd.get("prompt_preview") or "")
                    )
                # 兜底：如果 decision 没扫到入参字符数，至少用 tool_name 当占位 token
                if step_in_chars == 0 and step.tool_name:
                    step_in_chars = len(step.tool_name)
            elif step.step_type == "user_input":
                step_in_chars = len(step.content or "")
            # agent_event 不算 token

            step_in_t = round(step_in_chars / 4) if step_in_chars else 0
            step_out_t = round(step_out_chars / 4) if step_out_chars else 0

            # ── 按 kind 分流 model / provider / cost ──
            if kind == "host_tool":
                # OTel 规范里 host runtime 工具不属于 gen_ai LLM call
                step_model: Optional[str] = "host:cursor"
                step_provider: Optional[str] = "cursor-runtime"
                step_model_source = "host"
                step_token = _build_token_usage(
                    model_key, step_in_t, step_out_t, True, non_llm=True
                )
            elif kind == "user_input":
                step_model = "n/a (user)"
                step_provider = "client"
                step_model_source = "client"
                step_token = _build_token_usage(
                    model_key, step_in_t, step_out_t, True, non_llm=True
                )
            elif kind == "agent_event":
                step_model = "n/a (synthetic)"
                step_provider = "cot-extractor"
                step_model_source = "synthetic"
                step_token = _build_token_usage(
                    model_key, 0, 0, True, non_llm=True
                )
            else:
                # llm_call → 优先用 turn 自己生效的 model（renderer.log 时间轴
                # 解出来的），让"同 session 切换模型"的场景每个 turn 各显神通
                step_model = turn_model
                step_provider = turn_provider
                step_model_source = turn_model_source
                step_token = _build_token_usage(
                    turn_model_key, step_in_t, step_out_t, True
                )

            step_finish = _normalize_finish_reason(md, step.step_type)

            # operation.name 映射
            op_name_map = {
                "user_input": "user_input",
                "tool_decision": "chat",          # LLM 决策是一次 chat call
                "tool_execution": "execute_tool",  # host runtime 执行
                "thinking_inter": "chat",
                "thinking_intermediate": "chat",
                "thinking_explicit": "chat",
                "pre_tool_reasoning": "chat",
                "final_response": "chat",
                "strategy_shift": "agent_event",
                "error_recovery": "agent_event",
                "mode_transition": "agent_event",
                "plan_update": "agent_event",
                "todo_progress": "agent_event",
                "plan": "agent_event",
            }
            op_name = op_name_map.get(step.step_type, "chat")

            # span kind
            if step.step_type == "user_input":
                span_kind = "server"
            elif kind == "host_tool":
                span_kind = "client"   # host runtime 调外部工具，对 trace 而言是 client span
            else:
                span_kind = "internal"

            # ── attributes ─────────────────────────────
            otel_attrs: Dict[str, Any] = {
                "gen_ai.operation.name": op_name,
                "gen_ai.conversation.id": session_id,
                "gen_ai.agent.name": agent_name,
            }

            if kind == "llm_call":
                otel_attrs.update({
                    "gen_ai.provider.name": step_provider,
                    "gen_ai.request.model": step_model,
                    "gen_ai.response.model": step_model,
                    "gen_ai.response.finish_reasons": [step_finish],
                    "gen_ai.usage.input_tokens": step_token["input_tokens"],
                    "gen_ai.usage.output_tokens": step_token["output_tokens"],
                })
            elif kind == "host_tool":
                # OTel GenAI 1.27+：execute_tool span 标准 attribute
                # 同时保留旧 `tool.*` 私有命名以兼容已有前端 / Langfuse 视图
                otel_attrs.update({
                    "gen_ai.tool.name": step.tool_name,
                    "gen_ai.tool.call.id": step.tool_use_id,
                    "gen_ai.tool.type": "host_runtime",
                    "tool.kind": "host_runtime",
                    "tool.runtime": "cursor",
                    "tool.name": step.tool_name,
                    "tool.call.id": step.tool_use_id,
                    "tool.execution.input_chars": step_in_chars,
                    "tool.execution.output_chars": step_out_chars,
                })
                if md.get("is_error"):
                    otel_attrs["error.type"] = md.get("error_type") or "ToolExecutionError"
                if md.get("invocation_category"):
                    otel_attrs["tool.invocation.category"] = md["invocation_category"]
            elif kind == "user_input":
                otel_attrs.update({
                    "client.input.length": step_in_chars,
                    "client.input.kind": "text",
                })
            else:  # agent_event
                otel_attrs.update({
                    "agent.event.kind": step.step_type,
                })

            # decision step 也带上 tool 名字（它是 LLM 调用，但产物是 tool call）
            if step.step_type == "tool_decision":
                otel_attrs["gen_ai.tool.name"] = step.tool_name
                otel_attrs["gen_ai.tool.call.id"] = step.tool_use_id
                otel_attrs["tool.name"] = step.tool_name
                otel_attrs["tool.call.id"] = step.tool_use_id

            # gen_ai.client.operation.duration（OTel GenAI Metrics 标准；以 ms 入 attr）
            if getattr(step, "duration_ms", None) is not None:
                try:
                    otel_attrs["gen_ai.client.operation.duration"] = float(step.duration_ms)
                except (TypeError, ValueError):
                    pass

            # ── input/output messages ─────────────────
            input_msgs: List[Dict[str, Any]] = []
            output_msgs: List[Dict[str, Any]] = []
            if step.step_type == "tool_decision":
                input_msgs = _structured_input_messages(md)
            if step.step_type in ("final_response", "thinking_inter",
                                  "thinking_intermediate", "thinking_explicit",
                                  "pre_tool_reasoning"):
                output_msgs = _structured_output_messages(
                    step.content or "", md, step_finish
                )
            elif step.step_type == "tool_execution":
                # host runtime 工具的 output 不是 LLM message，但前端仍要看到执行结果文本，
                # 所以打成一个 ``role=tool`` 的 message（OTel/OpenInference 都支持 role=tool）
                if step.content:
                    output_msgs = [{
                        "role": "tool",
                        "parts": [{"type": "text", "content": step.content}],
                        "finish_reason": "stop",
                        "tool_call_id": step.tool_use_id,
                    }]

            # retrieval documents（仅 RAG / web_search 类 tool_execution）
            retrieval_docs: List[Dict[str, Any]] = []
            if (md.get("invocation_category") in ("rag_query", "web_search")
                    and step.step_type == "tool_execution"):
                rp = md.get("recall_preview") or ""
                if rp:
                    retrieval_docs = _split_retrieval_documents(str(rp))

            step_otel: Dict[str, Any] = {
                "trace_id": trace_id,
                "span_id": step_span_id,
                "parent_span_id": prev_span_id or turn_root_span,
                "kind": span_kind,
                "step_kind": kind,           # llm_call / host_tool / user_input / agent_event
                "operation_name": op_name,
                "model": step_model,
                "provider": step_provider,
                "model_source": step_model_source,
                "finish_reason": step_finish,
                "finish_reasons": [step_finish],
                "token_usage": step_token,
                "input_messages": input_msgs,
                "output_messages": output_msgs,
                "retrieval_documents": retrieval_docs,
                "attributes": otel_attrs,
            }
            if hasattr(step, "otel"):
                step.otel = step_otel
            prev_span_id = step_span_id

    # ── session-level OTel 视图 ─────────────────────────
    session_token_usage = _build_token_usage(
        model_key, session_input_tokens, session_output_tokens, not has_real_tokens
    )
    session_eval: Optional[Dict[str, Any]] = None
    if response_report:
        scores = response_report.get("scores") or {}
        overall = scores.get("overall")
        session_eval = {
            "metric_name": "response_quality",
            "score": overall,
            "label": (
                "ok" if (isinstance(overall, (int, float)) and overall >= 0.7)
                else ("warn" if (isinstance(overall, (int, float)) and overall >= 0.4)
                      else ("fail" if isinstance(overall, (int, float)) else None))
            ),
            "scores": scores,
            "summary": response_report.get("summary") or "",
            "checked_at": response_report.get("checked_at"),
        }

    # v0.11.2：把 events.jsonl 真实 token 用量算成「真值 token usage + cache-aware cost」
    actual_token_usage: Optional[Dict[str, Any]] = None
    actual_cost_usd: Optional[float] = None
    cost_breakdown: Optional[Dict[str, float]] = None
    if events_meta and events_meta.get("actual_token_usage"):
        atu = events_meta["actual_token_usage"]
        c, bd = _compute_cache_aware_cost_usd(
            model_key,
            atu["input_tokens"],
            atu["output_tokens"],
            atu["cache_read_tokens"],
            atu["cache_write_tokens"],
        )
        # 也算「无 cache 折扣下的全价 cost」用于前端对比展示，让用户直观看到 cache 省了多少
        full_price_cost, _full_reason = _compute_cost_usd(
            model_key, atu["input_tokens"], atu["output_tokens"]
        )
        actual_cost_usd = c
        cost_breakdown = bd if bd else None
        actual_token_usage = {
            **atu,
            "model": model,
            "model_key": model_key,
            "cost_usd": c,
            "cost_breakdown": cost_breakdown,
            "full_price_cost_usd": full_price_cost,
            "cost_reason": "ok" if c is not None else (
                "no_pricing" if model_key is None and model not in (None, "unknown") else "unknown_model"
            ),
            "is_estimate": False,
            "source": "cot-stream/events.jsonl",
        }

    # 给前端的修复 hint
    hints: List[Dict[str, str]] = []
    if model_source == "unknown":
        hints.append({
            "level": "warn",
            "code": "model_unknown",
            "message": (
                "未识别到当前 agent 使用的 LLM model。Cursor transcript 不暴露 model，"
                "请确认本机 ``~/.cursor/hooks/cot-stream.js`` 已挂上（推荐），"
                "events.jsonl 会自动写入真实 model；或在 cot-extractor/.env 里配 "
                "``COT_DEFAULT_MODEL=...`` 作为兜底。"
            ),
        })
    if not model_key and model and model != "unknown":
        hints.append({
            "level": "info",
            "code": "no_pricing",
            "message": (
                f"已识别到 model = ``{model}``，但 pricing 表中没有这个 key，"
                "cost_usd 无法计算。可在 cot_otel_enricher._PRICING_USD_PER_1K 添加条目。"
            ),
        })
    if events_meta and len(events_meta.get("model_distribution") or {}) > 1:
        dist = events_meta["model_distribution"]
        top2 = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:3]
        hints.append({
            "level": "info",
            "code": "multi_model",
            "message": (
                "本 session 内出现多个 model：" +
                ", ".join(f"{m}={c}" for m, c in top2) +
                "。当前以出现次数最多的为 dominant model，per-step 的精确 model "
                "已写入 step.otel.attributes['gen_ai.request.model']。"
            ),
        })

    # OTel GenAI 1.27+ 标准 attribute（session root span 用）
    # 这一块直接给 OTLP exporter / Phoenix / Langfuse / SigNoz 消费
    session_attrs: Dict[str, Any] = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.conversation.id": session_id,
        "gen_ai.agent.name": agent_name,
        "gen_ai.provider.name": provider or "unknown",
        "gen_ai.request.model": model or "unknown",
        "gen_ai.response.model": model or "unknown",
    }
    # Anthropic 扩展：cache token 拆分（Phoenix / Langfuse / SigNoz 都识别这两个 key）
    if actual_token_usage:
        session_attrs["gen_ai.usage.input_tokens"] = int(actual_token_usage.get("input_tokens", 0))
        session_attrs["gen_ai.usage.output_tokens"] = int(actual_token_usage.get("output_tokens", 0))
        if actual_token_usage.get("cache_read_tokens"):
            session_attrs["gen_ai.usage.cache_read_input_tokens"] = int(
                actual_token_usage["cache_read_tokens"]
            )
        if actual_token_usage.get("cache_write_tokens"):
            session_attrs["gen_ai.usage.cache_creation_input_tokens"] = int(
                actual_token_usage["cache_write_tokens"]
            )
    else:
        session_attrs["gen_ai.usage.input_tokens"] = int(session_input_tokens)
        session_attrs["gen_ai.usage.output_tokens"] = int(session_output_tokens)

    otel_view: Dict[str, Any] = {
        "schema": "opentelemetry-genai/0.1 (client-side derived)",
        "trace_id": trace_id,
        "root_span_id": session_root_span,
        "service": {
            "name": "cot-extractor",
            "version": "v0.12.0",
        },
        "model": model,
        "provider": provider,
        "model_source": model_source,
        # v0.13.x: 本 session 内出现过的所有非 default model（多模型场景：
        # 同一会话切了 claude → gpt-5 → glm 都会全部列出来给前端展示）
        "models_seen": (events_meta or {}).get("models_seen") or (
            [model] if model and model != "unknown" else []
        ),
        # 模型切换时间轴（用于前端在 turn 级精确标注）
        "model_timeline": (events_meta or {}).get("model_timeline") or [],
        "agent_name": agent_name,
        "session_id": session_id,
        "conversation_id": session_id,
        "attributes": session_attrs,
        "totals": {
            "turns": len(session_cot.turns),
            "steps": sum(len(t.steps) for t in session_cot.turns),
            "tool_calls": session_cot.total_tool_calls,
            # 估算口径（基于 transcript usage + char/4 fallback）
            "input_tokens": session_input_tokens,
            "output_tokens": session_output_tokens,
            "cost_usd": round(session_cost, 6) if model_key else None,
        },
        "token_usage": session_token_usage,
        # v0.11.2：来自 cot-stream hook 的真实 token + cache 计数（优先级高于 token_usage）
        "actual_token_usage": actual_token_usage,
        "actual_cost_usd": actual_cost_usd,
        # v0.11.2：来自 cot-stream hook 的运行时上下文
        "client_runtime": {
            "cursor_version": (events_meta or {}).get("cursor_version"),
            "user_email": (events_meta or {}).get("user_email"),
            "events_count": (events_meta or {}).get("events_count"),
            "events_path": (events_meta or {}).get("events_path"),
            "model_distribution": (events_meta or {}).get("model_distribution"),
        } if events_meta else None,
        "eval": session_eval,
        "request_params": {
            "temperature": None,
            "top_p": None,
            "max_tokens": None,
            "seed": None,
            "stop_sequences": None,
            "_note": "Cursor / Claude Code transcript 不暴露 LLM 调用参数，前端按 unavailable 渲染",
        },
        "missing_signals": [
            "gen_ai.request.temperature",
            "gen_ai.request.top_p",
            "gen_ai.request.max_tokens",
            "gen_ai.request.seed",
            "gen_ai.system_instructions",
            "gen_ai.tool.definitions",
            "gen_ai.response.id",
            "gen_ai.input.messages (full prompt; transcript 只给 preview)",
            "reasoning.signature (Anthropic redacted_thinking)",
        ],
        "hints": hints,
        "generated_at_ms": int(time.time() * 1000),
    }

    if hasattr(session_cot, "otel_view"):
        session_cot.otel_view = otel_view
    if hasattr(session_cot, "resource_attributes"):
        session_cot.resource_attributes = _build_resource_attributes()


__all__ = [
    "enrich_session_with_otel",
]
