"""mcp_traffic_reader — 把本地 MCP 代理写的流量日志回填到 ThoughtStep.observed_output。

为什么要这个模块：
  Cursor 的 hook 看到的是脱水版 metadata，CallMcpTool 真实返回的 result.content[].text
  经常被截断或丢失（前端"真实返回"长期空白）。本仓库另起了一个本地反向代理
  (cot-extractor/scripts/mcp_traffic_proxy.js)，在 Cursor 和远端 MCP 之间抓 wire 字节，
  完整的 result_summary.full_text 就落在 ~/.agent-cot/mcp-traffic/<server>/YYYY-MM-DD.jsonl。

  本模块负责：
    1. 加载日期窗口内的 traffic JSONL 行（lazy + filter，避免一次扫上 GB）
    2. 在 cot 树里找 CallMcpTool step，按 (server, tool, 调用次序) 配对到 traffic
    3. 把完整 full_text 写到 step.metadata.observed_output.result_text，并打上
       observed_source = "mcp_proxy"，observed_traffic_ts 留给 UI 排查溯源

匹配策略（按可靠性从高到低）：
  A. 时间锚命中：step.metadata.observed_at_ms 已被 _attach_cursor_events 写过，
     在 ±30 s 窗口内找 (server, tool) 匹配的 traffic，取最近的。
  B. 顺序回退：A 找不到 → 把 (server, tool) 维度上 step 序列与 traffic 序列按出现
     先后位置配对。这条用于历史 session（hook 时间锚缺失）或代理刚启用之前的步骤。
  C. 严格容错：完全找不到对应 traffic 就跳过（observed_output 保持原状），
     绝不破坏现有数据。

核心约束：
  - 不覆盖已有真实数据。如果 step.metadata.observed_output 已经有非空 result_text /
    stdout，就追加而不是替换，并且只在确认是 mcp_proxy 的"更全"版本时升级。
  - 失败容错。任何异常都返回 0 / 空，不抛出，避免破坏主提取流程。
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 默认在用户目录的 .agent-cot/mcp-traffic 下找日志，可以用环境变量覆盖
TRAFFIC_DIR_DEFAULT = Path(os.path.expanduser("~/.agent-cot/mcp-traffic"))
TRAFFIC_DIR = Path(os.environ.get("AGENT_COT_MCP_TRAFFIC_DIR", str(TRAFFIC_DIR_DEFAULT)))

# 时间锚命中窗口（毫秒）。Cursor hook 触发时间和代理收到响应时间通常差 < 1s，
# 给 30s 余量覆盖时钟漂移、SSE 长响应、event loop backpressure 等异常。
TIME_ANCHOR_WINDOW_MS = 30_000


@dataclass
class TrafficRecord:
    """一条已解析的 MCP 流量记录。"""
    ts_ms: int
    server: str           # "iWiki" / "shadow-folk" / "gongfengStreamable"
    tool: Optional[str]   # tools/call 的工具名，非 tools/call 时为 None
    args: Optional[Dict[str, Any]]
    full_text: Optional[str]   # result.content[].text 拼接，None = 没拿到
    is_error: bool
    n_blocks: int
    raw_size: int
    elapsed_ms: int
    rpc_method: str
    req_id: Optional[Any]
    # 用过一次就 mark，配对策略 B 用它做顺序游标；这样同一份 record 不会被两个 step 抢
    consumed: bool = False


def _strip_user_prefix(server: str) -> str:
    """Cursor 在 CallMcpTool.tool_input.server 里带 'user-' 前缀（namespace 标记），
    代理日志里没有。统一去掉前缀做匹配。"""
    if server and server.startswith("user-"):
        return server[len("user-"):]
    return server


def _coerce_ts_ms(rec: Dict[str, Any]) -> Optional[int]:
    ts = rec.get("ts")
    if isinstance(ts, (int, float)):
        return int(ts)
    iso = rec.get("ts_iso")
    if isinstance(iso, str):
        try:
            # ISO 8601 with Z
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except ValueError:
            return None
    return None


def _record_from_jsonl_line(line: str) -> Optional[TrafficRecord]:
    """把一行 jsonl 转成 TrafficRecord，解析失败返回 None（不抛错）。"""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except Exception:
        return None
    ts_ms = _coerce_ts_ms(obj)
    if ts_ms is None:
        return None

    server = obj.get("server") or "unknown"
    req = obj.get("req") or {}
    resp = obj.get("resp") or {}
    rpc_method = req.get("method") or "unknown"
    req_id = req.get("id")

    # tools/call 是我们关心的；其他 method 也存一份，但 tool=None
    tool = None
    args = None
    if rpc_method == "tools/call":
        params = req.get("params") or {}
        tool = params.get("name")
        args = params.get("arguments")

    # 找匹配请求 id 的响应消息（resp.messages 数组里）
    full_text: Optional[str] = None
    is_error = False
    n_blocks = 0
    msgs = resp.get("messages") or []
    for m in msgs:
        if req_id is not None and m.get("id") != req_id:
            continue
        summary = m.get("result_summary") or {}
        if summary.get("kind") == "tool_call":
            full_text = summary.get("full_text")
            is_error = bool(summary.get("isError"))
            n_blocks = int(summary.get("n_blocks") or 0)
            break
    return TrafficRecord(
        ts_ms=ts_ms,
        server=server,
        tool=tool,
        args=args,
        full_text=full_text,
        is_error=is_error,
        n_blocks=n_blocks,
        raw_size=int(resp.get("raw_size") or 0),
        elapsed_ms=int(obj.get("elapsed_ms") or 0),
        rpc_method=rpc_method,
        req_id=req_id,
    )


def load_traffic_records(
    ts_from_ms: Optional[int] = None,
    ts_to_ms: Optional[int] = None,
    traffic_dir: Optional[Path] = None,
) -> List[TrafficRecord]:
    """加载落在 [from, to] 时间窗口内的全部 traffic record。

    Why 时间窗口：一条 cot session 通常只跨几小时，给 ±1 天的 buffer 就够了；
    扫所有历史日志会越来越慢。如果窗口为 None 则不限制。
    """
    base = Path(traffic_dir) if traffic_dir else TRAFFIC_DIR
    if not base.is_dir():
        return []

    # 决定要扫哪些 YYYY-MM-DD.jsonl 文件
    candidate_dates: Optional[set] = None
    if ts_from_ms is not None and ts_to_ms is not None:
        candidate_dates = set()
        # 给 ±1 天 buffer 避免边界 session 漏 record
        d_from = datetime.fromtimestamp((ts_from_ms - 86_400_000) / 1000, tz=timezone.utc).date()
        d_to = datetime.fromtimestamp((ts_to_ms + 86_400_000) / 1000, tz=timezone.utc).date()
        d = d_from
        while d <= d_to:
            candidate_dates.add(d.strftime("%Y-%m-%d"))
            d += timedelta(days=1)
        # 同时考虑本地时区（traffic 日志按 UTC ISO 切片，但有些代理可能按 local）
        for delta_h in (8, -8):
            ts_shift = (ts_from_ms - 86_400_000) + delta_h * 3600_000
            candidate_dates.add(datetime.fromtimestamp(ts_shift / 1000, tz=timezone.utc).date().strftime("%Y-%m-%d"))

    out: List[TrafficRecord] = []
    for srv_dir in base.iterdir():
        if not srv_dir.is_dir():
            continue
        for f in srv_dir.glob("*.jsonl"):
            if candidate_dates is not None:
                stem = f.stem
                if stem not in candidate_dates:
                    continue
            try:
                with f.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        rec = _record_from_jsonl_line(line)
                        if rec is None:
                            continue
                        if ts_from_ms is not None and rec.ts_ms < ts_from_ms - TIME_ANCHOR_WINDOW_MS:
                            continue
                        if ts_to_ms is not None and rec.ts_ms > ts_to_ms + TIME_ANCHOR_WINDOW_MS:
                            continue
                        # tools/call 才有 RAG 价值；其它 RPC 也保留方便排查
                        out.append(rec)
            except Exception as e:
                logger.warning("mcp_traffic_reader: skip %s: %s", f, e)
    out.sort(key=lambda r: r.ts_ms)
    return out


# ─── 与 cot 树的合并 ────────────────────────────────────────────────

def _step_mcp_meta(step: Any) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """从一个 ThoughtStep 上抽 (server_normalized, tool_name, arguments)，
    不是 MCP 调用就返回 None。"""
    md = getattr(step, "metadata", None)
    if not isinstance(md, dict):
        return None
    tool_name = step.tool_name or md.get("tool_name")
    if tool_name != "CallMcpTool":
        return None
    ti = md.get("tool_input") or {}
    if not isinstance(ti, dict):
        return None
    server = ti.get("server")
    inner = ti.get("toolName") or ti.get("tool")
    args = ti.get("arguments") or {}
    if not server or not inner:
        return None
    return _strip_user_prefix(server), inner, args if isinstance(args, dict) else {}


def _existing_real_text(step: Any) -> Optional[str]:
    """Step 当前 observed_output 上是不是已经有真实 result_text。
    用于决定是否需要从 traffic 升级（避免覆盖更全的来源）。"""
    md = getattr(step, "metadata", None)
    if not isinstance(md, dict):
        return None
    out = md.get("observed_output")
    if not isinstance(out, dict):
        return None
    rt = out.get("result_text")
    if isinstance(rt, str) and rt.strip():
        # 排除空壳 JSON 占位
        sanity = rt.strip()
        if sanity not in (
            '{"content":[{"type":"text","text":""}],"isError":false}',
            '{"content":[],"isError":false}',
        ):
            return rt
    return None


def _attach_text_to_step(
    step: Any,
    rec: TrafficRecord,
) -> bool:
    """把 traffic 文本写到 step.metadata.observed_output；不破坏已有字段。
    返回是否真的更新了。"""
    md = step.metadata if isinstance(step.metadata, dict) else {}
    if not isinstance(step.metadata, dict):
        step.metadata = md  # type: ignore[attr-defined]

    out = md.get("observed_output")
    if not isinstance(out, dict):
        out = {}
    # 追加 result_text，保留 stdout/stderr/exit_code 等其它字段
    out["result_text"] = rec.full_text or ""
    out["result_is_error"] = rec.is_error
    out["result_n_blocks"] = rec.n_blocks
    out["result_raw_size"] = rec.raw_size
    md["observed_output"] = out
    # 多源 source 用列表 / 主源 + 辅源记录，方便前端 pill
    prev_source = md.get("observed_source")
    if prev_source and prev_source != "mcp_proxy":
        md["observed_source"] = "mcp_proxy+" + str(prev_source)
    else:
        md["observed_source"] = "mcp_proxy"
    md["mcp_traffic_ts_ms"] = rec.ts_ms
    md["mcp_traffic_elapsed_ms"] = rec.elapsed_ms
    if md.get("observed_at_ms") is None:
        md["observed_at_ms"] = rec.ts_ms
    return True


def _enumerate_mcp_steps(turns_cot: List[Any]) -> List[Tuple[Any, Any, str, str, Dict[str, Any]]]:
    """Yield 所有 (turn, step, server, tool, args) 三元组，按 (turn, step_index) 升序。

    只取 tool_decision 类型，因为 tool_input 在它身上；同时记录其紧邻的 tool_execution，
    UI 实际渲染"真实返回"用的是 execution step 上的 observed_output。
    """
    out = []
    for turn in turns_cot:
        steps_sorted = sorted(turn.steps, key=lambda s: s.step_index)
        for s in steps_sorted:
            if s.step_type != "tool_decision":
                continue
            meta = _step_mcp_meta(s)
            if meta is None:
                continue
            server, tool, args = meta
            out.append((turn, s, server, tool, args))
    return out


def _find_partner_execution(turn: Any, decision_step: Any) -> Optional[Any]:
    """tool_decision 之后第一条 tool_execution。前端"真实返回"显示在它身上。"""
    di = decision_step.step_index
    candidate = None
    for s in sorted(turn.steps, key=lambda s: s.step_index):
        if s.step_index <= di:
            continue
        if s.step_type == "tool_execution":
            candidate = s
            break
        # 只让"紧邻 N 步内"的 execution 算 partner，避免越过下一个 thinking
        if s.step_type in ("thinking_inter", "thinking_explicit", "tool_decision", "final_response"):
            break
    return candidate


def attach_mcp_traffic(
    turns_cot: List[Any],
    *,
    session_start_ms: Optional[int] = None,
    session_end_ms: Optional[int] = None,
    traffic_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """主入口：扫 MCP step 并尝试用 traffic 数据回填 result_text。

    返回 stats，便于上游写到 SessionCoT.observed_events。
    任何异常都被吞掉记 stats，不向上抛。
    """
    stats: Dict[str, Any] = {
        "available": False,
        "records_loaded": 0,
        "mcp_steps_total": 0,
        "matched_by_time_anchor": 0,
        "matched_by_sequence": 0,
        "matched_total": 0,
        "skipped_already_real": 0,
        "unmatched": 0,
    }

    try:
        records = load_traffic_records(session_start_ms, session_end_ms, traffic_dir=traffic_dir)
    except Exception as e:
        logger.warning("mcp_traffic_reader: load failed: %s", e)
        stats["error"] = str(e)
        return stats

    stats["records_loaded"] = len(records)
    if not records:
        return stats

    # 只保留 tools/call 的 record（其它 RPC 没意义）
    tool_records = [r for r in records if r.tool is not None and r.full_text is not None]
    if not tool_records:
        stats["available"] = True
        return stats

    # 按 (server, tool) 分桶，配对策略 B 要按出现顺序消费
    seq_buckets: Dict[Tuple[str, str], List[TrafficRecord]] = {}
    for rec in tool_records:
        seq_buckets.setdefault((rec.server, rec.tool), []).append(rec)

    mcp_steps = _enumerate_mcp_steps(turns_cot)
    stats["mcp_steps_total"] = len(mcp_steps)
    stats["available"] = True

    for turn, decision, server, tool, args in mcp_steps:
        partner_exec = _find_partner_execution(turn, decision)
        # 跳过那些已经有真实 result_text 的 step（多半是 hook 直接抓到的，
        # 那种不需要从代理回填）
        if _existing_real_text(decision) is not None or (
            partner_exec is not None and _existing_real_text(partner_exec) is not None
        ):
            stats["skipped_already_real"] += 1
            continue

        bucket_key = (server, tool)
        bucket = seq_buckets.get(bucket_key) or []
        if not bucket:
            stats["unmatched"] += 1
            continue

        chosen: Optional[TrafficRecord] = None
        anchor_ms: Optional[int] = None
        md = decision.metadata if isinstance(decision.metadata, dict) else {}
        if isinstance(md, dict):
            anchor_ms = md.get("observed_at_ms")
            if not isinstance(anchor_ms, (int, float)):
                anchor_ms = None

        # A. 时间锚优先
        if anchor_ms is not None:
            best_dt = None
            for rec in bucket:
                if rec.consumed:
                    continue
                dt = abs(rec.ts_ms - int(anchor_ms))
                if dt > TIME_ANCHOR_WINDOW_MS:
                    continue
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    chosen = rec
            if chosen is not None:
                stats["matched_by_time_anchor"] += 1

        # B. 顺序回退
        if chosen is None:
            for rec in bucket:
                if not rec.consumed:
                    chosen = rec
                    break
            if chosen is not None:
                stats["matched_by_sequence"] += 1

        if chosen is None:
            stats["unmatched"] += 1
            continue

        chosen.consumed = True
        stats["matched_total"] += 1

        # 同时回填到 decision 和 partner_exec —— UI 主要看 partner_exec，
        # 但 decision 上挂一份方便排查工具入参 ↔ 实际返回的对照
        _attach_text_to_step(decision, chosen)
        if partner_exec is not None:
            _attach_text_to_step(partner_exec, chosen)

    return stats


def attach_mcp_traffic_to_session(session_cot: Any, *, traffic_dir: Optional[Path] = None) -> Dict[str, Any]:
    """对外的便捷封装：传入 SessionCoT，自动从 turns 取时间窗口。"""
    turns = getattr(session_cot, "turns", []) or []
    if not turns:
        return {"available": False, "records_loaded": 0}
    # 取 [first.start, last.end] 的一个保守窗口
    starts = []
    ends = []
    for t in turns:
        for k in ("turn_start_ms_observed", "turn_start_time"):
            v = getattr(t, k, None)
            if isinstance(v, (int, float)):
                starts.append(int(v))
        for k in ("turn_end_ms_observed",):
            v = getattr(t, k, None)
            if isinstance(v, (int, float)):
                ends.append(int(v))
    s_ms = min(starts) if starts else None
    e_ms = max(ends) if ends else None
    return attach_mcp_traffic(
        turns,
        session_start_ms=s_ms,
        session_end_ms=e_ms,
        traffic_dir=traffic_dir,
    )
