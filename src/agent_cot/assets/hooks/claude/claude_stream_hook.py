#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""claude_stream_hook.py - per-session local events.jsonl stream for Claude.

v0.15.0 引入。Claude 27 个 hook 里有相当一部分事件 transcript 不会原生记录
（PreCompact / PostCompact / SubagentStart/Stop / Notification / Permission
/ Elicitation 等），而这些恰恰是前端"补齐 IDE 真值时间线"的关键来源。

本 hook 的职责：
  1. 读 stdin（UTF-8 强制），解析为 JSON dict（解析失败也兜底落原文）
  2. 从 payload 拿 ``hook_event_name`` / ``session_id`` / ``transcript_path``
  3. **追加一行** JSONL 到 ``~/.claude/state/events/<session_id>/events.jsonl``
  4. **v0.19.1 新增**：在 Stop / SubagentStop / SessionEnd / StopFailure 触发时
     后台 spawn ``extract_cot.py --session-id <sid> --transcript <tp>``，把
     transcript 转成 cot.json 落到 ``<data_root>/cot/<sid>_cot.json``，让 backend
     SessionList 立刻能看到这次 Claude 会话（之前唯一能跑 extract 的途径是用户
     手动跑 CLI，结果就是『跟 Claude 聊过但前端永远空』——见
     iwiki §13/Claude Pipeline Gap）。
  5. 永不向 stdout / stderr 写入任何内容（Claude 对 stderr 视为 hook error）
  6. 任何异常都吞掉，``exit 0``——绝不阻塞 IDE

它跟 langfuse_hook 完全独立：
  * langfuse_hook    → 上行到 Langfuse Cloud（依赖 SDK + 网络）
  * claude_stream_hook → 落地本地 events.jsonl + 后台触发 extract_cot
                         （无任何网络依赖，零失败成本）

cot_extractor 在 ``agent_type='claude'`` 时会读这个 events.jsonl 把
``compact_events`` / ``subagent_timeline`` / ``permission_events`` /
``notification_events`` 这四条新时间线注入到 SessionCoT。文件不存在时这些
字段保持空数组，前端零回归。

extract_cot 路径解析采用与 cot-bridge.js / cot-stream.js 相同的 4 层：
  1. 环境变量 ``COT_EXTRACTOR_ROOT`` / ``COT_PYTHON``
  2. ``~/.agent-cot/runtime.json`` 里的 ``cot_extractor_root`` / ``python_executable``
  3. ``~/.cursor-cot/runtime.json`` 同上 (兼容 cursor-cot 已经在跑的用户)
  4. 多个常见安装路径探测 + ``sys.executable``

任何一层都没找到就跳过（不打印不抛错），events.jsonl 仍照写。

安装方法（可选 / 用户自行决定）：

    python claude-code/hooks/claude_stream_hook.py --install

会把自身注册为所有 27 个 hook 事件的并行命令（紧跟现有 langfuse_hook 之后），
不破坏已有的 langfuse_hook / cot_hook / auto_check / export_transcript 命令。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

def _claude_home() -> Path:
    env = os.environ.get("AGENT_COT_CLAUDE_HOME")
    if env:
        return Path(env).expanduser()
    try:
        script = Path(__file__).resolve()
        if script.parent.name == "hooks" and script.parent.parent.name in {
            ".claude",
            ".claude-internal",
            ".claude-inertnal",
        }:
            return script.parent.parent
    except Exception:
        pass
    return Path.home() / ".claude"


CLAUDE_HOME = _claude_home()
STATE_DIR = CLAUDE_HOME / "state"
EVENTS_DIR = STATE_DIR / "events"

# v0.20.0：统一 pipeline.log。Cursor / CodeBuddy / Claude / extractor / backend
# 都向这里追加一行 breadcrumb，方便 ``tail -f`` 一眼看到链路在哪里断。
_PIPELINE_LOG = Path(
    os.environ.get("AGENT_COT_PIPELINE_LOG")
    or str(Path.home() / ".agent-cot" / "logs" / "pipeline.log")
).expanduser()


def _pipeline_log(event: str, sid: str = "-", ok: bool = True, **note: Any) -> None:
    """Append one breadcrumb to the unified pipeline log. Never raises."""
    try:
        _PIPELINE_LOG.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime as _dt, timezone as _tz
        ts = _dt.now(_tz.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        parts = []
        for k, v in note.items():
            if v is None:
                continue
            s = str(v)
            if " " in s or '"' in s:
                s = '"' + s.replace('"', '\\"') + '"'
            parts.append(f"{k}={s}")
        line = (
            f"[{ts}] [hook.claude] [claude] [sid={sid}] "
            f"event={event} status={'ok' if ok else 'FAIL'}"
            + (" " + " ".join(parts) if parts else "")
            + "\n"
        )
        with open(_PIPELINE_LOG, "a", encoding="utf-8", errors="replace") as f:
            f.write(line)
    except Exception:
        pass

# 当 stdin 为非 JSON 文本时，仍尝试用正则抢救出几个关键字段，跟 langfuse_hook
# 的 fallback 一致——任何能拿到 session_id / hook_event_name 的事件都比丢掉好。
_FIELD_RE = {
    "session_id": re.compile(r'"session_id"\s*:\s*"([^"]*)"'),
    "hook_event_name": re.compile(r'"hook_event_name"\s*:\s*"([^"]*)"'),
    "transcript_path": re.compile(r'"transcript_path"\s*:\s*"((?:[^"\\]|\\.)*)"'),
    "tool_name": re.compile(r'"tool_name"\s*:\s*"([^"]*)"'),
    "tool_use_id": re.compile(r'"tool_use_id"\s*:\s*"([^"]*)"'),
}


def _force_utf8_stdin() -> None:
    """强制 stdin 用 UTF-8 解码，跟 langfuse_hook 一致。"""
    try:
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        elif hasattr(sys.stdin, "buffer"):
            import io
            sys.stdin = io.TextIOWrapper(
                sys.stdin.buffer, encoding="utf-8", errors="replace"
            )
    except Exception:
        pass


def _read_payload() -> Dict[str, Any]:
    """读 stdin → dict。优先 JSON，失败回落正则抢救关键字段。

    这个函数被设计成**绝不抛**——任何异常都返回空 dict 让 main 走 no-op。
    """
    try:
        data = sys.stdin.read() or ""
    except Exception:
        return {}
    if not data.strip():
        return {}
    raw = data
    try:
        return json.loads(data)
    except Exception:
        pass
    # 二次清洗：删掉非打印字符再试
    try:
        cleaned = "".join(ch for ch in data if ch.isprintable() or ch in "\n\r\t")
        return json.loads(cleaned)
    except Exception:
        pass
    # 末招：正则提关键字段，附原文便于后期排查
    out: Dict[str, Any] = {"_raw_first200": raw[:200], "_raw_chars": len(raw)}
    for k, regex in _FIELD_RE.items():
        m = regex.search(raw)
        if m:
            out[k] = m.group(1)
    return out


def _detect_event_name(payload: Dict[str, Any]) -> str:
    """决定本次 hook 是哪个事件触发——优先级如下：

      1. payload['hook_event_name']   （最权威）
      2. 环境变量 CLAUDE_HOOK_EVENT  （某些版本会注入）
      3. argv[1]                     （install 时会显式带上）
      4. 'Unknown'
    """
    name = payload.get("hook_event_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    env_name = os.environ.get("CLAUDE_HOOK_EVENT") or os.environ.get("HOOK_EVENT_NAME")
    if env_name and env_name.strip():
        return env_name.strip()
    if len(sys.argv) >= 2 and sys.argv[1] not in ("--install", "--uninstall"):
        return str(sys.argv[1])
    return "Unknown"


def _build_record(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从 payload 构造一条 events.jsonl 行。session_id 缺失则返回 None（丢弃）。

    输出 schema 跟 cursor cot-stream.js 的 events.jsonl 大致对齐，但保留
    Claude 特有字段：
      {
        "t_ms":            wall-clock 毫秒时间戳（int）
        "iso_ts":          ISO8601 时间字符串（带 Z）
        "session_id":      session uuid
        "hook_event":      "PreCompact" | "SubagentStop" | ...
        "transcript_path": 主 transcript 路径（如有）
        "tool_name":       工具名（如有）
        "tool_use_id":     工具调用 id（如有）
        "payload":         原 payload 全文（已剥掉冗余字段，方便读）
      }
    """
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        return None
    now = datetime.now(timezone.utc)
    record: Dict[str, Any] = {
        "t_ms": int(now.timestamp() * 1000),
        "iso_ts": now.isoformat().replace("+00:00", "Z"),
        "session_id": session_id.strip(),
        "hook_event": _detect_event_name(payload),
    }
    for k in ("transcript_path", "tool_name", "tool_use_id", "cwd"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            record[k] = v.strip()
    # 把整个 payload 一并落盘但限制大小，防止个别 PostToolUse 带 100k+
    # 的工具结果撑爆 events.jsonl。超长的截断 + 标记 truncated。
    try:
        raw = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        raw = ""
    MAX_PAYLOAD = 32 * 1024  # 32KB / event 已经够用
    if len(raw) > MAX_PAYLOAD:
        record["payload_truncated"] = True
        record["payload_chars_full"] = len(raw)
        record["payload"] = raw[:MAX_PAYLOAD]
    else:
        record["payload"] = payload
    return record


def _append_event(record: Dict[str, Any]) -> None:
    """原子追加一行 JSONL 到 ``events/<session_id>/events.jsonl``。

    采用 ``open(... 'a')`` 的简单追加——Claude 的 hook 串行触发，不存在多
    进程并发写同一文件的真实风险（即便有，单行 JSONL 即使 interleave 也
    最多损失少数行，远比丢全部数据好）。
    """
    sid = record["session_id"]
    sess_dir = EVENTS_DIR / sid
    try:
        sess_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    target = sess_dir / "events.jsonl"
    try:
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str))
            f.write("\n")
    except Exception:
        return


# ═══════════════════════════════════════════════════════════
#  v0.19.1: 后台触发 extract_cot.py —— 让 Claude session 也能出现在
#  前端 SessionList。镜像 cot-bridge.js 的 4 层路径解析。
# ═══════════════════════════════════════════════════════════

# 只在这些事件触发 extract，避免每个工具调用都 spawn 一次 Python。
# Stop = 一次 turn 结束；SubagentStop = subagent turn 结束（Task 内部用 subagent
# 时常见，但 transcript 仍写到主 sid）；SessionEnd / StopFailure = 兜底。
_EXTRACT_TRIGGER_EVENTS = ("Stop", "SubagentStop", "SessionEnd", "StopFailure")
_CRITIC_TRIGGER_EVENTS = ("Stop", "SubagentStop", "SessionEnd", "StopFailure")

# 防抖：同一 session 60 秒内只 spawn 一次 extract（Stop 可能短时间内被 Claude
# 触发多次，例如用户连续按了取消 / 重发）。状态写到 ~/.claude/state/extract_debounce.json
_DEBOUNCE_WINDOW_SEC = 30
_DEBOUNCE_FILE = STATE_DIR / "extract_debounce.json"
_CRITIC_DEBOUNCE_FILE = STATE_DIR / "critic_debounce.json"


def _read_runtime_state(name: str) -> Optional[Dict[str, Any]]:
    """读 ``~/<name>/runtime.json``。失败返回 None，绝不抛。"""
    try:
        p = Path.home() / name / "runtime.json"
        if not p.is_file():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _looks_like_extractor_root(p: Path) -> bool:
    """判断给定路径是不是 cot-extractor 根 —— 必须含 scripts/extract_cot.py。"""
    try:
        return (p / "scripts" / "extract_cot.py").is_file()
    except Exception:
        return False


def _resolve_extractor_root() -> Optional[Path]:
    """4 层找 cot-extractor 根：env > agent-cot runtime > cursor-cot runtime > 探测。"""
    env_v = os.environ.get("COT_EXTRACTOR_ROOT")
    if env_v:
        p = Path(env_v).expanduser()
        if _looks_like_extractor_root(p):
            return p
    for ns in (".agent-cot", ".cursor-cot"):
        st = _read_runtime_state(ns)
        if st:
            v = st.get("cot_extractor_root")
            if isinstance(v, str) and v.strip():
                p = Path(v).expanduser()
                if _looks_like_extractor_root(p):
                    return p
    # Fallback 探测：开发树常见位置
    candidates = [
        Path.home() / ".agent-cot" / "cot-extractor",
        Path.home() / ".cursor-cot" / "cot-extractor",
    ]
    for c in candidates:
        if _looks_like_extractor_root(c):
            return c
    return None


def _resolve_python() -> str:
    """python 解释器：env > runtime > sys.executable。"""
    env_v = os.environ.get("COT_PYTHON")
    if env_v:
        return env_v
    for ns in (".agent-cot", ".cursor-cot"):
        st = _read_runtime_state(ns)
        if st:
            v = st.get("python_executable")
            if isinstance(v, str) and v.strip():
                return v
    return sys.executable or "python"


def _is_agent_quality_eval_exe(py: str) -> bool:
    name = Path(str(py or "")).name.lower()
    return name.startswith("agent-quality-eval") and name.endswith(".exe")


def _agent_quality_eval_runner_cmd(py: str, runner: str, args: list[str]) -> list[str]:
    if _is_agent_quality_eval_exe(py):
        return [py, "--agent-quality-eval-runner", runner, *args]
    if runner == "live-critic":
        code = "from agent_quality_eval.evaluation.live_critic import main; raise SystemExit(main())"
    else:
        code = "from agent_quality_eval.evaluation.critic import main; raise SystemExit(main())"
    return [py, "-c", code, *args]


def _normalize_data_root(p: Path) -> Path:
    """v0.19.3：data_root 自愈，纠正历史 init 把 runtime.json.data_root 写成
    ``~/.agent-cot``（无 /data）的损坏状态。

    判定规则：路径末段为 ``.agent-cot`` 或 ``.cursor-cot`` 时（agent-cot
    的 root 本身），自动补上 ``/data``；其他情况（包括用户已正确设置成
    ``...\\data`` 结尾，或自定义了完全不同的目录）保持原样。

    跟 cursor / codebuddy hook 的 _normalizeDataRoot 算法完全等价。
    """
    name = p.name
    if name == "data":
        return p
    if name in (".agent-cot", ".cursor-cot"):
        return p / "data"
    return p


def _resolve_data_root() -> Path:
    """``cot.json`` 落盘位置：env > runtime > 默认 ~/.agent-cot/data。

    cot-extractor 自己也会同样解析 ``AGENT_COT_DATA_ROOT``（见
    extract_cot.py:_resolve_output_dir），所以这里我们用环境变量传给子进程
    就够了；只有取不到任何线索时才落到 ``~/.agent-cot/data``。

    v0.19.3：runtime.json 中的 data_root 经过 _normalize_data_root 自愈，
    防止历史损坏状态被传播下去。env 显式值不做自愈（尊重使用者）。
    """
    env_v = os.environ.get("AGENT_COT_DATA_ROOT")
    if env_v:
        return Path(env_v).expanduser()
    for ns in (".agent-cot", ".cursor-cot"):
        st = _read_runtime_state(ns)
        if st:
            v = st.get("data_root")
            if isinstance(v, str) and v.strip():
                return _normalize_data_root(Path(v).expanduser())
    return Path.home() / ".agent-cot" / "data"


def _debounce_should_run(sid: str) -> bool:
    """同一 sid 在 _DEBOUNCE_WINDOW_SEC 内只允许跑一次 extract。"""
    now = time.time()
    state: Dict[str, float] = {}
    try:
        if _DEBOUNCE_FILE.is_file():
            raw = _DEBOUNCE_FILE.read_text(encoding="utf-8")
            loaded = json.loads(raw) if raw.strip() else {}
            if isinstance(loaded, dict):
                # cleanup stale entries (>1h)
                state = {k: float(v) for k, v in loaded.items()
                         if isinstance(v, (int, float)) and now - float(v) < 3600}
    except Exception:
        state = {}
    last = state.get(sid, 0.0)
    if now - last < _DEBOUNCE_WINDOW_SEC:
        return False
    state[sid] = now
    try:
        _DEBOUNCE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _DEBOUNCE_FILE.write_text(
            json.dumps(state, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass
    return True


def _critic_debounce_should_run(sid: str, ev: str) -> bool:
    now = time.time()
    state: Dict[str, float] = {}
    try:
        if _CRITIC_DEBOUNCE_FILE.is_file():
            raw = _CRITIC_DEBOUNCE_FILE.read_text(encoding="utf-8")
            loaded = json.loads(raw) if raw.strip() else {}
            if isinstance(loaded, dict):
                state = {k: float(v) for k, v in loaded.items()
                         if isinstance(v, (int, float)) and now - float(v) < 3600}
    except Exception:
        state = {}
    key = f"{sid}:{ev}"
    if now - state.get(key, 0.0) < _DEBOUNCE_WINDOW_SEC:
        return False
    state[key] = now
    try:
        _CRITIC_DEBOUNCE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CRITIC_DEBOUNCE_FILE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return True


def _maybe_trigger_extract(record: Dict[str, Any]) -> None:
    """在 Stop-类事件 + 拿得到 transcript_path 时后台跑 extract_cot。

    设计原则：
      * 任何错误一律吞掉（hook 失败会让 Claude 显示 hook 红字）
      * 一律 ``Popen + DETACHED_PROCESS``，主进程立即返回
      * 子进程 stdout/stderr 重定向到 ``~/.claude/state/extract.log``，
        方便排查 / 不污染 stdout
      * 子进程 cwd = extractor_root（让 ``import cot_extractor`` 找到 src）
    """
    try:
        ev = record.get("hook_event") or ""
        if ev not in _EXTRACT_TRIGGER_EVENTS:
            return
        sid = record.get("session_id") or ""
        if not sid:
            return
        tp = record.get("transcript_path") or ""
        if not tp or not Path(tp).is_file():
            # 没 transcript 路径就跑不出 cot.json，静默 skip
            return
        if not _debounce_should_run(sid):
            return
        extractor_root = _resolve_extractor_root()
        if not extractor_root:
            return
        py = _resolve_python()
        data_root = _resolve_data_root()
        # 给子进程注入 AGENT_COT_DATA_ROOT，让 extract_cot._resolve_output_dir
        # 拿到正确的写盘根。同时把 PYTHONPATH 加上 extractor src。
        env = os.environ.copy()
        env["AGENT_COT_DATA_ROOT"] = str(data_root)
        env["AGENT_COT_PIPELINE_LOG"] = str(_PIPELINE_LOG)
        env["PYTHONIOENCODING"] = "utf-8"
        _pipeline_log(
            "extract_spawn",
            sid=sid,
            python=py,
            cot_root=str(extractor_root),
            data_root=str(data_root),
        )
        # 日志：写到 ~/.claude/state/extract.log，便于事后排查
        log_path = STATE_DIR / "extract.log"
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            log_fh = open(log_path, "a", encoding="utf-8", errors="replace")
            log_fh.write(
                f"\n[{datetime.now(timezone.utc).isoformat()}Z] "
                f"event={ev} sid={sid} tp={tp} "
                f"extractor={extractor_root} python={py} "
                f"data_root={data_root}\n"
            )
            log_fh.flush()
        except Exception:
            log_fh = None  # 即便日志写不动也不影响主流程
        cmd = [
            py,
            str(extractor_root / "scripts" / "extract_cot.py"),
            "--session-id", sid,
            "--transcript", tp,
            "--no-upload",
        ]
        # Windows: DETACHED_PROCESS = 0x00000008，让子进程脱离父控制台
        # 不存在的 flag 在非 Windows 上会被忽略
        creationflags = 0
        if sys.platform == "win32":
            creationflags = 0x00000008 | 0x00000200  # DETACHED + NEW_PROCESS_GROUP
        import subprocess
        try:
            subprocess.Popen(
                cmd,
                cwd=str(extractor_root),
                env=env,
                stdout=log_fh if log_fh else subprocess.DEVNULL,
                stderr=log_fh if log_fh else subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                close_fds=True,
            )
        except Exception:
            pass
        # 注意：不 close log_fh —— 让子进程继续写
    except Exception:
        # 任何异常一律吞，hook 必须保持零失败
        return


def _maybe_trigger_critic(record: Dict[str, Any]) -> None:
    try:
        ev = str(record.get("hook_event") or "")
        if ev not in _CRITIC_TRIGGER_EVENTS:
            return
        if os.environ.get("AGENT_QUALITY_EVAL_CRITIC_DISABLE"):
            return
        sid = str(record.get("session_id") or "").strip()
        if not sid or not _critic_debounce_should_run(sid, ev):
            return
        py = _resolve_python()
        cmd = _agent_quality_eval_runner_cmd(py, "critic", [
            "--agent-type",
            "claude",
            "--source-event",
            f"claude-stream:{ev}",
            "--session-id",
            sid,
            "--wait-seconds",
            "75",
            "--no-persist-eval",
        ])
        env = os.environ.copy()
        env["AGENT_QUALITY_EVAL_CRITIC_DISABLE"] = "1"
        env["AGENT_COT_DATA_ROOT"] = str(_resolve_data_root())
        env.setdefault("PYTHONIOENCODING", "utf-8")
        creationflags = 0
        if sys.platform == "win32":
            creationflags = 0x00000008 | 0x00000200
        import subprocess
        try:
            log_path = Path.home() / ".agent-cot" / "logs" / "critic-runner.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = open(log_path, "a", encoding="utf-8", errors="replace")
        except Exception:
            log_fh = None
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(Path.home()),
                env=env,
                stdout=log_fh if log_fh else subprocess.DEVNULL,
                stderr=log_fh if log_fh else subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                close_fds=True,
            )
            _pipeline_log("critic_spawn", sid=sid, pid=proc.pid, source_event=ev)
        except Exception as exc:
            _pipeline_log("critic_spawn", sid=sid, ok=False, error=exc)
    except Exception:
        return


def _maybe_trigger_live_critic(record: Dict[str, Any]) -> None:
    try:
        if os.environ.get("AGENT_QUALITY_EVAL_LIVE_CRITIC_DISABLE"):
            return
        sid = str(record.get("session_id") or "").strip()
        ev = str(record.get("hook_event") or "")
        if not sid or not ev:
            return
        py = _resolve_python()
        cmd = _agent_quality_eval_runner_cmd(py, "live-critic", [
            "--agent-type",
            "claude",
            "--source-event",
            ev,
            "--session-id",
            sid,
        ])
        payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
        slim_payload = {
            key: payload.get(key)
            for key in (
                "hook_event_name",
                "tool",
                "tool_name",
                "toolName",
                "tool_use_id",
                "cwd",
            )
            if payload.get(key) is not None
        }
        if slim_payload:
            cmd.extend(["--payload-json", json.dumps(slim_payload, ensure_ascii=False)])
        env = os.environ.copy()
        env["AGENT_QUALITY_EVAL_LIVE_CRITIC_DISABLE"] = "1"
        env["AGENT_COT_DATA_ROOT"] = str(_resolve_data_root())
        env.setdefault("PYTHONIOENCODING", "utf-8")
        creationflags = 0
        if sys.platform == "win32":
            creationflags = 0x00000008 | 0x00000200
        import subprocess
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(Path.home()),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                close_fds=True,
            )
            _pipeline_log("live_critic_pulse", sid=sid, pid=proc.pid, source_event=ev)
        except Exception as exc:
            _pipeline_log("live_critic_pulse", sid=sid, ok=False, error=exc)
    except Exception:
        return


# ═══════════════════════════════════════════════════════════
#  --install / --uninstall：把自己挂到 settings.json 全部 27 个 hook 上
# ═══════════════════════════════════════════════════════════

ALL_EVENTS = [
    "SessionStart", "SessionEnd", "Setup",
    "UserPromptSubmit", "Stop", "StopFailure",
    "PreToolUse", "PostToolUse", "PostToolUseFailure",
    "SubagentStart", "SubagentStop",
    "TaskCreated", "TaskCompleted", "TeammateIdle",
    "PreCompact", "PostCompact",
    "PermissionRequest", "PermissionDenied",
    "Notification", "Elicitation", "ElicitationResult",
    "ConfigChange", "InstructionsLoaded", "CwdChanged",
    "FileChanged", "WorktreeCreate", "WorktreeRemove",
]


def _hook_command_for(event: str, claude_home: Path | None = None) -> Dict[str, Any]:
    """生成挂在 settings.json 里的 command 项。

    必须用绝对路径——Windows 下 Claude Code 派发 hook 时 ``$HOME`` 不会被
    cmd.exe 展开，导致 ``python "$HOME/..."`` 直接报 "can't open file"
    死亡（v0.15.0 现场实测：events.jsonl 完全没写入即此原因）。

    用 forward slash 是为了同一路径在 Windows / POSIX 都能跑：Python
    的 ``open()`` / cmd.exe 都能识别 forward slash 形式。
    """
    home = claude_home or CLAUDE_HOME
    script_path = str(home / "hooks" / "claude_stream_hook.py").replace("\\", "/")
    return {
        "type": "command",
        # 把事件名作为 argv[1] 传，以防 payload 里没有 hook_event_name
        "command": f'python "{script_path}" {event}',
    }


def _install(settings_path: Path) -> int:
    """把本 hook 注册到 settings.json 上，每个事件 idempotent。

    Upgrade-safe：每次 install 都会先把任何引用了 ``claude_stream_hook.py``
    的旧命令（包括 ``$HOME`` / 不同路径变体）清掉再写入新的绝对路径版本，
    避免命令字符串改变后产生重复挂载或残留死命令。
    """
    if not settings_path.exists():
        sys.stderr.write(f"settings.json not found at {settings_path}\n")
        return 1
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception as e:
        sys.stderr.write(f"failed to parse settings.json: {e}\n")
        return 1
    hooks = data.setdefault("hooks", {})
    # 第一步：清掉所有旧的 claude_stream_hook 命令（任何路径变体）
    purged = 0
    for ev, groups in list(hooks.items()):
        if not isinstance(groups, list):
            continue
        for g in groups:
            if not isinstance(g, dict):
                continue
            cmds = g.get("hooks") or []
            new_cmds = [
                c for c in cmds
                if not (
                    isinstance(c, dict)
                    and "claude_stream_hook.py" in str(c.get("command", ""))
                )
            ]
            if len(new_cmds) != len(cmds):
                purged += len(cmds) - len(new_cmds)
                g["hooks"] = new_cmds
    # 第二步：每个事件追加新的绝对路径命令
    added = 0
    for ev in ALL_EVENTS:
        groups = hooks.setdefault(ev, [{"hooks": []}])
        if not groups:
            groups.append({"hooks": []})
        cmds = groups[0].setdefault("hooks", [])
        cmds.append(_hook_command_for(ev, settings_path.parent))
        added += 1
    settings_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sys.stdout.write(
        f"installed claude_stream_hook on {added} events; "
        f"purged {purged} stale commands; total={len(ALL_EVENTS)}\n"
    )
    return 0


def _uninstall(settings_path: Path) -> int:
    if not settings_path.exists():
        return 0
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        return 1
    hooks = data.get("hooks") or {}
    removed = 0
    for ev, groups in list(hooks.items()):
        if not isinstance(groups, list):
            continue
        for g in groups:
            if not isinstance(g, dict):
                continue
            cmds = g.get("hooks") or []
            new_cmds = [
                c for c in cmds
                if not (
                    isinstance(c, dict)
                    and "claude_stream_hook.py" in str(c.get("command", ""))
                )
            ]
            if len(new_cmds) != len(cmds):
                removed += len(cmds) - len(new_cmds)
                g["hooks"] = new_cmds
    settings_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sys.stdout.write(f"uninstalled claude_stream_hook from {removed} command slots\n")
    return 0


# ═══════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════

def main() -> int:
    # --install / --uninstall 是 CLI 入口，不读 stdin
    argv = sys.argv[1:]
    if argv and argv[0] in ("--install", "--uninstall"):
        settings_path = CLAUDE_HOME / "settings.json"
        # 第二个参数允许显式指定 settings.json 路径（项目级也可）
        if len(argv) >= 2 and argv[1]:
            settings_path = Path(argv[1])
        if argv[0] == "--install":
            return _install(settings_path)
        else:
            return _uninstall(settings_path)

    _force_utf8_stdin()
    try:
        payload = _read_payload()
        record = _build_record(payload)
        if record is not None:
            _append_event(record)
            _pipeline_log(
                record["hook_event"],
                sid=record["session_id"],
                tool=record.get("tool_name"),
                transcript=record.get("transcript_path"),
            )
            # v0.19.1: Stop / SubagentStop / SessionEnd / StopFailure 时
            # 后台 spawn extract_cot.py，让 cot.json 写到 backend 扫的目录
            _maybe_trigger_live_critic(record)
            _maybe_trigger_extract(record)
            _maybe_trigger_critic(record)
        else:
            _pipeline_log(
                "drop",
                sid="-",
                ok=False,
                error="record build returned None (likely no session_id in payload)",
            )
    except Exception as e:
        _pipeline_log("hook_exception", ok=False, error=type(e).__name__ + ": " + str(e)[:200])
        # 任何兜底失败，绝不传播——hook 失败会污染 IDE
        pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        sys.exit(0)
