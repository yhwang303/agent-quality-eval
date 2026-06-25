#!/usr/bin/env python3
"""
Claude Code -> Langfuse 全量 Hook 脚本（增强版）

覆盖全部 27 个 hook 事件，支持分级启用。

核心功能：
- Thinking 内容完整记录（含传入上下文）
- 工具调用 PreToolUse/PostToolUse 精确配对（span create → update）
- Compact 前后记录
- 文件变更记录
- LLM Generation 含 usage + usageDetails（精确缓存成本计算）
- 全局开关 + 分级控制

配置方式：
  1. 放到 ~/.claude/hooks/langfuse_hook.py
  2. 在 ~/.claude/settings.json 注册所有 hook 事件
  3. 在项目 .claude/settings.local.json 配置 Langfuse 密钥和级别

环境变量：
  TRACE_TO_LANGFUSE      - 全局开关，"true" 启用（必须）
  CC_LANGFUSE_LEVEL      - 级别：basic / standard（默认）/ full
  LANGFUSE_PUBLIC_KEY    - Langfuse 公钥
  LANGFUSE_SECRET_KEY    - Langfuse 私钥
  LANGFUSE_BASE_URL      - Langfuse 地址（默认 https://cloud.langfuse.com）
  CC_LANGFUSE_DEBUG      - "true" 开启详细日志
  CC_LANGFUSE_MAX_CHARS  - 文本截断长度（默认 20000）
"""

import json
import os
import sys
import time
import uuid
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─── 无条件日志：确认脚本是否被调用 ─────────────────────
_BOOT_LOG = Path.home() / ".claude" / "state" / "langfuse_boot.log"
try:
    # ★ Windows stdin 编码修复：强制使用 UTF-8 读取 stdin
    # Claude Code 通过 Git Bash 管道传递的 JSON 是 UTF-8 编码的，
    # 但 Windows 的 sys.stdin 默认用系统编码（GBK/CP936），
    # 导致中文内容被错误解码为乱码。
    if hasattr(sys.stdin, 'reconfigure'):
        sys.stdin.reconfigure(encoding='utf-8', errors='replace')
    elif hasattr(sys.stdin, 'buffer'):
        import io
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')

    _BOOT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_BOOT_LOG, "a", encoding="utf-8") as _bf:
        _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _bf.write(f"{_ts} BOOT: script invoked, TRACE_TO_LANGFUSE={os.environ.get('TRACE_TO_LANGFUSE', 'NOT_SET')}, argv={sys.argv}\n")
        # 诊断：捕获 stdin 数据和关键环境变量
        try:
            import select
            _stdin_data = sys.stdin.read()
            _bf.write(f"{_ts} BOOT: stdin length={len(_stdin_data)}, first200={_stdin_data[:200]!r}\n")
            _bf.write(f"{_ts} BOOT: LANGFUSE_PUBLIC_KEY={'SET' if os.environ.get('LANGFUSE_PUBLIC_KEY') or os.environ.get('CC_LANGFUSE_PUBLIC_KEY') else 'MISSING'}\n")
            _bf.write(f"{_ts} BOOT: LANGFUSE_SECRET_KEY={'SET' if os.environ.get('LANGFUSE_SECRET_KEY') or os.environ.get('CC_LANGFUSE_SECRET_KEY') else 'MISSING'}\n")
            # 将已读取的 stdin 数据保存，供后续 read_hook_payload 使用
            _BOOT_STDIN_DATA = _stdin_data
        except Exception as _e:
            _bf.write(f"{_ts} BOOT: stdin read error: {_e}\n")
            _BOOT_STDIN_DATA = None
except Exception:
    _BOOT_STDIN_DATA = None

# ─── 快速退出检查（在 import langfuse 之前） ─────────────
if os.environ.get("TRACE_TO_LANGFUSE", "").lower() != "true":
    sys.exit(0)

# ─── 抑制 stderr 输出 ────────────────────────────────────
# Claude Code 的 hook 框架会将 stderr 中的任何输出视为 hook error，
# 在 UI 上显示 "PostToolUse:Edit hook error" 等警告。
# Langfuse SDK 和 Python warnings 可能向 stderr 写入内容，
# 因此在 import langfuse 之前将 stderr 重定向到 devnull。
_original_stderr = sys.stderr
try:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
except Exception:
    pass

try:
    from langfuse import Langfuse
except Exception:
    sys.exit(0)


# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════

STATE_DIR = Path.home() / ".claude" / "state"
LOG_FILE = STATE_DIR / "langfuse_hook.log"
STATE_FILE = STATE_DIR / "langfuse_state.json"
LOCK_FILE = STATE_DIR / "langfuse_state.lock"

DEBUG = os.environ.get("CC_LANGFUSE_DEBUG", "").lower() == "true"
MAX_CHARS = int(os.environ.get("CC_LANGFUSE_MAX_CHARS", "20000"))

# ─── 分级定义 ─────────────────────────────────────────────

ALL_EVENTS = {
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
}

LEVELS = {
    "basic": {
        "SessionStart", "Stop", "SessionEnd",
    },
    "standard": {
        "SessionStart", "Stop", "SessionEnd", "StopFailure",
        "UserPromptSubmit",
        "PreToolUse", "PostToolUse", "PostToolUseFailure",
        "PreCompact", "PostCompact",
        "FileChanged",
    },
    "full": ALL_EVENTS,
}


# ═══════════════════════════════════════════════════════════
# 日志
# ═══════════════════════════════════════════════════════════

def _log(level: str, message: str) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{ts} [{level}] {message}\n")
    except Exception:
        pass


def debug(msg: str) -> None:
    if DEBUG:
        _log("DEBUG", msg)


def info(msg: str) -> None:
    _log("INFO", msg)


def error(msg: str) -> None:
    _log("ERROR", msg)


# ═══════════════════════════════════════════════════════════
# 文件锁（并发安全）
# ═══════════════════════════════════════════════════════════

class FileLock:
    def __init__(self, path: Path, timeout_s: float = 2.0):
        self.path = path
        self.timeout_s = timeout_s
        self._fh = None

    def __enter__(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+", encoding="utf-8")
        try:
            import fcntl
            deadline = time.time() + self.timeout_s
            while True:
                try:
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() > deadline:
                        break
                    time.sleep(0.05)
        except ImportError:
            # Windows: fcntl 不可用
            try:
                import msvcrt
                deadline = time.time() + self.timeout_s
                while True:
                    try:
                        msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except (IOError, OSError):
                        if time.time() > deadline:
                            break
                        time.sleep(0.05)
            except Exception:
                pass
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            import fcntl
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
# 状态持久化
# ═══════════════════════════════════════════════════════════

def load_state() -> Dict[str, Any]:
    try:
        if not STATE_FILE.exists():
            return {}
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: Dict[str, Any]) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        debug(f"save_state failed: {e}")


def session_state_key(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]


def get_session_state(state: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    key = session_state_key(session_id)
    if key not in state:
        state[key] = {
            "trace_id": str(uuid.uuid4()),
            "turn_count": 0,
            "current_turn_span_id": None,
            "transcript_offset": 0,
            "transcript_buffer": "",
            "pending_spans": {},
            "updated": datetime.now(timezone.utc).isoformat(),
        }
    return state[key]


def update_session_state(state: Dict[str, Any], session_id: str, ss: Dict[str, Any]) -> None:
    key = session_state_key(session_id)
    ss["updated"] = datetime.now(timezone.utc).isoformat()
    state[key] = ss


# ═══════════════════════════════════════════════════════════
# Hook payload 解析
# ═══════════════════════════════════════════════════════════

def _regex_extract_fields(data: str) -> Dict[str, Any]:
    """
    当 JSON 解析完全失败时，用正则从原始字符串中提取关键字段。
    这样即使 tool_response 中有导致 JSON 解析失败的内容，
    也能至少提取出 hook_event_name / session_id / tool_use_id / tool_name
    来完成 span 的关闭。
    """
    import re
    result: Dict[str, Any] = {}

    # 提取简单的顶层字符串字段
    patterns = {
        "hook_event_name": r'"hook_event_name"\s*:\s*"([^"]*)"',
        "session_id": r'"session_id"\s*:\s*"([^"]*)"',
        "tool_use_id": r'"tool_use_id"\s*:\s*"([^"]*)"',
        "tool_name": r'"tool_name"\s*:\s*"([^"]*)"',
        "transcript_path": r'"transcript_path"\s*:\s*"([^"]*)"',
        "cwd": r'"cwd"\s*:\s*"([^"]*)"',
        "agent_id": r'"agent_id"\s*:\s*"([^"]*)"',
        "agent_type": r'"agent_type"\s*:\s*"([^"]*)"',
        "permission_mode": r'"permission_mode"\s*:\s*"([^"]*)"',
    }
    for field, pattern in patterns.items():
        m = re.search(pattern, data)
        if m:
            result[field] = m.group(1)

    if result:
        # 标记这是 regex 降级解析的结果
        result["_parsed_via"] = "regex_fallback"
        debug(f"regex_extract_fields: extracted {list(result.keys())} from {len(data)} bytes")

    return result


def read_hook_payload() -> Dict[str, Any]:
    try:
        # 优先使用 BOOT 阶段已读取的 stdin 数据（避免重复读取空 stdin）
        global _BOOT_STDIN_DATA
        if _BOOT_STDIN_DATA is not None:
            data = _BOOT_STDIN_DATA
            _BOOT_STDIN_DATA = None  # 只用一次
        else:
            data = sys.stdin.read()
        if not data.strip():
            return {}
        # 第 1 级：直接解析
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            pass
        # 第 2 级：清理 Windows 编码问题（BOM + 控制字符）
        cleaned = data.lstrip('\ufeff')
        cleaned = ''.join(
            c if c in ('\t', '\n', '\r') or ord(c) >= 0x20 else ' '
            for c in cleaned
        )
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # 第 3 级：数据截断恢复——从末尾向前找最后一个 '}'
        last_brace = cleaned.rfind('}')
        if last_brace > 0:
            try:
                return json.loads(cleaned[:last_brace + 1])
            except json.JSONDecodeError:
                pass
        # 第 4 级：正则降级提取关键字段
        # 这是最后的兜底——即使 tool_response 等大字段导致 JSON 损坏，
        # 也要提取出 hook_event_name 等关键字段，确保工具 span 能被关闭
        result = _regex_extract_fields(data)
        if result.get("hook_event_name"):
            _log("WARN", f"read_hook_payload: JSON parse failed, using regex fallback "
                 f"(data_len={len(data)}, event={result.get('hook_event_name')}, "
                 f"tool={result.get('tool_name', 'N/A')})")
            return result
        error(f"read_hook_payload failed after all attempts: data_len={len(data)}, first100={data[:100]!r}, last50={data[-50:]!r}")
        return {}
    except Exception as e:
        error(f"read_hook_payload failed: {e}")
        return {}


def get_session_id(payload: Dict[str, Any]) -> Optional[str]:
    return (
        payload.get("session_id")
        or payload.get("sessionId")
        or payload.get("session", {}).get("id")
    )


def get_transcript_path(payload: Dict[str, Any]) -> Optional[Path]:
    transcript = (
        payload.get("transcript_path")
        or payload.get("transcriptPath")
        or payload.get("transcript", {}).get("path")
    )
    if transcript:
        try:
            return Path(transcript).expanduser().resolve()
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════════════════
# 文本处理
# ═══════════════════════════════════════════════════════════

def truncate_text(s: Any, max_chars: int = MAX_CHARS) -> Tuple[str, Dict[str, Any]]:
    if s is None:
        return "", {"truncated": False, "orig_len": 0}
    if not isinstance(s, str):
        s = json.dumps(s, ensure_ascii=False, default=str)
    orig_len = len(s)
    if orig_len <= max_chars:
        return s, {"truncated": False, "orig_len": orig_len}
    return s[:max_chars], {
        "truncated": True,
        "orig_len": orig_len,
        "kept_len": max_chars,
    }


def safe_json(obj: Any) -> Any:
    """确保对象可 JSON 序列化"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [safe_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    return str(obj)


# ═══════════════════════════════════════════════════════════
# Transcript 解析
# ═══════════════════════════════════════════════════════════

def get_content(msg: Dict[str, Any]) -> Any:
    if not isinstance(msg, dict):
        return None
    if "message" in msg and isinstance(msg.get("message"), dict):
        return msg["message"].get("content")
    return msg.get("content")


def get_role(msg: Dict[str, Any]) -> Optional[str]:
    t = msg.get("type")
    if t in ("user", "assistant"):
        return t
    m = msg.get("message")
    if isinstance(m, dict):
        r = m.get("role")
        if r in ("user", "assistant"):
            return r
    return None


def is_tool_result(msg: Dict[str, Any]) -> bool:
    if get_role(msg) != "user":
        return False
    content = get_content(msg)
    if isinstance(content, list):
        return any(isinstance(x, dict) and x.get("type") == "tool_result" for x in content)
    return False


def iter_tool_results(content: Any) -> List[Dict[str, Any]]:
    out = []
    if isinstance(content, list):
        for x in content:
            if isinstance(x, dict) and x.get("type") == "tool_result":
                out.append(x)
    return out


def iter_tool_uses(content: Any) -> List[Dict[str, Any]]:
    out = []
    if isinstance(content, list):
        for x in content:
            if isinstance(x, dict) and x.get("type") == "tool_use":
                out.append(x)
    return out


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for x in content:
            if isinstance(x, dict) and x.get("type") == "text":
                parts.append(x.get("text", ""))
            elif isinstance(x, str):
                parts.append(x)
        return "\n".join([p for p in parts if p])
    return ""


def get_model(msg: Dict[str, Any]) -> str:
    m = msg.get("message")
    if isinstance(m, dict):
        return m.get("model") or "claude"
    return "claude"


def get_message_id(msg: Dict[str, Any]) -> Optional[str]:
    m = msg.get("message")
    if isinstance(m, dict):
        mid = m.get("id")
        if isinstance(mid, str) and mid:
            return mid
    return None


def get_usage(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    m = msg.get("message")
    if isinstance(m, dict):
        return m.get("usage")
    return None


def get_stop_reason(msg: Dict[str, Any]) -> Optional[str]:
    m = msg.get("message")
    if isinstance(m, dict):
        return m.get("stop_reason")
    return None


def extract_thinking(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts = []
    for x in content:
        if isinstance(x, dict) and x.get("type") == "thinking":
            thinking_text = x.get("thinking", "")
            if thinking_text:
                parts.append(thinking_text)
    return "\n---\n".join(parts)


def aggregate_usage(assistant_msgs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    聚合一个 turn 中所有 assistant 消息的 usage。
    input_tokens 取最后一条（最完整），output_tokens 累加。
    """
    total_output = 0
    total_cache_creation = 0
    total_cache_read = 0
    last_input = 0

    for am in assistant_msgs:
        u = get_usage(am)
        if not u:
            continue
        last_input = u.get("input_tokens", 0)
        total_output += u.get("output_tokens", 0)
        total_cache_creation += u.get("cache_creation_input_tokens", 0)
        total_cache_read += u.get("cache_read_input_tokens", 0)

    cache_hit_rate = 0.0
    if last_input > 0:
        cache_hit_rate = round(total_cache_read / last_input * 100, 2)

    return {
        "input_tokens": last_input,
        "output_tokens": total_output,
        "cache_creation_input_tokens": total_cache_creation,
        "cache_read_input_tokens": total_cache_read,
        "cache_hit_rate_percent": cache_hit_rate,
        "assistant_message_count": len(assistant_msgs),
    }


# ─── 增量读取 transcript ────────────────────────────────

def read_new_jsonl(
    transcript_path: Path, offset: int, buffer: str
) -> Tuple[List[Dict[str, Any]], int, str]:
    """增量读取 transcript，返回 (新消息列表, 新 offset, 剩余 buffer)"""
    if not transcript_path.exists():
        return [], offset, buffer
    try:
        with open(transcript_path, "rb") as f:
            f.seek(offset)
            chunk = f.read()
            new_offset = f.tell()
    except Exception as e:
        debug(f"read_new_jsonl failed: {e}")
        return [], offset, buffer
    if not chunk:
        return [], offset, buffer
    text = chunk.decode("utf-8", errors="replace")
    combined = buffer + text
    lines = combined.split("\n")
    remaining_buffer = lines[-1]
    msgs = []
    for line in lines[:-1]:
        line = line.strip()
        if not line:
            continue
        try:
            msgs.append(json.loads(line))
        except Exception:
            continue
    return msgs, new_offset, remaining_buffer


# ─── Turn 组装 ──────────────────────────────────────────

@dataclass
class Turn:
    user_msg: Dict[str, Any]
    assistant_msgs: List[Dict[str, Any]]
    tool_results_by_id: Dict[str, Any]


def build_turns(messages: List[Dict[str, Any]]) -> List[Turn]:
    """
    将 transcript 消息列表按 user turn 分组。
    保留所有 assistant 消息的完整序列（不做去重），以便识别中间思考轮。
    中间思考轮的特征：output_tokens=0 且无 stop_reason。
    """
    turns: List[Turn] = []
    current_user: Optional[Dict[str, Any]] = None
    all_assistants: List[Dict[str, Any]] = []
    tool_results_by_id: Dict[str, Any] = {}

    def flush_turn():
        nonlocal current_user, all_assistants, tool_results_by_id
        if current_user is None or not all_assistants:
            return
        turns.append(Turn(
            user_msg=current_user,
            assistant_msgs=list(all_assistants),
            tool_results_by_id=dict(tool_results_by_id),
        ))

    for msg in messages:
        role = get_role(msg)

        if is_tool_result(msg):
            for tr in iter_tool_results(get_content(msg)):
                tid = tr.get("tool_use_id")
                if tid:
                    tool_results_by_id[str(tid)] = tr.get("content")
            continue

        if role == "user":
            flush_turn()
            current_user = msg
            all_assistants = []
            tool_results_by_id = {}
            continue

        if role == "assistant":
            if current_user is None:
                continue
            all_assistants.append(msg)
            continue

    flush_turn()
    return turns


# ═══════════════════════════════════════════════════════════
# Langfuse 客户端初始化
# ═══════════════════════════════════════════════════════════

def create_langfuse() -> Optional[Langfuse]:
    public_key = os.environ.get("CC_LANGFUSE_PUBLIC_KEY") or os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("CC_LANGFUSE_SECRET_KEY") or os.environ.get("LANGFUSE_SECRET_KEY")
    host = (
        os.environ.get("CC_LANGFUSE_BASE_URL")
        or os.environ.get("LANGFUSE_BASE_URL")
        or "https://cloud.langfuse.com"
    )
    if not public_key or not secret_key:
        debug("Missing Langfuse keys")
        return None
    try:
        return Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    except Exception as e:
        debug(f"Failed to create Langfuse client: {e}")
        return None


# ═══════════════════════════════════════════════════════════
# Event Handlers
# ═══════════════════════════════════════════════════════════

def handle_session_start(
    langfuse: Langfuse, payload: Dict[str, Any],
    session_id: str, state: Dict[str, Any], ss: Dict[str, Any],
) -> None:
    """SessionStart: 创建整个 session 的 trace"""
    trace_id = ss["trace_id"]
    langfuse.trace(
        id=trace_id,
        session_id=session_id,
        name="Claude Code Session",
        input={"event": "SessionStart", "cwd": payload.get("cwd", "")},
        tags=["claude-code"],
        metadata={
            "source": "claude-code",
            "permission_mode": payload.get("permission_mode"),
            "cwd": payload.get("cwd"),
        },
    )
    info(f"SessionStart: trace_id={trace_id}, session={session_id}")


def handle_session_end(
    langfuse: Langfuse, payload: Dict[str, Any],
    session_id: str, state: Dict[str, Any], ss: Dict[str, Any],
) -> None:
    """SessionEnd: 更新 trace 的 output"""
    trace_id = ss["trace_id"]
    langfuse.trace(
        id=trace_id,
        output={"event": "SessionEnd", "total_turns": ss.get("turn_count", 0)},
    )
    info(f"SessionEnd: trace_id={trace_id}, turns={ss.get('turn_count', 0)}")


def handle_user_prompt(
    langfuse: Langfuse, payload: Dict[str, Any],
    session_id: str, state: Dict[str, Any], ss: Dict[str, Any],
) -> None:
    """UserPromptSubmit: 创建 Turn span"""
    ss["turn_count"] = ss.get("turn_count", 0) + 1
    turn_num = ss["turn_count"]
    turn_span_id = f"turn-{session_id}-{turn_num}"
    ss["current_turn_span_id"] = turn_span_id

    langfuse.span(
        id=turn_span_id,
        trace_id=ss["trace_id"],
        name=f"Turn {turn_num}",
        start_time=datetime.now(timezone.utc),
        input=safe_json(payload.get("tool_input", payload.get("content", ""))),
        metadata={
            "turn_number": turn_num,
            "event": "UserPromptSubmit",
        },
    )
    debug(f"UserPromptSubmit: turn={turn_num}, span_id={turn_span_id}")


def handle_stop(
    langfuse: Langfuse, payload: Dict[str, Any],
    session_id: str, state: Dict[str, Any], ss: Dict[str, Any],
) -> None:
    """
    Stop: 回合结束。
    1. 更新 Turn span 的 endTime
    2. 增量读取 transcript，提取 LLM generation（含 usage、thinking）
    """
    trace_id = ss["trace_id"]
    turn_span_id = ss.get("current_turn_span_id")

    # 更新 Turn span 的 endTime（必须带 trace_id，否则新 Langfuse 客户端会创建孤儿 trace）
    if turn_span_id:
        langfuse.span(
            id=turn_span_id,
            trace_id=trace_id,
            end_time=datetime.now(timezone.utc),
            output={"event": "Stop"},
        )

    # 增量读取 transcript
    transcript_path = get_transcript_path(payload)
    if not transcript_path:
        debug("Stop: no transcript_path")
        return

    offset = ss.get("transcript_offset", 0)
    buffer = ss.get("transcript_buffer", "")
    msgs, new_offset, new_buffer = read_new_jsonl(transcript_path, offset, buffer)
    ss["transcript_offset"] = new_offset
    ss["transcript_buffer"] = new_buffer

    if not msgs:
        debug("Stop: no new messages in transcript")
        return

    turns = build_turns(msgs)
    if not turns:
        debug("Stop: no turns built")
        return

    for turn in turns:
        _emit_turn_generation(langfuse, trace_id, turn_span_id, turn, transcript_path)

    # ★ 更新 Trace 的 Output（取最后一个 turn 的 assistant 回复），解决 Output=undefined 问题
    if turns:
        last_turn = turns[-1]
        if last_turn.assistant_msgs:
            last_assistant = last_turn.assistant_msgs[-1]
            assistant_output = extract_text(get_content(last_assistant))
            if assistant_output:
                output_text, _ = truncate_text(assistant_output, MAX_CHARS)
                usage_agg = aggregate_usage(last_turn.assistant_msgs)
                langfuse.trace(
                    id=trace_id,
                    output=output_text,
                    metadata={
                        "source": "claude-code",
                        "total_turns": ss.get("turn_count", 0),
                        "input_tokens": usage_agg["input_tokens"],
                        "output_tokens": usage_agg["output_tokens"],
                        "cache_read_input_tokens": usage_agg["cache_read_input_tokens"],
                    },
                )

    info(f"Stop: processed {len(turns)} turns from transcript")


def _is_thinking_msg(msg: Dict[str, Any]) -> bool:
    """
    判断一条 assistant 消息是否是中间思考轮。
    特征：output_tokens=0 且无 stop_reason。
    Claude Code 在流式传输中，中间思考会产生 text block，
    但 output_tokens 记为 0（token 计数归到最终消息上）。
    """
    usage = get_usage(msg)
    if usage and usage.get("output_tokens", 0) > 0:
        return False
    stop = get_stop_reason(msg)
    if stop:
        return False
    # 必须有 text 内容
    content = get_content(msg)
    text = extract_text(content)
    return len(text.strip()) > 0


def _emit_turn_generation(
    langfuse: Langfuse,
    trace_id: str,
    parent_span_id: Optional[str],
    turn: Turn,
    transcript_path: Path,
) -> None:
    """
    从 transcript turn 数据创建：
    1. Agent Thinking span（每个中间思考轮一个，output_tokens=0 且无 stop_reason）
    2. LLM Response generation（最终回复，含聚合 usage）
    """
    # 提取用户输入
    user_text_raw = extract_text(get_content(turn.user_msg))
    user_text, user_text_meta = truncate_text(user_text_raw)

    # ★ 识别中间思考消息和最终回复消息
    thinking_msgs = []
    final_msgs = []  # 有实际 output_tokens 或有 stop_reason 的消息
    for am in turn.assistant_msgs:
        if _is_thinking_msg(am):
            thinking_msgs.append(am)
        else:
            final_msgs.append(am)

    # ★ 为每个中间思考创建 Agent Thinking span
    for idx, tmsg in enumerate(thinking_msgs):
        thinking_text_raw = extract_text(get_content(tmsg))
        thinking_text, thinking_meta = truncate_text(thinking_text_raw, MAX_CHARS)

        thinking_usage = get_usage(tmsg) or {}
        thinking_input_tokens = thinking_usage.get("input_tokens", 0)

        thinking_kwargs = {
            "trace_id": trace_id,
            "name": f"Agent Thinking",
            "input": {
                "type": "intermediate_thinking",
                "thinking_index": idx + 1,
                "total_thinking_rounds": len(thinking_msgs),
            },
            "output": thinking_text,
            "metadata": {
                "thinking_index": idx + 1,
                "total_thinking_rounds": len(thinking_msgs),
                "text_length": len(thinking_text_raw),
                "input_tokens": thinking_input_tokens,
                "truncation": thinking_meta,
            },
            "start_time": datetime.now(timezone.utc),
            "end_time": datetime.now(timezone.utc),
        }
        if parent_span_id:
            thinking_kwargs["parent_observation_id"] = parent_span_id
        langfuse.span(**thinking_kwargs)

    # ★ 提取最终回复（取最后一条有 output 的消息）
    if not final_msgs:
        # 极端情况：全是中间思考没有最终回复
        debug("No final assistant message in turn, skipping LLM Response generation")
        return

    last_assistant = final_msgs[-1]
    assistant_text_raw = extract_text(get_content(last_assistant))
    assistant_text, assistant_text_meta = truncate_text(assistant_text_raw)

    # 模型名
    model = get_model(turn.assistant_msgs[0])

    # ★ 聚合 usage（只聚合有实际 output 的消息，避免重复计算）
    usage_agg = aggregate_usage(final_msgs)

    # ★ Langfuse usage 参数
    usage_param = None
    if usage_agg["input_tokens"] > 0 or usage_agg["output_tokens"] > 0:
        usage_param = {
            "input": usage_agg["input_tokens"],
            "output": usage_agg["output_tokens"],
            "unit": "TOKENS",
        }

    # ★ usageDetails（用于精确缓存成本计算）
    usage_details = {}
    if usage_agg["input_tokens"] > 0:
        usage_details["input_tokens"] = usage_agg["input_tokens"]
    if usage_agg["output_tokens"] > 0:
        usage_details["output_tokens"] = usage_agg["output_tokens"]
    if usage_agg["cache_creation_input_tokens"] > 0:
        usage_details["cache_creation_input_tokens"] = usage_agg["cache_creation_input_tokens"]
    if usage_agg["cache_read_input_tokens"] > 0:
        usage_details["cache_read_input_tokens"] = usage_agg["cache_read_input_tokens"]

    # ★ stop_reason
    stop_reason = get_stop_reason(last_assistant)

    # 创建 LLM Generation
    gen_kwargs = {
        "trace_id": trace_id,
        "name": "LLM Response",
        "model": model,
        "input": {"role": "user", "content": user_text},
        "output": {"role": "assistant", "content": assistant_text},
        "metadata": {
            "assistant_text_meta": assistant_text_meta,
            "user_text_meta": user_text_meta,
            "stop_reason": stop_reason,
            "input_tokens": usage_agg["input_tokens"],
            "output_tokens": usage_agg["output_tokens"],
            "cache_creation_input_tokens": usage_agg["cache_creation_input_tokens"],
            "cache_read_input_tokens": usage_agg["cache_read_input_tokens"],
            "cache_hit_rate_percent": usage_agg["cache_hit_rate_percent"],
            "assistant_message_count": usage_agg["assistant_message_count"],
            "thinking_rounds": len(thinking_msgs),
            "total_assistant_messages": len(turn.assistant_msgs),
        },
        "start_time": datetime.now(timezone.utc),
    }
    if parent_span_id:
        gen_kwargs["parent_observation_id"] = parent_span_id
    if usage_param:
        gen_kwargs["usage"] = usage_param
    if usage_details:
        gen_kwargs["usage_details"] = usage_details

    langfuse.generation(**gen_kwargs)


def handle_stop_failure(
    langfuse: Langfuse, payload: Dict[str, Any],
    session_id: str, state: Dict[str, Any], ss: Dict[str, Any],
) -> None:
    """StopFailure: 标记 Turn span 为 ERROR"""
    trace_id = ss["trace_id"]
    turn_span_id = ss.get("current_turn_span_id")
    if turn_span_id:
        langfuse.span(
            id=turn_span_id,
            trace_id=trace_id,
            end_time=datetime.now(timezone.utc),
            output={"event": "StopFailure"},
            level="ERROR",
            status_message="Turn ended with failure",
        )
    debug("StopFailure handled")


def handle_pre_tool_use(
    langfuse: Langfuse, payload: Dict[str, Any],
    session_id: str, state: Dict[str, Any], ss: Dict[str, Any],
) -> None:
    """PreToolUse: 创建工具调用 span（startTime，无 endTime）"""
    tool_use_id = payload.get("tool_use_id", "")
    tool_name = payload.get("tool_name", "unknown")
    tool_input = payload.get("tool_input", {})

    if not tool_use_id:
        debug("PreToolUse: no tool_use_id")
        return

    span_id = f"tool-{tool_use_id}"
    trace_id = ss["trace_id"]
    turn_span_id = ss.get("current_turn_span_id")

    # 截断工具输入
    input_display = safe_json(tool_input)
    if isinstance(input_display, str) and len(input_display) > MAX_CHARS:
        input_display, _ = truncate_text(input_display)

    span_kwargs = {
        "id": span_id,
        "trace_id": trace_id,
        "name": f"Tool: {tool_name}",
        "start_time": datetime.now(timezone.utc),
        "input": input_display,
        "metadata": {
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "agent_id": payload.get("agent_id"),
            "agent_type": payload.get("agent_type"),
        },
    }
    if turn_span_id:
        span_kwargs["parent_observation_id"] = turn_span_id

    langfuse.span(**span_kwargs)
    debug(f"PreToolUse: tool={tool_name}, span_id={span_id}")


def handle_post_tool_use(
    langfuse: Langfuse, payload: Dict[str, Any],
    session_id: str, state: Dict[str, Any], ss: Dict[str, Any],
) -> None:
    """PostToolUse: 更新工具 span（endTime + output）"""
    tool_use_id = payload.get("tool_use_id", "")
    if not tool_use_id:
        debug("PostToolUse: no tool_use_id")
        return

    span_id = f"tool-{tool_use_id}"
    tool_response = payload.get("tool_response", "")

    # 截断工具输出
    if isinstance(tool_response, str):
        output_display, output_meta = truncate_text(tool_response)
    else:
        output_str = json.dumps(tool_response, ensure_ascii=False, default=str)
        output_display, output_meta = truncate_text(output_str)

    langfuse.span(
        id=span_id,
        trace_id=ss["trace_id"],
        end_time=datetime.now(timezone.utc),
        output=output_display,
        metadata={
            "output_meta": output_meta,
            "tool_name": payload.get("tool_name", "unknown"),
        },
    )
    debug(f"PostToolUse: tool={payload.get('tool_name')}, span_id={span_id}")


def handle_post_tool_failure(
    langfuse: Langfuse, payload: Dict[str, Any],
    session_id: str, state: Dict[str, Any], ss: Dict[str, Any],
) -> None:
    """PostToolUseFailure: 更新工具 span 为 ERROR"""
    tool_use_id = payload.get("tool_use_id", "")
    if not tool_use_id:
        return

    span_id = f"tool-{tool_use_id}"
    langfuse.span(
        id=span_id,
        trace_id=ss["trace_id"],
        end_time=datetime.now(timezone.utc),
        output=safe_json(payload.get("tool_response", "Tool execution failed")),
        level="ERROR",
        status_message=str(payload.get("tool_response", "Tool execution failed"))[:500],
        metadata={
            "tool_name": payload.get("tool_name", "unknown"),
            "error": True,
        },
    )
    debug(f"PostToolUseFailure: tool={payload.get('tool_name')}")


def handle_subagent_start(
    langfuse: Langfuse, payload: Dict[str, Any],
    session_id: str, state: Dict[str, Any], ss: Dict[str, Any],
) -> None:
    """SubagentStart: 创建子代理 span"""
    agent_id = payload.get("agent_id", str(uuid.uuid4())[:8])
    span_id = f"subagent-{agent_id}"

    span_kwargs = {
        "id": span_id,
        "trace_id": ss["trace_id"],
        "name": f"Subagent: {payload.get('agent_type', 'unknown')}",
        "start_time": datetime.now(timezone.utc),
        "input": safe_json({
            "agent_id": agent_id,
            "agent_type": payload.get("agent_type"),
            "cwd": payload.get("cwd"),
        }),
        "metadata": {
            "event": "SubagentStart",
            "agent_id": agent_id,
            "agent_type": payload.get("agent_type"),
        },
    }
    turn_span_id = ss.get("current_turn_span_id")
    if turn_span_id:
        span_kwargs["parent_observation_id"] = turn_span_id

    langfuse.span(**span_kwargs)

    # 记录 pending span
    pending = ss.get("pending_spans", {})
    pending[f"subagent-{agent_id}"] = span_id
    ss["pending_spans"] = pending

    debug(f"SubagentStart: agent_id={agent_id}, span_id={span_id}")


def handle_subagent_stop(
    langfuse: Langfuse, payload: Dict[str, Any],
    session_id: str, state: Dict[str, Any], ss: Dict[str, Any],
) -> None:
    """SubagentStop: 结束子代理 span"""
    agent_id = payload.get("agent_id", "")
    pending = ss.get("pending_spans", {})
    span_id = pending.pop(f"subagent-{agent_id}", None)
    ss["pending_spans"] = pending

    if span_id:
        langfuse.span(
            id=span_id,
            trace_id=ss["trace_id"],
            end_time=datetime.now(timezone.utc),
            output={"event": "SubagentStop"},
        )
        debug(f"SubagentStop: agent_id={agent_id}")
    else:
        debug(f"SubagentStop: no pending span for agent_id={agent_id}")


def handle_pre_compact(
    langfuse: Langfuse, payload: Dict[str, Any],
    session_id: str, state: Dict[str, Any], ss: Dict[str, Any],
) -> None:
    """PreCompact: 创建压缩 span"""
    compact_id = f"compact-{session_id[:8]}-{int(time.time())}"
    langfuse.span(
        id=compact_id,
        trace_id=ss["trace_id"],
        name="Context Compact",
        start_time=datetime.now(timezone.utc),
        input=safe_json({
            "event": "PreCompact",
            "cwd": payload.get("cwd"),
        }),
        metadata={"event": "PreCompact"},
    )

    pending = ss.get("pending_spans", {})
    pending["compact"] = compact_id
    ss["pending_spans"] = pending

    debug(f"PreCompact: compact_id={compact_id}")


def handle_post_compact(
    langfuse: Langfuse, payload: Dict[str, Any],
    session_id: str, state: Dict[str, Any], ss: Dict[str, Any],
) -> None:
    """PostCompact: 结束压缩 span"""
    pending = ss.get("pending_spans", {})
    compact_id = pending.pop("compact", None)
    ss["pending_spans"] = pending

    if compact_id:
        langfuse.span(
            id=compact_id,
            trace_id=ss["trace_id"],
            end_time=datetime.now(timezone.utc),
            output={"event": "PostCompact"},
        )
        debug(f"PostCompact: compact_id={compact_id}")
    else:
        debug("PostCompact: no pending compact span")


def handle_elicitation_start(
    langfuse: Langfuse, payload: Dict[str, Any],
    session_id: str, state: Dict[str, Any], ss: Dict[str, Any],
) -> None:
    """Elicitation: 创建用户输入请求 span"""
    elicit_id = f"elicit-{session_id[:8]}-{int(time.time())}"
    langfuse.span(
        id=elicit_id,
        trace_id=ss["trace_id"],
        name="User Elicitation",
        start_time=datetime.now(timezone.utc),
        input=safe_json(payload),
        metadata={"event": "Elicitation"},
    )
    pending = ss.get("pending_spans", {})
    pending["elicitation"] = elicit_id
    ss["pending_spans"] = pending
    debug(f"Elicitation: elicit_id={elicit_id}")


def handle_elicitation_result(
    langfuse: Langfuse, payload: Dict[str, Any],
    session_id: str, state: Dict[str, Any], ss: Dict[str, Any],
) -> None:
    """ElicitationResult: 结束用户输入 span"""
    pending = ss.get("pending_spans", {})
    elicit_id = pending.pop("elicitation", None)
    ss["pending_spans"] = pending

    if elicit_id:
        langfuse.span(
            id=elicit_id,
            trace_id=ss["trace_id"],
            end_time=datetime.now(timezone.utc),
            output=safe_json(payload),
        )
        debug(f"ElicitationResult: elicit_id={elicit_id}")


def handle_generic_event(
    langfuse: Langfuse, payload: Dict[str, Any],
    session_id: str, state: Dict[str, Any], ss: Dict[str, Any],
) -> None:
    """通用事件处理：记录为 Langfuse event"""
    event_name = payload.get("hook_event_name", "Unknown")
    trace_id = ss["trace_id"]

    # 构造 event 的 input（去掉过大的字段）
    event_data = {}
    for k, v in payload.items():
        if k in ("hook_event_name", "session_id", "transcript_path"):
            continue
        if isinstance(v, str) and len(v) > 1000:
            event_data[k] = v[:1000] + "...(truncated)"
        else:
            event_data[k] = safe_json(v)

    langfuse.event(
        trace_id=trace_id,
        name=event_name,
        input=event_data,
        metadata={
            "event": event_name,
            "cwd": payload.get("cwd"),
        },
    )
    debug(f"Event: {event_name}")


# ═══════════════════════════════════════════════════════════
# Handler 路由表
# ═══════════════════════════════════════════════════════════

HANDLERS = {
    "SessionStart": handle_session_start,
    "SessionEnd": handle_session_end,
    "UserPromptSubmit": handle_user_prompt,
    "Stop": handle_stop,
    "StopFailure": handle_stop_failure,
    "PreToolUse": handle_pre_tool_use,
    "PostToolUse": handle_post_tool_use,
    "PostToolUseFailure": handle_post_tool_failure,
    "SubagentStart": handle_subagent_start,
    "SubagentStop": handle_subagent_stop,
    "PreCompact": handle_pre_compact,
    "PostCompact": handle_post_compact,
    "Elicitation": handle_elicitation_start,
    "ElicitationResult": handle_elicitation_result,
    # 以下全部走通用 event handler
    "Setup": handle_generic_event,
    "TaskCreated": handle_generic_event,
    "TaskCompleted": handle_generic_event,
    "TeammateIdle": handle_generic_event,
    "PermissionRequest": handle_generic_event,
    "PermissionDenied": handle_generic_event,
    "Notification": handle_generic_event,
    "ConfigChange": handle_generic_event,
    "InstructionsLoaded": handle_generic_event,
    "CwdChanged": handle_generic_event,
    "FileChanged": handle_generic_event,
    "WorktreeCreate": handle_generic_event,
    "WorktreeRemove": handle_generic_event,
}


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main() -> int:
    start = time.time()

    # 读取 payload
    payload = read_hook_payload()
    event = payload.get("hook_event_name", "")

    if not event:
        _log("WARN", f"No hook_event_name in payload, keys={list(payload.keys())}, len={len(payload)}")
        return 0

    # 检查级别
    level = os.environ.get("CC_LANGFUSE_LEVEL", "standard").lower()
    allowed_events = LEVELS.get(level, LEVELS["standard"])
    if event not in allowed_events:
        debug(f"Event {event} not in level {level}, skipping")
        return 0

    # 检查 handler
    handler = HANDLERS.get(event)
    if not handler:
        _log("WARN", f"No handler for event {event}")
        return 0

    # 获取 session_id
    session_id = get_session_id(payload)
    if not session_id:
        _log("WARN", f"No session_id for event {event}, payload_keys={list(payload.keys())}")
        return 0

    # 创建 Langfuse 客户端
    langfuse = create_langfuse()
    if not langfuse:
        _log("WARN", f"Failed to create Langfuse client for event {event}")
        return 0

    try:
        with FileLock(LOCK_FILE):
            state = load_state()
            ss = get_session_state(state, session_id)

            # 执行 handler
            handler(langfuse, payload, session_id, state, ss)

            # 保存状态
            update_session_state(state, session_id, ss)
            save_state(state)

        # Flush
        try:
            langfuse.flush()
        except Exception:
            pass

        dur = time.time() - start
        debug(f"Handled {event} in {dur:.3f}s (session={session_id[:16]}...)")
        return 0

    except Exception as e:
        error(f"Handler failed for {event}: {e}")
        return 0

    finally:
        try:
            langfuse.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
