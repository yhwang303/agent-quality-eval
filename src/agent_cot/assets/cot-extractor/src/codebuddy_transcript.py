"""CodeBuddy native transcript parser.

CodeBuddy IDE (Tencent CodeBuddy 4.x+) stores its conversation history in a
**double-layered** JSON layout under the user's AppData / .codebuddy
extension folder. This module is the single point that knows how to read
that layout and lift it into the same `SessionCoT` shape that
`cot_extractor.py` already produces for Cursor / Claude Code.

Empirical layout (Windows, CodeBuddyIDE 4.9.8 with hy3-preview-agent-ioa
混元 model — verified 2026-05-09)::

    %LOCALAPPDATA%\\CodeBuddyExtension\\Data
        <machine_uuid>\\
            CodeBuddyIDE\\<machine_uuid>\\
                history\\<workspace_hash>\\<session_id>\\
                    index.json                    # message id list + token usage
                    messages\\<msg_id>.json       # one file per message

Each ``messages/<id>.json`` looks like::

    {
      "role": "user" | "assistant" | "tool",
      "id":   "<msg_id>",
      "message": "<JSON-encoded string>",   # ← ⚠ stringified, must json.loads twice
      "extra":   "<JSON-encoded string>"    # ← ⚠ same, contains modelId/traceId
    }

Once you json.loads the inner ``message`` you get the real schema::

    user      → {"role":"user","content":[{"type":"text","text":"..."}]}
    assistant → {"role":"assistant","content":[
                   {"type":"reasoning",  "text":"<full CoT thinking>"},
                   {"type":"text",       "text":"<human-facing reply>"},
                   {"type":"tool-call",  "toolCallId":"...","toolName":"...","args":{...}},
                   ...
                 ],
                 "providerOptions":{"openaiCompatible":{"reasoning":"..."}}}
    tool      → {"role":"tool","content":[
                   {"type":"tool-result","toolCallId":"...","toolName":"...",
                    "result":{"status":"success","success":true,"result":{...}},
                    "isError":false},
                   ...    # one block per parallel tool-call from prev assistant
                 ]}

Why this matters
----------------
Unlike Claude Code, CodeBuddy keeps the **raw `reasoning` field on disk
unredacted** — even for the Hunyuan model. So we can recover the same
"agent thinking" depth that Cursor's `afterAgentThought` hook gives us,
just by parsing this transcript. The hooks (cot-stream-codebuddy.js) only
see PreToolUse / PostToolUse / Stop payloads which do NOT carry
`reasoning`; transcript parsing is the only path to the thinking content.

Discovery
---------
Two ways to find a transcript:

1. **From hook events** — every event written by ``cot-stream-codebuddy.js``
   carries ``payload.transcript_path`` pointing at the exact ``index.json``.
   Use :func:`transcript_path_from_events` for the most reliable lookup.

2. **From session_id alone** — for batch/CLI use without events. We walk
   ``%LOCALAPPDATA%\\CodeBuddyExtension\\Data\\*\\CodeBuddyIDE\\*\\history\\
   *\\<sid>\\index.json``. See :func:`find_transcript_by_session_id`.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Windows long-path support
# ─────────────────────────────────────────────────────────────────────────────
#
# CodeBuddy buries each session under a path like
#   %LOCALAPPDATA%\CodeBuddyExtension\Data\<machine-uuid>\CodeBuddyIDE\
#   <machine-uuid>\history\<workspace-hash>\<session-id>\messages\<msg-id>.json
# which routinely tips past Windows' 260-char MAX_PATH ceiling.
# Without the \\?\ extended-length prefix, plain ``Path.is_file()`` /
# ``open()`` silently report ENOENT for files that ``os.listdir`` happily
# enumerates, which makes the parser look broken when the data is fine.


def _long_path(p: Path) -> str:
    """Return a path string safe for filesystem ops on Windows long paths.

    The ``\\\\?\\`` prefix opts the call out of MAX_PATH normalization.
    No-ops on non-Windows platforms.
    """
    if sys.platform != "win32":
        return str(p)
    s = str(p)
    if s.startswith("\\\\?\\") or s.startswith("\\\\.\\"):
        return s
    if len(s) < 240 and not s.startswith("\\\\"):
        return s
    if s.startswith("\\\\"):
        # UNC path → \\?\UNC\server\share\...
        return "\\\\?\\UNC\\" + s.lstrip("\\")
    return "\\\\?\\" + s


def _exists_long(p: Path) -> bool:
    """``p.exists()`` that survives Windows long paths."""
    try:
        return os.path.exists(_long_path(p))
    except OSError:
        return False


def _read_text_long(p: Path, encoding: str = "utf-8-sig") -> str:
    """``p.read_text()`` that survives Windows long paths."""
    with open(_long_path(p), "r", encoding=encoding, errors="replace") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────────────────────
# Filesystem discovery
# ─────────────────────────────────────────────────────────────────────────────


def _codebuddy_data_roots() -> List[Path]:
    """All plausible roots that CodeBuddyExtension may store history under.

    Windows is the documented platform; macOS / Linux paths are best-effort
    based on Electron conventions. Adjust as we encounter more setups.
    """
    roots: List[Path] = []
    # Windows: %LOCALAPPDATA% / %APPDATA%
    for env_key in ("LOCALAPPDATA", "APPDATA"):
        base = os.environ.get(env_key)
        if base:
            p = Path(base) / "CodeBuddyExtension" / "Data"
            if p.is_dir():
                roots.append(p)
    # macOS
    home = Path.home()
    for cand in (
        home / "Library" / "Application Support" / "CodeBuddyExtension" / "Data",
        home / ".config" / "CodeBuddyExtension" / "Data",
        home / ".codebuddy" / "Data",
    ):
        if cand.is_dir():
            roots.append(cand)
    # de-dupe while preserving order
    seen: set = set()
    unique: List[Path] = []
    for r in roots:
        rs = str(r)
        if rs not in seen:
            seen.add(rs)
            unique.append(r)
    return unique


def find_transcript_by_session_id(session_id: str) -> Optional[Path]:
    """Walk every CodeBuddy data root looking for ``<sid>/index.json``.

    Returns the first match or ``None``. Cheap enough for interactive CLI
    use (a typical machine has O(10) workspaces × O(100) sessions).
    """
    for root in _codebuddy_data_roots():
        try:
            for index_path in root.rglob(f"{session_id}/index.json"):
                if _exists_long(index_path):
                    return index_path
        except OSError:
            continue
    return None


def transcript_path_from_events(events: List[Dict[str, Any]]) -> Optional[Path]:
    """Pull ``payload.transcript_path`` out of the first event that has it.

    ``cot-stream-codebuddy.js`` records the transcript_path on every event
    it forwards, so even the first SessionStart event is enough.
    """
    for ev in events or ():
        if not isinstance(ev, dict):
            continue
        payload = ev.get("payload")
        if not isinstance(payload, dict):
            continue
        tp = payload.get("transcript_path")
        if isinstance(tp, str) and tp.strip():
            p = Path(tp)
            # The path on Windows uses backslashes; Path() handles both fine.
            if _exists_long(p):
                return p
    return None


def iter_recent_transcripts(limit: int = 20) -> Iterator[Tuple[str, Path, float]]:
    """Yield ``(session_id, index_path, mtime)`` for the most recently
    modified CodeBuddy sessions across all data roots."""
    found: List[Tuple[str, Path, float]] = []
    for root in _codebuddy_data_roots():
        try:
            for index_path in root.rglob("history/*/*/index.json"):
                # parent dir name == session_id
                sid = index_path.parent.name
                try:
                    mtime = index_path.stat().st_mtime
                except OSError:
                    continue
                found.append((sid, index_path, mtime))
        except OSError:
            continue
    found.sort(key=lambda t: t[2], reverse=True)
    yield from found[:limit]


# ─────────────────────────────────────────────────────────────────────────────
# Parsing primitives
# ─────────────────────────────────────────────────────────────────────────────


def _safe_json_loads(blob: Any) -> Any:
    """Decode CodeBuddy's stringified JSON fields safely.

    They sometimes arrive as already-decoded dicts (newer versions may stop
    stringifying). Returning ``{}`` on any failure keeps the parser
    robust against schema drift — the worst case is a partially-empty step
    rather than a hard crash.
    """
    if blob is None:
        return {}
    if isinstance(blob, (dict, list)):
        return blob
    if isinstance(blob, (bytes, bytearray)):
        try:
            blob = blob.decode("utf-8", errors="replace")
        except Exception:
            return {}
    if not isinstance(blob, str):
        return {}
    try:
        return json.loads(blob)
    except Exception:
        return {}


@dataclass(frozen=True)
class CodeBuddyMessage:
    """A normalized view of one ``messages/<id>.json`` entry."""

    msg_id: str
    role: str                       # "user" | "assistant" | "tool"
    content_blocks: List[Dict[str, Any]]
    request_id: Optional[str] = None
    model_id: Optional[str] = None
    model_name: Optional[str] = None
    trace_id: Optional[str] = None
    response_id: Optional[str] = None
    # CodeBuddy 不在 message 文件里直接写 timestamp。但 assistant 的 responseId
    # 形如 ``gen-<unix_seconds>-<rand>`` —— OpenRouter / OpenAI Compatible API
    # 在生成时把 unix epoch (秒) 嵌进去了。我们就用它当这条 message 的真实生成
    # 时间（ms 精度只能到秒，但比靠 mtime 或 startedAt 推算可信得多）。
    # user / tool message 此字段为 None；调用方需要自己用相邻 assistant 的
    # generated_at_ms 插值。
    generated_at_ms: Optional[int] = None


_RESPONSE_ID_TS_RE = re.compile(r"^gen-(\d{10})(?:-|$)")


def _generated_at_ms_from_response_id(response_id: Optional[str]) -> Optional[int]:
    """Pull the unix timestamp out of an OpenRouter-style ``gen-<sec>-<rand>``.

    Returns ``None`` for any other shape (no fabrication — we'd rather not
    have a timestamp than make one up).
    """
    if not response_id:
        return None
    m = _RESPONSE_ID_TS_RE.match(response_id)
    if not m:
        return None
    try:
        return int(m.group(1)) * 1000
    except (TypeError, ValueError):
        return None


def _normalize_message(raw: Dict[str, Any]) -> Optional[CodeBuddyMessage]:
    """Convert a raw ``<msg_id>.json`` blob into a :class:`CodeBuddyMessage`.

    Returns ``None`` if the file lacks the minimum fields we need.
    """
    if not isinstance(raw, dict):
        return None
    msg_id = str(raw.get("id") or "").strip()
    if not msg_id:
        return None
    role = str(raw.get("role") or "").strip().lower() or "unknown"

    inner = _safe_json_loads(raw.get("message"))
    extra = _safe_json_loads(raw.get("extra"))

    content = []
    if isinstance(inner, dict):
        c = inner.get("content")
        if isinstance(c, list):
            content = [b for b in c if isinstance(b, dict)]
        elif isinstance(c, str):
            # User messages with plain string content (older format).
            content = [{"type": "text", "text": c}]

    response_id = str(extra.get("responseId")) if extra.get("responseId") else None
    return CodeBuddyMessage(
        msg_id=msg_id,
        role=role,
        content_blocks=content,
        request_id=str(extra.get("requestId")) if extra.get("requestId") else None,
        model_id=str(extra.get("modelId")) if extra.get("modelId") else None,
        model_name=str(extra.get("modelName")) if extra.get("modelName") else None,
        trace_id=str(extra.get("traceId")) if extra.get("traceId") else None,
        response_id=response_id,
        generated_at_ms=_generated_at_ms_from_response_id(response_id),
    )


def load_transcript(index_path: Path) -> Tuple[Dict[str, Any], List[CodeBuddyMessage]]:
    """Read ``index.json`` and resolve every referenced message file.

    Returns ``(index_blob, messages_in_order)``. Missing message files are
    skipped silently — CodeBuddy occasionally references messages that
    haven't been flushed yet during a live session, and skipping is
    preferable to crashing the whole extraction.
    """
    try:
        index_blob = json.loads(_read_text_long(index_path))
    except Exception:
        return {}, []

    messages_dir = index_path.parent / "messages"
    msg_id_order = index_blob.get("messages") if isinstance(index_blob, dict) else None
    if not isinstance(msg_id_order, list):
        msg_id_order = []

    out: List[CodeBuddyMessage] = []
    for entry in msg_id_order:
        if isinstance(entry, dict):
            msg_id = str(entry.get("id") or "")
        else:
            msg_id = str(entry or "")
        msg_id = msg_id.strip()
        if not msg_id:
            continue
        f = messages_dir / f"{msg_id}.json"
        if not _exists_long(f):
            continue
        try:
            raw = json.loads(_read_text_long(f))
        except Exception:
            continue
        norm = _normalize_message(raw)
        if norm is not None:
            out.append(norm)
    return index_blob if isinstance(index_blob, dict) else {}, out


# ─────────────────────────────────────────────────────────────────────────────
# Content-block helpers (the second layer of decoded JSON)
# ─────────────────────────────────────────────────────────────────────────────


def extract_user_text(msg: CodeBuddyMessage) -> str:
    """Concatenate every text block in a user message. The user's actual
    prompt typically lives inside an ``<user_query>...</user_query>`` tag
    nested in the larger context blob CodeBuddy injects — surface that
    when present, fall back to the whole text otherwise.

    v0.20.1 fix — *take the **last** ``<user_query>`` tag, not the first*.
    From turn 2 onward, CodeBuddy prepends the prior turn's full prompt
    block (including its own ``<user_query>...</user_query>``) into the
    new user message as conversation context. The naïve ``re.search``
    used to return that **stale** prompt as the current turn's
    ``user_query`` — that's why the dashboard's right-hand "内容" panel
    on turn-2 was showing turn-1's text. The actually-just-typed prompt
    is always the **last** tag in the message blob.
    """
    parts: List[str] = []
    for b in msg.content_blocks:
        if b.get("type") == "text":
            t = b.get("text")
            if isinstance(t, str):
                parts.append(t)
    full = "\n".join(parts).strip()
    if not full:
        return ""
    matches = re.findall(r"<user_query>([\s\S]*?)</user_query>", full)
    if matches:
        # The last tag is the just-submitted prompt; earlier tags are
        # historical context CodeBuddy injects on every turn ≥ 2.
        return matches[-1].strip()
    return full


def split_assistant_blocks(msg: CodeBuddyMessage) -> Dict[str, List[Dict[str, Any]]]:
    """Bucket assistant content blocks by type so callers can mint one
    `ThoughtStep` per logical event without re-iterating.

    Buckets returned: ``reasoning`` / ``text`` / ``tool_calls`` (preserving
    original order within each bucket).
    """
    out = {"reasoning": [], "text": [], "tool_calls": []}
    for b in msg.content_blocks:
        bt = b.get("type")
        if bt == "reasoning":
            out["reasoning"].append(b)
        elif bt == "text":
            out["text"].append(b)
        elif bt == "tool-call":
            out["tool_calls"].append(b)
    return out


def split_tool_results(msg: CodeBuddyMessage) -> List[Dict[str, Any]]:
    """Return the list of ``tool-result`` blocks in a role=tool message.

    A single CodeBuddy tool message bundles results for **all** parallel
    tool-calls from the previous assistant turn (unlike Cursor where each
    tool result is its own message). Callers should pair these by
    ``toolCallId`` against the prior assistant's tool_calls bucket.
    """
    return [b for b in msg.content_blocks if b.get("type") == "tool-result"]


def stringify_tool_input(args: Any, max_chars: int = 4096) -> str:
    """Compact JSON dump for tool inputs, truncated to keep step bodies sane."""
    if args is None:
        return ""
    try:
        s = json.dumps(args, ensure_ascii=False)
    except Exception:
        s = str(args)
    if len(s) > max_chars:
        return s[:max_chars] + f"…[+{len(s) - max_chars}ch]"
    return s


def stringify_tool_result(result: Any, max_chars: int = 8192) -> str:
    """Best-effort flatten of a CodeBuddy tool-result for a CoT step body.

    The shape is ``{"status": "success"|..., "success": bool,
    "result": <inner>, ...}``. We pull the inner result; if it's a dict
    with a ``content`` field (read_file etc.), surface that directly.
    """
    if result is None:
        return ""
    if isinstance(result, dict):
        inner = result.get("result", result)
        if isinstance(inner, dict):
            for key in ("content", "stdout", "text", "output", "data"):
                v = inner.get(key)
                if isinstance(v, str) and v:
                    s = v
                    break
            else:
                try:
                    s = json.dumps(inner, ensure_ascii=False)
                except Exception:
                    s = str(inner)
        else:
            try:
                s = json.dumps(inner, ensure_ascii=False)
            except Exception:
                s = str(inner)
    else:
        s = str(result)
    if len(s) > max_chars:
        return s[:max_chars] + f"…[+{len(s) - max_chars}ch]"
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Index → session-level metadata
# ─────────────────────────────────────────────────────────────────────────────


def index_request_summaries(index_blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pull the per-request summary list out of ``index.json``.

    Each entry typically has ``id`` (request_id), ``state``, ``startedAt``
    (ms), and ``usage = {inputTokens, outputTokens, totalTokens, lastTokens}``.
    Returns ``[]`` when the field is absent.
    """
    reqs = index_blob.get("requests") if isinstance(index_blob, dict) else None
    if not isinstance(reqs, list):
        return []
    out: List[Dict[str, Any]] = []
    for r in reqs:
        if not isinstance(r, dict):
            continue
        out.append({
            "request_id": str(r.get("id") or ""),
            "type": r.get("type"),
            "state": r.get("state"),
            "started_at_ms": r.get("startedAt"),
            "message_ids": list(r.get("messages") or []),
            "usage": dict(r.get("usage") or {}),
        })
    return out


def aggregate_token_usage(index_blob: Dict[str, Any]) -> Dict[str, int]:
    """Sum input/output/total tokens across every request in the index."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for r in index_request_summaries(index_blob):
        u = r.get("usage") or {}
        totals["input_tokens"] += int(u.get("inputTokens") or 0)
        totals["output_tokens"] += int(u.get("outputTokens") or 0)
        totals["total_tokens"] += int(u.get("totalTokens") or 0)
    return totals


def collect_models(messages: List[CodeBuddyMessage]) -> List[str]:
    """Distinct model_ids seen across messages (insertion order preserved)."""
    seen: set = set()
    out: List[str] = []
    for m in messages:
        if m.model_id and m.model_id not in seen:
            seen.add(m.model_id)
            out.append(m.model_id)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# v0.20.0: Thought-only diff API (Path B for CodeBuddy real-time thinking)
# ─────────────────────────────────────────────────────────────────────────────
#
# Background
# ----------
# CodeBuddy hooks (PreToolUse / PostToolUse / Stop) **don't carry the
# `reasoning` field** in their payloads, so the only way to surface
# Hunyuan/混元 等模型的真实思考链（与 Cursor 的 ``afterAgentThought`` 对齐）
# is to read ``messages/<id>.json`` files off disk.
#
# 历史上 ``transcript_watcher.py`` 的 codebuddy 分支已经在 mtime 变化时
# **整轮重抽** ``extract_session_cot``：正确，但慢——每条新消息都要重
# 写一次 cot.json，前端要等下一次 polling 才能看到。
#
# Path B 的目标：**只增量提取新消息里的 reasoning block**，把它们以
# ``agentThought`` 事件 append 到 events.jsonl。这个事件名跟 Cursor
# ``afterAgentThought`` 之后归一化的事件名一致，所以前端 SpanTree 已有
# 的紫色 🧠 + thinking phase 折叠逻辑能直接复用。本块新增的两个
# helper 是**纯 additive**：
#
#   - :func:`list_message_ids_in_index`  — 按 index.json 顺序列出 msg_id
#   - :func:`load_one_message`           — 读单条 message file → CodeBuddyMessage
#   - :func:`iter_reasoning_thoughts`    — 把单条 message 的 reasoning blocks
#                                          展开成 ``(text, idx_in_msg)`` 序列
#
# 它们都没有副作用，跟现有 ``load_transcript`` 行为完全一致。watcher
# 端通过持有 ``set[str] last_seen_msg_ids`` 实现 O(new_msgs) 增量扫描。
# 整个改动只服务 codebuddy ——cursor / claude code 路径未受影响。

def list_message_ids_in_index(index_path: Path) -> List[str]:
    """Return ``[msg_id, ...]`` in the order recorded by ``index.json``.

    Differs from :func:`load_transcript` by **not** opening any
    ``messages/<id>.json`` —— pure index-only walk for cheap polling.
    """
    try:
        index_blob = json.loads(_read_text_long(index_path))
    except Exception:
        return []
    if not isinstance(index_blob, dict):
        return []
    msg_id_order = index_blob.get("messages")
    if not isinstance(msg_id_order, list):
        return []
    out: List[str] = []
    for entry in msg_id_order:
        if isinstance(entry, dict):
            mid = str(entry.get("id") or "").strip()
        else:
            mid = str(entry or "").strip()
        if mid:
            out.append(mid)
    return out


def load_one_message(messages_dir: Path, msg_id: str) -> Optional[CodeBuddyMessage]:
    """Load + normalize a single ``messages/<msg_id>.json`` file.

    Returns ``None`` if the file is absent (CodeBuddy occasionally
    references a message id before flushing it) or unparseable. Uses
    the same long-path safe read as :func:`load_transcript`.
    """
    if not msg_id:
        return None
    f = messages_dir / f"{msg_id}.json"
    if not _exists_long(f):
        return None
    try:
        raw = json.loads(_read_text_long(f))
    except Exception:
        return None
    return _normalize_message(raw)


def iter_reasoning_thoughts(msg: CodeBuddyMessage) -> Iterator[Tuple[int, str]]:
    """Yield ``(idx_in_msg, text)`` for every non-empty reasoning block.

    ``idx_in_msg`` is the position **within the assistant message's
    content array** (not the global step index) — useful for stable
    sub-id generation when one message has multiple reasoning blocks.

    Only yields for ``role == "assistant"`` messages; user / tool
    messages return an empty iterator. Empty / whitespace reasoning
    bodies are skipped so the caller never emits empty thought events.
    """
    if msg is None or msg.role != "assistant":
        return
    for idx, block in enumerate(msg.content_blocks):
        if not isinstance(block, dict):
            continue
        if block.get("type") != "reasoning":
            continue
        text = block.get("text")
        if not isinstance(text, str):
            continue
        stripped = text.strip()
        if not stripped:
            continue
        yield idx, stripped


__all__ = [
    "CodeBuddyMessage",
    "_generated_at_ms_from_response_id",
    "aggregate_token_usage",
    "collect_models",
    "extract_user_text",
    "find_transcript_by_session_id",
    "index_request_summaries",
    "iter_reasoning_thoughts",
    "iter_recent_transcripts",
    "list_message_ids_in_index",
    "load_one_message",
    "load_transcript",
    "split_assistant_blocks",
    "split_tool_results",
    "stringify_tool_input",
    "stringify_tool_result",
    "transcript_path_from_events",
]
