"""Sidecar daemon watching Cursor's agent-transcripts/ + agent-tools/.

Background
----------
Cursor exposes 14 hook events (beforeReadFile / Shell / MCP / Edit / ...) via
``~/.cursor/hooks.json``. ``cot-stream.js`` already drains those and writes to
``<COT_EXTRACTOR_ROOT>/output/events/<sid>/events.jsonl``.

The remaining gap (verified empirically in this repo on 2026-04-28):
  * Glob / Grep / Delete / Task / SemanticSearch / WebFetch / WebSearch /
    TodoWrite / AskQuestion / ReadLints / AwaitShell / EditNotebook
    have **no Cursor hook event**, so cot-stream.js never sees them.
  * For tools that *do* produce results, the assistant transcript jsonl
    contains the ``tool_use`` block but **never** the ``tool_result`` —
    Cursor stores any oversized result as a separate
    ``agent-tools/<uuid>.txt`` artifact instead.

This watcher fills both gaps, **without** monkey-patching Cursor or
intercepting TLS:

    Layer B  (transcript_tail) :: tail <workspace>/agent-transcripts/<sid>/<sid>.jsonl
                                  -> emit ``agentToolCall`` event with full
                                     tool name + input for every tool_use.

    Layer C  (artifact_watch)  :: watch <workspace>/agent-tools/
                                  -> emit ``agentToolArtifact`` event for
                                     every new <uuid>.txt with size + sha256
                                     prefix + the first ~4 KB of content.

Both layers append line-JSON to the same ``events.jsonl`` schema that
``cot-stream.js`` writes:

    {"t": <ms>, "event": <name>, "cid": <session_id>, "tool": <name>,
     "brief_input": {...}?, "brief_output": {...}?, "payload": {...}}

so the existing cot_extractor consumers (and the dashboard frontend) can
ingest them with zero schema changes.

Why a daemon, not a hook
------------------------
Cursor hooks are short-lived synchronous one-shots that must finish < 100 ms.
A file watcher must be long-running. The two are orthogonal — this script
is intended to be launched once via ``agent-cot start`` or manually with
``python transcript_watcher.py`` and left running in the background.

Why polling instead of inotify / ReadDirectoryChangesW
------------------------------------------------------
Polling every 500 ms is well below any UX threshold and avoids one stdlib
abstraction per OS. Sessions per machine are O(10), files per session are
O(1) — the total work per tick is trivial.

Idempotency
-----------
Per-session state lives in ``<events_dir>/.watcher_state.json``:

    {
      "transcript_offset": <byte offset already consumed>,
      "transcript_tool_use_count": <int, monotonic>,
      "artifacts_seen": [<uuid>, ...]
    }

so re-launching the watcher mid-session never double-emits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Local sibling module — deterministic re-execution of Glob / Grep / Delete.
# Imported lazily-friendly so the watcher still starts if the file is absent
# (e.g. running an older deployment) — reproduction silently disables itself.
try:
    from tool_reproducer import reproduce as _tool_reproduce  # type: ignore
except Exception:  # pragma: no cover - best-effort
    _tool_reproduce = None  # type: ignore


LOG = logging.getLogger("transcript_watcher")

POLL_INTERVAL_S = 0.5
ARTIFACT_HEAD_BYTES = 4096
ARTIFACT_MAX_LINKED_BYTES = 64 * 1024  # never inline more than this
TRANSCRIPT_MAX_INPUT_CHARS = 4096  # truncate huge tool inputs

# Tools whose result we synthesize via deterministic local replay. Cursor
# emits no hook for them; their result also never lands on disk. We replay
# the call right after observing the agent's tool_use, while the workspace
# is still in (very nearly) the same state. See ``tool_reproducer.py``.
REPRODUCIBLE_TOOLS = frozenset({"Glob", "Grep", "rg", "Delete"})

# Tools that are known to be missing from cot-stream.js coverage. We still
# emit agentToolCall events for *all* tools (cheap and complete), but this
# list is referenced in metadata for the frontend to badge them clearly.
GAP_TOOLS = frozenset({
    "Glob", "Grep", "rg", "Delete", "Task", "SemanticSearch",
    "WebFetch", "WebSearch", "TodoWrite", "AskQuestion", "ReadLints",
    "AwaitShell", "EditNotebook", "GenerateImage", "FetchMcpResource",
    "ListMcpResources", "SwitchMode",
})


# ─────────────────────────────────────────────────────────────────────────────
# Path discovery
# ─────────────────────────────────────────────────────────────────────────────

def default_cursor_projects_root() -> Path:
    """Cursor stores per-workspace state under ~/.cursor/projects/<slug>/.

    The slug is the workspace path with separators flattened (e.g.
    ``d-ai-ide-langfuse`` for ``D:\\ai-ide-langfuse``). We don't need to
    decode it — we just iterate every subfolder and watch whatever is
    actively being written.
    """
    return Path.home() / ".cursor" / "projects"


def default_cot_root() -> Path:
    env = os.environ.get("COT_EXTRACTOR_ROOT")
    if env:
        return Path(env)
    # repo-relative fallback (this file lives at <repo>/scripts/)
    return Path(__file__).resolve().parent.parent


def default_events_root() -> Path:
    """v0.18.5: events.jsonl 父目录 —— 跟 cot-stream.js / cot_extractor 完全一致.

    优先级:
    1. ``$AGENT_COT_DATA_ROOT/events`` —— ``agent-cot start`` 注入。
    2. ``~/.agent-cot/data/events``    —— 用户级默认；wheel 安装态命中。
    3. ``<cot_root>/output/events``     —— 源码态开发兜底（回退由 caller
       负责，因为 ``cot_root`` 是 run() 时才知道）。

    返回前两档命中其一的绝对路径；都不命中时返回用户级默认（让 caller 直接
    mkdir 它）。源码态兜底的回退由 ``run()`` 自己处理。
    """
    env = os.environ.get("AGENT_COT_DATA_ROOT")
    if env:
        return Path(env).expanduser() / "events"
    return Path.home() / ".agent-cot" / "data" / "events"


# ─────────────────────────────────────────────────────────────────────────────
# CodeBuddy session discovery + extraction loop
# ─────────────────────────────────────────────────────────────────────────────
#
# Cursor's transcript layout (handled above) is workspace-shaped:
#   <ws>/agent-transcripts/<sid>/<sid>.jsonl
# CodeBuddy's is completely different and not workspace-rooted at all:
#   %LOCALAPPDATA%/CodeBuddyExtension/Data/<machine>/CodeBuddyIDE/<machine>/
#   history/<workspace_hash>/<sid>/{index.json, messages/<msg_id>.json}
# We bolt on a parallel mini-watcher rather than try to retrofit
# discover_sessions/tail_transcript, which deeply assume the Cursor
# layout (single jsonl tail + agent-tools artifact bucket).
#
# What this loop does each tick:
#   1. Find every codebuddy-* events.jsonl directory under cot_root.
#   2. From the first event in each, read payload.transcript_path → index.json.
#   3. Stat index.json + messages/ dir; if mtime changed since last tick,
#      re-run extract_session_cot for that session and overwrite
#      <cot_dir>/<sid>_cot.json. The session-list backend then picks it up.

@dataclass
class CodeBuddySessionState:
    """Per-session bookkeeping for the CodeBuddy parallel watcher.

    v0.20.0 (Path B – CodeBuddy thought-only fast path):
        ``seen_thought_keys`` / ``thought_state_path`` track which
        ``messages/<msg_id>.json`` reasoning blocks have already been
        emitted as ``agentThought`` events to ``events.jsonl``. This is
        independent of (and runs alongside) the existing mtime-based
        full-session re-extraction — it lets the dashboard show the
        Hunyuan/混元 model's thinking the moment a reasoning block lands
        on disk, without waiting for the slow ``extract_session_cot``
        rewrite. Persisted to ``.codebuddy_thoughts.json`` next to
        events.jsonl so a watcher restart never double-emits.

    Cursor / Claude code paths are completely unaffected — these
    fields are only consulted inside the codebuddy parallel branch.
    """
    session_id: str                 # bare sid (no codebuddy- prefix)
    events_dir: Path                # <cot_root>/output/events/codebuddy-<sid>/
    index_path: Optional[Path]      # CodeBuddy native transcript index.json
    last_index_mtime: float = 0.0
    last_messages_mtime: float = 0.0
    last_extract_mtime: float = 0.0
    # Path B fast-path bookkeeping:
    seen_thought_keys: set = field(default_factory=set)  # str keys
    thought_state_path: Optional[Path] = None
    thoughts_emitted: int = 0


def discover_codebuddy_sessions(cot_root: Path,
                                *, idle_seconds: float = 86400,
                                events_root: Optional[Path] = None) -> List[CodeBuddySessionState]:
    """Find every codebuddy-* events.jsonl active in the last N seconds.

    Resolution order for the transcript path:
      1. payload.transcript_path on the first event (most reliable; written
         directly by cot-stream-codebuddy.js).
      2. find_transcript_by_session_id from codebuddy_transcript (in case
         the events file has no payload yet).
    Sessions with neither resolved still get registered — extraction will
    just no-op until a transcript appears.

    v0.18.5: ``events_root`` 显式可传，缺省回退到老的 ``cot_root/output/events``，
    跟 SessionState.load 同样的兼容策略。
    """
    out: List[CodeBuddySessionState] = []
    if events_root is None:
        events_root = cot_root / "output" / "events"
    if not events_root.is_dir():
        return out
    cutoff = time.time() - idle_seconds
    try:
        from codebuddy_transcript import (  # type: ignore
            transcript_path_from_events,
            find_transcript_by_session_id,
        )
    except Exception:
        transcript_path_from_events = None  # type: ignore
        find_transcript_by_session_id = None  # type: ignore

    for entry in sorted(events_root.iterdir()):
        if not entry.is_dir():
            continue
        if not entry.name.startswith("codebuddy-"):
            continue
        sid = entry.name[len("codebuddy-"):]
        events_jsonl = entry / "events.jsonl"
        if not events_jsonl.is_file():
            continue
        try:
            mtime = events_jsonl.stat().st_mtime
        except OSError:
            continue
        if mtime < cutoff:
            continue

        # Peek first lines for payload.transcript_path.
        index_path: Optional[Path] = None
        try:
            with open(events_jsonl, "rb") as f:
                events_head: List[Dict[str, object]] = []
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        events_head.append(json.loads(raw.decode("utf-8", errors="replace")))
                    except Exception:
                        continue
                    if len(events_head) >= 5:
                        break
        except OSError:
            events_head = []
        if transcript_path_from_events is not None:
            try:
                index_path = transcript_path_from_events(events_head)  # type: ignore
            except Exception:
                index_path = None
        if index_path is None and find_transcript_by_session_id is not None:
            try:
                index_path = find_transcript_by_session_id(sid)  # type: ignore
            except Exception:
                index_path = None
        out.append(CodeBuddySessionState(
            session_id=sid,
            events_dir=entry,
            index_path=index_path,
            thought_state_path=entry / ".codebuddy_thoughts.json",
        ))
    return out


def _codebuddy_pair_mtimes(state: CodeBuddySessionState) -> Tuple[float, float]:
    """Return ``(index_mtime, messages_dir_mtime)`` for change detection."""
    try:
        from codebuddy_transcript import _exists_long  # type: ignore
    except Exception:
        return 0.0, 0.0
    if state.index_path is None or not _exists_long(state.index_path):
        return 0.0, 0.0
    try:
        idx_m = state.index_path.stat().st_mtime
    except OSError:
        idx_m = 0.0
    try:
        msgs_m = (state.index_path.parent / "messages").stat().st_mtime
    except OSError:
        msgs_m = 0.0
    return idx_m, msgs_m


def _run_codebuddy_extraction(state: CodeBuddySessionState, cot_root: Path) -> bool:
    """Re-run extract_session_cot for one CodeBuddy session, overwrite cot.json.

    Returns ``True`` on a successful write. Errors are logged at WARNING
    so the loop keeps making progress for siblings.
    """
    try:
        # Lazily import — keeps watcher startup fast when CodeBuddy isn't used.
        sys.path.insert(0, str(cot_root / "src"))
        from cot_extractor import extract_session_cot  # type: ignore
    except Exception as e:
        LOG.warning("cot_extractor import failed for codebuddy session: %s", e)
        return False
    if state.index_path is None:
        return False
    try:
        session_cot, _ = extract_session_cot(
            transcript_path=state.index_path,
            session_id=state.session_id,
            offset=0,
        )
    except Exception as e:
        LOG.warning("codebuddy extract failed for %s: %s", state.session_id, e)
        return False
    if session_cot is None:
        return False

    cot_dir_env = os.environ.get("COT_DIR")
    if cot_dir_env:
        cot_dir = Path(cot_dir_env)
    else:
        cot_dir = cot_root / "output" / "cot"
    cot_dir.mkdir(parents=True, exist_ok=True)
    target = cot_dir / f"{state.session_id}_cot.json"
    payload = json.dumps(session_cot.to_dict(), indent=2, ensure_ascii=False)
    try:
        target.write_text(payload, encoding="utf-8")
    except OSError as e:
        LOG.warning("codebuddy write failed for %s: %s", target, e)
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# v0.20.0 — Path B: CodeBuddy thought-only fast path
# ─────────────────────────────────────────────────────────────────────────────
#
# CodeBuddy's hooks (cot-stream-codebuddy.js) write PreToolUse / PostToolUse /
# Stop events but their payload **never** carries the ``reasoning`` field —
# CodeBuddy IDE only flushes that field into ``messages/<id>.json`` on disk.
# That makes the existing full-session re-extract (above) the only way to
# surface the model's thinking, but a single re-extract rewrites the whole
# cot.json (1-2s for long sessions) which feels laggy in the dashboard.
#
# The fast path below scans **only the reasoning blocks of newly-flushed
# messages** each tick and appends one tiny ``agentThought`` event per new
# block to ``events.jsonl`` — same shape that Cursor's afterAgentThought hook
# emits. Once the cot_extractor's normal ``codebuddy_reasoning`` path
# (``_extract_codebuddy_session_from_transcript``) catches up via the
# slower mtime-triggered branch, the thoughts also appear inside cot.json
# as ``THINKING_EXPLICIT`` steps — the front-end SpanTree's purple-brain +
# thinking-phase fold is **already** keyed off step_type, so it works
# unchanged. This file only needs to make the events.jsonl side honest;
# nothing here touches Cursor or Claude Code paths.
#
# Feature flag (rollback in 1 second, no code changes required):
#     AGENT_COT_CODEBUDDY_THOUGHT_STREAM = "0"  -> disable, behave exactly
#                                                  as before this change
#     AGENT_COT_CODEBUDDY_THOUGHT_STREAM = "1"  -> enable (default)
#
# Persistence: per-session ``.codebuddy_thoughts.json`` next to events.jsonl
# stores the set of (msg_id, idx_in_msg, hash[:8]) keys that were already
# emitted, so a watcher restart never double-emits the same thought.

def _codebuddy_thoughts_enabled() -> bool:
    """Path B master switch.

    Returns ``True`` unless the user explicitly set
    ``AGENT_COT_CODEBUDDY_THOUGHT_STREAM=0``. We default-on on purpose:
    the fast path is read-only on the IDE side, additive on disk
    (events.jsonl is already append-only), and adds < 5 ms per tick on
    a typical machine even when no codebuddy session is active.
    """
    raw = os.environ.get("AGENT_COT_CODEBUDDY_THOUGHT_STREAM")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _load_thought_state(state: "CodeBuddySessionState") -> None:
    """Restore ``seen_thought_keys`` from ``.codebuddy_thoughts.json``.

    No-op (and silent) if the file does not exist, is empty, or fails to
    parse — we'd rather emit an extra duplicate thought once than crash
    the watcher mid-tick. The only consequence of a corrupt state file
    is one extra duplicate ``agentThought`` event after the next IDE
    restart; the front-end de-dups by ``(msg_id, sub_idx)``.
    """
    p = state.thought_state_path
    if p is None or not p.exists():
        return
    try:
        blob = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception as e:
        LOG.warning("codebuddy thoughts state load failed for %s: %s",
                    state.session_id, e)
        return
    seen = blob.get("seen") if isinstance(blob, dict) else None
    if isinstance(seen, list):
        state.seen_thought_keys = {str(k) for k in seen if isinstance(k, str)}


def _save_thought_state(state: "CodeBuddySessionState") -> None:
    """Atomically persist ``seen_thought_keys`` (capped to last 8000 keys).

    Cap protects long-running sessions: even with 1 thought per second
    (≈ never, but defensive) 8000 keys keeps state under 600 KB. The
    cap drops oldest keys first (insertion order), which is safe
    because CodeBuddy never re-flushes an old reasoning block.
    """
    p = state.thought_state_path
    if p is None:
        return
    keys = list(state.seen_thought_keys)
    if len(keys) > 8000:
        keys = keys[-8000:]
        state.seen_thought_keys = set(keys)
    try:
        tmp = p.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"seen": keys}, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, p)
    except OSError as e:
        LOG.warning("codebuddy thoughts state save failed for %s: %s",
                    state.session_id, e)


def _thought_key(msg_id: str, idx_in_msg: int, text: str) -> str:
    """Stable de-dup key for a single reasoning block.

    Includes a short hash of the text so an in-place message rewrite
    (CodeBuddy occasionally appends to an existing reasoning block as
    streaming continues) emits a *fresh* event instead of being
    silently swallowed by the de-dup set.
    """
    h = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"{msg_id}:{idx_in_msg}:{h}"


def _emit_codebuddy_thoughts(state: "CodeBuddySessionState",
                              sink: "EventSink") -> int:
    """Diff-emit ``agentThought`` events for newly-flushed reasoning blocks.

    Returns the number of new events written this tick. Errors are
    logged at WARNING and the function returns ``0`` so the main loop
    keeps making progress for sibling sessions / the slow re-extract
    branch.

    Event schema (matches what the front-end's SpanTree already
    consumes for cursor's ``afterAgentThought``)::

        {
          "t": <epoch-ms>,
          "event": "agentThought",
          "cid": "codebuddy-<sid>",
          "provider": "codebuddy",
          "thinking_probe": "<reasoning text, capped at 8 KB>",
          "tool": "",
          "source": "codebuddy-thought-watcher",
          "payload": {
            "msg_id": "<assistant message id>",
            "sub_idx": <int>,                # idx of this block within the message
            "model_id": "<model id if known>",
            "request_id": "<request id if known>",
            "transcript_path": "<absolute path to index.json>"
          }
        }
    """
    if not _codebuddy_thoughts_enabled():
        return 0
    if state.index_path is None:
        return 0
    try:
        from codebuddy_transcript import (  # type: ignore
            list_message_ids_in_index,
            load_one_message,
            iter_reasoning_thoughts,
            _exists_long,
        )
    except Exception as e:
        LOG.warning("codebuddy_transcript import failed (thought stream): %s", e)
        return 0
    if not _exists_long(state.index_path):
        return 0
    messages_dir = state.index_path.parent / "messages"
    try:
        ids = list_message_ids_in_index(state.index_path)
    except Exception as e:
        LOG.warning("codebuddy thought stream: list ids failed for %s: %s",
                    state.session_id, e)
        return 0

    cid = f"codebuddy-{state.session_id}"
    transcript_path_str = str(state.index_path)
    emitted = 0

    for msg_id in ids:
        # Cheap pre-filter: if every reasoning slot we've ever seen for
        # this msg_id is already in seen_thought_keys with the latest
        # hash, skip the file read entirely. We approximate by checking
        # whether the *first* slot is recorded — if a message has any
        # reasoning at all, slot 0 is virtually always present, and on
        # cache hit we save an open() per known message every tick.
        # On any miss / hash change we still go read the file.
        try:
            msg = load_one_message(messages_dir, msg_id)
        except Exception:
            continue
        if msg is None or msg.role != "assistant":
            continue

        # Use the response_id-derived timestamp if available; otherwise
        # fall back to "now". Either way the front-end orders by t,
        # and we never invent timestamps that pretend to be older than
        # actually-observed-by-us.
        evt_t_ms = msg.generated_at_ms if msg.generated_at_ms else int(
            time.time() * 1000)

        for idx_in_msg, text in iter_reasoning_thoughts(msg):
            # Cap the probe to 8 KB so events.jsonl stays small even
            # for very chatty Hunyuan reasoning blocks. The cot.json
            # full-extract branch keeps the unredacted text — this
            # cap only affects the events stream.
            probe = text if len(text) <= 8192 else text[:8192] + \
                f"…[+{len(text) - 8192}ch]"
            key = _thought_key(msg_id, idx_in_msg, probe)
            if key in state.seen_thought_keys:
                continue
            evt = {
                "t": evt_t_ms,
                "event": "agentThought",
                "cid": cid,
                "provider": "codebuddy",
                "tool": "",
                "thinking_probe": probe,
                "source": "codebuddy-thought-watcher",
                "payload": {
                    "msg_id": msg_id,
                    "sub_idx": idx_in_msg,
                    "model_id": msg.model_id or "",
                    "model_name": msg.model_name or "",
                    "request_id": msg.request_id or "",
                    "response_id": msg.response_id or "",
                    "trace_id": msg.trace_id or "",
                    "thought_chars": len(text),
                    "transcript_path": transcript_path_str,
                },
            }
            try:
                # Reuse EventSink's writer so we share the same fsync
                # discipline and never interleave a half line.
                sink_state_shim = _CodeBuddyEventSinkAdapter(
                    session_id=cid,
                    events_path=state.events_dir / "events.jsonl",
                )
                sink.write(sink_state_shim, evt)  # type: ignore[arg-type]
            except Exception as e:
                LOG.warning("codebuddy thought emit failed for %s: %s",
                            state.session_id, e)
                # don't update seen_thought_keys — try again next tick.
                continue
            state.seen_thought_keys.add(key)
            emitted += 1

    if emitted > 0:
        state.thoughts_emitted += emitted
        _save_thought_state(state)

    return emitted


@dataclass
class _CodeBuddyEventSinkAdapter:
    """Tiny duck-type so :class:`EventSink.write` accepts our path.

    The codebuddy parallel watcher doesn't have a full ``SessionState``
    (those are Cursor-shaped), so this 2-field shim lets us reuse
    EventSink without weakening its type contract.
    """
    session_id: str
    events_path: Path



# ─────────────────────────────────────────────────────────────────────────────
# State per session
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    session_id: str
    transcript_path: Path
    artifacts_dir: Path
    events_path: Path
    state_path: Path

    transcript_offset: int = 0
    transcript_tool_use_count: int = 0
    artifacts_seen: List[str] = field(default_factory=list)

    @classmethod
    def load(cls, *, session_id: str, transcript_path: Path,
             artifacts_dir: Path, cot_root: Path,
             events_root: Optional[Path] = None) -> "SessionState":
        # v0.18.5: events_root 显式传 —— wheel 安装态走 ~/.agent-cot/data/events，
        # 跟 cot-stream.js / cot_extractor._load_cursor_events 完全对齐。
        # 只有 caller 显式不传（老代码路径）时才回退到老的 cot_root/output/events。
        base = events_root if events_root is not None else (cot_root / "output" / "events")
        events_dir = base / session_id
        events_dir.mkdir(parents=True, exist_ok=True)
        events_path = events_dir / "events.jsonl"
        state_path = events_dir / ".watcher_state.json"
        st = cls(
            session_id=session_id,
            transcript_path=transcript_path,
            artifacts_dir=artifacts_dir,
            events_path=events_path,
            state_path=state_path,
        )
        if state_path.exists():
            try:
                # utf-8-sig handles BOMs that Windows tools (PowerShell
                # Set-Content -Encoding UTF8) sometimes inject when humans
                # edit the state file by hand.
                blob = json.loads(state_path.read_text(encoding="utf-8-sig"))
                st.transcript_offset = int(blob.get("transcript_offset", 0))
                st.transcript_tool_use_count = int(
                    blob.get("transcript_tool_use_count", 0))
                st.artifacts_seen = list(blob.get("artifacts_seen", []))
            except Exception as e:
                LOG.warning("could not load state %s: %s", state_path, e)
        return st

    def save(self) -> None:
        blob = {
            "transcript_offset": self.transcript_offset,
            "transcript_tool_use_count": self.transcript_tool_use_count,
            # cap retained list to avoid unbounded growth — sessions rarely
            # exceed a few hundred artifacts; older entries are still on disk.
            "artifacts_seen": self.artifacts_seen[-2000:],
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(blob), encoding="utf-8")
        os.replace(tmp, self.state_path)


# ─────────────────────────────────────────────────────────────────────────────
# Discovery: which sessions are alive?
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiscoveredSession:
    session_id: str
    transcript_path: Path
    artifacts_dir: Path
    last_mtime: float


def discover_sessions(projects_root: Path,
                      *, idle_seconds: float = 86400) -> List[DiscoveredSession]:
    """Walk all Cursor workspaces, return sessions whose transcript was
    modified within ``idle_seconds`` (default 24 h).

    The default cap protects us from re-tailing weeks-old conversations on
    every restart while still picking up anything worked on today.
    """
    out: List[DiscoveredSession] = []
    if not projects_root.exists():
        return out
    cutoff = time.time() - idle_seconds
    for workspace in sorted(projects_root.iterdir()):
        transcripts = workspace / "agent-transcripts"
        artifacts = workspace / "agent-tools"
        if not transcripts.is_dir():
            continue
        for sid_dir in transcripts.iterdir():
            if not sid_dir.is_dir():
                continue
            sid = sid_dir.name
            jsonl = sid_dir / f"{sid}.jsonl"
            if not jsonl.exists():
                continue
            try:
                mtime = jsonl.stat().st_mtime
            except OSError:
                continue
            if mtime < cutoff:
                continue
            out.append(DiscoveredSession(
                session_id=sid,
                transcript_path=jsonl,
                artifacts_dir=artifacts,
                last_mtime=mtime,
            ))
    out.sort(key=lambda d: d.last_mtime, reverse=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Layer B: transcript tailer
# ─────────────────────────────────────────────────────────────────────────────

def _truncate(value: object) -> object:
    """Cap any string/bytes field at TRANSCRIPT_MAX_INPUT_CHARS."""
    if isinstance(value, str):
        if len(value) > TRANSCRIPT_MAX_INPUT_CHARS:
            return value[:TRANSCRIPT_MAX_INPUT_CHARS] + \
                f"…[+{len(value) - TRANSCRIPT_MAX_INPUT_CHARS}ch]"
        return value
    if isinstance(value, list):
        out = []
        cap = min(len(value), 50)
        for v in value[:cap]:
            out.append(_truncate(v))
        if len(value) > cap:
            out.append(f"…[+{len(value) - cap} items]")
        return out
    if isinstance(value, dict):
        return {k: _truncate(v) for k, v in value.items()}
    return value


def _extract_brief_input(tool_name: str, raw_input: object) -> Dict[str, object]:
    """Pick the few high-value fields per tool so the dashboard can render a
    compact summary without parsing the full payload."""
    if not isinstance(raw_input, dict):
        return {}
    brief: Dict[str, object] = {}
    take = ("path", "file_path", "filepath", "command", "cwd",
            "working_directory", "url", "query", "pattern", "glob_pattern",
            "target_directory", "search_queries", "objective", "subagent_type",
            "description")
    for k in take:
        if k in raw_input and raw_input[k] not in (None, ""):
            v = raw_input[k]
            if isinstance(v, (str, int, float, bool)):
                s = str(v)
                brief[k] = s[:300] + ("…" if len(s) > 300 else "")
            elif isinstance(v, list):
                brief[k] = _truncate(v)
            else:
                brief[k] = _truncate(v)
    # tool-specific extras
    if tool_name == "TodoWrite":
        todos = raw_input.get("todos")
        if isinstance(todos, list):
            brief["todos_count"] = len(todos)
            brief["todos_preview"] = [
                (t or {}).get("content", "") for t in todos[:5]
            ]
    if tool_name in {"Grep", "rg"} and "output_mode" in raw_input:
        brief["output_mode"] = raw_input["output_mode"]
    return brief


def _iter_complete_lines(buf: bytes) -> Tuple[List[bytes], bytes]:
    """Split ``buf`` on \\n boundaries; return (complete_lines, leftover)."""
    if not buf:
        return [], buf
    # Cursor writes line-JSON, so the last newline marks the end of the last
    # complete record. Anything after it is a partial write we'll re-read on
    # the next tick.
    last_nl = buf.rfind(b"\n")
    if last_nl < 0:
        return [], buf
    head = buf[: last_nl + 1]
    tail = buf[last_nl + 1 :]
    lines = [ln for ln in head.split(b"\n") if ln.strip()]
    return lines, tail


def tail_transcript(state: SessionState, sink: "EventSink",
                    *, reproduce_results: bool = True) -> int:
    """Read new bytes from the transcript jsonl and emit ``agentToolCall``
    events for every ``tool_use`` block found.

    For tools in ``REPRODUCIBLE_TOOLS`` (Glob / Grep / Delete), this also
    immediately re-runs the call locally and emits an ``agentToolResult``
    event right after, so downstream consumers see both halves of the pair.

    Returns the number of events emitted this tick (calls + results).
    """
    try:
        size = state.transcript_path.stat().st_size
    except OSError as e:
        LOG.debug("transcript stat failed: %s", e)
        return 0

    if size < state.transcript_offset:
        # File was rewound (rare; happens if Cursor truncates). Re-read from
        # start to be safe — we de-dup via tool_use_count, not offset.
        LOG.warning("transcript rewound for %s, restarting from byte 0",
                    state.session_id)
        state.transcript_offset = 0
        state.transcript_tool_use_count = 0

    if size == state.transcript_offset:
        return 0

    try:
        with open(state.transcript_path, "rb") as f:
            f.seek(state.transcript_offset)
            chunk = f.read(size - state.transcript_offset)
    except OSError as e:
        LOG.warning("transcript read failed: %s", e)
        return 0

    lines, leftover = _iter_complete_lines(chunk)
    consumed = len(chunk) - len(leftover)
    state.transcript_offset += consumed

    emitted = 0
    seen_tool_uses_so_far = 0  # counter for *all* tool_use seen so far this run
    for raw in lines:
        try:
            obj = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            continue
        msg = (obj.get("message") or {})
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        # Each transcript line is one assistant turn that may contain N
        # tool_use blocks. We count them globally so re-launches can resume.
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            seen_tool_uses_so_far += 1
            # Skip ones we already emitted in a previous run.
            if seen_tool_uses_so_far <= state.transcript_tool_use_count:
                continue
            tool_name = str(block.get("name") or "unknown")
            raw_input = block.get("input")
            tool_use_id = block.get("id")
            call_evt = {
                "t": int(time.time() * 1000),
                "event": "agentToolCall",
                "cid": state.session_id,
                "tool": tool_name,
                "source": "transcript",
                "brief_input": _extract_brief_input(tool_name, raw_input),
                "payload": {
                    "tool_call_idx": seen_tool_uses_so_far,
                    "tool_use_id": tool_use_id,
                    "name": tool_name,
                    "is_gap_tool": tool_name in GAP_TOOLS,
                    "input": _truncate(raw_input),
                },
            }
            sink.write(state, call_evt)
            state.transcript_tool_use_count = seen_tool_uses_so_far
            emitted += 1

            # Reproduction layer: for deterministic file-system tools we
            # re-run the same call ourselves so a paired result event lands
            # in events.jsonl. We deliberately do this in-line (not threaded)
            # — Glob/Grep on this repo cost < 30 ms, way below the 500 ms
            # poll interval, and ordering matters for downstream attribution.
            if (reproduce_results
                    and _tool_reproduce is not None
                    and tool_name in REPRODUCIBLE_TOOLS
                    and isinstance(raw_input, dict)):
                result = _tool_reproduce(tool_name, raw_input)
                if result is not None:
                    sink.write(state, {
                        "t": int(time.time() * 1000),
                        "event": "agentToolResult",
                        "cid": state.session_id,
                        "tool": tool_name,
                        "source": "reproduced",
                        "brief_output": _brief_reproduced_output(result),
                        "payload": {
                            "tool_call_idx": seen_tool_uses_so_far,
                            "tool_use_id": tool_use_id,
                            "name": tool_name,
                            # Full result, lightly truncated so events.jsonl
                            # stays line-buffered friendly.
                            "result": _truncate(result),
                        },
                    })
                    emitted += 1
    return emitted


def _brief_reproduced_output(result: Dict) -> Dict:
    """Compact one-liner summary of a reproduction result for SessionList /
    timeline rendering. Keeps the heavy detail in ``payload.result``.
    """
    if not isinstance(result, dict):
        return {}
    tool = result.get("tool")
    out: Dict[str, object] = {
        "ok": bool(result.get("ok", False)),
        "elapsed_ms": result.get("elapsed_ms"),
    }
    if tool == "Glob":
        out["match_count"] = result.get("match_count", 0)
        out["truncated"] = bool(result.get("truncated"))
    elif tool == "Grep":
        out["match_count"] = result.get("match_count", 0)
        out["file_count"] = result.get("file_count", 0)
        out["truncated"] = bool(result.get("truncated"))
        out["via"] = result.get("via")
    elif tool == "Delete":
        out["still_exists"] = bool(result.get("still_exists"))
    if not out["ok"] and result.get("error"):
        out["error"] = str(result["error"])[:200]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Layer C: artifact watcher
# ─────────────────────────────────────────────────────────────────────────────

def _file_digest(path: Path) -> Tuple[int, str, str]:
    """Return ``(size, sha256_prefix12, head_text)`` for an artifact.

    head_text is the first ~4 KB decoded as UTF-8 with replacement, suitable
    for inline preview in the dashboard. We never load > ARTIFACT_HEAD_BYTES
    bytes at this stage; full content stays on disk.
    """
    size = path.stat().st_size
    h = hashlib.sha256()
    head = b""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(64 * 1024)
            if not chunk:
                break
            h.update(chunk)
            if len(head) < ARTIFACT_HEAD_BYTES:
                head += chunk[: ARTIFACT_HEAD_BYTES - len(head)]
    return size, h.hexdigest()[:12], head.decode("utf-8", errors="replace")


def _session_active_window(events_path: Path) -> Optional[Tuple[float, float]]:
    """Read this session's existing events.jsonl and return (min_t, max_t)
    in seconds. cot-stream.js writes a ``t`` field (ms) on every hook event,
    so the existing log is a reliable proxy for "when this session was
    actually producing tool calls" — much more reliable than a transcript
    file mtime, which Cursor sometimes touches unrelatedly.

    Returns None if there are no existing events to anchor against.
    """
    if not events_path.exists():
        return None
    lo: Optional[float] = None
    hi: Optional[float] = None
    try:
        with open(events_path, "rb") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8", errors="replace"))
                except Exception:
                    continue
                # Skip our own self-emitted events when computing the window:
                #   - agentToolArtifact: t is the artifact's mtime (target of
                #     attribution, can't be used as a ground-truth anchor)
                #   - agentToolCall:     t is the watcher's wall-clock time at
                #     replay (not the real tool call time), so it would falsely
                #     stretch the window to "now" and steal artifacts from
                #     concurrent sessions
                #   - agentToolResult:   same caveat as agentToolCall
                if obj.get("event") in {"agentToolArtifact",
                                        "agentToolCall",
                                        "agentToolResult"}:
                    continue
                t = obj.get("t")
                if not isinstance(t, (int, float)):
                    continue
                ts = float(t) / 1000.0
                if lo is None or ts < lo:
                    lo = ts
                if hi is None or ts > hi:
                    hi = ts
    except OSError:
        return None
    if lo is None or hi is None:
        return None
    return (lo, hi)


def watch_artifacts(state: SessionState, sink: "EventSink",
                    *, all_sessions: Dict[str, "SessionState"]) -> int:
    """Scan the workspace's agent-tools/ for new files and emit
    ``agentToolArtifact`` events.

    agent-tools/ is *workspace*-scoped, not session-scoped: every session in
    the same Cursor workspace dumps oversized tool results into the same
    folder. Attribution rule (verified empirically against this repo's data
    on 2026-04-28):

      For each new artifact file, attribute it to the unique session whose
      events.jsonl time window [min_t-60s, max_t+300s] contains the
      artifact's mtime. If multiple sessions match (rare — would mean two
      sessions overlapped within 5 min), pick the one with the closest
      max_t. If none match, leave the artifact unattributed for now —
      future ticks (with newer events.jsonl entries) will retry.

    The 60s/300s slack handles two known cases:
      * pre-roll: artifact appears slightly before the first hook event
        (possible when the very first tool call is e.g. WebFetch which has
        no Cursor hook).
      * post-roll: artifact lingers a few minutes past the last hook event
        when the user keeps the chat open.
    """
    if not state.artifacts_dir.is_dir():
        return 0
    emitted = 0
    seen = set(state.artifacts_seen)
    try:
        candidates = sorted(state.artifacts_dir.iterdir())
    except OSError:
        return 0

    # Build (sid, lo, hi) windows for every live session sharing this folder.
    windows: List[Tuple[str, float, float]] = []
    for sid, st in all_sessions.items():
        if st.artifacts_dir != state.artifacts_dir:
            continue
        win = _session_active_window(st.events_path)
        if win is None:
            continue
        windows.append((sid, win[0] - 60.0, win[1] + 300.0))

    for entry in candidates:
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in {".txt", ".json", ".log"}:
            continue
        key = entry.name
        if key in seen:
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue

        # Find the matching session(s).
        matches = [(sid, lo, hi) for sid, lo, hi in windows
                   if lo <= mtime <= hi]
        if not matches:
            continue
        if len(matches) > 1:
            # Closest-by-end-of-window (most recent producer wins).
            matches.sort(key=lambda m: abs(mtime - m[2]))
        target_sid = matches[0][0]
        if target_sid != state.session_id:
            # A peer session will claim it on its own tick.
            continue

        try:
            size, digest, head = _file_digest(entry)
        except OSError as e:
            LOG.warning("artifact digest failed: %s", e)
            continue
        evt = {
            "t": int(mtime * 1000),
            "event": "agentToolArtifact",
            "cid": state.session_id,
            "tool": "",
            "source": "agent-tools",
            "brief_output": {
                "artifact_name": entry.name,
                "size": size,
                "head_chars": len(head),
            },
            "payload": {
                "artifact_path": str(entry),
                "artifact_size": size,
                "sha256_12": digest,
                "head_text": head if size <= ARTIFACT_MAX_LINKED_BYTES else
                             head + f"\n…[+{size - ARTIFACT_HEAD_BYTES}B on disk]",
                "matched_sessions": len(matches),
            },
        }
        sink.write(state, evt)
        state.artifacts_seen.append(key)
        emitted += 1
    return emitted


# ─────────────────────────────────────────────────────────────────────────────
# EventSink: append-only writer with batched fsync
# ─────────────────────────────────────────────────────────────────────────────

class EventSink:
    """Append-only writer to ``events.jsonl``.

    cot-stream.js writes via blocking appendFileSync — same file, same
    process group race conditions don't matter on Linux/macOS because both
    use O_APPEND, and on Windows because writes < 4 KB are atomic. We rely
    on that; no locking.
    """
    def __init__(self) -> None:
        self.events_written = 0

    def write(self, state: SessionState, evt: Dict[str, object]) -> None:
        line = json.dumps(evt, ensure_ascii=False, separators=(",", ":")) + "\n"
        with open(state.events_path, "ab") as f:
            f.write(line.encode("utf-8"))
        self.events_written += 1


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run(*, projects_root: Path, cot_root: Path,
        once: bool = False,
        only_session: Optional[str] = None,
        idle_seconds: float = 86400,
        reproduce_results: bool = True,
        events_root: Optional[Path] = None) -> None:
    # v0.18.5: events_root 默认走 ~/.agent-cot/data/events（用户级，跟 hook /
    # extractor / backend 同源）；老调用者不传时回退到 cot_root/output/events。
    if events_root is None:
        events_root = default_events_root()
    try:
        events_root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        LOG.warning("events_root mkdir failed (%s) — continuing, "
                    "individual session dirs will retry", e)
    LOG.info("watcher start: projects=%s, cot_root=%s, events_root=%s, once=%s",
             projects_root, cot_root, events_root, once)
    sink = EventSink()
    states: Dict[str, SessionState] = {}
    stop = {"flag": False}

    def _stop(signum, frame):  # noqa: ARG001
        LOG.info("signal %s received, exiting after current tick", signum)
        stop["flag"] = True

    if not once:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _stop)
            except (ValueError, OSError):
                pass  # signal not available on this platform/thread

    cb_states: Dict[str, CodeBuddySessionState] = {}

    while True:
        discovered = discover_sessions(projects_root,
                                       idle_seconds=idle_seconds)
        live_ids: List[str] = []
        for d in discovered:
            if only_session and d.session_id != only_session:
                continue
            live_ids.append(d.session_id)
            st = states.get(d.session_id)
            if st is None:
                st = SessionState.load(
                    session_id=d.session_id,
                    transcript_path=d.transcript_path,
                    artifacts_dir=d.artifacts_dir,
                    cot_root=cot_root,
                    events_root=events_root,
                )
                states[d.session_id] = st
                LOG.info("session attached: %s "
                         "(offset=%d, tool_use_seen=%d, artifacts_seen=%d)",
                         d.session_id, st.transcript_offset,
                         st.transcript_tool_use_count, len(st.artifacts_seen))
            tu = tail_transcript(st, sink,
                                 reproduce_results=reproduce_results)
            ar = watch_artifacts(st, sink, all_sessions=states)
            if tu or ar:
                st.save()
                LOG.info("session %s: +%d tool_use, +%d artifacts "
                         "(events_written=%d)",
                         st.session_id, tu, ar, sink.events_written)

        # ── Parallel mini-watcher for CodeBuddy ───────────────────────
        # Cheap (O(sessions) stat calls per tick) and skipped entirely
        # when no codebuddy-* events dirs exist.
        for cb in discover_codebuddy_sessions(cot_root,
                                              idle_seconds=idle_seconds,
                                              events_root=events_root):
            if only_session and cb.session_id != only_session:
                continue
            existing = cb_states.get(cb.session_id)
            if existing is None:
                cb_states[cb.session_id] = cb
                # v0.20.0 (Path B): restore the per-session de-dup set so
                # a watcher restart never re-emits a thought we already
                # appended to events.jsonl in a prior run.
                _load_thought_state(cb)
                LOG.info("codebuddy session attached: %s (index=%s, "
                         "thoughts_seen=%d)",
                         cb.session_id, cb.index_path,
                         len(cb.seen_thought_keys))
                existing = cb
            else:
                # Refresh in case index_path showed up after first tick.
                if existing.index_path is None and cb.index_path is not None:
                    existing.index_path = cb.index_path
                    if existing.thought_state_path is None:
                        existing.thought_state_path = cb.thought_state_path
                        _load_thought_state(existing)
                    LOG.info("codebuddy session %s: index now %s",
                             cb.session_id, cb.index_path)

            # ── Path B fast path: emit agentThought events for any new
            # reasoning blocks that have just landed on disk. Runs
            # **every tick**, even before we decide whether to
            # re-extract — this is what lets the dashboard see a
            # thought ~ 0.5 s after the model finishes streaming it,
            # instead of waiting for the slower full re-extract.
            new_thoughts = _emit_codebuddy_thoughts(existing, sink)
            if new_thoughts > 0:
                LOG.info("codebuddy session %s: +%d thought events "
                         "(events_written=%d)",
                         existing.session_id, new_thoughts,
                         sink.events_written)

            idx_m, msgs_m = _codebuddy_pair_mtimes(existing)
            if idx_m == 0.0 and msgs_m == 0.0:
                continue
            if (idx_m == existing.last_index_mtime
                    and msgs_m == existing.last_messages_mtime):
                continue
            ok = _run_codebuddy_extraction(existing, cot_root)
            if ok:
                existing.last_index_mtime = idx_m
                existing.last_messages_mtime = msgs_m
                existing.last_extract_mtime = time.time()
                LOG.info("codebuddy session %s: re-extracted "
                         "(index_mtime=%.3f, msgs_mtime=%.3f)",
                         existing.session_id, idx_m, msgs_m)

        if once:
            break
        if stop["flag"]:
            break
        time.sleep(POLL_INTERVAL_S)

    LOG.info("watcher stopped: total_events_written=%d", sink.events_written)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="transcript_watcher",
        description="Tail Cursor agent-transcripts/* and agent-tools/* "
                    "and append agentToolCall / agentToolArtifact events "
                    "to the same events.jsonl that cot-stream.js writes.",
    )
    p.add_argument("--projects-root", type=Path,
                   default=default_cursor_projects_root(),
                   help="Cursor projects root "
                        "(default: %(default)s)")
    p.add_argument("--cot-root", type=Path,
                   default=default_cot_root(),
                   help="cot-extractor root holding output/events/ "
                        "(default: %(default)s)")
    p.add_argument("--events-root", type=Path,
                   default=default_events_root(),
                   help="parent dir for events.jsonl per-session subfolders "
                        "(v0.18.5+; default ~/.agent-cot/data/events, "
                        "overridable via $AGENT_COT_DATA_ROOT)")
    p.add_argument("--once", action="store_true",
                   help="single sweep then exit (useful for tests / cron)")
    p.add_argument("--session", default=None,
                   help="only watch this session id; default: all live")
    p.add_argument("--idle-seconds", type=float, default=86400,
                   help="ignore transcripts not modified in last N seconds "
                        "(default 24 h)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="verbose logging")
    p.add_argument("--no-reproduce", action="store_true",
                   help="disable deterministic re-execution of Glob/Grep/"
                        "Delete (results will be missing from events.jsonl, "
                        "useful when you only want to capture inputs)")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    run(
        projects_root=args.projects_root,
        cot_root=args.cot_root,
        events_root=args.events_root,
        once=args.once,
        only_session=args.session,
        idle_seconds=args.idle_seconds,
        reproduce_results=not args.no_reproduce,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
