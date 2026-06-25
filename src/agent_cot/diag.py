"""Unified pipeline diagnostic logger.

This module exists to answer one question fast when a colleague reports
"I see less trace than you do, with the same prompt":

    Where did the pipeline drop the event?

Pipeline stages (in time order, every IDE):

    1. IDE fires hook               (cot-bridge.js / cot-stream.js /
                                     cot-stream-codebuddy.js /
                                     claude_stream_hook.py)
    2. hook writes events.jsonl      (<data_root>/events/<sid>/events.jsonl)
    3. hook spawns extract_cot.py    (cot-bridge / codebuddy / claude — Stop event)
    4. extract_cot.py writes cot.json (<data_root>/cot/<sid>_cot.json)
    5. backend session_scanner reads cot.json and serves /api/sessions
    6. frontend fetches /api/sessions/<sid>/cot and renders

Every stage now writes a one-line breadcrumb to
``~/.agent-cot/logs/pipeline.log`` with a uniform format:

    [<iso_ts>] [<stage>] [<ide>] [sid=<sid>] event=<name> ok=<bool> note="..."

Why all in one file (and not one per stage):
* Colleague runs ``tail -f ~/.agent-cot/logs/pipeline.log`` once and SEES
  every transition. If line N+1 never arrives, that's where the pipeline
  stopped.
* Hook → extractor → backend writes happen across multiple processes,
  potentially across reboot boundaries; a unified log makes correlation
  trivial via the ``sid=`` column.

The file is append-only, never rotated by us — keep it tiny (~50 bytes
per event), bound to ~1k events per session in practice. ``agent-cot
status`` shows the path; ``agent-cot doctor --deep`` tails the tail for
the user.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# Stable log path resolution — never raises, never relies on env beyond
# HOME. If even ``Path.home()`` fails (extremely degraded environment),
# we fall back to cwd; the diagnostic line still lands somewhere.
def log_path() -> Path:
    """Resolve the canonical pipeline log path.

    Resolution order (first hit wins):
      1. ``AGENT_COT_PIPELINE_LOG`` env (user override)
      2. ``~/.agent-cot/logs/pipeline.log``
      3. ``./agent-cot-pipeline.log`` (last-resort, cwd)
    """
    env = os.environ.get("AGENT_COT_PIPELINE_LOG")
    if env:
        return Path(env).expanduser()
    try:
        return Path.home() / ".agent-cot" / "logs" / "pipeline.log"
    except Exception:
        return Path.cwd() / "agent-cot-pipeline.log"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def log(
    stage: str,
    *,
    ide: str = "-",
    sid: str = "-",
    event: str = "-",
    ok: bool = True,
    **note: Any,
) -> None:
    """Append a single breadcrumb to the pipeline log.

    Parameters
    ----------
    stage:
        Short identifier for the pipeline stage. Conventional values:
        ``hook.cursor`` / ``hook.codebuddy`` / ``hook.claude`` /
        ``extractor`` / ``backend`` / ``cli``.
    ide:
        IDE / tool that triggered this stage. ``cursor`` / ``codebuddy``
        / ``claude`` / ``vscode`` / ``-`` (unknown).
    sid:
        Session id (with prefix if applicable, e.g. ``codebuddy-<sid>``).
    event:
        Hook event name (``PostToolUse`` / ``Stop`` / ``SessionEnd``…)
        or pipeline action (``extract_spawn`` / ``cot_written``).
    ok:
        ``True`` = step succeeded; ``False`` = step failed (note should
        carry the error reason).
    **note:
        Arbitrary key=value pairs to embed in the log line. Reserved
        keys: ``error`` (str), ``count`` (int), ``path`` (str), ``ms``
        (float duration since the previous breadcrumb at the same sid).

    The call is **defensive**: any I/O / encoding failure is swallowed
    silently. Diagnostic logging must never crash the pipeline.
    """
    try:
        p = log_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        status = "ok" if ok else "FAIL"
        note_kv = " ".join(
            f"{k}={_safe_kv(v)}" for k, v in note.items() if v is not None
        )
        line = (
            f"[{_iso_now()}] [{stage}] [{ide}] [sid={sid}] "
            f"event={event} status={status}"
            + (f" {note_kv}" if note_kv else "")
            + "\n"
        )
        with open(p, "a", encoding="utf-8", errors="replace") as f:
            f.write(line)
    except Exception:
        # Never raise from a logger — hooks treat any exception as fatal.
        pass


def _safe_kv(v: Any) -> str:
    """Stringify ``v`` for the log line. Quote spaces, escape quotes."""
    if isinstance(v, str):
        s = v
    elif isinstance(v, (int, float, bool)):
        return str(v)
    elif v is None:
        return "-"
    else:
        try:
            s = json.dumps(v, ensure_ascii=False)
        except Exception:
            s = repr(v)
    if any(ch in s for ch in (" ", "\t", '"')):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def tail(n: int = 50) -> str:
    """Return the last ``n`` lines of the pipeline log (best effort)."""
    try:
        p = log_path()
        if not p.is_file():
            return ""
        # cheap tail: read full file; the log is bounded by event count,
        # not by time, so it stays small in practice.
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def stamp_env() -> dict[str, str]:
    """Snapshot the relevant env vars + paths for embedding into the
    pipeline log on each ``agent-cot start``. Helpful when a colleague
    reports "events.jsonl exists but cot.json doesn't" — we want to know
    which extractor / Python they were running."""
    out: dict[str, str] = {
        "python": sys.executable or "",
        "platform": sys.platform,
        "cwd": str(Path.cwd()),
        "AGENT_COT_DATA_ROOT": os.environ.get("AGENT_COT_DATA_ROOT", ""),
        "AGENT_COT_PIPELINE_LOG": os.environ.get("AGENT_COT_PIPELINE_LOG", ""),
        "COT_EXTRACTOR_ROOT": os.environ.get("COT_EXTRACTOR_ROOT", ""),
        "COT_PYTHON": os.environ.get("COT_PYTHON", ""),
    }
    return out


def banner(stage: str, **kwargs: Any) -> None:
    """One-shot stamp at the top of a stage start. Emits two lines:

        [...] [agent-cot] === <stage> starting ===
        [...] [agent-cot] env=<pretty env dict>
    """
    log(stage, ide="-", sid="-", event="banner", note="=== starting ===", **kwargs)
    log(stage, ide="-", sid="-", event="env", **stamp_env())


__all__ = ["log", "log_path", "tail", "stamp_env", "banner"]
