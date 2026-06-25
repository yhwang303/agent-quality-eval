"""Backfill ``agentToolResult`` events into existing ``events.jsonl``.

Usage::

    python backfill_results.py [--cot-root PATH]
                               [--session SID | --all]
                               [--dry-run]

For every ``agentToolCall`` event whose tool is in
``tool_reproducer.REPRODUCIBLE_TOOLS`` and that does *not* yet have a paired
``agentToolResult`` (matched on ``payload.tool_use_id`` or, failing that,
``payload.tool_call_idx``), this script:

  1. Re-runs the call locally via ``tool_reproducer.reproduce``.
  2. Appends a fresh ``agentToolResult`` line to ``events.jsonl``.

It only ever appends — never rewrites — so it's safe to interrupt.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from tool_reproducer import reproduce as _reproduce    # noqa: E402

# Mirror of ``transcript_watcher.REPRODUCIBLE_TOOLS`` — duplicated here so
# this script doesn't pull in the entire watcher module (which has its own
# argparse main and signal handlers).
REPRODUCIBLE_TOOLS = frozenset({"Glob", "Grep", "rg", "Delete"})

LOG = logging.getLogger("backfill_results")


def _load_events(path: Path) -> List[Dict]:
    out: List[Dict] = []
    if not path.exists():
        return out
    # utf-8-sig swallows the leading BOM that PowerShell ``Set-Content
    # -Encoding UTF8`` leaves behind when humans (or earlier Python tools
    # using ``open(..., 'w', encoding='utf-8')`` on Windows for the first
    # line) write to events.jsonl. Without this, the very first event
    # silently fails to parse.
    with open(path, "rb") as f:
        first = True
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            text = raw.decode("utf-8", errors="replace")
            if first:
                if text.startswith("\ufeff"):
                    text = text[1:]
                first = False
            try:
                out.append(json.loads(text))
            except Exception:
                continue
    return out


def _existing_result_keys(events: List[Dict]) -> Tuple[set, set]:
    """Index existing agentToolResult events by tool_use_id and tool_call_idx."""
    by_tuid: set = set()
    by_idx: set = set()
    for e in events:
        if e.get("event") != "agentToolResult":
            continue
        payload = e.get("payload") or {}
        if payload.get("tool_use_id"):
            by_tuid.add(payload["tool_use_id"])
        if payload.get("tool_call_idx") is not None:
            by_idx.add(int(payload["tool_call_idx"]))
    return by_tuid, by_idx


def _brief_reproduced_output(result: Dict) -> Dict:
    """Mirror the helper in transcript_watcher.py so result events have the
    same brief_output shape regardless of whether the watcher emitted them
    live or this script backfilled them."""
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


def _truncate(o, max_chars: int = 4096):
    if isinstance(o, str):
        return o if len(o) <= max_chars else o[:max_chars] + f"…[+{len(o)-max_chars}ch]"
    if isinstance(o, list):
        return [_truncate(x, max_chars) for x in o[:200]]
    if isinstance(o, dict):
        return {k: _truncate(v, max_chars) for k, v in o.items()}
    return o


def backfill_session(events_path: Path, *, dry_run: bool = False) -> Dict[str, int]:
    """Append agentToolResult for every unprocessed reproducible call."""
    stats = {"calls_seen": 0, "results_added": 0, "skipped_existing": 0,
             "skipped_unsupported": 0, "errors": 0}
    events = _load_events(events_path)
    if not events:
        return stats
    seen_tuid, seen_idx = _existing_result_keys(events)

    new_lines: List[str] = []
    for e in events:
        if e.get("event") != "agentToolCall":
            continue
        tool = e.get("tool")
        if tool not in REPRODUCIBLE_TOOLS:
            stats["skipped_unsupported"] += 1
            continue
        stats["calls_seen"] += 1
        payload = e.get("payload") or {}
        tuid = payload.get("tool_use_id")
        idx = payload.get("tool_call_idx")
        if tuid and tuid in seen_tuid:
            stats["skipped_existing"] += 1
            continue
        if not tuid and idx is not None and int(idx) in seen_idx:
            stats["skipped_existing"] += 1
            continue
        raw_input = payload.get("input")
        if not isinstance(raw_input, dict):
            stats["errors"] += 1
            LOG.debug("call %s has non-dict input, skipping", tuid or idx)
            continue
        result = _reproduce(tool, raw_input)
        if result is None:
            stats["skipped_unsupported"] += 1
            continue
        evt = {
            "t": int(time.time() * 1000),
            "event": "agentToolResult",
            "cid": e.get("cid"),
            "tool": tool,
            "source": "reproduced",
            "backfilled": True,
            "brief_output": _brief_reproduced_output(result),
            "payload": {
                "tool_call_idx": idx,
                "tool_use_id": tuid,
                "name": tool,
                "result": _truncate(result),
            },
        }
        new_lines.append(json.dumps(evt, ensure_ascii=False))
        stats["results_added"] += 1
        if tuid:
            seen_tuid.add(tuid)
        if idx is not None:
            seen_idx.add(int(idx))

    if new_lines and not dry_run:
        with open(events_path, "ab") as f:
            for ln in new_lines:
                f.write(ln.encode("utf-8") + b"\n")
    return stats


def _iter_session_event_files(cot_root: Path) -> List[Tuple[str, Path]]:
    base = cot_root / "output" / "events"
    if not base.is_dir():
        return []
    out: List[Tuple[str, Path]] = []
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        ef = sub / "events.jsonl"
        if ef.exists():
            out.append((sub.name, ef))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cot-root", type=Path,
                   default=Path(__file__).resolve().parent.parent,
                   help="cot-extractor root holding output/events/ "
                        "(default: %(default)s)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", help="single session id to backfill")
    g.add_argument("--all", action="store_true",
                   help="backfill every session under cot-root")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be added but don't write")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.session:
        ef = args.cot_root / "output" / "events" / args.session / "events.jsonl"
        targets = [(args.session, ef)] if ef.exists() else []
        if not targets:
            LOG.error("no events.jsonl for session %s under %s",
                      args.session, args.cot_root)
            return 1
    else:
        targets = _iter_session_event_files(args.cot_root)
        if not targets:
            LOG.error("no sessions under %s/output/events", args.cot_root)
            return 1

    grand: Dict[str, int] = {"calls_seen": 0, "results_added": 0,
                             "skipped_existing": 0, "skipped_unsupported": 0,
                             "errors": 0}
    for sid, ef in targets:
        st = backfill_session(ef, dry_run=args.dry_run)
        for k, v in st.items():
            grand[k] += v
        LOG.info("session %s: calls=%d, added=%d, existing=%d, "
                 "errors=%d (dry_run=%s)",
                 sid, st["calls_seen"], st["results_added"],
                 st["skipped_existing"], st["errors"], args.dry_run)

    LOG.info("total: %s", grand)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
