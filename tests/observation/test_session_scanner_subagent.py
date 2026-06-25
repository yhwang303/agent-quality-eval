"""Tests for v0.20.6 session_scanner sub-agent folding.

Two contracts:

1. ``scan_sessions()`` MUST skip every cot whose sid starts with ``agent-``
   (Claude's sub-agent naming convention). Pre-0.20.6 these polluted the
   SessionList with "memory-related" sessions the user never asked for.

2. ``get_session_cot(main_sid)`` MUST scan disk for matching sub-agent
   cot files whose transcript path includes ``<main_sid>/subagents/``
   and merge their summaries into the returned cot's ``subagent_timeline``
   field. Pre-existing entries in ``subagent_timeline`` (written by the
   extractor at sub-agent stop time) MUST be preserved and de-duplicated
   by ``sub_agent_id``.

If either regresses, the dashboard's left list gets polluted again AND
the right panel's SubStart nodes stop appearing for Task() invocations.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def scanner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Load session_scanner with the bundled backend on sys.path + a tmp data dir."""
    # Point AGENT_COT_DATA_ROOT at tmp so the scanner walks our fixture tree
    # rather than the user's real ~/.agent-cot/data.
    monkeypatch.setenv("AGENT_COT_DATA_ROOT", str(tmp_path))

    # The backend's session_scanner lives under assets/backend/services/. It's
    # not a regular installable subpackage — we have to add the dir to sys.path
    # the same way commands/start.py does at runtime.
    from agent_cot import _assets

    backend_dir = _assets.bundled_backend_dir()
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    # Force re-import in case a previous test polluted module state.
    # session_scanner does ``from config import COT_SCAN_DIRS`` after
    # sys.path.insert — i.e. it imports the bare ``config`` module (NOT
    # ``services.config``), so a stale module key persists across tests
    # unless we clear it explicitly. Drop both forms to be safe.
    for mod in list(sys.modules):
        if mod.startswith("services.") or mod == "services" or mod == "config":
            del sys.modules[mod]
    return importlib.import_module("services.session_scanner")


def _write_cot(root: Path, sid: str, cot: dict) -> Path:
    """Write a cot.json under root/cot/<sid>_cot.json."""
    cot_dir = root / "cot"
    cot_dir.mkdir(parents=True, exist_ok=True)
    f = cot_dir / f"{sid}_cot.json"
    f.write_text(json.dumps(cot), encoding="utf-8")
    return f


def test_scan_sessions_filters_subagent_sids(tmp_path: Path, scanner) -> None:
    """Sessions whose sid starts with 'agent-' are sub-agents — never appear in list."""
    # 1. A real parent session.
    parent_sid = "abc12345-6789-4def-9012-3456789abcde"
    _write_cot(
        tmp_path,
        parent_sid,
        {
            "session_id": parent_sid,
            "transcript_path": str(
                tmp_path / "transcripts" / parent_sid / "main.jsonl"
            ),
            "events": [{"t_ms": 1, "kind": "user", "text": "hi"}],
        },
    )
    # 2. A sub-agent — must be filtered.
    _write_cot(
        tmp_path,
        "agent-ab919deadbeef",
        {
            "session_id": "agent-ab919deadbeef",
            "transcript_path": str(
                tmp_path
                / "transcripts"
                / parent_sid
                / "subagents"
                / "agent-ab919deadbeef.jsonl"
            ),
            "events": [{"t_ms": 1, "kind": "user", "text": "delegated task"}],
        },
    )

    sessions = scanner.scan_sessions()
    sids = [s.get("session_id") for s in sessions]
    assert parent_sid in sids
    assert not any(s.startswith("agent-") for s in sids), (
        "scan_sessions leaked a sub-agent sid into the dashboard list"
    )


def test_get_session_cot_merges_subagent_summary(tmp_path: Path, scanner) -> None:
    """Sub-agent cots are merged into the parent session's subagent_timeline."""
    parent_sid = "fedcba98-7654-3210-fedc-ba9876543210"
    _write_cot(
        tmp_path,
        parent_sid,
        {
            "session_id": parent_sid,
            "transcript_path": str(
                tmp_path / "transcripts" / parent_sid / "main.jsonl"
            ),
            "events": [{"t_ms": 1000, "kind": "user", "text": "use Task(x)"}],
        },
    )
    sub_sid = "agent-cafebabe1234"
    _write_cot(
        tmp_path,
        sub_sid,
        {
            "session_id": sub_sid,
            "agent_type": "claude-task",
            "transcript_path": str(
                tmp_path
                / "transcripts"
                / parent_sid
                / "subagents"
                / f"{sub_sid}.jsonl"
            ),
            # session_scanner._build_subagent_summary extracts from the
            # ``turns`` schema (this is the shape cot-extractor writes).
            "turns": [
                {
                    "user_query": "investigate memory leak",
                    "steps": [
                        {
                            "timestamp": "2026-05-19T01:00:00.000Z",
                            "tool_name": "Read",
                            "otel": {"model": "claude-4.6-sonnet-thinking"},
                        },
                        {
                            "timestamp": "2026-05-19T01:00:01.500Z",
                            "tool_name": "Read",
                        },
                        {
                            "timestamp": "2026-05-19T01:00:02.000Z",
                            "tool_name": "Read",
                        },
                        {
                            "timestamp": "2026-05-19T01:00:02.500Z",
                            "tool_name": "Read",
                        },
                        {
                            "timestamp": "2026-05-19T01:00:03.000Z",
                            "tool_name": "Edit",
                        },
                        {
                            "timestamp": "2026-05-19T01:00:03.200Z",
                            "tool_name": "Edit",
                        },
                        {
                            "timestamp": "2026-05-19T01:00:03.400Z",
                            "tool_name": "Edit",
                        },
                        {
                            "timestamp": "2026-05-19T01:00:03.500Z",
                            "step_type": "final_response",
                            "content": "found it: leak in foo.py",
                        },
                    ],
                },
            ],
            "otel_view": {
                "model": "claude-4.6-sonnet-thinking",
                "totals": {
                    "input_tokens": 5000,
                    "output_tokens": 1500,
                    "cost_usd": 0.04,
                },
                "actual_token_usage": {
                    "input_tokens": 5200,
                    "output_tokens": 1600,
                },
            },
        },
    )

    cot = scanner.get_session_cot(parent_sid)
    assert cot is not None
    timeline = cot.get("subagent_timeline") or []
    assert len(timeline) == 1
    entry = timeline[0]
    assert entry["sub_agent_id"] == sub_sid
    assert entry["agent_type"] == "claude-task"
    assert "investigate memory leak" in entry["prompt_preview"]
    assert entry["model"] == "claude-4.6-sonnet-thinking"
    assert entry["input_tokens"] == 5200  # actual wins over otel totals
    assert entry["output_tokens"] == 1600
    assert entry["cost_usd"] == 0.04
    assert entry["total_steps"] == 8
    assert entry["tool_call_distribution"] == {"Read": 4, "Edit": 3}
    # Marker so a maintainer can grep the cot to see it came from disk merge.
    assert entry["phase"] == "merged_from_disk"


def test_get_session_cot_preserves_extractor_written_subagent_timeline(
    tmp_path: Path, scanner
) -> None:
    """The extractor writes subagent_timeline at sub-agent stop; the disk
    merge must dedupe by sub_agent_id and prefer extractor entries."""
    parent_sid = "11111111-2222-3333-4444-555555555555"
    extractor_entry = {
        "sub_agent_id": "agent-already_written",
        "prompt_preview": "from extractor",
        "phase": "completed_by_extractor",
        "t_ms": 100,
    }
    _write_cot(
        tmp_path,
        parent_sid,
        {
            "session_id": parent_sid,
            "transcript_path": str(
                tmp_path / "transcripts" / parent_sid / "main.jsonl"
            ),
            "subagent_timeline": [extractor_entry],
        },
    )
    # Disk-only sub-agent with SAME sid as one already in timeline — must
    # NOT produce a duplicate.
    _write_cot(
        tmp_path,
        "agent-already_written",
        {
            "session_id": "agent-already_written",
            "transcript_path": str(
                tmp_path
                / "transcripts"
                / parent_sid
                / "subagents"
                / "agent-already_written.jsonl"
            ),
            "events": [{"t_ms": 1, "kind": "user", "text": "should be ignored"}],
        },
    )
    cot = scanner.get_session_cot(parent_sid)
    timeline = cot.get("subagent_timeline") or []
    # Exactly one entry, and it's the extractor-written one (phase pinned).
    assert len(timeline) == 1, timeline
    assert timeline[0]["phase"] == "completed_by_extractor"
