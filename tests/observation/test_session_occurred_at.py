from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "src" / "agent_cot" / "assets" / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from services.session_scanner import _session_occurred_at  # noqa: E402


def test_occurred_at_prefers_recent_top_level_extracted_at_over_stale_turn_start() -> None:
    """Bug repro: a still-in-progress latest turn has no turn_end_ms_observed
    and no per-step timestamps (harness doesn't emit them), so the only turn
    signal available is its *start* time. The observation panel must not show
    that stale start time when the file has clearly been touched much more
    recently (explicit user report: 时间显示的是session开始的时间，正确的应该是
    显示最近一次交互会话的时间).
    """
    cot_data = {
        "extracted_at": "2026-07-02T12:44:48.889484+00:00",
        "session_meta": {},
        "turns": [
            {"turn_start_time": "2026-07-02T08:25:16.927000+00:00", "steps": [{"step_type": "user_input"}]},
            {"turn_start_time": "2026-07-02T08:41:19.688000+00:00", "steps": [{"step_type": "tool_execution"}]},
        ],
    }
    assert _session_occurred_at(cot_data) == "2026-07-02T12:44:48.889484+00:00"


def test_occurred_at_prefers_observed_turn_end_when_newer_than_extraction() -> None:
    """When precise per-turn observed timestamps ARE available and are newer
    than the last extraction write, those should still win (they're the more
    precise signal)."""
    cot_data = {
        "extracted_at": "2026-07-02T08:00:00.000000+00:00",
        "turns": [
            {"turn_start_time": "2026-07-02T08:00:00.000000+00:00", "turn_end_ms_observed": 1782982800000},
        ],
    }
    result = _session_occurred_at(cot_data)
    assert result != "2026-07-02T08:00:00.000000+00:00"


def test_occurred_at_falls_back_to_extracted_at_when_no_turn_signal() -> None:
    cot_data = {"extracted_at": "2026-07-02T09:00:00.000000+00:00", "turns": []}
    assert _session_occurred_at(cot_data) == "2026-07-02T09:00:00.000000+00:00"


def test_occurred_at_handles_missing_extracted_at_gracefully() -> None:
    cot_data = {"turns": [{"turn_start_time": "2026-07-02T08:00:00.000000+00:00"}]}
    assert _session_occurred_at(cot_data) == "2026-07-02T08:00:00.000000+00:00"
