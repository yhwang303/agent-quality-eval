from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_collector():
    root = Path(__file__).resolve().parents[2]
    path = root / "src" / "agent_cot" / "assets" / "hooks" / "codex" / "codex_sidecar_collector.py"
    spec = importlib.util.spec_from_file_location("codex_sidecar_collector", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_codex_update_plan_becomes_plan_timeline(tmp_path: Path) -> None:
    collector = _load_collector()
    rollout = tmp_path / "rollout-2026-06-15T11-19-36-test-session.jsonl"
    rows = [
        {
            "timestamp": "2026-06-15T03:00:00.000Z",
            "type": "session_meta",
            "payload": {
                "id": "test-session",
                "model": "gpt-5-codex",
                "model_provider": "openai",
            },
        },
        {
            "timestamp": "2026-06-15T03:00:01.000Z",
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "build the thing"},
        },
        {
            "timestamp": "2026-06-15T03:00:02.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "update_plan",
                "call_id": "call_plan_1",
                "arguments": json.dumps(
                    {
                        "plan": [
                            {"step": "Audit inputs", "status": "completed"},
                            {"step": "Patch collector", "status": "in_progress"},
                            {"step": "Run tests", "status": "pending"},
                        ]
                    }
                ),
            },
        },
        {
            "timestamp": "2026-06-15T03:00:03.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "update_plan",
                "call_id": "call_plan_2",
                "arguments": json.dumps(
                    {
                        "plan": [
                            {"step": "Audit inputs", "status": "completed"},
                            {"step": "Patch collector", "status": "completed"},
                            {"step": "Run tests", "status": "completed"},
                        ]
                    }
                ),
            },
        },
    ]
    rollout.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    cot = collector._build_cot(rollout)

    assert cot is not None
    assert cot["tool_call_distribution"]["TodoWrite"] == 2
    assert len(cot["plan_timeline"]) == 2
    first, second = cot["plan_timeline"]
    assert first["completed"] == ["Audit inputs"]
    assert first["in_progress"] == ["Patch collector"]
    assert second["completed"] == ["Audit inputs", "Patch collector", "Run tests"]
    assert second["diff"]["newly_completed"] == [
        {"id": "2", "content": "Patch collector"},
        {"id": "3", "content": "Run tests"},
    ]

    plan_steps = [
        step
        for turn in cot["turns"]
        for step in turn["steps"]
        if step["tool_name"] == "TodoWrite" and step["step_type"] == "tool_decision"
    ]
    assert plan_steps[0]["metadata"]["tool_input"]["todos"][1] == {
        "id": "2",
        "content": "Patch collector",
        "status": "in_progress",
        "idx": 1,
    }
    assert plan_steps[1]["metadata"]["plan_completed_count"] == 3
