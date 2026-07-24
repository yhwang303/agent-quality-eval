from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_extractor():
    root = Path(__file__).resolve().parents[2]
    path = root / "src" / "agent_cot" / "assets" / "cot-extractor" / "src" / "cot_extractor.py"
    spec = importlib.util.spec_from_file_location("cot_extractor", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decision(module, step_index: int, turn_index: int, tool_name: str, tool_input: dict):
    return module.ThoughtStep(
        step_index=step_index,
        turn_index=turn_index,
        step_type=module.StepType.TOOL_DECISION,
        content="",
        tool_name=tool_name,
        metadata={"tool_input": tool_input},
    )


def test_claude_task_plan_display_numbering_resets_per_turn():
    extractor = _load_extractor()
    turns = [
        extractor.TurnCoT(
            turn_index=0,
            user_query="first request",
            steps=[
                _decision(extractor, 1, 0, "TaskCreate", {"subject": "Audit code"}),
                _decision(extractor, 2, 0, "TaskCreate", {"subject": "Patch code"}),
                _decision(extractor, 3, 0, "TaskUpdate", {"taskId": "2", "status": "completed"}),
            ],
        ),
        extractor.TurnCoT(
            turn_index=1,
            user_query="second request",
            steps=[
                _decision(extractor, 4, 1, "TaskCreate", {"subject": "Update README"}),
                _decision(extractor, 5, 1, "TaskUpdate", {"taskId": "3", "status": "completed"}),
            ],
        ),
    ]

    timeline = extractor._build_plan_timeline(turns)

    assert len(timeline) == 5
    second_turn_steps = turns[1].steps
    assert second_turn_steps[0].metadata["plan_snapshot_idx"] == 0
    assert second_turn_steps[0].metadata["plan_total"] == 1
    assert second_turn_steps[0].metadata["plan_full_todos"][0]["id"] == "1"
    assert second_turn_steps[0].metadata["plan_full_todos"][0]["source_id"] == "3"
    assert second_turn_steps[1].metadata["plan_snapshot_idx"] == 1
    assert second_turn_steps[1].metadata["plan_total"] == 1
    assert second_turn_steps[1].metadata["plan_completed_count"] == 1
    assert second_turn_steps[1].metadata["plan_display_task_id"] == "1"
    assert second_turn_steps[1].metadata["plan_source_task_id"] == "3"
