from __future__ import annotations

from agent_quality_eval.evaluation.session_eval import build_turn_eval_report, extract_turn_metrics


def _cot(user_query: str, *, final_response: str = "Done.") -> dict:
    return {
        "session_id": "boundary-test",
        "turns": [
            {
                "turn_index": 1,
                "user_query": user_query,
                "final_response": final_response,
                "steps": [
                    {"step_type": "user_input", "content": user_query, "metadata": {}},
                    {"step_type": "final_response", "content": final_response, "metadata": {}},
                ],
            }
        ],
    }


def test_user_boundary_constraints_are_extracted_into_metrics():
    cot = _cot("必须用 MCP 查看项目进度，不要联网，不要改文件。")
    turn = cot["turns"][0]

    metrics = extract_turn_metrics({"turn": turn, "cot": cot})

    assert metrics["user_boundary_constraint_count"] >= 1
    assert any("MCP" in item["text"] for item in metrics["user_boundary_constraints"])
    assert metrics["boundary_constraint_violation_count"] >= 1
    assert any("MCP" in item["reason"] for item in metrics["boundary_constraint_violations"])


def test_boundary_constraint_assertion_lands_in_instruction_following():
    cot = _cot("必须用 MCP 查看项目进度。")

    report = build_turn_eval_report("boundary-test", 1, cot=cot)
    boundary = next(
        item
        for item in report["assertion_results"]
        if item["key"] == "user-boundary-constraints-followed"
    )

    assert boundary["category"] == "instruction_following"
    assert boundary["passed"] is False
    assert boundary["evidence"]["violations"]
    assert report["eval_panel"]["safety_gate"]["status"] == "fail"
    assert any(
        item["key"] == "user_boundary_constraint_violation" and item["hit"]
        for item in report["eval_panel"]["safety_gate"]["items"]
    )

