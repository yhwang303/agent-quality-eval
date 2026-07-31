from __future__ import annotations

from agent_quality_eval.evaluation.critic import _normalize_instruction_evidence
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
    assert metrics["instruction_obligation_count"] >= metrics["user_boundary_constraint_count"] + 1
    assert metrics["instruction_obligations"][0]["category"] == "primary_request"


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
    group = next(item for item in report["assertion_groups"] if item["key"] == "instruction_following")
    assert group["label"] == "指令遵循"


def test_instruction_following_evidence_includes_behavior_not_only_prompt():
    cot = _cot("必须用 MCP 查看项目进度。", final_response="已查看项目进度并给出结论。")
    metrics = extract_turn_metrics({"turn": cot["turns"][0], "cot": cot})
    metrics["final_response"] = cot["turns"][0]["final_response"]

    dim = _normalize_instruction_evidence(
        {"verdict": "partial", "review": "遵循了用户指令。", "evidence": []},
        metrics,
    )

    refs = [item["ref"] for item in dim["evidence"]]
    assert any(ref.startswith(("final_response:", "assertion:", "metrics:tool_count")) for ref in refs)
    assert not all(ref.startswith("user_query:") for ref in refs)


def test_absent_harness_is_neutral_but_primary_request_is_kept():
    cot = _cot("帮我修复 eval 导出并打一个新 exe。")
    metrics = extract_turn_metrics({"turn": cot["turns"][0], "cot": cot})

    assert metrics["harness_constraint_count"] == 0
    assert metrics["skill_constraint_count"] == 0
    assert metrics["instruction_obligation_count"] >= 1
    assert metrics["instruction_obligations"][0]["category"] == "primary_request"
    assert metrics["instruction_obligation_violation_count"] == 0


def test_do_not_push_is_detected_as_instruction_violation():
    cot = _cot("修复问题，最后不要推送远端。")
    turn = cot["turns"][0]
    turn["steps"].insert(
        1,
        {
            "step_type": "tool_execution",
            "content": "git push origin main",
            "metadata": {"tool_name": "shell_command", "input": "git push origin main"},
        },
    )

    metrics = extract_turn_metrics({"turn": turn, "cot": cot})

    assert metrics["instruction_obligation_violation_count"] >= 1
    assert any("git push" in item["reason"] for item in metrics["instruction_obligation_violations"])


def test_describing_mcp_as_possible_harness_source_is_not_mcp_requirement():
    cot = _cot(
        "除了 SKILL，一般还有哪里会有这种流程的约束？",
        final_response="流程约束可能来自用户 prompt、SKILL.md、AGENTS/rules、system/developer 或工具/MCP 文档。",
    )
    metrics = extract_turn_metrics({"turn": cot["turns"][0], "cot": cot})

    assert not any("MCP" in item.get("reason", "") for item in metrics["instruction_obligation_violations"])


def test_visible_skill_harness_constraints_are_captured():
    cot = _cot("生成一个产品化页面。")
    turn = cot["turns"][0]
    turn["steps"].insert(
        1,
        {
            "step_type": "thinking_inter",
            "content": "SKILL.md 规定：必须先判断用户首次使用路径，并严格避免只输出产品分析。",
            "metadata": {},
        },
    )

    metrics = extract_turn_metrics({"turn": turn, "cot": cot})

    assert metrics["skill_constraint_count"] >= 1
    assert any(item["source"] == "skill" for item in metrics["instruction_obligations"])


def test_visible_skill_workflow_constraints_are_captured_separately():
    cot = _cot("补全 DPAR 资产组缩略图。")
    turn = cot["turns"][0]
    turn["steps"].insert(
        1,
        {
            "step_type": "thinking_inter",
            "content": (
                "SKILL.md\n## 工作流\n"
                "1. 取组上下文 dpar_get_group_analysis_context。\n"
                "2. 查直接引用者，只保留引用结果里的路径。\n"
                "3. 截取缩略图，再上传到组预览。\n"
                "## 约束\n- 禁止 Cursor ue-editor-mcp。"
            ),
            "metadata": {},
        },
    )
    turn["steps"].insert(
        2,
        {
            "step_type": "tool_execution",
            "content": "dpar_get_group_analysis_context('/Game/Foo')",
            "metadata": {"tool_name": "dpar_get_group_analysis_context"},
        },
    )

    metrics = extract_turn_metrics({"turn": turn, "cot": cot})

    assert metrics["workflow_constraint_count"] >= 1
    assert metrics["skill_workflow_constraint_count"] >= 1
    assert any(item["source"] == "skill" for item in metrics["workflow_constraints"])
    assert metrics["workflow_trace_events"]
