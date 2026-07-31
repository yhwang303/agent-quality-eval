from __future__ import annotations

import json
import sqlite3

import yaml

from agent_quality_eval.cli import bootstrap_workspace
from agent_quality_eval.evaluation.cbm_dataset import build_cbm_real_dataset
from agent_quality_eval.evaluation.compare import compare_experiments
from agent_quality_eval.evaluation.config import load_eval_config, write_default_config
from agent_quality_eval.evaluation.critic import CRITIC_REPORT_SCHEMA_VERSION
from agent_quality_eval.evaluation.runner import run_eval
from agent_quality_eval.evaluation.session_eval import (
    build_session_eval_report,
    build_turn_eval_report,
    _parse_and_validate_structured_turn_judge_response,
    _parse_structured_turn_judge_response,
)
from agent_quality_eval.evaluation.store import DatasetStore


def test_run_eval_writes_outputs_and_store(tmp_path):
    config = write_default_config(tmp_path / "eval.yaml")
    db = tmp_path / "eval.db"

    result = run_eval(config, db_path=str(db))

    assert result.overall_pass_rate >= 0.6
    assert db.exists()
    stored = DatasetStore(db).get_experiment_dict(result.experiment_id)
    assert stored["experiment_id"] == result.experiment_id
    assert (tmp_path / "results" / f"{result.experiment_id}.json").exists()
    assert (tmp_path / "results" / f"{result.experiment_id}.html").exists()


def test_compare_experiments(tmp_path):
    baseline_cfg = write_default_config(tmp_path / "baseline.yaml")
    candidate_cfg = write_default_config(tmp_path / "candidate.yaml")
    baseline_data = yaml.safe_load(baseline_cfg.read_text(encoding="utf-8"))
    baseline_data["providers"][0]["name"] = "agent_without_skill"
    baseline_cfg.write_text(yaml.safe_dump(baseline_data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    candidate_data = yaml.safe_load(candidate_cfg.read_text(encoding="utf-8"))
    candidate_data["providers"][0]["name"] = "agent_with_skill"
    first_question = candidate_data["tests"][0]["question"]
    candidate_data["providers"][0]["responses"][first_question] = "error"
    candidate_cfg.write_text(yaml.safe_dump(candidate_data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    db = tmp_path / "eval.db"

    baseline = run_eval(baseline_cfg, db_path=str(db))
    candidate = run_eval(candidate_cfg, db_path=str(db))

    comparison = compare_experiments(
        baseline.experiment_id,
        candidate.experiment_id,
        store=DatasetStore(db),
    )
    assert comparison.candidate_id == candidate.experiment_id
    assert comparison.loss >= 1

def test_store_promote_baseline(tmp_path):
    config = write_default_config(tmp_path / "eval.yaml")
    db = tmp_path / "eval.db"
    result = run_eval(config, db_path=str(db))
    store = DatasetStore(db)

    store.promote_baseline(result.experiment_id)

    assert store.get_baseline("sample", "v1") == result.experiment_id


def test_bootstrap_workspace_is_idempotent(tmp_path):
    result = bootstrap_workspace(home=tmp_path)
    assert result.sample_created is True
    assert result.db_path.exists()

    result.sample_config.write_text("custom: true\n", encoding="utf-8")
    second = bootstrap_workspace(home=tmp_path)

    assert second.sample_created is False
    assert second.sample_config.read_text(encoding="utf-8") == "custom: true\n"


def test_session_eval_extracts_usage_and_persists(tmp_path):
    report = build_session_eval_report(
        "codex-test",
        cot={
            "session_id": "codex-test",
            "total_tool_calls": 2,
            "otel_view": {
                "actual_token_usage": {
                    "input_tokens": 120,
                    "output_tokens": 45,
                    "cache_read_tokens": 10,
                },
                "actual_cost_usd": 0.0123,
            },
        },
        transcript={"messages": [{"role": "user"}, {"role": "assistant"}]},
        otel={"spans": [{"name": "llm_request", "duration_ms": 250}]},
    )
    assert report["metrics"]["input_tokens"] == 120
    assert report["metrics"]["output_tokens"] == 45
    assert report["metrics"]["total_tokens"] == 165
    assert report["metrics"]["cost_usd"] == 0.0123

    store = DatasetStore(tmp_path / "eval.db")
    report_id = store.save_session_eval(report)

    assert report_id > 0
    assert store.get_latest_session_eval("codex-test")["session_id"] == "codex-test"
    assert store.list_session_evals()[0]["total_tokens"] == 165


def test_turn_eval_reports_tokens_tps_and_persists(tmp_path, monkeypatch):
    from agent_quality_eval.evaluation import settings

    monkeypatch.setattr(settings, "SETTINGS_PATH", tmp_path / "settings.json")
    cot = {
        "session_id": "codex-test",
        "plan_timeline": [
            {
                "turn_index": 0,
                "todos": [
                    {"content": "inspect", "status": "completed"},
                    {"content": "test", "status": "completed"},
                ],
            }
        ],
        "turns": [
            {
                "turn_index": 0,
                "user_query": "请检查项目并运行测试",
                "final_response": "已经检查项目并运行测试，全部通过。",
                "tool_calls": ["shell", "shell"],
                "strategy_shifts": 0,
                "turn_duration_ms_observed": 5000,
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "steps": [
                    {"step_type": "user_input", "content": "请检查项目并运行测试", "metadata": {}},
                    {"step_type": "tool_decision", "content": "run tests", "metadata": {"tool_input": {"command": "pytest"}}},
                    {"step_type": "tool_execution", "content": "passed", "metadata": {"is_error": False}},
                    {"step_type": "final_response", "content": "已经检查项目并运行测试，全部通过。", "metadata": {}},
                ],
            }
        ],
    }

    report = build_turn_eval_report("codex-test", 0, cot=cot)

    assert report["turn_index"] == 0
    assert report["metrics"]["input_tokens"] == 100
    assert report["metrics"]["output_tokens"] == 50
    assert report["metrics"]["total_tokens"] == 150
    assert report["metrics"]["tokens_per_second"] == 10
    assert "cost_usd" not in report["metrics"]
    assert report["eval_version"] == "v3"
    assert report["overall_score"] == report["quality_score"]
    assert report["score_breakdown"] is None
    assert report["eval_panel"]["method"] == "no_weighted_total_v1"
    assert len(report["eval_panel"]["core_dimensions"]) == 6
    assert any(item["key"] == "workflow_adherence" for item in report["eval_panel"]["core_dimensions"])
    assert report["eval_panel"]["diagnostics"]["efficiency"]["tokens"] == 150
    assert report["eval_panel"]["safety_gate"]["status"] in {"pass", "fail"}
    assert report["task_profile"]["primary"] in report["task_profile"]["labels"]
    assert report["assertion_set"]["version"] == "turn-v3.7"
    assert report["assertion_results"]
    categories = {item["category"] for item in report["assertion_results"]}
    assert categories >= {"task_outcome", "execution_integrity", "code_delivery", "tool_use"}
    assert "trace_evidence" not in categories
    assert report["metrics"]["tool_count"] == 2
    assert report["metrics"]["tool_kind_count"] == 1
    assert report["metrics"]["tool_category_counts"]["shell"] == 2
    assert report["metrics"]["tool_name_counts"]["shell"] == 2
    assert "error_breakdown" in report["metrics"]
    assert "mcp_tool_count" in report["metrics"]
    assert report["assertion_groups"]
    assert report["judge"]["status"] == "missing"
    assert "Agent Critic" in report["judge"]["reason"]
    assert "ab_testing" in report["pipeline"]
    assert "regression_gate" not in report["pipeline"]

    store = DatasetStore(tmp_path / "eval.db")
    report_id = store.save_turn_eval(report)

    assert report_id > 0
    assert store.get_latest_turn_eval("codex-test", 0)["report_id"] == report["report_id"]
    assert store.list_turn_evals(session_id="codex-test")[0]["tokens_per_second"] == 10
    assert store.list_turn_evals(session_id="codex-test")[0]["eval_version"] == "v3"
    assert store.list_turn_evals(session_id="codex-test")[0]["assertion_groups"]


def test_codebuddy_successful_tool_results_do_not_count_as_failures():
    from agent_quality_eval.evaluation.session_eval import extract_turn_metrics

    payload = {
        "turn": {
            "turn_index": 1,
            "tool_calls": [
                "mcp_get_tool_description",
                "mcp_get_tool_description",
                "mcp_call_tool",
                "mcp_call_tool",
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "steps": [
                {
                    "step_type": "tool_execution",
                    "tool_name": "mcp_get_tool_description",
                    "content": '{"type":"mcp_call_tool_result","isError":false}',
                    "metadata": {
                        "is_error": False,
                        "raw_result": {
                            "status": "success",
                            "success": True,
                            "result": {"isError": False},
                        },
                    },
                },
                {
                    "step_type": "tool_execution",
                    "tool_name": "mcp_call_tool",
                    "content": '{"type":"mcp_call_tool_result","data":[],"isError":false}',
                    "metadata": {
                        "is_error": False,
                        "raw_result": {
                            "status": "success",
                            "success": True,
                            "result": {"isError": False},
                        },
                    },
                },
            ],
        }
    }

    metrics = extract_turn_metrics(payload)

    assert metrics["tool_count"] == 4
    assert metrics["tool_error_count"] == 0
    assert metrics["tool_error_by_tool"] == {}


def test_turn_eval_ingests_agent_critic_report(tmp_path, monkeypatch):
    from agent_quality_eval.evaluation.critic import critic_report_path

    monkeypatch.setenv("AGENT_COT_DATA_ROOT", str(tmp_path / "agent-cot-data"))
    cot = {
        "session_id": "codex-test",
        "turns": [
            {
                "turn_index": 0,
                "user_query": "please inspect the project",
                "final_response": "I inspected the project and found no blocking issue.",
                "tool_calls": ["shell"],
                "strategy_shifts": 0,
                "usage": {"input_tokens": 10, "output_tokens": 20},
                "steps": [
                    {"step_type": "user_input", "content": "please inspect the project", "metadata": {}},
                    {"step_type": "tool_execution", "content": "ok", "metadata": {"is_error": False, "tool_input": {"command": "pytest"}}},
                    {"step_type": "final_response", "content": "I inspected the project and found no blocking issue.", "metadata": {}},
                ],
            }
        ],
    }
    path = critic_report_path("codex-test", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": CRITIC_REPORT_SCHEMA_VERSION,
                "eval_method": "agent_critic_v1",
                "status": "completed",
                "session_id": "codex-test",
                "turn_index": 0,
                "provider": "timiai",
                "model": "gpt-4o-mini",
                "summary_conclusion": "结论：本轮完成了项目检查并给出明确交付，主要风险较低。",
                "overall_verdict": "resolved",
                "structured": {
                    "summary_conclusion": "结论：本轮完成了项目检查并给出明确交付，主要风险较低。",
                    "overall_verdict": "resolved",
                    "task_completion": {"verdict": "resolved", "review": "覆盖了用户检查诉求。"},
                    "tool_use": {"verdict": "correct", "review": "工具调用与检查目标一致。"},
                    "reasoning": {"verdict": "on_track", "review": "执行路径直接。"},
                    "instruction_following": {"verdict": "yes", "review": "遵循了用户请求。"},
                    "faithfulness": {"verdict": "grounded", "review": "最终回复有工具结果支撑。"},
                    "efficiency": {"verdict": "normal", "review": "资源消耗合理。"},
                    "reliability": {"verdict": "clear", "review": "未见阻断失败。"},
                    "review_markdown": "**结论**：本轮完成了项目检查并给出明确交付，主要风险较低。",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = build_turn_eval_report("codex-test", 0, cot=cot)

    assert report["judge"]["status"] == "completed"
    assert report["judge"]["eval_method"] == "agent_critic_v1"
    assert report["judge"]["structured"]["summary_conclusion"].startswith("结论：")
    assert report["eval_panel"]["core_dimensions"][0]["source"].startswith("Agent Critic")


def _legacy_llm_judge_template_normalizes_to_strict_schema():
    raw = {
        "review": "本轮部分解决了打包诉求，token 消耗适中，但 tool_call#2 失败后仍需要补充安装验证；最终回复与用户问题相关，推理没有明显跑偏。",
        "intent": {
            "user_goal": "验证打包结果并确认新版 exe 可安装",
            "explicit_constraints": ["必须生成 exe", "不能只说结论", "数组超限 1", "数组超限 2", "数组超限 3"],
            "intent_understood": "yes",
            "intent_note": "",
        },
        "tool_actions": {
            "expected_actions": ["运行构建", "检查产物", "启动 smoke test", "返回路径", "多余项"],
            "actual_key_actions": [
                {"step": 1, "action": "shell pyinstaller", "verdict": "on_target"},
                {"step": 2, "action": "tool_call#2 启动 exe", "verdict": "error"},
                {"step": 3, "action": "多余动作", "verdict": "invalid"},
                {"step": 4, "action": "读取日志", "verdict": "on_target"},
                {"step": 5, "action": "超过上限", "verdict": "on_target"},
            ],
            "tool_action_aligned": "no",
            "alignment_note": "用户要求验证安装，但 tool_call#2 启动失败",
        },
        "reasoning": {
            "reasoning_on_track": "yes",
            "drift_evidence": "",
        },
        "outcome": {
            "delivered": "给出了打包产物路径",
            "task_resolved": "yes",
            "missing_or_wrong": "",
        },
        "verdict": "resolved",
        "headline": "打包完成但安装验证失败",
    }

    structured = _parse_structured_turn_judge_response(json.dumps(raw, ensure_ascii=False))

    assert set(structured) == {"review", "intent", "tool_actions", "reasoning", "outcome", "verdict", "headline"}
    assert "dimension_scores" not in structured
    assert "trace_findings" not in structured
    assert "claim_audit" not in structured
    assert "overall" not in structured
    assert "部分解决" not in structured["review"]
    assert "tool_call#2" in structured["review"]
    assert len(structured["intent"]["explicit_constraints"]) == 4
    assert len(structured["tool_actions"]["expected_actions"]) == 4
    assert len(structured["tool_actions"]["actual_key_actions"]) == 4
    assert structured["tool_actions"]["actual_key_actions"][2]["verdict"] == "off_target"
    assert structured["outcome"]["task_resolved"] == "yes"
    assert structured["verdict"] == "unresolved"


def _legacy_llm_judge_template_normalizes_to_strict_schema_v3():
    raw = {
        "metrics": {
            "total_tokens": 150,
            "elapsed_seconds": 5.0,
            "tool_calls_total": 2,
            "tool_calls_failed": 1,
            "efficiency_verdict": "normal",
            "efficiency_note": "150 tokens、5.0s、2次调用，低于5000 token基准。",
        },
        "relevance": {
            "verdict": "aligned",
            "user_goal": "验证打包结果并确认新版exe",
            "evidence": "turn#1 最终回复回应打包验证请求",
        },
        "instruction_following": {
            "verdict": "yes",
            "constraints": ["必须生成exe", "返回产物路径"],
            "violations": "",
        },
        "tool_use": {
            "verdict": "suboptimal",
            "steps": [
                {"step": 1, "expected": "运行打包", "actual": "tool_call#1 执行 pyinstaller", "tag": "match"},
                {"step": 2, "expected": "验证安装", "actual": "tool_call#2 启动失败", "tag": "failed"},
            ],
            "note": "tool_call#2 暴露启动失败，需补验。",
        },
        "reasoning": {
            "verdict": "on_track",
            "evidence": "turn#2 按构建到验证推进",
        },
        "faithfulness": {
            "verdict": "partial",
            "unsupported_claims": ["安装完成缺少 tool_call#2 支撑"],
        },
        "task_completion": {
            "verdict": "partial",
            "delivered": "给出打包产物路径和验证过程",
            "missing": "tool_call#2 失败后缺少安装成功证据",
        },
        "overall_verdict": "partial",
        "headline": "打包完成但安装验证不足",
        "review_markdown": (
            "**结论**：打包完成但安装验证不足\n\n"
            "**效率** · normal\n"
            "- 150 tokens / 5.0s / 2 次工具调用（失败 1）\n"
            "- 150 tokens、5.0s、2次调用，低于5000 token基准。\n\n"
            "**相关性** · aligned\n"
            "- 用户目标：验证打包结果并确认新版exe\n"
            "- turn#1 最终回复回应打包验证请求\n\n"
            "**指令遵循** · yes\n"
            "- 硬约束：必须生成exe，返回产物路径\n"
            "- 所有硬约束已满足\n\n"
            "**工具使用** · suboptimal\n"
            "- tool_call#2 暴露启动失败，需补验。\n"
            "- 偏离步骤：step 2 引用 tool_call#2\n\n"
            "**推理路径** · on_track\n"
            "- turn#2 按构建到验证推进\n\n"
            "**忠实度** · partial\n"
            "- 安装完成缺少 tool_call#2 支撑\n\n"
            "**任务完成** · partial\n"
            "- 交付：给出打包产物路径和验证过程\n"
            "- 缺口：tool_call#2 失败后缺少安装成功证据"
        ),
    }

    structured, errors = _parse_and_validate_structured_turn_judge_response(
        json.dumps(raw, ensure_ascii=False),
        expected_runtime_metrics={
            "total_tokens": 150,
            "elapsed_seconds": 5.0,
            "tool_calls_total": 2,
            "tool_calls_failed": 1,
        },
    )

    assert errors == []
    assert set(structured) == {
        "metrics",
        "relevance",
        "instruction_following",
        "tool_use",
        "reasoning",
        "faithfulness",
        "task_completion",
        "overall_verdict",
        "headline",
        "review_markdown",
    }
    assert structured["overall_verdict"] == "partial"
    assert structured["tool_use"]["steps"][1]["tag"] == "failed"
    assert "tool_call#2" in structured["review_markdown"]
    assert "部分解决" not in structured["review_markdown"]

    parsed = _parse_structured_turn_judge_response(json.dumps(raw, ensure_ascii=False))
    assert parsed["headline"] == "打包完成但安装验证不足"


def _legacy_llm_judge_v3_rejects_template_drift():
    raw = {
        "metrics": {
            "total_tokens": 150,
            "elapsed_seconds": 5.0,
            "tool_calls_total": 2,
            "tool_calls_failed": 1,
            "efficiency_verdict": "normal",
            "efficiency_note": "150 tokens、5.0s、2次调用，低于5000 token基准。",
        },
        "relevance": {"verdict": "aligned", "user_goal": "验证打包", "evidence": "turn#1 回应目标"},
        "instruction_following": {"verdict": "yes", "constraints": [], "violations": ""},
        "tool_use": {"verdict": "correct", "steps": [], "note": "tool_call#1 已验证。"},
        "reasoning": {"verdict": "on_track", "evidence": "turn#2 按目标推进"},
        "faithfulness": {"verdict": "grounded", "unsupported_claims": []},
        "task_completion": {"verdict": "resolved", "delivered": "交付exe", "missing": ""},
        "overall_verdict": "partial",
        "headline": "基本完成打包验证",
        "review_markdown": "基本完成打包验证",
    }

    structured, errors = _parse_and_validate_structured_turn_judge_response(json.dumps(raw, ensure_ascii=False))

    assert structured["overall_verdict"] == "resolved"
    assert any("overall_verdict" in error for error in errors)
    assert any("禁用短语" in error for error in errors)
    assert any("review_markdown" in error for error in errors)


def _legacy_llm_judge_template_normalizes_to_strict_schema_v31():
    raw = {
        "summary": "本轮围绕新版 exe 打包与安装验证展开，最终回复给出了产物路径和验证过程，tool_call#1 支撑了构建动作，tool_call#2 暴露启动失败且未形成安装成功证据，因此交付仍有影响验收的缺口。",
        "metrics": {
            "total_tokens": 150,
            "elapsed_seconds": 5.0,
            "tool_calls_total": 2,
            "tool_calls_failed": 1,
            "efficiency_verdict": "normal",
            "efficiency_note": "本轮 150 tokens、5.0s、2 次工具调用，低于中等打包验证任务常见 5000 tokens 与 30s 范围，效率本身未影响最终交付。",
        },
        "relevance": {
            "verdict": "aligned",
            "user_goal": "验证打包结果并确认新版 exe 可安装",
            "final_response_addresses": "turn#1 最终回复回应了用户要求，说明已生成新版 exe，并给出构建产物路径、验证过程和后续安装验收依据，能够对应用户要看到可安装结果的核心诉求。",
            "evidence": "",
        },
        "instruction_following": {
            "verdict": "yes",
            "constraints": [
                {"constraint": "必须生成 exe", "satisfied": "yes", "evidence": "tool_call#1 执行打包并产生产物"},
                {"constraint": "返回产物路径", "satisfied": "yes", "evidence": "turn#1 最终回复包含 exe 路径"},
            ],
            "note": "",
        },
        "tool_use": {
            "verdict": "suboptimal",
            "expected_actions": ["运行打包", "验证安装启动"],
            "actual_key_actions": [
                {"step": 1, "tool_call_ref": "tool_call#1", "action": "PyInstaller 打包 exe", "tag": "match"},
                {"step": 2, "tool_call_ref": "tool_call#2", "action": "启动 exe 验证安装", "tag": "impactful_failure"},
            ],
            "note": "实际工具链覆盖了打包与启动验证，tool_call#1 与预期匹配并提供了构建证据；tool_call#2 的失败没有被后续成功安装证据恢复，因此它不只是路径波动，而是影响了用户验收新版 exe 是否可安装的最终判断。",
        },
        "reasoning": {
            "verdict": "on_track",
            "trajectory_summary": "turn#1 到 turn#2 的执行轨迹从打包推进到启动验证，方向与用户验收新版 exe 的目标一致。",
            "key_moments": [
                {"turn_ref": "turn#1", "observation": "构建产物路径被记录", "impact": "none"},
                {"turn_ref": "turn#2", "observation": "安装启动验证失败", "impact": "major"},
            ],
            "note": "",
        },
        "faithfulness": {
            "verdict": "partial",
            "grounded_examples": ["exe 产物路径由 tool_call#1 支撑"],
            "unsupported_claims": ["安装成功缺少 tool_call#2 支撑"],
            "note": "构建相关说法有工具证据，安装成功类结论没有成功启动结果支撑，因此忠实度只能判为 partial。",
        },
        "task_completion": {
            "verdict": "partial",
            "delivered": "用户实际收到了新版 exe 的构建产物路径、打包过程说明和一次启动验证结果；这些内容可以支持继续定位安装问题，也能证明打包动作已经执行，并为下一步重新安装、修复启动失败、复测安装向导和确认覆盖更新行为提供了依据。",
            "missing": "缺少安装或启动成功证据，直接影响用户确认新版 exe 可安装的验收目标。",
        },
        "overall_verdict": "partial",
    }

    structured, errors = _parse_and_validate_structured_turn_judge_response(
        json.dumps(raw, ensure_ascii=False),
        expected_runtime_metrics={
            "total_tokens": 150,
            "elapsed_seconds": 5.0,
            "tool_calls_total": 2,
            "tool_calls_failed": 1,
        },
    )

    assert errors == []
    assert "review_markdown" in structured
    assert "150 tokens / 5.0s / 2 次工具调用（失败 1）" in structured["review_markdown"]
    assert structured["overall_verdict"] == "partial"
    assert structured["tool_use"]["actual_key_actions"][1]["tag"] == "impactful_failure"
    assert "headline" not in structured


def _legacy_llm_judge_v31_rejects_template_drift_and_uses_real_fallback_metrics():
    structured, errors = _parse_and_validate_structured_turn_judge_response(
        "not json",
        expected_runtime_metrics={
            "total_tokens": 777,
            "elapsed_seconds": 12.5,
            "tool_calls_total": 9,
            "tool_calls_failed": 2,
        },
    )

    assert errors
    assert structured["metrics"]["total_tokens"] == 777
    assert structured["metrics"]["elapsed_seconds"] == 12.5
    assert structured["metrics"]["tool_calls_total"] == 9
    assert structured["metrics"]["tool_calls_failed"] == 2
    assert "777 tokens / 12.5s / 9 次工具调用（失败 2）" in structured["review_markdown"]


def test_llm_judge_template_normalizes_to_loose_v32_schema():
    raw = {
        "summary": "本轮围绕新版 exe 打包与安装验证展开，agent 调用了打包和启动验证相关工具，最终回复给出了产物路径、验证过程和后续风险。整体交付与用户目标部分对齐，但启动失败没有被成功恢复，因此仍会影响用户安装验收。",
        "efficiency": {
            "verdict": "normal",
            "review": "本轮 runtime_metrics 显示 150 tokens、5.0s、2 次工具调用且失败 1 次；以打包验证任务复杂度看，资源消耗不高，失败属于关键验收步骤的波动，而不是 token 或耗时层面的浪费。",
        },
        "relevance": {
            "verdict": "aligned",
            "review": "用户目标是确认新版 exe 是否完成并可用于安装验收，最终回复围绕产物路径、构建过程和启动验证展开，能够回应核心诉求；其中 turn#1 的交付说明与用户要看到结果的要求一致。",
        },
        "instruction_following": {
            "verdict": "yes",
            "review": "用户硬约束是更新后需要打包 exe 并提供可验收结果。最终回复说明了 exe 产物和验证情况，没有只给抽象结论；虽然安装成功证据不足，但这属于任务完成缺口，不是忽略显式格式或步骤要求。",
        },
        "tool_use": {
            "verdict": "suboptimal",
            "review": "预期工具调用应覆盖构建和安装验证，实际 tool_call#1 支撑打包动作，tool_call#2 进入启动验证但失败。该失败没有被后续成功结果恢复，因此工具链路不是 wrong，但确实留下影响最终验收的缺口。",
        },
        "reasoning": {
            "verdict": "on_track",
            "review": "推理轨迹从生成产物推进到验证启动，方向与用户验收新版 exe 的目标一致。中间没有明显长期绕路；主要问题不是思路迷失，而是关键验证步骤失败后缺少进一步恢复或替代验证。",
        },
        "faithfulness": {
            "verdict": "partial",
            "review": "最终回复中关于打包产物的说法有 tool_call#1 支撑，但如果声称安装已经完全可用，则缺少 tool_call#2 成功结果支撑。因此忠实度不是 hallucinated，但对安装成功的表达需要更克制。",
        },
        "task_completion": {
            "verdict": "partial",
            "review": "用户实际获得了新版 exe 的构建路径、打包说明和一次启动验证结果，这些内容能支持继续排查；但用户最关心的安装验收仍缺少成功证据，启动失败未恢复，所以任务只能算部分完成。",
        },
        "overall_verdict": "resolved",
    }

    structured, errors = _parse_and_validate_structured_turn_judge_response(json.dumps(raw, ensure_ascii=False))

    assert errors == []
    assert structured["overall_verdict"] == "partial"
    assert structured["efficiency"]["verdict"] == "normal"
    assert "150 tokens、5.0s、2 次工具调用" in structured["review_markdown"]
    assert "- " not in structured["review_markdown"]
    assert "**工具使用** · suboptimal" in structured["review_markdown"]


def test_llm_judge_v32_parse_failure_falls_back_without_error_status_shape():
    structured, errors = _parse_and_validate_structured_turn_judge_response(
        "not json",
        expected_runtime_metrics={
            "total_tokens": 777,
            "elapsed_seconds": 12.5,
            "tool_calls_total": 9,
            "tool_calls_failed": 2,
        },
    )

    assert errors
    assert structured["overall_verdict"] == "partial"
    assert "777 tokens" in structured["efficiency"]["review"]
    assert "9 次工具调用" in structured["review_markdown"]
    assert "error" not in structured


def test_agent_critic_review_markdown_is_canonical_standard_dimensions():
    from agent_quality_eval.evaluation.critic import _normalize_structured

    structured = _normalize_structured(
        {
            "summary_conclusion": "结论：本轮可用但需要复核。",
            "overall_verdict": "partial",
            "task_completion": {"verdict": "partial", "review": "交付有缺口。"},
            "tool_use": {"verdict": "suboptimal", "review": "工具证据不足。"},
            "review_markdown": (
                "**结论**：本轮可用但需要复核。\n\n"
                "### 任务完成度\n"
                "- 完成了主要动作。\n\n"
                "### 最终判定\n"
                "- overall_verdict: partial\n"
                "- 是否足以验收：否"
            ),
        },
        {},
    )

    assert "**用户诉求覆盖情况**" in structured["review_markdown"]
    assert "**任务完成** · partial" in structured["review_markdown"]
    assert "任务完成度" not in structured["review_markdown"]
    assert "最终判定" not in structured["review_markdown"]
    assert "overall_verdict: partial" not in structured["review_markdown"]


def test_agent_critic_reuses_completed_hook_report_for_manual_display():
    from agent_quality_eval.evaluation.critic import should_reuse_completed_hook_report

    assert should_reuse_completed_hook_report(
        {"status": "completed", "source_event": "codebuddy-stream:Stop"}
    )
    assert not should_reuse_completed_hook_report(
        {"status": "completed", "source_event": "api-rerun"}
    )
    assert not should_reuse_completed_hook_report(
        {"status": "running", "source_event": "codebuddy-stream:Stop"}
    )


def test_legacy_turn_eval_cache_is_not_returned(tmp_path):
    store = DatasetStore(tmp_path / "eval.db")
    store.save_turn_eval(
        {
            "report_id": "legacy-turn-eval",
            "session_id": "codex-test",
            "turn_index": 0,
            "created_at": "2026-01-01T00:00:00Z",
            "passed": True,
            "overall_score": 0.9,
            "quality_score": 0.9,
            "metrics": {"tokens_per_second": 1.0},
            "scores": [{"key": "task_completion", "score": 0.9}],
        }
    )
    store.save_turn_eval(
        {
            "report_id": "old-v3-turn-eval",
            "session_id": "codex-test",
            "turn_index": 0,
            "created_at": "2026-01-02T00:00:00Z",
            "eval_version": "v3",
            "passed": True,
            "overall_score": 0.9,
            "quality_score": 0.9,
            "assertion_set": {"version": "turn-v3.3"},
            "assertion_results": [{"key": "optional", "passed": True}],
            "metrics": {"tokens_per_second": 1.0},
            "scores": [],
        }
    )

    assert store.get_latest_turn_eval("codex-test", 0) is None
    assert store.list_turn_evals(session_id="codex-test") == []


def test_build_cbm_real_dataset_from_temp_sqlite_redacts_and_loads(tmp_path):
    db_path = tmp_path / "codebuddy-mem.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE session_summaries (
                id INTEGER PRIMARY KEY,
                memory_session_id TEXT,
                project TEXT,
                request TEXT,
                investigated TEXT,
                completed TEXT,
                next_steps TEXT,
                notes TEXT,
                files_read TEXT,
                files_edited TEXT,
                created_at TEXT,
                created_at_epoch INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE observations (
                memory_session_id TEXT,
                title TEXT,
                subtitle TEXT,
                text TEXT,
                facts TEXT,
                evidence TEXT,
                type TEXT,
                created_at_epoch INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO session_summaries VALUES (
                1,
                'real-session-123',
                'D:/work/agent-quality-eval',
                'Fix a regression where the eval panel reports a subjective score and leaks C:\\Users\\milkwang\\repo\\secret.py plus user@example.com.',
                'Investigated trace evidence, assertion pass rate, A/B testing, A/B comparison gate, and token=sk-testsecret1234567890.',
                'Completed a v3 assertion-based fix, added deterministic no-pii and no-error checks, and ran pytest plus frontend build validation.',
                'Next step is to monitor regression risk and avoid claiming validation without captured evidence.',
                'Installer artifact CBMem-Setup-1.2.3.exe and local DB codebuddy-mem.db should be redacted.',
                'C:\\Users\\milkwang\\repo\\secret.py',
                'C:\\Users\\milkwang\\repo\\eval.py',
                '2026-06-18T00:00:00Z',
                1
            )
            """
        )
        conn.execute(
            """
            INSERT INTO observations VALUES (
                'real-session-123',
                'Bugfix validation',
                'Eval trace',
                'The bugfix was verified with pytest and npm run build; email user@example.com and C:\\Users\\milkwang\\repo\\private.env must not appear.',
                'A A/B comparison gate blocks critical failures.',
                'Evidence includes tests, assertion groups, and A/B baseline comparison.',
                'bugfix',
                1
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    yaml_path = tmp_path / "cbm_real_cases.yaml"
    evidence_path = tmp_path / "cbm_real_cases_evidence.jsonl"
    result = build_cbm_real_dataset(db_path, yaml_path=yaml_path, evidence_path=evidence_path, max_cases=1)

    assert result["cases"] == 1
    config = load_eval_config(yaml_path)
    assert len(config.tests) == 1
    assert config.tests[0].metadata["source_project"] == "agent-quality-eval"
    assert config.tests[0].assertions[-1]["type"] == "llm-rubric"
    assert config.tests[0].assertions[-1]["optional"] is True

    yaml_text = yaml_path.read_text(encoding="utf-8")
    evidence_text = evidence_path.read_text(encoding="utf-8")
    combined = yaml_text + "\n" + evidence_text
    assert "C:\\Users" not in combined
    assert "milkwang" not in combined
    assert "user@example.com" not in combined
    assert "sk-testsecret1234567890" not in combined
    assert "CBMem-Setup-1.2.3.exe" not in combined
    assert "codebuddy-mem.db" not in combined

    evidence = json.loads(evidence_text)
    assert evidence["memory_session_id"].startswith("session-")
    assert evidence["source_table"] == "session_summaries"
    assert evidence["source_row_id"] == 1
