"""Regression tests for cross-dimension duplicate detection/repair.

Ground-truth bug report: a real regression compare showed capability_preservation,
behavioral_change_risk and workflow_integrity all citing the exact same
``metrics:tool_calls_failed`` evidence with near-identical review text — the
"three dimensions all just talk about tool failures" complaint. These tests
lock in the code-level safety net (independent of how well the LLM follows
prompt instructions):

* ``_find_cross_dimension_duplicates`` — detects shared refs / near-identical
  quotes / near-identical review narratives across dimensions in one report.
* ``_resolve_dimension_duplicates`` — deterministically neutralizes the loser
  of a colliding pair instead of shipping duplicated content.
* ``_deterministic_capability_preservation`` — capability_preservation is
  grounded in code-computed assertion diffs, never the LLM's own words.
* End-to-end via ``_compare_turn_reports(mode="regression")`` with a mocked
  provider that reproduces the exact broken output from the bug report.
"""

import json
from typing import Any

from agent_quality_eval.evaluation.api import (
    AB_COMPARE_DIMENSION_FALLBACK_NOTES,
    AB_COMPARE_DIMENSIONS,
    AB_COMPARE_REF_WHITELIST,
    REGRESSION_DIMENSION_FALLBACK_NOTES,
    REGRESSION_DIMENSIONS,
    REGRESSION_REF_WHITELIST,
    _REGRESSION_LLM_COMPARE_CACHE,
    _compare_turn_reports,
    _deterministic_capability_preservation,
    _find_cross_dimension_duplicates,
    _regression_dimension,
    _resolve_dimension_duplicates,
)
from agent_quality_eval.evaluation.models import PerformanceMetrics, ProviderResponse


def _dup_metric_dim(review: str, count: str) -> dict[str, Any]:
    return {
        "verdict": "degraded",
        "review": review,
        "baseline_evidence": [{"ref": "metrics:tool_calls_failed", "quote": "工具调用失败次数为 0。", "source": "metrics"}],
        "candidate_evidence": [{"ref": "metrics:tool_calls_failed", "quote": f"工具调用失败次数为 {count} 次。", "source": "metrics"}],
    }


def _bug_report_regression_result() -> dict[str, Any]:
    """Reproduces the exact duplication from the user's screenshot: capability
    preservation / behavioral change risk / workflow integrity all cite the
    same tool_calls_failed metric with near-identical wording."""
    return {
        "gate_verdict": "WARN",
        "summary_conclusion": "回归检测结论：WARN。",
        "capability_preservation": _dup_metric_dim(
            "在能力保持方面，candidate 相比 baseline 存在明显退化，工具调用失败次数增加。", "2"
        ),
        "user_goal_coverage": {
            "verdict": "preserved",
            "review": "用户目标仍被覆盖。",
            "baseline_evidence": [{"ref": "user_query:主诉求", "quote": "了解项目进度", "source": "transcript"}],
            "candidate_evidence": [{"ref": "final_response", "quote": "已梳理项目进度", "source": "transcript"}],
        },
        "behavioral_change_risk": _dup_metric_dim(
            "在行为变化风险方面，candidate 的工具调用失败增加可能导致未来交互出现更多问题。", "2"
        ),
        "evidence_faithfulness": {
            "verdict": "preserved",
            "review": "candidate 的最终回复内容与工具调用结果相符。",
            "baseline_evidence": [{"ref": "claim:未提出独立声称", "quote": "无独立声称", "source": "final_response"}],
            "candidate_evidence": [{"ref": "claim:未提出独立声称", "quote": "无独立声称", "source": "final_response"}],
        },
        "workflow_integrity": _dup_metric_dim(
            "在工作流完整性方面，candidate 的工具调用失败导致整体工作流的完整性下降。", "2"
        ),
        "efficiency_regression": {
            "verdict": "warning",
            "review": "candidate 的 token 用量高于 baseline。",
            "baseline_evidence": [{"ref": "metrics:total_tokens", "quote": "总 token 使用量为 11407595。", "source": "metrics"}],
            "candidate_evidence": [{"ref": "metrics:total_tokens", "quote": "总计使用了 1755350 个 tokens。", "source": "metrics"}],
        },
    }


def test_find_cross_dimension_duplicates_flags_shared_ref_and_similar_review():
    result = _bug_report_regression_result()
    collisions = _find_cross_dimension_duplicates(
        result, REGRESSION_DIMENSIONS, ("baseline_evidence", "candidate_evidence")
    )
    pairs = {frozenset((c["dim_a"], c["dim_b"])) for c in collisions}
    assert frozenset({"capability_preservation", "behavioral_change_risk"}) in pairs
    assert frozenset({"capability_preservation", "workflow_integrity"}) in pairs
    assert frozenset({"behavioral_change_risk", "workflow_integrity"}) in pairs
    # Dimensions that never shared evidence must not be flagged.
    assert frozenset({"user_goal_coverage", "evidence_faithfulness"}) not in pairs


def test_find_cross_dimension_duplicates_clean_report_has_no_collisions():
    result = _bug_report_regression_result()
    # workflow_integrity legitimately owns metrics:tool_error_count-style
    # evidence; give the other two dimensions their own real signals instead.
    result["capability_preservation"] = _regression_dimension(
        verdict="preserved",
        review="能力保持方面，candidate 仍通过 baseline 已通过的全部 70 个断言。",
        baseline_evidence=[{"ref": "assertion:preserved_all", "quote": "baseline 通过 70 项断言。", "source": "assertion"}],
        candidate_evidence=[{"ref": "assertion:preserved_all", "quote": "candidate 同样保持通过这 70 项断言。", "source": "assertion"}],
    )
    result["behavioral_change_risk"] = _regression_dimension(
        verdict="low",
        review="行为变化风险方面，candidate 与 baseline 的策略转换次数相近，未发现明显策略变化。",
        baseline_evidence=[{"ref": "metrics:strategy_shifts", "quote": "baseline 策略转换 1 次。", "source": "metrics"}],
        candidate_evidence=[{"ref": "metrics:strategy_shifts", "quote": "candidate 策略转换 1 次。", "source": "metrics"}],
    )
    collisions = _find_cross_dimension_duplicates(
        result, REGRESSION_DIMENSIONS, ("baseline_evidence", "candidate_evidence")
    )
    assert collisions == []


def test_resolve_dimension_duplicates_keeps_whitelist_owner_and_neutralizes_others():
    result = _bug_report_regression_result()
    collisions = _find_cross_dimension_duplicates(
        result, REGRESSION_DIMENSIONS, ("baseline_evidence", "candidate_evidence")
    )
    resolved = _resolve_dimension_duplicates(
        result,
        collisions,
        ("baseline_evidence", "candidate_evidence"),
        REGRESSION_REF_WHITELIST,
        REGRESSION_DIMENSIONS,
        fallback_notes=REGRESSION_DIMENSION_FALLBACK_NOTES,
    )
    # workflow_integrity's evidence (metrics:tool_error_count-style) matches its
    # own whitelist better than the other two, so it should survive untouched.
    assert resolved["workflow_integrity"]["candidate_evidence"][0]["ref"] == "metrics:tool_calls_failed"
    # No two dimensions may still share a ref or near-identical review text.
    leftover = _find_cross_dimension_duplicates(
        resolved, REGRESSION_DIMENSIONS, ("baseline_evidence", "candidate_evidence")
    )
    assert leftover == []
    # Neutralized dimensions must be honest about having no independent evidence
    # (using their own distinct wording), not just silently blank or copying
    # another dimension's placeholder.
    for key in ("capability_preservation", "behavioral_change_risk"):
        if resolved[key]["review"] != _bug_report_regression_result()[key]["review"]:
            assert resolved[key]["review"] == REGRESSION_DIMENSION_FALLBACK_NOTES[key]
            ref = resolved[key]["candidate_evidence"][0]["ref"]
            assert ref.startswith("metrics:no_signal#")
            # Refs render as compact <code> chips in a narrow evidence column;
            # a long ref (e.g. the full dimension name) overflows the UI.
            assert len(ref) <= 24, f"ref too long, will overflow evidence chip: {ref!r}"


def test_resolve_dimension_duplicates_ab_uses_comparable_verdict_and_tie_winner():
    result = {
        "task_completion": {
            "verdict": "stronger", "winner": "candidate", "review": "候选完成了新版 exe 的交付验证工作。",
            "baseline_evidence": [{"ref": "final_response", "quote": "交付了 exe", "source": "transcript"}],
            "candidate_evidence": [{"ref": "final_response", "quote": "交付了 exe", "source": "transcript"}],
        },
        "faithfulness": {
            "verdict": "stronger", "winner": "candidate", "review": "候选完成了新版 exe 的交付验证工作。",
            "baseline_evidence": [{"ref": "final_response", "quote": "交付了 exe", "source": "transcript"}],
            "candidate_evidence": [{"ref": "final_response", "quote": "交付了 exe", "source": "transcript"}],
        },
    }
    partial_dims = ("task_completion", "faithfulness")
    collisions = _find_cross_dimension_duplicates(result, partial_dims, ("baseline_evidence", "candidate_evidence"))
    assert collisions
    resolved = _resolve_dimension_duplicates(
        result, collisions, ("baseline_evidence", "candidate_evidence"), AB_COMPARE_REF_WHITELIST, AB_COMPARE_DIMENSIONS,
        neutral_verdict="comparable", fallback_notes=AB_COMPARE_DIMENSION_FALLBACK_NOTES,
    )
    loser = "faithfulness" if resolved["faithfulness"]["review"] != result["faithfulness"]["review"] else "task_completion"
    assert resolved[loser]["verdict"] == "comparable"
    assert resolved[loser]["winner"] == "tie"


def _gate(*, regressed: bool) -> dict[str, Any]:
    if regressed:
        return {
            "verdict": "WARN",
            "new_failed_assertions": [{"key": "tool-safety", "label_zh": "工具安全断言"}],
            "missing_assertion_coverage": [],
            "preserved_passed_assertions": 69,
            "baseline_passed_assertions": 70,
        }
    return {
        "verdict": "PASS",
        "new_failed_assertions": [],
        "missing_assertion_coverage": [],
        "preserved_passed_assertions": 70,
        "baseline_passed_assertions": 70,
    }


def test_deterministic_capability_preservation_overrides_metric_based_llm_output():
    llm_dim = _dup_metric_dim("candidate 相比 baseline 存在能力退化。", "2")
    result = _deterministic_capability_preservation(_gate(regressed=False), llm_dim)
    # The LLM's evidence used a metrics: ref, not assertion:, so it must be
    # replaced wholesale with the deterministic, assertion-grounded answer.
    assert result["verdict"] == "preserved"
    assert result["baseline_evidence"][0]["ref"].startswith("assertion:")
    assert result["candidate_evidence"][0]["ref"].startswith("assertion:")


def test_deterministic_capability_preservation_flags_real_assertion_regression():
    llm_dim = {"verdict": "unclear", "review": "", "baseline_evidence": [], "candidate_evidence": []}
    result = _deterministic_capability_preservation(_gate(regressed=True), llm_dim)
    assert result["verdict"] == "degraded"
    assert "工具安全断言" in result["review"]
    assert result["baseline_evidence"][0]["ref"] == "assertion:tool-safety"


def test_deterministic_capability_preservation_trusts_llm_when_already_assertion_grounded():
    llm_dim = _regression_dimension(
        verdict="preserved",
        review="candidate 仍通过所有断言。",
        baseline_evidence=[{"ref": "assertion:preserved_all", "quote": "baseline 通过 70 项。", "source": "assertion"}],
        candidate_evidence=[{"ref": "assertion:preserved_all", "quote": "candidate 通过 70 项。", "source": "assertion"}],
    )
    result = _deterministic_capability_preservation(_gate(regressed=False), llm_dim)
    assert result is llm_dim


# ---------------------------------------------------------------------------
# End-to-end: mocked provider reproduces the exact bug report output.
# ---------------------------------------------------------------------------


class _Settings:
    provider = "timiai"
    model = "gpt-4o-mini"
    enabled = True

    def to_provider_config(self) -> dict[str, Any] | None:
        return {"type": "mock", "model": self.model}


class _SequentialProvider:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.calls: list[str] = []

    def call(self, input_text: str, **kwargs: Any) -> ProviderResponse:
        self.calls.append(input_text)
        idx = min(len(self.calls) - 1, len(self.outputs) - 1)
        return ProviderResponse(output=self.outputs[idx], performance=PerformanceMetrics(token_usage={"total_tokens": 10}), error=None)


def _turn_report(session_id: str, *, passed: bool, tool_errors: int) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "turn_index": 0,
        "quality_score": 1.0 if passed else 0.9,
        "assertion_pass_rate": 1.0 if passed else 0.9,
        "created_at": "2026-01-01T00:00:00Z",
        "metrics": {
            "user_query": "你用MCP看一下项目进度",
            "total_tokens": 1755350,
            "input_tokens": 800000,
            "output_tokens": 955350,
            "tool_count": 20,
            "tool_error_count": tool_errors,
            "duration_ms": 45000,
            "step_count": 40,
            "strategy_shifts": 1,
            "plan_update_count": 1,
            "repeated_tool_calls": 2,
            "error_recovery_steps": max(0, tool_errors - 1),
            "unrecovered_failures": 1 if tool_errors else 0,
            "thinking_steps": 5,
        },
        "assertion_results": [
            {"key": "task-complete", "label_zh": "任务完成", "category": "task_outcome", "severity": "high", "passed": True, "score": 1.0, "reason": "done"}
        ],
        "assertion_groups": [{"key": "task_outcome", "label": "任务结果", "passed": 1, "total": 1}],
        "judge": {"status": "completed", "provider": "timiai", "model": "gpt-4o-mini", "source_event": "cursor-stream:Stop", "structured": {}},
        "eval_panel": {"overall_verdict": "pass"},
    }


def _turn_context(session_id: str) -> dict[str, Any]:
    turn = {
        "turn_index": 0,
        "user_query": "你用MCP看一下项目进度",
        "final_response": "我已经通过 MCP 记忆库和自接查资代码，完整梳理了项目从早期到 0.1.66 的演进过程。",
        "duration_ms": 45000,
        "steps": [{"step_type": "user_input", "content": "你用MCP看一下项目进度", "metadata": {}}],
    }
    return {
        "session_id": session_id, "turn_index": 0, "available": True,
        "cot": {"session_id": session_id, "agent_type": "cursor", "turns": [turn]},
        "turn": turn, "transcript": {}, "otel": {}, "overview": {"agent_type": "cursor"},
    }


def test_regression_end_to_end_repairs_duplicated_dimensions(monkeypatch):
    import agent_quality_eval.evaluation.providers as providers
    import agent_quality_eval.evaluation.settings as settings

    _REGRESSION_LLM_COMPARE_CACHE.clear()
    broken = json.dumps(_bug_report_regression_result(), ensure_ascii=False)
    fixed = json.dumps(
        {
            **_bug_report_regression_result(),
            "capability_preservation": {
                "verdict": "preserved", "review": "candidate 仍通过 baseline 已通过的全部断言。",
                "baseline_evidence": [{"ref": "assertion:preserved_all", "quote": "baseline 通过全部断言。", "source": "assertion"}],
                "candidate_evidence": [{"ref": "assertion:preserved_all", "quote": "candidate 通过全部断言。", "source": "assertion"}],
            },
            "behavioral_change_risk": {
                "verdict": "low", "review": "策略转换次数两侧相近，未见明显行为变化。",
                "baseline_evidence": [{"ref": "metrics:strategy_shifts", "quote": "baseline 策略转换 1 次。", "source": "metrics"}],
                "candidate_evidence": [{"ref": "metrics:strategy_shifts", "quote": "candidate 策略转换 1 次。", "source": "metrics"}],
            },
        },
        ensure_ascii=False,
    )
    provider = _SequentialProvider([broken, fixed])
    monkeypatch.setattr(settings, "load_critic_settings", lambda: _Settings())
    monkeypatch.setattr(providers, "load_provider", lambda config: provider)

    result = _compare_turn_reports(
        _turn_report("base-e2e", passed=True, tool_errors=0),
        _turn_report("cand-e2e", passed=True, tool_errors=2),
        baseline_context=_turn_context("base-e2e"),
        candidate_context=_turn_context("cand-e2e"),
        mode="regression",
    )

    compare = result["regression_compare"]
    assert compare["status"] == "completed"
    # The repair round should have been triggered (2 provider calls: initial + dedup repair).
    assert len(provider.calls) == 2

    for key in REGRESSION_DIMENSIONS:
        assert compare[key]["review"].strip(), f"{key} must not be empty"
    leftover = _find_cross_dimension_duplicates(compare, REGRESSION_DIMENSIONS, ("baseline_evidence", "candidate_evidence"))
    assert leftover == [], f"dimensions still duplicated after repair: {leftover}"
    # capability_preservation must always be assertion-grounded, never metrics-based.
    assert compare["capability_preservation"]["baseline_evidence"][0]["ref"].startswith("assertion:")
    _REGRESSION_LLM_COMPARE_CACHE.clear()


def test_regression_end_to_end_falls_back_when_repair_still_duplicated(monkeypatch):
    import agent_quality_eval.evaluation.providers as providers
    import agent_quality_eval.evaluation.settings as settings

    _REGRESSION_LLM_COMPARE_CACHE.clear()
    broken = json.dumps(_bug_report_regression_result(), ensure_ascii=False)
    # The model repeats the same mistake even after a repair prompt.
    provider = _SequentialProvider([broken, broken])
    monkeypatch.setattr(settings, "load_critic_settings", lambda: _Settings())
    monkeypatch.setattr(providers, "load_provider", lambda config: provider)

    result = _compare_turn_reports(
        _turn_report("base-fallback", passed=True, tool_errors=0),
        _turn_report("cand-fallback", passed=True, tool_errors=2),
        baseline_context=_turn_context("base-fallback"),
        candidate_context=_turn_context("cand-fallback"),
        mode="regression",
    )

    compare = result["regression_compare"]
    assert len(provider.calls) == 2
    leftover = _find_cross_dimension_duplicates(compare, REGRESSION_DIMENSIONS, ("baseline_evidence", "candidate_evidence"))
    assert leftover == [], f"deterministic fallback must eliminate all duplicates, got: {leftover}"
    # Even in the worst case, capability_preservation is deterministically grounded.
    assert compare["capability_preservation"]["baseline_evidence"][0]["ref"].startswith("assertion:")
    assert compare["capability_preservation"]["verdict"] == "preserved"
    _REGRESSION_LLM_COMPARE_CACHE.clear()
