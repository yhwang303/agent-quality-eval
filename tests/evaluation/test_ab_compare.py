import json
from typing import Any

from agent_quality_eval.evaluation.api import (
    _AB_LLM_COMPARE_CACHE,
    _REGRESSION_LLM_COMPARE_CACHE,
    _compare_turn_reports,
    _record_compare_eval_event,
)
from agent_quality_eval.evaluation.models import PerformanceMetrics, ProviderResponse
from agent_quality_eval.evaluation.store import DatasetStore


class _Settings:
    def __init__(self, *, enabled: bool = True, configured: bool = True):
        self.enabled = enabled
        self.provider = "timiai"
        self.model = "gpt-4o-mini"
        self.configured = configured

    def to_provider_config(self) -> dict[str, Any] | None:
        if not self.enabled or not self.configured:
            return None
        return {"type": "mock", "model": self.model}


class _Provider:
    def __init__(self, output: str | list[str] = "", error: str | None = None):
        self.outputs = output if isinstance(output, list) else [output]
        self.error = error
        self.calls: list[str] = []

    def call(self, input_text: str, **kwargs: Any) -> ProviderResponse:
        self.calls.append(input_text)
        output = self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]
        return ProviderResponse(
            output=output,
            performance=PerformanceMetrics(token_usage={"total_tokens": 123}),
            error=self.error,
        )


def _report(
    session_id: str,
    *,
    passed: bool,
    score: float,
    tokens: int = 100,
    tools: int = 1,
    severity: str = "high",
    category: str = "task_outcome",
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "turn_index": 0,
        "quality_score": score,
        "assertion_pass_rate": score,
        "created_at": "2026-01-01T00:00:00Z",
        "metrics": {
            "user_query": "fix the A/B report",
            "total_tokens": tokens,
            "input_tokens": tokens // 2,
            "output_tokens": tokens // 2,
            "tool_count": tools,
            "tool_error_count": 0,
            "duration_ms": 1500,
        },
        "assertion_results": [
            {
                "key": "task-complete",
                "label_zh": "任务完成",
                "category": category,
                "severity": severity,
                "passed": passed,
                "score": score,
                "reason": "done" if passed else "missing",
            }
        ],
        "assertion_groups": [{"key": "task_outcome", "label": "任务结果", "passed": 1 if passed else 0, "total": 1}],
        "judge": {
            "status": "completed",
            "provider": "timiai",
            "model": "gpt-4o-mini",
            "source_event": "codex-stream:Stop",
            "structured": {
                "summary_conclusion": "结论：已有单 trace 评审。",
                "task_completion": {"verdict": "resolved", "review": "完成。"},
            },
        },
        "eval_panel": {"overall_verdict": "pass" if passed else "needs_attention"},
    }


def _context(session_id: str) -> dict[str, Any]:
    turn = {
        "turn_index": 0,
        "user_query": "fix the A/B report",
        "final_response": "implemented",
        "duration_ms": 1500,
        "steps": [
            {"step_type": "user_input", "content": "fix the A/B report", "metadata": {}},
            {"step_type": "tool_execution", "content": "ok", "metadata": {"tool_name": "shell", "is_error": False}},
            {"step_type": "final_response", "content": "implemented", "metadata": {}},
        ],
    }
    return {
        "session_id": session_id,
        "turn_index": 0,
        "available": True,
        "cot": {"session_id": session_id, "agent_type": "codex", "turns": [turn]},
        "turn": turn,
        "transcript": {},
        "otel": {},
        "overview": {"agent_type": "codex"},
    }


def _completed_json() -> str:
    data: dict[str, Any] = {
        "comparison_verdict": "candidate_better",
        "summary_conclusion": "结论：候选 trace 在任务完成和可靠性上更好。",
        "user_request_coverage": "候选更完整覆盖用户对 A/B 报告的诉求，Base 只完成了部分断言。",
    }
    for key in ("task_completion", "tool_use", "reasoning", "instruction_following", "faithfulness", "efficiency", "reliability"):
        data[key] = {"verdict": "candidate_stronger", "winner": "candidate", "review": f"候选在 {key} 上更稳定。"}
    return json.dumps(data, ensure_ascii=False)


def test_turn_compare_includes_completed_llm_compare(monkeypatch):
    import agent_quality_eval.evaluation.providers as providers
    import agent_quality_eval.evaluation.settings as settings

    provider = _Provider(_completed_json())
    monkeypatch.setattr(settings, "load_critic_settings", lambda: _Settings())
    monkeypatch.setattr(providers, "load_provider", lambda config: provider)

    result = _compare_turn_reports(
        _report("base", passed=False, score=0.0),
        _report("cand", passed=True, score=1.0, tokens=120, tools=2),
        baseline_context=_context("base"),
        candidate_context=_context("cand"),
    )

    assert result["summary"]["improvement_count"] == 1
    assert result["baseline"]["trace_meta"]["agent_type"] == "codex"
    assert result["baseline"]["trace_meta"]["trace_fingerprint"]
    assert result["baseline"]["trace_meta"]["task_signature"]["request_hash"]
    assert result["baseline"]["trace_meta"]["timeline_signature"]["count"] == 3
    assert result["llm_compare"]["status"] == "completed"
    assert result["llm_compare"]["comparison_verdict"] == "candidate_better"
    assert set(("task_completion", "tool_use", "reasoning", "instruction_following", "faithfulness", "efficiency", "reliability")) <= set(result["llm_compare"])
    assert result["llm_compare"]["cache_hit"] is False
    assert "hook_subagent_eval_report" in provider.calls[0]
    assert "必须突出差异" in provider.calls[0]


def test_turn_compare_reuses_completed_llm_compare_for_same_pair(monkeypatch):
    import agent_quality_eval.evaluation.providers as providers
    import agent_quality_eval.evaluation.settings as settings

    _AB_LLM_COMPARE_CACHE.clear()
    calls = {"count": 0}

    def load_provider(config: dict[str, Any]) -> _Provider:
        calls["count"] += 1
        return _Provider(_completed_json())

    monkeypatch.setattr(settings, "load_critic_settings", lambda: _Settings())
    monkeypatch.setattr(providers, "load_provider", load_provider)

    baseline = _report("base-cache", passed=False, score=0.0)
    candidate = _report("cand-cache", passed=True, score=1.0, tokens=120, tools=2)
    first = _compare_turn_reports(
        baseline,
        candidate,
        baseline_context=_context("base-cache"),
        candidate_context=_context("cand-cache"),
    )
    second = _compare_turn_reports(
        baseline,
        candidate,
        baseline_context=_context("base-cache"),
        candidate_context=_context("cand-cache"),
    )

    assert calls["count"] == 1
    assert first["llm_compare"]["cache_hit"] is False
    assert second["llm_compare"]["cache_hit"] is True

    _compare_turn_reports(
        baseline,
        _report("cand-cache-2", passed=True, score=1.0, tokens=120, tools=2),
        baseline_context=_context("base-cache"),
        candidate_context=_context("cand-cache-2"),
    )
    assert calls["count"] == 2
    _AB_LLM_COMPARE_CACHE.clear()


def test_turn_compare_repairs_missing_completed_dimensions_with_llm(monkeypatch):
    import agent_quality_eval.evaluation.providers as providers
    import agent_quality_eval.evaluation.settings as settings

    _AB_LLM_COMPARE_CACHE.clear()
    partial = {
        "comparison_verdict": "baseline_better",
        "summary_conclusion": "结论：Base 的证据更完整。",
        "user_request_coverage": "Base 与候选请求不完全一致，Base 覆盖更贴近用户诉求。",
        "task_completion": {"verdict": "mixed", "winner": "unclear", "review": "两边都有完成证据，但 Base 更稳。"},
    }
    provider = _Provider([json.dumps(partial, ensure_ascii=False), _completed_json()])
    monkeypatch.setattr(settings, "load_critic_settings", lambda: _Settings())
    monkeypatch.setattr(providers, "load_provider", lambda config: provider)

    result = _compare_turn_reports(
        _report("base-partial", passed=True, score=1.0, tokens=100, tools=1),
        _report("cand-partial", passed=True, score=1.0, tokens=140, tools=3),
        baseline_context=_context("base-partial"),
        candidate_context=_context("cand-partial"),
    )

    assert result["llm_compare"]["status"] == "completed"
    assert result["llm_compare"]["tool_use"]["review"]
    assert "模型没有单独展开" not in result["llm_compare"]["tool_use"]["review"]
    assert result["llm_compare"]["tool_use"]["winner"] == "candidate"
    assert result["llm_compare"]["efficiency"]["review"]
    assert len(provider.calls) == 2
    assert "不完整" in provider.calls[1]
    _AB_LLM_COMPARE_CACHE.clear()


def test_turn_compare_errors_when_llm_repair_still_missing_dimensions(monkeypatch):
    import agent_quality_eval.evaluation.providers as providers
    import agent_quality_eval.evaluation.settings as settings

    _AB_LLM_COMPARE_CACHE.clear()
    partial = {
        "comparison_verdict": "baseline_better",
        "summary_conclusion": "结论：Base 的证据更完整。",
        "user_request_coverage": "Base 与候选请求不完全一致。",
    }
    provider = _Provider(json.dumps(partial, ensure_ascii=False))
    monkeypatch.setattr(settings, "load_critic_settings", lambda: _Settings())
    monkeypatch.setattr(providers, "load_provider", lambda config: provider)

    result = _compare_turn_reports(
        _report("base-partial-error", passed=True, score=1.0),
        _report("cand-partial-error", passed=True, score=1.0),
        baseline_context=_context("base-partial-error"),
        candidate_context=_context("cand-partial-error"),
    )

    assert result["llm_compare"]["status"] == "error"
    assert result["llm_compare"]["missing_fields"]
    assert len(provider.calls) == 2
    _AB_LLM_COMPARE_CACHE.clear()


def test_turn_compare_llm_disabled_does_not_break_numeric_compare(monkeypatch):
    import agent_quality_eval.evaluation.settings as settings

    monkeypatch.setattr(settings, "load_critic_settings", lambda: _Settings(enabled=False))

    result = _compare_turn_reports(_report("base", passed=True, score=1.0), _report("cand", passed=True, score=1.0))

    assert result["llm_compare"]["status"] == "disabled"
    assert result["diffs"]


def test_turn_compare_llm_unconfigured_does_not_break_numeric_compare(monkeypatch):
    import agent_quality_eval.evaluation.settings as settings

    monkeypatch.setattr(settings, "load_critic_settings", lambda: _Settings(configured=False))

    result = _compare_turn_reports(_report("base", passed=True, score=1.0), _report("cand", passed=True, score=1.0))

    assert result["llm_compare"]["status"] == "unconfigured"
    assert result["baseline"]["metrics"]["total_tokens"] == 100


def test_turn_compare_provider_error_does_not_break_numeric_compare(monkeypatch):
    import agent_quality_eval.evaluation.providers as providers
    import agent_quality_eval.evaluation.settings as settings

    monkeypatch.setattr(settings, "load_critic_settings", lambda: _Settings())
    monkeypatch.setattr(providers, "load_provider", lambda config: _Provider(error="boom"))

    result = _compare_turn_reports(
        _report("base", passed=True, score=1.0),
        _report("cand", passed=True, score=1.0),
        baseline_context=_context("base"),
        candidate_context=_context("cand"),
    )

    assert result["llm_compare"]["status"] == "error"
    assert result["summary"]["changed_count"] == 0


def test_turn_compare_invalid_json_and_missing_source_are_non_fatal(monkeypatch):
    import agent_quality_eval.evaluation.providers as providers
    import agent_quality_eval.evaluation.settings as settings

    monkeypatch.setattr(settings, "load_critic_settings", lambda: _Settings())
    monkeypatch.setattr(providers, "load_provider", lambda config: _Provider("not json"))

    invalid = _compare_turn_reports(
        _report("base", passed=True, score=1.0),
        _report("cand", passed=True, score=1.0),
        baseline_context=_context("base"),
        candidate_context=_context("cand"),
    )
    assert invalid["llm_compare"]["status"] == "error"

    missing = _compare_turn_reports(
        _report("base", passed=True, score=1.0),
        _report("cand", passed=True, score=1.0),
        baseline_context={"available": False, "error": "missing cot"},
        candidate_context=_context("cand"),
    )
    assert missing["llm_compare"]["status"] == "error"


def _regression_completed_json(verdict: str = "PASS") -> str:
    data: dict[str, Any] = {
        "gate_verdict": verdict,
        "summary_conclusion": f"回归检测结论：{verdict}。candidate 保持了 baseline 的核心能力。",
        "blocking_reasons": [] if verdict != "FAIL" else ["Candidate broke a baseline capability."],
        "warning_reasons": [] if verdict == "PASS" else ["需要人工复核。"],
        "preserved_capabilities": ["Task completion capability is preserved."],
        "new_regressions": [] if verdict == "PASS" else ["New candidate regression."],
        "manual_review_notes": ["复核断言差异。"],
    }
    for key in (
        "capability_preservation",
        "user_goal_coverage",
        "behavioral_change_risk",
        "evidence_faithfulness",
        "workflow_integrity",
        "efficiency_regression",
    ):
        data[key] = {"verdict": "preserved", "review": f"{key} is supported by the eval evidence."}
    return json.dumps(data, ensure_ascii=False)


def test_regression_mode_uses_regression_judge_not_ab_judge(monkeypatch):
    import agent_quality_eval.evaluation.api as api_module
    import agent_quality_eval.evaluation.providers as providers
    import agent_quality_eval.evaluation.settings as settings

    _REGRESSION_LLM_COMPARE_CACHE.clear()
    provider = _Provider(_regression_completed_json("PASS"))
    monkeypatch.setattr(settings, "load_critic_settings", lambda: _Settings())
    monkeypatch.setattr(providers, "load_provider", lambda config: provider)

    def fail_ab(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("standard A/B judge should not run in regression mode")

    monkeypatch.setattr(api_module, "_run_ab_llm_compare", fail_ab)

    result = _compare_turn_reports(
        _report("base-reg", passed=True, score=1.0),
        _report("cand-reg", passed=True, score=1.0),
        baseline_context=_context("base-reg"),
        candidate_context=_context("cand-reg"),
        mode="regression",
    )

    assert result["compare_mode"] == "regression"
    assert "llm_compare" not in result
    assert result["regression_gate"]["verdict"] == "PASS"
    assert result["regression_compare"]["status"] == "completed"
    assert "Agent 回归检测评审器" in provider.calls[0]
    _REGRESSION_LLM_COMPARE_CACHE.clear()


def test_regression_mode_critical_decline_fails_without_llm(monkeypatch):
    import agent_quality_eval.evaluation.settings as settings

    monkeypatch.setattr(settings, "load_critic_settings", lambda: _Settings(enabled=False))

    result = _compare_turn_reports(
        _report("base-critical", passed=True, score=1.0, severity="critical"),
        _report("cand-critical", passed=False, score=0.0, severity="critical"),
        mode="regression",
    )

    assert result["regression_gate"]["verdict"] == "FAIL"
    assert result["regression_gate"]["critical_regressions"]
    assert result["regression_compare"]["status"] == "disabled"


def test_regression_mode_medium_decline_warns(monkeypatch):
    import agent_quality_eval.evaluation.settings as settings

    monkeypatch.setattr(settings, "load_critic_settings", lambda: _Settings(enabled=False))

    result = _compare_turn_reports(
        _report("base-medium", passed=True, score=1.0, severity="medium", category="workflow_integrity"),
        _report("cand-medium", passed=False, score=0.95, severity="medium", category="workflow_integrity"),
        mode="regression",
    )

    assert result["regression_gate"]["verdict"] == "WARN"
    assert result["regression_gate"]["new_failed_assertions"]


def test_regression_mode_task_outcome_decline_fails(monkeypatch):
    import agent_quality_eval.evaluation.settings as settings

    monkeypatch.setattr(settings, "load_critic_settings", lambda: _Settings(enabled=False))

    result = _compare_turn_reports(
        _report("base-outcome", passed=True, score=1.0, severity="high", category="task_outcome"),
        _report("cand-outcome", passed=False, score=0.8, severity="high", category="task_outcome"),
        mode="regression",
    )

    assert result["regression_gate"]["verdict"] == "FAIL"
    assert result["regression_gate"]["task_outcome_regressions"]


def test_regression_mode_passes_when_assertions_are_preserved(monkeypatch):
    import agent_quality_eval.evaluation.settings as settings

    monkeypatch.setattr(settings, "load_critic_settings", lambda: _Settings(enabled=False))

    result = _compare_turn_reports(
        _report("base-pass", passed=True, score=1.0),
        _report("cand-pass", passed=True, score=1.0),
        mode="regression",
    )

    assert result["regression_gate"]["verdict"] == "PASS"
    assert result["regression_gate"]["preserved_passed_assertions"] == 1
    assert result["regression_compare"]["gate_verdict"] == "PASS"


def test_eval_log_records_ab_conclusion_metadata(tmp_path, monkeypatch):
    import agent_quality_eval.evaluation.providers as providers
    import agent_quality_eval.evaluation.settings as settings

    monkeypatch.setattr(settings, "load_critic_settings", lambda: _Settings())
    monkeypatch.setattr(providers, "load_provider", lambda config: _Provider(_completed_json()))
    result = _compare_turn_reports(
        _report("base-log", passed=False, score=0.0),
        _report("cand-log", passed=True, score=1.0),
        baseline_context=_context("base-log"),
        candidate_context=_context("cand-log"),
    )
    store = DatasetStore(tmp_path / "eval.db")

    _record_compare_eval_event(
        store,
        result,
        baseline_context=_context("base-log"),
        candidate_context=_context("cand-log"),
        reference_answer=None,
    )

    events = store.list_eval_events(event_type="ab")
    assert len(events) == 1
    assert events[0]["winner"] == "candidate"
    assert events[0]["summary"]["comparison_verdict"] == "candidate_better"
    assert events[0]["summary"]["baseline"]["session_id"] == "base-log"
    assert events[0]["summary"]["candidate"]["session_id"] == "cand-log"
    assert "quality_delta" in events[0]["summary"]


def test_eval_log_records_regression_gate_and_gold_metadata(tmp_path, monkeypatch):
    import agent_quality_eval.evaluation.settings as settings

    monkeypatch.setattr(settings, "load_critic_settings", lambda: _Settings(enabled=False))
    reference_answer = {
        "standard_answer": "done",
        "process_requirements": {"required_tools": ["shell"]},
    }
    result = _compare_turn_reports(
        _report("base-reg-log", passed=True, score=1.0, severity="critical"),
        _report("cand-reg-log", passed=False, score=0.0, severity="critical"),
        baseline_context=_context("base-reg-log"),
        candidate_context=_context("cand-reg-log"),
        mode="regression",
        reference_answer=reference_answer,
    )
    store = DatasetStore(tmp_path / "eval.db")

    _record_compare_eval_event(
        store,
        result,
        baseline_context=_context("base-reg-log"),
        candidate_context=_context("cand-reg-log"),
        reference_answer=reference_answer,
    )

    events = store.list_eval_events(event_type="regression", has_gold=True)
    assert len(events) == 1
    assert events[0]["verdict"] == "FAIL"
    assert events[0]["gold_hash"]
    assert events[0]["summary"]["gate_verdict"] == "FAIL"
    assert events[0]["summary"]["has_regression"] is True
    assert events[0]["summary"]["blocking_reasons"]
