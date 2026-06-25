import json
from typing import Any

from agent_quality_eval.evaluation.api import _AB_LLM_COMPARE_CACHE, _compare_turn_reports
from agent_quality_eval.evaluation.models import PerformanceMetrics, ProviderResponse


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


def _report(session_id: str, *, passed: bool, score: float, tokens: int = 100, tools: int = 1) -> dict[str, Any]:
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
                "category": "task_outcome",
                "severity": "high",
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
