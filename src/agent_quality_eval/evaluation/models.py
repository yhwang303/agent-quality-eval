"""Shared dataclasses for agent evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class PerformanceMetrics:
    total_duration: float = 0.0
    time_to_first_token: float | None = None
    token_usage: dict[str, int | float] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    cost: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_duration": self.total_duration,
            "time_to_first_token": self.time_to_first_token,
            "token_usage": self.token_usage,
            "tool_calls": self.tool_calls,
            "steps": self.steps,
            "cost": self.cost,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PerformanceMetrics":
        if not data:
            return cls()
        return cls(
            total_duration=float(data.get("total_duration") or data.get("response_time") or 0.0),
            time_to_first_token=data.get("time_to_first_token"),
            token_usage=dict(data.get("token_usage") or {}),
            tool_calls=list(data.get("tool_calls") or data.get("tool_call_timings") or []),
            steps=dict(data.get("steps") or data.get("step_timings") or {}),
            cost=data.get("cost"),
        )


@dataclass
class ProviderResponse:
    output: str
    conversation_id: str = ""
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    raw_response: Any = None
    trace: dict[str, Any] | None = None
    error: str | None = None

    @property
    def response_time(self) -> float:
        return self.performance.total_duration

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "conversation_id": self.conversation_id,
            "performance": self.performance.to_dict(),
            "raw_response": self.raw_response,
            "trace": self.trace,
            "error": self.error,
        }


@dataclass
class TestCase:
    id: str
    question: str
    assertions: list[dict[str, Any]] = field(default_factory=list)
    expected_answer: str | None = None
    context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: str = "normal"
    num_trials: int | None = None
    trace: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int = 1) -> "TestCase":
        assertions = data.get("assert", data.get("assertions", [])) or []
        if isinstance(assertions, dict):
            assertions = [assertions]
        return cls(
            id=str(data.get("id") or data.get("name") or index),
            question=str(data.get("question") or data.get("input") or ""),
            assertions=list(assertions),
            expected_answer=data.get("expected_answer", data.get("expected")),
            context=data.get("context"),
            metadata=dict(data.get("metadata") or {}),
            priority=str(data.get("priority") or data.get("metadata", {}).get("priority") or "normal"),
            num_trials=data.get("num_trials"),
            trace=data.get("trace"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "assertions": self.assertions,
            "expected_answer": self.expected_answer,
            "context": self.context,
            "metadata": self.metadata,
            "priority": self.priority,
            "num_trials": self.num_trials,
            "trace": self.trace,
        }


@dataclass
class ScoreResult:
    name: str
    type: str
    score: float
    passed: bool
    reason: str
    threshold: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "score": self.score,
            "passed": self.passed,
            "reason": self.reason,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }


@dataclass
class TrialResult:
    trial_id: str
    trial_number: int
    test_case_id: str
    provider_name: str
    answer: str
    response_time: float
    conversation_id: str = ""
    timestamp: str = field(default_factory=utc_now)
    scores: list[ScoreResult] = field(default_factory=list)
    passed: bool = False
    pass_rate: float = 0.0
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    error: str | None = None
    trace: dict[str, Any] | None = None

    def compute(self, pass_threshold: float) -> None:
        if not self.scores:
            self.pass_rate = 0.0
            self.passed = False
            return
        passed_count = sum(1 for score in self.scores if score.passed)
        self.pass_rate = passed_count / len(self.scores)
        self.passed = self.pass_rate >= pass_threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "trial_number": self.trial_number,
            "test_case_id": self.test_case_id,
            "provider_name": self.provider_name,
            "answer": self.answer,
            "response_time": self.response_time,
            "conversation_id": self.conversation_id,
            "timestamp": self.timestamp,
            "scores": [s.to_dict() for s in self.scores],
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "performance": self.performance.to_dict(),
            "error": self.error,
            "trace": self.trace,
        }


@dataclass
class CaseResult:
    test_case: TestCase
    provider_name: str
    trials: list[TrialResult] = field(default_factory=list)
    pass_at_k: float = 0.0
    pass_power_k: float = 0.0
    avg_pass_rate: float = 0.0
    avg_score: float = 0.0
    avg_response_time: float = 0.0

    def compute(self) -> None:
        if not self.trials:
            return
        passed_trials = sum(1 for trial in self.trials if trial.passed)
        self.pass_at_k = 1.0 if passed_trials else 0.0
        self.pass_power_k = 1.0 if passed_trials == len(self.trials) else 0.0
        self.avg_pass_rate = sum(t.pass_rate for t in self.trials) / len(self.trials)
        self.avg_response_time = sum(t.response_time for t in self.trials) / len(self.trials)
        all_scores = [s.score for t in self.trials for s in t.scores]
        self.avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_case": self.test_case.to_dict(),
            "provider_name": self.provider_name,
            "trials": [t.to_dict() for t in self.trials],
            "pass_at_k": self.pass_at_k,
            "pass_power_k": self.pass_power_k,
            "avg_pass_rate": self.avg_pass_rate,
            "avg_score": self.avg_score,
            "avg_response_time": self.avg_response_time,
        }


@dataclass
class ExperimentResult:
    experiment_id: str
    name: str
    dataset_name: str
    dataset_version: str
    providers: list[str]
    started_at: str = field(default_factory=utc_now)
    ended_at: str | None = None
    case_results: list[CaseResult] = field(default_factory=list)
    status: str = "running"
    overall_pass_rate: float = 0.0
    average_score: float = 0.0
    average_response_time: float = 0.0
    total_trials: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute(self) -> None:
        trials = [trial for case in self.case_results for trial in case.trials]
        self.total_trials = len(trials)
        if trials:
            self.overall_pass_rate = sum(t.pass_rate for t in trials) / len(trials)
            self.average_response_time = sum(t.response_time for t in trials) / len(trials)
            all_scores = [s.score for t in trials for s in t.scores]
            self.average_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        self.ended_at = utc_now()
        self.status = "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "providers": self.providers,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "overall_pass_rate": self.overall_pass_rate,
            "average_score": self.average_score,
            "average_response_time": self.average_response_time,
            "total_trials": self.total_trials,
            "metadata": self.metadata,
            "case_results": [c.to_dict() for c in self.case_results],
        }
