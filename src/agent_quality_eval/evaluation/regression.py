"""Baseline regression detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import quantiles
from typing import Any

from .compare import compare_experiment_dicts
from .store import DatasetStore


@dataclass
class RegressionPolicy:
    max_pass_rate_drop: float = 0.03
    max_p95_latency_increase_ratio: float = 0.20
    max_cost_increase_ratio: float = 0.20
    block_on_high_priority_failure: bool = True
    block_on_safety_failure: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RegressionPolicy":
        data = data or {}
        return cls(
            max_pass_rate_drop=float(data.get("max_pass_rate_drop", 0.03)),
            max_p95_latency_increase_ratio=float(data.get("max_p95_latency_increase_ratio", 0.20)),
            max_cost_increase_ratio=float(data.get("max_cost_increase_ratio", 0.20)),
            block_on_high_priority_failure=bool(data.get("block_on_high_priority_failure", True)),
            block_on_safety_failure=bool(data.get("block_on_safety_failure", True)),
        )


@dataclass
class RegressionResult:
    passed: bool
    baseline_id: str
    candidate_id: str
    reasons: list[str] = field(default_factory=list)
    comparison: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "baseline_id": self.baseline_id,
            "candidate_id": self.candidate_id,
            "reasons": self.reasons,
            "comparison": self.comparison,
            "metrics": self.metrics,
        }


def _metric_check(
    *,
    name: str,
    baseline: float,
    candidate: float,
    limit: float,
    direction: str,  # 'drop' (smaller candidate is bad) or 'increase' (bigger candidate is bad)
    ratio_basis: str = "absolute",  # 'absolute' or 'ratio'
) -> dict[str, Any]:
    """Compute one regression sub-metric and report it as a structured row.

    The point of pulling this out is so the frontend / CI can show three
    independent rows ("pass rate", "p95 latency", "tokens") instead of one
    opaque overall delta. Each row tells you what was measured, what the
    baseline / candidate values were, what threshold the policy uses, and
    whether this metric on its own would block.
    """
    delta = candidate - baseline
    if direction == "drop":
        breach_value = baseline - candidate
        unit_threshold = limit
        triggered = baseline > 0 and breach_value > limit if ratio_basis == "ratio" else breach_value > limit
        if ratio_basis == "ratio" and baseline > 0:
            triggered = breach_value / baseline > limit
    else:
        if ratio_basis == "ratio" and baseline > 0:
            triggered = candidate > baseline * (1 + limit)
        else:
            triggered = candidate > baseline + limit
        unit_threshold = limit
    return {
        "name": name,
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "direction": direction,
        "limit": unit_threshold,
        "ratio_basis": ratio_basis,
        "triggered": bool(triggered),
    }


def detect_regression(
    baseline_id: str,
    candidate_id: str,
    *,
    policy: RegressionPolicy | None = None,
    store: DatasetStore | None = None,
) -> RegressionResult:
    store = store or DatasetStore()
    policy = policy or RegressionPolicy()
    baseline = store.get_experiment_dict(baseline_id)
    candidate = store.get_experiment_dict(candidate_id)
    reasons: list[str] = []

    base_pass_rate = float(baseline.get("overall_pass_rate", 0))
    cand_pass_rate = float(candidate.get("overall_pass_rate", 0))
    pass_drop = base_pass_rate - cand_pass_rate
    pass_metric = _metric_check(
        name="pass_rate",
        baseline=base_pass_rate,
        candidate=cand_pass_rate,
        limit=policy.max_pass_rate_drop,
        direction="drop",
    )
    if pass_metric["triggered"]:
        reasons.append(
            f"overall pass rate dropped by {pass_drop:.3f}, limit {policy.max_pass_rate_drop:.3f}"
        )

    b_p95 = _p95_latency(baseline)
    c_p95 = _p95_latency(candidate)
    latency_metric = _metric_check(
        name="p95_latency_seconds",
        baseline=b_p95,
        candidate=c_p95,
        limit=policy.max_p95_latency_increase_ratio,
        direction="increase",
        ratio_basis="ratio",
    )
    if latency_metric["triggered"]:
        reasons.append(f"p95 latency increased from {b_p95:.2f}s to {c_p95:.2f}s")

    base_cost = _avg_cost(baseline)
    cand_cost = _avg_cost(candidate)
    cost_metric = _metric_check(
        name="avg_cost",
        baseline=base_cost,
        candidate=cand_cost,
        limit=policy.max_cost_increase_ratio,
        direction="increase",
        ratio_basis="ratio",
    )
    if cost_metric["triggered"]:
        reasons.append(
            f"average cost increased from {base_cost:.4f} to {cand_cost:.4f}"
        )

    if policy.block_on_high_priority_failure:
        failures = _new_high_priority_failures(baseline, candidate)
        if failures:
            reasons.append(f"new high-priority failures: {', '.join(failures)}")

    if policy.block_on_safety_failure:
        safety = _safety_failures(candidate)
        if safety:
            reasons.append(f"safety/PII failures: {', '.join(safety)}")

    comparison = compare_experiment_dicts(baseline, candidate).to_dict()
    metrics = {
        "pass_rate": pass_metric,
        "p95_latency": latency_metric,
        "avg_cost": cost_metric,
    }
    return RegressionResult(
        passed=not reasons,
        baseline_id=baseline_id,
        candidate_id=candidate_id,
        reasons=reasons,
        comparison=comparison,
        metrics=metrics,
    )


def _avg_cost(exp: dict[str, Any]) -> float:
    """Average per-trial cost surfaced by deterministic scorers, defaulting to 0.

    Some experiments don't carry cost (free local models). Returning 0 in that
    case keeps the cost row visible in the regression report without ever
    triggering, which is the right behavior for a free-tier baseline.
    """
    costs: list[float] = []
    for case in exp.get("case_results", []):
        for trial in case.get("trials", []):
            performance = trial.get("performance") or {}
            cost = performance.get("cost")
            try:
                cost_f = float(cost) if cost is not None else 0.0
            except (TypeError, ValueError):
                cost_f = 0.0
            costs.append(cost_f)
    if not costs:
        return 0.0
    return sum(costs) / len(costs)


def _trial_latencies(exp: dict[str, Any]) -> list[float]:
    return [
        float(trial.get("response_time", 0))
        for case in exp.get("case_results", [])
        for trial in case.get("trials", [])
    ]


def _p95_latency(exp: dict[str, Any]) -> float:
    values = sorted(_trial_latencies(exp))
    if not values:
        return 0.0
    if len(values) < 20:
        return values[-1]
    return quantiles(values, n=20)[-1]


def _case_status(exp: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for case in exp.get("case_results", []):
        test = case.get("test_case", {})
        out[str(test.get("id"))] = {
            "passed": float(case.get("avg_pass_rate", 0)) >= 0.999,
            "priority": str(test.get("priority", "normal")).lower(),
        }
    return out


def _new_high_priority_failures(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    b = _case_status(baseline)
    c = _case_status(candidate)
    failures = []
    for case_id, item in c.items():
        if item["priority"] in {"high", "critical", "p0", "p1"} and not item["passed"]:
            if b.get(case_id, {}).get("passed", True):
                failures.append(case_id)
    return failures


def _safety_failures(exp: dict[str, Any]) -> list[str]:
    failures = []
    safety_names = {"no-pii", "pii-leakage", "safety", "misuse", "role-violation"}
    for case in exp.get("case_results", []):
        case_id = str(case.get("test_case", {}).get("id"))
        for trial in case.get("trials", []):
            for score in trial.get("scores", []):
                if str(score.get("name")).lower() in safety_names and not score.get("passed", False):
                    failures.append(case_id)
                    break
    return sorted(set(failures))