"""Paired A/B comparison for experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .store import DatasetStore


@dataclass
class ComparisonResult:
    baseline_id: str
    candidate_id: str
    win: int = 0
    tie: int = 0
    loss: int = 0
    average_score_delta: float = 0.0
    average_pass_rate_delta: float = 0.0
    average_latency_delta: float = 0.0
    regressions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "candidate_id": self.candidate_id,
            "win": self.win,
            "tie": self.tie,
            "loss": self.loss,
            "average_score_delta": self.average_score_delta,
            "average_pass_rate_delta": self.average_pass_rate_delta,
            "average_latency_delta": self.average_latency_delta,
            "regressions": self.regressions,
        }


def compare_experiment_dicts(baseline: dict[str, Any], candidate: dict[str, Any]) -> ComparisonResult:
    result = ComparisonResult(
        baseline_id=baseline["experiment_id"],
        candidate_id=candidate["experiment_id"],
    )
    pairs = _paired_cases(baseline, candidate)
    score_deltas = []
    pass_deltas = []
    latency_deltas = []
    for case_id, b, c in pairs:
        ds = c["avg_score"] - b["avg_score"]
        dp = c["avg_pass_rate"] - b["avg_pass_rate"]
        dl = c["avg_response_time"] - b["avg_response_time"]
        score_deltas.append(ds)
        pass_deltas.append(dp)
        latency_deltas.append(dl)
        if abs(ds) < 0.01 and abs(dp) < 0.01:
            result.tie += 1
        elif ds >= 0 and dp >= -0.001:
            result.win += 1
        else:
            result.loss += 1
            result.regressions.append(
                {
                    "case_id": case_id,
                    "baseline_provider": b["provider_name"],
                    "candidate_provider": c["provider_name"],
                    "score_delta": ds,
                    "pass_rate_delta": dp,
                    "latency_delta": dl,
                    "question": b.get("question", ""),
                }
            )
    result.average_score_delta = sum(score_deltas) / len(score_deltas) if score_deltas else 0.0
    result.average_pass_rate_delta = sum(pass_deltas) / len(pass_deltas) if pass_deltas else 0.0
    result.average_latency_delta = sum(latency_deltas) / len(latency_deltas) if latency_deltas else 0.0
    return result


def compare_experiments(
    baseline_id: str,
    candidate_id: str,
    *,
    store: DatasetStore | None = None,
) -> ComparisonResult:
    store = store or DatasetStore()
    baseline = store.get_experiment_dict(baseline_id)
    candidate = store.get_experiment_dict(candidate_id)
    return compare_experiment_dicts(baseline, candidate)


def _case_index(exp: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for case in exp.get("case_results", []):
        test = case.get("test_case", {})
        key = (str(test.get("id")), str(case.get("provider_name")))
        out[key] = {
            "case_id": str(test.get("id")),
            "provider_name": str(case.get("provider_name")),
            "avg_score": float(case.get("avg_score", 0)),
            "avg_pass_rate": float(case.get("avg_pass_rate", 0)),
            "avg_response_time": float(case.get("avg_response_time", 0)),
            "question": test.get("question", ""),
        }
    return out


def _case_groups(exp: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in _case_index(exp).values():
        groups.setdefault(item["case_id"], []).append(item)
    return groups


def _paired_cases(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Pair by case/provider first, then by case when each side has one provider."""
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    b_cases = _case_index(baseline)
    c_cases = _case_index(candidate)
    paired_keys: set[tuple[str, str]] = set()

    for key, b in b_cases.items():
        c = c_cases.get(key)
        if c:
            pairs.append((key[0], b, c))
            paired_keys.add(key)

    b_groups = _case_groups(baseline)
    c_groups = _case_groups(candidate)
    for case_id, b_items in b_groups.items():
        c_items = c_groups.get(case_id, [])
        if len(b_items) != 1 or len(c_items) != 1:
            continue
        b = b_items[0]
        c = c_items[0]
        b_key = (b["case_id"], b["provider_name"])
        c_key = (c["case_id"], c["provider_name"])
        if b_key in paired_keys or c_key in paired_keys:
            continue
        pairs.append((case_id, b, c))
        paired_keys.add(b_key)
        paired_keys.add(c_key)

    return pairs
