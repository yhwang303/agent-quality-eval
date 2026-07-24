"""Tests for skill-creator-inspired enhancements.

Covers four narrow contracts that we picked up from Anthropic's skill-creator
evaluator and grafted onto the three eval lanes (single-trace critic, A/B
compare, regression gate):

* critic._normalize_evidence_list / _normalize_claims — keep frontend payloads
  bounded and well-shaped.
* compare.classify_assertion_pattern — the 4-paradox diagnostic surfaced in
  the A/B view.
* api._compare_turn_reports — assertion_patterns counters propagate through
  ``summary`` so the frontend can paint the strip.
* api._blind_pair_assignment / _unblind_ab_compare — blind-mode round-trip
  preserves baseline/candidate semantics for the user.
* regression.detect_regression — exposes per-metric rows (pass_rate, p95
  latency, avg cost) instead of one opaque delta.
"""

from agent_quality_eval.evaluation import critic
from agent_quality_eval.evaluation.api import (
    _blind_pair_assignment,
    _blind_top_diff,
    _compare_turn_reports,
    _unblind_ab_compare,
)
from agent_quality_eval.evaluation.compare import (
    ASSERTION_PATTERNS,
    classify_assertion_pattern,
)
from agent_quality_eval.evaluation.regression import (
    RegressionPolicy,
    detect_regression,
)


class _StaticStore:
    def __init__(self, baseline, candidate):
        self._exps = {baseline["experiment_id"]: baseline, candidate["experiment_id"]: candidate}

    def get_experiment_dict(self, exp_id):
        return self._exps[exp_id]


def _exp(exp_id, *, pass_rate, latency, cost):
    return {
        "experiment_id": exp_id,
        "overall_pass_rate": pass_rate,
        "case_results": [
            {
                "test_case": {"id": "case-1", "priority": "normal"},
                "avg_pass_rate": pass_rate,
                "trials": [
                    {
                        "response_time": latency,
                        "performance": {"cost": cost},
                        "scores": [],
                    }
                ],
            }
        ],
    }


def test_normalize_evidence_list_accepts_strings_and_dicts():
    rows = critic._normalize_evidence_list(
        [
            "  step:3 wrote tables.csv  ",
            {"ref": "tool_call:7", "quote": "exit_code=0", "source": "tool_result"},
            {"reference": "transcript:42", "text": "user asked for csv"},
            {"foo": "bar"},  # neither ref nor quote — must be dropped
            "",  # blank — must be dropped
        ]
    )
    refs = [r["ref"] for r in rows]
    quotes = [r["quote"] for r in rows]
    assert "tool_call:7" in refs
    assert "transcript:42" in refs
    assert any("step:3 wrote tables.csv" == q for q in quotes)
    assert all(set(r.keys()) == {"ref", "quote", "source"} for r in rows)
    assert len(rows) == 3


def test_normalize_evidence_list_caps_at_six():
    rows = critic._normalize_evidence_list([f"step:{i} note" for i in range(20)])
    assert len(rows) == 6


def test_normalize_claims_filters_and_types_correctly():
    claims = critic._normalize_claims(
        [
            {"claim": "Wrote tables.csv with 27 rows", "type": "factual", "verified": True, "evidence": [{"ref": "step:5", "quote": "27 rows"}]},
            {"claim": "Ran pytest --all", "type": "process", "verified": False},
            {"claim": "Output is concise", "type": "quality", "verified": "unknown"},
            {"claim": "", "type": "factual"},  # blank — drop
            {"type": "factual"},  # no claim text — drop
            {"claim": "Bogus", "type": "weird"},  # unknown type → 'unknown'
        ]
    )
    types = [c["type"] for c in claims]
    verified = [c["verified"] for c in claims]
    assert types == ["factual", "process", "quality", "unknown"]
    assert verified[0] is True
    assert verified[1] is False
    assert verified[2] is None
    assert verified[3] is None
    assert claims[0]["evidence"] and claims[0]["evidence"][0]["ref"] == "step:5"


def test_normalize_claims_drops_final_response_self_evidence():
    claims = critic._normalize_claims(
        [
            {
                "claim": "修复了会导致 Agent 混淆的核心缺陷",
                "type": "quality",
                "verified": True,
                "evidence": [
                    {
                        "ref": "final_response",
                        "quote": "修复了会导致 Agent 混淆的核心缺陷",
                        "source": "FINAL_RESPONSE",
                    }
                ],
            },
            {
                "claim": "输出目录改成 summary/sessions/",
                "type": "factual",
                "verified": True,
                "evidence": [
                    {
                        "ref": "tool_call#1",
                        "quote": "Task #38 created successfully: 输出目录改成 summary/sessions/",
                        "source": "TOOL_RESULT",
                    },
                    {
                        "ref": "final_response",
                        "quote": "输出目录改成 summary/sessions/",
                        "source": "FINAL_RESPONSE",
                    },
                ],
            },
        ]
    )

    assert claims[0]["verified"] is None
    assert claims[0]["evidence"] == []
    assert claims[1]["verified"] is True
    assert [e["ref"] for e in claims[1]["evidence"]] == ["tool_call#1"]


def test_classify_assertion_pattern_covers_four_diagnostics():
    assert classify_assertion_pattern(baseline_passed=True, candidate_passed=True) == ASSERTION_PATTERNS["non_discriminating"]
    assert classify_assertion_pattern(baseline_passed=False, candidate_passed=False) == ASSERTION_PATTERNS["always_failing"]
    assert classify_assertion_pattern(baseline_passed=False, candidate_passed=True) == ASSERTION_PATTERNS["candidate_helps"]
    assert classify_assertion_pattern(baseline_passed=True, candidate_passed=False) == ASSERTION_PATTERNS["candidate_hurts"]
    assert classify_assertion_pattern(baseline_passed=None, candidate_passed=True) == ASSERTION_PATTERNS["mixed"]


def _fake_turn_report(*, session, turn_index, assertions):
    return {
        "session_id": session,
        "turn_index": turn_index,
        "assertion_results": assertions,
        "assertion_pass_rate": sum(1 for a in assertions if a["passed"]) / len(assertions),
        "quality_score": 0.5,
        "metrics": {"total_tokens": 100, "tool_count": 1, "duration_ms": 1000},
        "score_breakdown": {},
        "eval_panel": {},
        "assertion_groups": [],
        "judge": {"status": "completed", "structured": {}},
    }


def test_compare_turn_reports_emits_assertion_pattern_counts(monkeypatch):
    """``_compare_turn_reports`` must aggregate paired diff rows into the four
    skill-creator paradox buckets so the frontend can render the strip."""
    # Force LLM compare to a no-op so the test stays deterministic.
    from agent_quality_eval.evaluation import api as api_mod

    monkeypatch.setattr(api_mod, "_run_ab_llm_compare", lambda *a, **kw: {"status": "disabled"})
    baseline = _fake_turn_report(
        session="s-base",
        turn_index=1,
        assertions=[
            {"key": "a", "passed": True, "score": 1.0, "label_zh": "A"},
            {"key": "b", "passed": True, "score": 1.0, "label_zh": "B"},
            {"key": "c", "passed": False, "score": 0.0, "label_zh": "C"},
            {"key": "d", "passed": True, "score": 1.0, "label_zh": "D"},
        ],
    )
    candidate = _fake_turn_report(
        session="s-cand",
        turn_index=1,
        assertions=[
            {"key": "a", "passed": True, "score": 1.0, "label_zh": "A"},  # non-discrim
            {"key": "b", "passed": False, "score": 0.0, "label_zh": "B"},  # candidate_hurts
            {"key": "c", "passed": True, "score": 1.0, "label_zh": "C"},  # candidate_helps
            {"key": "d", "passed": True, "score": 1.0, "label_zh": "D"},  # non-discrim
        ],
    )
    result = _compare_turn_reports(baseline, candidate, mode="ab")
    counts = result["summary"]["assertion_patterns"]
    assert counts["non_discriminating"] == 2
    assert counts["candidate_helps"] == 1
    assert counts["candidate_hurts"] == 1
    patterns = {row["key"]: row["pattern"] for row in result["diffs"]}
    assert patterns == {"a": "non_discriminating", "b": "candidate_hurts", "c": "candidate_helps", "d": "non_discriminating"}


def test_blind_pair_assignment_is_deterministic_and_round_trips():
    baseline = {"session_id": "s-b", "turn_index": 1}
    candidate = {"session_id": "s-c", "turn_index": 1}
    pair_first = _blind_pair_assignment(baseline, candidate)
    pair_second = _blind_pair_assignment(baseline, candidate)
    assert pair_first == pair_second
    assert set(pair_first) == {"side_a", "side_b"}

    baseline_side, candidate_side = pair_first
    blind_winner = baseline_side  # judge chose baseline-equivalent
    blind_result = {
        "comparison_verdict": "side_a_better" if blind_winner == "side_a" else "side_b_better",
        "task_completion": {"winner": blind_winner},
    }
    unblinded = _unblind_ab_compare(blind_result, baseline_side, candidate_side)
    assert unblinded["comparison_verdict"] == "baseline_better"
    assert unblinded["task_completion"]["winner"] == "baseline"
    assert unblinded["blind_assignment"] == {"baseline": baseline_side, "candidate": candidate_side}


def test_unblind_replaces_side_labels_in_narrative_text():
    """Screenshot regression: the report used to leak "Side A / Side B" into
    the free-text sections (summary, per-dimension review, review_markdown).
    That's meaningless to a user reading a Base vs 候选 report — unblind must
    substitute the identity words with the real labels so no residual
    Side A/B tokens survive to the UI."""
    baseline = {"session_id": "s-b", "turn_index": 1}
    candidate = {"session_id": "s-c", "turn_index": 1}
    baseline_side, candidate_side = _blind_pair_assignment(baseline, candidate)

    blind_result = {
        "comparison_verdict": "side_a_better",
        "summary_conclusion": "结论：Side A 在任务完成度上更好，Side B 存在多次工具失败。",
        "user_request_coverage": "在用户诉求覆盖方面，Side A 完全满足，而 Side B 覆盖不全。",
        "task_completion": {
            "verdict": "stronger",
            "winner": "side_a",
            "review": "Side A 完成了所有功能，Side B 仅部分完成。",
        },
        "tool_use": {
            "verdict": "weaker",
            "winner": "side_b",
            "review": "尽管两侧都存在失败，但 side_a 的失败率更低。",
        },
        "review_markdown": "**任务完成** · stronger\nSide A 完成了所有功能。",
    }

    unblinded = _unblind_ab_compare(blind_result, baseline_side, candidate_side)

    # Determine which real label each side maps to so the assertion works
    # regardless of which side was randomly assigned baseline.
    label_of_a = "Base" if baseline_side == "side_a" else "候选"
    label_of_b = "Base" if baseline_side == "side_b" else "候选"

    for field in ("summary_conclusion", "user_request_coverage"):
        assert "Side A" not in unblinded[field]
        assert "Side B" not in unblinded[field]
        assert "side_a" not in unblinded[field]
        assert "side_b" not in unblinded[field]
        assert label_of_a in unblinded[field]
        assert label_of_b in unblinded[field]

    for dim in ("task_completion", "tool_use"):
        review = unblinded[dim]["review"]
        assert "Side A" not in review and "Side B" not in review
        assert "side_a" not in review and "side_b" not in review

    md = unblinded["review_markdown"]
    assert "Side A" not in md and "Side B" not in md
    assert "side_a" not in md and "side_b" not in md

    # Winners must round-trip to the real labels driven by the deterministic
    # side assignment, not hard-coded expectations.
    expected_task = "baseline" if baseline_side == "side_a" else "candidate"
    expected_tool = "baseline" if baseline_side == "side_b" else "candidate"
    assert unblinded["task_completion"]["winner"] == expected_task
    assert unblinded["tool_use"]["winner"] == expected_tool


def test_blind_top_diff_renames_paired_fields():
    row = {
        "key": "x",
        "baseline_passed": True,
        "candidate_passed": False,
        "baseline_score": 1.0,
        "candidate_score": 0.0,
        "baseline_reason": "ok",
        "candidate_reason": "fail",
        "delta": -1.0,
    }
    out = _blind_top_diff(row, baseline_side="side_b", candidate_side="side_a")
    assert "baseline_passed" not in out
    assert "candidate_passed" not in out
    assert out["side_b_passed"] is True
    assert out["side_a_passed"] is False
    assert out["side_b_score"] == 1.0
    assert out["side_a_score"] == 0.0


def test_detect_regression_exposes_three_independent_metric_rows():
    """Skill-creator-style: the gate must be auditable per metric. We check the
    three rows are all present and each carries baseline/candidate/delta/limit
    so the frontend can paint them independently."""
    baseline = _exp("baseline", pass_rate=0.95, latency=1.0, cost=0.001)
    # candidate: slight pass-rate drop within tolerance, slow latency, equal cost.
    candidate = _exp("candidate", pass_rate=0.94, latency=2.0, cost=0.001)
    store = _StaticStore(baseline, candidate)
    result = detect_regression(
        "baseline",
        "candidate",
        store=store,
        policy=RegressionPolicy(max_pass_rate_drop=0.05, max_p95_latency_increase_ratio=0.5, max_cost_increase_ratio=0.5),
    )
    assert set(result.metrics.keys()) == {"pass_rate", "p95_latency", "avg_cost"}
    pass_row = result.metrics["pass_rate"]
    assert pass_row["baseline"] == 0.95
    assert pass_row["candidate"] == 0.94
    assert pass_row["triggered"] is False  # within tolerance
    latency_row = result.metrics["p95_latency"]
    assert latency_row["triggered"] is True  # 2x latency exceeds 50% ratio
    assert "p95 latency increased" in " ".join(result.reasons)
    cost_row = result.metrics["avg_cost"]
    assert cost_row["triggered"] is False
    assert result.passed is False  # latency triggered the gate
