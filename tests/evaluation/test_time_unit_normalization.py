"""Regression tests for consistent time-unit reporting.

Ground-truth bug report: efficiency evidence mixed milliseconds ("2917757.0
毫秒") and seconds ("2917.8 秒") within the same report, and the LLM sometimes
cited the raw duration_ms value labeled as milliseconds without converting it.
User requirement: "用时时间单位要统一啊，用min，别一会儿一个毫秒一会儿又是秒".

These tests lock in the deterministic normalization that always overwrites
duration evidence (and scrubs stray ms/seconds mentions from the review prose)
with a canonical minutes-based value, regardless of what the model wrote.
"""

from agent_quality_eval.evaluation.api import (
    _ensure_duration_evidence,
    _ensure_efficiency_evidence,
    _ensure_reliability_evidence,
    _format_minutes,
)
from agent_quality_eval.evaluation.critic import (
    _normalize_duration_evidence,
    _normalize_efficiency_evidence,
    _normalize_reliability_evidence,
)


def test_format_minutes_converts_ms_to_minutes():
    assert _format_minutes(2917757.0) == "48.6 分钟"
    assert _format_minutes(0) is None
    assert _format_minutes(None) is None


def test_ensure_duration_evidence_overwrites_bad_model_output_with_minutes():
    # Model wrote a raw-ms value mislabeled as milliseconds instead of converting.
    dim = {
        "review": "Base 的总耗时为 2917757.0 毫秒，而候选的总耗时为 724142.0 毫秒。",
        "baseline_evidence": [
            {"ref": "metrics:duration_ms", "quote": "总耗时为 2917757.0 毫秒。", "source": "metrics"}
        ],
        "candidate_evidence": [
            {"ref": "metrics:duration_ms", "quote": "总耗时为 724142.0 毫秒。", "source": "metrics"}
        ],
    }
    result = _ensure_duration_evidence(
        dim, {"baseline_evidence": 2917757.0, "candidate_evidence": 724142.0}
    )
    # Exactly one duration_ms evidence item per side, always in minutes.
    for key in ("baseline_evidence", "candidate_evidence"):
        duration_items = [e for e in result[key] if e["ref"].startswith("metrics:duration_ms")]
        assert len(duration_items) == 1
        assert "分钟" in duration_items[0]["quote"]
        assert "毫秒" not in duration_items[0]["quote"]
        assert "秒" not in duration_items[0]["quote"].replace("分钟", "")
    # Review prose is scrubbed of raw ms mentions too.
    assert "毫秒" not in result["review"]
    assert "分钟" in result["review"]


def test_ensure_duration_evidence_scrubs_seconds_from_review():
    dim = {
        "review": "Base 的总耗时为 2917.76 秒，而候选的总耗时为 724.14 秒，显示出更高的效率。",
        "baseline_evidence": [],
        "candidate_evidence": [],
    }
    result = _ensure_duration_evidence(
        dim, {"baseline_evidence": 2917760, "candidate_evidence": 724140}
    )
    assert "秒" not in result["review"]
    assert "分钟" in result["review"]


def test_ensure_duration_evidence_is_idempotent_and_always_minutes():
    dim = {"review": "", "baseline_evidence": [], "candidate_evidence": []}
    once = _ensure_duration_evidence(dim, {"baseline_evidence": 60000, "candidate_evidence": 120000})
    twice = _ensure_duration_evidence(once, {"baseline_evidence": 60000, "candidate_evidence": 120000})
    assert once["baseline_evidence"] == twice["baseline_evidence"]
    assert len(twice["baseline_evidence"]) == 1
    assert "1.0 分钟" in twice["baseline_evidence"][0]["quote"]
    assert "2.0 分钟" in twice["candidate_evidence"][0]["quote"]


def test_critic_normalize_duration_evidence_forces_minutes():
    dim = {
        "verdict": "normal",
        "review": "本轮耗时 4200000 毫秒，资源消耗正常。",
        "evidence": [
            {"ref": "metrics:duration_ms", "quote": "耗时 4200000 毫秒。", "source": "metrics"},
            {"ref": "metrics:total_tokens", "quote": "共消耗 5000 tokens。", "source": "metrics"},
        ],
    }
    result = _normalize_duration_evidence(dim, 4200000)
    duration_items = [e for e in result["evidence"] if e["ref"].startswith("metrics:duration_ms")]
    assert len(duration_items) == 1
    assert "70.0 分钟" in duration_items[0]["quote"]
    assert "毫秒" not in result["review"]
    # Non-duration evidence must be preserved untouched.
    assert any(e["ref"] == "metrics:total_tokens" for e in result["evidence"])


def test_critic_normalize_duration_evidence_noop_without_duration():
    dim = {"review": "no timing info", "evidence": []}
    result = _normalize_duration_evidence(dim, 0)
    assert result == dim


# ---------------------------------------------------------------------------
# Structured-evidence enforcement for efficiency (tokens + duration + tool
# count must always appear as evidence, not just prose). User requirement:
# "强制让llm输出token消耗，耗时，工具调用次数三个维度的内容作为证据展示出来，
# 不要只在自然语言中进行说明".
# ---------------------------------------------------------------------------


def test_critic_normalize_efficiency_evidence_forces_all_three_metrics():
    dim = {"verdict": "normal", "review": "资源消耗正常。", "evidence": []}
    result = _normalize_efficiency_evidence(dim, {"duration_ms": 4200000, "total_tokens": 5000, "tool_count": 12})
    refs = {e["ref"] for e in result["evidence"]}
    assert {"metrics:duration_ms", "metrics:total_tokens", "metrics:tool_count"} <= refs


def test_critic_normalize_efficiency_evidence_does_not_duplicate_existing():
    dim = {
        "verdict": "normal",
        "review": "资源消耗正常。",
        "evidence": [{"ref": "metrics:total_tokens", "quote": "模型已经写好的 token 证据。", "source": "trace"}],
    }
    result = _normalize_efficiency_evidence(dim, {"duration_ms": 60000, "total_tokens": 5000, "tool_count": 3})
    token_items = [e for e in result["evidence"] if e["ref"] == "metrics:total_tokens"]
    assert len(token_items) == 1
    assert token_items[0]["quote"] == "模型已经写好的 token 证据。"


def test_ensure_efficiency_evidence_forces_all_three_metrics_per_side():
    dim = {"verdict": "comparable", "review": "两侧资源消耗相当。", "baseline_evidence": [], "candidate_evidence": []}
    result = _ensure_efficiency_evidence(
        dim,
        {
            "baseline_evidence": {"duration_ms": 60000, "total_tokens": 1000, "tool_count": 5},
            "candidate_evidence": {"duration_ms": 120000, "total_tokens": 2000, "tool_count": 10},
        },
    )
    for key in ("baseline_evidence", "candidate_evidence"):
        refs = {e["ref"] for e in result[key]}
        assert {"metrics:duration_ms", "metrics:total_tokens", "metrics:tool_count"} <= refs


# ---------------------------------------------------------------------------
# Structured-evidence enforcement for reliability's three required aspects:
# failure recovery / edge case handling / state consistency. User requirement:
# "对于可靠性，还应该考虑失败恢复能力，边界情况处理，状态是否一致这三个部分。你
# 需要强制让llm按照模板进行输出...保证内容的完整性".
# ---------------------------------------------------------------------------


def test_critic_normalize_reliability_evidence_backfills_missing_aspects():
    dim = {"verdict": "clear", "review": "过程稳健。", "evidence": []}
    result = _normalize_reliability_evidence(dim, {"unrecovered_failures": 0, "error_recovery_steps": 2})
    refs = {e["ref"] for e in result["evidence"]}
    assert refs == {"reliability:failure_recovery", "reliability:edge_case_handling", "reliability:state_consistency"}


def test_critic_normalize_reliability_evidence_keeps_model_provided_aspects():
    dim = {
        "verdict": "clear",
        "review": "过程稳健。",
        "evidence": [
            {"ref": "reliability:edge_case_handling", "quote": "模型自己写的边界处理证据。", "source": "trace"},
        ],
    }
    result = _normalize_reliability_evidence(dim, {})
    edge_items = [e for e in result["evidence"] if e["ref"] == "reliability:edge_case_handling"]
    assert len(edge_items) == 1
    assert edge_items[0]["quote"] == "模型自己写的边界处理证据。"
    refs = {e["ref"] for e in result["evidence"]}
    assert "reliability:failure_recovery" in refs
    assert "reliability:state_consistency" in refs


def test_ensure_reliability_evidence_backfills_all_three_aspects_per_side():
    dim = {"verdict": "comparable", "review": "两侧过程都很稳健。", "baseline_evidence": [], "candidate_evidence": []}
    result = _ensure_reliability_evidence(
        dim,
        {
            "baseline_evidence": {"unrecovered_failures": 0, "error_recovery_steps": 1},
            "candidate_evidence": {"unrecovered_failures": 2, "error_recovery_steps": 0},
        },
    )
    for key in ("baseline_evidence", "candidate_evidence"):
        refs = {e["ref"] for e in result[key]}
        assert refs >= {"reliability:failure_recovery", "reliability:edge_case_handling", "reliability:state_consistency"}


def test_efficiency_and_reliability_evidence_do_not_cross_dimension_collide():
    """The deterministically-injected metrics:tool_count (efficiency) and
    reliability's failure-count-derived evidence must never read as
    near-identical text, or the cross-dimension dedup detector would flag a
    false-positive collision between two semantically distinct dimensions."""
    import difflib

    efficiency_dim = _normalize_efficiency_evidence(
        {"verdict": "normal", "review": "", "evidence": []},
        {"duration_ms": 60000, "total_tokens": 500, "tool_count": 8},
    )
    reliability_dim = _normalize_reliability_evidence(
        {"verdict": "clear", "review": "", "evidence": []},
        {"unrecovered_failures": 2, "error_recovery_steps": 0},
    )
    tool_count_quote = next(e["quote"] for e in efficiency_dim["evidence"] if e["ref"] == "metrics:tool_count")
    failure_quote = next(e["quote"] for e in reliability_dim["evidence"] if e["ref"] == "reliability:failure_recovery")
    ratio = difflib.SequenceMatcher(None, tool_count_quote, failure_quote).ratio()
    assert ratio < 0.85
