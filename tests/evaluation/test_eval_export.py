"""Eval 结果导出：全量、摘要与本体一致、会话隔离。

界面上「Agent 评估维度面板」标题旁那个导出按钮走这条路径。用户明确要求「全部
内容都导出（断言，hook 评审等）」，所以这套测试守的第一条就是**不许丢字段**。
"""

from __future__ import annotations

import json

import pytest

from agent_quality_eval.evaluation.eval_export import (
    EVAL_EXPORT_SCHEMA,
    export_turn_eval,
    sanitize_id,
)


def _report(**overrides):
    """一份形状贴近真实的 turn eval 报告。"""
    report = {
        "report_id": "rpt-1",
        "session_id": "sess-a",
        "turn_index": 3,
        "created_at": "2026-07-20T10:00:00+00:00",
        "passed": False,
        "overall_score": 0.72,
        "assertion_pass_rate": 0.8,
        "eval_mode": "assertion",
        "eval_version": "v3",
        "critical_failures": 1,
        "assertion_results": [
            {"id": "a1", "passed": True, "description": "回答覆盖了用户问题"},
            {"id": "a2", "passed": False, "description": "没有验证改动"},
            {"id": "a3", "passed": True},
        ],
        "assertion_groups": {"correctness": ["a1"], "verification": ["a2"]},
        "eval_panel": {
            "overall_verdict": "fail",
            "core_dimensions": [{"key": "correctness", "score": 0.9}],
            "safety_gate": {"passed": True},
            "diagnostics": ["未跑测试"],
        },
        "judge": {
            "status": "done",
            "source_event": "cursor-bridge:stop",
            "model": "claude-opus-5",
            "provider": "anthropic",
            "reason": "改动没有验证",
            "structured": {
                "overall_verdict": "partial",
                "summary_conclusion": "结论：本轮存在需要复核的风险。",
                "user_request_coverage": "用户要求导出完整评审内容，当前需要确认 LLM 评审已进入 JSON。",
                "instruction_following": {
                    "verdict": "partial",
                    "review": "用户要求全部导出，但旧 JSON 缺少显式 LLM 展示块。",
                    "evidence": [{"ref": "user_query:constraint_1", "quote": "全部内容都要输出到json中"}],
                },
                "reliability": {
                    "verdict": "minor_issues",
                    "review": "主要风险包含导出完整性，而不只是工具调用失败率。",
                    "evidence": [{"ref": "instruction_following:user_boundary_constraint_violation", "quote": "缺少 LLM 评审导出"}],
                },
                "review_markdown": "**结论**：本轮存在需要复核的风险。",
                "claims": [{"claim": "已导出 LLM 评审", "verified": False, "evidence": []}],
            },
            "live_supervisor": {
                "status": "completed",
                "risk_level": "medium",
                "live_summary": "观察到导出缺口。",
                "observations": [{"event": "stop", "message": "Turn boundary reached"}],
            },
            "input_sources": ["turn.raw_object", "raw.transcript_file"],
        },
        "metrics": {"tool_count": 12},
        "lineage": {"gold_binding_hash": "abc"},
    }
    report.update(overrides)
    return report


def _envelope(report):
    return json.loads(export_turn_eval(report)["content"])


# ── 完整性 ──────────────────────────────────────────────────

def test_report_is_carried_verbatim_with_no_field_dropped():
    """报告本体原样带上。少一个字段，下游就少一条可分析的信号。"""
    report = _report()
    envelope = _envelope(report)
    assert envelope["report"] == report


def test_assertion_details_and_hook_review_are_both_present():
    envelope = _envelope(_report())
    assert len(envelope["report"]["assertion_results"]) == 3
    assert envelope["report"]["judge"]["reason"] == "改动没有验证"
    assert envelope["report"]["judge"]["source_event"] == "cursor-bridge:stop"
    assert envelope["report"]["eval_panel"]["safety_gate"] == {"passed": True}


def test_llm_review_is_exported_as_a_display_ready_block():
    envelope = _envelope(_report())
    llm = envelope["llm_review"]
    assert llm["summary_conclusion"].startswith("结论：")
    assert llm["review_markdown"].startswith("**结论**")
    assert llm["dimensions"][0]["key"] == "instruction_following"
    assert llm["dimensions"][0]["evidence"][0]["ref"] == "user_query:constraint_1"
    assert llm["live_supervisor"]["risk_level"] == "medium"
    assert any(block["key"] == "live_supervisor" for block in llm["display_blocks"])


def test_export_is_valid_json_and_declares_its_schema():
    result = export_turn_eval(_report())
    assert result["media_type"].startswith("application/json")
    envelope = json.loads(result["content"])
    assert envelope["schema"] == EVAL_EXPORT_SCHEMA
    assert result["content"].endswith("\n")


# ── 摘要 ────────────────────────────────────────────────────

def test_summary_reflects_the_report_it_summarises():
    """摘要全部是报告里字段的副本，不能出现第三方口径。"""
    report = _report()
    summary = _envelope(report)["summary"]
    assert summary["passed"] is False
    assert summary["overall_verdict"] == "fail"
    assert summary["assertion_pass_rate"] == report["assertion_pass_rate"]
    assert summary["assertion_total"] == 3
    assert summary["judge_source_event"] == "cursor-bridge:stop"


def test_summary_names_the_failed_assertions():
    """打开文件先看到「哪几条炸了」，不用自己翻几十个字段。"""
    assert _envelope(_report())["summary"]["failed_assertions"] == ["a2"]


def test_summary_survives_a_report_missing_optional_blocks():
    """老报告或中断的报告字段不全，导出不能因此炸掉。"""
    envelope = _envelope({"session_id": "sess-a", "turn_index": 1})
    assert envelope["summary"]["overall_verdict"] is None
    assert envelope["summary"]["failed_assertions"] == []
    assert envelope["summary"]["assertion_total"] is None


# ── 会话隔离 ────────────────────────────────────────────────

def test_identity_is_pinned_in_both_envelope_and_filename():
    result = export_turn_eval(_report())
    envelope = json.loads(result["content"])
    assert envelope["session_id"] == "sess-a"
    assert envelope["turn_index"] == 3
    assert result["filename"] == "eval-sess-a-turn3.json"


@pytest.mark.parametrize("raw,expected", [
    ("zhangsan::abc-123", "zhangsan-abc-123"),
    ("../../etc/passwd", "etc-passwd"),
    ("", "session"),
])
def test_session_id_is_sanitised_before_it_reaches_a_filename(raw, expected):
    safe = sanitize_id(raw)
    assert safe == expected
    assert "/" not in safe and "\\" not in safe and ":" not in safe


def test_non_dict_report_is_rejected_loudly():
    with pytest.raises(ValueError, match="无法导出"):
        export_turn_eval(None)  # type: ignore[arg-type]
