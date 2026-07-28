"""单轮 eval 结果导出。

界面上「Agent 评估维度面板」标题旁那个导出按钮走这里。

**为什么是 JSON 而不是 trace 那边用的 JSONL**：eval 报告是一个嵌套文档
（断言结果、维度打分、safety gate、judge provenance 互相引用），不是一条按
时间推进的事件流。硬塞成 JSONL 只会得到「一行装完整篇报告」这种没有意义的
文件；而 trace 是流，一行一个事件才有价值。两边各用各自领域里最标准的形态。

**完整性**：报告字典原样放进 ``report``，一个字段都不删。断言明细
（``assertion_results``）、hook 阶段评审（``judge``）、维度面板
（``eval_panel``）都在里面。外层只加一层信封记录 schema 版本和身份信息，
让下游能判断「这是哪个会话哪一轮的什么版本的报告」。
"""

from __future__ import annotations

import json
import re
from typing import Any

EVAL_EXPORT_SCHEMA = "agent-quality-eval.turn-eval/v1"

MEDIA_TYPE = "application/json; charset=utf-8"

LLM_DIMENSION_LABELS = {
    "task_completion": "任务完成",
    "tool_use": "工具使用",
    "reasoning": "推理路径",
    "instruction_following": "指令遵循",
    "faithfulness": "忠实度",
    "efficiency": "效率",
    "reliability": "可靠性",
}


def sanitize_id(raw: str) -> str:
    """把 id 变成安全的文件名片段（``::`` 在 Windows 文件名里非法）。"""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", raw or "")
    cleaned = cleaned.strip("-.") or "session"
    return cleaned[:120]


def _headline(report: dict[str, Any]) -> dict[str, Any]:
    """给文件开头一段结论摘要。

    报告本体几十个字段，人或 LLM 打开文件时最想先知道的就是「过没过、哪几条
    断言炸了」。这几个字段全部是从 ``report`` 里读出来的副本，不是新事实，
    所以摘要和本体不可能对不上。
    """
    panel = report.get("eval_panel") if isinstance(report.get("eval_panel"), dict) else {}
    judge = report.get("judge") if isinstance(report.get("judge"), dict) else {}
    results = report.get("assertion_results")
    failed: list[str] = []
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            if item.get("passed") is False:
                label = str(
                    item.get("id")
                    or item.get("assertion_id")
                    or item.get("name")
                    or item.get("description")
                    or ""
                ).strip()
                if label:
                    failed.append(label)
    return {
        "passed": report.get("passed"),
        "overall_verdict": panel.get("overall_verdict"),
        "overall_score": report.get("overall_score"),
        "assertion_pass_rate": report.get("assertion_pass_rate"),
        "assertion_total": len(results) if isinstance(results, list) else None,
        "failed_assertions": failed,
        "critical_failures": report.get("critical_failures"),
        "eval_mode": report.get("eval_mode"),
        "eval_version": report.get("eval_version"),
        # provenance：这份结论是 hook 阶段自动跑的还是手动重跑的
        "judge_source_event": judge.get("source_event"),
        "judge_status": judge.get("status"),
        "judge_model": judge.get("model"),
        "judge_provider": judge.get("provider"),
    }


def _llm_review_export(report: dict[str, Any]) -> dict[str, Any]:
    """Build an explicit UI-shaped LLM review block for downstream harnesses.

    The original report is still exported verbatim under ``report``. This block
    is a denormalized convenience copy matching what the eval panel renders:
    conclusion, dimension reviews, evidence, claims, and live supervisor notes.
    """
    judge = report.get("judge") if isinstance(report.get("judge"), dict) else {}
    structured = judge.get("structured") if isinstance(judge.get("structured"), dict) else {}
    dimensions: list[dict[str, Any]] = []
    display_blocks: list[dict[str, Any]] = []

    coverage = structured.get("user_request_coverage") or structured.get("user_need_coverage") or ""
    if coverage:
        display_blocks.append({"key": "user_request_coverage", "label": "用户诉求覆盖情况", "text": coverage})

    for key, label in LLM_DIMENSION_LABELS.items():
        section = structured.get(key) if isinstance(structured.get(key), dict) else {}
        if not section:
            continue
        block = {
            "key": key,
            "label": label,
            "verdict": section.get("verdict"),
            "review": section.get("review"),
            "evidence": section.get("evidence") if isinstance(section.get("evidence"), list) else [],
        }
        dimensions.append(block)
        display_blocks.append({"type": "dimension", **block})

    live = judge.get("live_supervisor") if isinstance(judge.get("live_supervisor"), dict) else {}
    observations = live.get("observations") if isinstance(live.get("observations"), list) else []
    model_snapshots = live.get("model_snapshots") if isinstance(live.get("model_snapshots"), list) else []
    if live:
        display_blocks.append(
            {
                "type": "live_supervisor",
                "key": "live_supervisor",
                "label": "Live Critic Supervisor",
                "status": live.get("status"),
                "risk_level": live.get("risk_level"),
                "summary": live.get("live_summary"),
                "observations": observations,
                "model_snapshots": model_snapshots,
            }
        )

    return {
        "status": judge.get("status"),
        "provider": judge.get("provider"),
        "model": judge.get("model"),
        "source_event": judge.get("source_event"),
        "eval_method": judge.get("eval_method"),
        "report_path": judge.get("report_path"),
        "created_at": judge.get("created_at"),
        "verdict": judge.get("verdict") or structured.get("overall_verdict"),
        "summary_conclusion": structured.get("summary_conclusion") or judge.get("reason"),
        "user_request_coverage": coverage,
        "review_markdown": structured.get("review_markdown") or "",
        "dimensions": dimensions,
        "claims": structured.get("claims") if isinstance(structured.get("claims"), list) else [],
        "live_supervisor": live,
        "display_blocks": display_blocks,
        "raw_structured": structured,
    }


def export_turn_eval(report: dict[str, Any]) -> dict[str, Any]:
    """把一轮的 eval 报告导出成 JSON。

    Args:
        report: ``DatasetStore.get_latest_turn_eval()`` 的返回值。调用方必须
            保证它就是目标会话目标轮次的报告——本模块不做任何查找或兜底，
            免得导出时悄悄串了别的会话。

    Returns:
        ``{"content", "media_type", "filename", "session_id", "turn_index"}``
    """
    if not isinstance(report, dict):
        raise ValueError("eval 报告为空或格式不对，无法导出")

    from .models import utc_now

    session_id = str(report.get("session_id") or "")
    turn_index = report.get("turn_index")

    envelope = {
        "schema": EVAL_EXPORT_SCHEMA,
        "exported_at": utc_now(),
        "session_id": session_id,
        "turn_index": turn_index,
        "report_id": report.get("report_id") or report.get("id"),
        "created_at": report.get("created_at"),
        "summary": _headline(report),
        "llm_review": _llm_review_export(report),
        "report": report,
    }

    turn_part = f"turn{turn_index}" if turn_index is not None else "turn"
    return {
        "content": json.dumps(envelope, ensure_ascii=False, indent=2) + "\n",
        "media_type": MEDIA_TYPE,
        "filename": f"eval-{sanitize_id(session_id)}-{turn_part}.json",
        "session_id": session_id,
        "turn_index": turn_index,
        "schema": EVAL_EXPORT_SCHEMA,
    }
