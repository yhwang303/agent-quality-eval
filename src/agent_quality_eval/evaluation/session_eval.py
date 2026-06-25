"""Observed-session and turn-level evaluation metric extraction."""

from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Any

from .models import utc_now


ERROR_TERMS = ("error", "exception", "traceback", "failed", "failure", "错误", "异常", "失败")
PII_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b(?:api[_-]?key|secret|token|password|passwd)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}", re.I),
)

TURN_EVAL_ASSERTION_SET_VERSION = "turn-v3.7"

JUDGE_V3_TOP_LEVEL_KEYS = {
    "summary",
    "efficiency",
    "relevance",
    "instruction_following",
    "tool_use",
    "reasoning",
    "faithfulness",
    "task_completion",
    "overall_verdict",
}
JUDGE_V3_BLACKLIST = (
    "基本完成",
    "整体不错",
    "略有不足",
    "较为充分",
    "部分解决",
    "已解决",
    "未解决",
    "看起来",
    "似乎",
    "高概率",
    "大致",
    "缺少可验证 X",
    "缺少可解析 X",
    "缺少可核验 X",
)
JUDGE_V3_REVIEW_HEADINGS = (
    "**结论**：",
    "**效率** · ",
    "**相关性** · ",
    "**指令遵循** · ",
    "**工具使用** · ",
    "**推理路径** · ",
    "**忠实度** · ",
    "**任务完成** · ",
)
JUDGE_V3_ENUMS = {
    "efficiency.verdict": {"normal", "high", "excessive"},
    "relevance.verdict": {"aligned", "partial", "off"},
    "instruction_following.verdict": {"yes", "partial", "no"},
    "tool_use.verdict": {"correct", "suboptimal", "wrong"},
    "reasoning.verdict": {"on_track", "drift", "redundant", "lost"},
    "faithfulness.verdict": {"grounded", "partial", "hallucinated"},
    "task_completion.verdict": {"resolved", "partial", "unresolved"},
    "overall_verdict": {"resolved", "partial", "unresolved"},
}
JUDGE_V3_REQUIRED_OBJECT_KEYS = {
    "efficiency": {"verdict", "review"},
    "relevance": {"verdict", "review"},
    "instruction_following": {"verdict", "review"},
    "tool_use": {"verdict", "review"},
    "reasoning": {"verdict", "review"},
    "faithfulness": {"verdict", "review"},
    "task_completion": {"verdict", "review"},
}


TURN_SCORE_WEIGHTS: dict[str, float] = {
    "task_completion": 0.16,
    "answer_relevance": 0.10,
    "response_completeness": 0.08,
    "instruction_adherence": 0.08,
    "tool_correctness": 0.10,
    "argument_correctness": 0.08,
    "tool_efficiency": 0.08,
    "plan_quality": 0.07,
    "plan_adherence": 0.07,
    "error_free_execution": 0.08,
    "trace_completeness": 0.05,
    "safety_privacy": 0.03,
}


def build_turn_eval_report(
    session_id: str,
    turn_index: int,
    *,
    cot: dict[str, Any] | None = None,
    transcript: dict[str, Any] | None = None,
    otel: dict[str, Any] | None = None,
    overview: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a bilingual, trace-aware evaluation report for one user-agent turn."""
    cot = cot or {}
    turn = _find_turn(cot, turn_index)
    if turn is None:
        raise KeyError(f"turn not found: {turn_index}")

    return _build_turn_eval_report_v3(
        session_id,
        turn_index,
        turn=turn,
        cot=cot,
        transcript=transcript or {},
        otel=otel or {},
        overview=overview or {},
    )

    context = {
        "session_id": session_id,
        "turn_index": turn_index,
        "cot": cot,
        "turn": turn,
        "transcript": transcript or {},
        "otel": otel or {},
        "overview": overview or {},
    }
    metrics = extract_turn_metrics(context)
    scores = _score_turn(metrics, turn, cot)
    overall_score = _weighted_average(scores)
    passed = overall_score >= 0.75 and metrics["error_count"] == 0 and not metrics["pii_or_secret_risk"]
    return {
        "report_id": f"turn-eval-{session_id}-{turn_index}",
        "session_id": session_id,
        "turn_index": turn_index,
        "created_at": utc_now(),
        "passed": passed,
        "overall_score": overall_score,
        "quality_score": overall_score,
        "score_formula": {
            "description_zh": "Quality Score 是各维度得分的加权平均；没有人工 gold label 或 LLM judge 时使用 trace 启发式评分。",
            "description_en": "Quality Score is a weighted average of metric scores; without gold labels or an LLM judge, it uses trace-based heuristic scoring.",
            "weights": TURN_SCORE_WEIGHTS,
        },
        "metrics": metrics,
        "scores": scores,
        "summary": _build_turn_summary(metrics, scores),
        "ab_ready_dimensions": [
            "task_completion",
            "answer_relevance",
            "tool_correctness",
            "argument_correctness",
            "tool_efficiency",
            "plan_quality",
            "plan_adherence",
            "error_free_execution",
            "trace_completeness",
            "safety_privacy",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "tokens_per_second",
        ],
        "lineage": {
            "letsgoagenteval_retained": [
                "deterministic assertions",
                "LLM rubric/judge contract",
                "provider/scorer separation",
                "JSON/HTML report shape",
                "human review dimensions",
            ],
            "implementation_note_zh": "本 turn 评估没有直接 import LetsGoAgentEval 源码；保留的是评估语义和指标设计，运行时代码在 agent_quality_eval 内重写。",
            "implementation_note_en": "This turn evaluator does not directly import LetsGoAgentEval code; it retains the evaluation semantics and metric design, reimplemented inside agent_quality_eval.",
        },
        "source": {
            "has_cot": bool(cot),
            "has_transcript": bool(transcript),
            "has_otel": bool(otel),
            "has_overview": bool(overview),
        },
    }


def _build_turn_eval_report_v3(
    session_id: str,
    turn_index: int,
    *,
    turn: dict[str, Any],
    cot: dict[str, Any],
    transcript: dict[str, Any],
    otel: dict[str, Any],
    overview: dict[str, Any],
) -> dict[str, Any]:
    context = {
        "session_id": session_id,
        "turn_index": turn_index,
        "cot": cot,
        "turn": turn,
        "transcript": transcript,
        "otel": otel,
        "overview": overview,
    }
    metrics = extract_turn_metrics(context)
    task_profile = {
        "primary": "comprehensive_agent_eval",
        "labels": ["comprehensive_agent_eval"],
        "confidence": 1.0,
        "routing_basis": {
            "mode": "unified_assertion_template",
            "tool_count": metrics.get("tool_count", 0),
            "step_count": metrics.get("step_count", 0),
            "plan_update_count": metrics.get("plan_update_count", 0),
        },
    }
    turn_eval_config = _load_turn_eval_config()
    assertion_set = _build_v3_assertion_set(task_profile, metrics, turn, cot, turn_eval_config)
    assertion_results = _run_v3_turn_assertions(assertion_set["assertions"], metrics, turn, cot)
    raw_eval_context = _build_raw_eval_context(
        session_id=session_id,
        turn_index=turn_index,
        cot=cot,
        turn=turn,
        transcript=transcript,
        otel=otel,
        overview=overview,
    )
    judge = _run_optional_turn_judge(turn_eval_config, metrics, turn, raw_eval_context)

    scored = [item for item in assertion_results if not item.get("skipped")]
    passed_count = sum(1 for item in scored if item.get("passed"))
    assertion_pass_rate = passed_count / len(scored) if scored else 0.0
    critical_failures = [
        item for item in scored
        if not item.get("passed") and item.get("severity") in {"critical", "high"}
    ]
    assertion_groups = _group_assertion_results(assertion_results)
    eval_panel = _build_agent_eval_panel(
        assertion_results=assertion_results,
        assertion_groups=assertion_groups,
        assertion_pass_rate=assertion_pass_rate,
        metrics=metrics,
        turn=turn,
        judge=judge,
        critical_failures=critical_failures,
    )
    legacy_score = assertion_pass_rate
    passed = bool(eval_panel.get("overall_verdict") == "pass")
    summary = _build_v3_turn_summary(metrics, task_profile, assertion_results, assertion_pass_rate, eval_panel)
    pipeline = _build_v3_pipeline(task_profile, assertion_results, metrics)

    return {
        "report_id": f"turn-eval-{session_id}-{turn_index}",
        "session_id": session_id,
        "turn_index": turn_index,
        "created_at": utc_now(),
        "eval_version": "v3",
        "passed": passed,
        "overall_score": legacy_score,
        "quality_score": legacy_score,
        "assertion_pass_rate": assertion_pass_rate,
        "task_profile": task_profile,
        "assertion_set": {
            "version": assertion_set["version"],
            "source": assertion_set["source"],
            "default_assertions": assertion_set["default_assertions"],
            "specialized_assertions": assertion_set["specialized_assertions"],
            "total_assertions": len(assertion_set["assertions"]),
        },
        "assertion_results": assertion_results,
        "assertion_groups": assertion_groups,
        "critical_failures": critical_failures,
        "judge": {k: v for k, v in judge.items() if k != "assertion"},
        "eval_panel": eval_panel,
        "pipeline": pipeline,
        "score_breakdown": None,
        "score_formula": {
            "description_zh": "本版本取消跨维度加权综合分，改为核心 verdict 面板、诊断指标和 Safety Gate 并列展示。",
            "description_en": "This version does not compute a weighted quality score; it reports independent verdict panels, diagnostics, and safety gates.",
            "weights": {},
        },
        "metrics": metrics,
        "scores": _compat_scores_from_assertions(assertion_results),
        "summary": summary,
        "ab_ready_dimensions": [
            "assertion_pass_rate",
            "critical_failures",
            "task_outcome",
            "tool_use",
            "code_delivery",
            "research_grounding",
            "planning_execution",
            "computer_use",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "tokens_per_second",
            "tool_count",
            "tool_kind_count",
            "mcp_tool_count",
            "rag_tool_count",
            "retrieval_tool_count",
            "search_tool_count",
            "shell_tool_count",
            "file_tool_count",
            "browser_tool_count",
        ],
        "lineage": {
            "letsgoagenteval_retained": [
                "v3 declarative assertions",
                "provider/judge separation",
                "weighted outcome/trajectory scoring",
                "A/B-ready dimensions",
            "A/B comparison metadata",
            ],
            "implementation_note_zh": "Turn eval v3 is reimplemented locally and does not import LetsGoAgentEval source.",
            "implementation_note_en": "Turn eval v3 is reimplemented locally and does not import LetsGoAgentEval source.",
        },
        "source": {
            "has_cot": bool(cot),
            "has_transcript": bool(transcript),
            "has_otel": bool(otel),
            "has_overview": bool(overview),
            "raw_eval_sources": raw_eval_context.get("sources", []),
        },
    }


def _load_turn_eval_config() -> dict[str, Any]:
    config_path = os.environ.get("AGENT_QUALITY_EVAL_TURN_CONFIG")
    path = Path(config_path) if config_path else Path(os.environ.get("AGENT_QUALITY_EVAL_HOME", Path.home() / ".agent-quality-eval")) / "configs" / "turn_eval.yaml"
    try:
        from .settings import load_llm_judge_settings

        settings_judge = load_llm_judge_settings().to_judge_config()
    except Exception:
        settings_judge = None
    if not path.exists():
        data = {"config_path": str(path), "loaded": False}
        if settings_judge:
            data["judge"] = settings_judge
        return data
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict):
            data["config_path"] = str(path)
            data["loaded"] = True
            if settings_judge and not (data.get("judge") or data.get("llm_judge")):
                data["judge"] = settings_judge
            return data
    except Exception as exc:
        return {"config_path": str(path), "loaded": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"config_path": str(path), "loaded": False}


def _classify_turn_task(metrics: dict[str, Any], turn: dict[str, Any], cot: dict[str, Any]) -> dict[str, Any]:
    query = str(metrics.get("user_query") or "")
    text = (query + "\n" + _flatten_text(turn)).lower()
    tool_calls = [str(x).lower() for x in metrics.get("tool_calls") or []]
    labels: list[str] = []

    def add(label: str) -> None:
        if label not in labels:
            labels.append(label)

    if any(term in text for term in ("code", "test", "pytest", "build", "bug", "fix", "file", "repo", "代码", "测试", "修复", "文件", "仓库")):
        add("coding")
    if any(term in text for term in ("write", "edit", "patch", "apply_patch", "文件", "修改", "编辑")):
        add("file_editing")
    if any(term in text for term in ("research", "search", "source", "citation", "调研", "搜索", "资料", "来源", "报告")):
        add("research")
    if any(term in text for term in ("browser", "click", "screenshot", "gui", "页面", "点击", "截图")):
        add("computer_use")
    if metrics.get("plan_update_count") or any(term in text for term in ("plan", "todo", "步骤", "计划")):
        add("planning")
    if metrics.get("tool_count", 0) > 0 or tool_calls:
        add("tool_orchestration")
    if not labels:
        add("conversation")

    primary = labels[0]
    confidence = min(0.95, 0.45 + 0.12 * len(labels) + (0.15 if metrics.get("tool_count", 0) else 0))
    return {
        "primary": primary,
        "labels": labels,
        "confidence": confidence,
        "routing_basis": {
            "tool_count": metrics.get("tool_count", 0),
            "step_count": metrics.get("step_count", 0),
            "plan_update_count": metrics.get("plan_update_count", 0),
        },
    }


def _classify_turn_task(metrics: dict[str, Any], turn: dict[str, Any], cot: dict[str, Any]) -> dict[str, Any]:
    """Classify the turn with evidence-weighted routing, not a single keyword hit."""
    query = str(metrics.get("user_query") or "")
    final_response = str(turn.get("final_response") or "")
    text = (query + "\n" + final_response + "\n" + _flatten_text(turn)).lower()
    tool_calls = [str(x).lower() for x in metrics.get("tool_calls") or []]
    step_types = metrics.get("step_type_counts") or {}

    def hits(patterns: tuple[str, ...]) -> int:
        return sum(1 for pattern in patterns if re.search(pattern, text, re.I))

    file_tool_hits = sum(1 for name in tool_calls if any(x in name for x in ("read", "write", "edit", "patch", "glob", "grep", "ls", "file", "shell", "bash")))
    edit_tool_hits = sum(1 for name in tool_calls if any(x in name for x in ("write", "edit", "patch", "apply_patch")))
    browser_tool_hits = sum(1 for name in tool_calls if any(x in name for x in ("browser", "click", "screenshot", "mouse", "keyboard", "playwright")))
    retrieval_tool_hits = sum(1 for name in tool_calls if any(x in name for x in ("search", "web", "fetch", "query", "read")))

    scores = {
        "coding": 0.0,
        "file_editing": 0.0,
        "research": 0.0,
        "tool_orchestration": 0.0,
        "planning": 0.0,
        "computer_use": 0.0,
        "conversation": 0.0,
    }
    scores["coding"] += 0.18 * hits((r"\b(code|test|pytest|build|bug|fix|repo|compile|lint|package|release)\b", r"代码|测试|修复|构建|仓库|打包|发布"))
    scores["coding"] += min(0.35, 0.08 * file_tool_hits)
    scores["coding"] += 0.15 if step_types.get("tool_execution", 0) and file_tool_hits else 0.0
    scores["file_editing"] += 0.22 * hits((r"\b(write|edit|patch|apply_patch|modify|save|create file)\b", r"修改|编辑|写入|保存|创建文件"))
    scores["file_editing"] += min(0.45, 0.12 * edit_tool_hits)
    scores["research"] += 0.2 * hits((r"\b(research|search|source|citation|paper|report|doc|look up|verify)\b", r"调研|搜索|资料|来源|引用|报告|查一下|验证"))
    scores["research"] += min(0.3, 0.06 * retrieval_tool_hits)
    scores["tool_orchestration"] += min(0.8, 0.08 * metrics.get("tool_count", 0))
    scores["tool_orchestration"] += 0.2 if tool_calls else 0.0
    scores["planning"] += 0.25 if metrics.get("plan_update_count") else 0.0
    scores["planning"] += 0.16 * hits((r"\b(plan|todo|steps|checklist|strategy)\b", r"计划|步骤|待办|清单|方案"))
    scores["planning"] += 0.15 if metrics.get("step_count", 0) >= 8 else 0.0
    scores["computer_use"] += 0.25 * hits((r"\b(browser|click|screenshot|mouse|keyboard|gui|playwright|page)\b", r"页面|点击|截图|浏览器"))
    scores["computer_use"] += min(0.55, 0.18 * browser_tool_hits)
    scores["conversation"] = 0.35 if metrics.get("tool_count", 0) == 0 else 0.0

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    labels = [name for name, score in ranked if score >= 0.35]
    if not labels:
        labels = ["conversation"]

    return {
        "primary": labels[0],
        "labels": labels,
        "confidence": min(0.96, max(0.35, ranked[0][1])),
        "scores": {k: round(v, 3) for k, v in sorted(scores.items()) if v > 0},
        "routing_basis": {
            "tool_count": metrics.get("tool_count", 0),
            "step_count": metrics.get("step_count", 0),
            "plan_update_count": metrics.get("plan_update_count", 0),
            "file_tool_hits": file_tool_hits,
            "edit_tool_hits": edit_tool_hits,
            "browser_tool_hits": browser_tool_hits,
            "retrieval_tool_hits": retrieval_tool_hits,
        },
    }


def _build_v3_assertion_set(
    task_profile: dict[str, Any],
    metrics: dict[str, Any],
    turn: dict[str, Any],
    cot: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build one comprehensive Anthropic-style agent assertion template."""
    query = str(metrics.get("user_query") or "")
    full_trace_text = (query + "\n" + str(turn.get("final_response") or "") + "\n" + _flatten_text(turn)).lower()
    has_tools = int(metrics.get("tool_count", 0) or 0) > 0
    code_applicable = _query_mentions_file_edit(query) or _has_file_edit_evidence(turn, cot) or bool(
        re.search(r"\b(code|bug|fix|test|build|repo|pytest|npm|package|release)\b|代码|修复|测试|构建|仓库|打包|发布", full_trace_text, re.I)
    )
    research_applicable = _query_mentions_research(query) or _has_retrieval_evidence(turn) or bool(
        metrics.get("retrieval_tool_count", 0) or metrics.get("search_tool_count", 0) or metrics.get("rag_tool_count", 0)
    )
    planning_applicable = bool(metrics.get("plan_update_count")) or int(metrics.get("step_count", 0) or 0) >= 8
    computer_applicable = bool(metrics.get("browser_tool_count", 0)) or bool(
        re.search(r"\b(browser|click|screenshot|mouse|keyboard|gui|playwright)\b|页面|点击|截图|浏览器", full_trace_text, re.I)
    )
    assertions: list[dict[str, Any]] = [
        {"name": "final-response-present", "label_zh": "最终回答存在", "type": "non-empty", "category": "task_outcome", "severity": "critical", "binary": True},
        {"name": "response-specific-enough", "label_zh": "回答具有足够信息量", "type": "min-length", "category": "task_outcome", "severity": "medium", "value": 80, "quantitative": True},
        {"name": "error-free-execution", "label_zh": "执行过程无明显错误", "type": "no-error", "category": "execution_integrity", "severity": "high", "threshold": 0.8, "quantitative": True},
        {"name": "pii-or-secret-absent", "label_zh": "未泄露隐私或密钥", "type": "no-pii", "category": "execution_integrity", "severity": "high", "binary": True},
    ]
    specialized: list[str] = ["non-empty", "min-length", "no-error", "no-pii"]

    if has_tools:
        assertions.extend([
            {"name": "tool-errors-absent", "label_zh": "工具调用无错误", "type": "tool-error-free", "category": "tool_use", "severity": "high", "binary": True},
            {"name": "tool-args-valid", "label_zh": "工具参数有效", "type": "tool-args-valid", "category": "tool_use", "severity": "high", "threshold": 0.8, "quantitative": True},
            {"name": "tool-results-used-in-final", "label_zh": "最终回答引用工具结果", "type": "tool-results-used-in-final", "category": "tool_use", "severity": "medium", "binary": True},
            {"name": "tool-taxonomy-captured", "label_zh": "工具类型统计可观测", "type": "tool-taxonomy-captured", "category": "tool_use", "severity": "medium", "binary": True},
        ])
        specialized.extend(["tool-error-free", "tool-args-valid", "tool-results-used-in-final", "tool-taxonomy-captured"])
    else:
        assertions.append({"name": "no-unnecessary-tool-use", "label_zh": "简单任务未滥用工具", "type": "no-unnecessary-tool-use", "category": "tool_use", "severity": "medium", "binary": True})
        specialized.append("no-unnecessary-tool-use")

    if code_applicable:
        assertions.extend([
            {"name": "file-edit-observed-if-requested", "label_zh": "按需产生文件修改证据", "type": "file-edit-observed-if-requested", "category": "code_delivery", "severity": "high", "binary": True},
            {"name": "validation-run-after-edit", "label_zh": "修改后有验证动作", "type": "validation-run-after-edit", "category": "code_delivery", "severity": "high", "binary": True},
            {"name": "changed-files-mentioned", "label_zh": "说明变更文件或路径", "type": "changed-files-mentioned", "category": "code_delivery", "severity": "medium", "binary": True},
            {"name": "no-unverified-code-claim", "label_zh": "未声称未经证实的验证结果", "type": "no-unverified-code-claim", "category": "code_delivery", "severity": "medium", "binary": True},
        ])
        specialized.extend(["file-edit-observed-if-requested", "validation-run-after-edit", "changed-files-mentioned", "no-unverified-code-claim"])

    if research_applicable:
        assertions.extend([
            {"name": "retrieval-used-if-needed", "label_zh": "研究问题使用检索证据", "type": "retrieval-used-if-needed", "category": "research_grounding", "severity": "high", "binary": True},
            {"name": "source-grounding-present", "label_zh": "回答具备来源支撑", "type": "source-grounding-present", "category": "research_grounding", "severity": "high", "binary": True},
            {"name": "evidence-synthesis-present", "label_zh": "形成证据综合而非堆砌", "type": "evidence-synthesis-present", "category": "research_grounding", "severity": "medium", "binary": True},
        ])
        specialized.extend(["retrieval-used-if-needed", "source-grounding-present", "evidence-synthesis-present"])

    if planning_applicable:
        assertions.extend([
            {"name": "plan-created-for-complex-task", "label_zh": "复杂任务有计划痕迹", "type": "plan-created-for-complex-task", "category": "planning_execution", "severity": "medium", "binary": True},
            {"name": "plan-final-alignment", "label_zh": "计划进度与最终回答一致", "type": "plan-final-alignment", "category": "planning_execution", "severity": "medium", "threshold": 0.65, "quantitative": True},
        ])
        specialized.extend(["plan-created-for-complex-task", "plan-final-alignment"])

    if computer_applicable:
        assertions.extend([
            {"name": "gui-action-observed", "label_zh": "GUI/浏览器动作已捕获", "type": "gui-action-observed", "category": "computer_use", "severity": "high", "binary": True},
            {"name": "final-state-evidence-present", "label_zh": "GUI 最终状态有证据", "type": "final-state-evidence-present", "category": "computer_use", "severity": "high", "binary": True},
        ])
        specialized.extend(["gui-action-observed", "final-state-evidence-present"])

    for item in config.get("assertions", []) if isinstance(config.get("assertions"), list) else []:
        if isinstance(item, dict):
            category = str(item.get("category") or "").lower()
            atype = str(item.get("type") or item.get("name") or "").lower().replace("_", "-")
            if category in {"trace_evidence", "efficiency", "safety", "optional_judge"}:
                continue
            if atype in {"llm-rubric", "llm", "task-completion", "plan-quality", "plan-adherence"}:
                continue
            assertions.append({**item, "source": item.get("source", "turn_eval_config")})

    return {
        "version": TURN_EVAL_ASSERTION_SET_VERSION,
        "source": "built-in+turn_eval_config" if config.get("loaded") else "built-in",
        "default_assertions": ["non-empty", "min-length", "no-error", "no-pii"],
        "specialized_assertions": specialized,
        "assertions": assertions,
    }


def _run_v3_turn_assertions(
    assertions: list[dict[str, Any]],
    metrics: dict[str, Any],
    turn: dict[str, Any],
    cot: dict[str, Any],
) -> list[dict[str, Any]]:
    return [_run_v3_turn_assertion(assertion, metrics, turn, cot) for assertion in assertions]


def _run_v3_turn_assertion(
    assertion: dict[str, Any],
    metrics: dict[str, Any],
    turn: dict[str, Any],
    cot: dict[str, Any],
) -> dict[str, Any]:
    atype = str(assertion.get("type") or assertion.get("name") or "").lower().replace("_", "-")
    threshold = float(assertion.get("threshold", 0.5))
    final_response = str(turn.get("final_response") or "")
    query = str(metrics.get("user_query") or "")
    steps_text = _flatten_text(turn.get("steps") or [])

    if atype == "non-empty":
        return _v3_result(assertion, bool(final_response.strip()), 1.0 if final_response.strip() else 0.0, "已生成最终回答。" if final_response.strip() else "最终回答为空。", {"chars": len(final_response)})
    if atype == "no-error":
        score = 1.0 if metrics.get("error_count", 0) == 0 else max(0.0, 1.0 - 0.25 * metrics.get("error_count", 0))
        evidence = {
            "error_count": metrics.get("error_count", 0),
            "error_breakdown": metrics.get("error_breakdown", {}),
            "error_samples": metrics.get("error_samples", []),
            "tool_error_by_tool": metrics.get("tool_error_by_tool", {}),
        }
        return _v3_result(
            assertion,
            score >= threshold,
            score,
            "未检测到明显错误信号。" if score == 1.0 else f"检测到 {metrics.get('error_count')} 个错误或恢复信号，已按类型展开到证据明细。",
            evidence,
        )
    if atype == "no-pii":
        return _v3_result(assertion, not metrics.get("pii_or_secret_risk"), 0.0 if metrics.get("pii_or_secret_risk") else 1.0, "未发现明显 PII 或密钥模式。" if not metrics.get("pii_or_secret_risk") else "检测到疑似 PII 或密钥模式。", {"hits": metrics.get("pii_or_secret_hits", [])})
    if atype == "trace-complete":
        score = _trace_completeness_score(metrics)
        return _v3_result(assertion, score >= threshold, score, "根据步骤、耗时、Token、工具和最终回答计算 Trace 证据覆盖度。", metrics.get("trace_fields_present", {}))
    if atype == "token-usage-captured":
        passed = metrics.get("total_tokens", 0) > 0
        return _v3_result(assertion, passed, 1.0 if passed else 0.0, "已采集 Token 用量。" if passed else "未采集到 Token 用量。", {"total_tokens": metrics.get("total_tokens", 0)})
    if atype == "min-length":
        limit = int(assertion.get("value") or assertion.get("min") or 1)
        length = len(final_response.strip())
        score = min(1.0, length / max(1, limit))
        return _v3_result(assertion, length >= limit, score, f"最终回答长度 {length} 字符，最低要求 {limit} 字符。", {"chars": length, "min": limit})
    if atype == "max-length":
        limit = int(assertion.get("value") or assertion.get("max") or 10000)
        length = len(final_response.strip())
        score = 1.0 if length <= limit else max(0.0, limit / max(1, length))
        return _v3_result(assertion, length <= limit, score, f"最终回答长度 {length} 字符，最高限制 {limit} 字符。", {"chars": length, "max": limit})
    if atype == "contains":
        value = str(assertion.get("value") or "")
        passed = value.lower() in final_response.lower()
        return _v3_result(assertion, passed, 1.0 if passed else 0.0, f"最终回答{'包含' if passed else '未包含'}指定文本：{value}", {"value": value})
    if atype == "not-contains":
        value = str(assertion.get("value") or "")
        passed = value.lower() not in final_response.lower()
        return _v3_result(assertion, passed, 1.0 if passed else 0.0, f"最终回答{'未包含' if passed else '包含了'}禁用文本：{value}", {"value": value})
    if atype == "equals":
        value = str(assertion.get("value") or "").strip()
        passed = final_response.strip() == value
        return _v3_result(assertion, passed, 1.0 if passed else 0.0, "最终回答与期望文本完全一致。" if passed else "最终回答与期望文本不一致。", {"value": value})
    if atype == "starts-with":
        value = str(assertion.get("value") or "")
        passed = final_response.strip().startswith(value)
        return _v3_result(assertion, passed, 1.0 if passed else 0.0, f"最终回答{'以' if passed else '未以'}指定文本开头：{value}", {"value": value})
    if atype == "ends-with":
        value = str(assertion.get("value") or "")
        passed = final_response.strip().endswith(value)
        return _v3_result(assertion, passed, 1.0 if passed else 0.0, f"最终回答{'以' if passed else '未以'}指定文本结尾：{value}", {"value": value})
    if atype == "regex":
        pattern = str(assertion.get("value") or "")
        try:
            passed = bool(re.search(pattern, final_response, re.I))
            return _v3_result(assertion, passed, 1.0 if passed else 0.0, f"最终回答{'匹配' if passed else '未匹配'}正则：{pattern}", {"pattern": pattern})
        except re.error as exc:
            return _v3_result(assertion, False, 0.0, f"正则表达式无效：{exc}", {"pattern": pattern})
    if atype == "contains-json":
        passed = _contains_json(final_response)
        return _v3_result(assertion, passed, 1.0 if passed else 0.0, "最终回答包含有效 JSON。" if passed else "最终回答不包含有效 JSON。", {})
    if atype in {"keyword", "keywords"}:
        values = assertion.get("value") or assertion.get("keywords") or []
        keywords = [str(x) for x in (values if isinstance(values, list) else [values]) if str(x)]
        found = [kw for kw in keywords if kw.lower() in final_response.lower()]
        all_required = bool(assertion.get("all_required", True))
        passed = len(found) == len(keywords) if all_required else bool(found)
        score = (len(found) / max(1, len(keywords))) if keywords else 1.0
        return _v3_result(assertion, passed, score, f"命中关键词 {found}；期望关键词 {keywords}。", {"found": found, "keywords": keywords})
    if atype == "latency-captured":
        passed = metrics.get("duration_ms") is not None
        return _v3_result(assertion, passed, 1.0 if passed else 0.0, "已采集本轮耗时。" if passed else "未采集到本轮耗时。", {"duration_ms": metrics.get("duration_ms")})
    if atype == "tool-error-free":
        passed = metrics.get("tool_error_count", 0) == 0
        return _v3_result(
            assertion,
            passed,
            1.0 if passed else 0.0,
            "未检测到工具错误。" if passed else f"检测到 {metrics.get('tool_error_count')} 次工具错误，已按工具名统计。",
            {
                "tool_error_count": metrics.get("tool_error_count", 0),
                "tool_error_by_tool": metrics.get("tool_error_by_tool", {}),
                "error_samples": metrics.get("error_samples", []),
            },
        )
    if atype == "tool-args-valid":
        score = _argument_correctness_score(turn)
        return _v3_result(assertion, score >= threshold, score, "工具参数结构看起来有效。" if score >= threshold else "工具参数存在缺失、空值或与错误相关。", {})
    if atype == "tool-results-used-in-final":
        passed = metrics.get("tool_count", 0) == 0 or _tool_results_referenced(final_response, turn)
        return _v3_result(assertion, passed, 1.0 if passed else 0.0, "最终回答引用了工具结果，或本轮无需工具。" if passed else "本轮使用了工具，但最终回答缺少明确的工具结果引用。", {})
    if atype == "tool-taxonomy-captured":
        counts = {
            "tool_count": metrics.get("tool_count", 0),
            "tool_kind_count": metrics.get("tool_kind_count", 0),
            "tool_category_counts": metrics.get("tool_category_counts", {}),
            "tool_name_counts": metrics.get("tool_name_counts", {}),
        }
        passed = int(counts["tool_count"] or 0) > 0
        return _v3_result(assertion, passed, 1.0 if passed else 0.0, f"已统计工具总次数、工具类别数和具体工具调用次数。", counts)
    if atype == "max-tool-calls":
        limit = int(assertion.get("max") or assertion.get("value") or 20)
        count = int(metrics.get("tool_count", 0) or 0)
        score = 1.0 if count <= limit else max(0.0, limit / max(1, count))
        return _v3_result(assertion, count <= limit, score, f"工具调用 {count} 次，限制 {limit} 次。", {"tool_count": count, "limit": limit})
    if atype == "no-unnecessary-tool-use":
        likely = _query_likely_needs_tool(query)
        passed = likely or metrics.get("tool_count", 0) <= 2
        score = 1.0 if passed else 0.4
        return _v3_result(assertion, passed, score, "工具使用与任务形态匹配。" if passed else "简单对话任务疑似使用了不必要的工具。", {"tool_count": metrics.get("tool_count", 0), "likely_tool_needed": likely})
    if atype == "file-edit-observed-if-requested":
        requested = _query_mentions_file_edit(query)
        observed = _has_file_edit_evidence(turn, cot)
        passed = (not requested) or observed
        return _v3_result(assertion, passed, 1.0 if passed else 0.0, "存在文件修改证据，或本轮不需要修改文件。" if passed else "用户请求了文件/代码修改，但 Trace 中没有捕获到修改证据。", {"requested": requested, "observed": observed})
    if atype == "validation-run-after-edit":
        edit_observed = _has_file_edit_evidence(turn, cot)
        validation = _has_validation_evidence(turn)
        passed = (not edit_observed) or validation
        return _v3_result(assertion, passed, 1.0 if passed else 0.0, "观察到验证动作，或本轮没有发生文件修改。" if passed else "发生了文件/代码修改，但缺少测试、构建或检查证据。", {"edit_observed": edit_observed, "validation_observed": validation})
    if atype == "no-unverified-code-claim":
        claims_verified = not re.search(r"\b(test|tests|build|pytest|passed|all green|全部通过)\b", final_response, re.I) or _has_validation_evidence(turn)
        return _v3_result(assertion, claims_verified, 1.0 if claims_verified else 0.0, "未发现未经证实的验证成功声明。" if claims_verified else "最终回答声称验证成功，但 Trace 中没有捕获到验证证据。", {})
    if atype == "retrieval-used-if-needed":
        needed = _query_mentions_research(query)
        observed = _has_retrieval_evidence(turn)
        passed = (not needed) or observed
        return _v3_result(assertion, passed, 1.0 if passed else 0.0, "存在检索/搜索证据，或本轮不需要检索。" if passed else "研究型问题缺少检索/搜索证据。", {"needed": needed, "observed": observed})
    if atype == "changed-files-mentioned":
        edit_observed = _has_file_edit_evidence(turn, cot)
        mentioned = bool(re.search(r"\b[\w./\\-]+\.(py|ts|tsx|js|jsx|json|yaml|yml|md|css|html|toml|ini)\b|file|path|文件|路径", final_response, re.I))
        passed = (not edit_observed) or mentioned
        return _v3_result(assertion, passed, 1.0 if passed else 0.0, "最终回答说明了变更文件/路径，或本轮没有发生文件修改。" if passed else "发生了文件/代码修改，但最终回答没有说明变更文件。", {"edit_observed": edit_observed, "mentioned": mentioned})
    if atype == "source-grounding-present":
        needed = _query_mentions_research(query)
        grounded = bool(re.search(r"https?://|source|citation|引用|来源|根据|参考", final_response, re.I)) or _has_retrieval_evidence(turn)
        passed = (not needed) or grounded
        return _v3_result(assertion, passed, 1.0 if passed else 0.0, "找到来源支撑证据，或本轮不需要来源支撑。" if passed else "研究型回答缺少来源支撑证据。", {})
    if atype == "no-unsupported-research-claim":
        risky_claim = _query_mentions_research(query) and len(final_response) > 200 and not _has_retrieval_evidence(turn) and not re.search(r"来源|引用|https?://", final_response, re.I)
        return _v3_result(assertion, not risky_claim, 0.0 if risky_claim else 1.0, "未发现缺少证据支撑的研究结论。" if not risky_claim else "较长研究结论缺少检索或引用证据。", {})
    if atype == "evidence-synthesis-present":
        needed = _query_mentions_research(query)
        synthesized = bool(re.search(r"because|therefore|based on|evidence|source|原因|依据|因此|结论|来源|引用|对比", final_response, re.I))
        passed = (not needed) or synthesized
        return _v3_result(assertion, passed, 1.0 if passed else 0.0, "研究回答包含证据综合表达，或本轮不需要研究综合。" if passed else "研究回答缺少明确的证据综合。", {"needed": needed, "synthesized": synthesized})
    if atype == "plan-created-for-complex-task":
        complex_task = metrics.get("step_count", 0) >= 6 or metrics.get("tool_count", 0) >= 3
        has_plan = metrics.get("plan_update_count", 0) > 0
        passed = (not complex_task) or has_plan
        return _v3_result(assertion, passed, 1.0 if passed else 0.0, "存在计划证据，或本轮任务较简单。" if passed else "复杂任务缺少计划或 Todo 更新证据。", {"complex_task": complex_task, "plan_update_count": metrics.get("plan_update_count", 0)})
    if atype == "plan-final-alignment":
        score = _plan_adherence_score(metrics, turn, cot)
        return _v3_result(assertion, score >= threshold, score, "计划进度与最终回答基本一致。" if score >= threshold else "计划进度与最终回答不一致或证据不足。", {})
    if atype == "gui-action-observed":
        observed = bool(re.search(r"browser|click|screenshot|mouse|keyboard|gui|页面|点击|截图", steps_text, re.I))
        return _v3_result(assertion, observed, 1.0 if observed else 0.0, "已捕获 GUI/浏览器动作证据。" if observed else "未捕获到 GUI/浏览器动作证据。", {})
    if atype == "final-state-evidence-present":
        evidence = bool(re.search(r"done|success|saved|created|updated|完成|成功|已保存|已创建|已更新", final_response, re.I))
        return _v3_result(assertion, evidence, 1.0 if evidence else 0.0, "最终回答包含 GUI 操作后的状态证据。" if evidence else "最终回答缺少 GUI 操作后的状态证据。", {})
    return _v3_result(assertion, False, 0.0, f"未知 V3 断言类型：{atype}", {})


def _v3_result(
    assertion: dict[str, Any],
    passed: bool,
    score: float,
    reason: str,
    evidence: dict[str, Any] | list[Any] | None,
    *,
    skipped: bool = False,
) -> dict[str, Any]:
    score = max(0.0, min(1.0, float(score)))
    threshold = float(assertion.get("threshold", 0.5))
    name = str(assertion.get("name") or assertion.get("type") or "assertion")
    atype = str(assertion.get("type") or name)
    label_zh = str(assertion.get("label_zh") or name)
    return {
        "key": name,
        "name": name,
        "label_zh": label_zh,
        "label_en": str(assertion.get("label_en") or name),
        "type": atype,
        "source": assertion.get("source", "built-in"),
        "category": assertion.get("category", "outcome"),
        "severity": assertion.get("severity", "medium"),
        "score": score,
        "threshold": threshold,
        "passed": bool(passed),
        "skipped": skipped,
        "reason": reason,
        "reason_en": reason,
        "reason_zh": reason,
        "evidence": evidence or {},
        "binary": bool(assertion.get("binary")),
        "quantitative": bool(assertion.get("quantitative")),
    }


def _build_raw_eval_context(
    *,
    session_id: str,
    turn_index: int,
    cot: dict[str, Any],
    turn: dict[str, Any],
    transcript: dict[str, Any],
    otel: dict[str, Any],
    overview: dict[str, Any] | None,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """Collect bounded original session evidence for LLM judge input.

    The visual trace is still useful for humans, but semantic judging should
    prefer original transcript / hook / OTel payloads when they are available.
    The returned object stores only source metadata plus a bounded excerpt.
    """
    sources: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    def add_source(label: str, value: Any, *, source_path: str | None = None, limit: int = 3000) -> None:
        if value in (None, "", [], {}):
            return
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                text = str(value)
        text = text.strip()
        if not text:
            return
        excerpt = text[:limit]
        evt_id = f"evt:{len(events)}"
        events.append({"id": evt_id, "source": label, "text": excerpt})
        sources.append(
            {
                "label": label,
                "evt_id": evt_id,
                "chars": len(text),
                "included_chars": len(excerpt),
                "truncated": len(text) > len(excerpt),
                "path_name": Path(source_path).name if source_path else None,
            }
        )

    add_source("turn.raw_object", turn, limit=2500)
    add_source("session.transcript_cache", transcript, limit=3000)
    if isinstance(otel, dict):
        add_source("session.otel_summary", otel.get("summary") or {}, limit=1800)
        add_source("session.otel_events", otel.get("events") or [], limit=2000)
        add_source("session.otel_spans", otel.get("spans") or [], limit=2000)
    add_source("session.overview", overview or {}, limit=1000)

    transcript_path = str(cot.get("transcript_path") or "").strip() if isinstance(cot, dict) else ""
    if transcript_path:
        try:
            path = Path(transcript_path).expanduser()
            if path.exists() and path.is_file():
                add_source("raw.transcript_file", path.read_text(encoding="utf-8", errors="replace"), source_path=str(path), limit=3500)
        except Exception as exc:
            sources.append({"label": "raw.transcript_file", "error": f"{type(exc).__name__}: {exc}", "path_name": Path(transcript_path).name})

    text = "\n\n".join(f"{event['id']} [{event['source']}]\n{event['text']}" for event in events)
    if len(text) > max_chars:
        text = text[:max_chars]
    return {
        "session_id": session_id,
        "turn_index": turn_index,
        "text": text,
        "chars": len(text),
        "sources": sources,
        "events": [{"id": event["id"], "source": event["source"]} for event in events],
    }


def _run_optional_turn_judge(
    config: dict[str, Any],
    metrics: dict[str, Any],
    turn: dict[str, Any],
    raw_eval_context: dict[str, Any],
) -> dict[str, Any]:
    session_id = str(raw_eval_context.get("session_id") or "")
    turn_index = _to_int(raw_eval_context.get("turn_index"))
    live_supervisor = None
    try:
        from .critic import (
            critic_report_path,
            is_stale_incomplete_critic_report,
            load_best_live_critic_state,
            load_critic_report,
        )

        critic_report = load_critic_report(session_id, turn_index)
        try:
            live_supervisor = load_best_live_critic_state(session_id, turn_index)
        except Exception:
            pass
    except Exception:
        critic_report = None
    if isinstance(critic_report, dict):
        status = str(critic_report.get("status") or "").lower()
        report_live_supervisor = critic_report.get("live_supervisor")
        if report_live_supervisor:
            live_supervisor = report_live_supervisor
        try:
            live_supervisor = live_supervisor or load_best_live_critic_state(session_id, turn_index)
        except Exception:
            pass
        if status in {"queued", "running"}:
            stale = is_stale_incomplete_critic_report(critic_report)
            display_status = "interrupted" if stale else status
            reason = (
                "Hook-stage Agent Critic started but did not finish within the expected window; click Eval to regenerate the missing hook report for this trace."
                if stale
                else "Hook-stage Agent Critic is still generating the final report for this trace."
            )
            return {
                "status": display_status,
                "provider": critic_report.get("provider"),
                "model": critic_report.get("model"),
                "score": None,
                "verdict": None,
                "structured": {},
                "reason": reason,
                "input_sources": critic_report.get("input_sources") or raw_eval_context.get("sources", []),
                "eval_method": critic_report.get("eval_method") or "agent_critic_v1",
                "source_event": critic_report.get("source_event"),
                "live_supervisor": live_supervisor,
                "report_path": str(critic_report_path(session_id, turn_index)),
                "created_at": critic_report.get("created_at"),
            }
        structured = critic_report.get("structured") if isinstance(critic_report.get("structured"), dict) else {}
        if not structured and any(key in critic_report for key in ("summary_conclusion", "task_completion", "tool_use")):
            structured = {k: critic_report.get(k) for k in (
                "summary_conclusion",
                "overall_verdict",
                "user_request_coverage",
                "task_completion",
                "tool_use",
                "reasoning",
                "instruction_following",
                "faithfulness",
                "efficiency",
                "reliability",
                "review_markdown",
            ) if k in critic_report}
        verdict = structured.get("overall_verdict") or critic_report.get("overall_verdict")
        reason = (
            structured.get("summary_conclusion")
            or critic_report.get("summary_conclusion")
            or critic_report.get("reason")
            or "Agent Critic report loaded."
        )
        return {
            "status": critic_report.get("status") or "completed",
            "provider": critic_report.get("provider"),
            "model": critic_report.get("model"),
            "score": _judge_v3_compat_score(verdict),
            "verdict": verdict,
            "structured": structured,
            "reason": reason,
            "input_sources": critic_report.get("input_sources") or raw_eval_context.get("sources", []),
            "eval_method": critic_report.get("eval_method") or "agent_critic_v1",
            "source_event": critic_report.get("source_event"),
            "live_supervisor": live_supervisor,
            "report_path": str(critic_report_path(session_id, turn_index)),
            "created_at": critic_report.get("created_at"),
        }
    try:
        from .critic import critic_report_path

        report_path = str(critic_report_path(session_id, turn_index))
    except Exception:
        report_path = None
    if isinstance(live_supervisor, dict):
        return {
            "status": "interrupted",
            "reason": "Hook/live supervisor observed this trace, but the final Agent Critic report is missing. Click Eval to regenerate the missing hook report.",
            "config_path": config.get("config_path"),
            "input_sources": raw_eval_context.get("sources", []),
            "eval_method": "agent_critic_v1",
            "source_event": live_supervisor.get("last_event") or live_supervisor.get("source_event"),
            "live_supervisor": live_supervisor,
            "report_path": report_path,
        }
    return {
        "status": "missing",
        "reason": "本轮尚未发现 hook 阶段生成的 Agent Critic report；可能是 IDE hook/subagent 没有触发，或评审仍未落盘。点击 Eval 会优先读取 hook 产物，缺失时启动 Agent Critic 手动兜底并标明来源。",
        "config_path": config.get("config_path"),
        "input_sources": raw_eval_context.get("sources", []),
        "eval_method": "agent_critic_v1",
        "report_path": report_path,
    }


def _build_structured_turn_judge_prompt(
    *,
    metrics: dict[str, Any],
    turn: dict[str, Any],
    raw_eval_context: dict[str, Any],
    validation_errors: list[str] | None = None,
) -> str:
    judge_input = _build_judge_v3_input(metrics, turn, raw_eval_context)
    schema = {
        "summary": "120-220字 一段完整自然语言总结。先给整体判断，再概述用户诉求、agent 关键动作、最终交付质量、主要影响因素。",
        "efficiency": {
            "verdict": "normal | high | excessive",
            "review": "80-180字。围绕 runtime_metrics 的 token、耗时、工具调用次数与失败数，对照任务复杂度判断是否合理。",
        },
        "relevance": {
            "verdict": "aligned | partial | off",
            "review": "80-180字。说明用户实际目标、最终回复如何回应，以及二者对齐程度。",
        },
        "instruction_following": {
            "verdict": "yes | partial | no",
            "review": "80-180字。识别用户硬约束并判断是否满足；无显式约束时说明隐含意图。",
        },
        "tool_use": {
            "verdict": "correct | suboptimal | wrong",
            "review": "100-220字。说明预期工具调用 vs 实际调用；区分已恢复失败与影响结果失败；引用 tool_call#N。",
        },
        "reasoning": {
            "verdict": "on_track | drift | redundant | lost",
            "review": "80-180字。客观描述推理轨迹与关键节点；若任务最终完成，不应判 lost。",
        },
        "faithfulness": {
            "verdict": "grounded | partial | hallucinated",
            "review": "80-180字。评估最终回复关键声称是否有原始 tool_result 支撑。",
        },
        "task_completion": {
            "verdict": "resolved | partial | unresolved",
            "review": "120-220字。说明用户最初请求与最终交付的对应关系、产出形式、实质缺口。",
        },
        "overall_verdict": "resolved | partial | unresolved",
    }
    retry_note = ""
    if validation_errors:
        retry_note = (
            "\n【上一次输出无法解析为 JSON】\n"
            + "\n".join(f"- {error}" for error in validation_errors[:4])
            + "\n请仅修正 JSON 格式后重新输出。\n"
        )
    return (
        "你是一名资深 Agent 行为评审员。只输出一个 JSON 对象，不要 Markdown、不要解释、不要代码块包裹。所有自然语言字段使用中文。\n\n"
        "【评审输入】（原始数据，请直接据此评审，无需任何兜底）\n"
        f"- 用户消息：{judge_input['user_message']}\n"
        f"- assistant 各轮（含 thinking、tool_calls、content）：{json.dumps(judge_input['assistant_turns'], ensure_ascii=False, indent=2)}\n"
        f"- 工具返回：{json.dumps(judge_input['tool_results'], ensure_ascii=False, indent=2)}\n"
        f"- 最终回复：{judge_input['final_response']}\n"
        f"- 运行指标：{json.dumps(judge_input['runtime_metrics'], ensure_ascii=False, indent=2)}\n\n"
        "【评审思路】\n"
        "你拥有完整的 transcript、工具调用与返回、最终回复以及运行统计，完全足以给出一份详尽的评审。不要因为输入看起来不完整就回避结论；当原始数据无法支撑某个维度的判断时，直接以中性 verdict（partial / suboptimal 等）加一段说明带过即可。\n\n"
        "请按 7 个维度产出评审，每个维度给一个 verdict + 一段 80-180 字的自然语言 review：\n"
        "1. efficiency：基于 runtime_metrics 与任务复杂度判断 token / 耗时 / 工具次数是否合理。\n"
        "2. relevance：最终回复是否对齐用户原始目标。\n"
        "3. instruction_following：用户的硬约束是否被满足；若无硬约束，明确说明并基于隐含意图做合理评判。\n"
        "4. tool_use：工具调用是否匹配预期；区分已恢复的失败与影响结果的失败。\n"
        "5. reasoning：推理轨迹是否健康；若任务最终完成，不应判 lost。\n"
        "6. faithfulness：最终回复的关键声称是否有 tool_result 支撑。\n"
        "7. task_completion：用户的最初请求是否被实际满足。\n\n"
        "最后产出 summary：120-220 字的一段完整自然语言总结，覆盖整体判断 + 用户诉求 + agent 关键动作 + 最终交付质量 + 主要影响因素。\n\n"
        "【影响导向】\n"
        "- 单步失败、单点跑偏不必上升为整体问题；只有真正影响最终交付时才作为负面结论。\n"
        "- 失败被重试或替代方案恢复的，应被自然描述为路径有波动但已恢复。\n"
        "- 不要因为局部异常就给出 wrong / lost / unresolved。\n\n"
        "【输出风格】\n"
        "- 每个 review 字段是一段连贯自然语言，不要用 bullet 列点。\n"
        "- 不要写未提供、无法判断、需重新生成之类的系统话术；如证据不足，以基于现有数据，整体判断为……的口吻继续给出评审。\n"
        "- summary 是用户唯一会通读的段落，必须自然、专业、信息量充足。\n"
        f"{retry_note}"
        "【输出 schema】\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "只返回 JSON 对象。"
    )

def _build_judge_v3_input(metrics: dict[str, Any], turn: dict[str, Any], raw_eval_context: dict[str, Any]) -> dict[str, Any]:
    """Shape original turn data into the v3 judge input contract."""
    user_message = str(metrics.get("user_query") or turn.get("user_query") or "").strip()
    final_response = str(turn.get("final_response") or "").strip()
    assistant_turns: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    tool_call_index = 1
    elapsed_seconds = round((float(metrics.get("duration_ms") or 0.0) / 1000.0), 2)
    runtime_metrics = {
        "total_tokens": max(0, _to_int(metrics.get("total_tokens"))),
        "elapsed_seconds": elapsed_seconds,
        "tool_calls_total": max(0, _to_int(metrics.get("tool_count"))),
        "tool_calls_failed": max(0, _to_int(metrics.get("tool_error_count"))),
        "tool_category_counts": metrics.get("tool_category_counts") or {},
        "tool_name_counts": metrics.get("tool_name_counts") or {},
    }

    for idx, step in enumerate(turn.get("steps") or [], start=1):
        if not isinstance(step, dict):
            continue
        step_type = str(step.get("step_type") or step.get("type") or "")
        metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        tool_name = _step_tool_name(step) or str(step.get("tool_name") or metadata.get("tool_name") or metadata.get("name") or "").strip()
        tool_input = metadata.get("tool_input") or metadata.get("input") or metadata.get("arguments")
        content = _limit_text(step.get("content"), 1200)
        if step_type == "tool_execution":
            tool_results.append(
                {
                    "ref": f"tool_call#{tool_call_index}",
                    "turn_ref": f"turn#{idx}",
                    "tool": tool_name or "unknown",
                    "duration_ms": step.get("duration_ms") or metadata.get("duration_ms"),
                    "is_error": bool(metadata.get("is_error")),
                    "input": _limit_text(tool_input, 360),
                    "result": content,
                }
            )
            tool_call_index += 1
            continue
        assistant_turns.append(
            {
                "ref": f"turn#{idx}",
                "kind": step_type or "assistant",
                "thinking": content if step_type in {"thinking_inter", "thinking_intermediate", "thinking_explicit", "pre_tool_reasoning", "tool_decision"} else "",
                "duration_ms": step.get("duration_ms") or metadata.get("duration_ms"),
                "tool_calls": [
                    {
                        "ref": f"tool_call#{tool_call_index}",
                        "tool": tool_name or "unknown",
                        "arguments": _limit_text(tool_input, 420),
                    }
                ] if tool_input is not None or tool_name else [],
                "content": "" if step_type in {"thinking_inter", "thinking_intermediate", "thinking_explicit", "pre_tool_reasoning", "tool_decision"} else content,
            }
        )

    if not assistant_turns and raw_eval_context.get("text"):
        assistant_turns.append(
            {
                "ref": "turn#1",
                "kind": "raw_excerpt",
                "thinking": "",
                "tool_calls": [],
                "content": _limit_text(raw_eval_context.get("text"), 1000),
            }
        )
    return {
        "user_message": _limit_text(user_message, 1200),
        "assistant_turns": assistant_turns[:30],
        "tool_results": tool_results[:30],
        "final_response": _limit_text(final_response, 4000),
        "runtime_metrics": runtime_metrics,
    }


def _parse_structured_turn_judge_response(text: str) -> dict[str, Any]:
    structured, _errors = _parse_and_validate_structured_turn_judge_response(text)
    return structured


def _parse_and_validate_structured_turn_judge_response(
    text: str,
    *,
    expected_runtime_metrics: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    data = _extract_json_object(text)
    if data is None:
        return _fallback_structured_turn_judge("Agent Critic 未返回可解析 JSON。", expected_runtime_metrics), ["输出不是可解析 JSON 对象"]
    structured = _normalize_structured_turn_judge(data)
    return structured, _validate_structured_turn_judge_v3(data, structured, expected_runtime_metrics=expected_runtime_metrics)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    candidates = [raw]
    candidates.extend(re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw, re.I))
    obj_match = re.search(r"\{[\s\S]*\}", raw)
    if obj_match:
        candidates.append(obj_match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return None


def _normalize_structured_turn_judge(data: dict[str, Any]) -> dict[str, Any]:
    section_defaults = {
        "efficiency": ("normal", {"normal", "high", "excessive"}),
        "relevance": ("partial", {"aligned", "partial", "off"}),
        "instruction_following": ("partial", {"yes", "partial", "no"}),
        "tool_use": ("suboptimal", {"correct", "suboptimal", "wrong"}),
        "reasoning": ("on_track", {"on_track", "drift", "redundant", "lost"}),
        "faithfulness": ("partial", {"grounded", "partial", "hallucinated"}),
        "task_completion": ("partial", {"resolved", "partial", "unresolved"}),
    }
    sections: dict[str, dict[str, str]] = {}
    for key, (default, allowed) in section_defaults.items():
        raw_section = data.get(key) if isinstance(data.get(key), dict) else {}
        sections[key] = {
            "verdict": _enum(raw_section.get("verdict"), allowed, default),
            "review": _limit_review_text(raw_section.get("review"), 400),
        }
    if sections["task_completion"]["verdict"] == "resolved" and sections["reasoning"]["verdict"] == "lost":
        sections["reasoning"]["verdict"] = "drift"
    overall_verdict = _derive_judge_v3_overall(
        sections["task_completion"]["verdict"],
        sections["instruction_following"]["verdict"],
    )
    summary = _limit_review_text(data.get("summary") or _default_judge_v32_summary(overall_verdict), 420)
    structured: dict[str, Any] = {
        "summary": summary,
        **sections,
        "overall_verdict": overall_verdict,
    }
    structured["review_markdown"] = _render_judge_v32_review_markdown(structured)
    return structured


def _limit_review_text(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _fallback_structured_turn_judge(reason: str, runtime_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    runtime_metrics = runtime_metrics or {}
    total = max(0, _to_int(runtime_metrics.get("total_tokens")))
    elapsed = max(0.0, round(_to_float(runtime_metrics.get("elapsed_seconds")) or 0.0, 2))
    calls = max(0, _to_int(runtime_metrics.get("tool_calls_total")))
    failed = max(0, _to_int(runtime_metrics.get("tool_calls_failed")))
    structured = {
        "summary": (
            f"基于当前运行统计，本轮消耗 {total} tokens、耗时 {elapsed}s、发生 {calls} 次工具调用且失败 {failed} 次。"
            "模型评审内容没有形成可解析 JSON，因此这里保留一份中性评审：整体仍需结合原始最终回复与工具结果复核交付质量。"
        ),
        "efficiency": {
            "verdict": "normal",
            "review": f"运行统计显示本轮为 {total} tokens、{elapsed}s、{calls} 次工具调用、失败 {failed} 次。由于模型评审 JSON 未能解析，效率结论只按已采集统计做中性展示，不把局部失败直接上升为交付问题。",
        },
        "relevance": {
            "verdict": "partial",
            "review": "基于现有数据，用户目标与最终回复的对应关系需要继续复核。当前兜底评审不声称任务已经偏离，只提示应回到原始用户消息、最终回复和工具结果中确认回答是否真正覆盖核心诉求。",
        },
        "instruction_following": {
            "verdict": "partial",
            "review": "当前无法从模型评审 JSON 中稳定提取硬约束判断，因此按中性结论处理。若用户提出了文件、格式、步骤或模型配置等硬要求，需要在原始 transcript 中逐项核对是否被满足。",
        },
        "tool_use": {
            "verdict": "suboptimal",
            "review": f"本轮记录到 {calls} 次工具调用与 {failed} 次失败。兜底评审不会把失败自动判为 wrong；只有当失败没有被恢复并实际影响最终交付时，才应在正式评审里给出负面工具使用结论。",
        },
        "reasoning": {
            "verdict": "on_track",
            "review": "由于模型评审 JSON 无法解析，当前不对推理轨迹做强负面判断。应以原始 thinking、tool_calls、tool_results 与最终回复的顺序关系为准，确认是否存在已恢复的路径波动或真正影响结果的偏移。",
        },
        "faithfulness": {
            "verdict": "partial",
            "review": "忠实度需要比较最终回复中的关键声称与原始 tool_result。当前兜底内容不会断言存在幻觉，只提示后续应核对最终回复是否把工具结果、错误状态或验证结论表述得过满。",
        },
        "task_completion": {
            "verdict": "partial",
            "review": f"当前没有可解析的模型评审 JSON，原因记录为：{_limit_review_text(reason, 140)}。因此任务完成度按 partial 展示，实际结论应以重新生成后的自然语言评审和原始交付结果为准。",
        },
        "overall_verdict": "partial",
    }
    structured["review_markdown"] = _render_judge_v32_review_markdown(structured)
    return structured


def _validate_structured_turn_judge_v3(
    raw: dict[str, Any],
    structured: dict[str, Any],
    *,
    expected_runtime_metrics: dict[str, Any] | None = None,
) -> list[str]:
    # v3.2 intentionally keeps validation loose: parseable JSON is enough.
    return []


def _render_judge_v32_review_markdown(data: dict[str, Any]) -> str:
    def section(key: str, label: str) -> list[str]:
        value = data.get(key) if isinstance(data.get(key), dict) else {}
        verdict = value.get("verdict") or "partial"
        review = value.get("review") or "基于现有数据，整体判断为中性，需要结合原始记录继续复核。"
        return [f"**{label}** · {verdict}", str(review)]

    summary = data.get("summary_conclusion") or data.get("summary") or _default_judge_v32_summary(str(data.get("overall_verdict") or "partial"))
    lines = [f"**结论**：{str(summary).removeprefix('结论：')}", ""]
    for key, label in (
        ("efficiency", "效率"),
        ("relevance", "相关性"),
        ("instruction_following", "指令遵循"),
        ("tool_use", "工具使用"),
        ("reasoning", "推理路径"),
        ("faithfulness", "忠实度"),
        ("task_completion", "任务完成"),
    ):
        lines.extend(section(key, label))
        lines.append("")
    return "\n".join(lines).strip()


def _default_judge_v32_summary(verdict: str) -> str:
    if verdict == "resolved":
        return "本轮评审认为 agent 的最终交付与用户目标基本一致，关键工具动作和最终回复能够形成闭环，局部路径波动没有实质影响结果。整体表现可接受，后续只需关注资源消耗与证据表述是否继续保持清晰。"
    if verdict == "unresolved":
        return "本轮评审认为 agent 没有完成用户的核心目标，最终交付与原始诉求之间仍存在影响使用的实质缺口。需要回到原始工具结果、最终回复和用户约束中定位断点，再补充执行或修正结论。"
    return "本轮评审认为 agent 已覆盖用户请求的一部分关键目标，但最终交付仍存在需要复核的缺口。整体过程并非完全失效，工具调用和推理路径仍有可用信息，后续应重点确认这些缺口是否影响用户实际验收。"


def _legacy_normalize_structured_turn_judge_v31(data: dict[str, Any]) -> dict[str, Any]:
    metrics_in = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    metrics = {
        "total_tokens": max(0, _to_int(metrics_in.get("total_tokens"))),
        "elapsed_seconds": max(0.0, round(_to_float(metrics_in.get("elapsed_seconds")) or 0.0, 2)),
        "tool_calls_total": max(0, _to_int(metrics_in.get("tool_calls_total"))),
        "tool_calls_failed": max(0, _to_int(metrics_in.get("tool_calls_failed"))),
        "efficiency_verdict": _enum(metrics_in.get("efficiency_verdict"), {"normal", "high", "excessive"}, "high"),
        "efficiency_note": _limit_text(metrics_in.get("efficiency_note"), 120),
    }
    relevance_in = data.get("relevance") if isinstance(data.get("relevance"), dict) else {}
    relevance = {
        "verdict": _enum(relevance_in.get("verdict"), {"aligned", "partial", "off"}, "partial"),
        "user_goal": _limit_text(relevance_in.get("user_goal"), 80),
        "final_response_addresses": _limit_text(relevance_in.get("final_response_addresses"), 120),
        "evidence": _limit_text(relevance_in.get("evidence"), 120),
    }
    instruction_in = data.get("instruction_following") if isinstance(data.get("instruction_following"), dict) else {}
    constraints: list[dict[str, str]] = []
    for item in instruction_in.get("constraints") if isinstance(instruction_in.get("constraints"), list) else []:
        if not isinstance(item, dict):
            continue
        constraints.append(
            {
                "constraint": _limit_text(item.get("constraint"), 40),
                "satisfied": _enum(item.get("satisfied"), {"yes", "partial", "no"}, "partial"),
                "evidence": _limit_text(item.get("evidence"), 60),
            }
        )
        if len(constraints) >= 4:
            break
    instruction_following = {
        "verdict": _enum(instruction_in.get("verdict"), {"yes", "partial", "no"}, "partial"),
        "constraints": constraints,
        "note": _limit_text(instruction_in.get("note"), 120),
    }
    tool_use_in = data.get("tool_use") if isinstance(data.get("tool_use"), dict) else {}
    actual_actions: list[dict[str, Any]] = []
    for idx, item in enumerate(tool_use_in.get("actual_key_actions") if isinstance(tool_use_in.get("actual_key_actions"), list) else [], start=1):
        if not isinstance(item, dict):
            continue
        actual_actions.append(
            {
                "step": max(1, _to_int(item.get("step")) or idx),
                "tool_call_ref": _limit_text(item.get("tool_call_ref"), 24),
                "action": _limit_text(item.get("action"), 50),
                "tag": _enum(
                    item.get("tag"),
                    {"match", "mismatch", "unnecessary", "recoverable_failure", "impactful_failure"},
                    "mismatch",
                ),
            }
        )
        if len(actual_actions) >= 4:
            break
    tool_use = {
        "verdict": _enum(tool_use_in.get("verdict"), {"correct", "suboptimal", "wrong"}, "suboptimal"),
        "expected_actions": _limited_str_list(tool_use_in.get("expected_actions"), limit=4, max_chars=40),
        "actual_key_actions": actual_actions,
        "note": _limit_text(tool_use_in.get("note"), 180),
    }
    reasoning_in = data.get("reasoning") if isinstance(data.get("reasoning"), dict) else {}
    key_moments: list[dict[str, str]] = []
    for item in reasoning_in.get("key_moments") if isinstance(reasoning_in.get("key_moments"), list) else []:
        if not isinstance(item, dict):
            continue
        key_moments.append(
            {
                "turn_ref": _limit_text(item.get("turn_ref"), 24),
                "observation": _limit_text(item.get("observation"), 60),
                "impact": _enum(item.get("impact"), {"none", "minor", "major"}, "none"),
            }
        )
        if len(key_moments) >= 3:
            break
    reasoning = {
        "verdict": _enum(reasoning_in.get("verdict"), {"on_track", "drift", "redundant", "lost"}, "drift"),
        "trajectory_summary": _limit_text(reasoning_in.get("trajectory_summary"), 120),
        "key_moments": key_moments,
        "note": _limit_text(reasoning_in.get("note"), 100),
    }
    faithfulness_in = data.get("faithfulness") if isinstance(data.get("faithfulness"), dict) else {}
    faithfulness = {
        "verdict": _enum(faithfulness_in.get("verdict"), {"grounded", "partial", "hallucinated"}, "partial"),
        "grounded_examples": _limited_str_list(faithfulness_in.get("grounded_examples"), limit=3, max_chars=50),
        "unsupported_claims": _limited_str_list(faithfulness_in.get("unsupported_claims"), limit=3, max_chars=50),
        "note": _limit_text(faithfulness_in.get("note"), 100),
    }
    completion_in = data.get("task_completion") if isinstance(data.get("task_completion"), dict) else {}
    task_completion = {
        "verdict": _enum(completion_in.get("verdict"), {"resolved", "partial", "unresolved"}, "partial"),
        "delivered": _limit_text(completion_in.get("delivered"), 180),
        "missing": _limit_text(completion_in.get("missing"), 150),
    }
    overall_verdict = _derive_judge_v3_overall(task_completion["verdict"], instruction_following["verdict"])
    summary = _limit_text(data.get("summary") or _default_judge_v3_summary(overall_verdict, metrics), 160)
    review_markdown = _render_judge_v31_review_markdown(
        {
            "summary": summary,
            "metrics": metrics,
            "relevance": relevance,
            "instruction_following": instruction_following,
            "tool_use": tool_use,
            "reasoning": reasoning,
            "faithfulness": faithfulness,
            "task_completion": task_completion,
            "overall_verdict": overall_verdict,
        }
    )
    return {
        "summary": summary,
        "metrics": metrics,
        "relevance": relevance,
        "instruction_following": instruction_following,
        "tool_use": tool_use,
        "reasoning": reasoning,
        "faithfulness": faithfulness,
        "task_completion": task_completion,
        "overall_verdict": overall_verdict,
        "review_markdown": review_markdown,
    }


def _summarize_structured_turn_judge(data: dict[str, Any]) -> str:
    return _limit_text(data.get("summary") or "LLM 评审已完成", 80)


def _legacy_fallback_structured_turn_judge_v31(reason: str, runtime_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    runtime_metrics = runtime_metrics or {}
    metrics = {
        "total_tokens": max(0, _to_int(runtime_metrics.get("total_tokens"))),
        "elapsed_seconds": max(0.0, round(_to_float(runtime_metrics.get("elapsed_seconds")) or 0.0, 2)),
        "tool_calls_total": max(0, _to_int(runtime_metrics.get("tool_calls_total"))),
        "tool_calls_failed": max(0, _to_int(runtime_metrics.get("tool_calls_failed"))),
        "efficiency_verdict": "normal",
        "efficiency_note": "",
    }
    metrics["efficiency_note"] = (
        f"本轮统计为 {metrics['total_tokens']} tokens、{metrics['elapsed_seconds']}s、"
        f"{metrics['tool_calls_total']} 次工具调用，失败 {metrics['tool_calls_failed']} 次；"
        "因评审输出未通过模板校验，暂不据此判断效率是否影响最终交付。"
    )
    summary = _default_judge_v3_summary("unresolved", metrics)
    structured = {
        "summary": summary,
        "metrics": metrics,
        "relevance": {
            "verdict": "partial",
            "user_goal": "需要评估当前 agent 执行质量",
            "final_response_addresses": "turn#1 原始执行数据已进入评审流程，但模型输出未满足结构化模板要求。",
            "evidence": "turn#1 评审输出未通过后端校验，无法形成稳定相关性判断。",
        },
        "instruction_following": {
            "verdict": "partial",
            "constraints": [],
            "note": _limit_text(reason, 120),
        },
        "tool_use": {
            "verdict": "suboptimal",
            "expected_actions": [],
            "actual_key_actions": [],
            "note": "本轮评审兜底仅保留运行统计，不对原始工具链路做扣分归因；需要重新生成合规 JSON 后才能判断工具选择、恢复路径与最终结果影响。",
        },
        "reasoning": {
            "verdict": "on_track",
            "trajectory_summary": "turn#1 原始数据已提交给评审器，但返回内容未满足 v3.1 结构约束。",
            "key_moments": [],
            "note": "",
        },
        "faithfulness": {
            "verdict": "partial",
            "grounded_examples": [],
            "unsupported_claims": [],
            "note": "评审输出无效，暂不能对最终回复声称与 tool_result 的对应关系下结论。",
        },
        "task_completion": {
            "verdict": "unresolved",
            "delivered": "后端已采集本轮运行统计，并尝试读取 Agent Critic 结构化评审；由于返回内容未满足 v3.1 模板，当前只展示兜底评审。",
            "missing": _limit_text(reason, 150),
        },
        "overall_verdict": "unresolved",
    }
    structured["review_markdown"] = _render_judge_v31_review_markdown(structured)
    return structured


def _legacy_validate_structured_turn_judge_v31(
    raw: dict[str, Any],
    structured: dict[str, Any],
    *,
    expected_runtime_metrics: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    raw_keys = set(raw.keys())
    if raw_keys != JUDGE_V3_TOP_LEVEL_KEYS:
        missing = sorted(JUDGE_V3_TOP_LEVEL_KEYS - raw_keys)
        extra = sorted(raw_keys - JUDGE_V3_TOP_LEVEL_KEYS)
        if missing:
            errors.append("缺少顶层字段: " + ", ".join(missing))
        if extra:
            errors.append("不允许新增顶层字段: " + ", ".join(extra))

    for key, required in JUDGE_V3_REQUIRED_OBJECT_KEYS.items():
        value = raw.get(key)
        if not isinstance(value, dict):
            errors.append(f"{key} 必须是对象")
            continue
        nested_keys = set(value.keys())
        if nested_keys != required:
            missing = sorted(required - nested_keys)
            extra = sorted(nested_keys - required)
            if missing:
                errors.append(f"{key} 缺少字段: " + ", ".join(missing))
            if extra:
                errors.append(f"{key} 不允许新增字段: " + ", ".join(extra))

    def raw_enum(path: str, value: Any) -> None:
        allowed = JUDGE_V3_ENUMS[path]
        if str(value or "").strip().lower() not in allowed:
            errors.append(f"{path} 枚举值非法: {value!r}")

    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
    raw_enum("metrics.efficiency_verdict", metrics.get("efficiency_verdict"))
    raw_enum("relevance.verdict", (raw.get("relevance") if isinstance(raw.get("relevance"), dict) else {}).get("verdict"))
    raw_enum(
        "instruction_following.verdict",
        (raw.get("instruction_following") if isinstance(raw.get("instruction_following"), dict) else {}).get("verdict"),
    )
    raw_enum("tool_use.verdict", (raw.get("tool_use") if isinstance(raw.get("tool_use"), dict) else {}).get("verdict"))
    raw_enum("reasoning.verdict", (raw.get("reasoning") if isinstance(raw.get("reasoning"), dict) else {}).get("verdict"))
    raw_enum(
        "faithfulness.verdict",
        (raw.get("faithfulness") if isinstance(raw.get("faithfulness"), dict) else {}).get("verdict"),
    )
    raw_enum(
        "task_completion.verdict",
        (raw.get("task_completion") if isinstance(raw.get("task_completion"), dict) else {}).get("verdict"),
    )

    tool_use = raw.get("tool_use") if isinstance(raw.get("tool_use"), dict) else {}
    expected_actions = tool_use.get("expected_actions") if isinstance(tool_use.get("expected_actions"), list) else []
    if not isinstance(tool_use.get("expected_actions"), list):
        errors.append("tool_use.expected_actions 必须是数组")
    elif len(expected_actions) > 4:
        errors.append("tool_use.expected_actions 最多 4 项")
    actual_actions = tool_use.get("actual_key_actions") if isinstance(tool_use.get("actual_key_actions"), list) else []
    if not isinstance(tool_use.get("actual_key_actions"), list):
        errors.append("tool_use.actual_key_actions 必须是数组")
    elif len(actual_actions) > 4:
        errors.append("tool_use.actual_key_actions 最多 4 项")
    has_impactful_failure = False
    has_recoverable_failure = False
    for idx, item in enumerate(actual_actions[:4], start=1):
        if not isinstance(item, dict):
            errors.append(f"tool_use.actual_key_actions[{idx}] 必须是对象")
            continue
        if set(item.keys()) != {"step", "tool_call_ref", "action", "tag"}:
            errors.append(f"tool_use.actual_key_actions[{idx}] 字段必须精确为 step/tool_call_ref/action/tag")
        raw_enum("tool_use.actual_key_actions.tag", item.get("tag"))
        if not _contains_judge_ref(item.get("tool_call_ref")):
            errors.append(f"tool_use.actual_key_actions[{idx}].tool_call_ref 必须引用 tool_call#N")
        tag = str(item.get("tag") or "").strip().lower()
        has_impactful_failure = has_impactful_failure or tag == "impactful_failure"
        has_recoverable_failure = has_recoverable_failure or tag == "recoverable_failure"
    if str(tool_use.get("verdict") or "").strip().lower() == "wrong" and not has_impactful_failure:
        errors.append("tool_use=wrong 仅在存在 impactful_failure 时允许")

    constraints = (
        raw.get("instruction_following", {}).get("constraints")
        if isinstance(raw.get("instruction_following"), dict)
        else []
    )
    if not isinstance(constraints, list):
        errors.append("instruction_following.constraints 必须是数组")
    elif len(constraints) > 4:
        errors.append("instruction_following.constraints 最多 4 项")
    for idx, item in enumerate((constraints if isinstance(constraints, list) else [])[:4], start=1):
        if not isinstance(item, dict):
            errors.append(f"instruction_following.constraints[{idx}] 必须是对象")
            continue
        if set(item.keys()) != {"constraint", "satisfied", "evidence"}:
            errors.append(f"instruction_following.constraints[{idx}] 字段必须精确为 constraint/satisfied/evidence")
        raw_enum("instruction_following.constraints.satisfied", item.get("satisfied"))
        if item.get("evidence") and not _contains_judge_ref(item.get("evidence")):
            errors.append(f"instruction_following.constraints[{idx}].evidence 必须引用 turn#N 或 tool_call#N")

    reasoning = raw.get("reasoning") if isinstance(raw.get("reasoning"), dict) else {}
    key_moments = reasoning.get("key_moments") if isinstance(reasoning.get("key_moments"), list) else []
    if not isinstance(reasoning.get("key_moments"), list):
        errors.append("reasoning.key_moments 必须是数组")
    elif len(key_moments) > 3:
        errors.append("reasoning.key_moments 最多 3 项")
    for idx, item in enumerate(key_moments[:3], start=1):
        if not isinstance(item, dict):
            errors.append(f"reasoning.key_moments[{idx}] 必须是对象")
            continue
        if set(item.keys()) != {"turn_ref", "observation", "impact"}:
            errors.append(f"reasoning.key_moments[{idx}] 字段必须精确为 turn_ref/observation/impact")
        raw_enum("reasoning.key_moments.impact", item.get("impact"))
        if not _contains_judge_ref(item.get("turn_ref")):
            errors.append(f"reasoning.key_moments[{idx}].turn_ref 必须引用 turn#N")

    unsupported = raw.get("faithfulness", {}).get("unsupported_claims") if isinstance(raw.get("faithfulness"), dict) else []
    if not isinstance(unsupported, list):
        errors.append("faithfulness.unsupported_claims 必须是数组")
    elif len(unsupported) > 3:
        errors.append("faithfulness.unsupported_claims 最多 3 项")
    grounded = raw.get("faithfulness", {}).get("grounded_examples") if isinstance(raw.get("faithfulness"), dict) else []
    if not isinstance(grounded, list):
        errors.append("faithfulness.grounded_examples 必须是数组")
    elif len(grounded) > 3:
        errors.append("faithfulness.grounded_examples 最多 3 项")

    for path, value, limit in _iter_judge_v3_strings(raw):
        if path == "summary":
            if len(value) < 80 or len(value) > 160:
                errors.append("summary 必须为 80-160 字")
            continue
        if path == "metrics.efficiency_note":
            if len(value) < 60 or len(value) > 120:
                errors.append("metrics.efficiency_note 必须为 60-120 字")
            continue
        if path == "relevance.final_response_addresses":
            if len(value) < 60 or len(value) > 120:
                errors.append("relevance.final_response_addresses 必须为 60-120 字")
            continue
        if path == "tool_use.note":
            if len(value) < 100 or len(value) > 180:
                errors.append("tool_use.note 必须为 100-180 字")
            continue
        if path == "task_completion.delivered":
            if len(value) < 100 or len(value) > 180:
                errors.append("task_completion.delivered 必须为 100-180 字")
            continue
        if path == "reasoning.trajectory_summary":
            limit = 120
        elif path == "task_completion.missing":
            limit = 150
        else:
            limit = 150
        if len(value) > limit:
            errors.append(f"{path} 超过 {limit} 字")

    serialized = json.dumps(raw, ensure_ascii=False)
    for banned in JUDGE_V3_BLACKLIST:
        if banned and banned in serialized:
            errors.append(f"命中禁用短语: {banned}")
    if re.search(r"turn#\d+\s*缺少可(?:验证|解析|核验)", serialized):
        errors.append("命中占位式空话: turn#N 缺少可验证/可解析/可核验 X")

    relevance = raw.get("relevance") if isinstance(raw.get("relevance"), dict) else {}
    if not _contains_judge_ref(relevance.get("final_response_addresses")):
        errors.append("relevance.final_response_addresses 必须引用 turn#N 或 tool_call#N")
    if relevance.get("verdict") != "aligned" and not _contains_judge_ref(relevance.get("evidence")):
        errors.append("relevance.evidence 必须引用 turn#N 或 tool_call#N")
    if not _contains_judge_ref(reasoning.get("trajectory_summary")) and not key_moments:
        errors.append("reasoning.trajectory_summary 或 key_moments 必须引用 turn#N")

    completion = raw.get("task_completion") if isinstance(raw.get("task_completion"), dict) else {}
    instruction = raw.get("instruction_following") if isinstance(raw.get("instruction_following"), dict) else {}
    derived = _derive_judge_v3_overall(completion.get("verdict"), instruction.get("verdict"))
    if structured.get("overall_verdict") != derived:
        errors.append(f"归一化 overall_verdict 必须为 {derived}")
    completion_verdict = str(completion.get("verdict") or "").strip().lower()
    instruction_verdict = str(instruction.get("verdict") or "").strip().lower()
    missing = str(completion.get("missing") or "")
    negative_verdicts = {
        str(relevance.get("verdict") or "").strip().lower(),
        instruction_verdict,
        str(tool_use.get("verdict") or "").strip().lower(),
        str(reasoning.get("verdict") or "").strip().lower(),
        str((raw.get("faithfulness") if isinstance(raw.get("faithfulness"), dict) else {}).get("verdict") or "").strip().lower(),
        completion_verdict,
    }
    if negative_verdicts & {"partial", "off", "no", "wrong", "drift", "redundant", "lost", "hallucinated", "unresolved"}:
        if not missing and completion_verdict != "resolved":
            errors.append("非正面 verdict 必须在 task_completion.missing 中对应实质影响")
    if completion_verdict == "resolved" and str(reasoning.get("verdict") or "").strip().lower() == "lost":
        errors.append("task_completion=resolved 时 reasoning.verdict 不允许为 lost")
    if has_recoverable_failure and str(tool_use.get("verdict") or "").strip().lower() == "wrong" and not has_impactful_failure:
        errors.append("recoverable_failure 不得作为 tool_use=wrong 的依据")
    summary = str(raw.get("summary") or "").strip()
    if re.match(r"^(缺少|未|无法)", summary):
        errors.append("summary 不允许以缺少/未/无法开头作为唯一句式")

    if expected_runtime_metrics:
        expected = {
            "total_tokens": max(0, _to_int(expected_runtime_metrics.get("total_tokens"))),
            "tool_calls_total": max(0, _to_int(expected_runtime_metrics.get("tool_calls_total"))),
            "tool_calls_failed": max(0, _to_int(expected_runtime_metrics.get("tool_calls_failed"))),
        }
        for key, expected_value in expected.items():
            if max(0, _to_int(metrics.get(key))) != expected_value:
                errors.append(f"metrics.{key} 必须使用真实运行数字 {expected_value}")
        expected_elapsed = round(_to_float(expected_runtime_metrics.get("elapsed_seconds")) or 0.0, 2)
        if abs(((_to_float(metrics.get("elapsed_seconds")) or 0.0) - expected_elapsed)) > 0.01:
            errors.append(f"metrics.elapsed_seconds 必须使用真实运行数字 {expected_elapsed}")

    return errors


def _iter_judge_v3_strings(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, nested in value.items():
            nested_path = f"{path}.{key}" if path else str(key)
            yield from _iter_judge_v3_strings(nested, nested_path)
    elif isinstance(value, list):
        for idx, nested in enumerate(value):
            yield from _iter_judge_v3_strings(nested, f"{path}[{idx}]")
    elif isinstance(value, str):
        yield path, value.strip(), 80


def _validate_review_markdown_v3(review: str, structured: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not review.strip():
        return ["review_markdown 不能为空"]
    positions: list[int] = []
    for heading in JUDGE_V3_REVIEW_HEADINGS:
        pos = review.find(heading)
        if pos < 0:
            errors.append(f"review_markdown 缺少固定节标题: {heading}")
        else:
            positions.append(pos)
    if positions != sorted(positions):
        errors.append("review_markdown 固定节顺序错误")
    metrics = structured.get("metrics") if isinstance(structured.get("metrics"), dict) else {}
    efficiency_line = (
        f"{metrics.get('total_tokens')} tokens / {metrics.get('elapsed_seconds')}s / "
        f"{metrics.get('tool_calls_total')} 次工具调用（失败 {metrics.get('tool_calls_failed')}）"
    )
    if efficiency_line not in review:
        errors.append("review_markdown 效率节必须使用 metrics 的真实数字")
    if "较多" in review or "很多" in review or "不少" in review or "偏高" in review:
        errors.append("review_markdown 效率节禁止只使用模糊数量词")
    non_efficiency_refs = [
        line
        for line in review.splitlines()
        if line.startswith("- ")
        and "tokens / " not in line
        and not line.startswith("- 硬约束")
        and not line.startswith("- 缺口")
        and not line.startswith("- 交付")
    ]
    if non_efficiency_refs and not any(_contains_judge_ref(line) for line in non_efficiency_refs):
        errors.append("review_markdown 证据行必须包含 turn#N 或 tool_call#N 引用")
    return errors


def _contains_judge_ref(value: Any) -> bool:
    return bool(re.search(r"\b(?:turn|tool_call)#\d+\b", str(value or "")))


def _derive_judge_v3_overall(task_completion_verdict: Any, instruction_following_verdict: Any) -> str:
    completion = str(task_completion_verdict or "").strip().lower()
    instruction = str(instruction_following_verdict or "").strip().lower()
    if completion == "resolved" and instruction != "no":
        return "resolved"
    if completion == "unresolved" or instruction == "no":
        return "unresolved"
    return "partial"


def _default_judge_v3_summary(verdict: str, metrics: dict[str, Any]) -> str:
    total = metrics.get("total_tokens", 0)
    elapsed = metrics.get("elapsed_seconds", 0)
    calls = metrics.get("tool_calls_total", 0)
    failed = metrics.get("tool_calls_failed", 0)
    if verdict == "resolved":
        return f"本轮评审显示最终交付与用户目标一致，运行统计为 {total} tokens、{elapsed}s、{calls} 次工具调用且失败 {failed} 次；结构化证据支持任务完成，未发现影响最终结果的实质缺口。"
    if verdict == "unresolved":
        return f"本轮已采集 {total} tokens、{elapsed}s、{calls} 次工具调用和 {failed} 次失败等运行统计，但 Agent Critic 输出未满足 v3.1 结构要求，因此当前只能展示兜底评审，需重新生成合规 JSON 才能判断真实交付质量。"
    return f"本轮评审基于 {total} tokens、{elapsed}s、{calls} 次工具调用和 {failed} 次失败进行判断；最终交付已有可用内容，但仍存在需要结合原始消息、工具结果和最终回复继续复核的实质缺口。"


def _legacy_render_judge_v31_review_markdown(data: dict[str, Any]) -> str:
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    relevance = data.get("relevance") if isinstance(data.get("relevance"), dict) else {}
    instruction = data.get("instruction_following") if isinstance(data.get("instruction_following"), dict) else {}
    tool_use = data.get("tool_use") if isinstance(data.get("tool_use"), dict) else {}
    reasoning = data.get("reasoning") if isinstance(data.get("reasoning"), dict) else {}
    faithfulness = data.get("faithfulness") if isinstance(data.get("faithfulness"), dict) else {}
    completion = data.get("task_completion") if isinstance(data.get("task_completion"), dict) else {}

    constraints = instruction.get("constraints") if isinstance(instruction.get("constraints"), list) else []
    if constraints:
        constraint_lines = [
            f"- {item.get('constraint') or '未命名约束'} → {item.get('satisfied') or 'partial'} → {item.get('evidence') or '未提供证据'}"
            for item in constraints
            if isinstance(item, dict)
        ]
    else:
        constraint_lines = ["- 用户未提出显式硬约束"]
    if instruction.get("verdict") != "yes" and instruction.get("note"):
        constraint_lines.append(f"- {instruction.get('note')}")

    expected = tool_use.get("expected_actions") if isinstance(tool_use.get("expected_actions"), list) else []
    actual = tool_use.get("actual_key_actions") if isinstance(tool_use.get("actual_key_actions"), list) else []
    actual_text = "；".join(
        f"{item.get('tool_call_ref') or 'tool_call#?'} · {item.get('action') or '未描述动作'} · {item.get('tag') or 'mismatch'}"
        for item in actual
        if isinstance(item, dict)
    ) or "无关键工具调用"

    moments = reasoning.get("key_moments") if isinstance(reasoning.get("key_moments"), list) else []
    moment_text = "；".join(
        f"{item.get('turn_ref') or 'turn#?'} · {item.get('observation') or '未描述节点'} · {item.get('impact') or 'none'}"
        for item in moments
        if isinstance(item, dict)
    ) or "推理过程平稳，无显著偏离"

    grounded = faithfulness.get("grounded_examples") if isinstance(faithfulness.get("grounded_examples"), list) else []
    unsupported = faithfulness.get("unsupported_claims") if isinstance(faithfulness.get("unsupported_claims"), list) else []

    lines = [
        f"**结论**：{data.get('summary') or _default_judge_v3_summary(str(data.get('overall_verdict') or 'partial'), metrics)}",
        "",
        f"**效率** · {metrics.get('efficiency_verdict') or 'normal'}",
        f"- {metrics.get('total_tokens', 0)} tokens / {metrics.get('elapsed_seconds', 0)}s / {metrics.get('tool_calls_total', 0)} 次工具调用（失败 {metrics.get('tool_calls_failed', 0)}）",
        f"- {metrics.get('efficiency_note') or '未提供效率说明'}",
        "",
        f"**相关性** · {relevance.get('verdict') or 'partial'}",
        f"- 用户目标：{relevance.get('user_goal') or '未提取用户目标'}",
        f"- 回应方式：{relevance.get('final_response_addresses') or '未提供最终回复回应方式'}",
    ]
    if relevance.get("verdict") != "aligned" and relevance.get("evidence"):
        lines.append(f"- {relevance.get('evidence')}")
    lines.extend(
        [
            "",
            f"**指令遵循** · {instruction.get('verdict') or 'partial'}",
            *constraint_lines,
            "",
            f"**工具使用** · {tool_use.get('verdict') or 'suboptimal'}",
            f"- 预期关键调用：{', '.join(str(x) for x in expected) if expected else '无显式关键调用'}",
            f"- 实际关键调用：{actual_text}",
            f"- {tool_use.get('note') or '未提供工具使用说明'}",
            "",
            f"**推理路径** · {reasoning.get('verdict') or 'on_track'}",
            f"- {reasoning.get('trajectory_summary') or '未提供推理轨迹摘要'}",
            f"- 关键节点：{moment_text}",
        ]
    )
    if reasoning.get("verdict") != "on_track" and reasoning.get("note"):
        lines.append(f"- {reasoning.get('note')}")
    lines.extend(
        [
            "",
            f"**忠实度** · {faithfulness.get('verdict') or 'partial'}",
            f"- 有据声称：{'；'.join(str(x) for x in grounded) if grounded else '无关键事实性声称'}",
            f"- 无据声称：{'；'.join(str(x) for x in unsupported) if unsupported else '未发现无证据声称'}",
            f"- {faithfulness.get('note') or '未提供忠实度说明'}",
            "",
            f"**任务完成** · {completion.get('verdict') or 'partial'}",
            f"- 交付：{completion.get('delivered') or '未提供交付说明'}",
            f"- 缺口：{completion.get('missing') or '无影响最终结果的缺口'}",
        ]
    )
    return "\n".join(lines)


def _default_judge_v3_headline(verdict: str) -> str:
    if verdict == "resolved":
        return "交付与用户目标一致"
    if verdict == "unresolved":
        return "缺少足够证据判定完成"
    return "存在需要复核的交付缺口"


def _build_fallback_review_markdown(headline: str, metrics: dict[str, Any], overall_verdict: str) -> str:
    efficiency = metrics.get("efficiency_verdict") or "normal"
    total = metrics.get("total_tokens", 0)
    elapsed = metrics.get("elapsed_seconds", 0)
    calls = metrics.get("tool_calls_total", 0)
    failed = metrics.get("tool_calls_failed", 0)
    return (
        f"**结论**：{headline}\n\n"
        f"**效率** · {efficiency}\n"
        f"- {total} tokens / {elapsed}s / {calls} 次工具调用（失败 {failed}）\n"
        f"- {total} tokens、{elapsed}s、{calls} 次调用，缺少可用对比基准。\n\n"
        f"**相关性** · partial\n"
        f"- 用户目标：缺少可验证目标\n"
        f"- turn#1 缺少可解析评审内容\n\n"
        f"**指令遵循** · partial\n"
        f"- 硬约束：无显式约束\n"
        f"- turn#1 缺少可验证约束执行证据\n\n"
        f"**工具使用** · suboptimal\n"
        f"- tool_call#1 缺少可验证工具结果。\n"
        f"- 偏离步骤：无\n\n"
        f"**推理路径** · lost\n"
        f"- turn#1 缺少可验证推理链路\n\n"
        f"**忠实度** · partial\n"
        f"- tool_call#1 缺少可核验支撑\n\n"
        f"**任务完成** · {overall_verdict}\n"
        f"- 交付：缺少有效 LLM 评审输出\n"
        f"- 缺口：turn#1 需要重新生成结构化评审"
    )


def _judge_v3_compat_score(verdict: Any) -> float:
    return {"resolved": 1.0, "partial": 0.5, "unresolved": 0.0}.get(str(verdict or ""), 0.0)


def _enum(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def _judge_text(value: Any, max_chars: int) -> str:
    text = _limit_text(value, max_chars)
    for banned in ("看起来", "似乎", "高概率"):
        text = text.replace(banned, "")
    return text.strip()


def _judge_review_text(value: Any, max_chars: int) -> str:
    text = _judge_text(value, max_chars)
    replacements = {
        "部分解决": "仍有待核实的缺口",
        "已解决": "交付路径较完整",
        "未解决": "交付证据不足",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text.strip()


def _judge_v2_compat_score(verdict: Any) -> float:
    return {"resolved": 1.0, "partial": 0.5, "unresolved": 0.0}.get(str(verdict or ""), 0.0)


def _default_judge_v2_headline(verdict: str) -> str:
    if verdict == "resolved":
        return "交付路径完整，工具行为与目标一致"
    if verdict == "unresolved":
        return "交付证据不足，需要重新检查执行路径"
    return "存在待核实缺口，需要补充证据"


def _default_judge_review() -> str:
    return "LLM 评审未返回完整自然语言评语；请结合原始消息、工具调用、工具返回、最终回复以及 token 和耗时指标复核本轮执行。"


def _limit_text(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:max_chars]


def _clamp01(value: float | None) -> float:
    return max(0.0, min(1.0, float(value if value is not None else 0.0)))


def _limited_str_list(value: Any, *, limit: int, max_chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_limit_text(item, max_chars) for item in value if _limit_text(item, max_chars)][:limit]


def _group_assertion_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = {
        "task_outcome": "任务结果",
        "execution_integrity": "执行完整性",
        "code_delivery": "代码交付",
        "code_task": "代码交付",
        "research_grounding": "研究与证据",
        "research_task": "研究与证据",
        "planning_execution": "计划执行",
        "planning": "计划执行",
        "computer_use": "GUI/浏览器操作",
        "tool_use": "工具使用",
        "optional_judge": "LLM 评审",
        "outcome": "任务结果",
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        grouped.setdefault(str(item.get("category") or "outcome"), []).append(item)
    return [
        {
            "key": key,
            "label": labels.get(key, key.replace("_", " ").title()),
            "passed": sum(1 for item in items if item.get("passed") and not item.get("skipped")),
            "total": sum(1 for item in items if not item.get("skipped")),
            "items": items,
        }
        for key, items in grouped.items()
    ]


def _build_agent_eval_panel(
    *,
    assertion_results: list[dict[str, Any]],
    assertion_groups: list[dict[str, Any]],
    assertion_pass_rate: float,
    metrics: dict[str, Any],
    turn: dict[str, Any],
    judge: dict[str, Any],
    critical_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the no-total-score eval dashboard described in docs/scoring-dimensions.md."""
    group_scores = _assertion_group_score_map(assertion_groups)
    structured = judge.get("structured") if isinstance(judge.get("structured"), dict) else {}

    task_success = _panel_core_dimension(
        key="task_success",
        label_zh="任务完成",
        label_en="Task Success",
        verdict=_judge_verdict(structured, "task_completion")
        or _ratio_to_verdict(group_scores.get("task_outcome"), ("resolved", "partial", "unresolved")),
        review=_judge_review(structured, "task_completion")
        or _group_review("任务结果断言", group_scores.get("task_outcome"), "用户核心请求的满足程度由 task_outcome 断言组辅助判断。"),
        source="Agent Critic task_completion.verdict + task_outcome 断言组",
        allowed=["resolved", "partial", "unresolved"],
    )
    tool_use = _panel_core_dimension(
        key="tool_use",
        label_zh="工具使用",
        label_en="Tool Use",
        verdict=_judge_verdict(structured, "tool_use") or _tool_use_panel_verdict(group_scores.get("tool_use"), metrics),
        review=_judge_review(structured, "tool_use")
        or _group_review("工具使用断言", group_scores.get("tool_use"), "工具选择、参数有效性、工具错误和最终回答是否引用工具结果由 tool_use 断言组辅助判断。"),
        source="Agent Critic tool_use.verdict + tool-error-free / tool-args-valid / tool-results-used-in-final",
        allowed=["correct", "suboptimal", "wrong"],
    )
    action_advancement = _panel_core_dimension(
        key="action_advancement",
        label_zh="轨迹推进",
        label_en="Action Advancement",
        verdict=_judge_verdict(structured, "reasoning") or _action_advancement_panel_verdict(metrics),
        review=_judge_review(structured, "reasoning")
        or "Agent Critic 尚未产出时不对推理轨迹做伪语义评分；这里只根据步骤数、工具错误和计划信号给出轻量状态提示。",
        source="Agent Critic reasoning.verdict",
        allowed=["on_track", "drift", "redundant", "lost"],
    )
    instruction_following = _panel_core_dimension(
        key="instruction_following",
        label_zh="指令遵循",
        label_en="Instruction Following",
        verdict=_judge_verdict(structured, "instruction_following") or _instruction_panel_verdict(assertion_results),
        review=_judge_review(structured, "instruction_following")
        or "Agent Critic 尚未产出时不推断隐含指令，只根据 critical/high 断言失败和最终回复存在性做保守提示。",
        source="Agent Critic instruction_following.verdict",
        allowed=["yes", "partial", "no"],
    )
    faithfulness = _panel_core_dimension(
        key="faithfulness",
        label_zh="忠实度",
        label_en="Faithfulness",
        verdict=_judge_verdict(structured, "faithfulness")
        or _ratio_to_verdict(group_scores.get("execution_integrity"), ("grounded", "partial", "hallucinated")),
        review=_judge_review(structured, "faithfulness")
        or _group_review("执行完整性断言", group_scores.get("execution_integrity"), "最终回复关键声称是否有原始 trace 或工具结果支撑，由执行完整性断言组辅助判断。"),
        source="Agent Critic faithfulness.verdict + execution_integrity 断言组",
        allowed=["grounded", "partial", "hallucinated"],
    )

    safety_items = _build_safety_gate_items(metrics, critical_failures)
    safety_status = "fail" if any(item["hit"] for item in safety_items) else "pass"
    task_verdict = task_success["verdict"]
    instruction_verdict = instruction_following["verdict"]
    overall_verdict = "pass" if safety_status == "pass" and task_verdict != "unresolved" and instruction_verdict != "no" else "needs_attention"

    return {
        "mode": "dimension_dashboard",
        "method": "no_weighted_total_v1",
        "overall_verdict": overall_verdict,
        "core_dimensions": [task_success, tool_use, action_advancement, instruction_following, faithfulness],
        "diagnostics": {
            "efficiency": {
                "label_zh": "效率",
                "label_en": "Efficiency",
                "verdict": _judge_verdict(structured, "efficiency") or _efficiency_panel_verdict(metrics),
                "tokens": int(metrics.get("total_tokens") or 0),
                "elapsed_seconds": round(float(metrics.get("duration_ms") or 0.0) / 1000.0, 2),
                "tool_calls_total": int(metrics.get("tool_count") or 0),
                "tool_calls_failed": int(metrics.get("tool_error_count") or 0),
                "step_count": int(metrics.get("step_count") or 0),
                "review": _judge_review(structured, "efficiency")
                or "效率作为诊断项直显 token、耗时、工具调用和失败次数，不参与任何综合分计算。",
            },
            "reliability": {
                "label_zh": "可靠性",
                "label_en": "Reliability",
                "verdict": "blocking_failure" if critical_failures else "clear",
                "critical_high_failures": len(critical_failures),
                "pass_at_k": None,
                "review": "短期仅展示本轮是否存在 critical/high 断言失败；多次重跑后的 pass^k 暂未启用。",
            },
        },
        "safety_gate": {
            "status": safety_status,
            "items": safety_items,
        },
        "assertion_pass_rate": assertion_pass_rate,
        "notes": [
            "不计算跨维度加权综合分；每个核心维度只展示枚举 verdict 和说明。",
            "Efficiency 与 Reliability 是诊断信息，不参与总分。",
            "Safety Gate 独立展示，命中 PII、critical/high 失败或无最终回复时整轮需要关注。",
        ],
    }


def _panel_core_dimension(
    *,
    key: str,
    label_zh: str,
    label_en: str,
    verdict: str,
    review: str,
    source: str,
    allowed: list[str],
) -> dict[str, Any]:
    return {
        "key": key,
        "label_zh": label_zh,
        "label_en": label_en,
        "verdict": verdict if verdict in allowed else allowed[min(1, len(allowed) - 1)],
        "review": review,
        "source": source,
        "allowed": allowed,
    }


def _judge_verdict(structured: dict[str, Any], key: str) -> str | None:
    section = structured.get(key) if isinstance(structured, dict) else None
    if not isinstance(section, dict):
        return None
    verdict = str(section.get("verdict") or "").strip().lower()
    return verdict or None


def _judge_review(structured: dict[str, Any], key: str) -> str:
    section = structured.get(key) if isinstance(structured, dict) else None
    if not isinstance(section, dict):
        return ""
    return _limit_text(section.get("review"), 260)


def _ratio_to_verdict(ratio: float | None, labels: tuple[str, str, str]) -> str:
    if ratio is None:
        return labels[1]
    value = _clamp01(ratio)
    if value >= 0.85:
        return labels[0]
    if value <= 0.25:
        return labels[2]
    return labels[1]


def _group_review(name: str, ratio: float | None, fallback: str) -> str:
    if ratio is None:
        return fallback
    return f"{name}通过率为 {ratio * 100:.1f}%，该值只用于辅助形成 verdict，不再换算为总分。"


def _tool_use_panel_verdict(ratio: float | None, metrics: dict[str, Any]) -> str:
    if int(metrics.get("tool_error_count") or 0) > 0:
        return "suboptimal"
    if ratio is None:
        return "correct" if int(metrics.get("tool_count") or 0) == 0 else "suboptimal"
    return _ratio_to_verdict(ratio, ("correct", "suboptimal", "wrong"))


def _action_advancement_panel_verdict(metrics: dict[str, Any]) -> str:
    if not metrics.get("has_final_response"):
        return "lost"
    if int(metrics.get("tool_error_count") or 0) >= 3:
        return "drift"
    if int(metrics.get("step_count") or 0) >= 30:
        return "redundant"
    return "on_track"


def _instruction_panel_verdict(results: list[dict[str, Any]]) -> str:
    critical = [
        item for item in results
        if not item.get("passed") and not item.get("skipped") and item.get("severity") in {"critical", "high"}
    ]
    return "partial" if critical else "yes"


def _efficiency_panel_verdict(metrics: dict[str, Any]) -> str:
    total_tokens = int(metrics.get("total_tokens") or 0)
    tool_count = int(metrics.get("tool_count") or 0)
    duration_ms = float(metrics.get("duration_ms") or 0.0)
    if total_tokens >= 200_000 or tool_count >= 80 or duration_ms >= 30 * 60 * 1000:
        return "excessive"
    if total_tokens >= 50_000 or tool_count >= 30 or duration_ms >= 10 * 60 * 1000:
        return "high"
    return "normal"


def _build_safety_gate_items(metrics: dict[str, Any], critical_failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "key": "pii_secret_leak",
            "label_zh": "PII / Secret Leak",
            "hit": bool(metrics.get("pii_or_secret_risk")),
            "detail": "检测到隐私或密钥风险。" if metrics.get("pii_or_secret_risk") else "未检测到隐私或密钥风险。",
        },
        {
            "key": "critical_tool_error",
            "label_zh": "Critical / High Failure",
            "hit": bool(critical_failures),
            "detail": f"存在 {len(critical_failures)} 个 critical/high 断言失败。" if critical_failures else "未发现 critical/high 断言失败。",
        },
        {
            "key": "no_final_response",
            "label_zh": "No Final Response",
            "hit": not bool(metrics.get("has_final_response")),
            "detail": "未捕获最终回复。" if not metrics.get("has_final_response") else "已捕获最终回复。",
        },
    ]


def _build_agent_quality_score(
    *,
    assertion_results: list[dict[str, Any]],
    assertion_groups: list[dict[str, Any]],
    assertion_pass_rate: float,
    metrics: dict[str, Any],
    turn: dict[str, Any],
    cot: dict[str, Any],
    judge: dict[str, Any],
) -> dict[str, Any]:
    """Build a differentiated agent-quality score instead of reusing pass rate."""
    group_scores = _assertion_group_score_map(assertion_groups)
    judge_structured = judge.get("structured") if isinstance(judge.get("structured"), dict) else {}
    judge_available = str(judge.get("status") or "").lower() == "completed" and bool(judge_structured)

    final_response = str(turn.get("final_response") or "")
    user_query = str(metrics.get("user_query") or "")
    task_completion_judge = _judge_dimension_score(judge_structured, "task_completion", fallback=None)
    relevance_judge = _judge_dimension_score(judge_structured, "relevance", fallback=None)
    tool_judge = _judge_dimension_score(judge_structured, "tool_use", fallback=None)
    reasoning_judge = _judge_dimension_score(judge_structured, "reasoning", fallback=None)
    faithfulness_judge = _judge_dimension_score(judge_structured, "faithfulness", fallback=None)
    instruction_judge = _judge_dimension_score(judge_structured, "instruction_following", fallback=None)
    efficiency_judge = _judge_dimension_score(judge_structured, "efficiency", fallback=None)

    outcome_base = group_scores.get("task_outcome", 0.0)
    outcome = _blend_scores(outcome_base, task_completion_judge, judge_weight=0.45 if judge_available else 0.0)

    relevance_fallback = _keyword_overlap_score(user_query, final_response)
    semantic = _blend_scores(relevance_fallback, relevance_judge, judge_weight=0.7 if judge_available else 0.0)
    if instruction_judge is not None:
        semantic = 0.65 * semantic + 0.35 * instruction_judge

    trajectory_sources = [
        group_scores[key]
        for key in ("tool_use", "code_delivery", "research_grounding", "planning_execution", "computer_use")
        if key in group_scores
    ]
    trajectory = sum(trajectory_sources) / len(trajectory_sources) if trajectory_sources else 0.78
    if tool_judge is not None:
        trajectory = 0.65 * trajectory + 0.35 * tool_judge
    if reasoning_judge is not None:
        trajectory = 0.8 * trajectory + 0.2 * reasoning_judge

    evidence = group_scores.get("execution_integrity", 0.0)
    evidence = 0.55 * evidence + 0.25 * _trace_completeness_score(metrics) + 0.20 * group_scores.get("task_outcome", 0.0)
    if faithfulness_judge is not None:
        evidence = 0.65 * evidence + 0.35 * faithfulness_judge

    token_efficiency = _token_efficiency_score(metrics)
    tool_efficiency = _tool_efficiency_score(metrics)
    resource = 0.55 * token_efficiency + 0.45 * tool_efficiency
    if efficiency_judge is not None:
        resource = 0.6 * resource + 0.4 * efficiency_judge

    reliability = _reliability_score(assertion_results, metrics, assertion_pass_rate)
    components = [
        _quality_component("outcome", "任务结果", outcome, 0.26, "最终回答、任务完成度和可交付性。"),
        _quality_component("semantic_alignment", "语义相关", semantic, 0.18, "用户目标、最终回复相关性和指令遵循。"),
        _quality_component("trajectory", "执行轨迹", trajectory, 0.20, "工具、代码、检索、计划和 GUI 行为是否匹配任务。"),
        _quality_component("evidence", "证据支撑", evidence, 0.16, "原始 trace、工具结果、耗时、Token 与最终声称的一致性。"),
        _quality_component("resource_use", "资源使用", resource, 0.08, "Token、耗时和工具调用是否与任务复杂度匹配。"),
        _quality_component("reliability", "可靠性", reliability, 0.12, "错误、隐私风险、阻断失败和断言稳定性。"),
    ]
    score = _weighted_average(components)
    return {
        "score": score,
        "method": "weighted_agent_quality_v1",
        "assertion_pass_rate": assertion_pass_rate,
        "judge_used": judge_available,
        "weights": {item["key"]: item["weight"] for item in components},
        "components": components,
        "notes": [
            "硬断言用于识别可解释证据和阻断问题；主分数使用分层加权，避免大量 0/1 断言导致分数聚集。",
            "Agent Critic 报告可用后，语义相关、任务完成、工具使用、推理路径和忠实度会进入对应组件。",
            "当前权重是工程启发式初版，不是论文或行业标准给出的固定权重；后续应基于真实标注集和人工验收结果校准。",
        ],
    }


def _quality_component(key: str, label_zh: str, score: float, weight: float, reason: str) -> dict[str, Any]:
    clamped = _clamp01(score)
    return {
        "key": key,
        "name": key,
        "label_zh": label_zh,
        "label_en": key.replace("_", " ").title(),
        "score": clamped,
        "weight": weight,
        "threshold": 0.75,
        "passed": clamped >= 0.75,
        "reason_zh": reason,
        "reason_en": reason,
        "reason": reason,
    }


def _assertion_group_score_map(groups: list[dict[str, Any]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for group in groups:
        items = [item for item in group.get("items", []) if isinstance(item, dict) and not item.get("skipped")]
        if not items:
            continue
        scores[str(group.get("key"))] = sum(_clamp01(item.get("score")) for item in items) / len(items)
    return scores


def _blend_scores(base: float, judge_score: float | None, *, judge_weight: float) -> float:
    if judge_score is None or judge_weight <= 0:
        return _clamp01(base)
    return _clamp01(base) * (1 - judge_weight) + _clamp01(judge_score) * judge_weight


def _judge_dimension_score(structured: dict[str, Any], key: str, fallback: float | None = 0.0) -> float | None:
    if not structured:
        return fallback
    if key == "overall_verdict":
        return _judge_v3_compat_score(structured.get("overall_verdict"))
    data = structured.get(key)
    if not isinstance(data, dict):
        return fallback
    verdict = str(data.get("verdict") or "").strip().lower()
    maps = {
        "efficiency": {"normal": 1.0, "high": 0.72, "excessive": 0.35},
        "relevance": {"aligned": 1.0, "partial": 0.58, "off": 0.12},
        "instruction_following": {"yes": 1.0, "partial": 0.58, "no": 0.0},
        "tool_use": {"correct": 1.0, "suboptimal": 0.62, "wrong": 0.12},
        "reasoning": {"on_track": 1.0, "redundant": 0.72, "drift": 0.42, "lost": 0.08},
        "faithfulness": {"grounded": 1.0, "partial": 0.58, "hallucinated": 0.0},
        "task_completion": {"resolved": 1.0, "partial": 0.55, "unresolved": 0.0},
    }
    return maps.get(key, {}).get(verdict, fallback)


def _reliability_score(results: list[dict[str, Any]], metrics: dict[str, Any], assertion_pass_rate: float) -> float:
    score = 0.55 + 0.45 * _clamp01(assertion_pass_rate)
    high_failures = sum(
        1 for item in results
        if not item.get("passed") and not item.get("skipped") and item.get("severity") in {"critical", "high"}
    )
    medium_failures = sum(
        1 for item in results
        if not item.get("passed") and not item.get("skipped") and item.get("severity") == "medium"
    )
    score -= min(0.35, high_failures * 0.14)
    score -= min(0.15, medium_failures * 0.04)
    score -= min(0.18, int(metrics.get("tool_error_count", 0) or 0) * 0.05)
    if metrics.get("pii_or_secret_risk"):
        score -= 0.35
    if not metrics.get("has_final_response"):
        score -= 0.45
    return _clamp01(score)


def _compat_scores_from_assertions(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "key": item["key"],
            "name": item["name"],
            "label_zh": item["name"],
            "label_en": item["name"],
            "score": item["score"],
            "weight": 0,
            "threshold": item["threshold"],
            "passed": item["passed"],
            "reason_zh": item["reason"],
            "reason_en": item["reason"],
            "reason": item["reason"],
        }
        for item in results
    ]


def _build_v3_turn_summary(
    metrics: dict[str, Any],
    task_profile: dict[str, Any],
    results: list[dict[str, Any]],
    assertion_pass_rate: float,
    eval_panel: dict[str, Any],
) -> dict[str, Any]:
    failures = [item for item in results if not item.get("passed") and not item.get("skipped")]
    safety_gate = eval_panel.get("safety_gate") if isinstance(eval_panel.get("safety_gate"), dict) else {}
    return {
        "zh": "本报告取消跨维度加权综合分，改为任务完成、工具使用、轨迹推进、指令遵循、忠实度五个 verdict 并列展示；效率和可靠性仅作为诊断项。",
        "en": "This report does not compute a weighted total score; it shows independent verdicts plus diagnostics.",
        "verdict": eval_panel.get("overall_verdict") or "needs_attention",
        "task_labels": [],
        "failed_assertions": [{"key": x["key"], "severity": x["severity"], "reason": x["reason"]} for x in failures[:6]],
        "tokens_per_second": metrics.get("tokens_per_second"),
        "assertion_pass_rate": assertion_pass_rate,
        "safety_gate": safety_gate.get("status"),
    }


def _build_v3_pipeline(
    task_profile: dict[str, Any],
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "automation_ready": {
            "ready": True,
            "dataset_case_shape": "用户请求 + 原始会话数据 + 结果/轨迹/证据评分",
            "recommended_trials": 3 if metrics.get("tool_count", 0) or metrics.get("step_count", 0) >= 8 else 1,
            "pass_threshold": 0.75,
        },
        "ab_testing": {
            "unit": "同一用户请求或同类真实 Trace 的 baseline/candidate 成对比较",
            "primary_metric": "core_dimension_verdicts",
            "secondary_metrics": ["assertion_pass_rate", "matched_assertion_delta", "total_tokens", "tool_count", "tool_kind_count", "mcp_tool_count", "rag_tool_count", "retrieval_tool_count", "shell_tool_count", "tool_error_count"],
            "dimensions": ["task_success", "tool_use", "action_advancement", "instruction_following", "faithfulness"],
            "requires_pair": True,
        },
    }


def _contains_json(text: str) -> bool:
    for pattern in (r"\{.*?\}", r"\[.*?\]"):
        for match in re.findall(pattern, text or "", re.S):
            try:
                json.loads(match)
                return True
            except Exception:
                continue
    return False


def _tool_results_referenced(final_response: str, turn: dict[str, Any]) -> bool:
    if re.search(r"checked|found|ran|read|updated|created|edited|failed|passed|查看|读取|运行|更新|创建|修改|失败|通过", final_response, re.I):
        return True
    text = _flatten_text(turn.get("steps") or [])
    return bool(re.search(r"observation|result|stdout|stderr|output|tool_result|结果|输出", text, re.I))


def _query_likely_needs_tool(query: str) -> bool:
    return bool(re.search(r"search|read|file|run|test|build|code|repo|查|搜索|文件|运行|测试|构建|代码|仓库", query, re.I))


def _query_mentions_file_edit(query: str) -> bool:
    return bool(re.search(r"edit|write|modify|patch|fix|implement|refactor|修改|编辑|修复|实现|重构|写入", query, re.I))


def _query_mentions_research(query: str) -> bool:
    return bool(re.search(r"research|search|source|citation|report|调研|搜索|来源|引用|报告|资料", query, re.I))


def _has_file_edit_evidence(turn: dict[str, Any], cot: dict[str, Any]) -> bool:
    text = _flatten_text(turn)
    if re.search(r"apply_patch|write_file|edit_file|strreplace|multi_edit|file_op|afterFileEdit|Write|Edit|Patch", text, re.I):
        return True
    for artifact in cot.get("script_artifacts") or []:
        if isinstance(artifact, dict) and _to_int(artifact.get("edit_count")) > 0:
            return True
    return False


def _has_validation_evidence(turn: dict[str, Any]) -> bool:
    text = _flatten_text(turn)
    return bool(re.search(r"pytest|npm run|pnpm|yarn test|tsc|ruff|mypy|build|lint|doctor|测试|构建|检查", text, re.I))


def _has_retrieval_evidence(turn: dict[str, Any]) -> bool:
    text = _flatten_text(turn)
    return bool(re.search(r"web_search|search_query|retrieval|rag|source|citation|browser|open_url|搜索|检索|引用|来源", text, re.I))


def _collect_turn_tool_calls(turn: dict[str, Any]) -> list[str]:
    explicit = [str(x) for x in (turn.get("tool_calls") or []) if x]
    if explicit:
        return explicit
    calls: list[str] = []
    steps = turn.get("steps") if isinstance(turn.get("steps"), list) else []
    for step in steps:
        if not isinstance(step, dict):
            continue
        meta = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        candidates = [
            step.get("tool_name"),
            step.get("name"),
            step.get("tool"),
            meta.get("tool_name"),
            meta.get("name"),
            meta.get("tool"),
            meta.get("server"),
        ]
        for value in candidates:
            if isinstance(value, str) and value.strip():
                calls.append(value.strip())
                break
    return calls


def _normalize_tool_name(name: str) -> str:
    clean = re.sub(r"\s+", " ", str(name or "").strip())
    return clean or "unknown"


def _classify_tool_name(name: str) -> str:
    n = _normalize_tool_name(name).lower()
    if re.search(r"mcp__|^mcp\b|\bmcp[-_:]", n):
        return "mcp"
    if re.search(r"\brag\b|vector|embedding|semantic|knowledge|memory|知识库|向量|召回", n):
        return "rag"
    if re.search(r"browser|playwright|click|screenshot|mouse|keyboard|navigate|页面|点击|截图|浏览器", n):
        return "browser"
    if re.search(r"apply_patch|patch|strreplace|replace|edit|multi_edit|afterfileedit|aftertabfileedit|修改|编辑", n):
        return "edit"
    if re.search(r"\bwrite\b|write_file|create_file|save|touch|new_file|写入|创建文件", n):
        return "write"
    if re.search(r"\bread\b|open|cat|sed|head|tail|get-content|view|读取|查看", n):
        return "read"
    if re.search(r"shell|bash|powershell|command|cmd|terminal|exec|run_command|运行命令", n):
        return "shell"
    if re.search(r"search|grep|rg|find|glob|ripgrep|web_search|search_query|搜索|查找", n):
        return "search"
    if re.search(r"retriev|fetch|query|lookup|检索|查询", n):
        return "retrieval"
    if re.search(r"todo|plan|taskcreate|taskupdate|createplan|switchmode|计划", n):
        return "plan"
    if re.search(r"file|path|ls|dir|copy|move|delete|remove|文件|路径", n):
        return "file"
    if re.search(r"llm|model|completion|chat", n):
        return "llm"
    return "other"


def _tool_taxonomy_counts(tool_calls: list[str], turn: dict[str, Any]) -> dict[str, Any]:
    text = "\n".join(tool_calls) + "\n" + _flatten_text(turn.get("steps") or [])
    name_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for raw_name in tool_calls:
        name = _normalize_tool_name(raw_name)
        name_counts[name] = name_counts.get(name, 0) + 1
        category = _classify_tool_name(name)
        category_counts[category] = category_counts.get(category, 0) + 1

    def cat_count(*names: str) -> int:
        return sum(int(category_counts.get(name, 0) or 0) for name in names)

    sorted_names = dict(sorted(name_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:50])
    sorted_categories = dict(sorted(category_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return {
        "tool_kind_count": len([k for k, v in category_counts.items() if v > 0]),
        "tool_category_counts": sorted_categories,
        "tool_name_counts": sorted_names,
        "mcp_tool_count": cat_count("mcp"),
        "rag_tool_count": cat_count("rag"),
        "retrieval_tool_count": cat_count("retrieval", "read"),
        "search_tool_count": cat_count("search"),
        "browser_tool_count": cat_count("browser"),
        "file_tool_count": cat_count("file", "read", "write", "edit", "search"),
        "shell_tool_count": cat_count("shell"),
        "read_tool_count": cat_count("read"),
        "write_tool_count": cat_count("write"),
        "edit_tool_count": cat_count("edit"),
        "plan_tool_count": cat_count("plan"),
        "other_tool_count": cat_count("other"),
        "tool_trace_mentions_mcp": len(re.findall(r"\bmcp\b|mcp__", text, re.I)),
        "tool_trace_mentions_rag": len(re.findall(r"\brag\b|retrieval|vector|embedding|知识库|向量", text, re.I)),
    }


def _step_tool_name(step: Any) -> str:
    if not isinstance(step, dict):
        return "unknown"
    meta = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
    for value in (
        step.get("tool_name"),
        step.get("name"),
        step.get("tool"),
        meta.get("tool_name"),
        meta.get("name"),
        meta.get("tool"),
        meta.get("server"),
    ):
        if isinstance(value, str) and value.strip():
            return _normalize_tool_name(value)
    return "unknown"


def _short_error_message(step: Any, limit: int = 220) -> str:
    if not isinstance(step, dict):
        return ""
    meta = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
    candidates = [
        meta.get("error"),
        meta.get("stderr"),
        meta.get("message"),
        step.get("error"),
        step.get("content"),
        step.get("summary"),
    ]
    for value in candidates:
        text = _flatten_text(value).strip()
        if text:
            return re.sub(r"\s+", " ", text)[:limit]
    return ""


def _error_term_counts(value: Any) -> dict[str, int]:
    text = _flatten_text(value).lower()
    counts: dict[str, int] = {}
    for term in ERROR_TERMS:
        count = len(re.findall(re.escape(term.lower()), text))
        if count:
            counts[term] = count
    return counts


def _build_error_observability(steps: list[Any], final_response: str, turn: dict[str, Any]) -> dict[str, Any]:
    tool_error_by_tool: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    for idx, step in enumerate(steps):
        if not _is_tool_error(step):
            continue
        tool_name = _step_tool_name(step)
        tool_error_by_tool[tool_name] = tool_error_by_tool.get(tool_name, 0) + 1
        if len(samples) < 12:
            samples.append(
                {
                    "source": "tool",
                    "step_index": step.get("step_index", idx) if isinstance(step, dict) else idx,
                    "tool": tool_name,
                    "message": _short_error_message(step),
                }
            )

    recovery_steps = [
        step for step in steps
        if isinstance(step, dict) and step.get("step_type") == "error_recovery"
    ]
    final_term_counts = _error_term_counts(final_response)
    step_term_counts = _error_term_counts(
        [
            {"content": s.get("content"), "metadata": s.get("metadata")}
            for s in steps
            if isinstance(s, dict) and not _is_tool_error(s)
        ]
    )
    breakdown = {
        "tool_errors": sum(tool_error_by_tool.values()),
        "error_recovery_steps": len(recovery_steps),
        "turn_error_recovery_flag": 1 if turn.get("has_error_recovery") else 0,
        "final_response_error_terms": sum(final_term_counts.values()),
        "step_text_error_terms": sum(step_term_counts.values()),
    }
    breakdown = {key: value for key, value in breakdown.items() if value}
    if recovery_steps and len(samples) < 12:
        for step in recovery_steps[: max(0, 12 - len(samples))]:
            samples.append(
                {
                    "source": "recovery",
                    "step_index": step.get("step_index"),
                    "tool": _step_tool_name(step),
                    "message": _short_error_message(step),
                }
            )

    return {
        "error_count": sum(breakdown.values()),
        "error_breakdown": breakdown,
        "error_term_counts": {
            "final_response": final_term_counts,
            "steps": step_term_counts,
        },
        "tool_error_count": sum(tool_error_by_tool.values()),
        "tool_error_by_tool": dict(sorted(tool_error_by_tool.items(), key=lambda kv: (-kv[1], kv[0]))),
        "error_samples": samples,
    }


def extract_turn_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    turn = payload.get("turn") or {}
    cot = payload.get("cot") or {}
    steps = turn.get("steps") if isinstance(turn.get("steps"), list) else []
    final_response = str(turn.get("final_response") or "")
    user_query = str(turn.get("user_query") or turn.get("question") or "")
    tool_calls = _collect_turn_tool_calls(turn)
    tool_taxonomy = _tool_taxonomy_counts(tool_calls, turn)
    usage = _extract_turn_usage(turn, payload)
    duration_ms = _extract_turn_duration_ms(turn, steps)
    duration_sec = duration_ms / 1000 if duration_ms and duration_ms > 0 else None
    total_tokens = usage["total_tokens"]
    output_tokens = usage["output_tokens"]
    tps_basis = "output_tokens" if output_tokens > 0 else "total_tokens"
    tps_tokens = output_tokens if output_tokens > 0 else total_tokens
    tokens_per_second = (tps_tokens / duration_sec) if duration_sec and tps_tokens > 0 else None

    error_terms = _find_errors(final_response)
    error_observability = _build_error_observability(steps, final_response, turn)
    plan_updates = _count_turn_plan_updates(cot, turn.get("turn_index"))
    pii_hits = _find_pii_or_secret(final_response + "\n" + _flatten_text(steps))
    step_type_counts: dict[str, int] = {}
    missing_duration_steps = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_type = str(step.get("step_type") or "unknown")
        step_type_counts[step_type] = step_type_counts.get(step_type, 0) + 1
        if not step.get("timestamp") and not step.get("duration_ms"):
            missing_duration_steps += 1

    return {
        "user_query": user_query,
        "final_response_chars": len(final_response),
        "has_final_response": bool(final_response.strip()),
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cache_read_tokens": usage["cache_read_tokens"],
        "cache_write_tokens": usage["cache_write_tokens"],
        "total_tokens": total_tokens,
        "duration_ms": duration_ms,
        "tokens_per_second": tokens_per_second,
        "tokens_per_second_basis": tps_basis,
        "tool_count": len(tool_calls),
        "unique_tool_count": len(set(tool_calls)),
        "tool_calls": tool_calls,
        **tool_taxonomy,
        "step_count": len(steps),
        "step_type_counts": step_type_counts,
        "strategy_shifts": _to_int(turn.get("strategy_shifts")),
        "plan_update_count": plan_updates,
        "error_count": error_observability["error_count"],
        "tool_error_count": error_observability["tool_error_count"],
        "error_terms": sorted(set(error_terms)),
        "error_breakdown": error_observability["error_breakdown"],
        "error_term_counts": error_observability["error_term_counts"],
        "tool_error_by_tool": error_observability["tool_error_by_tool"],
        "error_samples": error_observability["error_samples"],
        "pii_or_secret_risk": bool(pii_hits),
        "pii_or_secret_hits": pii_hits,
        "trace_fields_present": {
            "steps": bool(steps),
            "turn_timing": bool(duration_ms),
            "token_usage": total_tokens > 0,
            "final_response": bool(final_response.strip()),
            "tool_calls": bool(tool_calls),
        },
        "missing_duration_steps": missing_duration_steps,
    }


def _score_turn(metrics: dict[str, Any], turn: dict[str, Any], cot: dict[str, Any]) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []

    def add(
        key: str,
        zh: str,
        en: str,
        score: float,
        reason_zh: str,
        reason_en: str,
        *,
        threshold: float = 0.75,
        weight: float | None = None,
    ) -> None:
        clamped = max(0.0, min(1.0, float(score)))
        scores.append(
            {
                "key": key,
                "name": key,
                "label_zh": zh,
                "label_en": en,
                "score": clamped,
                "weight": TURN_SCORE_WEIGHTS.get(key, 0.0) if weight is None else weight,
                "threshold": threshold,
                "passed": clamped >= threshold,
                "reason_zh": reason_zh,
                "reason_en": reason_en,
                "reason": f"{reason_zh} / {reason_en}",
            }
        )

    final_chars = metrics["final_response_chars"]
    step_count = metrics["step_count"]
    tool_count = metrics["tool_count"]
    errors = metrics["error_count"]
    strategy_shifts = metrics["strategy_shifts"]
    user_query = metrics["user_query"]
    final_response = str(turn.get("final_response") or "")

    completion = 0.0 if not metrics["has_final_response"] else min(1.0, 0.65 + final_chars / 1200)
    if errors:
        completion *= 0.75
    add(
        "task_completion",
        "任务完成度",
        "Task Completion",
        completion,
        f"最终回复 {final_chars} 字符，错误信号 {errors} 个。",
        f"Final response has {final_chars} chars with {errors} error signal(s).",
    )

    relevance = _keyword_overlap_score(user_query, final_response)
    add(
        "answer_relevance",
        "回答相关性",
        "Answer Relevance",
        relevance,
        "基于用户输入关键词与最终回复的覆盖度估算。",
        "Estimated from keyword overlap between the user query and final answer.",
        threshold=0.6,
    )

    completeness = 0.0 if not final_response.strip() else min(1.0, 0.45 + final_chars / 900)
    if "?" in user_query or "？" in user_query:
        completeness = max(completeness, 0.55 if final_chars > 20 else 0.2)
    add(
        "response_completeness",
        "输出完整性",
        "Response Completeness",
        completeness,
        "按回复长度、是否有最终回复和问题形态进行启发式估算。",
        "Heuristically estimated from answer length, final-answer presence, and query shape.",
        threshold=0.65,
    )

    instruction_score = _instruction_adherence_score(user_query, final_response, tool_count)
    add(
        "instruction_adherence",
        "指令遵循",
        "Instruction Adherence",
        instruction_score,
        "检查是否出现明显拒答、空转、格式违背或只复述问题。",
        "Checks for refusal, empty loops, format misses, or simply echoing the request.",
        threshold=0.65,
    )

    tool_score = _tool_correctness_score(metrics, user_query)
    add(
        "tool_correctness",
        "工具选择正确性",
        "Tool Correctness",
        tool_score,
        f"本轮调用 {tool_count} 次工具，唯一工具 {metrics['unique_tool_count']} 个。",
        f"This turn used {tool_count} tool call(s) across {metrics['unique_tool_count']} unique tool(s).",
        threshold=0.65,
    )

    argument_score = _argument_correctness_score(turn)
    add(
        "argument_correctness",
        "参数正确性",
        "Argument Correctness",
        argument_score,
        "检查工具调用参数是否存在空参数、错误标记或明显缺失。",
        "Checks tool arguments for empty inputs, error markers, and obvious omissions.",
        threshold=0.65,
    )

    efficiency = _tool_efficiency_score(metrics)
    add(
        "tool_efficiency",
        "工具效率",
        "Tool Efficiency",
        efficiency,
        f"步骤 {step_count} 个，工具 {tool_count} 次，策略转换 {strategy_shifts} 次。",
        f"{step_count} steps, {tool_count} tool calls, {strategy_shifts} strategy shift(s).",
        threshold=0.65,
    )

    plan_quality = _plan_quality_score(metrics, turn)
    add(
        "plan_quality",
        "计划质量",
        "Plan Quality",
        plan_quality,
        f"检测到 {metrics['plan_update_count']} 次 plan/todo 更新。",
        f"Detected {metrics['plan_update_count']} plan/todo update(s).",
        threshold=0.6,
    )

    plan_adherence = _plan_adherence_score(metrics, turn, cot)
    add(
        "plan_adherence",
        "计划执行一致性",
        "Plan Adherence",
        plan_adherence,
        "基于 TodoWrite 进度、最终回复和错误恢复信号估算。",
        "Estimated from TodoWrite progress, final response, and error-recovery signals.",
        threshold=0.6,
    )

    error_free = 1.0 if errors == 0 else max(0.0, 1.0 - errors * 0.25)
    add(
        "error_free_execution",
        "错误与异常",
        "Error-Free Execution",
        error_free,
        "未检测到错误信号。" if errors == 0 else f"检测到 {errors} 个错误/恢复信号。",
        "No error signals detected." if errors == 0 else f"Detected {errors} error/recovery signal(s).",
    )

    trace_score = _trace_completeness_score(metrics)
    add(
        "trace_completeness",
        "Trace 完整性",
        "Trace Completeness",
        trace_score,
        "检查 steps、token、耗时、工具和最终回复是否被采集。",
        "Checks whether steps, tokens, timing, tools, and final answer were captured.",
        threshold=0.7,
    )

    safety = 0.0 if metrics["pii_or_secret_risk"] else 1.0
    add(
        "safety_privacy",
        "安全与隐私",
        "Safety / Privacy",
        safety,
        "未发现明显 PII/secret 模式。" if safety else "检测到疑似 PII/secret 模式。",
        "No obvious PII/secret pattern found." if safety else "Potential PII/secret pattern detected.",
    )

    return scores


def _weighted_average(scores: list[dict[str, Any]]) -> float:
    total_weight = sum(float(item.get("weight") or 0) for item in scores)
    if total_weight <= 0:
        return sum(float(item.get("score") or 0) for item in scores) / len(scores) if scores else 0.0
    return sum(float(item.get("score") or 0) * float(item.get("weight") or 0) for item in scores) / total_weight


def _build_turn_summary(metrics: dict[str, Any], scores: list[dict[str, Any]]) -> dict[str, Any]:
    weak = sorted(scores, key=lambda x: x["score"])[:3]
    strong = sorted(scores, key=lambda x: x["score"], reverse=True)[:3]
    return {
        "zh": "本报告基于当前子会话 trace 自动评分，适合用于单次会话质检和 A/B 对齐维度分析。",
        "en": "This report scores the current interaction trace and can feed per-turn QA and A/B comparison.",
        "strongest": [{"key": x["key"], "label_zh": x["label_zh"], "label_en": x["label_en"], "score": x["score"]} for x in strong],
        "needs_attention": [{"key": x["key"], "label_zh": x["label_zh"], "label_en": x["label_en"], "score": x["score"]} for x in weak],
        "tokens_per_second": metrics.get("tokens_per_second"),
    }


def _find_turn(cot: dict[str, Any], turn_index: int) -> dict[str, Any] | None:
    for turn in cot.get("turns") or []:
        if isinstance(turn, dict) and _to_int(turn.get("turn_index")) == int(turn_index):
            return turn
    return None


def _extract_turn_usage(turn: dict[str, Any], payload: dict[str, Any]) -> dict[str, int]:
    usage = _normalize_usage_dict(turn.get("usage") or {})
    input_tokens = usage.get("input_tokens", 0) + usage.get("prompt_tokens", 0)
    output_tokens = usage.get("output_tokens", 0) + usage.get("completion_tokens", 0)
    cache_read = usage.get("cache_read_tokens", 0) + _to_int((turn.get("usage") or {}).get("cache_read_input_tokens"))
    cache_write = usage.get("cache_write_tokens", 0) + _to_int((turn.get("usage") or {}).get("cache_creation_input_tokens"))
    total = usage.get("total_tokens", 0)
    if total <= 0:
        total = input_tokens + output_tokens + cache_read + cache_write
    if total <= 0:
        best = _extract_token_usage(payload)
        for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "total_tokens"):
            usage[key] = int(best.get(key, 0) or 0)
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        cache_read = usage["cache_read_tokens"]
        cache_write = usage["cache_write_tokens"]
        total = usage["total_tokens"]
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "total_tokens": total,
    }


def _extract_turn_duration_ms(turn: dict[str, Any], steps: list[Any]) -> float | None:
    for key in ("turn_duration_ms_observed", "turn_duration_ms", "duration_ms", "latency_ms"):
        value = _to_float(turn.get(key))
        if value is not None and value > 0:
            return value
    total = 0.0
    for step in steps:
        if isinstance(step, dict):
            value = _to_float(step.get("duration_ms"))
            if value is not None and value > 0:
                total += value
    return total if total > 0 else None


def _keyword_overlap_score(question: str, answer: str) -> float:
    q = question.strip()
    a = answer.lower()
    if not q:
        return 0.8 if answer.strip() else 0.0
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_\-]{2,}|\d{3,}", q)
    stop = {"什么", "怎么", "如何", "帮我", "请问", "这个", "那个", "please", "what", "how", "the", "and"}
    tokens = [t for t in dict.fromkeys(tokens) if t.lower() not in stop]
    if not tokens:
        return 0.75 if answer.strip() else 0.0
    hits = sum(1 for t in tokens if t.lower() in a)
    return max(0.35 if answer.strip() else 0.0, hits / len(tokens))


def _instruction_adherence_score(question: str, answer: str, tool_count: int) -> float:
    if not answer.strip():
        return 0.0
    q = question.strip().lower()
    a = answer.strip().lower()
    if q and a == q:
        return 0.1
    refusal_terms = ("抱歉", "无法", "不能", "sorry", "cannot", "can't")
    score = 0.85
    if any(term in a for term in refusal_terms):
        score -= 0.25
    if ("json" in q or "JSON" in question) and not ("{" in answer and "}" in answer):
        score -= 0.25
    if ("表格" in question or "table" in q) and "|" not in answer:
        score -= 0.2
    if tool_count == 0 and any(term in q for term in ("查", "search", "read", "文件", "file", "源码", "代码")):
        score -= 0.15
    return score


def _tool_correctness_score(metrics: dict[str, Any], question: str) -> float:
    tool_count = metrics["tool_count"]
    q = question.lower()
    likely_tool_needed = any(term in q for term in ("查", "搜索", "文件", "源码", "代码", "运行", "执行", "search", "read", "file", "run", "test"))
    if tool_count == 0:
        return 0.65 if likely_tool_needed else 0.85
    score = 0.88
    if metrics["tool_error_count"]:
        score -= min(0.35, metrics["tool_error_count"] * 0.12)
    if metrics["unique_tool_count"] == 1 and tool_count > 8:
        score -= 0.12
    return score


def _argument_correctness_score(turn: dict[str, Any]) -> float:
    steps = turn.get("steps") if isinstance(turn.get("steps"), list) else []
    tool_steps = [s for s in steps if isinstance(s, dict) and str(s.get("step_type")) in {"tool_decision", "tool_execution"}]
    if not tool_steps:
        return 0.85
    bad = 0
    checked = 0
    for step in tool_steps:
        meta = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        tool_input = meta.get("tool_input") or meta.get("input") or meta.get("arguments") or meta.get("args")
        if tool_input is not None:
            checked += 1
            text = _flatten_text(tool_input).strip()
            if not text or text in {"{}", "[]", "null", "None"}:
                bad += 1
        if _is_tool_error(step):
            bad += 1
    if checked == 0 and tool_steps:
        return 0.7
    return max(0.0, 1.0 - bad / max(1, len(tool_steps)))


def _tool_efficiency_score(metrics: dict[str, Any]) -> float:
    tool_count = metrics["tool_count"]
    step_count = metrics["step_count"]
    shifts = metrics["strategy_shifts"]
    if step_count <= 0:
        return 0.0
    score = 0.9
    if tool_count > 0:
        ratio = step_count / max(1, tool_count)
        if ratio > 8:
            score -= min(0.25, (ratio - 8) * 0.02)
    if tool_count > 15:
        score -= min(0.25, (tool_count - 15) * 0.02)
    if shifts > 2:
        score -= min(0.25, (shifts - 2) * 0.08)
    return score


def _plan_quality_score(metrics: dict[str, Any], turn: dict[str, Any]) -> float:
    if metrics["plan_update_count"] > 0:
        return 0.9
    if metrics["step_count"] >= 6 and metrics["tool_count"] >= 2:
        return 0.65
    if metrics["tool_count"] == 0:
        return 0.78
    return 0.6


def _plan_adherence_score(metrics: dict[str, Any], turn: dict[str, Any], cot: dict[str, Any]) -> float:
    plans = [
        p for p in (cot.get("plan_timeline") or [])
        if isinstance(p, dict) and _to_int(p.get("turn_index")) == _to_int(turn.get("turn_index"))
    ]
    if not plans:
        return 0.75 if metrics["has_final_response"] else 0.35
    last = plans[-1]
    todos = last.get("todos") if isinstance(last.get("todos"), list) else []
    if todos:
        completed = sum(1 for t in todos if isinstance(t, dict) and t.get("status") == "completed")
        return completed / max(1, len(todos))
    completed = len(last.get("completed") or [])
    total = _to_int(last.get("total")) or completed + len(last.get("pending") or []) + len(last.get("in_progress") or [])
    if total > 0:
        return completed / total
    return 0.8


def _trace_completeness_score(metrics: dict[str, Any]) -> float:
    fields = metrics["trace_fields_present"]
    weights = {"steps": 0.35, "turn_timing": 0.2, "token_usage": 0.2, "final_response": 0.15, "tool_calls": 0.1}
    score = sum(weight for key, weight in weights.items() if fields.get(key))
    if metrics["tool_count"] == 0:
        score += weights["tool_calls"]
    missing_steps = metrics.get("missing_duration_steps") or 0
    if metrics["step_count"] and missing_steps / metrics["step_count"] > 0.6:
        score -= 0.1
    return score


def _token_efficiency_score(metrics: dict[str, Any]) -> float:
    total = metrics["total_tokens"]
    output = metrics["output_tokens"]
    if total <= 0:
        return 0.45
    if output <= 0:
        return 0.55
    ratio = output / total
    score = 0.75 + min(0.2, ratio)
    if total > 250000:
        score -= 0.2
    elif total > 100000:
        score -= 0.1
    return score


def _count_turn_plan_updates(cot: dict[str, Any], turn_index: Any) -> int:
    idx = _to_int(turn_index)
    return sum(
        1 for p in (cot.get("plan_timeline") or [])
        if isinstance(p, dict) and _to_int(p.get("turn_index")) == idx
    )


def _is_tool_error(step: Any) -> bool:
    if not isinstance(step, dict):
        return False
    meta = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
    explicit_is_error = meta.get("is_error")
    if explicit_is_error is True or str(explicit_is_error).lower() == "true":
        return True
    raw_result = meta.get("raw_result")
    if isinstance(raw_result, dict):
        if raw_result.get("success") is True or str(raw_result.get("status") or "").lower() == "success":
            return False
        result = raw_result.get("result")
        if isinstance(result, dict) and (
            result.get("isError") is False
            or result.get("is_error") is False
            or str(result.get("status") or "").lower() == "success"
        ):
            return False
    if meta.get("error"):
        return True
    if explicit_is_error is False:
        return False
    content = step.get("content")
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = None
        if isinstance(parsed, dict) and (
            parsed.get("isError") is False
            or parsed.get("is_error") is False
            or parsed.get("success") is True
            or str(parsed.get("status") or "").lower() == "success"
        ):
            return False
    text = _flatten_text({"content": step.get("content"), "metadata": meta}).lower()
    return any(term in text for term in ERROR_TERMS)


def _find_pii_or_secret(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in PII_PATTERNS:
        for match in pattern.findall(text or ""):
            label = match if isinstance(match, str) else str(match)
            hits.append(label[:48])
    return hits[:10]


def build_session_eval_report(
    session_id: str,
    *,
    cot: dict[str, Any] | None = None,
    transcript: dict[str, Any] | None = None,
    otel: dict[str, Any] | None = None,
    overview: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "session_id": session_id,
        "cot": cot or {},
        "transcript": transcript or {},
        "otel": otel or {},
        "overview": overview or {},
    }
    metrics = extract_session_metrics(payload)
    scores = _score_session(metrics)
    overall_score = sum(item["score"] for item in scores) / len(scores) if scores else 0.0
    return {
        "report_id": f"session-eval-{session_id}",
        "session_id": session_id,
        "created_at": utc_now(),
        "passed": overall_score >= 0.75 and metrics["error_count"] == 0,
        "overall_score": overall_score,
        "metrics": metrics,
        "scores": scores,
        "source": {
            "has_cot": bool(cot),
            "has_transcript": bool(transcript),
            "has_otel": bool(otel),
            "has_overview": bool(overview),
        },
    }


def extract_session_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    overview = payload.get("overview") or {}
    cot = payload.get("cot") or {}
    otel = payload.get("otel") or {}
    transcript = payload.get("transcript") or {}

    usage = _extract_token_usage(payload)
    cost = _first_positive_number(
        overview,
        cot,
        otel,
        transcript,
        keys=("actual_cost_usd", "cost_usd", "full_price_cost_usd", "total_cost", "cost"),
    )
    latency_ms = _extract_latency_ms(payload)
    tool_count = _to_int(overview.get("total_tool_calls")) or _count_tools(payload)
    llm_call_count = _count_llm_calls(payload)
    turn_count = _to_int(overview.get("total_turns")) or _count_turns(payload)
    error_hits = _find_errors(payload)
    span_count = _count_spans(payload)

    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "cache_read_tokens": int(usage.get("cache_read_tokens", 0) or 0),
        "cache_write_tokens": int(usage.get("cache_write_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "cost_usd": cost,
        "latency_ms": latency_ms,
        "tool_count": tool_count,
        "llm_call_count": llm_call_count,
        "turn_count": turn_count,
        "span_count": span_count,
        "error_count": len(error_hits),
        "error_terms": sorted(set(error_hits)),
        "trace_present": bool(cot or otel),
        "transcript_present": bool(transcript),
    }


def _score_session(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    scores = []

    def add(name: str, score: float, reason: str, threshold: float = 0.75) -> None:
        clamped = max(0.0, min(1.0, float(score)))
        scores.append(
            {
                "name": name,
                "score": clamped,
                "threshold": threshold,
                "passed": clamped >= threshold,
                "reason": reason,
            }
        )

    add(
        "trace coverage",
        1.0 if metrics["trace_present"] else 0.0,
        "Trace captured" if metrics["trace_present"] else "No trace found for this session",
    )
    add(
        "transcript coverage",
        1.0 if metrics["transcript_present"] else 0.4,
        "Transcript captured"
        if metrics["transcript_present"]
        else "Transcript not found; score uses trace only",
        threshold=0.5,
    )
    add(
        "error free",
        1.0 if metrics["error_count"] == 0 else 0.0,
        "No error markers found"
        if metrics["error_count"] == 0
        else "Error markers found: " + ", ".join(metrics["error_terms"][:8]),
    )
    add(
        "tool observability",
        1.0 if metrics["tool_count"] > 0 else 0.7,
        f"{metrics['tool_count']} tool call(s) visible"
        if metrics["tool_count"] > 0
        else "No tool calls visible; acceptable for simple no-tool tasks, but weak for agent workflows",
        threshold=0.7,
    )
    add(
        "token accounting",
        1.0 if metrics["total_tokens"] > 0 else 0.3,
        f"{metrics['total_tokens']} total tokens accounted"
        if metrics["total_tokens"] > 0
        else "No token usage captured",
        threshold=0.7,
    )
    add(
        "cost accounting",
        1.0 if metrics["cost_usd"] is not None else 0.5,
        f"Cost captured: ${metrics['cost_usd']:.6f}"
        if metrics["cost_usd"] is not None
        else "Cost not available; model pricing or token data may be missing",
        threshold=0.5,
    )
    return scores


def _extract_token_usage(payload: dict[str, Any]) -> dict[str, int]:
    direct = _find_best_usage(payload)
    input_tokens = direct.get("input_tokens", 0) + direct.get("prompt_tokens", 0)
    output_tokens = direct.get("output_tokens", 0) + direct.get("completion_tokens", 0)
    cache_read = direct.get("cache_read_tokens", 0)
    cache_write = direct.get("cache_write_tokens", 0)
    total = direct.get("total_tokens", 0)
    if total <= 0:
        total = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "total_tokens": total,
    }


def _find_best_usage(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        overview_otel = ((value.get("overview") or {}).get("otel") or {})
        normalized = _normalize_usage_dict(overview_otel)
        if any(normalized.values()):
            return normalized
    candidates: list[dict[str, int]] = []

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            for key in ("actual_token_usage", "token_usage", "usage", "totals"):
                nested = item.get(key)
                if isinstance(nested, dict):
                    normalized = _normalize_usage_dict(nested)
                    if any(normalized.values()):
                        candidates.append(normalized)
            normalized = _normalize_usage_dict(item)
            if any(normalized.values()):
                candidates.append(normalized)
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    if not candidates:
        return {}
    return max(candidates, key=lambda data: data.get("total_tokens", 0) or sum(data.values()))


def _normalize_usage_dict(data: dict[str, Any]) -> dict[str, int]:
    return {
        "input_tokens": _to_int(data.get("input_tokens")),
        "prompt_tokens": _to_int(data.get("prompt_tokens")),
        "output_tokens": _to_int(data.get("output_tokens")),
        "completion_tokens": _to_int(data.get("completion_tokens")),
        "cache_read_tokens": _to_int(data.get("cache_read_tokens")),
        "cache_write_tokens": _to_int(data.get("cache_write_tokens")),
        "total_tokens": _to_int(data.get("total_tokens")),
    }


def _extract_latency_ms(value: Any) -> float | None:
    numbers = []

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            for key in (
                "duration_ms",
                "latency_ms",
                "elapsed_ms",
                "total_duration_ms",
                "response_time_ms",
            ):
                n = _to_float(item.get(key))
                if n is not None and n >= 0:
                    numbers.append(n)
            for key in ("duration", "latency", "response_time", "total_duration"):
                n = _to_float(item.get(key))
                if n is not None and 0 <= n < 100000:
                    numbers.append(n * 1000 if n < 1000 else n)
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    return max(numbers) if numbers else None


def _first_number(*items: Any, keys: tuple[str, ...]) -> float | None:
    for item in items:
        value = _find_number(item, keys)
        if value is not None:
            return value
    return None


def _first_positive_number(*items: Any, keys: tuple[str, ...]) -> float | None:
    for item in items:
        value = _find_number(item, keys)
        if value is None:
            continue
        if value > 0:
            return value
    return None


def _find_number(value: Any, keys: tuple[str, ...]) -> float | None:
    if isinstance(value, dict):
        for key in keys:
            n = _to_float(value.get(key))
            if n is not None:
                return n
        for child in value.values():
            n = _find_number(child, keys)
            if n is not None:
                return n
    elif isinstance(value, list):
        for child in value:
            n = _find_number(child, keys)
            if n is not None:
                return n
    return None


def _count_tools(value: Any) -> int:
    count = 0
    if isinstance(value, dict):
        for key, child in value.items():
            low = str(key).lower()
            if low in {"tool_calls", "tools"} and isinstance(child, list):
                count += len(child)
            elif low in {"tool_name", "tool", "tool_use_id"} and child:
                count += 1
            else:
                count += _count_tools(child)
    elif isinstance(value, list):
        for child in value:
            count += _count_tools(child)
    return count


def _count_llm_calls(value: Any) -> int:
    text = _flatten_text(value).lower()
    markers = ("llm_request", "chat.completion", "completion", "model_request", "api_request")
    return sum(text.count(marker) for marker in markers)


def _count_turns(value: Any) -> int:
    if isinstance(value, dict):
        for key in ("turns", "messages", "interactions"):
            child = value.get(key)
            if isinstance(child, list):
                return len(child)
        return max((_count_turns(child) for child in value.values()), default=0)
    if isinstance(value, list):
        return max((_count_turns(child) for child in value), default=0)
    return 0


def _count_spans(value: Any) -> int:
    if isinstance(value, dict):
        total = 0
        for key, child in value.items():
            if str(key).lower() in {"spans", "resource_spans", "scope_spans"} and isinstance(child, list):
                total += len(child)
            total += _count_spans(child)
        return total
    if isinstance(value, list):
        return sum(_count_spans(child) for child in value)
    return 0


def _find_errors(value: Any) -> list[str]:
    text = _flatten_text(value).lower()
    return [term for term in ERROR_TERMS if term.lower() in text]


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_text(v) for v in value)
    return str(value)


def _to_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

