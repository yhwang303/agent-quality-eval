"""Agent Critic sidecar runner and report ingestion helpers."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import utc_now

CRITIC_REPORT_SCHEMA_VERSION = "agent-critic-v1"
DIMENSION_KEYS = (
    "task_completion",
    "tool_use",
    "reasoning",
    "instruction_following",
    "faithfulness",
    "efficiency",
    "reliability",
)
USER_REQUEST_COVERAGE_KEY = "user_request_coverage"
INCOMPLETE_REPORT_STALE_SECONDS = 600


def critic_log_path() -> Path:
    return Path.home() / ".agent-cot" / "logs" / "critic-runner.log"


def log_critic_event(event: str, **fields: Any) -> None:
    try:
        path = critic_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "event": event,
            **fields,
        }
        with path.open("a", encoding="utf-8", errors="replace") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def data_root() -> Path:
    env = os.environ.get("AGENT_COT_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    runtime = Path.home() / ".agent-cot" / "runtime.json"
    try:
        raw = json.loads(runtime.read_text(encoding="utf-8"))
        value = raw.get("data_root") if isinstance(raw, dict) else None
        if isinstance(value, str) and value.strip():
            path = Path(value).expanduser()
            if path.name in {".agent-cot", ".cursor-cot"}:
                return path / "data"
            return path
    except Exception:
        pass
    return Path.home() / ".agent-cot" / "data"


def critic_reports_root() -> Path:
    return data_root() / "critic"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value or "unknown")[:180]


def critic_report_path(session_id: str, turn_index: int) -> Path:
    return critic_reports_root() / _safe_name(session_id) / f"turn_{int(turn_index)}.json"


def load_critic_report(session_id: str, turn_index: int) -> dict[str, Any] | None:
    path = critic_report_path(session_id, turn_index)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema_version") != CRITIC_REPORT_SCHEMA_VERSION:
        return None
    if (
        str(data.get("status") or "").lower() == "queued"
        and str(data.get("source_event") or "") == "api-auto-queued"
    ):
        return None
    # Product contract: frontend Eval only ingests reports produced by the IDE
    # hook/sidecar path. Older builds wrote api-rerun reports when the user
    # clicked Eval; those are not hook-stage artifacts and must not be reused.
    if str(data.get("source_event") or "") == "api-rerun":
        return None
    return data


def write_critic_report(report: dict[str, Any]) -> Path:
    session_id = str(report.get("session_id") or "unknown")
    turn_index = int(report.get("turn_index") or 0)
    path = critic_report_path(session_id, turn_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    log_critic_event(
        "report_written",
        session_id=session_id,
        turn_index=turn_index,
        status=report.get("status"),
        source_event=report.get("source_event"),
        path=str(path),
    )
    return path


def _cot_dir() -> Path:
    return Path(os.environ.get("COT_DIR") or data_root() / "cot").expanduser()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def find_cot_file(session_id: str | None, *, wait_seconds: float = 45.0) -> Path | None:
    log_critic_event("find_cot_start", session_id=session_id or "", wait_seconds=wait_seconds, cot_dir=str(_cot_dir()))
    deadline = time.time() + max(0.0, wait_seconds)
    names: list[str] = []
    if session_id:
        sid = str(session_id).strip()
        names.extend([sid, sid.removeprefix("codex-")])
        if not sid.startswith("codex-"):
            names.append(f"codex-{sid}")
        if not sid.startswith("codebuddy-"):
            names.append(f"codebuddy-{sid}")
        if sid.startswith("codebuddy-"):
            names.append(sid.removeprefix("codebuddy-"))
    while True:
        root = _cot_dir()
        for name in dict.fromkeys(n for n in names if n):
            path = root / f"{name}_cot.json"
            if path.is_file():
                log_critic_event("find_cot_ok", session_id=session_id or "", path=str(path))
                return path
        try:
            files = list(root.glob("*_cot.json"))
        except Exception:
            files = []
        if files and not session_id:
            latest = max(files, key=lambda p: p.stat().st_mtime)
            log_critic_event("find_cot_latest", session_id="", path=str(latest))
            return latest
        if time.time() >= deadline:
            log_critic_event("find_cot_missing", session_id=session_id or "", cot_dir=str(root))
            return None
        time.sleep(1.0)


def _latest_turn(cot: dict[str, Any]) -> dict[str, Any] | None:
    turns = [t for t in cot.get("turns") or [] if isinstance(t, dict)]
    if not turns:
        return None
    return max(turns, key=lambda t: int(t.get("turn_index") or 0))


def _find_turn(cot: dict[str, Any], turn_index: int) -> dict[str, Any] | None:
    for turn in cot.get("turns") or []:
        if isinstance(turn, dict) and int(turn.get("turn_index") or 0) == int(turn_index):
            return turn
    return None


def _report_age_seconds(report: dict[str, Any]) -> float | None:
    raw = str(report.get("created_at") or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except Exception:
        return None


def is_incomplete_critic_report(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    return str(report.get("status") or "").lower() in {"queued", "running"}


def is_stale_incomplete_critic_report(
    report: dict[str, Any] | None,
    *,
    max_age_seconds: float = INCOMPLETE_REPORT_STALE_SECONDS,
) -> bool:
    if not is_incomplete_critic_report(report):
        return False
    age = _report_age_seconds(report or {})
    return age is not None and age >= max_age_seconds


def _should_reuse_existing_report(report: dict[str, Any] | None, source_event: str) -> bool:
    if not isinstance(report, dict):
        return False
    existing_source = str(report.get("source_event") or "")
    if (
        existing_source in {"manual-agent-critic-fallback", "manual-hook-report-recovery"}
        and source_event != existing_source
    ):
        return False
    if source_event == "api-rerun":
        return False
    status = str(report.get("status") or "").lower()
    if status == "completed":
        return True
    if status in {"queued", "running"}:
        age = _report_age_seconds(report)
        return age is None or age < INCOMPLETE_REPORT_STALE_SECONDS
    return False


def should_reuse_completed_hook_report(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    status = str(report.get("status") or "").lower()
    source_event = str(report.get("source_event") or "")
    return status == "completed" and source_event and source_event != "api-rerun"


def _parse_iso_timestamp(value: Any) -> float:
    raw = str(value or "").strip()
    if not raw:
        return 0.0
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _live_state_score(state: dict[str, Any] | None) -> tuple[int, int, float]:
    if not isinstance(state, dict):
        return (-1, -1, 0.0)
    status = str(state.get("status") or "").lower()
    status_rank = {
        "completed": 4,
        "error": 3,
        "running": 1,
        "queued": 0,
    }.get(status, 0)
    try:
        events = int(state.get("event_count") or 0)
    except Exception:
        events = 0
    updated = _parse_iso_timestamp(state.get("updated_at") or state.get("completed_at") or state.get("started_at"))
    return (status_rank, events, updated)


def load_best_live_critic_state(session_id: str, turn_index: int | None = None) -> dict[str, Any] | None:
    try:
        from .live_critic import load_live_critic_state

        candidates = []
        if turn_index is not None:
            candidates.append(load_live_critic_state(session_id, turn_index))
        candidates.append(load_live_critic_state(session_id))
        states = [state for state in candidates if isinstance(state, dict)]
        if not states:
            return None
        return max(states, key=_live_state_score)
    except Exception:
        return None


def _dimension(verdict: str, review: str) -> dict[str, str]:
    return {"verdict": str(verdict or "partial"), "review": str(review or "").strip()}


def _deterministic_structured(
    *,
    reason: str,
    metrics: dict[str, Any],
    overall_verdict: str = "partial",
) -> dict[str, Any]:
    total = int(metrics.get("total_tokens") or 0)
    elapsed = round(float(metrics.get("duration_ms") or 0.0) / 1000.0, 2)
    calls = int(metrics.get("tool_count") or 0)
    failed = int(metrics.get("tool_error_count") or 0)
    summary = (
        f"结论：Agent Critic 未完成模型评审，当前仅保留确定性断言兜底。"
        f"本轮记录到 {total} tokens、{elapsed}s、{calls} 次工具调用、失败 {failed} 次；"
        f"需要结合最终回复与工具结果继续复核交付质量。原因：{reason}"
    )
    structured = {
        "summary_conclusion": summary,
        "overall_verdict": overall_verdict,
        USER_REQUEST_COVERAGE_KEY: (
            "当前未完成模型评审，用户诉求覆盖情况只能依据 trace、最终回复和确定性断言做保守判断；"
            "需要在 critic 模型可用后复核用户目标、交付内容和未完成风险之间的对应关系。"
        ),
        "task_completion": _dimension("partial", "模型评审暂不可用，任务完成度暂按确定性断言与最终回复存在性保守展示。"),
        "tool_use": _dimension("suboptimal", f"本轮记录到 {calls} 次工具调用，其中失败 {failed} 次；失败是否影响交付需结合原始工具结果判断。"),
        "reasoning": _dimension("on_track", "未调用 critic 模型时不对推理轨迹做强语义判断，仅提示回看步骤顺序与恢复动作。"),
        "instruction_following": _dimension("partial", "未调用 critic 模型时不推断隐含指令，只保留硬性断言的客观结果。"),
        "faithfulness": _dimension("partial", "最终回复关键声称仍需对照 tool_result 与原始 transcript 复核。"),
        "efficiency": _dimension("normal", f"运行统计为 {total} tokens、{elapsed}s、{calls} 次工具调用。"),
        "reliability": _dimension("clear" if failed == 0 else "minor_issues", f"工具失败数为 {failed}，由确定性断言继续标记是否阻断。"),
    }
    structured["review_markdown"] = _render_review_markdown(structured)
    return structured


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    candidates = [raw]
    candidates.extend(re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw, re.I))
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return None


def _normalize_structured(data: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    fallback = _deterministic_structured(reason="critic JSON 字段不完整", metrics=metrics)
    out: dict[str, Any] = {}
    summary = str(data.get("summary_conclusion") or data.get("summary") or "").strip()
    if not summary:
        summary = fallback["summary_conclusion"]
    if not summary.startswith("结论"):
        summary = "结论：" + summary
    out["summary_conclusion"] = summary[:900]
    overall = str(data.get("overall_verdict") or data.get("verdict") or fallback["overall_verdict"]).strip()
    if overall not in {"resolved", "partial", "unresolved"}:
        overall = "partial"
    out["overall_verdict"] = overall
    coverage = str(data.get(USER_REQUEST_COVERAGE_KEY) or data.get("user_need_coverage") or "").strip()
    out[USER_REQUEST_COVERAGE_KEY] = (coverage or fallback.get(USER_REQUEST_COVERAGE_KEY) or "")[:900]
    for key in DIMENSION_KEYS:
        raw = data.get(key) if isinstance(data.get(key), dict) else {}
        fb = fallback[key]
        out[key] = _dimension(raw.get("verdict") or fb["verdict"], raw.get("review") or fb["review"])
    # Always rebuild the frontend markdown from normalized fields. The model may
    # still return legacy heuristic sections despite the prompt; the product
    # surface should remain locked to the standard eval dimensions.
    out["review_markdown"] = _render_review_markdown(out)
    return out


def _strip_final_verdict_section(review: str) -> str:
    return re.sub(
        r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:\*\*)?最终判[定断](?:\*\*)?\s*(?:[:：])?\s*[\s\S]*$",
        "",
        str(review or ""),
        flags=re.MULTILINE,
    ).strip()


def _render_review_markdown(data: dict[str, Any]) -> str:
    labels = {
        "task_completion": "任务完成",
        "tool_use": "工具使用",
        "reasoning": "推理路径",
        "instruction_following": "指令遵循",
        "faithfulness": "忠实度",
        "efficiency": "效率",
        "reliability": "可靠性",
    }
    coverage = str(data.get(USER_REQUEST_COVERAGE_KEY) or "").strip()
    if not coverage:
        coverage = "本轮评审会先对照用户原始诉求、最终交付内容和 trace 中的关键动作，判断主 agent 是否覆盖了用户真正要求完成的事项。"
    lines = ["**用户诉求覆盖情况**", coverage, ""]
    for key in DIMENSION_KEYS:
        value = data.get(key) if isinstance(data.get(key), dict) else {}
        lines.append(f"**{labels[key]}** · {value.get('verdict') or 'partial'}")
        lines.append(str(value.get("review") or "基于现有数据，整体判断为中性，需要继续复核。"))
        lines.append("")
    return "\n".join(lines).strip()


def _build_prompt(judge_input: dict[str, Any]) -> str:
    schema = {
        "summary_conclusion": "必须以“结论：”开头的一段自然语言。覆盖整体判断、用户诉求、agent关键动作、交付价值、主要风险。",
        "overall_verdict": "resolved | partial | unresolved",
        USER_REQUEST_COVERAGE_KEY: "80-160字自然语言段落。只判断用户诉求是否被覆盖：用户明确要求了什么、主 agent 实际交付了什么、哪些诉求已覆盖、哪些仍缺证据或未完成。",
        "task_completion": {"verdict": "resolved | partial | unresolved", "review": "80-160字完整自然语言段落。围绕任务完成度判断最终交付是否满足可验收标准，不写启发式优缺点清单。"},
        "tool_use": {"verdict": "correct | suboptimal | wrong", "review": "80-160字完整自然语言段落。评估工具选择、工具顺序、失败恢复和 tool_result 使用情况；可引用 tool_call#N，但不要罗列流水账。"},
        "reasoning": {"verdict": "on_track | drift | redundant | lost", "review": "80-160字完整自然语言段落。评估推理路径是否围绕目标推进、是否有偏航、重复检索、过早下结论或缺少验证。"},
        "instruction_following": {"verdict": "yes | partial | no", "review": "80-160字完整自然语言段落。评估显式指令、边界条件、禁止项、输出格式要求是否被遵循。"},
        "faithfulness": {"verdict": "grounded | partial | hallucinated", "review": "80-160字完整自然语言段落。评估最终说法是否被 trace、工具结果、文件内容或用户输入支持；证据不足时标 partial。"},
        "efficiency": {"verdict": "normal | high | excessive", "review": "80-160字完整自然语言段落。使用真实 token、耗时、工具数、失败数，结合任务复杂度判断是否高效。"},
        "reliability": {"verdict": "clear | minor_issues | blocking_failure", "review": "80-160字完整自然语言段落。评估稳定性、错误恢复、未完成状态、运行异常和结果可复现风险。"},
        "review_markdown": "面向前端详情区的 Markdown。只能包含以下 8 个部分，顺序固定：**用户诉求覆盖情况**、**任务完成**、**工具使用**、**推理路径**、**指令遵循**、**忠实度**、**效率**、**可靠性**。每个部分下面写一段完整自然语言评审。不要写“做得好的部分”“主要不足与风险”“工具使用评审”“最终判定/最终判断”等额外章节。",
    }
    return (
        "你是只读 Agent Critic sidecar，职责是评审主 agent 刚完成的一轮交互。"
        "只能根据给定 trace/transcript/tool_result 判断，不要假设外部事实，不要要求重新提供数据。"
        "如果证据不足，应说明风险和需要复核的点，但仍给出明确结论。\n\n"
        "评分口径：单个工具失败被恢复时不要上升为整体失败；只有影响用户最终验收时才判 unresolved/wrong。"
        "评审内容对齐主流 agent eval 维度：任务完成、工具使用、推理路径、指令遵循、忠实度、效率、可靠性。"
        "不要输出启发式复盘口吻，不要写泛化的优点/缺点列表；每个维度必须是一段可读的自然语言判断，且不得超过160字。"
        "review_markdown 不要追加“最终判定/最终判断”章节。\n\n"
        "【必须输出 JSON 对象，字段完全使用以下 schema】\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "【本轮原始材料】\n"
        f"{json.dumps(judge_input, ensure_ascii=False, indent=2, default=str)}"
    )


def _build_report(
    *,
    cot: dict[str, Any],
    turn: dict[str, Any],
    session_id: str,
    turn_index: int,
    agent_type: str,
    source_event: str,
) -> dict[str, Any]:
    from .session_eval import _build_judge_v3_input, _build_raw_eval_context, extract_turn_metrics
    from .settings import load_critic_settings
    from .providers import load_provider

    context = {
        "session_id": session_id,
        "turn_index": turn_index,
        "cot": cot,
        "turn": turn,
        "transcript": {},
        "otel": {},
        "overview": {},
    }
    metrics = extract_turn_metrics(context)
    settings = load_critic_settings()
    provider_config = settings.to_provider_config()
    base = {
        "schema_version": CRITIC_REPORT_SCHEMA_VERSION,
        "eval_method": "agent_critic_v1",
        "critic_agent": "agent-quality-critic",
        "agent_type": agent_type or cot.get("agent_type") or "unknown",
        "session_id": session_id,
        "turn_index": int(turn_index),
        "source_event": source_event,
        "created_at": utc_now(),
        "provider": settings.provider,
        "model": settings.model,
        "input_sources": [],
    }
    try:
        live_state = load_best_live_critic_state(session_id, turn_index)
        if isinstance(live_state, dict):
            base["live_supervisor"] = {
                key: live_state.get(key)
                for key in (
                    "schema_version",
                    "eval_method",
                    "status",
                    "trigger_source",
                    "critic_runtime",
                    "started_at",
                    "updated_at",
                    "completed_at",
                    "event_count",
                    "event_counts",
                    "turn_index_approx",
                    "risk_level",
                    "live_summary",
                    "observations",
                    "model_status",
                    "model_snapshots",
                )
                if key in live_state
            }
    except Exception:
        pass
    if not settings.enabled:
        structured = _deterministic_structured(reason="Critic 模型已在设置中关闭", metrics=metrics)
        return {**base, "status": "disabled", "structured": structured, **structured, "reason": "Critic 模型已关闭"}
    if not provider_config:
        structured = _deterministic_structured(reason="Critic 模型未配置 API Key", metrics=metrics)
        return {**base, "status": "unconfigured", "structured": structured, **structured, "reason": "Critic 模型未配置 API Key"}

    raw_context = _build_raw_eval_context(
        session_id=session_id,
        turn_index=turn_index,
        cot=cot,
        turn=turn,
        transcript={},
        otel={},
        overview={},
    )
    judge_input = _build_judge_v3_input(metrics, turn, raw_context)
    base["input_sources"] = raw_context.get("sources", [])
    started = time.time()
    provider = load_provider(provider_config)
    response = provider.call(_build_prompt(judge_input))
    latency_ms = int((time.time() - started) * 1000)
    if response.error:
        structured = _deterministic_structured(reason=response.error, metrics=metrics)
        return {
            **base,
            "status": "error",
            "structured": structured,
            **structured,
            "reason": response.error,
            "latency_ms": latency_ms,
        }
    parsed = _extract_json_object(response.output)
    if parsed is None:
        structured = _deterministic_structured(reason="Critic 模型未返回可解析 JSON", metrics=metrics)
        return {
            **base,
            "status": "error",
            "structured": structured,
            **structured,
            "reason": "Critic 模型未返回可解析 JSON",
            "raw_output": response.output[:2000],
            "latency_ms": latency_ms,
        }
    structured = _normalize_structured(parsed, metrics)
    return {
        **base,
        "status": "completed",
        "structured": structured,
        **structured,
        "reason": structured["summary_conclusion"],
        "latency_ms": latency_ms,
        "token_usage": response.performance.token_usage if response.performance else {},
    }


def run_critic_for_cot(
    cot: dict[str, Any],
    *,
    session_id: str | None = None,
    turn_index: int | None = None,
    agent_type: str = "unknown",
    source_event: str = "manual",
    persist_eval: bool = True,
) -> dict[str, Any]:
    sid = str(session_id or cot.get("session_id") or "").strip()
    if not sid:
        raise ValueError("session_id is required")
    turn = _find_turn(cot, turn_index) if turn_index is not None else _latest_turn(cot)
    if turn is None:
        raise ValueError(f"turn not found: {turn_index}")
    idx = int(turn.get("turn_index") or 0)
    existing = load_critic_report(sid, idx)
    if _should_reuse_existing_report(existing, source_event):
        log_critic_event(
            "run_turn_reuse_existing",
            session_id=sid,
            turn_index=idx,
            source_event=source_event,
            existing_status=existing.get("status"),
            existing_source_event=existing.get("source_event"),
        )
        return existing
    log_critic_event(
        "run_turn_start",
        session_id=sid,
        turn_index=idx,
        agent_type=agent_type or cot.get("agent_type") or "unknown",
        source_event=source_event,
        persist_eval=persist_eval,
    )
    running = {
        "schema_version": CRITIC_REPORT_SCHEMA_VERSION,
        "eval_method": "agent_critic_v1",
        "critic_agent": "agent-quality-critic",
        "agent_type": agent_type or cot.get("agent_type") or "unknown",
        "session_id": sid,
        "turn_index": idx,
        "source_event": source_event,
        "created_at": utc_now(),
        "status": "running",
        "summary_conclusion": "结论：Agent Critic 正在评审本轮交互，稍后会刷新为完整自然语言结论。",
        "overall_verdict": "partial",
        "reason": "Agent Critic is running; this placeholder must not be treated as a completed eval.",
    }
    write_critic_report(running)
    report = _build_report(
        cot=cot,
        turn=turn,
        session_id=sid,
        turn_index=idx,
        agent_type=agent_type,
        source_event=source_event,
    )
    write_critic_report(report)
    if persist_eval:
        try:
            from .session_eval import build_turn_eval_report
            from .store import DatasetStore

            turn_eval = build_turn_eval_report(sid, idx, cot=cot)
            DatasetStore().save_turn_eval(turn_eval)
            log_critic_event("turn_eval_persisted", session_id=sid, turn_index=idx)
        except Exception as exc:
            report["persist_error"] = f"{type(exc).__name__}: {exc}"
            write_critic_report(report)
            log_critic_event(
                "turn_eval_persist_error",
                session_id=sid,
                turn_index=idx,
                error=traceback.format_exc()[-4000:],
            )
    log_critic_event("run_turn_done", session_id=sid, turn_index=idx, status=report.get("status"))
    return report


def run_critic_for_session(
    *,
    session_id: str | None,
    turn_index: int | None = None,
    agent_type: str = "unknown",
    source_event: str = "hook",
    wait_seconds: float = 45.0,
    persist_eval: bool = True,
) -> dict[str, Any]:
    log_critic_event(
        "run_session_start",
        session_id=session_id or "",
        turn_index=turn_index,
        agent_type=agent_type,
        source_event=source_event,
        wait_seconds=wait_seconds,
    )
    path = find_cot_file(session_id, wait_seconds=wait_seconds)
    if path is None:
        sid = session_id or "unknown"
        report = {
            "schema_version": CRITIC_REPORT_SCHEMA_VERSION,
            "eval_method": "agent_critic_v1",
            "critic_agent": "agent-quality-critic",
            "agent_type": agent_type,
            "session_id": sid,
            "turn_index": int(turn_index or 0),
            "source_event": source_event,
            "created_at": utc_now(),
            "status": "queued",
            "reason": "等待 cot.json 落盘，Agent Critic 尚未拿到可评审的 turn 数据。",
            "summary_conclusion": "结论：Agent Critic 已排队，但当前还没有找到该会话的 cot.json；确定性断言会先作为兜底展示。",
            "overall_verdict": "partial",
        }
        write_critic_report(report)
        log_critic_event("run_session_queued_no_cot", session_id=sid, turn_index=report["turn_index"])
        return report
    cot = _read_json(path)
    if not cot:
        raise ValueError(f"invalid cot json: {path}")
    # The hook may pass a shortened filename filter (notably Codex rollout ids
    # without the product prefix). Once the cot file is found, persist the
    # report under the canonical session_id inside that cot so the frontend can
    # resolve it by the displayed trace id.
    sid = str(cot.get("session_id") or session_id or path.name.removesuffix("_cot.json"))
    return run_critic_for_cot(
        cot,
        session_id=sid,
        turn_index=turn_index,
        agent_type=agent_type,
        source_event=source_event,
        persist_eval=persist_eval,
    )


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Agent Critic sidecar for a completed turn.")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--turn-index", type=int, default=None)
    parser.add_argument("--agent-type", default="unknown")
    parser.add_argument("--source-event", default="hook")
    parser.add_argument("--wait-seconds", type=float, default=45.0)
    parser.add_argument("--no-persist-eval", action="store_true")
    args = parser.parse_args(argv)
    os.environ["AGENT_QUALITY_EVAL_CRITIC_DISABLE"] = "1"
    try:
        run_critic_for_session(
            session_id=args.session_id,
            turn_index=args.turn_index,
            agent_type=args.agent_type,
            source_event=args.source_event,
            wait_seconds=args.wait_seconds,
            persist_eval=not args.no_persist_eval,
        )
        return 0
    except Exception as exc:
        log_critic_event(
            "runner_error",
            session_id=args.session_id or "",
            turn_index=args.turn_index,
            agent_type=args.agent_type,
            source_event=args.source_event,
            error=traceback.format_exc()[-4000:],
        )
        sid = args.session_id or "unknown"
        report = {
            "schema_version": CRITIC_REPORT_SCHEMA_VERSION,
            "eval_method": "agent_critic_v1",
            "critic_agent": "agent-quality-critic",
            "agent_type": args.agent_type,
            "session_id": sid,
            "turn_index": int(args.turn_index or 0),
            "source_event": args.source_event,
            "created_at": _iso_now(),
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "summary_conclusion": f"结论：Agent Critic 运行失败，当前仅能展示确定性断言兜底。错误为 {type(exc).__name__}: {exc}",
            "overall_verdict": "partial",
        }
        try:
            write_critic_report(report)
        except Exception:
            pass
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
