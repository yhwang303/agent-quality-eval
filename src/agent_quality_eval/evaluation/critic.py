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

CRITIC_REPORT_SCHEMA_VERSION = "agent-critic-v2"
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


def _dimension(
    verdict: str,
    review: str,
    evidence: list[Any] | None = None,
) -> dict[str, Any]:
    return {
        "verdict": str(verdict or "partial"),
        "review": str(review or "").strip(),
        "evidence": _normalize_evidence_list(evidence),
    }


def _normalize_evidence_list(value: Any) -> list[dict[str, str]]:
    """Coerce model-supplied evidence into a bounded list of {ref, quote} rows.

    Accepts list of strings or list of {ref, quote, source} dicts. We keep this
    intentionally permissive because LLMs return both shapes, but always emit
    the same dict shape to the frontend so detail panels stay simple.
    """
    if value is None:
        return []
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, dict):
        candidates = [value]
    elif isinstance(value, list):
        candidates = list(value)
    else:
        return []
    out: list[dict[str, str]] = []
    for raw in candidates:
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                continue
            out.append({"ref": "", "quote": text[:240], "source": ""})
        elif isinstance(raw, dict):
            ref = str(raw.get("ref") or raw.get("reference") or raw.get("location") or raw.get("step") or "").strip()
            quote = str(raw.get("quote") or raw.get("text") or raw.get("excerpt") or raw.get("evidence") or "").strip()
            source = str(raw.get("source") or raw.get("kind") or "").strip()
            if not (ref or quote):
                continue
            out.append({"ref": ref[:80], "quote": quote[:240], "source": source[:40]})
        if len(out) >= 6:
            break
    return out


def _is_final_response_ref(item: dict[str, str]) -> bool:
    ref = str(item.get("ref") or "").strip().lower()
    source = str(item.get("source") or "").strip().lower()
    return ref in {"final_response", "final-response"} or source in {"final_response", "final-response"}


def _compact_claim_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip().lower()


def _filter_claim_evidence(claim: str, evidence: list[dict[str, str]]) -> list[dict[str, str]]:
    claim_key = _compact_claim_text(claim)
    if not claim_key:
        return evidence
    return [
        item for item in evidence
        if not (_is_final_response_ref(item) and _compact_claim_text(item.get("quote") or "") == claim_key)
    ]


def _has_independent_claim_evidence(evidence: list[dict[str, str]]) -> bool:
    return any(not _is_final_response_ref(item) for item in evidence)


def _cap_evidence_keeping_required(items: list[Any], required_refs: set[str], max_total: int = 8) -> list[Any]:
    """Trim an evidence list to `max_total` items without ever dropping the
    deterministically-guaranteed entries in `required_refs` (a naive [:N]
    slice can silently cut off a just-appended guaranteed item).
    """
    required = [i for i in items if isinstance(i, dict) and str(i.get("ref") or "") in required_refs]
    others = [i for i in items if not (isinstance(i, dict) and str(i.get("ref") or "") in required_refs)]
    budget = max(0, max_total - len(required))
    return others[:budget] + required


def _normalize_duration_evidence(dim: dict[str, Any], duration_ms: Any) -> dict[str, Any]:
    """Guarantee the efficiency dimension always cites duration in minutes.

    The model is asked to convert duration_ms to minutes in its evidence
    quote, but can't be trusted to do so consistently (sometimes seconds,
    sometimes raw milliseconds) — explicit user requirement: "用时时间单位
    要统一啊，用min，别一会儿一个毫秒一会儿又是秒". Always replace whatever the
    model wrote for metrics:duration_ms with a canonical minutes-based entry
    computed from the real metric.
    """
    if not isinstance(dim, dict):
        return dim
    minutes = float(duration_ms or 0) / 60000
    if minutes <= 0:
        return dim
    dim = dict(dim)
    duration_ref_prefixes = ("metrics:duration_ms", "metrics:duration_min", "metrics:elapsed_minutes", "metrics:elapsed_seconds")
    evidence = [
        item for item in (dim.get("evidence") or [])
        if not (isinstance(item, dict) and str(item.get("ref") or "").startswith(duration_ref_prefixes))
    ]
    evidence.append({"ref": "metrics:duration_ms", "quote": f"本轮耗时 {minutes:.1f} 分钟。", "source": "trace"})
    dim["evidence"] = _cap_evidence_keeping_required(evidence, {"metrics:duration_ms"})
    review = str(dim.get("review") or "")
    review = re.sub(r"(\d[\d,\.]*)\s*(毫秒|ms)\b", lambda m: (_ms_to_minutes_text(m.group(1)) or m.group(0)), review)
    review = re.sub(r"(\d[\d,\.]*)\s*秒(?!\d)", lambda m: (_seconds_to_minutes_text(m.group(1)) or m.group(0)), review)
    dim["review"] = review
    return dim


def _normalize_efficiency_evidence(dim: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Explicit user requirement: "强制让llm输出token消耗，耗时，工具调用次数三个维度的
    内容作为证据展示出来，不要只在自然语言中进行说明" — efficiency must always surface
    tokens, duration and tool-call count as structured evidence chips, computed
    deterministically so it never depends on the model remembering to cite them.
    """
    dim = _normalize_duration_evidence(dim, metrics.get("duration_ms"))
    if not isinstance(dim, dict):
        return dim
    dim = dict(dim)
    evidence = [item for item in (dim.get("evidence") or []) if isinstance(item, dict)]
    existing_refs = {str(item.get("ref") or "") for item in evidence}
    total_tokens = metrics.get("total_tokens")
    if "metrics:total_tokens" not in existing_refs and total_tokens:
        evidence.append({"ref": "metrics:total_tokens", "quote": f"共消耗 {int(total_tokens)} tokens。", "source": "trace"})
    tool_count = metrics.get("tool_count")
    if "metrics:tool_count" not in existing_refs and tool_count is not None:
        # Deliberately worded to avoid the "工具调用...次数...次" phrasing used by
        # reliability's failure-count evidence, which would otherwise trip the
        # cross-dimension near-duplicate detector on similarity alone.
        evidence.append({"ref": "metrics:tool_count", "quote": f"全程累计触发 {int(tool_count)} 次工具执行动作（成功与失败合计）。", "source": "trace"})
    dim["evidence"] = _cap_evidence_keeping_required(
        evidence, {"metrics:duration_ms", "metrics:total_tokens", "metrics:tool_count"}
    )
    return dim


def _normalize_reliability_evidence(dim: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Explicit user requirement: reliability must consider three aspects —
    失败恢复能力 (failure recovery), 边界情况处理 (edge case handling), 状态是否
    一致 (state consistency) — not just raw tool-call failure counts. This is
    enforced via both the prompt and here structurally: all three must always
    appear as evidence, backfilling any the model omits so content is never
    incomplete.
    """
    if not isinstance(dim, dict):
        return dim
    dim = dict(dim)
    required = ("reliability:failure_recovery", "reliability:edge_case_handling", "reliability:state_consistency")
    evidence = [item for item in (dim.get("evidence") or []) if isinstance(item, dict)]
    existing_refs = {str(item.get("ref") or "") for item in evidence}
    if "reliability:failure_recovery" not in existing_refs:
        unrecovered = metrics.get("unrecovered_failures")
        recovered_steps = metrics.get("error_recovery_steps")
        if unrecovered is not None or recovered_steps is not None:
            quote = f"未恢复失败数为 {int(unrecovered or 0)}，恢复动作 {int(recovered_steps or 0)} 次。"
        else:
            quote = "未提供失败恢复相关的独立指标，按中性处理。"
        evidence.append({"ref": "reliability:failure_recovery", "quote": quote, "source": "trace"})
    if "reliability:edge_case_handling" not in existing_refs:
        evidence.append({"ref": "reliability:edge_case_handling", "quote": "未针对边界情况处理给出独立证据，按中性处理。", "source": "trace"})
    if "reliability:state_consistency" not in existing_refs:
        evidence.append({"ref": "reliability:state_consistency", "quote": "未针对状态一致性给出独立证据，按中性处理。", "source": "trace"})
    dim["evidence"] = _cap_evidence_keeping_required(evidence, set(required))
    return dim


def _ms_to_minutes_text(raw: str) -> str | None:
    try:
        return f"{float(raw.replace(',', '')) / 60000:.1f} 分钟"
    except ValueError:
        return None


def _seconds_to_minutes_text(raw: str) -> str | None:
    try:
        return f"{float(raw.replace(',', '')) / 60:.1f} 分钟"
    except ValueError:
        return None


def _normalize_claims(value: Any) -> list[dict[str, Any]]:
    """Bound claims to a small list of typed, evidence-backed records.

    Frontend expects: {claim, type, verified, evidence}. type is constrained to
    factual/process/quality/unknown so the detail panel can color-code at a
    glance; unknown values are kept rather than dropped.
    """
    if not isinstance(value, list):
        return []
    allowed_types = {"factual", "process", "quality"}
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = str(item.get("claim") or item.get("text") or "").strip()
        if not text:
            continue
        ctype = str(item.get("type") or item.get("category") or "").strip().lower()
        if ctype not in allowed_types:
            ctype = "unknown"
        verified_raw = item.get("verified")
        if isinstance(verified_raw, str):
            verified: bool | None = verified_raw.strip().lower() in {"true", "yes", "passed", "1"}
            if verified_raw.strip().lower() in {"unknown", "uncertain", "n/a", ""}:
                verified = None
        elif isinstance(verified_raw, bool):
            verified = verified_raw
        else:
            verified = None
        evidence = _filter_claim_evidence(text, _normalize_evidence_list(item.get("evidence")))
        if verified is True and not _has_independent_claim_evidence(evidence):
            verified = None
        out.append(
            {
                "claim": text[:240],
                "type": ctype,
                "verified": verified,
                "evidence": evidence,
            }
        )
        if len(out) >= 12:
            break
    return out


def _deterministic_structured(
    *,
    reason: str,
    metrics: dict[str, Any],
    overall_verdict: str = "partial",
) -> dict[str, Any]:
    total = int(metrics.get("total_tokens") or 0)
    elapsed = round(float(metrics.get("duration_ms") or 0.0) / 60000.0, 2)
    calls = int(metrics.get("tool_count") or 0)
    failed = int(metrics.get("tool_error_count") or 0)
    summary = (
        f"结论：Agent Critic 未完成模型评审，当前仅保留确定性断言兜底。"
        f"本轮记录到 {total} tokens、{elapsed} 分钟、{calls} 次工具调用、失败 {failed} 次；"
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
        "efficiency": _dimension("normal", f"运行统计为 {total} tokens、{elapsed} 分钟、{calls} 次工具调用。"),
        "reliability": _dimension("clear" if failed == 0 else "minor_issues", f"工具失败数为 {failed}，由确定性断言继续标记是否阻断。"),
        "claims": [],
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
        out[key] = _dimension(
            raw.get("verdict") or fb["verdict"],
            raw.get("review") or fb["review"],
            raw.get("evidence"),
        )
    out["efficiency"] = _normalize_efficiency_evidence(out["efficiency"], metrics)
    out["reliability"] = _normalize_reliability_evidence(out["reliability"], metrics)
    out["claims"] = _normalize_claims(data.get("claims"))
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
    dimension_schema_with_evidence = lambda verdict_enum, review_hint: {
        "verdict": verdict_enum,
        "review": review_hint,
        "evidence": [
            {
                "ref": "step:N | tool_call:N | transcript:line | otel:span",
                "quote": "trace 中的原文片段，最多 240 字。",
                "source": "transcript | tool_result | trace | otel",
            }
        ],
    }
    schema = {
        "summary_conclusion": "必须以“结论：”开头的一段自然语言。覆盖整体判断、用户诉求、agent关键动作、交付价值、主要风险。",
        "overall_verdict": "resolved | partial | unresolved",
        USER_REQUEST_COVERAGE_KEY: "80-160字自然语言段落。只判断用户诉求是否被覆盖：用户明确要求了什么、主 agent 实际交付了什么、哪些诉求已覆盖、哪些仍缺证据或未完成。",
        "task_completion": dimension_schema_with_evidence(
            "resolved | partial | unresolved",
            "80-160字完整自然语言段落。围绕任务完成度判断最终交付是否满足可验收标准，只看最终产出（文件/成功响应/成功状态），不因过程中的工具失败率或重试次数而下调判定。",
        ),
        "tool_use": dimension_schema_with_evidence(
            "correct | suboptimal | wrong",
            "80-160字完整自然语言段落。只评估**工具选型正确性**：是否选对了工具、参数是否恰当、组合是否合理、有没有漏用应用的工具或错用非最优工具。**不写**失败次数与失败恢复（那属于 reliability），也不写 token/耗时（那属于 efficiency）。",
        ),
        "reasoning": dimension_schema_with_evidence(
            "on_track | drift | redundant | lost",
            "80-160字完整自然语言段落。评估推理路径是否围绕目标推进、是否有偏航、重复检索、过早下结论或缺少验证。",
        ),
        "instruction_following": dimension_schema_with_evidence(
            "yes | partial | no",
            "80-160字完整自然语言段落。评估显式指令、边界条件、禁止项、以及用户指定的实现手段/工具/MCP/skill（哪怕没指名具体是哪个）是否被遵循和真正使用。",
        ),
        "faithfulness": dimension_schema_with_evidence(
            "grounded | partial | hallucinated",
            "80-160字完整自然语言段落。评估 agent 在过程或结论中说的具体声称（数字/状态词/操作声称）是否被 trace、工具结果、文件内容支持；**不评估交付物本身是否存在**，那是 task_completion 的职责。证据不足时标 partial。",
        ),
        "efficiency": dimension_schema_with_evidence(
            "normal | high | excessive",
            "80-160字完整自然语言段落。**只看资源消耗**：token、耗时（必须明确写出本轮总耗时，直接使用 runtime_metrics.elapsed_minutes 字段的分钟数，不允许自己用 duration_ms/elapsed_seconds 换算，也不允许用秒或毫秒）、步骤数、思考步骤、同一工具重复调用次数、无效轮次。**不看**失败率、错误恢复。多次重试导致的资源浪费应在此判 excessive，不在 reliability 判定里重复。",
        ),
        "reliability": dimension_schema_with_evidence(
            "clear | minor_issues | blocking_failure",
            "80-160字完整自然语言段落，**必须依次覆盖三个方面，缺一不可**：①失败恢复能力——未被恢复的失败、崩溃、超时、safety 断言退化；②边界情况处理——是否有意识地处理了输入缺失/异常参数/极端场景等边界条件，而不是只走 happy path；③状态是否一致——多次操作/重试之间状态有没有出现自相矛盾、重复副作用或数据不一致。工具失败但完成恢复应判 minor_issues；仅当有具体失败直接阻断最终交付、或出现状态不一致/边界条件处理缺失导致的实质风险才判 blocking_failure。**不写**成功工具的选型（那属于 tool_use）与资源消耗（那属于 efficiency）。",
        ),
        "claims": [
            {
                "claim": "Agent 在最终回复中或 trace 中作出的具体声称，例如“写入了 foo.py”、“运行了 pytest 全绿”、“查询返回了 N 条”。",
                "type": "factual | process | quality",
                "verified": "true | false | unknown。true 需独立执行证据（tool_result 内容/文件状态/success=true）支持；false 需 evidence 能读出反驳；不确定一律 unknown。禁止拿 assistant 自述当 true 的证据。",
                "evidence": [{"ref": "step:N | tool_call:N", "quote": "支持或反驳该 claim 的原文片段。", "source": "transcript | tool_result"}],
            }
        ],
        "review_markdown": "面向前端详情区的 Markdown。只能包含以下 8 个部分，顺序固定：**用户诉求覆盖情况**、**任务完成**、**工具使用**、**推理路径**、**指令遵循**、**忠实度**、**效率**、**可靠性**。每个部分下面写一段完整自然语言评审。不要写“做得好的部分”“主要不足与风险”“工具使用评审”“最终判定/最终判断”等额外章节。",
    }
    reference = judge_input.get("reference_answer") if isinstance(judge_input.get("reference_answer"), dict) else None
    reference_instructions = ""
    if reference:
        reference_instructions = (
            "\n【Gold Standard 评审约束】\n"
            "本轮用户已上传标准答案。你必须把 standard_answer 作为验收金标准来评估，不要把它当作普通参考资料。\n"
            "保持输出 JSON schema、维度名称和 verdict 枚举完全不变，但每个维度的判断都必须说明主 agent 最终回复与标准答案的符合程度。\n"
            "- task_completion：以是否覆盖标准答案的核心结论和必要步骤作为 resolved/partial/unresolved 的首要依据。\n"
            "- 用户诉求覆盖情况：判断最终回复是否围绕标准答案要求的输出，而不是泛泛回应用户问题。\n"
            "- instruction_following：结合用户原始要求和标准答案中的格式/边界/必要条件判断。\n"
            "- faithfulness：最终回复若偏离、遗漏或新增与标准答案冲突的内容，应标为 partial 或 hallucinated，并指出差异。\n"
            "- tool_use / reasoning：评估工具和推理是否支持得到标准答案，不要仅因过程很长就判好。\n"
            "- efficiency：在答案未更接近标准答案时，额外 token、耗时和工具调用应视为负面。\n"
            "不要改写标准答案，不要替标准答案补充新逻辑；只能基于 standard_answer、rubric、keywords、assertions 与 trace 证据评审。\n"
        )
    if reference and isinstance(reference.get("process_requirements"), dict) and reference.get("process_requirements"):
        reference_instructions += (
            "\nGold process_requirements are present. Evaluate these generic process constraints against trace/tool/reasoning evidence. "
            "Do not hard-code any tool family; judge only the normalized requirements supplied by the user. "
            "If process_requirements are absent, keep the existing trace heuristic.\n"
        )
    return (
        "你是只读 Agent Critic sidecar，职责是评审主 agent 刚完成的一轮交互。"
        "只能根据给定 trace/transcript/tool_result 判断，不要假设外部事实，不要要求重新提供数据。"
        "如果证据不足，应说明风险和需要复核的点，但仍给出明确结论。\n\n"
        "评分口径：单个工具失败被恢复时不要上升为整体失败；只有影响用户最终验收时才判 unresolved/wrong。"
        "评审内容对齐主流 agent eval 维度：任务完成、工具使用、推理路径、指令遵循、忠实度、效率、可靠性。"
        "不要输出启发式复盘口吻，不要写泛化的优点/缺点列表；每个维度必须是一段可读的自然语言判断，且不得超过160字。"
        "review_markdown 不要追加“最终判定/最终判断”章节。\n"
        "【证据先行硬约束】每个维度都必须给出 evidence 数组，且 evidence 必须来自 transcript / tool_result / trace / otel 的具体引用，"
        "可以是 step:N、tool_call:N、transcript 行号或 otel span 名；找不到证据就把该维度的 verdict 朝保守一侧打分。\n"
        "【三维度语义边界——不允许重叠】tool_use、efficiency、reliability 长期被误当成“同一件事”，本次严格切分：\n"
        "  * tool_use 只回答“**选对了工具吗、参数对吗、组合合理吗**”——是选型正确性问题，与失败次数无关。\n"
        "  * efficiency 只回答“**用了多少资源**”——只看 tokens、耗时、步骤、重复调用；不看失败率、不看恢复情况。\n"
        "  * reliability 回答“**过程是否稳健**”，覆盖三个子方面（缺一不可）：失败恢复能力、边界情况处理、状态一致性，不看正常成功的工具选型，也不是单纯数一数工具调用失败次数。\n"
        "过程颠簸但最终恢复 → tool_use 与 reliability 应给中性，efficiency 判 excessive。\n"
        "过程颠簸且最终阻断交付 → reliability 独占 blocking_failure。\n"
        "工具选错或参数错 → tool_use 独占 wrong，与失败次数无关。\n"
        "【每维度证据数量】每个维度的 evidence 数组必须包含**至少 2 条**引用（找不到 2 条时至少 1 条），最多 4 条。\n"
        "【三步语义抽取——先做这个，再举证】task_completion / instruction_following / faithfulness 长期被误当成同一件事，本次强制先做抽取动作再找证据：\n"
        "  ① 从 user_query 抽出【主诉求】：用户到底要什么。示例：主诉求=生成新版 exe。**只有 task_completion 引用主诉求**。\n"
        "  ② 从 user_query 抽出【约束/边界/禁止项清单】：主诉求之外用户附带的规则，覆盖两类，缺一不可：\n"
        "     (a) 边界/禁止类：例如“不要破坏现有结构”“按 dist 现有格式”“只改 X 不动 Y”“中英文一致”“禁止硬编码”。\n"
        "     (b) 手段/工具/harness 指定类：用户明确点名了“要用什么方式/工具/MCP/skill 去做”，哪怕没有指名具体是哪一个（例如“你用 MCP 看一下”“调用某个 skill 处理”“用工具去查”“借助 XX 能力完成”）。"
        "只要用户点名了实现手段本身，这就是独立于主诉求的一条约束——即使目标和手段写在同一句话里也必须拆开：目标进主诉求，手段进约束清单，**不允许把“用 MCP 看一下项目进度”整体揉进主诉求就当没有约束**。\n"
        "**只有 instruction_following 引用这个清单**，逐条对照 agent 是否遵守：对手段类约束，要看 trace 里是否真的调用了对应类型的工具/MCP/skill（例如是否有 tool_call 命中 MCP 工具），而不是绕开约束直接凭自身知识/其他方式达成目标。"
        "若用户原话里除主诉求外**确实没有任何附加约束**（包括没有指定任何手段/工具/MCP/skill），instruction_following 才可以给 verdict=yes 并写 quote=\"用户未提附加约束\"；**只要用户指定了任何手段，哪怕很笼统，都不允许再套用这句话**，必须逐条核实该手段是否被真正使用。**禁止把主诉求当作约束再讲一遍**。\n"
        "  ③ 从 final_response 抽出【agent 具体声称清单】：faithfulness 回答的是“agent 说过的话是否属实”，不是“东西有没有交付”——判断对象是**话**，不是**物**。合格声称必须是可独立核验的过程性/数量性/状态性陈述，例如“265 项测试全部通过”“doctor 通过”“已升 v6→v7”“已修改 X 处”“共调用了 N 次工具”。"
        "**反例（不算声称，禁止当作 faithfulness 证据）**：单纯陈述“生成/交付/完成了 XX 文件或结果”本身——这只是复述交付物，属于 task_completion 的判断对象，即便它出现在 final_response 里也不能被 faithfulness 拿来当 claim。自检方法：把这句话从 final_response 里去掉后，task_completion 的证据是否也随之消失——如果是，说明它就是交付物本身，两个维度不能共用。"
        "**若逐句检查后除交付物陈述外确实没有其他可验证声称**，faithfulness 必须写 claim=\"agent 未在回复中提出独立于交付物的可验证声称\"，evidence 用 final_response:无独立声称，verdict 给 grounded（没有可证伪的声称就谈不上失实），**不允许为了凑证据数量硬把交付物包装成一个 claim**。"
        "**只有 faithfulness 引用这些声称**，逐条用 tool_result / metrics 反查是否属实（例如是否有 pytest 调用、是否有对应文件写入等）。**禁止把交付物本身当作声称**——交付物是 task_completion 的事，不是 faithfulness 的事。\n"
        "【每维度证据 ref 前缀白名单——违反即失败】ref 必须以下列前缀开头，且**不同维度不允许共用同一 ref**：\n"
        "  - task_completion → 允许前缀：final_response、assertion:xxx、file:path、artifact:xxx。quote 描述【主诉求】达成情况：用户要 X，交付了 X 或未交付 X。**严禁 tool_call#N**（那是过程证据不是交付证据）。\n"
        "  - tool_use → 允许前缀：tool_choice:tool_name、metrics:tool_kind_count、metrics:tool_count。quote 写选型判断，例如 quote=\"多次使用 Read 循环读文件，用 Grep 一次搜索更合适\"。**严禁 metrics:tool_error_count**。\n"
        "  - reasoning → 允许前缀：step:strategy_shift、step:plan_update、step:thinking、metrics:strategy_shifts_count。quote 写路径特征（绕路/直达/重复），不放 transcript 原文。\n"
        "  - instruction_following → 允许前缀：user_query:约束点。quote 必须逐条写【约束/边界/禁止项】的具体名字，边界类和手段/工具/MCP/skill 指定类都算，例如 quote=\"约束『不要破坏现有结构』：agent 未破坏\" 或 quote=\"约束『需用 MCP』：agent 是否调用了 MCP 工具\"。**严禁把主诉求（生成 exe）当约束再讲一遍**。只有用户确实没指定任何边界或手段时才明写 quote=\"用户未提附加约束\"，evidence 用 user_query:无附加约束。\n"
        "  - faithfulness → 允许前缀 **必须成对**：claim:XX + tool_call#N。claim 必须是从 final_response 抽出的**具体可验证声称**（数字/状态词/操作声称，例如“265 项测试通过”“doctor 通过”“v6→v7”），而不是“生成了 exe”这种交付物本身。**严禁只用 final_response 或只用一般 tool_call**。**例外**：若 final_response 除交付物陈述外确实没有其他可验证声称，允许写 claim:agent未提出独立声称 + final_response:无独立声称 这一对，quote=\"agent 未在回复中提出独立于交付物的可验证声称\"。\n"
        "  - efficiency → 允许前缀：metrics:total_tokens、metrics:input_tokens、metrics:output_tokens、metrics:duration_ms、metrics:tool_count、metrics:step_count、metrics:tool_kind_count、metrics:thinking_steps、metrics:repeated_tool_calls。**必须同时包含 ref=metrics:duration_ms、ref=metrics:total_tokens、ref=metrics:tool_count 这三条**（分别对应耗时/token消耗/工具调用次数，三者都要以证据条目的形式列出，不能只在 review 自然语言里提一句；duration_ms 的分钟数值必须直接取自 runtime_metrics.elapsed_minutes 字段，不要自己换算），可以再加其他前缀补充。**严禁 metrics:tool_error_count**。\n"
        "  - reliability → 允许前缀：step:error_recovery、metrics:tool_error_count、metrics:unrecovered_failures、assertion:safety_xxx、event:crash、event:timeout、**reliability:failure_recovery、reliability:edge_case_handling、reliability:state_consistency**。**必须同时包含这三条固定 ref**：reliability:failure_recovery（quote 体现“是否恢复/是否影响交付”，例如“N 次失败全部恢复，未影响交付”）、reliability:edge_case_handling（quote 说明是否处理了边界/异常输入，没处理就明说“未观察到边界情况处理”）、reliability:state_consistency（quote 说明多次操作间状态是否一致，没有可判断的证据就明说“未观察到状态不一致的证据”）。**只报失败次数不说恢复情况/边界处理/状态一致性，视为证据不合格**。**严禁 tool_call#N**（那是原始工具事件不是恢复观察点）。\n"
        "【数字方向硬约束】效率维度：数字大 = 消耗多 = verdict 朝 excessive；可靠性维度：**未恢复且影响交付**的失败数 > 0 = verdict 朝 minor_issues 或 blocking_failure；失败已全部恢复、未影响交付时，即使失败次数不少，也应判 clear 或 minor_issues，不得仅凭失败次数本身判 blocking_failure。绝不允许写“消耗更多但更高效”这种前后矛盾的话。\n"
        "证据 ref 必须能映射到上面白名单前缀之一；不允许拿 assistant 自述做证据。\n"
        "【拒绝表面合规】不要因为文件名/工具名/字段对就判 PASS：必须确认对应内容真实非空、与用户诉求一致；"
        "巧合命中、空文件、声称做过但 trace 找不到对应 tool_result 的情况一律视为不达标。\n"
        "【三维度严格解耦】task_completion 只回答“最终产物/交付是否达到用户可验收标准”。"
        "工具调用失败次数、重试次数、错误恢复次数不进入 task_completion 判定。"
        "只要 trace 中最终存在满足用户诉求的成功产出（例如目标文件已生成、目标 API 已调用成功、目标数据已返回），"
        "即便过程中经历了多次工具失败与重试，task_completion 也应判 resolved；"
        "只有能明确指出“某个具体失败没被恢复且直接阻断了最终交付”，才可以判 unresolved。"
        "过程中的高失败率、多次重试、耗时长应归入 efficiency（判 excessive）与 reliability（判 minor_issues 直至 blocking_failure），"
        "不允许在 task_completion 与用户诉求覆盖情况维度里写“工具失败率高，因此交付存在风险/影响验收”这类跨维度串味结论。"
        "reliability 判 blocking_failure 仅限“最终交付被阻断”的情况；只是过程颠簸但最终产出成功，应判 minor_issues。\n"
        "【claims 提取与 verified 判定】请额外输出 claims 数组：从最终回复或 trace 中抽取 agent 作出的可验证声称（factual/process/quality）。"
        "verified 三态判定必须严格：\n"
        "  - verified=true 必须要有独立的执行证据支持该声称（tool_result 中的 stdout/文件内容、显式 success=true、apiResult 明确成功、observed 文件系统状态等）；"
        "不允许拿 assistant 自己在最终回复里的表述当验证证据（自证不算证）；也不允许仅凭“工具调用发生过”就判 true。\n"
        "  - verified=false 必须能从 evidence 中读出反驳信号（明确 error / 空输出 / 输出与声称不一致 / success=false）；"
        "仅仅“找不到直接执行证据”不足以判 false。\n"
        "  - 找不到充分证据时一律判 unknown，并在 evidence 中说明缺哪类证据；不要为了给出“确定”结论就硬判 true 或 false。\n"
        "  - 数据源类型影响判定：来自 IDE hook 的真实 trace 通常有 tool_result 内容可作为独立证据；"
        "来自 OTel/telemetry 类的上传 trace（source_event 为 user-upload-eval）通常只有 metadata（success/error_type/duration），没有 stdout；"
        "此时工具类声称的 verified 应偏向 unknown，除非 attributes 里有 success=true 明确对应该声称。\n"
        "  - evidence 与 verified 必须语义一致：若 evidence 引用的是支持性文本，verified 不应为 false；若 evidence 引用的是反驳性文本，verified 不应为 true。\n"
        f"{reference_instructions}\n"
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
    reference_answer: dict[str, Any] | None = None,
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
    if reference_answer:
        base["reference_answer"] = reference_answer
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
    if reference_answer:
        judge_input["reference_answer"] = reference_answer
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
    reference_answer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sid = str(session_id or cot.get("session_id") or "").strip()
    if not sid:
        raise ValueError("session_id is required")
    turn = _find_turn(cot, turn_index) if turn_index is not None else _latest_turn(cot)
    if turn is None:
        raise ValueError(f"turn not found: {turn_index}")
    idx = int(turn.get("turn_index") or 0)
    existing = load_critic_report(sid, idx)
    if not reference_answer and _should_reuse_existing_report(existing, source_event):
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
        has_reference_answer=bool(reference_answer),
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
    if reference_answer:
        running["reference_answer"] = reference_answer
    write_critic_report(running)
    report = _build_report(
        cot=cot,
        turn=turn,
        session_id=sid,
        turn_index=idx,
        agent_type=agent_type,
        source_event=source_event,
        reference_answer=reference_answer,
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
