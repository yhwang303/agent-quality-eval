"""Live Agent Critic supervisor state.

This module is intentionally lightweight enough to be called from IDE hooks
while the primary agent is still working. Each hook invocation records a
single "pulse" into a rolling read-only supervisor state. The post-turn critic
still produces the final report; this live state is the in-flight supervision
surface.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .critic import data_root, log_critic_event

LIVE_CRITIC_SCHEMA_VERSION = "agent-critic-live-v1"
LIVE_START_EVENTS = {"UserPromptSubmit", "SessionStart", "user_prompt_submit", "start"}
LIVE_STOP_EVENTS = {"Stop", "SessionEnd", "StopFailure", "SubagentStop", "stop", "session_end"}
LIVE_FAILURE_EVENTS = {
    "StopFailure",
    "PostToolUseFailure",
    "PermissionDenied",
    "tool_error",
    "error",
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value or "unknown")[:180]


def live_critic_root() -> Path:
    return data_root() / "critic_live"


def live_critic_state_path(session_id: str) -> Path:
    return live_critic_root() / _safe_name(session_id) / "state.json"


def live_critic_turn_path(session_id: str, turn_index: int) -> Path:
    return live_critic_root() / _safe_name(session_id) / f"turn_{int(turn_index)}.json"


def load_live_critic_state(session_id: str, turn_index: int | None = None) -> dict[str, Any] | None:
    path = live_critic_turn_path(session_id, turn_index) if turn_index is not None else live_critic_state_path(session_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema_version") != LIVE_CRITIC_SCHEMA_VERSION:
        return None
    return data


def write_live_critic_state(state: dict[str, Any]) -> Path:
    session_id = str(state.get("session_id") or "unknown")
    path = live_critic_state_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    log_critic_event(
        "live_state_written",
        session_id=session_id,
        status=state.get("status"),
        hook_event=state.get("last_event"),
        path=str(path),
    )
    turn_index = _state_turn_index(state)
    if turn_index is not None:
        turn_path = live_critic_turn_path(session_id, turn_index)
        turn_tmp = turn_path.with_suffix(".tmp")
        turn_tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(turn_tmp, turn_path)
        log_critic_event(
            "live_turn_state_written",
            session_id=session_id,
            turn_index=turn_index,
            status=state.get("status"),
            hook_event=state.get("last_event"),
            path=str(turn_path),
        )
    return path


def _state_turn_index(state: dict[str, Any]) -> int | None:
    try:
        value = int(state.get("turn_index_approx") or 0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _new_state(session_id: str, agent_type: str, source_event: str) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema_version": LIVE_CRITIC_SCHEMA_VERSION,
        "eval_method": "live_critic_supervisor_v1",
        "critic_agent": "agent-quality-live-supervisor",
        "trigger_source": "ide_hook_live",
        "critic_runtime": "hook_pulse_sidecar",
        "session_id": session_id,
        "agent_type": agent_type or "unknown",
        "status": "running",
        "started_at": now,
        "updated_at": now,
        "completed_at": None,
        "last_event": source_event,
        "event_count": 0,
        "event_counts": {},
        "turn_index_approx": 1 if source_event in LIVE_START_EVENTS else 0,
        "risk_level": "low",
        "live_summary": "Live Critic Supervisor is watching this session in read-only mode.",
        "observations": [],
        "recent_events": [],
        "model_status": "not_called",
        "model_snapshots": [],
    }


def _bounded_append(items: list[Any], item: Any, limit: int) -> list[Any]:
    items.append(item)
    if len(items) > limit:
        del items[: len(items) - limit]
    return items


def _risk_rank(level: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(level, 0)


def _raise_risk(state: dict[str, Any], level: str) -> None:
    if _risk_rank(level) > _risk_rank(str(state.get("risk_level") or "low")):
        state["risk_level"] = level


def _observation_for_event(source_event: str, event_payload: dict[str, Any] | None) -> tuple[str, str] | None:
    event = source_event or "unknown"
    payload = event_payload or {}
    tool = payload.get("tool") or payload.get("tool_name") or ""
    if event in LIVE_START_EVENTS:
        return "info", "Turn/session started; live supervisor attached before final evaluation."
    if event in LIVE_FAILURE_EVENTS or "Failure" in event or "Denied" in event:
        return "high", f"Risk event observed during execution: {event}{f' ({tool})' if tool else ''}."
    if event in {"PreToolUse", "beforeShellExecution", "beforeMCPExecution"}:
        return "info", f"Tool/action about to run{f': {tool}' if tool else ''}."
    if event in {"PostToolUse", "afterShellExecution", "afterMCPExecution"}:
        return "info", f"Tool/action completed{f': {tool}' if tool else ''}."
    if event in {"FileChanged", "afterFileEdit", "afterTabFileEdit"}:
        return "medium", "File mutation observed; final response should be backed by validation or clear evidence."
    if event in {"SubagentStart", "TaskCreated"}:
        return "info", "Primary agent delegated work to a subagent/task."
    if event in {"SubagentStop", "TaskCompleted"}:
        return "info", "Subagent/task completed; final critic should reconcile delegated output with the user request."
    if event in {"Stop", "SessionEnd", "stop", "session_end"}:
        return "info", "Turn/session boundary reached; live supervision will hand off to the final Turn Critic."
    return None


def _summarize_state(state: dict[str, Any]) -> str:
    counts = state.get("event_counts") if isinstance(state.get("event_counts"), dict) else {}
    errors = sum(int(counts.get(name, 0) or 0) for name in LIVE_FAILURE_EVENTS)
    file_events = sum(int(counts.get(name, 0) or 0) for name in ("FileChanged", "afterFileEdit", "afterTabFileEdit"))
    subagents = sum(int(counts.get(name, 0) or 0) for name in ("SubagentStart", "TaskCreated"))
    status = state.get("status") or "running"
    return (
        f"Live supervisor {status}: observed {state.get('event_count', 0)} hook events, "
        f"risk={state.get('risk_level', 'low')}, failures={errors}, "
        f"file_changes={file_events}, subagent_delegations={subagents}. "
        "Final judgment remains pending until the post-turn critic sees the completed trace."
    )


def run_live_pulse(
    *,
    session_id: str,
    agent_type: str,
    source_event: str,
    event_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required")
    event = str(source_event or "unknown")
    state = load_live_critic_state(sid)
    if state is None or (event in LIVE_START_EVENTS and state.get("status") == "completed"):
        state = _new_state(sid, agent_type, event)

    now = _utc_now()
    state["agent_type"] = agent_type or state.get("agent_type") or "unknown"
    state["updated_at"] = now
    state["last_event"] = event
    state["event_count"] = int(state.get("event_count") or 0) + 1
    counts = state.get("event_counts") if isinstance(state.get("event_counts"), dict) else {}
    counts[event] = int(counts.get(event) or 0) + 1
    state["event_counts"] = counts
    if event in LIVE_START_EVENTS and state["event_count"] > 1:
        state["turn_index_approx"] = int(state.get("turn_index_approx") or 0) + 1
        state["status"] = "running"
        state["completed_at"] = None
        state["risk_level"] = "low"
        state["observations"] = []

    recent = state.get("recent_events") if isinstance(state.get("recent_events"), list) else []
    _bounded_append(
        recent,
        {
            "ts": now,
            "event": event,
            "tool": (event_payload or {}).get("tool") or (event_payload or {}).get("tool_name"),
        },
        32,
    )
    state["recent_events"] = recent

    obs = _observation_for_event(event, event_payload)
    if obs:
        severity, message = obs
        observations = state.get("observations") if isinstance(state.get("observations"), list) else []
        _bounded_append(
            observations,
            {"ts": now, "severity": severity, "event": event, "message": message},
            20,
        )
        state["observations"] = observations
        if severity in {"medium", "high"}:
            _raise_risk(state, severity)

    if event in LIVE_STOP_EVENTS:
        state["status"] = "completed" if event not in {"StopFailure"} else "error"
        state["completed_at"] = now
    else:
        state["status"] = "running"

    state["live_summary"] = _summarize_state(state)
    _maybe_add_model_snapshot(state)
    write_live_critic_state(state)
    return state


def _maybe_add_model_snapshot(state: dict[str, Any]) -> None:
    event = str(state.get("last_event") or "")
    should_call = (
        event in LIVE_FAILURE_EVENTS
        or "Failure" in event
        or (int(state.get("event_count") or 0) % 12 == 0 and _seconds_since_last_model(state) > 45)
    )
    if not should_call:
        return
    try:
        from .providers import load_provider
        from .settings import load_critic_settings

        settings = load_critic_settings()
        provider_config = settings.to_provider_config()
        if not provider_config:
            state["model_status"] = "unconfigured"
            return
        prompt = (
            "You are a read-only live critic supervisor watching an agent while it works. "
            "Given recent hook events and observations, write one concise Chinese supervision note. "
            "Do not claim final success; final judgment happens after Stop. Return JSON with keys "
            "risk_level (low|medium|high), note, watch_next.\n\n"
            f"{json.dumps({k: state.get(k) for k in ('session_id', 'agent_type', 'event_count', 'event_counts', 'observations', 'recent_events')}, ensure_ascii=False, default=str)}"
        )
        response = load_provider(provider_config).call(prompt)
        if response.error:
            state["model_status"] = "error"
            state["model_error"] = response.error
            return
        data = _extract_json(response.output)
        if not isinstance(data, dict):
            state["model_status"] = "parse_error"
            state["model_raw"] = response.output[:1000]
            return
        snapshots = state.get("model_snapshots") if isinstance(state.get("model_snapshots"), list) else []
        _bounded_append(
            snapshots,
            {
                "ts": _utc_now(),
                "model": settings.model,
                "provider": settings.provider,
                "risk_level": data.get("risk_level") or state.get("risk_level"),
                "note": data.get("note") or "",
                "watch_next": data.get("watch_next") or "",
            },
            6,
        )
        state["model_snapshots"] = snapshots
        state["model_status"] = "completed"
        state["last_model_at"] = time.time()
        if data.get("risk_level") in {"medium", "high"}:
            _raise_risk(state, str(data.get("risk_level")))
    except Exception as exc:
        state["model_status"] = "error"
        state["model_error"] = f"{type(exc).__name__}: {exc}"


def _seconds_since_last_model(state: dict[str, Any]) -> float:
    try:
        return time.time() - float(state.get("last_model_at") or 0.0)
    except Exception:
        return 999999.0


def _extract_json(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    candidates = [raw]
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a live Agent Critic supervisor pulse.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--agent-type", default="unknown")
    parser.add_argument("--source-event", default="hook")
    parser.add_argument("--payload-json", default="")
    args = parser.parse_args(argv)
    os.environ["AGENT_QUALITY_EVAL_LIVE_CRITIC_DISABLE"] = "1"
    payload: dict[str, Any] | None = None
    if args.payload_json:
        try:
            data = json.loads(args.payload_json)
            payload = data if isinstance(data, dict) else None
        except Exception:
            payload = None
    try:
        run_live_pulse(
            session_id=args.session_id,
            agent_type=args.agent_type,
            source_event=args.source_event,
            event_payload=payload,
        )
    except Exception as exc:
        log_critic_event(
            "live_pulse_error",
            session_id=args.session_id,
            agent_type=args.agent_type,
            source_event=args.source_event,
            error=f"{type(exc).__name__}: {exc}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
