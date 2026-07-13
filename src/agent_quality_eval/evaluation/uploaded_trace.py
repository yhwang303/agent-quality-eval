"""Normalize a user-uploaded trace into the cot.json shape used by the rest
of the eval pipeline.

The product requirement: users may bring traces from many sources (Langfuse,
OpenTelemetry, Knot, hand-rolled JSON, ...). We do NOT hard-code a per-source
schema — instead we duck-type a small set of common shapes, then fall back to
a single-turn wrapper that just preserves the raw payload as one step. The
goal is "good enough for LLM critic to reason about", not "full fidelity to
the IDE-hook cot.json".

Concretely the normalizer guarantees:

* `session_id` is set (caller provides; we don't generate here so the upload
  endpoint can choose the storage path first).
* `turns` is a non-empty list, each turn has `turn_index`, `user_query`,
  `final_response`, `steps`. Token / latency fields are intentionally absent
  when the source doesn't carry them — the LLM critic must tolerate missing
  metrics, and we already do via `_deterministic_structured` fallback.
* `_uploaded = True` and `_uploaded_source` are set so the project resolver
  can short-circuit to the virtual "Uploaded Traces" project.

This module is intentionally pure: no filesystem writes, no time.now() calls
that aren't passed in. The upload API decides where to land the result.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

UPLOADED_PROJECT_ID = "__uploaded__"
UPLOADED_PROJECT_NAME = "Uploaded Traces"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _coerce_text(value: Any, *, max_chars: int = 8000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    text = text.strip()
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def _looks_like_message_list(value: Any) -> bool:
    """Match Langfuse/OpenAI-style message arrays: [{role, content}, ...]."""
    if not isinstance(value, list) or not value:
        return False
    head = value[0]
    return isinstance(head, dict) and "role" in head and ("content" in head or "parts" in head)


def _pick_first(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, "", [], {}):
            return payload[key]
    return None


def _extract_user_query_from_messages(messages: list[Any]) -> str:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").lower()
        if role in {"user", "human", "input"}:
            return _coerce_text(msg.get("content") or msg.get("parts"), max_chars=4000)
    return ""


def _extract_final_response_from_messages(messages: list[Any]) -> str:
    last_assistant = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").lower()
        if role in {"assistant", "model", "ai", "output"}:
            last_assistant = _coerce_text(msg.get("content") or msg.get("parts"), max_chars=8000)
    return last_assistant


def _step_from_event(event: Any, idx: int) -> dict[str, Any]:
    if not isinstance(event, dict):
        return {
            "step_index": idx,
            "turn_index": 1,
            "step_type": "tool_result_input",
            "content": _coerce_text(event, max_chars=4000),
            "metadata": {},
            "tool_name": "",
            "tool_use_id": "",
            "tokens": 0,
        }
    step_type = str(
        event.get("step_type")
        or event.get("type")
        or event.get("kind")
        or event.get("event_type")
        or "tool_execution"
    )
    tool_name = str(event.get("tool_name") or event.get("name") or event.get("tool") or "")
    content = (
        event.get("content")
        or event.get("output")
        or event.get("result")
        or event.get("input")
        or event.get("text")
        or event
    )
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    return {
        "step_index": idx,
        "turn_index": int(event.get("turn_index") or 1),
        "step_type": step_type,
        "content": _coerce_text(content, max_chars=4000),
        "metadata": dict(metadata),
        "tool_name": tool_name,
        "tool_use_id": str(event.get("tool_use_id") or event.get("id") or ""),
        "tokens": int(event.get("tokens") or 0),
    }


def _normalize_steps(raw_steps: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_steps, list):
        return []
    return [_step_from_event(item, idx) for idx, item in enumerate(raw_steps)]


def _build_turn(
    *,
    turn_index: int,
    user_query: str,
    final_response: str,
    steps: list[dict[str, Any]],
    raw_payload: Any,
) -> dict[str, Any]:
    """Final shape mirrors the cot.json turn structure used by session_eval."""
    return {
        "turn_index": turn_index,
        "user_query": user_query,
        "final_response": final_response,
        "steps": steps,
        "tool_calls": [step.get("tool_name") for step in steps if step.get("tool_name")],
        "strategy_shifts": 0,
        "thinking_depth": 0,
        "total_steps": len(steps),
        "has_error_recovery": False,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        "complexity_score": 0,
        "_uploaded_raw": raw_payload if isinstance(raw_payload, (dict, list)) else None,
    }


def _looks_like_otel_event_stream(payload: dict[str, Any]) -> bool:
    """OTel-style trace: {events: [{event_name, attributes, ...}]}.

    We match on the shape rather than the vendor. Any exporter that emits an
    `events` list where each entry has `event_name` + `attributes` (which is
    what OpenTelemetry SDKs produce for log-style events) will land here. In
    particular the Claude Code telemetry export follows this exact schema, but
    so do many home-grown OTel bridges — we don't hard-code either one.
    """
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        return False
    head = events[0]
    if not isinstance(head, dict):
        return False
    return "event_name" in head and "attributes" in head


def _otel_event_ts(event: dict[str, Any]) -> str:
    for key in ("ts", "observed_ts", "timestamp"):
        val = event.get(key)
        if val:
            return str(val)
    attrs = event.get("attributes") if isinstance(event.get("attributes"), dict) else {}
    for key in ("event.timestamp", "timestamp"):
        val = attrs.get(key) if isinstance(attrs, dict) else None
        if val:
            return str(val)
    return ""


# Events that don't map to anything a user cares about in the trace tree.
# Defaults follow the Claude Code telemetry spec but are matched by name only —
# any exporter that emits similarly-named registration/lifecycle events will
# be filtered too, without hard-coding the exporter identity.
_OTEL_NOISE_EVENT_NAMES = {
    "hook_registered",
    "hook_execution_start",
    "hook_execution_complete",
    "hook_execution_end",
    "plugin_loaded",
    "plugin_unloaded",
    "mcp_server_connection",
    "mcp_server_disconnection",
    "session_start",
    "session_end",
}


def _otel_event_is_noise(event_name: str) -> bool:
    return str(event_name or "").lower() in _OTEL_NOISE_EVENT_NAMES


def _otel_tokens_from_attrs(attrs: dict[str, Any]) -> int:
    """Best-effort output-token count from an api_request event."""
    for key in ("output_tokens", "completion_tokens"):
        val = attrs.get(key)
        try:
            n = int(val) if val is not None else 0
        except (TypeError, ValueError):
            n = 0
        if n:
            return n
    return 0


def _otel_duration_ms(attrs: dict[str, Any]) -> int | None:
    for key in ("duration_ms", "latency_ms"):
        val = attrs.get(key)
        try:
            if val is None:
                continue
            return int(float(val))
        except (TypeError, ValueError):
            continue
    return None


def _otel_thinking_step(event: dict[str, Any], idx: int, turn_index: int) -> dict[str, Any]:
    """`api_request` event → LLM Thinking step with token/model chips."""
    attrs = event.get("attributes") if isinstance(event.get("attributes"), dict) else {}
    duration_ms = _otel_duration_ms(attrs)
    step: dict[str, Any] = {
        "step_index": idx,
        "turn_index": turn_index,
        "step_type": "thinking_intermediate",
        "content": "",  # OTel telemetry does not carry assistant text.
        "metadata": {
            "otel_event_name": "api_request",
            "ts": _otel_event_ts(event),
            "model": attrs.get("model"),
            "input_tokens": attrs.get("input_tokens"),
            "output_tokens": attrs.get("output_tokens"),
            "cache_read_tokens": attrs.get("cache_read_tokens"),
            "cache_creation_tokens": attrs.get("cache_creation_tokens"),
            "cost_usd": attrs.get("cost_usd"),
            "duration_ms": duration_ms,
            "query_source": attrs.get("query_source"),
        },
        "tool_name": "",
        "tool_use_id": "",
        "tokens": _otel_tokens_from_attrs(attrs),
    }
    if duration_ms is not None:
        step["duration_ms"] = duration_ms
    return step


def _otel_tool_pair_step(
    decision: dict[str, Any] | None,
    result: dict[str, Any] | None,
    idx: int,
    turn_index: int,
) -> list[dict[str, Any]]:
    """Turn a tool_decision/tool_result pair into 1 or 2 canonical steps.

    Frontend renders `tool_decision` as "LLM Thinking → Use Tool <tool>" and
    `tool_execution` as "Tool Execution → <tool>". Emitting both when we have
    both events matches how real IDE-hooked traces display, so the uploaded
    trace visually blends in with the rest of the platform.
    """
    steps: list[dict[str, Any]] = []
    d_attrs = (decision or {}).get("attributes") if decision else None
    d_attrs = d_attrs if isinstance(d_attrs, dict) else {}
    r_attrs = (result or {}).get("attributes") if result else None
    r_attrs = r_attrs if isinstance(r_attrs, dict) else {}
    tool_name = str(
        d_attrs.get("tool_name")
        or r_attrs.get("tool_name")
        or ""
    )
    tool_use_id = str(
        d_attrs.get("tool_use_id")
        or r_attrs.get("tool_use_id")
        or ""
    )
    tool_input = d_attrs.get("tool_input") or d_attrs.get("tool_parameters") or r_attrs.get("tool_input")
    if decision:
        steps.append({
            "step_index": idx,
            "turn_index": turn_index,
            "step_type": "tool_decision",
            "content": _coerce_text(tool_input, max_chars=2000),
            "metadata": {
                "otel_event_name": "tool_decision",
                "ts": _otel_event_ts(decision),
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "tool_input": tool_input,
                "decision": d_attrs.get("decision"),
                "decision_source": d_attrs.get("source") or d_attrs.get("decision_source"),
            },
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "tokens": 0,
        })
    if result:
        duration_ms = _otel_duration_ms(r_attrs)
        success_raw = r_attrs.get("success")
        # OTel exporters send `success` as bool OR the string "true"/"false".
        # Coerce so downstream step_type / UI logic doesn't miss failures.
        if isinstance(success_raw, bool):
            success = success_raw
        elif isinstance(success_raw, str):
            success = success_raw.strip().lower() != "false"
            if success_raw.strip().lower() not in {"true", "false"}:
                success = None
        elif success_raw is None:
            success = None
        else:
            success = bool(success_raw)
        error = r_attrs.get("error") or r_attrs.get("error_type")
        content_source = r_attrs.get("output") or r_attrs.get("result") or (
            f"error: {error}" if error else ("success" if success else "")
        )
        step_type = "error_recovery" if success is False else "tool_execution"
        exec_step: dict[str, Any] = {
            "step_index": idx + (1 if decision else 0),
            "turn_index": turn_index,
            "step_type": step_type,
            "content": _coerce_text(content_source, max_chars=4000),
            "metadata": {
                "otel_event_name": "tool_result",
                "ts": _otel_event_ts(result),
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "success": success,
                "error": error,
                "error_content": _coerce_text(error, max_chars=2000) if error else None,
                "duration_ms": duration_ms,
            },
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "tokens": 0,
        }
        if duration_ms is not None:
            exec_step["duration_ms"] = duration_ms
        steps.append(exec_step)
    return steps


def _pair_tool_events(events: list[dict[str, Any]]) -> tuple[list[tuple[dict[str, Any] | None, dict[str, Any] | None, int]], set[int]]:
    """Match tool_decision events with tool_result events by tool_use_id.

    Returns (pairs, consumed_indexes). Each pair is (decision, result,
    anchor_index) where anchor_index is the position of the earliest event in
    the pair — used later to preserve chronological ordering when we splice
    the merged steps back into the timeline.
    """
    pairs: list[tuple[dict[str, Any] | None, dict[str, Any] | None, int]] = []
    consumed: set[int] = set()
    decisions_by_id: dict[str, tuple[int, dict[str, Any]]] = {}
    for i, evt in enumerate(events):
        name = str(evt.get("event_name") or "").lower()
        attrs = evt.get("attributes") if isinstance(evt.get("attributes"), dict) else {}
        tool_use_id = str(attrs.get("tool_use_id") or "")
        if name == "tool_decision" and tool_use_id:
            decisions_by_id[tool_use_id] = (i, evt)
        elif name == "tool_result" and tool_use_id and tool_use_id in decisions_by_id:
            d_idx, d_evt = decisions_by_id.pop(tool_use_id)
            pairs.append((d_evt, evt, d_idx))
            consumed.add(d_idx)
            consumed.add(i)
    # Any decisions without a matching result stand on their own.
    for tool_use_id, (d_idx, d_evt) in decisions_by_id.items():
        pairs.append((d_evt, None, d_idx))
        consumed.add(d_idx)
    return pairs, consumed


def _normalize_otel_event_stream(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Split an OTel event stream into turns keyed by `user_prompt` events.

    Design choices:

    * Each `user_prompt` event opens a new turn. Everything until the next
      `user_prompt` is that turn's steps.
    * Registration / lifecycle noise (hook_registered, plugin_loaded, ...) is
      not rendered as steps — it's counted into `metadata.otel_noise_counts`
      on the turn so we don't silently drop it, but users don't have to
      scroll past 40 identical rows to reach the actual work.
    * `tool_decision` + matching `tool_result` are emitted as one canonical
      pair: an "LLM Thinking → Use Tool <tool>" step followed by a
      "Tool Execution → <tool>" step (or an "Error Recovery" step when the
      tool failed). This mirrors how IDE-hooked traces already render.
    * `api_request` becomes an "LLM Thinking" step with model/token metadata
      so the trace tree shows realistic thinking anchors.
    * OTel telemetry rarely carries the assistant text; final_response stays
      empty when absent, and the LLM critic is prompted to acknowledge that.
    """
    events = [e for e in payload.get("events") or [] if isinstance(e, dict)]
    if not events:
        return []
    boundaries = [
        i for i, e in enumerate(events)
        if str(e.get("event_name") or "").lower() in {"user_prompt", "user_input", "user_message"}
    ]
    if not boundaries:
        boundaries = [0]

    turns: list[dict[str, Any]] = []
    for turn_i, start in enumerate(boundaries):
        end = boundaries[turn_i + 1] if turn_i + 1 < len(boundaries) else len(events)
        # Preamble events go with the first turn so we don't lose them entirely.
        turn_events = events[0:end] if (turn_i == 0 and start > 0) else events[start:end]
        prompt_event = next(
            (
                e for e in turn_events
                if str(e.get("event_name") or "").lower() in {"user_prompt", "user_input", "user_message"}
            ),
            None,
        )
        prompt_attrs = prompt_event.get("attributes") if prompt_event and isinstance(prompt_event.get("attributes"), dict) else {}
        user_query = _coerce_text(prompt_attrs.get("prompt") if prompt_attrs else "", max_chars=4000)

        # Assistant text is best-effort — most OTel exporters omit it.
        final_response = ""
        for evt in reversed(turn_events):
            attrs = evt.get("attributes") if isinstance(evt.get("attributes"), dict) else {}
            for key in ("response", "assistant_message", "output", "final_response"):
                val = attrs.get(key)
                if val:
                    final_response = _coerce_text(val, max_chars=8000)
                    break
            if final_response:
                break

        # Filter noise and pair up tool events. `consumed` holds indexes we've
        # already turned into pairs so we don't re-emit them from the fallback
        # loop below.
        pairs, consumed = _pair_tool_events(turn_events)
        pair_at: dict[int, tuple[dict[str, Any] | None, dict[str, Any] | None]] = {
            anchor: (d, r) for (d, r, anchor) in pairs
        }

        noise_counts: dict[str, int] = {}
        steps: list[dict[str, Any]] = []
        step_idx = 0
        for i, evt in enumerate(turn_events):
            name = str(evt.get("event_name") or "").lower()
            if _otel_event_is_noise(name):
                noise_counts[name] = noise_counts.get(name, 0) + 1
                continue
            if name in {"user_prompt", "user_input", "user_message"}:
                # Already surfaced as `user_query`; don't duplicate.
                continue
            if i in pair_at:
                pair_steps = _otel_tool_pair_step(*pair_at[i], idx=step_idx, turn_index=turn_i + 1)
                for s in pair_steps:
                    s["step_index"] = step_idx
                    step_idx += 1
                    steps.append(s)
                continue
            if i in consumed:
                # This event's mate is emitted at the anchor position.
                continue
            if name == "api_request":
                steps.append(_otel_thinking_step(evt, step_idx, turn_i + 1))
                step_idx += 1
                continue
            # Unknown event — keep it as a generic step so we don't hide data,
            # but avoid the ambiguous "Tool Result" label.
            attrs = evt.get("attributes") if isinstance(evt.get("attributes"), dict) else {}
            steps.append({
                "step_index": step_idx,
                "turn_index": turn_i + 1,
                "step_type": "tool_result_input",
                "content": _coerce_text(evt.get("body") or attrs, max_chars=2000),
                "metadata": {"otel_event_name": name, "ts": _otel_event_ts(evt)},
                "tool_name": "",
                "tool_use_id": "",
                "tokens": 0,
            })
            step_idx += 1

        turns.append(
            _build_turn(
                turn_index=turn_i + 1,
                user_query=user_query,
                final_response=final_response,
                steps=steps,
                raw_payload=None,
            )
        )
        turns[-1]["metadata"] = {
            "otel_noise_counts": noise_counts,
            "otel_event_total": len(turn_events),
            "otel_step_total": len(steps),
        }
    return turns


def _normalize_object_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Object payload with a familiar 'turns' or messages-style shape."""
    if _looks_like_otel_event_stream(payload):
        return _normalize_otel_event_stream(payload)
    if isinstance(payload.get("turns"), list) and payload["turns"]:
        out: list[dict[str, Any]] = []
        for idx, raw_turn in enumerate(payload["turns"]):
            if not isinstance(raw_turn, dict):
                continue
            steps_raw = raw_turn.get("steps") or raw_turn.get("events") or raw_turn.get("spans")
            steps = _normalize_steps(steps_raw)
            out.append(
                _build_turn(
                    turn_index=int(raw_turn.get("turn_index") or idx + 1),
                    user_query=_coerce_text(_pick_first(raw_turn, ["user_query", "input", "prompt", "question"]), max_chars=4000),
                    final_response=_coerce_text(_pick_first(raw_turn, ["final_response", "output", "answer", "response"]), max_chars=8000),
                    steps=steps,
                    raw_payload=raw_turn,
                )
            )
        return out

    messages = _pick_first(payload, ["messages", "conversation", "history"])
    if _looks_like_message_list(messages):
        steps_raw = _pick_first(payload, ["steps", "events", "spans", "trace"])
        steps = _normalize_steps(steps_raw)
        return [
            _build_turn(
                turn_index=1,
                user_query=_extract_user_query_from_messages(messages),
                final_response=_extract_final_response_from_messages(messages),
                steps=steps,
                raw_payload=payload,
            )
        ]

    # Common single-turn fields (Langfuse "input"/"output", OTel "input"/"output").
    user_query = _coerce_text(_pick_first(payload, ["input", "prompt", "user_query", "question"]), max_chars=4000)
    final_response = _coerce_text(_pick_first(payload, ["output", "response", "final_response", "answer"]), max_chars=8000)
    if user_query or final_response:
        steps_raw = _pick_first(payload, ["steps", "events", "spans", "trace", "tool_calls"])
        steps = _normalize_steps(steps_raw)
        return [
            _build_turn(
                turn_index=1,
                user_query=user_query,
                final_response=final_response,
                steps=steps,
                raw_payload=payload,
            )
        ]
    return []


def _normalize_array_payload(payload: list[Any]) -> list[dict[str, Any]]:
    if _looks_like_message_list(payload):
        return [
            _build_turn(
                turn_index=1,
                user_query=_extract_user_query_from_messages(payload),
                final_response=_extract_final_response_from_messages(payload),
                steps=_normalize_steps(payload),
                raw_payload=payload,
            )
        ]
    # Treat as one-turn step list when items don't look like messages.
    return [
        _build_turn(
            turn_index=1,
            user_query="",
            final_response="",
            steps=_normalize_steps(payload),
            raw_payload=payload,
        )
    ]


def _fallback_single_turn(raw_payload: Any) -> list[dict[str, Any]]:
    """Last resort: wrap whatever we got as a single dump step.

    The LLM critic tolerates this because its prompt explicitly asks the model
    to extract evidence from raw transcript text when structure is missing.
    """
    return [
        _build_turn(
            turn_index=1,
            user_query="",
            final_response="",
            steps=[
                {
                    "step_index": 0,
                    "turn_index": 1,
                    "step_type": "tool_result_input",
                    "content": _coerce_text(raw_payload, max_chars=8000),
                    "metadata": {},
                    "tool_name": "",
                    "tool_use_id": "",
                    "tokens": 0,
                }
            ],
            raw_payload=raw_payload,
        )
    ]


def normalize_uploaded_trace(
    *,
    session_id: str,
    raw_trace: Any,
    transcript: str | None = None,
    source: str = "user-upload",
    title: str | None = None,
) -> dict[str, Any]:
    """Build a cot-compatible dict from arbitrary user-supplied trace data.

    `raw_trace` may be a dict, list, or string. If it's a string we first try
    JSON, then treat it as raw transcript text. The resulting cot is marked
    with `_uploaded=True` so the session scanner routes it into the virtual
    "Uploaded Traces" project.
    """
    payload: Any = raw_trace
    if isinstance(payload, str):
        text = payload.strip()
        if text:
            try:
                payload = json.loads(text)
            except Exception:
                payload = {"input": "", "output": "", "text": text}
        else:
            payload = {}

    turns: list[dict[str, Any]]
    if isinstance(payload, dict):
        turns = _normalize_object_payload(payload) or _fallback_single_turn(payload)
    elif isinstance(payload, list):
        turns = _normalize_array_payload(payload)
    else:
        turns = _fallback_single_turn(payload)

    if transcript:
        # Append transcript as an extra step on the first turn so it is visible
        # to the LLM critic without inventing a synthetic turn.
        turns[0]["steps"].append(
            {
                "step_index": len(turns[0]["steps"]),
                "turn_index": turns[0].get("turn_index", 1),
                "step_type": "user_input",
                "content": _coerce_text(transcript, max_chars=8000),
                "metadata": {"uploaded_transcript": True},
                "tool_name": "",
                "tool_use_id": "",
                "tokens": 0,
            }
        )

    cot = {
        "session_id": session_id,
        "transcript_path": "",
        "extracted_at": _utc_iso(),
        "turns": turns,
        "total_tool_calls": sum(len(t.get("tool_calls") or []) for t in turns),
        "total_strategy_shifts": 0,
        "total_thinking_steps": 0,
        "tool_call_distribution": {},
        "avg_steps_per_turn": (sum(len(t.get("steps") or []) for t in turns) / len(turns)) if turns else 0,
        "avg_complexity": 0,
        "agent_type": "uploaded",
        "_uploaded": True,
        "_uploaded_source": source,
        "_uploaded_title": title or "",
        "_uploaded_at": _utc_iso(),
    }
    return cot
