"""Tests for the user-uploaded trace pathway.

Two contracts here:

1. The source-agnostic normalizer turns multiple realistic shapes (Langfuse-
   style messages, OTel-style input/output, raw text) into a cot.json the rest
   of the platform can read. Token / latency fields may be missing — that's
   intentional, and the test guards against accidental `KeyError`s in the
   shape we produce.

2. Uploaded traces are routed to the virtual `__uploaded__` project so the
   sidebar can render them as a separate, source-isolated group. We don't
   test the scanner directly (that's a backend service that depends on the
   filesystem), only the project resolver short-circuit.
"""

from __future__ import annotations

import json

from agent_quality_eval.evaluation.uploaded_trace import (
    UPLOADED_PROJECT_ID,
    UPLOADED_PROJECT_NAME,
    normalize_uploaded_trace,
)


def test_normalize_langfuse_messages_picks_user_and_assistant():
    raw = {
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "How do I extract tables from invoice.pdf?"},
            {"role": "assistant", "content": "I extracted 3 tables. They are saved to tables.csv."},
        ],
        "steps": [
            {"type": "tool_call", "name": "pdf_extract", "result": {"rows": 27}},
        ],
    }
    cot = normalize_uploaded_trace(session_id="uploaded-abc", raw_trace=raw, source="langfuse")
    assert cot["session_id"] == "uploaded-abc"
    assert cot["_uploaded"] is True
    assert cot["_uploaded_source"] == "langfuse"
    assert cot["agent_type"] == "uploaded"
    assert len(cot["turns"]) == 1
    turn = cot["turns"][0]
    assert "invoice.pdf" in turn["user_query"]
    assert "tables.csv" in turn["final_response"]
    assert turn["steps"][0]["tool_name"] == "pdf_extract"


def test_normalize_otel_style_input_output_pair():
    raw = {"input": "What is the capital of France?", "output": "Paris."}
    cot = normalize_uploaded_trace(session_id="uploaded-otel", raw_trace=raw, source="otel")
    assert cot["turns"][0]["user_query"] == "What is the capital of France?"
    assert cot["turns"][0]["final_response"] == "Paris."


def test_normalize_raw_string_falls_back_to_single_step():
    cot = normalize_uploaded_trace(
        session_id="uploaded-blob",
        raw_trace="just a plain text dump from some tool",
        source="custom",
    )
    assert cot["_uploaded"] is True
    turn = cot["turns"][0]
    # The text fallback packs the payload as either user_query/final_response
    # via _normalize_object_payload (we wrap into {input, output, text}) or as
    # a single tool_result_input step. Either way the text must be reachable.
    serialized = json.dumps(turn, ensure_ascii=False, default=str)
    assert "just a plain text dump from some tool" in serialized


def test_normalize_array_of_messages():
    raw = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    cot = normalize_uploaded_trace(session_id="uploaded-arr", raw_trace=raw)
    assert cot["turns"][0]["user_query"] == "hi"
    assert cot["turns"][0]["final_response"] == "hello"


def test_optional_transcript_attached_as_extra_step():
    raw = {"input": "q", "output": "a"}
    cot = normalize_uploaded_trace(
        session_id="uploaded-tx",
        raw_trace=raw,
        transcript="raw transcript text from external system",
    )
    steps = cot["turns"][0]["steps"]
    assert any(
        s.get("step_type") == "user_input" and "raw transcript text" in s.get("content", "")
        for s in steps
    )


def test_explicit_turns_shape_is_preserved():
    raw = {
        "turns": [
            {
                "turn_index": 1,
                "user_query": "first question",
                "final_response": "first answer",
                "steps": [{"type": "tool_execution", "name": "search"}],
            },
            {
                "turn_index": 2,
                "input": "second question",
                "output": "second answer",
                "events": [{"kind": "tool_call", "tool_name": "calculator"}],
            },
        ]
    }
    cot = normalize_uploaded_trace(session_id="uploaded-multi", raw_trace=raw)
    assert len(cot["turns"]) == 2
    assert cot["turns"][0]["user_query"] == "first question"
    assert cot["turns"][1]["user_query"] == "second question"
    assert cot["turns"][0]["steps"][0]["tool_name"] == "search"
    assert cot["turns"][1]["steps"][0]["tool_name"] == "calculator"


def test_uploaded_project_constants_match_scanner_short_circuit():
    """The session scanner short-circuits to `__uploaded__` when it sees
    `cot_data["_uploaded"]`. If we ever rename the constants, this test will
    flag it before the sidebar grouping silently breaks."""
    assert UPLOADED_PROJECT_ID == "__uploaded__"
    assert UPLOADED_PROJECT_NAME == "Uploaded Traces"


def test_uploaded_cot_carries_marker_and_source():
    cot = normalize_uploaded_trace(
        session_id="uploaded-knot",
        raw_trace={"input": "a", "output": "b"},
        source="knot",
        title="my pipeline trace",
    )
    assert cot["_uploaded"] is True
    assert cot["_uploaded_source"] == "knot"
    assert cot["_uploaded_title"] == "my pipeline trace"


def test_normalize_otel_event_stream_splits_turns_by_user_prompt():
    """Real OTel exporters (Claude Code, home-grown OTel bridges, ...) emit a
    flat `events` list where user_prompt marks the start of each turn. The
    normalizer now filters registration/lifecycle noise and pairs
    tool_decision/tool_result into canonical "Use Tool" + "Tool Execution"
    steps so the uploaded trace visually matches IDE-hooked traces."""
    raw = {
        "session_id": "otel-abc",
        "provider": "claude",
        "events": [
            # Preamble noise — must be filtered from the step list, not shown.
            {"event_name": "plugin_loaded", "attributes": {"plugin.name": "x"}, "ts": "t0"},
            {"event_name": "hook_registered", "attributes": {"hook_event": "Stop"}, "ts": "t0.1"},
            # Turn 1
            {
                "event_name": "user_prompt",
                "attributes": {"prompt": "please list the files", "prompt.id": "p1"},
                "ts": "t1",
            },
            {
                "event_name": "tool_decision",
                "attributes": {"tool_name": "Bash", "tool_use_id": "u1", "decision": "accept",
                               "tool_input": {"command": "ls"}},
                "ts": "t2",
            },
            {
                "event_name": "tool_result",
                "attributes": {"tool_name": "Bash", "tool_use_id": "u1", "success": "false", "error": "not found",
                               "duration_ms": 250},
                "ts": "t3",
            },
            {
                "event_name": "api_request",
                "attributes": {"model": "claude-sonnet", "input_tokens": 10, "output_tokens": 20,
                               "duration_ms": 900},
                "ts": "t4",
            },
            # Turn 2
            {
                "event_name": "user_prompt",
                "attributes": {"prompt": "try again"},
                "ts": "t5",
            },
            {
                "event_name": "tool_decision",
                "attributes": {"tool_name": "Read", "tool_use_id": "u2",
                               "tool_input": {"path": "a.txt"}},
                "ts": "t6",
            },
            {
                "event_name": "tool_result",
                "attributes": {"tool_name": "Read", "tool_use_id": "u2", "success": True,
                               "duration_ms": 30},
                "ts": "t7",
            },
        ],
    }
    cot = normalize_uploaded_trace(session_id="uploaded-otel-events", raw_trace=raw, source="otel")
    assert len(cot["turns"]) == 2
    t1, t2 = cot["turns"]
    assert t1["user_query"] == "please list the files"
    assert t2["user_query"] == "try again"

    # Turn 1 must contain: 1 tool_decision + 1 error_recovery (failed Bash) + 1 thinking. Noise is filtered.
    step_types_t1 = [s["step_type"] for s in t1["steps"]]
    assert step_types_t1 == ["tool_decision", "error_recovery", "thinking_intermediate"]
    # Noise counts must be preserved on turn.metadata so nothing is silently dropped.
    noise = t1.get("metadata", {}).get("otel_noise_counts") or {}
    assert noise.get("plugin_loaded") == 1
    assert noise.get("hook_registered") == 1

    # tool_decision step must carry tool_input / tool_name / tool_use_id so
    # the frontend can render "LLM Thinking → Use Tool Bash" and expand args.
    dec = t1["steps"][0]
    assert dec["tool_name"] == "Bash"
    assert dec["metadata"]["tool_use_id"] == "u1"
    assert dec["metadata"]["tool_input"] == {"command": "ls"}

    # Failed tool_result becomes error_recovery with error_content set.
    err = t1["steps"][1]
    assert err["step_type"] == "error_recovery"
    assert err["metadata"]["success"] is False
    assert err["metadata"]["error_content"] == "not found"

    # api_request → LLM Thinking with tokens surfaced.
    thinking = t1["steps"][2]
    assert thinking["step_type"] == "thinking_intermediate"
    assert thinking["metadata"]["output_tokens"] == 20
    assert thinking["duration_ms"] == 900

    # Turn 2's successful Read pair renders as tool_decision + tool_execution.
    step_types_t2 = [s["step_type"] for s in t2["steps"]]
    assert step_types_t2 == ["tool_decision", "tool_execution"]
    assert t2["steps"][1]["metadata"]["success"] is True


def test_otel_event_stream_without_user_prompt_still_produces_one_turn():
    """Some exporters may emit only tool events (no user_prompt marker). The
    stream must still yield one non-empty turn instead of the empty-turn
    fallback path."""
    raw = {
        "events": [
            {"event_name": "tool_decision", "attributes": {"tool_name": "Bash", "tool_use_id": "u1"}, "ts": "t1"},
            {"event_name": "tool_result", "attributes": {"tool_name": "Bash", "tool_use_id": "u1", "success": True}, "ts": "t2"},
        ]
    }
    cot = normalize_uploaded_trace(session_id="uploaded-no-prompt", raw_trace=raw)
    assert len(cot["turns"]) == 1
    assert cot["turns"][0]["user_query"] == ""
    # tool_decision + tool_execution (paired), even without user_prompt.
    types = [s["step_type"] for s in cot["turns"][0]["steps"]]
    assert types == ["tool_decision", "tool_execution"]


def test_otel_lone_tool_decision_still_emitted():
    """A tool_decision without matching tool_result should stand on its own —
    users deserve to see 'agent tried to call this tool but it never returned'
    rather than have the row silently disappear."""
    raw = {
        "events": [
            {"event_name": "user_prompt", "attributes": {"prompt": "hi"}, "ts": "t0"},
            {"event_name": "tool_decision", "attributes": {"tool_name": "Bash", "tool_use_id": "u1"}, "ts": "t1"},
        ]
    }
    cot = normalize_uploaded_trace(session_id="u", raw_trace=raw)
    steps = cot["turns"][0]["steps"]
    assert [s["step_type"] for s in steps] == ["tool_decision"]
    assert steps[0]["tool_name"] == "Bash"
