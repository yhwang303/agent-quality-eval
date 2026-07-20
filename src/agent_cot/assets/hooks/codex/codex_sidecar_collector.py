#!/usr/bin/env python
"""Materialize Codex rollout JSONL into agent-cot ``*_cot.json`` files.

This is intentionally transcript-first.  Codex already keeps structured local
rollouts under ``$CODEX_HOME/sessions``; the hook uses this sidecar to convert
that native transcript into the dashboard schema without touching Cursor,
Claude or CodeBuddy paths.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()


def _data_root() -> Path:
    raw = os.environ.get("AGENT_COT_DATA_ROOT")
    if raw:
        return Path(raw).expanduser()
    try:
        runtime = Path.home() / ".agent-cot" / "runtime.json"
        if runtime.is_file():
            data = json.loads(runtime.read_text(encoding="utf-8"))
            value = data.get("data_root")
            if isinstance(value, str) and value:
                p = Path(value).expanduser()
                return p if p.name == "data" else p / "data"
    except Exception:
        pass
    return Path.home() / ".agent-cot" / "data"


def _read_runtime_python() -> str | None:
    try:
        runtime = Path.home() / ".agent-cot" / "runtime.json"
        if runtime.is_file():
            data = json.loads(runtime.read_text(encoding="utf-8"))
            value = data.get("python_executable") if isinstance(data, dict) else None
            return str(value) if value else None
    except Exception:
        return None
    return None


def _is_agent_quality_eval_exe(py: str) -> bool:
    name = Path(str(py or "")).name.lower()
    return name.startswith("agent-quality-eval") and name.endswith(".exe")


def _agent_quality_eval_runner_cmd(py: str, runner: str, args: list[str]) -> list[str]:
    if _is_agent_quality_eval_exe(py):
        return [py, "--agent-quality-eval-runner", runner, *args]
    if runner == "live-critic":
        code = "from agent_quality_eval.evaluation.live_critic import main; raise SystemExit(main())"
    else:
        code = "from agent_quality_eval.evaluation.critic import main; raise SystemExit(main())"
    return [py, "-c", code, *args]


def _critic_debounce_path() -> Path:
    return _data_root().parent / "state" / "codex-collector-critic-debounce.json"


def _should_spawn_critic(sid: str, source_mtime: float, payload_changed: bool) -> bool:
    if os.environ.get("AGENT_QUALITY_EVAL_CRITIC_DISABLE"):
        return False
    if not payload_changed:
        return False
    # The collector may rewrite a recent scan window on app startup. Only a
    # transcript touched in the last few minutes is considered a new turn.
    if time.time() - source_mtime > 600:
        return False
    now = time.time()
    key = str(sid)
    path = _critic_debounce_path()
    try:
        state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if not isinstance(state, dict):
            state = {}
    except Exception:
        state = {}
    last = float(state.get(key) or 0)
    if now - last < 45:
        return False
    state[key] = now
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return True


def _spawn_critic(sid: str) -> None:
    py = os.environ.get("COT_PYTHON") or _read_runtime_python() or sys.executable or "python"
    cmd = _agent_quality_eval_runner_cmd(py, "critic", [
        "--agent-type",
        "codex",
        "--source-event",
        "codex-collector:cot_written",
        "--session-id",
        sid,
        "--wait-seconds",
        "5",
        "--no-persist-eval",
    ])
    env = os.environ.copy()
    env["AGENT_QUALITY_EVAL_CRITIC_DISABLE"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    popen_kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) | 0x00000008 | 0x00000200
        )
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        popen_kwargs["startupinfo"] = startupinfo
    try:
        log_path = Path.home() / ".agent-cot" / "logs" / "critic-runner.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "a", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            cmd,
            cwd=str(_codex_home()),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            close_fds=True,
            **popen_kwargs,
        )
        _pipeline_log("critic_spawn", sid=sid, pid=proc.pid, source_event="codex-collector:cot_written")
    except Exception as exc:
        _pipeline_log("critic_spawn", sid=sid, ok=False, error=exc, source_event="codex-collector:cot_written")


def _stable_critic_payload(value: str | None) -> str | None:
    if not value:
        return None
    try:
        data = json.loads(value)
    except Exception:
        return value
    if not isinstance(data, dict):
        return value
    data.pop("collected_at", None)
    otel_view = data.get("otel_view")
    if isinstance(otel_view, dict):
        otel_view.pop("generated_at_ms", None)
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _pipeline_log(event: str, sid: str = "-", ok: bool = True, **note: Any) -> None:
    try:
        lp = Path(
            os.environ.get("AGENT_COT_PIPELINE_LOG")
            or str(Path.home() / ".agent-cot" / "logs" / "pipeline.log")
        ).expanduser()
        lp.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        parts = []
        for key, value in note.items():
            if value is None:
                continue
            text = str(value)
            if " " in text or '"' in text:
                text = '"' + text.replace('"', '\\"') + '"'
            parts.append(f"{key}={text}")
        line = (
            f"[{ts}] [codex.collector] [codex] [sid={sid}] "
            f"event={event} status={'ok' if ok else 'FAIL'}"
            + (" " + " ".join(parts) if parts else "")
            + "\n"
        )
        with open(lp, "a", encoding="utf-8", errors="replace") as fh:
            fh.write(line)
    except Exception:
        pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict):
                    out.append(item)
    except OSError:
        pass
    return out


def _find_rollouts(session_id: str | None, recent_seconds: int) -> list[Path]:
    root = _codex_home() / "sessions"
    if not root.is_dir():
        return []
    pattern = str(root / "*" / "*" / "*" / "*.jsonl")
    files = [Path(p) for p in glob.glob(pattern)]
    if session_id:
        needle = session_id.lower()
        files = [p for p in files if needle in p.name.lower()]
    elif recent_seconds > 0:
        cutoff = time.time() - recent_seconds
        files = [p for p in files if p.stat().st_mtime >= cutoff]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def _payload(item: dict[str, Any]) -> Any:
    return item.get("payload")


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    return ""


def _message_text(payload: Any) -> tuple[str | None, str]:
    if not isinstance(payload, dict):
        return None, ""
    if payload.get("type") == "message":
        role = payload.get("role")
        text = _text_from_content(payload.get("content"))
        return (str(role) if role else None), text
    if "message" in payload and isinstance(payload["message"], str):
        return "assistant", payload["message"]
    if "message" in payload and isinstance(payload["message"], dict):
        msg = payload["message"]
        role = msg.get("role")
        return (str(role) if role else None), _text_from_content(msg.get("content"))
    return None, ""


def _compact_text(text: str, limit: int = 800) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _is_noise_user_text(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if stripped.startswith("<environment_context>"):
        return True
    if stripped.startswith("<turn_aborted>"):
        return True
    if stripped.startswith("<permissions instructions>"):
        return True
    if stripped.startswith("<app-context>"):
        return True
    if stripped.startswith("# AGENTS.md"):
        return True
    if stripped.startswith("Another language model started to solve this problem"):
        return True
    return False


def _clean_user_prompt(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^My request for Codex:\s*", "", text, flags=re.I)
    text = re.sub(r"\n?# Files mentioned by the user:.*\Z", "", text, flags=re.S)
    return text.strip()


def _display_model(model: Any) -> str | None:
    if not isinstance(model, str):
        return None
    text = model.strip()
    if not text:
        return None
    if text.lower() in {"timi", "openai", "codex"}:
        return None
    return text


def _model_from_config() -> str | None:
    cfg = _codex_home() / "config.toml"
    try:
        data = cfg.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    in_table = False
    for raw in data.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            in_table = True
            continue
        if in_table:
            continue
        m = re.match(r"""model\s*=\s*["']([^"']+)["']""", line)
        if m:
            return _display_model(m.group(1))
    return None


def _canonical_tool_name(name: str, args: Any = None) -> str:
    raw = (name or "").strip()
    low = raw.lower()
    if low == "update_plan":
        return "TodoWrite"
    if low == "exec_command":
        if isinstance(args, dict):
            shell = str(args.get("shell") or "").strip().lower()
            if shell:
                return shell
        return "shell"
    if low in {"shell_command", "bash", "shell"}:
        return low
    if low == "write_stdin":
        return "write"
    if low.startswith("read_") or low == "read_thread_terminal":
        return "read"
    if low == "apply_patch":
        return "apply_patch"
    if low in {"web_search", "web_search_call"}:
        return "web_search"
    if low in {"js", "node_repl"} or "node_repl" in low or low.endswith(".js"):
        return "node_repl"
    return raw or "Tool"


def _parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _tool_input_summary(tool_name: str, tool_input: Any) -> str:
    if isinstance(tool_input, dict):
        command = tool_input.get("command")
        if isinstance(command, str) and command.strip():
            return command.strip()
        code = tool_input.get("code")
        if isinstance(code, str) and code.strip():
            return _compact_text(code, 500)
        q = tool_input.get("query") or tool_input.get("queries")
        if q:
            return _compact_text(json.dumps(q, ensure_ascii=False, default=str), 500)
    if isinstance(tool_input, str):
        return _compact_text(tool_input, 500)
    return _compact_text(json.dumps(tool_input, ensure_ascii=False, default=str), 500)


def _tool_result_summary(output: str) -> str:
    text = output or ""
    match = re.search(r"Exit code:\s*([^\n]+)", text)
    if match:
        return f"Exit code: {match.group(1).strip()}"
    return _compact_text(text, 300)


def _normalize_plan_status(status: Any) -> str:
    text = str(status or "").strip().lower().replace("-", "_")
    if text in {"complete", "completed", "done", "success"}:
        return "completed"
    if text in {"inprogress", "in_progress", "active", "running", "doing"}:
        return "in_progress"
    if text in {"cancelled", "canceled", "skipped"}:
        return "cancelled"
    return "pending"


def _extract_codex_plan_todos(value: Any) -> list[dict[str, Any]]:
    """Normalize Codex ``update_plan`` arguments to the dashboard Todo shape."""
    data = _parse_jsonish(value)
    if isinstance(data, dict):
        items = data.get("plan") or data.get("todos") or data.get("items")
    else:
        items = data
    if not isinstance(items, list):
        return []

    todos: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if isinstance(item, str):
            content = item.strip()
            status = "pending"
            raw_id = None
        elif isinstance(item, dict):
            content = str(
                item.get("step")
                or item.get("content")
                or item.get("title")
                or item.get("description")
                or ""
            ).strip()
            status = _normalize_plan_status(item.get("status"))
            raw_id = item.get("id")
        else:
            continue
        if not content:
            continue
        todos.append(
            {
                "id": str(raw_id if raw_id not in (None, "") else idx + 1),
                "content": content,
                "status": status,
                "idx": idx,
            }
        )
    return todos


def _diff_plan_todos(prev: list[dict[str, Any]] | None, curr: list[dict[str, Any]]) -> dict[str, Any]:
    diff = {
        "newly_completed": [],
        "newly_started": [],
        "newly_added": [],
        "removed": [],
        "status_changes": [],
    }
    prev_map: dict[str, dict[str, Any]] = {}
    for item in prev or []:
        if isinstance(item, dict):
            prev_map[str(item.get("id") or item.get("content") or "")] = item

    curr_keys: set[str] = set()
    for item in curr:
        key = str(item.get("id") or item.get("content") or "")
        curr_keys.add(key)
        prev_item = prev_map.get(key)
        status = _normalize_plan_status(item.get("status"))
        entry = {"id": item.get("id"), "content": item.get("content")}
        if prev_item is None:
            diff["newly_added"].append({**entry, "status": status})
            if status == "completed":
                diff["newly_completed"].append(entry)
            elif status == "in_progress":
                diff["newly_started"].append(entry)
            continue

        prev_status = _normalize_plan_status(prev_item.get("status"))
        if prev_status == status:
            continue
        if status == "completed":
            diff["newly_completed"].append(entry)
        elif status == "in_progress":
            diff["newly_started"].append(entry)
        else:
            diff["status_changes"].append(
                {**entry, "from": prev_status, "to": status}
            )

    for key, item in prev_map.items():
        if key not in curr_keys:
            diff["removed"].append({"id": item.get("id"), "content": item.get("content")})
    return diff


def _enrich_codex_plan_timeline(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    prev: list[dict[str, Any]] | None = None
    for turn in turns:
        for step in turn.get("steps") or []:
            if (
                not isinstance(step, dict)
                or step.get("tool_name") != "TodoWrite"
                or step.get("step_type") != "tool_decision"
            ):
                continue
            md = step.setdefault("metadata", {})
            tool_input = _parse_jsonish(md.get("tool_input"))
            todos = _extract_codex_plan_todos(tool_input)
            if not todos:
                todos = _extract_codex_plan_todos(md.get("payload"))
            if not todos:
                continue

            counts = Counter(_normalize_plan_status(t.get("status")) for t in todos)
            diff = _diff_plan_todos(prev, todos)
            snapshot_idx = len(timeline) + 1
            md["tool_input"] = {"todos": todos}
            md["plan_full_todos"] = todos
            md["plan_diff"] = diff
            md["plan_snapshot_idx"] = snapshot_idx
            md["plan_total"] = len(todos)
            md["plan_completed_count"] = int(counts.get("completed", 0))
            md["plan_in_progress_count"] = int(counts.get("in_progress", 0))
            md["plan_pending_count"] = int(counts.get("pending", 0))
            md["plan_cancelled_count"] = int(counts.get("cancelled", 0))
            md["codex_plan_source"] = "update_plan"

            timeline.append(
                {
                    "at_step": int(step.get("step_index") or 0),
                    "turn_index": int(step.get("turn_index") or 0),
                    "timestamp": step.get("timestamp"),
                    "in_progress": [
                        str(t.get("content") or "")
                        for t in todos
                        if _normalize_plan_status(t.get("status")) == "in_progress"
                    ],
                    "completed": [
                        str(t.get("content") or "")
                        for t in todos
                        if _normalize_plan_status(t.get("status")) == "completed"
                    ],
                    "pending": [
                        str(t.get("content") or "")
                        for t in todos
                        if _normalize_plan_status(t.get("status")) == "pending"
                    ],
                    "cancelled": [
                        str(t.get("content") or "")
                        for t in todos
                        if _normalize_plan_status(t.get("status")) == "cancelled"
                    ],
                    "total": len(todos),
                    "todos": todos,
                    "diff": diff,
                    "snapshot_index": snapshot_idx,
                }
            )
            prev = todos
    return timeline


def _ts_ms(ts: str | None) -> int | None:
    if not ts:
        return None
    try:
        return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return None


def _iso_from_ms(ms: int | None) -> str | None:
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _usage_numbers(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage if isinstance(usage, dict) else {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cache_creation_input_tokens": int(
            usage.get("cache_creation_input_tokens")
            or usage.get("cache_write_input_tokens")
            or 0
        ),
        "cache_read_input_tokens": int(
            usage.get("cached_input_tokens")
            or usage.get("cache_read_input_tokens")
            or 0
        ),
        "reasoning_output_tokens": int(usage.get("reasoning_output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def _token_usage_view(
    usage: dict[str, Any] | None,
    *,
    model: str,
    source: str = "codex_token_count",
    shared: bool = False,
) -> dict[str, Any]:
    nums = _usage_numbers(usage)
    if shared:
        nums = {k: 0 for k in nums}
        source = "shared_with_anchor"
    return {
        "input_tokens": nums["input_tokens"],
        "output_tokens": nums["output_tokens"],
        "total_tokens": nums["total_tokens"]
        or nums["input_tokens"] + nums["output_tokens"],
        "cache_read_tokens": nums["cache_read_input_tokens"],
        "cache_creation_tokens": nums["cache_creation_input_tokens"],
        "cache_read_input_tokens": nums["cache_read_input_tokens"],
        "cache_creation_input_tokens": nums["cache_creation_input_tokens"],
        "reasoning_output_tokens": nums["reasoning_output_tokens"],
        "cost_usd": None,
        "cost_reason": "codex_native_usage",
        "currency": "USD",
        "is_estimate": False,
        "model_key": model,
        "source": source,
    }


def _attach_llm_otel(
    step: dict[str, Any],
    usage: dict[str, Any] | None,
    *,
    model: str,
    provider: str,
    shared: bool = False,
    duration_ms: int | None = None,
) -> None:
    token_usage = _token_usage_view(usage, model=model, shared=shared)
    step["otel"] = {
        "step_kind": "llm_call_shared" if shared else "llm_call",
        "operation_name": "codex.llm",
        "model": model,
        "provider": provider,
        "model_source": "transcript",
        "token_usage": token_usage,
        "attributes": {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": model,
            "gen_ai.provider.name": provider,
            "gen_ai.usage.input_tokens": token_usage["input_tokens"],
            "gen_ai.usage.output_tokens": token_usage["output_tokens"],
            "codex.usage.reasoning_output_tokens": token_usage["reasoning_output_tokens"],
        },
    }
    if duration_ms is not None and duration_ms >= 0:
        step["duration_ms"] = duration_ms
        step["otel"]["duration_ms"] = duration_ms
        step["otel"]["operation_duration_ms"] = duration_ms
    md = step.setdefault("metadata", {})
    if not shared:
        nums = _usage_numbers(usage)
        md["input_tokens"] = nums["input_tokens"]
        md["output_tokens"] = nums["output_tokens"]
        md["cache_read_input_tokens"] = nums["cache_read_input_tokens"]
        md["reasoning_output_tokens"] = nums["reasoning_output_tokens"]
        md["usage_source"] = "codex_token_count"
    else:
        md["usage_source"] = "shared_with_anchor"


def _new_step(
    idx: int,
    turn_index: int,
    step_type: str,
    content: str,
    *,
    timestamp: str | None = None,
    tool_name: str = "",
    tool_use_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "step_index": idx,
        "turn_index": turn_index,
        "step_type": step_type,
        "content": content or "",
        "metadata": metadata or {},
        "tool_name": tool_name,
        "tool_use_id": tool_use_id,
        "tokens": max(1, len(content or "") // 4) if content else 0,
        "timestamp": timestamp,
        "duration_ms": None,
        "reasoning_digest": None,
        "decision_trace": None,
        "state_evolution": None,
        "error_trace": None,
        "otel": None,
    }


def _norm_snapshot_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _snapshot_relation(prev: str | None, new: str | None) -> str | None:
    """Detect cumulative Codex UI snapshots.

    Codex rollout streams sometimes persist the same assistant message more
    than once: first as a partial snapshot, then as a longer snapshot with the
    same prefix. Keeping both makes the trace look like two separate thoughts.
    """
    a = _norm_snapshot_text(prev)
    b = _norm_snapshot_text(new)
    if not a or not b or a == b:
        return None
    short_enough_to_be_label = min(len(a), len(b)) < 40
    if short_enough_to_be_label:
        return None
    if b.startswith(a) or a in b:
        return "new_extends_prev"
    if a.startswith(b) or b in a:
        return "prev_extends_new"
    return None


def _finalize_turn(turn: dict[str, Any]) -> None:
    steps = turn["steps"]
    tool_calls = [
        s.get("tool_name")
        for s in steps
        if s.get("step_type") == "tool_decision" and s.get("tool_name")
    ]
    turn["tool_calls"] = tool_calls
    turn["strategy_shifts"] = 0
    turn["thinking_depth"] = len(
        [
            s
            for s in steps
            if s.get("step_type") in {"thinking_explicit", "pre_tool_reasoning"}
        ]
    )
    turn["total_steps"] = len(steps)
    turn["has_error_recovery"] = any("error" in (s.get("content") or "").lower() for s in steps)
    turn["usage"] = turn.get("usage") or {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    turn["complexity_score"] = min(1.0, 0.2 + 0.05 * len(steps) + 0.04 * len(tool_calls))
    if not turn.get("final_response"):
        finals = [s["content"] for s in steps if s.get("step_type") == "final_response"]
        turn["final_response"] = finals[-1] if finals else ""


def _rollout_id(path: Path, rows: list[dict[str, Any]]) -> str:
    for row in rows:
        if row.get("type") == "session_meta":
            payload = row.get("payload") or {}
            sid = payload.get("id")
            if isinstance(sid, str) and sid:
                return sid
    match = re.search(r"rollout-[^-]+-[^-]+-[^-]+-(.+)\.jsonl$", path.name)
    if match:
        return match.group(1)
    return path.stem.replace("rollout-", "")


def _build_cot(path: Path) -> dict[str, Any] | None:
    rows = _read_jsonl(path)
    if not rows:
        return None
    sid = _rollout_id(path, rows)
    session_meta = next((r.get("payload") for r in rows if r.get("type") == "session_meta"), {}) or {}

    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    step_idx = 0
    call_tools: dict[str, str] = {}
    call_plan_todos: dict[str, list[dict[str, Any]]] = {}
    seen_steps: set[tuple[str, str, str]] = set()
    model_seen: str | None = _display_model(session_meta.get("model"))
    provider = str(session_meta.get("model_provider") or "codex")
    last_token_usage: dict[str, Any] | None = None
    llm_call_count = 0
    pending_llm_steps: list[dict[str, Any]] = []
    compact_events: list[dict[str, Any]] = []

    def ensure_turn(ts: str | None = None) -> dict[str, Any]:
        nonlocal current
        if current is None:
            current = {
                "turn_index": len(turns),
                "user_query": "",
                "steps": [],
                "tool_calls": [],
                "strategy_shifts": 0,
                "thinking_depth": 0,
                "total_steps": 0,
                "has_error_recovery": False,
                "final_response": "",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
                "complexity_score": 0.0,
                "turn_start_time": ts,
                "turn_start_ms_observed": _ts_ms(ts),
                "turn_end_ms_observed": _ts_ms(ts),
                "turn_duration_ms": None,
                "turn_duration_ms_observed": None,
                "turn_wallclock_span_ms": None,
                "turn_idle_ms": 0,
                "otel": None,
                "eval": None,
            }
        return current

    def close_turn() -> None:
        nonlocal current
        if current is not None and current["steps"]:
            if pending_llm_steps:
                pending_llm_steps.clear()
            _finalize_turn(current)
            turns.append(current)
        current = None

    def add_step(
        turn: dict[str, Any],
        step_type: str,
        content: str,
        *,
        timestamp: str | None = None,
        tool_name: str = "",
        tool_use_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        nonlocal step_idx
        if not content and step_type not in {"tool_decision", "tool_execution"}:
            return None

        if step_type in {"thinking_explicit", "final_response"}:
            for prev in reversed(turn["steps"]):
                if prev.get("step_type") != step_type:
                    continue
                relation = _snapshot_relation(prev.get("content"), content)
                if relation == "new_extends_prev":
                    turn["steps"].remove(prev)
                    try:
                        pending_llm_steps.remove(prev)
                    except ValueError:
                        pass
                    prev.setdefault("metadata", {})["superseded_by_snapshot"] = True
                    break
                if relation == "prev_extends_new":
                    return None
                break

        sig = (step_type, tool_use_id or "", _compact_text(content, 1200))
        if sig in seen_steps:
            return None
        seen_steps.add(sig)
        step_idx += 1
        step = _new_step(
            step_idx,
            turn["turn_index"],
            step_type,
            content,
            timestamp=timestamp,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            metadata=metadata,
        )
        if step_type in {"user_input", "tool_execution"}:
            step["tokens"] = 0
        turn["steps"].append(step)
        t_ms = _ts_ms(timestamp)
        if t_ms:
            start_ms = turn.get("turn_start_ms_observed")
            end_ms = turn.get("turn_end_ms_observed")
            turn["turn_start_ms_observed"] = min(start_ms or t_ms, t_ms)
            turn["turn_end_ms_observed"] = max(end_ms or t_ms, t_ms)
        return step

    def apply_token_usage(turn: dict[str, Any], usage: dict[str, Any] | None) -> None:
        if not isinstance(usage, dict):
            return
        nums = _usage_numbers(usage)
        existing = turn.get("usage") or {}
        turn["usage"] = {
            "input_tokens": int(existing.get("input_tokens") or 0) + nums["input_tokens"],
            "output_tokens": int(existing.get("output_tokens") or 0) + nums["output_tokens"],
            "cache_creation_input_tokens": int(existing.get("cache_creation_input_tokens") or 0) + nums["cache_creation_input_tokens"],
            "cache_read_input_tokens": int(existing.get("cache_read_input_tokens") or 0) + nums["cache_read_input_tokens"],
            "reasoning_output_tokens": int(existing.get("reasoning_output_tokens") or 0) + nums["reasoning_output_tokens"],
            "source": "codex_token_count",
        }

    def add_compact_event(ts: str | None, summary: str, payload: Any) -> None:
        t_ms = _ts_ms(ts)
        if not t_ms:
            return
        text = summary or ""
        turn_index = current.get("turn_index") if current is not None else None
        compact_events.append(
            {
                "t_ms": t_ms,
                "phase": "after",
                "trigger": "auto",
                "source": "codex_rollout",
                "turn_index": turn_index,
                "summary": text,
                "summary_preview": _compact_text(text, 240),
                "summary_chars": len(text),
                "payload": payload if isinstance(payload, dict) else None,
            }
        )
        if current is not None:
            current["turn_end_ms_observed"] = max(
                current.get("turn_end_ms_observed") or t_ms,
                t_ms,
            )

    def mark_pending_llm(step: dict[str, Any] | None) -> None:
        if step is not None:
            pending_llm_steps.append(step)

    def flush_llm_usage(turn: dict[str, Any], usage: dict[str, Any] | None, ts: str | None) -> None:
        nonlocal llm_call_count
        if not isinstance(usage, dict):
            pending_llm_steps.clear()
            return
        if not pending_llm_steps:
            return
        anchor = next(
            (
                s
                for s in reversed(pending_llm_steps)
                if s.get("step_type") in {"tool_decision", "final_response"}
            ),
            pending_llm_steps[-1],
        )
        start_candidates = [_ts_ms(s.get("timestamp")) for s in pending_llm_steps]
        start_candidates = [x for x in start_candidates if x]
        end_ms = _ts_ms(ts)
        start_ms = min(start_candidates) if start_candidates else None
        dur_ms = (end_ms - start_ms) if (end_ms and start_ms and end_ms >= start_ms) else None
        model = model_seen or _display_model(session_meta.get("model")) or _model_from_config() or "unknown"
        _attach_llm_otel(
            anchor,
            usage,
            model=model,
            provider=provider,
            shared=False,
            duration_ms=dur_ms,
        )
        anchor.setdefault("metadata", {})["llm_call_index"] = llm_call_count
        anchor.setdefault("metadata", {})["invocation_category"] = "llm_call"
        for step in pending_llm_steps:
            if step is anchor:
                continue
            _attach_llm_otel(step, usage, model=model, provider=provider, shared=True)
            step.setdefault("metadata", {})["llm_call_index"] = llm_call_count
        llm_call_count += 1
        if end_ms:
            turn["turn_end_ms_observed"] = max(turn.get("turn_end_ms_observed") or end_ms, end_ms)
        apply_token_usage(turn, usage)
        pending_llm_steps.clear()

    for row in rows:
        ts = row.get("timestamp")
        rtype = row.get("type")
        payload = _payload(row)

        if rtype == "turn_context":
            if isinstance(payload, dict):
                model_seen = _display_model(payload.get("model")) or model_seen
            ensure_turn(ts)
            continue

        if rtype == "compacted":
            text = ""
            if isinstance(payload, dict):
                text = str(payload.get("message") or payload.get("summary") or "")
            else:
                text = str(payload or "")
            add_compact_event(ts, text, payload)
            continue

        if rtype == "event_msg" and isinstance(payload, dict):
            etype = payload.get("type")
            if etype == "user_message":
                text = _clean_user_prompt(str(payload.get("message") or ""))
                if _is_noise_user_text(text):
                    continue
                if current is not None and current["steps"]:
                    close_turn()
                turn = ensure_turn(ts)
                turn["user_query"] = text
                add_step(turn, "user_input", text, timestamp=ts)
            elif etype == "agent_message":
                turn = ensure_turn(ts)
                text = str(payload.get("message") or "")
                phase = str(payload.get("phase") or "")
                if phase == "final":
                    turn["final_response"] = text
                    mark_pending_llm(add_step(turn, "final_response", text, timestamp=ts))
                else:
                    mark_pending_llm(
                        add_step(
                            turn,
                            "thinking_explicit",
                            text,
                            timestamp=ts,
                            metadata={"codex_event": etype, "phase": phase or "commentary"},
                        )
                    )
            elif etype == "plan_update":
                turn = ensure_turn(ts)
                todos = _extract_codex_plan_todos(payload)
                if todos:
                    summary = "\n".join(
                        f"[{t.get('status')}] {t.get('content')}" for t in todos
                    )
                    mark_pending_llm(
                        add_step(
                            turn,
                            "tool_decision",
                            summary or "Codex plan update",
                            timestamp=ts,
                            tool_name="TodoWrite",
                            metadata={
                                "codex_event": etype,
                                "payload": payload,
                                "tool_input": {"todos": todos},
                                "input_summary": summary,
                                "tool_intent": summary,
                            },
                        )
                    )
                else:
                    add_step(
                        turn,
                        "strategy_shift",
                        json.dumps(payload, ensure_ascii=False),
                        timestamp=ts,
                        metadata={"codex_event": etype, "payload": payload},
                    )
            elif etype == "token_count":
                turn = ensure_turn(ts)
                info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
                last_token_usage = info.get("total_token_usage") or last_token_usage
                flush_llm_usage(turn, info.get("last_token_usage"), ts)
            elif etype == "task_complete":
                turn = ensure_turn(ts)
                text = str(payload.get("last_agent_message") or "")
                if text:
                    turn["final_response"] = text
                    mark_pending_llm(add_step(turn, "final_response", text, timestamp=ts))
                close_turn()
            continue

        if rtype == "response_item":
            turn = ensure_turn(ts)
            role, text = _message_text(payload)
            if role == "user":
                # Codex writes the same user prompt twice: first as a
                # response_item message, then as event_msg.user_message.  The
                # event_msg carries client metadata and lines up with hooks, so
                # use that as the canonical turn boundary and ignore this copy.
                continue
            if role == "assistant" and text:
                phase = payload.get("phase") if isinstance(payload, dict) else None
                if phase == "final":
                    turn["final_response"] = text
                    mark_pending_llm(add_step(turn, "final_response", text, timestamp=ts))
                else:
                    mark_pending_llm(
                        add_step(
                            turn,
                            "thinking_explicit",
                            text,
                            timestamp=ts,
                            metadata={"codex_item_type": "message", "phase": phase or "commentary"},
                        )
                    )
                continue

            if not isinstance(payload, dict):
                continue

            ptype = payload.get("type")
            if ptype == "reasoning":
                summary = payload.get("summary") or []
                parts: list[str] = []
                for item in summary:
                    if isinstance(item, dict):
                        parts.append(str(item.get("text") or item.get("summary") or ""))
                    elif item:
                        parts.append(str(item))
                reason_text = "\n".join(p for p in parts if p.strip()).strip()
                if reason_text:
                    mark_pending_llm(
                        add_step(
                            turn,
                            "thinking_explicit",
                            reason_text,
                            timestamp=ts,
                            metadata={"codex_item_type": ptype},
                        )
                    )
            elif ptype in {"function_call", "custom_tool_call", "web_search_call"}:
                raw_name = str(payload.get("name") or ptype)
                args = payload.get("arguments")
                if ptype == "custom_tool_call":
                    args = payload.get("input")
                if ptype == "web_search_call":
                    raw_name = "web_search_call"
                    args = payload.get("action") or payload
                parsed_args = _parse_jsonish(args)
                name = _canonical_tool_name(raw_name, parsed_args)
                call_id = str(payload.get("call_id") or payload.get("id") or "")
                if call_id:
                    call_tools[call_id] = name
                todos = _extract_codex_plan_todos(parsed_args) if name == "TodoWrite" else []
                if call_id and todos:
                    call_plan_todos[call_id] = todos
                summary = (
                    "\n".join(f"[{t.get('status')}] {t.get('content')}" for t in todos)
                    if todos
                    else _tool_input_summary(name, parsed_args)
                )
                tool_input = {"todos": todos} if todos else parsed_args
                mark_pending_llm(
                    add_step(
                        turn,
                        "tool_decision",
                        summary or name,
                        timestamp=ts,
                        tool_name=name,
                        tool_use_id=call_id,
                        metadata={
                            "codex_item_type": ptype,
                            "payload": payload,
                            "tool_input": tool_input,
                            "input_summary": summary,
                            "tool_intent": summary,
                            **({"codex_plan_source": raw_name} if todos else {}),
                        },
                    )
                )
            elif ptype in {"function_call_output", "custom_tool_call_output"}:
                output = str(payload.get("output") or "")
                call_id = str(payload.get("call_id") or payload.get("id") or "")
                name = call_tools.get(call_id) or _canonical_tool_name(str(payload.get("name") or "Tool"))
                todos = call_plan_todos.get(call_id) if call_id else None
                metadata = {
                    "codex_item_type": ptype,
                    "payload": payload,
                    "observed_output": output,
                }
                if todos:
                    metadata.update(
                        {
                            "tool_input": {"todos": todos},
                            "plan_full_todos": todos,
                            "codex_plan_source": "update_plan_output",
                        }
                    )
                add_step(
                    turn,
                    "tool_execution",
                    _tool_result_summary(output) or output,
                    timestamp=ts,
                    tool_name=name,
                    tool_use_id=call_id,
                    metadata=metadata,
                )
            continue

        if rtype == "event_msg" and isinstance(payload, dict):
            continue

    close_turn()
    if not turns:
        return None

    # Drop any setup-only turns left behind by environment snapshots.
    turns = [
        t for t in turns if t.get("user_query") or any(s.get("step_type") != "user_input" for s in t.get("steps") or [])
    ]
    if not turns:
        return None
    for idx, turn in enumerate(turns):
        turn["turn_index"] = idx
        start_ms = turn.get("turn_start_ms_observed")
        end_ms = turn.get("turn_end_ms_observed")
        if start_ms and end_ms and end_ms >= start_ms:
            turn["turn_duration_ms_observed"] = end_ms - start_ms
            turn["turn_duration_ms"] = end_ms - start_ms
            turn["turn_wallclock_span_ms"] = end_ms - start_ms
        for step in turn.get("steps") or []:
            step["turn_index"] = idx

    plan_timeline = _enrich_codex_plan_timeline(turns)

    tool_counter: Counter[str] = Counter()
    for turn in turns:
        for name in turn.get("tool_calls") or []:
            if name:
                tool_counter[str(name)] += 1

    topic = ""
    for turn in turns:
        q = _clean_user_prompt(str(turn.get("user_query") or ""))
        if q and not _is_noise_user_text(q):
            topic = q
            break
    topic = topic or str(session_meta.get("thread_name") or "Codex Session")
    model = (
        model_seen
        or _display_model(session_meta.get("model"))
        or _model_from_config()
        or "unknown"
    )
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    session_start_candidates = [
        int(t.get("turn_start_ms_observed") or 0)
        for t in turns
        if int(t.get("turn_start_ms_observed") or 0) > 0
    ]
    if not session_start_candidates:
        session_start_candidates = [
            ms for ms in (_ts_ms(row.get("timestamp")) for row in rows) if ms
        ]
    session_activity_candidates = [
        int(t.get("turn_end_ms_observed") or 0)
        for t in turns
        if int(t.get("turn_end_ms_observed") or 0) > 0
    ]
    for turn in turns:
        for step in turn.get("steps") or []:
            if isinstance(step, dict):
                ms = _ts_ms(step.get("timestamp"))
                if ms:
                    session_activity_candidates.append(ms)
    if not session_activity_candidates:
        session_activity_candidates = list(session_start_candidates)
    session_started_at = _iso_from_ms(min(session_start_candidates)) or now
    session_last_activity_at = _iso_from_ms(max(session_activity_candidates)) or session_started_at
    for turn in turns:
        u = turn.get("usage") or {}
        turn["otel"] = {
            "schema": "opentelemetry-genai/0.1",
            "step_kind": "llm_call",
            "operation_name": "codex.turn",
            "model": str(model),
            "provider": provider,
            "model_source": "transcript",
            "models_seen": [str(model)] if model != "unknown" else [],
            "token_usage": _token_usage_view(u, model=str(model), source="codex_turn_sum"),
        }

    total_input = sum(int((t.get("usage") or {}).get("input_tokens") or 0) for t in turns)
    total_output = sum(int((t.get("usage") or {}).get("output_tokens") or 0) for t in turns)
    cached_input = sum(int((t.get("usage") or {}).get("cache_read_input_tokens") or 0) for t in turns)
    reasoning_output = sum(int((t.get("usage") or {}).get("reasoning_output_tokens") or 0) for t in turns)
    cot = {
        "session_id": f"codex-{sid}" if not str(sid).startswith("codex-") else sid,
        "transcript_path": str(path),
        "extracted_at": session_last_activity_at,
        "session_started_at": session_started_at,
        "session_last_activity_at": session_last_activity_at,
        "collected_at": now,
        "agent_type": "codex",
        "turns": turns,
        "total_tool_calls": sum(tool_counter.values()),
        "total_strategy_shifts": sum(t.get("strategy_shifts", 0) for t in turns),
        "total_thinking_steps": sum(t.get("thinking_depth", 0) for t in turns),
        "tool_call_distribution": dict(tool_counter),
        "avg_steps_per_turn": sum(len(t.get("steps") or []) for t in turns) / max(1, len(turns)),
        "avg_complexity": sum(t.get("complexity_score", 0.0) for t in turns) / max(1, len(turns)),
        "is_parent": False,
        "sub_sessions": [],
        "plan_timeline": plan_timeline,
        "observed_events": None,
        "invocation_stats": {
            "llm_calls": llm_call_count,
            "rag_queries": 0,
            "web_searches": int(tool_counter.get("WebSearch", 0)),
            "llm_call_distribution": {str(model): llm_call_count} if model != "unknown" else {},
            "rag_query_distribution": {},
            "web_search_distribution": {"WebSearch": int(tool_counter.get("WebSearch", 0))}
            if tool_counter.get("WebSearch")
            else {},
        },
        "script_artifacts": [],
        "script_stats": None,
        "mode_transitions": [],
        "plan_proposals": [],
        "otel_view": {
            "schema": "opentelemetry-genai/0.1",
            "trace_id": sid.replace("-", "")[:32].ljust(32, "0"),
            "root_span_id": sid.replace("-", "")[:16].ljust(16, "0"),
            "service": {"name": "codex", "version": str(session_meta.get("cli_version") or "")},
            "model": str(model),
            "provider": provider,
            "model_source": "transcript",
            "models_seen": [str(model)] if model != "unknown" else [],
            "agent_name": "codex",
            "session_id": f"codex-{sid}" if not str(sid).startswith("codex-") else sid,
            "conversation_id": str(sid),
            "totals": {
                "turns": len(turns),
                "steps": sum(len(t.get("steps") or []) for t in turns),
                "tool_calls": sum(tool_counter.values()),
                "input_tokens": total_input,
                "output_tokens": total_output,
                "cache_read_input_tokens": cached_input,
                "reasoning_output_tokens": reasoning_output,
                "cost_usd": None,
            },
            "token_usage": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_input + total_output,
                "cache_read_input_tokens": cached_input,
                "reasoning_output_tokens": reasoning_output,
                "cost_usd": None,
                "cost_reason": "codex_native_usage",
                "currency": "USD",
                "is_estimate": False,
                "model_key": str(model),
                "source": "codex_token_count",
            },
            "actual_token_usage": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_input + total_output,
                "cache_read_tokens": cached_input,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "reasoning_output_tokens": reasoning_output,
                "cost_usd": None,
                "currency": "USD",
                "source": "codex_token_count",
            },
            "actual_cost_usd": None,
            "client_runtime": {
                "cursor_version": None,
                "user_email": None,
                "events_count": len(rows),
                "events_path": str(path),
                "model_distribution": {str(model): 1} if model != "unknown" else {},
            },
            "eval": None,
            "request_params": {
                "temperature": None,
                "top_p": None,
                "max_tokens": None,
                "seed": None,
                "stop_sequences": None,
                "_note": "Codex rollout transcript does not expose request params here.",
            },
            "missing_signals": [],
            "hints": [],
            "generated_at_ms": int(time.time() * 1000),
        },
        "resource_attributes": {
            "service.name": "codex",
            "service.version": str(session_meta.get("cli_version") or ""),
            "service.namespace": "agent-cot",
        },
        "session_meta": {
            "cursor_version": None,
            "user_email": None,
            "workspace_roots": [session_meta.get("cwd")] if session_meta.get("cwd") else [],
            "transcript_path": str(path),
            "hook_events_observed": {},
            "model_id": model,
            "model_provider": provider,
            "codex_originator": session_meta.get("originator"),
            "codex_source": session_meta.get("source"),
        },
        "user_activity": [],
        "subagent_timeline": [],
        "permission_events": [],
        "compact_events": compact_events,
        "notification_events": [],
        "environment_events": [],
        "topic": _compact_text(str(topic).splitlines()[0], 120),
    }
    return cot


def collect(session_id: str | None, recent_seconds: int) -> int:
    files = _find_rollouts(session_id, recent_seconds)
    if not files:
        _pipeline_log("no_rollout", sid=session_id or "-", ok=False)
        return 1
    out_dir = _data_root() / "cot"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in files:
        cot = _build_cot(path)
        if not cot:
            continue
        sid = cot["session_id"]
        target = out_dir / f"{sid}_cot.json"
        tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        payload = json.dumps(cot, ensure_ascii=False, indent=2)
        try:
            previous_payload = target.read_text(encoding="utf-8") if target.is_file() else None
        except Exception:
            previous_payload = None
        payload_changed = _stable_critic_payload(previous_payload) != _stable_critic_payload(payload)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        written += 1
        _pipeline_log("cot_written", sid=sid, transcript=str(path), target=str(target))
        try:
            source_mtime = path.stat().st_mtime
        except Exception:
            source_mtime = 0
        if _should_spawn_critic(sid, source_mtime, payload_changed):
            _spawn_critic(sid)
    return 0 if written else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--recent-seconds", type=int, default=7200)
    args = parser.parse_args(argv)
    return collect(args.session_id, args.recent_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
