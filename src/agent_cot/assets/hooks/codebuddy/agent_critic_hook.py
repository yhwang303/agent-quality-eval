#!/usr/bin/env python
"""Fire-and-forget Agent Critic sidecar hook.

This script is intentionally tiny and defensive. IDE hooks call it on turn
boundary events, it extracts the best available session id from stdin, then
spawns the real runner in a detached Python process. The runner waits for
cot.json to land, calls the configured critic model when available, writes a
JSON report under ~/.agent-cot/data/critic, and persists the normal turn eval.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _log(event: str, **fields: Any) -> None:
    try:
        path = Path.home() / ".agent-cot" / "logs" / "critic-hook.log"
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


def _read_payload() -> dict[str, Any]:
    try:
        raw = sys.stdin.read() or ""
    except Exception:
        return {}
    if raw and raw[0] == "\ufeff":
        raw = raw[1:]
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {"payload": data}
    except Exception:
        return {"_raw_first200": raw[:200]}


def _walk(obj: Any) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                out.append((str(key), value))
            else:
                out.extend(_walk(value))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_walk(item))
    return out


def _pick(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    flat = _walk(payload)
    for want in keys:
        for key, value in flat:
            if key == want and value.strip():
                return value.strip()
    return None


def _read_runtime_python() -> str | None:
    try:
        path = Path.home() / ".agent-cot" / "runtime.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        value = data.get("python_executable") if isinstance(data, dict) else None
        return str(value) if value else None
    except Exception:
        return None


def _is_agent_quality_eval_exe(py: str) -> bool:
    name = Path(str(py or "")).name.lower()
    return name.startswith("agent-quality-eval") and name.endswith(".exe")


def _agent_quality_eval_runner_cmd(py: str, args: list[str]) -> list[str]:
    if _is_agent_quality_eval_exe(py):
        return [py, "--agent-quality-eval-runner", "critic", *args]
    runner_code = "from agent_quality_eval.evaluation.critic import main; raise SystemExit(main())"
    return [py, "-c", runner_code, *args]


def _spawn(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    if os.environ.get("AGENT_QUALITY_EVAL_CRITIC_DISABLE"):
        _log("disabled", agent=args.agent, source_event=args.event)
        return
    event = args.event or _pick(payload, ("hook_event_name", "event_name", "event", "type")) or "hook"
    session_id = args.session_id or _pick(
        payload,
        (
            "session_id",
            "sessionId",
            "conversation_id",
            "conversationId",
            "thread_id",
            "chatSessionId",
        ),
    )
    if args.agent == "codebuddy" and session_id and not session_id.startswith("codebuddy-"):
        session_id = "codebuddy-" + session_id
    py = os.environ.get("COT_PYTHON") or _read_runtime_python() or sys.executable or "python"
    cmd = _agent_quality_eval_runner_cmd(py, ["--agent-type", args.agent, "--source-event", event])
    if session_id:
        cmd.extend(["--session-id", session_id])
    _log(
        "spawn_prepare",
        agent=args.agent,
        source_event=event,
        session_id=session_id or "",
        python=py,
        command=cmd,
        payload_keys=sorted(payload.keys())[:20],
    )
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
        log_fh = log_path.open("a", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(Path.home()),
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            close_fds=True,
            **popen_kwargs,
        )
        _log("spawn_ok", agent=args.agent, source_event=event, session_id=session_id or "", pid=proc.pid)
    except Exception:
        _log(
            "spawn_error",
            agent=args.agent,
            source_event=event,
            session_id=session_id or "",
            error=traceback.format_exc()[-4000:],
        )
        return


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--agent", default="unknown")
    parser.add_argument("--event", default="")
    parser.add_argument("--session-id", default="")
    args, _unknown = parser.parse_known_args()
    try:
        payload = _read_payload()
        _log("invoked", agent=args.agent, source_event=args.event, argv=sys.argv, payload_keys=sorted(payload.keys())[:20])
        _spawn(args, payload)
    except Exception:
        _log("hook_error", agent=args.agent, source_event=args.event, error=traceback.format_exc()[-4000:])
    try:
        sys.stdout.write('{"continue":true}')
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
