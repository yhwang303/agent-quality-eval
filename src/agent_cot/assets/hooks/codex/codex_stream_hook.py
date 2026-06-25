#!/usr/bin/env python
"""Codex lifecycle hook stream for agent-cot.

Captures official Codex hook payloads into ``$CODEX_HOME/state/events`` and
triggers the sidecar transcript collector on turn/session boundary events.
The script never writes to stdout/stderr and always exits 0 so a logging issue
cannot block Codex.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()


CODEX_HOME = _codex_home()
STATE_DIR = CODEX_HOME / "state"
EVENTS_DIR = STATE_DIR / "events"
PIPELINE_LOG = Path(
    os.environ.get("AGENT_COT_PIPELINE_LOG")
    or str(Path.home() / ".agent-cot" / "logs" / "pipeline.log")
).expanduser()

_COLLECT_TRIGGER_EVENTS = {"PostToolUse", "PostCompact", "Stop", "SubagentStop"}
_CRITIC_TRIGGER_EVENTS = {"Stop", "SubagentStop"}
_DEBOUNCE_FILE = STATE_DIR / "agent-cot-collect-debounce.json"
_CRITIC_DEBOUNCE_FILE = STATE_DIR / "agent-cot-critic-debounce.json"
_DEBOUNCE_SEC = 2
_CRITIC_DEBOUNCE_SEC = 30


def _force_utf8_stdin() -> None:
    try:
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        elif hasattr(sys.stdin, "buffer"):
            import io

            sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass


def _pipeline_log(event: str, sid: str = "-", ok: bool = True, **note: Any) -> None:
    try:
        PIPELINE_LOG.parent.mkdir(parents=True, exist_ok=True)
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
            f"[{ts}] [hook.codex] [codex] [sid={sid}] "
            f"event={event} status={'ok' if ok else 'FAIL'}"
            + (" " + " ".join(parts) if parts else "")
            + "\n"
        )
        with open(PIPELINE_LOG, "a", encoding="utf-8", errors="replace") as fh:
            fh.write(line)
    except Exception:
        pass


def _read_payload() -> dict[str, Any]:
    try:
        raw = sys.stdin.read() or ""
    except Exception:
        raw = ""
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {"payload": data}
    except Exception:
        return {"_raw_first200": raw[:200], "_raw_chars": len(raw)}


def _walk_values(obj: Any) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                out.append((str(k), v))
            else:
                out.extend(_walk_values(v))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_walk_values(item))
    return out


def _detect_event(payload: dict[str, Any]) -> str:
    for key in ("hook_event_name", "event", "event_name", "type"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    if len(sys.argv) >= 2:
        return str(sys.argv[1])
    return "Unknown"


def _detect_session_id(payload: dict[str, Any]) -> str:
    preferred = (
        "session_id",
        "thread_id",
        "conversation_id",
        "conversationId",
        "codex_session_id",
    )
    flat = _walk_values(payload)
    for want in preferred:
        for key, value in flat:
            if key == want and value:
                return value.strip()
    latest = _latest_rollout_id()
    if latest:
        return latest
    for _key, value in flat:
        if len(value) >= 26 and "-" in value and value.count("-") >= 3:
            return value.strip()
    return "codex-orphan"


def _latest_rollout_id() -> str | None:
    root = CODEX_HOME / "sessions"
    try:
        files = list(root.glob("*/*/*/*.jsonl"))
    except Exception:
        return None
    if not files:
        return None
    latest = max(files, key=lambda p: p.stat().st_mtime)
    name = latest.stem
    if "rollout-" in name:
        return name.split("rollout-", 1)[-1]
    return name


def _append_event(record: dict[str, Any]) -> None:
    sid = str(record.get("session_id") or "codex-orphan")
    target = EVENTS_DIR / sid / "events.jsonl"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str))
            fh.write("\n")
    except Exception:
        pass


def _should_collect(key: str) -> bool:
    now = time.time()
    state: dict[str, float] = {}
    try:
        if _DEBOUNCE_FILE.is_file():
            data = json.loads(_DEBOUNCE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                state = {str(k): float(v) for k, v in data.items() if now - float(v) < 3600}
    except Exception:
        state = {}
    if now - float(state.get(key, 0.0)) < _DEBOUNCE_SEC:
        return False
    state[key] = now
    try:
        _DEBOUNCE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _DEBOUNCE_FILE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return True


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


def _agent_quality_eval_runner_cmd(py: str, runner: str, args: list[str]) -> list[str]:
    if _is_agent_quality_eval_exe(py):
        return [py, "--agent-quality-eval-runner", runner, *args]
    if runner == "live-critic":
        code = "from agent_quality_eval.evaluation.live_critic import main; raise SystemExit(main())"
    else:
        code = "from agent_quality_eval.evaluation.critic import main; raise SystemExit(main())"
    return [py, "-c", code, *args]


def _should_run_critic(key: str) -> bool:
    now = time.time()
    state: dict[str, float] = {}
    try:
        if _CRITIC_DEBOUNCE_FILE.is_file():
            data = json.loads(_CRITIC_DEBOUNCE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                state = {str(k): float(v) for k, v in data.items() if now - float(v) < 3600}
    except Exception:
        state = {}
    if now - float(state.get(key, 0.0)) < _CRITIC_DEBOUNCE_SEC:
        return False
    state[key] = now
    try:
        _CRITIC_DEBOUNCE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CRITIC_DEBOUNCE_FILE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return True


def _trigger_collect(session_id: str) -> None:
    if not _should_collect(session_id):
        return
    collector = Path(__file__).with_name("codex_sidecar_collector.py")
    if not collector.is_file():
        return
    py = sys.executable or "python"
    sid_filter = _rollout_filter_for_session(session_id)
    cmd = [py, str(collector), "--recent-seconds", "7200"]
    if sid_filter:
        cmd.extend(["--session-id", sid_filter])
    # Codex appends rollout rows asynchronously around hook boundaries.  Run
    # a short delayed sweep series so Stop/PostToolUse cannot snapshot a half
    # flushed turn and leave the current trace incomplete until the next user
    # message.
    helper_code = (
        "import subprocess,sys,time\n"
        f"cmd={cmd!r}\n"
        "run_kwargs = {}\n"
        "if sys.platform == 'win32':\n"
        "    creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)\n"
        "    run_kwargs['creationflags'] = creationflags\n"
        "    startupinfo = subprocess.STARTUPINFO()\n"
        "    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW\n"
        "    startupinfo.wShowWindow = 0\n"
        "    run_kwargs['startupinfo'] = startupinfo\n"
        "for delay in (1, 5, 15, 45):\n"
        "    time.sleep(delay)\n"
        "    subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60, **run_kwargs)\n"
    )
    helper_cmd = [py, "-c", helper_code]
    popen_kwargs = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) | 0x00000008 | 0x00000200
        )
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        popen_kwargs["startupinfo"] = startupinfo
    log_path = STATE_DIR / "agent-cot-collector.log"
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "a", encoding="utf-8", errors="replace")
    except Exception:
        log_fh = subprocess.DEVNULL
    try:
        subprocess.Popen(
            helper_cmd,
            cwd=str(CODEX_HOME),
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            close_fds=True,
            **popen_kwargs,
        )
        _pipeline_log("collector_spawn", sid=session_id, collector=str(collector))
    except Exception as exc:
        _pipeline_log("collector_spawn", sid=session_id, ok=False, error=exc)


def _live_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in (
            "hook_event_name",
            "event",
            "tool",
            "tool_name",
            "toolName",
            "tool_use_id",
            "cwd",
        )
        if payload.get(key) is not None
    }


def _trigger_live_critic(session_id: str, event: str, payload: dict[str, Any] | None = None) -> None:
    if os.environ.get("AGENT_QUALITY_EVAL_LIVE_CRITIC_DISABLE"):
        return
    py = os.environ.get("COT_PYTHON") or _read_runtime_python() or sys.executable or "python"
    cmd = _agent_quality_eval_runner_cmd(py, "live-critic", [
        "--agent-type",
        "codex",
        "--source-event",
        event,
        "--session-id",
        session_id,
    ])
    slim_payload = _live_payload(payload or {})
    if slim_payload:
        cmd.extend(["--payload-json", json.dumps(slim_payload, ensure_ascii=False)])
    env = os.environ.copy()
    env["AGENT_QUALITY_EVAL_LIVE_CRITIC_DISABLE"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    popen_kwargs = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) | 0x00000008 | 0x00000200
        )
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        popen_kwargs["startupinfo"] = startupinfo
    try:
        subprocess.Popen(
            cmd,
            cwd=str(CODEX_HOME),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            **popen_kwargs,
        )
        _pipeline_log("live_critic_pulse", sid=session_id, source_event=event)
    except Exception as exc:
        _pipeline_log("live_critic_pulse", sid=session_id, ok=False, error=exc)


def _trigger_critic(session_id: str, event: str) -> None:
    if os.environ.get("AGENT_QUALITY_EVAL_CRITIC_DISABLE"):
        return
    key = f"{session_id}:{event}"
    if not _should_run_critic(key):
        return
    py = os.environ.get("COT_PYTHON") or _read_runtime_python() or sys.executable or "python"
    sid_filter = _rollout_filter_for_session(session_id)
    runner_cmd = _agent_quality_eval_runner_cmd(py, "critic", [
        "--agent-type",
        "codex",
        "--source-event",
        f"codex-stream:{event}",
        "--wait-seconds",
        "75",
        "--no-persist-eval",
    ])
    if sid_filter:
        runner_cmd.extend(["--session-id", sid_filter])
    delay = 2 if sid_filter else 12
    helper_code = (
        "import subprocess,sys,time\n"
        f"cmd={runner_cmd!r}\n"
        f"time.sleep({delay!r})\n"
        "run_kwargs = {}\n"
        "if sys.platform == 'win32':\n"
        "    creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)\n"
        "    run_kwargs['creationflags'] = creationflags\n"
        "    startupinfo = subprocess.STARTUPINFO()\n"
        "    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW\n"
        "    startupinfo.wShowWindow = 0\n"
        "    run_kwargs['startupinfo'] = startupinfo\n"
        "subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180, **run_kwargs)\n"
    )
    helper_cmd = [py, "-c", helper_code]
    env = os.environ.copy()
    env["AGENT_QUALITY_EVAL_CRITIC_DISABLE"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    popen_kwargs = {}
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
        if _is_agent_quality_eval_exe(py):
            proc = subprocess.Popen(
                runner_cmd,
                cwd=str(CODEX_HOME),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=log_fh,
                close_fds=True,
                **popen_kwargs,
            )
            _pipeline_log("critic_spawn", sid=session_id, pid=proc.pid, session_arg=sid_filter or "", delay=0)
            return
        proc = subprocess.Popen(
            helper_cmd,
            cwd=str(CODEX_HOME),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            close_fds=True,
            **popen_kwargs,
        )
        _pipeline_log("critic_spawn", sid=session_id, pid=proc.pid, session_arg=sid_filter or "", delay=delay)
    except Exception as exc:
        _pipeline_log("critic_spawn", sid=session_id, ok=False, error=exc)


def _rollout_filter_for_session(session_id: str) -> str | None:
    """Return a safe rollout filename filter, or None for recent-file sweep.

    Codex hook payloads often expose ``turn_id`` but not the rollout/session
    id. Filtering by that turn id finds no transcript, so the collector silently
    misses the just-finished turn. Only apply ``--session-id`` when the value
    actually appears in a rollout filename; otherwise scan recent rollouts.
    """
    sid = (session_id or "").replace("codex-", "").strip()
    if not sid or sid == "codex-orphan":
        return None
    try:
        root = CODEX_HOME / "sessions"
        for path in root.glob("*/*/*/*.jsonl"):
            if sid.lower() in path.name.lower():
                return sid
    except Exception:
        return None
    return None


def main() -> int:
    _force_utf8_stdin()
    try:
        payload = _read_payload()
        event = _detect_event(payload)
        sid = _detect_session_id(payload)
        now = datetime.now(timezone.utc)
        record = {
            "t_ms": int(now.timestamp() * 1000),
            "iso_ts": now.isoformat().replace("+00:00", "Z"),
            "session_id": sid,
            "hook_event": event,
            "cwd": os.getcwd(),
            "payload": payload,
        }
        _append_event(record)
        _pipeline_log(event, sid=sid)
        _trigger_live_critic(sid, event, payload)
        if event in _COLLECT_TRIGGER_EVENTS:
            _trigger_collect(sid)
        if event in _CRITIC_TRIGGER_EVENTS:
            _trigger_critic(sid, event)
    except Exception as exc:
        _pipeline_log("hook_exception", ok=False, error=exc)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        raise SystemExit(0)
