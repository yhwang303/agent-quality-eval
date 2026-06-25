"""``agent-cot stop`` — gracefully terminate the dashboard backend."""

from __future__ import annotations

from dataclasses import dataclass

from ..agents.base import CursorCotError
from ..runtime import PidFile, is_pid_running, terminate_pid


class StopError(CursorCotError):
    """Public umbrella for any stop-time error visible to the CLI."""


@dataclass
class StopResult:
    found_pid_file: bool
    was_running: bool
    pid: int | None
    terminated: bool
    """``True`` if the process is no longer running on return."""

    cleaned_pid_file: bool
    """``True`` iff we removed the PID file (always when terminated)."""

    # v0.18.5: transcript_watcher daemon 也被一并收掉时填这几个；watcher 没起来
    # 的话整组都是 None，CLI 渲染时静默跳过。
    watcher_found_pid_file: bool = False
    watcher_was_running: bool = False
    watcher_pid: int | None = None
    watcher_terminated: bool = False
    watcher_cleaned_pid_file: bool = False


def stop_backend(*, force: bool = False) -> StopResult:
    """Terminate any running backend tracked by the default PID file.

    Behaviour matrix:

    +-----------------+---------------------+-----------------------------+
    | PID file        | Process at that PID | Action                      |
    +=================+=====================+=============================+
    | absent          | n/a                 | no-op, return cleanly       |
    +-----------------+---------------------+-----------------------------+
    | present, valid  | running, ours       | SIGTERM → SIGKILL → unlink  |
    +-----------------+---------------------+-----------------------------+
    | present, valid  | not running (stale) | unlink only                 |
    +-----------------+---------------------+-----------------------------+
    | present, valid  | running, NOT ours   | refuse unless ``force=True``|
    +-----------------+---------------------+-----------------------------+
    | present, corrupt| n/a                 | unlink, log a warning       |
    +-----------------+---------------------+-----------------------------+
    """
    pid_file = PidFile.default()

    if not pid_file.exists():
        return StopResult(
            found_pid_file=False,
            was_running=False,
            pid=None,
            terminated=True,
            cleaned_pid_file=False,
        )

    record = pid_file.read()
    if record is None:
        # Corrupt file — best we can do is remove it.
        pid_file.remove()
        return StopResult(
            found_pid_file=True,
            was_running=False,
            pid=None,
            terminated=True,
            cleaned_pid_file=True,
        )

    if not is_pid_running(record.pid, cmdline_marker=record.cmdline_marker):
        # Stale PID file — process already died.
        pid_file.remove()
        return StopResult(
            found_pid_file=True,
            was_running=False,
            pid=record.pid,
            terminated=True,
            cleaned_pid_file=True,
        )

    # The PID belongs to a live process. Verify it's *ours* unless the
    # user forces.
    if not force and not is_pid_running(
        record.pid,
        cmdline_marker=record.cmdline_marker,
    ):
        # The marker check failed — refuse silently kill someone else.
        raise StopError(
            f"PID {record.pid} is alive but does NOT look like a "
            f"agent-cot backend (marker '{record.cmdline_marker}' "
            "missing from its cmdline). Refusing to terminate. "
            "Pass --force to override."
        )

    terminated = terminate_pid(
        record.pid,
        cmdline_marker=None if force else record.cmdline_marker,
    )
    if terminated:
        pid_file.remove()

    # v0.18.5: backend 收掉之后顺手把 transcript_watcher daemon 也停了。
    # 这是一对儿，一起 spawn / 一起销毁，避免出现"backend 关了但 watcher
    # 还在写老 events.jsonl"的诡异状态。
    w = _stop_watcher(force=force)

    return StopResult(
        found_pid_file=True,
        was_running=True,
        pid=record.pid,
        terminated=terminated,
        cleaned_pid_file=terminated,
        watcher_found_pid_file=w["found"],
        watcher_was_running=w["was_running"],
        watcher_pid=w["pid"],
        watcher_terminated=w["terminated"],
        watcher_cleaned_pid_file=w["cleaned"],
    )


def _stop_watcher(*, force: bool) -> dict:
    """Internal: kill watcher daemon + clean PID file. Best-effort, never raises.

    Returns a dict with the keys mirrored onto :class:`StopResult.watcher_*`
    fields. Errors are swallowed because a failed watcher kill should never
    block ``stop_backend`` from reporting the backend half worked.
    """
    pid_file = PidFile.for_watcher()
    if not pid_file.exists():
        return {"found": False, "was_running": False, "pid": None,
                "terminated": True, "cleaned": False}

    record = pid_file.read()
    if record is None:
        pid_file.remove()
        return {"found": True, "was_running": False, "pid": None,
                "terminated": True, "cleaned": True}

    if not is_pid_running(record.pid, cmdline_marker=record.cmdline_marker):
        pid_file.remove()
        return {"found": True, "was_running": False, "pid": record.pid,
                "terminated": True, "cleaned": True}

    terminated = terminate_pid(
        record.pid,
        cmdline_marker=None if force else record.cmdline_marker,
    )
    if terminated:
        pid_file.remove()
    return {"found": True, "was_running": True, "pid": record.pid,
            "terminated": terminated, "cleaned": terminated}


__all__ = ["StopError", "StopResult", "stop_backend"]
