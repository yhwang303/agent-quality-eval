"""runtime/ — process-lifecycle primitives for the local dashboard.

This package owns *only* the cross-cutting concerns of running a
long-lived backend on the user's machine:

* persisting a PID + metadata so ``status`` / ``stop`` can find it,
* spawning a detached child process that survives the CLI exit,
* probing the backend's HTTP endpoint to know when it's actually up.

High-level ``commands.start`` / ``commands.stop`` glue these together;
``installer/`` is for one-time disk side-effects (hooks.json, config),
``runtime/`` is for repeated-during-normal-use side-effects.
"""

from __future__ import annotations

from .health import HealthCheckError, wait_for_backend
from .pid_file import (
    PidFile,
    PidRecord,
    is_pid_running,
    read_pid_record,
    remove_pid_file,
    write_pid_record,
)
from .process import (
    SPAWN_CMDLINE_MARKER,
    WATCHER_CMDLINE_MARKER,
    ProcessSpawnError,
    spawn_backend,
    spawn_python_script,
    terminate_pid,
)

__all__ = [
    "HealthCheckError",
    "PidFile",
    "PidRecord",
    "ProcessSpawnError",
    "SPAWN_CMDLINE_MARKER",
    "WATCHER_CMDLINE_MARKER",
    "is_pid_running",
    "read_pid_record",
    "remove_pid_file",
    "spawn_backend",
    "spawn_python_script",
    "terminate_pid",
    "wait_for_backend",
    "write_pid_record",
]
