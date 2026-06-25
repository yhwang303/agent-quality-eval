"""PID file with metadata, plus cross-OS process-existence checks.

A naked PID file is fragile for two reasons:

1. **PID reuse**: the OS may hand the old PID to an unrelated process
   in the seconds between our backend dying and the next ``status``
   call. We dodge this by persisting a "started_at" timestamp that the
   verifier can match against the live process's CreationDate.
2. **Crash safety**: if the file exists but the PID is gone, callers
   need to treat it as "stale" not "running". :func:`is_pid_running`
   gives that distinction.

We persist as JSON (not bare integer) so future fields — backend port,
log path, agent name — can be added without breaking the schema.
"""

from __future__ import annotations

import errno
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..installer.platform_paths import agent_cot_root, ensure_dir

PID_RECORD_VERSION = 1
DEFAULT_PID_FILENAME = "backend.pid"
# v0.18.5: ``transcript_watcher.py`` 由 ``agent-cot start`` 一并 spawn —— 单独
# 一个 PID 文件，跟 backend 分开管，``stop`` 能把两个都收掉。
WATCHER_PID_FILENAME = "watcher.pid"


@dataclass
class PidRecord:
    """In-memory representation of a backend PID file."""

    pid: int
    port: int
    started_at: float = field(default_factory=time.time)
    """Unix timestamp; matched against ``psutil`` create_time when verifying."""

    schema: int = PID_RECORD_VERSION
    """File-format version; incremented on breaking layout changes."""

    log_path: str | None = None
    """Where the spawned process is writing its stdout/stderr."""

    cmdline_marker: str | None = None
    """Substring guaranteed to appear in the process's command line.

    Lets ``status`` validate that the PID hasn't been recycled into
    something completely unrelated (e.g. "explorer.exe" stealing 41148).
    """

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class PidFile:
    """Filesystem-bound view of a PID file."""

    path: Path

    @classmethod
    def default(cls) -> PidFile:
        """The default location: ``~/.agent-cot/run/backend.pid``."""
        return cls(path=agent_cot_root() / "run" / DEFAULT_PID_FILENAME)

    @classmethod
    def for_watcher(cls) -> PidFile:
        """v0.18.5: ``~/.agent-cot/run/watcher.pid`` —— transcript_watcher daemon."""
        return cls(path=agent_cot_root() / "run" / WATCHER_PID_FILENAME)

    def exists(self) -> bool:
        return self.path.is_file()

    def read(self) -> PidRecord | None:
        return read_pid_record(self.path)

    def write(self, record: PidRecord) -> Path:
        return write_pid_record(record, self.path)

    def remove(self) -> bool:
        return remove_pid_file(self.path)


# ---------------------------------------------------------------------------
# Module-level helpers (module-state-free, easy to test)
# ---------------------------------------------------------------------------


def write_pid_record(record: PidRecord, path: Path) -> Path:
    """Persist ``record`` atomically to ``path``."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(record.to_json(), encoding="utf-8")
    tmp.replace(path)
    return path


def read_pid_record(path: Path) -> PidRecord | None:
    """Load a previously-written :class:`PidRecord`. Returns ``None`` if
    the file is missing or unparseable.

    Tolerant of partial writes (returns ``None`` rather than raising)
    so a corrupt PID file never breaks ``status``.
    """
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    try:
        return PidRecord(
            pid=int(raw["pid"]),
            port=int(raw["port"]),
            started_at=float(raw.get("started_at", 0.0)),
            schema=int(raw.get("schema", PID_RECORD_VERSION)),
            log_path=raw.get("log_path"),
            cmdline_marker=raw.get("cmdline_marker"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def remove_pid_file(path: Path) -> bool:
    """Delete the PID file. Returns ``True`` if a file was removed."""
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# is_pid_running — cross-OS process existence
# ---------------------------------------------------------------------------


def is_pid_running(
    pid: int,
    *,
    cmdline_marker: str | None = None,
    started_at: float | None = None,
) -> bool:
    """Return ``True`` if ``pid`` belongs to a still-alive process owned
    by us.

    "Owned by us" is determined progressively:

    1. The PID exists at the OS level.
    2. (optional) The process command line contains
       ``cmdline_marker`` — defends against PID reuse where the new
       owner is unrelated.
    3. (optional) The process create_time is within ±2s of
       ``started_at`` — additional defence-in-depth when ``psutil`` is
       installed.

    Returns ``False`` for PID 0 and any non-positive value, since those
    aren't real targets and the kill(0) trick on POSIX would no-op
    against the calling process group.
    """
    if pid is None or pid <= 0:
        return False

    if not _os_pid_alive(pid):
        return False

    # If the caller didn't ask for stronger verification, the OS-level
    # answer is good enough.
    if cmdline_marker is None and started_at is None:
        return True

    psutil = _try_import_psutil()
    if psutil is None:
        # We were asked for verification but can't deliver it; be
        # permissive (better a false-positive "running" than to nuke a
        # working backend).
        return True

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True  # Better safe than sorry: assume it's ours.

    if started_at is not None:
        try:
            create_time = proc.create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return True
        if abs(create_time - started_at) > 2.0:
            return False

    if cmdline_marker is not None:
        try:
            cmdline = " ".join(proc.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return True
        if cmdline_marker not in cmdline:
            return False

    return True


def _os_pid_alive(pid: int) -> bool:
    """Cheapest possible "is this PID running" probe.

    On POSIX we send signal 0 (no-op delivery test). On Windows we use
    ``psutil`` if available; otherwise we fall back to opening the
    process via ``OpenProcess`` through ``ctypes`` — that always works
    even on a fresh interpreter.
    """
    if os.name == "nt":
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't own it; for our purposes that's
        # still "alive".
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return True
    return True


def _windows_pid_alive(pid: int) -> bool:
    """Return True if a Windows PID is alive, without depending on psutil."""
    psutil = _try_import_psutil()
    if psutil is not None:
        return psutil.pid_exists(pid)

    # ctypes fallback so we work even before psutil is installed.
    try:
        import ctypes
        import ctypes.wintypes as wt
    except ImportError:  # pragma: no cover - ctypes always ships with CPython
        return True

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32
    OpenProcess = kernel32.OpenProcess
    OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    OpenProcess.restype = wt.HANDLE

    handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        exit_code = wt.DWORD()
        GetExitCodeProcess = kernel32.GetExitCodeProcess
        GetExitCodeProcess.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD)]
        if not GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _try_import_psutil():  # pragma: no cover - trivial wrapper
    try:
        import psutil

        return psutil
    except ImportError:
        return None


__all__ = [
    "DEFAULT_PID_FILENAME",
    "PID_RECORD_VERSION",
    "PidFile",
    "PidRecord",
    "WATCHER_PID_FILENAME",
    "is_pid_running",
    "read_pid_record",
    "remove_pid_file",
    "write_pid_record",
]
