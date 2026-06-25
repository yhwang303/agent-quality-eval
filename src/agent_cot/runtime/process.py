"""Cross-OS detached spawn + graceful kill for the dashboard backend.

The trick we need is "spawn a long-lived child that survives the
parent CLI exiting", which is unfortunately not one-line on either
POSIX or Windows:

* POSIX: ``start_new_session=True`` puts the child in its own session
  so SIGHUP from the closing terminal doesn't reach it.
* Windows: ``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`` does the
  equivalent — child has no console attached, won't die when the CLI
  exits.

Both platforms also need stdout / stderr redirected to a log file,
because the user's terminal is going to disappear.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from ..agents.base import CursorCotError
from ..installer.platform_paths import ensure_dir
from .pid_file import is_pid_running


class ProcessSpawnError(CursorCotError):
    """Raised when the backend fails to spawn (rare; usually file-not-found)."""


# Marker substring written into the spawn command line so we can later
# verify a PID belongs to *us* and not a recycled unrelated process.
SPAWN_CMDLINE_MARKER = "agent-cot-backend"

# v0.18.5: 同样的标记机制给 transcript_watcher daemon —— 跟 backend 用同一套
# is_pid_running / terminate_pid 校验逻辑，避免 PID 复用时误杀别的进程。
WATCHER_CMDLINE_MARKER = "agent-cot-watcher"


def _build_uvicorn_invocation(
    *,
    port: int,
    host: str,
    backend_dir: Path,
    extra_env: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Return ``(argv, env)`` for spawning the dashboard backend.

    We invoke uvicorn through ``python -c`` rather than the
    ``uvicorn`` console-script so the binding to *our* venv is
    explicit (even if the user has a different uvicorn on PATH).

    The ``agent-cot-backend`` marker comment in the command lets
    :func:`pid_file.is_pid_running` recognise the process by its
    cmdline.
    """
    snippet = (
        # `# agent-cot-backend` is a no-op comment used purely so the
        # marker shows up in the OS-level command-line listing.
        f"# {SPAWN_CMDLINE_MARKER}\n"
        "import uvicorn\n"
        f"uvicorn.run('main:app', host={host!r}, port={int(port)}, reload=False)\n"
    )
    argv = [sys.executable, "-c", snippet]

    env = os.environ.copy()
    # PYTHONPATH so `main:app` resolves even when cwd-shenanigans fail
    # on Windows. We push backend_dir to the *front*.
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{backend_dir}{os.pathsep}{pp}" if pp else str(backend_dir)
    )
    # Make stdout/stderr line-buffered so users tailing the log get
    # immediate feedback (default Python is fully buffered when piped).
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    return argv, env


def spawn_backend(
    *,
    port: int,
    backend_dir: Path,
    log_path: Path,
    host: str = "127.0.0.1",
    extra_env: dict[str, str] | None = None,
) -> tuple[int, list[str]]:
    """Detach-spawn the FastAPI backend.

    Returns ``(pid, argv)``. The caller is responsible for persisting
    them via :class:`pid_file.PidFile`.

    Raises :class:`ProcessSpawnError` if ``backend_dir`` is missing
    or doesn't contain a ``main.py``.
    """
    backend_dir = backend_dir.resolve()
    if not backend_dir.is_dir():
        raise ProcessSpawnError(f"backend dir does not exist: {backend_dir}")
    if not (backend_dir / "main.py").is_file():
        raise ProcessSpawnError(
            f"backend dir is missing main.py: {backend_dir}"
        )

    argv, env = _build_uvicorn_invocation(
        port=port, host=host, backend_dir=backend_dir, extra_env=extra_env
    )

    ensure_dir(log_path.parent)
    # Open in append so successive starts share one file the user can
    # tail; truncation would lose the previous crash's last words.
    log_fh = open(log_path, "ab")

    creationflags = 0
    start_new_session = False
    if os.name == "nt":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_NO_WINDOW = 0x08000000
        creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    else:
        start_new_session = True

    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(backend_dir),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            creationflags=creationflags,
            start_new_session=start_new_session,
        )
    except OSError as exc:
        log_fh.close()
        raise ProcessSpawnError(f"failed to spawn backend: {exc}") from exc

    # We deliberately do NOT close ``log_fh`` here — the child still
    # needs the FD on POSIX. The OS will reap it when the child exits.
    return proc.pid, argv


def terminate_pid(
    pid: int,
    *,
    cmdline_marker: str | None = SPAWN_CMDLINE_MARKER,
    grace_seconds: float = 5.0,
    poll_interval: float = 0.1,
) -> bool:
    """Ask ``pid`` to exit, escalating to SIGKILL after ``grace_seconds``.

    Returns ``True`` if the process is no longer running by the time
    we return. Returns ``False`` only if SIGKILL itself failed
    (extremely rare; usually means the PID never existed).

    The ``cmdline_marker`` guard prevents us from killing an unrelated
    process that has reused the PID — important on long-running
    laptops where ``backend.pid`` can outlive its original process.
    """
    if pid <= 0:
        return True
    if not is_pid_running(pid, cmdline_marker=cmdline_marker):
        return True

    try:
        if os.name == "nt":
            # On Windows ``terminate`` sends WM_CLOSE/TerminateProcess;
            # there is no SIGTERM analogue. We fall through to the
            # poll loop and only escalate to a hard kill if needed
            # (which is the same call here, but we still wait the
            # grace period in case the process can clean up).
            _windows_terminate(pid)
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError:
        return False

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not is_pid_running(pid):
            return True
        time.sleep(poll_interval)

    # Hard kill.
    try:
        if os.name == "nt":
            _windows_terminate(pid, force=True)
        else:
            os.kill(pid, signal.SIGKILL)
    except OSError:
        return False

    # Final check.
    time.sleep(0.2)
    return not is_pid_running(pid)


def _windows_terminate(pid: int, *, force: bool = False) -> None:
    """Best-effort Windows process termination via taskkill."""
    cmd = ["taskkill", "/PID", str(pid), "/T"]
    if force:
        cmd.append("/F")
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        kwargs["startupinfo"] = startupinfo
    # Run quietly; failures bubble up as non-zero exit which we ignore.
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        **kwargs,
    )


def render_argv(argv: list[str]) -> str:
    """Pretty-print ``argv`` for ``status`` and log diagnostics."""
    return " ".join(shlex.quote(a) for a in argv)


def spawn_python_script(
    *,
    script_path: Path,
    script_args: list[str] | None = None,
    log_path: Path,
    cmdline_marker: str,
    cwd: Path | None = None,
    extra_env: dict[str, str] | None = None,
    python_executable: str | None = None,
) -> tuple[int, list[str]]:
    """Detach-spawn an arbitrary Python script (e.g. transcript_watcher.py).

    Modelled on :func:`spawn_backend` but generic — we want one code path for
    any long-lived child the CLI launches alongside the backend. Same OS
    detach semantics (``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`` on
    Windows, ``start_new_session`` on POSIX), same append-mode log file.

    The ``cmdline_marker`` is injected as a no-op ``-c`` bootstrap so
    ``is_pid_running`` / ``terminate_pid`` can verify the PID is ours when
    we later try to kill it. Without this, on a PID-reused machine
    ``agent-cot stop`` could nuke an unrelated python process (rare, but
    bad enough to be worth guarding against — same reasoning as backend).
    """
    script_path = script_path.resolve()
    if not script_path.is_file():
        raise ProcessSpawnError(f"script does not exist: {script_path}")

    py = python_executable or sys.executable

    # We exec the script via "python -c bootstrap" so the marker comment
    # actually appears in the OS-level command line. If we just did
    # ``python <script.py>`` the cmdline wouldn't carry our marker and PID
    # verification would degrade to "is this PID alive at all".
    bootstrap = (
        f"# {cmdline_marker}\n"
        "import runpy, sys\n"
        f"sys.argv = [{str(script_path)!r}]"
        + ("".join(f" + [{a!r}]" for a in (script_args or [])))
        + "\n"
        f"runpy.run_path({str(script_path)!r}, run_name='__main__')\n"
    )
    argv = [py, "-c", bootstrap]

    env = os.environ.copy()
    if getattr(sys, "frozen", False):
        env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)

    ensure_dir(log_path.parent)
    log_fh = open(log_path, "ab")

    creationflags = 0
    start_new_session = False
    if os.name == "nt":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_NO_WINDOW = 0x08000000
        creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    else:
        start_new_session = True

    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            creationflags=creationflags,
            start_new_session=start_new_session,
        )
    except OSError as exc:
        log_fh.close()
        raise ProcessSpawnError(f"failed to spawn script: {exc}") from exc

    # Same FD-handoff trick as spawn_backend — child still owns log_fh on POSIX.
    return proc.pid, argv


__all__ = [
    "SPAWN_CMDLINE_MARKER",
    "WATCHER_CMDLINE_MARKER",
    "ProcessSpawnError",
    "render_argv",
    "spawn_backend",
    "spawn_python_script",
    "terminate_pid",
]
