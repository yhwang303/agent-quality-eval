"""Frozen executable entry point for observation-agent.

This module makes the PyInstaller-built executable act like three things:

* ``agent-cot`` CLI when arguments are provided.
* A one-click launcher when double-clicked with no arguments.
* A tiny Python-compatible runner for the package's own subprocess calls
  (``exe -c <code>`` and ``exe path/to/script.py ...``).

The last mode is what lets the existing backend / watcher / extractor spawn
code keep working after ``sys.executable`` becomes the frozen exe path.
"""

from __future__ import annotations

import runpy
import os
import shutil
import sys
import webbrowser
from pathlib import Path


def _prepare_frozen_assets() -> None:
    if not getattr(sys, "frozen", False):
        return
    if os.environ.get("AGENT_COT_ASSETS_ROOT"):
        return

    from . import __version__

    source = Path(getattr(sys, "_MEIPASS", "")) / "agent_cot" / "assets"
    if not source.is_dir():
        return

    target = Path.home() / ".agent-cot" / "frozen-assets" / __version__ / "assets"
    marker = target / ".complete"
    ready = (
        (target / "backend" / "main.py").is_file()
        and (target / "frontend-dist" / "index.html").is_file()
        and (target / "cot-extractor" / "scripts" / "extract_cot.py").is_file()
    )
    if ready:
        os.environ["AGENT_COT_ASSETS_ROOT"] = str(target)
        _pin_config_to_frozen_assets(target)
        return

    if not marker.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True)
        marker.write_text(__version__ + "\n", encoding="utf-8")
    os.environ["AGENT_COT_ASSETS_ROOT"] = str(target)
    _pin_config_to_frozen_assets(target)


def _pin_config_to_frozen_assets(target: Path) -> None:
    try:
        from .installer.config import load_config, save_config

        cfg = load_config()
        cfg.dashboard_repo = str(target)
        cfg.cot_extractor_repo = str(target / "cot-extractor")
        save_config(cfg)
    except Exception:
        return


def _seed_import_path() -> None:
    entries: list[str] = [str(Path.cwd())]
    pp = os.environ.get("PYTHONPATH", "")
    if pp:
        entries.extend(p for p in pp.split(os.pathsep) if p)
    for entry in reversed(entries):
        if entry and entry not in sys.path:
            sys.path.insert(0, entry)


def _run_dash_c() -> None:
    code = sys.argv[2] if len(sys.argv) > 2 else ""
    sys.argv = [sys.executable, "-c", *sys.argv[3:]]
    _seed_import_path()
    glb = {
        "__name__": "__main__",
        "__file__": "<agent-cot-frozen-c>",
        "__package__": None,
    }
    exec(compile(code, "<agent-cot-frozen-c>", "exec"), glb, glb)


def _run_script() -> None:
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    _seed_import_path()
    runpy.run_path(str(script), run_name="__main__")


def _process_command_line(pid: int) -> str | None:
    try:
        import psutil  # type: ignore

        return " ".join(psutil.Process(pid).cmdline())
    except Exception:
        return None


def _observation_backend_pids() -> list[int]:
    from .runtime.process import SPAWN_CMDLINE_MARKER

    pids: list[int] = []
    try:
        import psutil  # type: ignore

        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                argv = proc.info.get("cmdline") or []
                exe_name = Path(argv[0]).name.lower() if argv else ""
                cmdline = " ".join(argv)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if (
                exe_name.startswith("observation-agent")
                and SPAWN_CMDLINE_MARKER in cmdline
                and "uvicorn.run" in cmdline
                and "main:app" in cmdline
            ):
                pids.append(int(proc.info["pid"]))
    except Exception:
        return pids
    return pids


def _is_current_frozen_backend(pid: int) -> bool:
    if not getattr(sys, "frozen", False):
        return True
    cmdline = _process_command_line(pid)
    if not cmdline:
        return True
    current = os.path.normcase(os.path.abspath(sys.executable))
    return current in os.path.normcase(cmdline)


def _replace_stale_observation_backends(log) -> None:
    from .runtime import PidFile, is_pid_running, terminate_pid
    from .runtime.process import SPAWN_CMDLINE_MARKER

    pid_file = PidFile.default()
    record = pid_file.read()
    candidates = set(_observation_backend_pids())
    if (
        record
        and is_pid_running(record.pid, cmdline_marker=record.cmdline_marker)
    ):
        candidates.add(record.pid)

    removed_pid_file = False
    for pid in sorted(candidates):
        if not is_pid_running(pid, cmdline_marker=SPAWN_CMDLINE_MARKER):
            continue
        if _is_current_frozen_backend(pid):
            continue
        log(f"replacing observation backend pid={pid} with {sys.executable}")
        if terminate_pid(pid, cmdline_marker=SPAWN_CMDLINE_MARKER):
            if record and record.pid == pid and not removed_pid_file:
                pid_file.remove()
                removed_pid_file = True
        else:
            log(f"failed to stop observation backend pid={pid}")


def _one_click() -> None:
    from .commands import start as start_cmd
    from .installer.platform_paths import agent_cot_root, ensure_dir
    from .runtime import PidFile, is_pid_running

    log_dir = ensure_dir(agent_cot_root() / "logs")
    bootstrap_log = log_dir / "one-click.log"

    def log(line: str) -> None:
        with bootstrap_log.open("a", encoding="utf-8", errors="replace") as f:
            f.write(line.rstrip() + "\n")

    log("observation-agent one-click launcher")
    _replace_stale_observation_backends(log)

    try:
        result = start_cmd.start_backend(open_browser=True)
        log(f"started backend pid={result.pid} port={result.port}")
        if result.auto_bootstrap_agents:
            log(f"bootstrapped agents: {', '.join(result.auto_bootstrap_agents)}")
        if result.auto_bootstrap_skip_reason:
            log(f"bootstrap partial: {result.auto_bootstrap_skip_reason}")
        return
    except start_cmd.StartError as exc:
        message = str(exc)
        log(f"start warning: {message}")
        if "backend already running" not in message:
            raise

    pid_file = PidFile.default()
    record = pid_file.read()
    if record and is_pid_running(record.pid, cmdline_marker=record.cmdline_marker):
        url = f"http://127.0.0.1:{record.port}/"
        webbrowser.open(url)
        log(f"opened existing backend {url}")


def main() -> None:
    _prepare_frozen_assets()
    if len(sys.argv) >= 2 and sys.argv[1] == "-c":
        _run_dash_c()
        return
    if len(sys.argv) >= 2 and sys.argv[1].lower().endswith(".py"):
        _run_script()
        return
    if len(sys.argv) == 1:
        _one_click()
        return

    from .cli import _entrypoint

    _entrypoint()


if __name__ == "__main__":
    main()
