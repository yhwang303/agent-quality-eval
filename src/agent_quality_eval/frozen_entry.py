"""Frozen executable entry point for agent-quality-eval."""

from __future__ import annotations

import os
import runpy
import shutil
import sys
import urllib.request
import webbrowser
from pathlib import Path


def _prepare_frozen_assets() -> None:
    """Persist bundled observation assets outside PyInstaller's temp dir."""
    if not getattr(sys, "frozen", False):
        return

    from . import __version__

    source = Path(getattr(sys, "_MEIPASS", "")) / "agent_cot" / "assets"
    if not source.is_dir():
        return

    try:
        exe_mtime = Path(sys.executable).stat().st_mtime_ns
    except OSError:
        exe_mtime = 0
    target = (
        Path.home()
        / ".agent-quality-eval"
        / "frozen-assets"
        / f"{__version__}-{exe_mtime}"
        / "assets"
    )
    ready = (
        (target / "backend" / "main.py").is_file()
        and (target / "frontend-dist" / "index.html").is_file()
        and (target / "cot-extractor" / "scripts" / "extract_cot.py").is_file()
    )
    if not ready:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True)

    os.environ["AGENT_COT_ASSETS_ROOT"] = str(target)


def _refresh_frozen_runtime_state() -> None:
    """Write runtime.json for the current stable exe and bundled assets."""
    if not getattr(sys, "frozen", False):
        return
    assets_root = os.environ.get("AGENT_COT_ASSETS_ROOT")
    if not assets_root:
        return
    try:
        from . import __version__ as eval_version
        from agent_cot.installer.runtime_state import write_runtime_state

        root = Path(assets_root)
        write_runtime_state(
            cot_extractor_root=root / "cot-extractor",
            python_executable=sys.executable,
            extras={
                "agent_quality_eval_version": eval_version,
                "launcher_executable": sys.executable,
                "launcher_assets_root": str(root.resolve()),
            },
        )
    except Exception:
        pass


def _seed_import_path() -> None:
    entries: list[str] = [str(Path.cwd())]
    pythonpath = os.environ.get("PYTHONPATH", "")
    if pythonpath:
        entries.extend(p for p in pythonpath.split(os.pathsep) if p)
    for entry in reversed(entries):
        if entry and entry not in sys.path:
            sys.path.insert(0, entry)


def _run_dash_c() -> None:
    code = sys.argv[2] if len(sys.argv) > 2 else ""
    sys.argv = [sys.executable, "-c", *sys.argv[3:]]
    _seed_import_path()
    glb = {
        "__name__": "__main__",
        "__file__": "<agent-quality-eval-frozen-c>",
        "__package__": None,
    }
    exec(compile(code, "<agent-quality-eval-frozen-c>", "exec"), glb, glb)


def _run_script() -> None:
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    _seed_import_path()
    runpy.run_path(str(script), run_name="__main__")


def _run_agent_quality_eval_runner() -> None:
    runner = sys.argv[2] if len(sys.argv) > 2 else ""
    argv = sys.argv[3:]
    _seed_import_path()
    if runner == "critic":
        from .evaluation.critic import main as runner_main

        raise SystemExit(runner_main(argv))
    if runner == "live-critic":
        from .evaluation.live_critic import main as runner_main

        raise SystemExit(runner_main(argv))
    raise SystemExit(f"unknown agent-quality-eval runner: {runner}")


def _open_running_eval_workbench() -> bool:
    try:
        from agent_cot.runtime import PidFile, SPAWN_CMDLINE_MARKER, is_pid_running

        existing = PidFile.default().read()
        if existing and is_pid_running(existing.pid, cmdline_marker=SPAWN_CMDLINE_MARKER):
            webbrowser.open(f"http://127.0.0.1:{existing.port}/")
            return True
    except Exception:
        return False
    return False


def _restart_dashboard() -> bool:
    try:
        from .cli import bootstrap_observation_runtime, bootstrap_workspace
        from agent_cot.commands import start as start_cmd
        from agent_cot.commands.stop import stop_backend

        stop_backend(force=False)
        bootstrap_workspace(overwrite_config=False)
        bootstrap_observation_runtime()
        result = start_cmd.start_backend(open_browser=False)
        webbrowser.open(f"http://127.0.0.1:{result.port}/")
        return True
    except Exception:
        return False


def _stop_other_eval_processes() -> None:
    """Treat a double-clicked exe as an update: remove old trays/backends."""
    if not getattr(sys, "frozen", False):
        return
    try:
        import psutil  # type: ignore
    except Exception:
        return

    current_pid = os.getpid()
    try:
        current_proc = psutil.Process(current_pid)
        current_started = current_proc.create_time()
        protected_pids = {current_pid, *(proc.pid for proc in current_proc.parents())}
    except Exception:
        current_started = 0.0
        protected_pids = {current_pid}

    def is_runner_process(proc) -> bool:
        try:
            cmdline = [str(part) for part in (proc.cmdline() or [])]
        except Exception:
            cmdline = []
        joined = " ".join(cmdline).lower()
        return (
            "--agent-quality-eval-runner" in joined
            or "agent_quality_eval.evaluation.critic" in joined
            or "agent_quality_eval.evaluation.live_critic" in joined
        )

    victims = []
    for proc in psutil.process_iter(["pid", "name", "exe"]):
        try:
            if proc.info.get("pid") in protected_pids:
                continue
            if is_runner_process(proc):
                continue
            name = str(proc.info.get("name") or "").lower()
            exe = str(proc.info.get("exe") or "").lower()
            exe_name = Path(exe).name
            is_eval = (
                name.startswith("agent-quality-eval")
                or exe_name.startswith("agent-quality-eval")
            )
            if not is_eval:
                continue
            # If the user double-clicks the same stable exe twice, keep the
            # newest launcher and remove older trays/backends.
            if proc.create_time() <= current_started:
                victims.append(proc)
        except Exception:
            continue

    for proc in victims:
        try:
            for child in proc.children(recursive=True):
                if child.pid in protected_pids:
                    continue
                if is_runner_process(child):
                    continue
                try:
                    child.terminate()
                except Exception:
                    pass
            proc.terminate()
        except Exception:
            pass
    _, alive = psutil.wait_procs(victims, timeout=2)
    for proc in alive:
        try:
            for child in proc.children(recursive=True):
                if child.pid in protected_pids:
                    continue
                if is_runner_process(child):
                    continue
                try:
                    child.kill()
                except Exception:
                    pass
            proc.kill()
        except Exception:
            pass


def _make_tray_image():
    from PIL import Image

    try:
        from .desktop import _logo_path

        logo = _logo_path()
        if logo is not None:
            return Image.open(logo).convert("RGBA")
    except Exception:
        pass

    from PIL import ImageDraw

    img = Image.new("RGBA", (64, 64), (11, 17, 32, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((6, 6, 58, 58), radius=14, fill=(37, 99, 235, 255))
    draw.text((18, 22), "AE", fill=(255, 255, 255, 255))
    return img


def _run_tray(port: int) -> None:
    try:
        import pystray
    except Exception:
        return

    url = f"http://127.0.0.1:{port}/"

    def open_app(_icon=None, _item=None) -> None:
        webbrowser.open(url)

    def open_settings(_icon=None, _item=None) -> None:
        try:
            request = urllib.request.Request(
                url + "api/evals/ui/open-settings",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=1):
                pass
        except Exception:
            pass

    def quit_app(icon, _item=None) -> None:
        try:
            from agent_cot.commands.stop import stop_backend

            stop_backend(force=False)
        except Exception:
            pass
        icon.stop()

    try:
        image = _make_tray_image()
    except Exception:
        return

    icon = pystray.Icon(
        "agent-quality-eval",
        image,
        "Agent Observation",
        pystray.Menu(
            pystray.MenuItem("Open", open_app, default=True),
            pystray.MenuItem("Settings", open_settings),
            pystray.MenuItem("Quit", quit_app),
        ),
    )
    icon.run()


def main() -> None:
    if len(sys.argv) == 1:
        _stop_other_eval_processes()
    _prepare_frozen_assets()
    _refresh_frozen_runtime_state()
    if len(sys.argv) >= 3 and sys.argv[1] == "--agent-quality-eval-runner":
        _run_agent_quality_eval_runner()
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "-c":
        _run_dash_c()
        return
    if len(sys.argv) >= 2 and sys.argv[1].lower().endswith(".py"):
        _run_script()
        return

    if len(sys.argv) == 1:
        # Real desktop-app presentation: an immediate splash + a native
        # WebView2 window instead of a silent-then-sudden browser tab. This
        # changes presentation only — the backend, bootstrap and data flow
        # are identical. If the native window can't be created we fall through
        # to the historical browser + tray path so the app always comes up.
        if getattr(sys, "frozen", False):
            try:
                from .desktop import run_app as run_desktop_app

                if run_desktop_app():
                    return
            except Exception:
                pass

        # One-click behavior: first-run setup, then start the copied observation dashboard.
        try:
            from .cli import bootstrap_observation_runtime, bootstrap_workspace
            from agent_cot.commands import start as start_cmd
            from agent_cot.commands.stop import stop_backend

            # A packaged upgrade must not keep serving the previously spawned
            # backend/static bundle. Restart the tracked backend before opening
            # the dashboard so the new exe's assets and API code are active.
            try:
                stop_backend(force=False)
            except Exception:
                pass
            bootstrap_workspace(overwrite_config=False)
            bootstrap_observation_runtime()
            _refresh_frozen_runtime_state()
            result = start_cmd.start_backend(open_browser=False)
            webbrowser.open(f"http://127.0.0.1:{result.port}/")
            if getattr(sys, "frozen", False):
                _run_tray(result.port)
            return
        except Exception:
            if _open_running_eval_workbench():
                return
            if _restart_dashboard():
                return
            # Fall back to the CLI help if the observation runtime cannot start.
            pass

    if len(sys.argv) >= 2 and sys.argv[1] == "--open-home":
        webbrowser.open(str(Path.home() / ".agent-quality-eval"))
        return

    from .cli import _entrypoint

    _entrypoint()


if __name__ == "__main__":
    main()
