"""``agent-cot start`` — bring up the local dashboard backend.

Three flavours:

* ``--apply`` (default in CLI is "background detach"): spawn uvicorn
  detached, write PID file, wait until healthy, optionally open the
  browser, return.
* ``--foreground``: don't detach; uvicorn runs in the current
  terminal. Simpler debugging path; Ctrl-C kills it.
* ``--no-browser``: skip ``webbrowser.open``. Used by CI and by users
  who run the backend headless.

The backend itself is the existing FastAPI app at
``agent-dashboard/backend/main.py``. P3 will move it inside the wheel
and let us serve a frontend bundle from the same port; until then,
``start`` only brings up the API and points users at the dev frontend.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from .. import diag
from .._assets import (
    bundled_backend_dir,
    bundled_extractor_root,
    frontend_dist,
    has_bundled_backend,
    has_bundled_extractor,
    has_frontend_dist,
)
from ..agents.base import CursorCotError
from ..installer.config import CursorCotConfig, load_config, save_config
from ..installer.platform_paths import agent_cot_root, ensure_dir
from ..installer.port_picker import is_port_free, pick_port
# v0.19.0: write fresh runtime.json on every start so on-disk hook scripts
# (cursor + codebuddy) self-heal even after `pip install -U agent-cot`
# without `agent-cot upgrade --apply`.
from ..installer.runtime_state import write_runtime_state
from ..runtime import (
    HealthCheckError,
    PidFile,
    PidRecord,
    is_pid_running,
    spawn_backend,
    spawn_python_script,
    wait_for_backend,
)
from ..runtime.process import (
    SPAWN_CMDLINE_MARKER,
    WATCHER_CMDLINE_MARKER,
    ProcessSpawnError,
    render_argv,
)


class StartError(CursorCotError):
    """Public umbrella for any start-time error visible to the CLI."""


@dataclass
class StartResult:
    pid: int
    port: int
    backend_dir: Path
    log_path: Path
    pid_file: Path
    foreground: bool
    opened_browser: bool
    # v0.18.5: 同时拉起的 transcript_watcher daemon —— 失败不影响 backend 启动，
    # 三个字段都允许为 None / "skipped"，CLI 渲染时按状态分别提示。
    watcher_pid: int | None = None
    watcher_log_path: Path | None = None
    watcher_pid_file: Path | None = None
    watcher_skip_reason: str | None = None
    # v0.19.0: hooks self-heal report —— how many on-disk hook scripts
    # (across cursor + codebuddy) were stale and got refreshed (idempotent;
    # usually 0). None when the self-heal pass was skipped (no IDE
    # detected) or errored. ``self_heal_skip_reason`` holds the
    # combined per-IDE skip reasons separated by ``; ``.
    self_heal_scripts_refreshed: int | None = None
    self_heal_runtime_state: Path | None = None
    self_heal_skip_reason: str | None = None
    # v0.19.0: per-IDE breakdown so CLI can render "cursor: 2 stale; codebuddy: 0 stale".
    self_heal_per_ide: dict[str, int] | None = None
    # v0.20.4: Claude settings.json env.OTEL_EXPORTER_OTLP_ENDPOINT self-heal.
    # ``"updated"`` when we rewrote a stale loopback URL to match the actual
    # backend port; ``"ok"`` when nothing needed changing; ``"foreign"`` when
    # the user pointed Claude at a non-loopback endpoint (corporate collector
    # / langfuse / phoenix) — we leave that alone; ``"absent"`` when Claude
    # isn't installed or settings.json has no OTel env at all; ``None`` when
    # the self-heal pass errored. ``otel_endpoint_self_heal_detail`` holds
    # the previous vs. new URL for CLI rendering.
    otel_endpoint_self_heal: str | None = None
    otel_endpoint_self_heal_detail: str | None = None
    auto_bootstrap_agents: list[str] | None = None
    auto_bootstrap_skip_reason: str | None = None


# ---------------------------------------------------------------------------
# Backend dir resolution
# ---------------------------------------------------------------------------


def _find_backend_dir(config: CursorCotConfig) -> Path | None:
    """Locate the FastAPI backend for spawning uvicorn.

    Order of trust (v0.20.6, *reversed* from earlier releases):

    1. ``config.dashboard_repo`` if it points at a real ``backend``
       containing ``main.py`` (lets advanced users override explicitly).
    2. The wheel-bundled copy at ``agent_cot/assets/backend/``
       (production: this is what end-users get from ``pip install``).
    3. A sibling ``agent-dashboard/backend`` in the developer checkout
       (last-resort fallback while iterating on backend code).

    Why the reordering: pre-0.20.6 the sibling search ran BEFORE the
    bundled fallback, which means a maintainer with an old/stale
    ``agent-dashboard/`` git checkout next to ``agent-cot/`` would
    silently load the **old backend** instead of the freshly built
    wheel. Sub-agent folding, OTel route fixes, basically every
    code change the maintainer made to the new backend was invisible.
    The bundled copy is what ``pip install -e .`` syncs from the
    current source tree (via ``_build_assets sync``), so making it
    the trusted source restores "edit src → restart backend → see
    your change" — and end-users never had a sibling dir to hit
    anyway, so this is a no-op for them.

    Returns ``None`` only when *all three* fail, which means the
    install is broken — better to surface a clear error than to spawn
    nothing.
    """
    candidates: list[Path] = []
    if config.dashboard_repo:
        candidates.append(Path(config.dashboard_repo) / "backend")
        candidates.append(Path(config.dashboard_repo))

    for cand in candidates:
        if cand.is_dir() and (cand / "main.py").is_file():
            return cand.resolve()

    # v0.20.6: prefer bundled over sibling dev checkout. See docstring.
    if has_bundled_backend():
        return bundled_backend_dir().resolve()

    # Last resort: walk up the file system looking for a co-located dev
    # checkout. Only contributors hacking on agent-dashboard hit this path
    # now; end-users have already returned with the bundled copy above.
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        cand = parent / "agent-dashboard" / "backend"
        if cand.is_dir() and (cand / "main.py").is_file():
            return cand.resolve()
    return None


def _build_backend_env() -> dict[str, str]:
    """Compute extra env vars to pass into the spawned backend.

    Variables we set (kept in lock-step with ``cot-bridge.js`` and
    ``extract_cot.py`` so the whole capture → store → display chain
    points at the same on-disk root):

    * ``AGENT_COT_FRONTEND_DIST`` — directory of the bundled SPA.
      Only set when the bundle actually contains an ``index.html``,
      so dev users running with a live ``vite`` can avoid serving a
      stale prebuilt SPA.
    * ``AGENT_COT_EXTRACTOR_SRC`` — for the OTLP exporter import,
      reusing the value already detected during ``init`` if any.
    * ``AGENT_COT_DATA_ROOT`` (v0.18.2) — single source of truth for
      where ``cot.json`` / reports / transcripts live. backend/config.py
      reads this; cot-bridge.js spawns extract_cot.py with it set so
      the hook writes to the same place the backend scans. Without this,
      wheel-installed users hit "SESSIONS 0" forever (data lands in
      site-packages while backend scans ``~/.agent-cot/data/``).
    * ``COT_EXTRACTOR_ROOT`` (v0.18.2) — fallback path used by hook
      scripts to locate ``scripts/extract_cot.py``. Set here so a
      ``agent-cot start`` *without* a prior ``agent-cot init`` still
      gives subsequent hook spawns a valid path.
    """
    env: dict[str, str] = {}
    if getattr(sys, "frozen", False):
        env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    if has_frontend_dist():
        env["AGENT_COT_FRONTEND_DIST"] = str(frontend_dist().resolve())
    config = load_config()
    if config.cot_extractor_repo:
        repo = Path(config.cot_extractor_repo)
        src = repo / "src"
        if src.is_dir():
            env["AGENT_COT_EXTRACTOR_SRC"] = str(src.resolve())

    # v0.18.2: 跨进程数据根 + extractor 入口路径兜底注入
    data_root = ensure_dir(agent_cot_root() / "data")
    env["AGENT_COT_DATA_ROOT"] = str(data_root)
    # backend 历史代码用 COT_DIR 直接指 cot 子目录，保留同步注入避免任何
    # 老路径计算逻辑漏到这里（双保险）。
    env.setdefault("COT_DIR", str(ensure_dir(data_root / "cot")))
    # v0.20.0: backend 也写 pipeline.log；从 CLI 显式指定路径，确保
    # 跟 hook / extractor / CLI 写到同一个文件。
    env.setdefault("AGENT_COT_PIPELINE_LOG", str(diag.log_path()))

    extractor_root: Path | None = None
    if config.cot_extractor_repo:
        candidate = Path(config.cot_extractor_repo)
        if (candidate / "scripts" / "extract_cot.py").is_file():
            extractor_root = candidate.resolve()
    if extractor_root is None and has_bundled_extractor():
        extractor_root = bundled_extractor_root().resolve()
    if extractor_root is not None:
        env.setdefault("COT_EXTRACTOR_ROOT", str(extractor_root))

    return env


# ---------------------------------------------------------------------------
# v0.19.0: self-heal on-disk hook scripts + runtime.json
# ---------------------------------------------------------------------------


def _self_heal_hooks() -> tuple[
    int | None, Path | None, str | None, dict[str, int]
]:
    """Refresh on-disk hook scripts for every detected IDE + write ``runtime.json``.

    The three killer failure modes this plugs:

    1. User did ``pip install -U agent-cot`` then ``agent-cot start``
       *without* re-running ``agent-cot upgrade --apply``. The on-disk
       ``~/.cursor/hooks/cot-bridge.js`` still holds the **previous**
       wheel's patched ``COT_ROOT``. If that path no longer resolves
       (uninstall / venv recreation / Python upgrade), every Cursor
       stop event silently no-ops and the dashboard goes stale.
    2. Same problem variant for codebuddy: ``cot-stream-codebuddy.js``
       in ``~/.codebuddy/hooks/`` was copied from a previous wheel; the
       hook itself is literal-free (no ``COT_ROOT`` patch needed) but
       its source code may have improved since (e.g. v0.19.0 added the
       ``runtime.json`` fallback). Re-copying the bytes from the
       currently-installed wheel is harmless and idempotent.
    3. **v0.19.1 new**: same pattern for Claude Internal —
       ``~/.claude/hooks/claude_stream_hook.py`` is the Python hook that
       both writes events.jsonl AND backgrounds extract_cot.py on Stop /
       SubagentStop / SessionEnd / StopFailure. Older wheels didn't have
       the extract trigger, so re-copying the bundled bytes is what gets
       Claude session traces flowing into the dashboard. NOT patching
       it means the user does NOT need to re-run init — pure self-heal.

    All fixed by re-running ``upgrade --apply``-equivalent logic here,
    then dropping a fresh ``runtime.json`` so hook scripts can self-heal
    even mid-session.

    Iteration order: ``cursor`` → ``codebuddy`` → ``claude`` (so the
    CLI lines appear in the same order each time). Each adapter is
    consulted via ``detect_installed()`` — if the IDE isn't on this box,
    we skip its self-heal silently (no IDE = nothing to refresh).

    Returns ``(total_refreshed, runtime_path, skip_reason, per_ide_counts)``.
    Failures are *non-fatal* — backend still comes up. ``skip_reason``
    is surfaced via :class:`StartResult` so CLI can print a one-liner.
    """
    from . import upgrade as _upgrade
    from ..agents import AgentNotImplementedError, get_adapter

    total_refreshed = 0
    runtime_path: Path | None = None
    skip_reasons: list[str] = []
    per_ide: dict[str, int] = {}

    for agent_name in ("cursor", "codebuddy", "claude", "codex"):
        try:
            adapter = get_adapter(agent_name)
        except AgentNotImplementedError:
            continue
        except Exception as exc:
            skip_reasons.append(f"{agent_name}: adapter unavailable ({exc})")
            continue

        try:
            installed = adapter.detect_installed()
        except Exception as exc:
            skip_reasons.append(f"{agent_name}: detect failed ({exc})")
            continue
        if not installed:
            # IDE not present on this machine — nothing to self-heal.
            continue

        try:
            ctx = _upgrade.build_context(agent_name=agent_name)
        except AgentNotImplementedError:
            continue
        except Exception as exc:
            skip_reasons.append(
                f"{agent_name}: upgrade plan unavailable "
                f"({type(exc).__name__}: {exc})"
            )
            continue

        try:
            res = _upgrade.apply(ctx)
            count = len(res.scripts_replaced) + len(res.scripts_installed)
            per_ide[agent_name] = count
            total_refreshed += count
        except Exception as exc:
            skip_reasons.append(f"{agent_name}: refresh failed ({exc})")

    # Always (try to) write runtime.json — even when no IDE detected,
    # so a future install / IDE switch benefits without a re-init.
    try:
        runtime_path = write_runtime_state()
    except Exception as exc:
        if not skip_reasons:
            skip_reasons.append(f"runtime.json write failed: {exc}")

    skip_reason = "; ".join(skip_reasons) if skip_reasons else None
    refreshed_total = total_refreshed if per_ide else None
    return (refreshed_total, runtime_path, skip_reason, per_ide)


def _auto_bootstrap_installed_agents(*, backend_port: int) -> tuple[list[str], str | None]:
    """Register hooks for detected agents during ``start``.

    ``_self_heal_hooks`` refreshes script bytes for already-initialized users,
    but a fresh one-click install also needs ``hooks.json`` / settings entries
    created.  This path is intentionally port-aware: Claude OTel env, when
    applicable, is written with the backend port that ``start`` just selected.

    Codex is hook-only here.  We do not write ``~/.codex/config.toml`` from the
    product launcher because invalid Codex TOML can break the desktop client;
    the Codex adapter currently exposes no ``recommended_env`` hook, so
    ``init.apply_plan`` only touches ``hooks.json`` and hook scripts.
    """
    from . import init as _init
    from ..agents import AgentNotImplementedError, get_adapter, list_agents

    applied: list[str] = []
    skip_reasons: list[str] = []

    for agent_name in list_agents():
        try:
            adapter = get_adapter(agent_name)
        except Exception as exc:
            skip_reasons.append(f"{agent_name}: adapter unavailable ({exc})")
            continue

        try:
            installed = adapter.detect_installed()
        except Exception as exc:
            skip_reasons.append(f"{agent_name}: detect failed ({exc})")
            continue
        if not installed:
            continue

        try:
            plan = _init.build_plan(
                agent_name=agent_name,
                port_backend=backend_port,
                write_otel_env=True,
            )
        except AgentNotImplementedError:
            continue
        except Exception as exc:
            skip_reasons.append(
                f"{agent_name}: bootstrap plan failed "
                f"({type(exc).__name__}: {exc})"
            )
            continue

        try:
            _init.apply_plan(plan, force=True)
            applied.append(agent_name)
        except Exception as exc:
            skip_reasons.append(
                f"{agent_name}: bootstrap apply failed "
                f"({type(exc).__name__}: {exc})"
            )

    return applied, "; ".join(skip_reasons) if skip_reasons else None


# ---------------------------------------------------------------------------
# v0.20.4: Claude OTel endpoint self-heal
# ---------------------------------------------------------------------------


# Loopback host fragments we consider "ours" — anything else (foo.corp.io,
# us.cloud.langfuse.com, my-phoenix:6006) is treated as a user-managed
# endpoint and left untouched. The user explicitly set those, agent-cot
# never overwrites them.
_LOOPBACK_HOSTS: frozenset[str] = frozenset(
    {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
)


def _parse_loopback_url(url: str) -> tuple[str, int] | None:
    """Return ``(host, port)`` iff ``url`` looks like ``http(s)://<loopback>:<port>``.

    Returns ``None`` for everything else (DNS name, IPv6 with brackets,
    schemeless strings, or a malformed URL) — those signal "user-managed
    endpoint", and the self-heal pass must NOT touch them.
    """
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url.strip())
    except Exception:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    host = (parsed.hostname or "").lower()
    if host not in _LOOPBACK_HOSTS:
        return None
    if parsed.port is None:
        return None
    return host, parsed.port


def _self_heal_claude_otel_endpoint(
    *,
    backend_port: int,
) -> tuple[str, str | None]:
    """Make sure ``~/.claude/settings.json`` env points at the live backend port.

    Claude Code reads ``settings.json`` once at process start and caches
    every env var into the spawned Claude process. If ``init --apply`` ran
    while port 8765 was free but ``start`` ended up on 8766 (because 8765
    got occupied between runs), the cached endpoint inside Claude would
    push OTel data to whatever process now holds 8765 — i.e. nowhere
    useful. This self-heal pass closes that gap so the documented
    "one-click" promise survives port-eviction.

    Behaviour matrix:

    ====================================  ==============  ============================
    settings.json shape                    Action          Return status
    ====================================  ==============  ============================
    ``~/.claude/`` missing                 nothing         ``"absent"``
    no ``settings.json``                   nothing         ``"absent"``
    invalid JSON                           nothing         ``"absent"``
    no ``env`` block or no endpoint key    nothing         ``"absent"``
    endpoint is non-loopback (corp/cloud)  nothing         ``"foreign"``
    endpoint already matches               nothing         ``"ok"``
    endpoint loopback but wrong port       rewrite atomic  ``"updated"``
    ====================================  ==============  ============================

    Returns ``(status, detail)`` where ``detail`` is a human-readable
    fragment like ``"http://127.0.0.1:8765 -> http://127.0.0.1:8766"``
    when we actually wrote, else ``None``. Any I/O error is swallowed
    and reported as ``("error", "<message>")`` — a failed self-heal
    must NEVER block backend startup.
    """
    import json

    from ..agents.claude import ClaudeAdapter

    key = ClaudeAdapter.OTEL_ENDPOINT_KEY
    settings_paths = [
        Path.home() / ".claude" / "settings.json",
        Path.home() / ".claude-internal" / "settings.json",
        Path.home() / ".claude-inertnal" / "settings.json",
    ]
    statuses: list[str] = []
    details: list[str] = []

    for settings_path in settings_paths:
        if not settings_path.is_file():
            continue
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8") or "{}")
        except (OSError, ValueError):
            statuses.append("absent")
            continue
        if not isinstance(settings, dict):
            statuses.append("absent")
            continue
        env_block = settings.get("env")
        if not isinstance(env_block, dict):
            statuses.append("absent")
            continue
        current = env_block.get(key)
        if not isinstance(current, str) or not current.strip():
            statuses.append("absent")
            continue
        parsed = _parse_loopback_url(current)
        if parsed is None:
            statuses.append("foreign")
            continue
        _host, current_port = parsed
        if current_port == backend_port:
            statuses.append("ok")
            continue

        new_url = ClaudeAdapter.otel_endpoint_for_port(backend_port)
        env_block[key] = new_url
        try:
            payload = json.dumps(settings, indent=2, ensure_ascii=False)
            tmp = settings_path.with_suffix(settings_path.suffix + ".tmp")
            tmp.write_text(payload + "\n", encoding="utf-8")
            tmp.replace(settings_path)
            statuses.append("updated")
            details.append(f"{settings_path}: {current} -> {new_url}")
        except OSError as exc:
            statuses.append("error")
            details.append(f"could not rewrite {settings_path}: {exc}")

    if not statuses:
        return ("absent", None)
    if "error" in statuses:
        return ("error", "; ".join(details) if details else None)
    if "updated" in statuses:
        return ("updated", "; ".join(details) if details else None)
    if "foreign" in statuses and all(s == "foreign" for s in statuses):
        return ("foreign", None)
    if "ok" in statuses:
        return ("ok", None)
    return ("absent", None)


# ---------------------------------------------------------------------------
# v0.18.5: transcript_watcher daemon 拉起 / 关闭
# ---------------------------------------------------------------------------


def _resolve_watcher_script() -> Path | None:
    """Locate ``transcript_watcher.py`` for spawning by the daemon launcher.

    Same precedence as ``_build_backend_env`` extractor lookup:
    1. ``cot_extractor_repo`` from config (advanced users with a source checkout)
    2. wheel-bundled ``assets/cot-extractor/scripts/transcript_watcher.py``

    Returns ``None`` if neither carries the script (broken install or a
    very old wheel) — caller must skip the spawn and surface a reason.
    """
    config = load_config()
    if config.cot_extractor_repo:
        cand = Path(config.cot_extractor_repo) / "scripts" / "transcript_watcher.py"
        if cand.is_file():
            return cand.resolve()
    if has_bundled_extractor():
        cand = bundled_extractor_root() / "scripts" / "transcript_watcher.py"
        if cand.is_file():
            return cand.resolve()
    return None


def start_watcher(
    *,
    backend_env: dict[str, str],
    health_timeout: float = 5.0,
) -> tuple[int | None, Path | None, Path | None, str | None]:
    """Spawn the gap-tool watcher daemon. Returns ``(pid, log_path, pid_file, skip_reason)``.

    Failures here are *non-fatal* for ``agent-cot start``: the dashboard
    can still display whatever cot-stream.js manages to capture (Read /
    Edit / Shell / MCP), just without the deterministic Glob / Grep /
    WebFetch / SemanticSearch / TodoWrite enrichment. We surface the
    reason so ``status`` / docs can point users at the right knob.
    """
    pid_file = PidFile.for_watcher()

    # Refuse to silently double-spawn (consistent with backend behaviour).
    existing = pid_file.read()
    if existing is not None and is_pid_running(
        existing.pid, cmdline_marker=existing.cmdline_marker
    ):
        # Already healthy; just return its details so CLI can echo them.
        return (
            existing.pid,
            Path(existing.log_path) if existing.log_path else None,
            pid_file.path,
            None,
        )

    script = _resolve_watcher_script()
    if script is None:
        return (
            None,
            None,
            pid_file.path,
            "transcript_watcher.py not found in cot-extractor (wheel "
            "is missing scripts bundle — pip install --upgrade agent-cot).",
        )

    log_path = ensure_dir(agent_cot_root() / "logs") / "watcher.log"

    # The watcher reads AGENT_COT_DATA_ROOT to decide where events.jsonl
    # lives. ``backend_env`` already carries it, but we copy explicitly so
    # the dependency is obvious to anyone reading this file in 6 months.
    extra_env: dict[str, str] = {}
    for k in ("AGENT_COT_DATA_ROOT", "COT_EXTRACTOR_ROOT", "COT_DIR"):
        if k in backend_env:
            extra_env[k] = backend_env[k]

    try:
        pid, _argv = spawn_python_script(
            script_path=script,
            script_args=[],
            log_path=log_path,
            cmdline_marker=WATCHER_CMDLINE_MARKER,
            cwd=script.parent,
            extra_env=extra_env,
        )
    except ProcessSpawnError as exc:
        return (None, log_path, pid_file.path, f"spawn failed: {exc}")

    record = PidRecord(
        pid=pid,
        port=0,  # watcher has no HTTP port; field kept for schema parity
        started_at=time.time(),
        log_path=str(log_path),
        cmdline_marker=WATCHER_CMDLINE_MARKER,
    )
    pid_file.write(record)

    # Cheap liveness check: poll PID for a beat before declaring success.
    # Avoids false-positive "running" when the script crashes immediately
    # (e.g. import error). Keep it short — health_timeout default 5s is
    # plenty for "process started and didn't immediately die".
    deadline = time.time() + max(0.5, min(health_timeout, 5.0))
    while time.time() < deadline:
        if is_pid_running(pid, cmdline_marker=WATCHER_CMDLINE_MARKER):
            time.sleep(0.2)
            if is_pid_running(pid, cmdline_marker=WATCHER_CMDLINE_MARKER):
                return (pid, log_path, pid_file.path, None)
        time.sleep(0.1)

    # Did not stay alive. Clean up PID file so next start can retry cleanly.
    pid_file.remove()
    return (None, log_path, pid_file.path,
            f"daemon exited within {health_timeout:.1f}s — see {log_path}")


# ---------------------------------------------------------------------------
# Foreground (non-detached) execution
# ---------------------------------------------------------------------------


def _run_foreground(
    *,
    port: int,
    backend_dir: Path,
    host: str = "127.0.0.1",
    extra_env: dict[str, str] | None = None,
) -> int:
    """Run uvicorn synchronously in this terminal until Ctrl-C.

    Returns the child's exit code. We do not write a PID file in this
    mode — there's nothing to ``stop`` later because the user owns
    the process directly.
    """
    snippet = (
        f"# {SPAWN_CMDLINE_MARKER}\n"
        "import uvicorn\n"
        f"uvicorn.run('main:app', host={host!r}, port={int(port)}, reload=False)\n"
    )
    argv = [sys.executable, "-c", snippet]
    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{backend_dir}{os.pathsep}{pp}" if pp else str(backend_dir)
    )
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)

    try:
        result = subprocess.run(
            argv,
            cwd=str(backend_dir),
            env=env,
            check=False,
        )
        return result.returncode
    except KeyboardInterrupt:
        # uvicorn handles SIGINT itself; we just unwind cleanly.
        return signal.SIGINT


# ---------------------------------------------------------------------------
# Background detached execution (the default)
# ---------------------------------------------------------------------------


def start_backend(
    *,
    port_override: int | None = None,
    foreground: bool = False,
    open_browser: bool = True,
    health_timeout: float = 15.0,
) -> StartResult:
    """High-level orchestration used by both the CLI and tests.

    The function is split out from the click command so unit tests
    can drive it directly, mocking spawn/health if needed.
    """
    config = load_config()

    diag.banner("cli.start")

    # v0.19.0: self-heal first so the hook scripts on disk match the
    # currently-installed wheel before we even bring up the backend.
    # Idempotent — a no-op when nothing changed since the last invocation.
    # Covers BOTH cursor (cot-bridge.js / cot-stream.js literal patches)
    # AND codebuddy (cot-stream-codebuddy.js byte-level refresh).
    self_heal_count, self_heal_runtime, self_heal_skip, self_heal_per_ide = (
        _self_heal_hooks()
    )

    backend_dir = _find_backend_dir(config)
    if backend_dir is None:
        raise StartError(
            "could not locate the dashboard backend (agent-dashboard/backend "
            "with main.py). Run `agent-cot init` first or set "
            "dashboard_repo in ~/.agent-cot/config.toml."
        )

    # Decide on a port. Priority:
    #   --port override  >  config.backend_port (if still free)  >  auto-pick
    #
    # v0.20.8: 加占用复查。0.20.7 之前如果 ``config.backend_port`` 已经被
    # 别的进程占用（例：同事第二天开机时 8765 被 Jupyter / Vite / 公司其他
    # dashboard 抢了），start 会无视占用直接尝试 bind 那个端口，uvicorn 立
    # 即崩，backend 起不来，前端访问不到 —— 还不会自动 fallback 到
    # ``pick_port`` 的 8800-8900 备用段。现在加一道 ``is_port_free`` 复查：
    # config 写的端口被占就当它不存在，照样走 ``pick_port`` 重新挑一个空的。
    pid_file = PidFile.default()
    existing = pid_file.read()
    existing_running = (
        existing is not None
        and is_pid_running(existing.pid, cmdline_marker=existing.cmdline_marker)
    )

    if existing_running and existing is not None:
        port = existing.port
    elif port_override is not None:
        port = port_override
    elif config.backend_port and is_port_free(config.backend_port):
        port = config.backend_port
    else:
        port = pick_port(prefer=8765)

    auto_bootstrap_agents, auto_bootstrap_skip = _auto_bootstrap_installed_agents(
        backend_port=port,
    )

    # Refuse to silently double-spawn.
    if existing_running and existing is not None:
        # The dashboard is already alive, but a first-time one-click launch may
        # still have needed the hook registration above.  Keep the historical
        # CLI behaviour (surface "already running") after the best-effort
        # bootstrap has had a chance to run against the live port.
        raise StartError(
            f"backend already running (pid={existing.pid}, "
            f"port={existing.port}). Run `agent-cot stop` first."
        )

    if foreground:
        # No PID file, no health check, no browser open — just run.
        rc = _run_foreground(
            port=port,
            backend_dir=backend_dir,
            extra_env=_build_backend_env(),
        )
        raise SystemExit(rc)

    log_path = ensure_dir(agent_cot_root() / "logs") / "backend.log"
    backend_env = _build_backend_env()
    pid, _argv = spawn_backend(
        port=port,
        backend_dir=backend_dir,
        log_path=log_path,
        extra_env=backend_env,
    )

    record = PidRecord(
        pid=pid,
        port=port,
        started_at=time.time(),
        log_path=str(log_path),
        cmdline_marker=SPAWN_CMDLINE_MARKER,
    )
    pid_file.write(record)

    # Persist the chosen port so the next `start` is deterministic.
    config.backend_port = port
    save_config(config)

    # v0.20.4: keep Claude's settings.json OTel endpoint in sync with the
    # port we just locked in. NEVER touches non-loopback URLs (corp /
    # cloud collectors set by the user). Errors are swallowed —— self-heal
    # is best-effort and must not gate dashboard startup.
    otel_endpoint_status, otel_endpoint_detail = _self_heal_claude_otel_endpoint(
        backend_port=port,
    )

    try:
        wait_for_backend(port, timeout=health_timeout)
    except HealthCheckError as exc:
        # Don't kill the process — it might just be slow. But surface
        # the health failure so the user knows.
        raise StartError(
            f"{exc}\nLog: {log_path}\nPID: {pid} (kept alive; run "
            f"`agent-cot stop` once you've inspected the log)"
        ) from exc

    # v0.18.5: 拉起 transcript_watcher daemon —— 失败不影响 backend 启动，
    # 只在 StartResult 上挂一条 skip_reason 让 CLI 提示用户。
    watcher_pid, watcher_log, watcher_pid_file, watcher_skip = start_watcher(
        backend_env=backend_env,
        health_timeout=5.0,
    )

    opened = False
    if open_browser:
        # When the SPA bundle is present, take the user straight to the
        # dashboard root. Otherwise fall back to /api/sessions so they
        # at least see live data (and discover something is wrong).
        landing = "/" if has_frontend_dist() else "/api/sessions"
        try:
            webbrowser.open(f"http://127.0.0.1:{port}{landing}")
            opened = True
        except (webbrowser.Error, OSError):
            opened = False

    return StartResult(
        pid=pid,
        port=port,
        backend_dir=backend_dir,
        log_path=log_path,
        pid_file=pid_file.path,
        foreground=False,
        opened_browser=opened,
        watcher_pid=watcher_pid,
        watcher_log_path=watcher_log,
        watcher_pid_file=watcher_pid_file,
        watcher_skip_reason=watcher_skip,
        self_heal_scripts_refreshed=self_heal_count,
        self_heal_runtime_state=self_heal_runtime,
        self_heal_skip_reason=self_heal_skip,
        self_heal_per_ide=self_heal_per_ide if self_heal_per_ide else None,
        otel_endpoint_self_heal=otel_endpoint_status,
        otel_endpoint_self_heal_detail=otel_endpoint_detail,
        auto_bootstrap_agents=auto_bootstrap_agents or None,
        auto_bootstrap_skip_reason=auto_bootstrap_skip,
    )


# Internal hook so tests can substitute a fake "argv preview" without
# actually launching anything.
__all__ = [
    "StartError",
    "StartResult",
    "render_argv",
    "start_backend",
    "start_watcher",
]
