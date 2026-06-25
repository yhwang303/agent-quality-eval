"""Individual diagnostic checks.

Each check is a function returning a single :class:`Check`. They are
intentionally *flat* (no inheritance, no plug-in registry) — the order
matters for human readability, and a flat list makes that order
obvious in :mod:`runner`.

Design rules:

* **Read-only.** No filesystem writes, no env mutations.
* **Cheap.** No HTTP roundtrips, no subprocess spawns. Doctor must
  finish in well under a second on a healthy machine.
* **Resilient.** A check that itself crashes is reported as ``fail``
  with the exception text; never escapes.
"""

from __future__ import annotations

import importlib
import os
import socket
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .. import __version__
from .._assets import (
    bundled_backend_dir,
    bundled_extractor_root,
    frontend_dist,
    has_bundled_backend,
    has_bundled_extractor,
    has_frontend_dist,
)
from ..agents import (
    AgentNotImplementedError,
    UnknownAgentError,
    get_adapter,
    list_agents,
)
from ..installer.config import config_path, load_config
from ..installer.platform_paths import agent_cot_root, cursor_root
from ..installer.runtime_state import read_runtime_state, runtime_state_path
from ..runtime import PidFile, is_pid_running


class CheckStatus(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class Check:
    """The atomic unit of a doctor report."""

    name: str
    status: CheckStatus
    message: str = ""
    hint: str | None = None
    """Free-form 'how to fix this' text. Surfaced only on warn / fail."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Machine-readable side-channel for ``--json``."""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


# ---------------------------------------------------------------------------
# Internal helpers (not exported)
# ---------------------------------------------------------------------------


def _safe(fn):
    """Decorate a check fn so its own crashes turn into FAIL Checks
    instead of aborting the whole doctor pass."""

    def wrapper(*args, **kwargs) -> Check:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            return Check(
                name=getattr(fn, "__name__", "check").replace("check_", ""),
                status=CheckStatus.FAIL,
                message=f"check raised {type(exc).__name__}: {exc}",
                hint=(
                    "this is a bug in agent-cot; please open an issue with "
                    "the doctor --verbose --json output."
                ),
            )

    wrapper.__name__ = getattr(fn, "__name__", "check")
    return wrapper


def _can_import(module: str) -> tuple[bool, str | None]:
    """Return ``(ok, version_or_error)``.

    Version lookup prefers ``importlib.metadata`` (the modern,
    deprecation-free path; click 8.2 already removed ``__version__``)
    and falls back to the module attribute for things that don't ship
    standard package metadata (e.g. namespaced submodules).
    """
    try:
        importlib.import_module(module)
    except ImportError as exc:
        return (False, str(exc))

    from importlib.metadata import PackageNotFoundError, version

    # ``opentelemetry.sdk`` etc. live under a parent distribution.
    candidates = [module, module.split(".")[0]]
    for cand in candidates:
        try:
            return (True, version(cand))
        except PackageNotFoundError:
            continue

    mod = importlib.import_module(module)
    return (True, getattr(mod, "__version__", None))


def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Cheap probe: try to bind. We don't care who's holding it,
    only whether *we* could pick it for a fresh ``agent-cot start``.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((host, port))
    except OSError:
        return False
    finally:
        s.close()
    return True


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@_safe
def check_python_version() -> Check:
    major, minor = sys.version_info[:2]
    cur = f"{major}.{minor}.{sys.version_info[2]}"
    if (major, minor) < (3, 10):
        return Check(
            name="python.version",
            status=CheckStatus.FAIL,
            message=f"running on Python {cur}; agent-cot requires >= 3.10",
            hint="install Python 3.10+ and re-create your venv.",
        )
    return Check(
        name="python.version",
        status=CheckStatus.OK,
        message=f"Python {cur}",
    )


@_safe
def check_agent_cot_version() -> Check:
    return Check(
        name="agent-cot.version",
        status=CheckStatus.OK,
        message=f"agent-cot {__version__}",
    )


@_safe
def check_cursor_renderer_logs() -> Check:
    """The 'rich trace' check: does ``cot_otel_enricher.py`` have access
    to Cursor's renderer.log on this machine?

    Background: the per-turn ``actual_token_usage`` / ``actual_cost_usd``
    and ``model_timeline`` that drive the dashboard's right-panel OTel
    view come from Cursor's renderer.log, NOT from the agent's transcript
    (Cursor's transcript schema doesn't carry token counts per turn).
    The extractor probes:
      Windows: %APPDATA%/Cursor/logs/
      macOS:   ~/Library/Application Support/Cursor/logs/
      Linux:   ~/.config/Cursor/logs/
      Override: CURSOR_LOGS_ROOT env var

    When colleagues report 'my trace is missing tokens / says model
    unknown / cost is 0', this is the first thing to check. The
    extractor does NOT fail on missing renderer.log; it just falls back
    to ``model_source = events_jsonl`` or ``transcript``, which is
    materially leaner than what the user (whose .cursor/logs is
    populated) sees.
    """
    import os as _os
    override = (_os.environ.get("CURSOR_LOGS_ROOT") or "").strip()
    if override:
        if _os.path.isdir(override):
            return Check(
                name="cursor.renderer_logs",
                status=CheckStatus.OK,
                message=f"CURSOR_LOGS_ROOT override -> {override}",
            )
        return Check(
            name="cursor.renderer_logs",
            status=CheckStatus.FAIL,
            message=f"CURSOR_LOGS_ROOT points to a non-existent dir: {override}",
            hint="Unset CURSOR_LOGS_ROOT or point it at a real Cursor logs dir.",
        )
    candidates: list[str] = []
    if _os.name == "nt":
        appdata = _os.environ.get("APPDATA") or ""
        if appdata:
            candidates.append(_os.path.join(appdata, "Cursor", "logs"))
    else:
        home = _os.path.expanduser("~")
        candidates.extend([
            _os.path.join(home, "Library", "Application Support", "Cursor", "logs"),
            _os.path.join(home, ".config", "Cursor", "logs"),
        ])
    found = next((c for c in candidates if _os.path.isdir(c)), None)
    if not found:
        return Check(
            name="cursor.renderer_logs",
            status=CheckStatus.WARN,
            message=(
                "Cursor renderer.log dir not found; cot.json will fall "
                "back to transcript-only model/token data (leaner than "
                "the developer's reference)."
            ),
            hint=(
                "Open Cursor once so it creates its logs dir, or set "
                "CURSOR_LOGS_ROOT to point at Cursor's logs folder. "
                "Looked at: " + ", ".join(candidates)
            ),
            extra={"candidates": candidates},
        )
    # try to enumerate at least one renderer.log inside it
    try:
        run_dirs = [d for d in _os.listdir(found) if _os.path.isdir(_os.path.join(found, d))]
    except OSError:
        run_dirs = []
    renderer_count = 0
    for run in run_dirs[:20]:
        rp = _os.path.join(found, run)
        try:
            for sub in _os.listdir(rp):
                if sub.startswith("window") and _os.path.isfile(
                    _os.path.join(rp, sub, "renderer.log")
                ):
                    renderer_count += 1
        except OSError:
            continue
    if renderer_count == 0:
        return Check(
            name="cursor.renderer_logs",
            status=CheckStatus.WARN,
            message=(
                f"Cursor logs dir exists at {found} but no "
                f"window*/renderer.log files inside. The dashboard "
                f"will still work but per-turn model/token data may be "
                f"missing for older sessions."
            ),
            hint="Use Cursor at least once with this Python env; renderer.log appears as Cursor writes to it.",
        )
    return Check(
        name="cursor.renderer_logs",
        status=CheckStatus.OK,
        message=f"{renderer_count} renderer.log files found under {found}",
    )


@_safe
def check_distribution_conflict() -> Check:
    """Detect the 0.19.x -> 0.20.x upgrade footgun.

    Pip lets ``observation-agent`` and ``Agent-CoT`` co-exist as separate
    distributions, but both ship the same on-disk ``agent_cot/`` package
    (161 shared files). The colleague-side timeline is:

      pip install Agent-CoT==0.19.6      # 162 files owned by Agent-CoT
      pip install observation-agent==0.20.0  # overwrites 161, adds 2
        -> import agent_cot resolves to 0.20.0 code
        -> ``pip list`` shows BOTH packages
      pip uninstall Agent-CoT            # nukes the 161 shared files
        -> observation-agent metadata stays, but agent_cot/__init__.py
           is gone, so ``import agent_cot`` half-works and crashes
           with ``no attribute __version__`` at random places.

    Catching this at ``agent-cot doctor`` time gives the colleague a
    loud, explicit instruction (uninstall the OLD one FIRST, then
    install the new one) instead of a silent half-broken venv.
    """
    try:
        import importlib.metadata as md
    except ImportError:
        return Check(
            name="install.distribution_conflict",
            status=CheckStatus.SKIP,
            message="importlib.metadata not available",
        )
    # PEP 503 canonical form for dedup: lowercase + ``[-_.]+`` → ``-``.
    # Without this, ``distribution("Agent-CoT")`` and
    # ``distribution("agent-cot")`` resolve to the same dist but get
    # reported twice in the error message — looks like a doctor bug.
    def _canon(name: str) -> str:
        import re as _re
        return _re.sub(r"[-_.]+", "-", name).lower()

    legacy_names = ("Agent-CoT", "cursor-cot-observer")
    new_names = ("observation-agent",)
    present_legacy: dict[str, tuple[str, str]] = {}
    present_new: dict[str, tuple[str, str]] = {}
    for name in legacy_names:
        try:
            d = md.distribution(name)
            key = _canon(d.metadata["Name"])
            present_legacy.setdefault(key, (d.metadata["Name"], d.version))
        except md.PackageNotFoundError:
            continue
    for name in new_names:
        try:
            d = md.distribution(name)
            key = _canon(d.metadata["Name"])
            present_new.setdefault(key, (d.metadata["Name"], d.version))
        except md.PackageNotFoundError:
            continue
    if present_legacy and present_new:
        legacy_str = ", ".join(f"{n}=={v}" for n, v in present_legacy.values())
        new_str = ", ".join(f"{n}=={v}" for n, v in present_new.values())
        return Check(
            name="install.distribution_conflict",
            status=CheckStatus.FAIL,
            message=(
                f"BOTH legacy and new distributions installed: "
                f"{legacy_str}  AND  {new_str}. "
                f"This will break on the next uninstall of either side."
            ),
            hint=(
                "Run in order: 1) pip uninstall -y "
                + " ".join(n for n, _ in present_legacy.values())
                + "   2) pip install --force-reinstall "
                "observation-agent  (re-resolves the shared files)."
            ),
            extra={
                "legacy": [{"name": n, "version": v} for n, v in present_legacy.values()],
                "new": [{"name": n, "version": v} for n, v in present_new.values()],
            },
        )
    return Check(
        name="install.distribution_conflict",
        status=CheckStatus.OK,
        message="no legacy/new distribution conflict",
    )


@_safe
def check_required_dep(module: str, hint: str) -> Check:
    ok, ver = _can_import(module)
    if ok:
        return Check(
            name=f"deps.{module}",
            status=CheckStatus.OK,
            message=f"{module} {ver or '(version unknown)'}",
        )
    return Check(
        name=f"deps.{module}",
        status=CheckStatus.FAIL,
        message=f"required module '{module}' not importable",
        hint=hint,
        extra={"error": ver},
    )


@_safe
def check_optional_dep(module: str, install_extra: str) -> Check:
    """Probe an optional extra. Missing → warn, not fail."""
    ok, ver = _can_import(module)
    if ok:
        return Check(
            name=f"deps.{module}",
            status=CheckStatus.OK,
            message=f"{module} {ver or '(version unknown)'}",
        )
    return Check(
        name=f"deps.{module}",
        status=CheckStatus.WARN,
        message=f"optional module '{module}' not installed",
        hint=f"install with `pip install 'agent-cot[{install_extra}]'`",
        extra={"error": ver},
    )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@_safe
def check_agent_cot_root() -> Check:
    root = agent_cot_root()
    if not root.exists():
        return Check(
            name="paths.agent_cot_root",
            status=CheckStatus.WARN,
            message=f"data root does not exist yet: {root}",
            hint="run `agent-cot init --apply` (or `agent-cot start`) to create it.",
        )
    if not os.access(root, os.W_OK):
        return Check(
            name="paths.agent_cot_root",
            status=CheckStatus.FAIL,
            message=f"data root not writable: {root}",
            hint="check directory permissions; doctor / start need write access here.",
        )
    return Check(
        name="paths.agent_cot_root",
        status=CheckStatus.OK,
        message=str(root),
    )


@_safe
def check_config_file() -> Check:
    p = config_path()
    if not p.is_file():
        return Check(
            name="paths.config",
            status=CheckStatus.WARN,
            message=f"config file missing: {p}",
            hint="run `agent-cot init --apply` to write the default config.",
        )
    cfg = load_config()
    return Check(
        name="paths.config",
        status=CheckStatus.OK,
        message=str(p),
        extra={
            "backend_port": cfg.backend_port,
            "installed_agents": list(cfg.installed_agents),
            "cot_extractor_repo": cfg.cot_extractor_repo,
        },
    )


@_safe
def check_hooks_json() -> Check:
    p = cursor_root() / "hooks.json"
    if not p.is_file():
        return Check(
            name="paths.hooks_json",
            status=CheckStatus.WARN,
            message=f"~/.cursor/hooks.json missing: {p}",
            hint="run `agent-cot init --agent cursor --apply` to install hooks.",
        )
    try:
        import json

        json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return Check(
            name="paths.hooks_json",
            status=CheckStatus.FAIL,
            message=f"hooks.json exists but is not valid JSON: {exc}",
            hint=(
                "back it up (cp hooks.json hooks.json.bak) and re-run "
                "`agent-cot init --agent cursor --apply --force`."
            ),
        )
    return Check(
        name="paths.hooks_json",
        status=CheckStatus.OK,
        message=str(p),
    )


# v0.19.4 (P-12): doctor's default check set used to look at Cursor's
# ``hooks.json`` only. CodeBuddy / Claude have their own paths
# (``~/.codebuddy/settings.json``, ``~/.claude/settings.json``) and
# without symmetric checks, ``agent-cot doctor`` on a CodeBuddy-only
# machine reported "all green" even when settings.json was missing or
# malformed. The helper below is parametric, then we ship one Check
# instance per agent.


def _check_agent_settings_json(
    *, name: str, display: str, path: Path, init_agent: str
) -> Check:
    if not path.is_file():
        return Check(
            name=name,
            status=CheckStatus.WARN,
            message=f"{display} settings.json missing: {path}",
            hint=f"run `agent-cot init --agent {init_agent} --apply` to install hooks.",
        )
    try:
        import json

        blob = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, ValueError) as exc:
        return Check(
            name=name,
            status=CheckStatus.FAIL,
            message=f"{display} settings.json is not valid JSON: {exc}",
            hint=(
                f"back it up and re-run "
                f"`agent-cot init --agent {init_agent} --apply --force`."
            ),
        )
    if not isinstance(blob, dict) or "hooks" not in blob:
        return Check(
            name=name,
            status=CheckStatus.WARN,
            message=f"{display} settings.json present but has no `hooks` block",
            hint=f"run `agent-cot init --agent {init_agent} --apply` to register hooks.",
        )
    return Check(name=name, status=CheckStatus.OK, message=str(path))


@_safe
def check_codebuddy_settings_json() -> Check:
    return _check_agent_settings_json(
        name="paths.codebuddy_settings_json",
        display="CodeBuddy",
        path=Path.home() / ".codebuddy" / "settings.json",
        init_agent="codebuddy",
    )


@_safe
def check_claude_settings_json() -> Check:
    # v0.20.6: prefer the Claude variant that actually exists on disk for
    # the report path. Single-OSS users keep seeing ``~/.claude/``; users
    # who only have Tencent's Cursor-embedded variant see
    # ``~/.claude-internal/``. Dual-install users still get ``~/.claude/``
    # reported here (since ``init`` mirrors writes to the other one
    # transparently — see :func:`check_claude_variants`).
    oss = Path.home() / ".claude" / "settings.json"
    internal = Path.home() / ".claude-internal" / "settings.json"
    path = (
        oss if oss.parent.is_dir()
        else internal if internal.parent.is_dir()
        else oss  # neither installed → fall back to OSS path for the error msg
    )
    return _check_agent_settings_json(
        name="paths.claude_settings_json",
        display="Claude",
        path=path,
        init_agent="claude",
    )


@_safe
def check_claude_variants() -> Check:
    """v0.20.6: surface whether Claude is installed as OSS, Internal, or both.

    Claude Code ships in two flavors:

    * ``~/.claude/``           — Anthropic OSS ``claude`` CLI
    * ``~/.claude-internal/``  — Tencent ``@tencent/claude-code-internal``
      (the variant Cursor embeds)

    Each reads settings.json from its own home directory. ``agent-cot init
    --apply --agent claude`` from v0.20.6+ writes to whichever ones are
    installed (and mirrors to both when both exist). This check exists
    so the doctor report explicitly answers "which Claude are you on?"
    so users (and us during support) can match dashboard symptoms to
    config files.

    Status mapping:

    * **SKIP** — neither variant installed.
    * **OK + 'dual'** — both installed, mirror writes active.
    * **OK + 'oss'** — only ``~/.claude/`` installed (OSS CLI user).
    * **OK + 'internal'** — only ``~/.claude-internal/`` installed
      (Cursor-embedded Tencent variant — the most common司内 case).
    """
    oss_home = Path.home() / ".claude"
    internal_home = Path.home() / ".claude-internal"
    oss_present = oss_home.is_dir()
    internal_present = internal_home.is_dir()

    if not oss_present and not internal_present:
        return Check(
            name="claude.variants",
            status=CheckStatus.SKIP,
            message="No Claude Code variant installed.",
        )

    extra: dict[str, Any] = {}
    if oss_present and internal_present:
        extra["mode"] = "dual"
        msg = (
            "Both ~/.claude/ (OSS) AND ~/.claude-internal/ (Tencent) installed — "
            "agent-cot init mirrors hooks + OTel env to both."
        )
    elif internal_present:
        extra["mode"] = "internal"
        msg = (
            "Only ~/.claude-internal/ installed — using Tencent's "
            "@tencent/claude-code-internal (the variant Cursor embeds). "
            "Hooks + OTel env will be written to ~/.claude-internal/settings.json."
        )
    else:
        extra["mode"] = "oss"
        msg = "Only ~/.claude/ installed (Anthropic OSS claude CLI)."

    return Check(
        name="claude.variants",
        status=CheckStatus.OK,
        message=msg,
        hint=(
            "After agent-cot init, restart your Claude Code process "
            "(close + reopen the terminal / Cursor embedded shell) so the "
            "new OTel env block in settings.json actually loads — env is "
            "read once at startup, not picked up live."
        ),
        extra=extra,
    )


# ---------------------------------------------------------------------------
# v0.20.4: Claude OTel env — is the native Claude Code OpenTelemetry pipe
# wired up to OUR backend, and does the endpoint port still match what
# ``agent-cot start`` ended up on?
#
# This check stays at WARN (never FAIL) because Claude OTel is *opt-in*:
# users who don't care about API-true cost / TTFT / prompt.id linkage
# can run perfectly happily on the hooks + transcript channels alone.
# The check exists to answer "why is the dashboard's ClaudeOtelPanel
# empty even though my hooks fire?" and "why are token counts in the UI
# wildly different from what Claude itself reports?".
# ---------------------------------------------------------------------------


# Keys we require to see Claude's native OTel data land in our backend.
# The values are deliberately NOT enforced — a user with a corp collector
# could be on ``http/protobuf`` and want it that way. We only flag when
# the OTel signal is wired in a way that *cannot possibly* reach our
# claude_otel_receiver.py (which only speaks http/json).
_CLAUDE_OTEL_REQUIRED_KEYS: tuple[str, ...] = (
    "CLAUDE_CODE_ENABLE_TELEMETRY",
    "OTEL_LOGS_EXPORTER",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
)


@_safe
def check_claude_otel_env() -> Check:
    """Inspect ``~/.claude/settings.json`` env for OTel readiness.

    Status mapping:

    * **SKIP** — Claude not installed on this machine
      (``~/.claude/`` missing or no ``settings.json``).
    * **WARN** — Claude installed but ``env`` block is missing entirely.
      Fixable with ``agent-cot init --apply --agent claude`` (writes env
      with fill-missing semantics).
    * **WARN** — ``env`` block present but missing one or more of the
      keys listed in :data:`_CLAUDE_OTEL_REQUIRED_KEYS`. Either the user
      partially configured it, or they upgraded from a release older
      than 0.20.4. Hint points at re-running init.
    * **WARN** — ``CLAUDE_CODE_ENABLE_TELEMETRY != "1"`` (telemetry
      disabled). Some users intentionally disable; we just flag it so
      the answer to "why no OTel data?" is in this report.
    * **WARN** — ``OTEL_EXPORTER_OTLP_PROTOCOL`` not ``http/json``.
      Our receiver only parses JSON. Honors user choice; just notes it.
    * **WARN** — loopback endpoint but port differs from ``config.backend_port``
      (the port ``agent-cot start`` last used). Self-heal in start.py
      handles this on the next ``agent-cot start``, but doctor surfaces
      it for explicit awareness.
    * **OK** + 'foreign endpoint' marker — non-loopback endpoint
      (corporate collector / langfuse / phoenix). User-managed; we
      acknowledge but don't second-guess.
    * **OK** — env is consistent with our pipeline.
    """
    claude_dir = Path.home() / ".claude"
    settings_path = claude_dir / "settings.json"
    if not claude_dir.is_dir():
        return Check(
            name="claude.otel_env",
            status=CheckStatus.SKIP,
            message="Claude Internal not installed on this machine.",
        )
    if not settings_path.is_file():
        return Check(
            name="claude.otel_env",
            status=CheckStatus.SKIP,
            message=f"{settings_path} does not exist.",
            hint=(
                "run `agent-cot init --apply --agent claude` to install "
                "hooks + write the OTel env."
            ),
        )

    import json as _json

    try:
        settings = _json.loads(settings_path.read_text(encoding="utf-8") or "{}")
    except (OSError, ValueError) as exc:
        return Check(
            name="claude.otel_env",
            status=CheckStatus.WARN,
            message=f"could not parse {settings_path}: {exc}",
        )

    if not isinstance(settings, dict):
        return Check(
            name="claude.otel_env",
            status=CheckStatus.WARN,
            message="settings.json top level is not an object",
        )

    env_block = settings.get("env")
    extra: dict[str, Any] = {"settings_path": str(settings_path)}

    if not isinstance(env_block, dict) or not env_block:
        return Check(
            name="claude.otel_env",
            status=CheckStatus.WARN,
            message=(
                "Claude settings.json has no 'env' block — native OTel "
                "data won't flow into the dashboard's ClaudeOtelPanel."
            ),
            hint=(
                "run `agent-cot init --apply --agent claude` to inject the "
                "recommended env (fill-missing only; never overwrites your keys), "
                "then fully quit + reopen Claude Code so the new env takes effect."
            ),
            extra=extra,
        )

    missing = [k for k in _CLAUDE_OTEL_REQUIRED_KEYS if k not in env_block]
    if missing:
        return Check(
            name="claude.otel_env",
            status=CheckStatus.WARN,
            message=f"env block is missing required keys: {', '.join(missing)}",
            hint=(
                "re-run `agent-cot init --apply --agent claude` (no overwrites; "
                "fills only the missing keys). Restart Claude Code afterwards."
            ),
            extra={**extra, "missing_keys": missing},
        )

    telemetry_enabled = str(env_block.get("CLAUDE_CODE_ENABLE_TELEMETRY", "")).strip()
    if telemetry_enabled not in ("1", "true", "True"):
        return Check(
            name="claude.otel_env",
            status=CheckStatus.WARN,
            message=(
                f"CLAUDE_CODE_ENABLE_TELEMETRY={telemetry_enabled!r} — Claude OTel "
                "pipeline is disabled."
            ),
            hint="set CLAUDE_CODE_ENABLE_TELEMETRY='1' in settings.json then restart Claude Code.",
            extra={**extra, "telemetry_enabled_raw": telemetry_enabled},
        )

    protocol = str(env_block.get("OTEL_EXPORTER_OTLP_PROTOCOL", "")).strip().lower()
    if protocol != "http/json":
        return Check(
            name="claude.otel_env",
            status=CheckStatus.WARN,
            message=(
                f"OTEL_EXPORTER_OTLP_PROTOCOL={protocol!r} — our "
                "claude_otel_receiver only speaks http/json. The protobuf "
                "payload will be rejected."
            ),
            hint="set OTEL_EXPORTER_OTLP_PROTOCOL='http/json' then restart Claude Code.",
            extra={**extra, "protocol": protocol},
        )

    endpoint = str(env_block.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")).strip()
    extra["endpoint"] = endpoint
    if not endpoint:
        return Check(
            name="claude.otel_env",
            status=CheckStatus.WARN,
            message="OTEL_EXPORTER_OTLP_ENDPOINT is empty",
            hint="re-run `agent-cot init --apply --agent claude`.",
            extra=extra,
        )

    try:
        from urllib.parse import urlparse

        parsed = urlparse(endpoint)
    except Exception:
        parsed = None

    is_loopback = bool(
        parsed
        and parsed.scheme in ("http", "https")
        and (parsed.hostname or "").lower() in ("127.0.0.1", "localhost", "::1", "0.0.0.0")
    )

    if not is_loopback:
        # User explicitly aimed Claude at a non-loopback target. Our
        # backend can't receive these, but that's a user choice — we
        # just note it so "why is ClaudeOtelPanel empty?" has a clear
        # answer in the report.
        return Check(
            name="claude.otel_env",
            status=CheckStatus.OK,
            message=(
                f"endpoint={endpoint} (user-managed — not pointing at our backend; "
                "ClaudeOtelPanel will stay empty by design)"
            ),
            extra=extra,
        )

    if parsed is None or parsed.port is None:
        return Check(
            name="claude.otel_env",
            status=CheckStatus.WARN,
            message=f"could not parse port out of endpoint {endpoint!r}",
            extra=extra,
        )

    config = load_config()
    expected_port = config.backend_port or 8765
    if parsed.port != expected_port:
        return Check(
            name="claude.otel_env",
            status=CheckStatus.WARN,
            message=(
                f"endpoint port {parsed.port} differs from configured backend "
                f"port {expected_port} — Claude OTel data is being pushed at the "
                f"wrong process (or nowhere)."
            ),
            hint=(
                "run `agent-cot start` once — it will auto-heal the endpoint "
                "to match the current backend port. Then fully quit + reopen "
                "Claude Code so the cached env refreshes."
            ),
            extra={**extra, "expected_port": expected_port, "actual_port": parsed.port},
        )

    return Check(
        name="claude.otel_env",
        status=CheckStatus.OK,
        message=f"endpoint={endpoint}; telemetry on; protocol http/json",
        extra={**extra, "expected_port": expected_port},
    )


# ---------------------------------------------------------------------------
# Bundled assets (P3)
# ---------------------------------------------------------------------------


@_safe
def check_bundled_frontend() -> Check:
    if has_frontend_dist():
        return Check(
            name="bundle.frontend_dist",
            status=CheckStatus.OK,
            message=str(frontend_dist().resolve()),
        )
    return Check(
        name="bundle.frontend_dist",
        status=CheckStatus.WARN,
        message="frontend SPA bundle is empty",
        hint=(
            "run `npm run build` in agent-dashboard/frontend, then "
            "`python -m agent_cot._build_assets sync`."
        ),
    )


@_safe
def check_bundled_backend() -> Check:
    if has_bundled_backend():
        return Check(
            name="bundle.backend",
            status=CheckStatus.OK,
            message=str(bundled_backend_dir().resolve()),
        )
    return Check(
        name="bundle.backend",
        status=CheckStatus.WARN,
        message="bundled backend missing",
        hint="run `python -m agent_cot._build_assets sync`.",
    )


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


@_safe
def check_agent(name: str) -> Check:
    try:
        adapter = get_adapter(name)
    except UnknownAgentError as exc:
        return Check(
            name=f"agents.{name}",
            status=CheckStatus.FAIL,
            message=str(exc),
        )

    detected = adapter.detect_installed()
    extra = {"detected": detected, "minimum_version": adapter.minimum_version}

    try:
        entries = adapter.hook_entries()
        extra["hook_entries"] = len(entries)
    except AgentNotImplementedError:
        return Check(
            name=f"agents.{name}",
            status=CheckStatus.SKIP,
            message=f"{adapter.display_name} adapter is a stub (planned for v0.14)",
            hint=(
                "no action needed; agent-cot does not yet manage hooks "
                f"for {adapter.display_name}."
            ),
            extra=extra,
        )

    if not detected:
        return Check(
            name=f"agents.{name}",
            status=CheckStatus.WARN,
            message=f"{adapter.display_name} not detected on this machine",
            hint=(
                f"if you don't use {adapter.display_name} you can ignore this; "
                "otherwise check that the agent's config dir exists."
            ),
            extra=extra,
        )

    return Check(
        name=f"agents.{name}",
        status=CheckStatus.OK,
        message=f"{adapter.display_name} detected; "
        f"adapter exposes {extra['hook_entries']} hook entries",
        extra=extra,
    )


# ---------------------------------------------------------------------------
# cot-extractor bridge (for OTLP forwarding)
# ---------------------------------------------------------------------------


@_safe
def check_cot_extractor() -> Check:
    """Importable + has the contract we depend on?"""
    from ..commands import otlp_bridge

    try:
        mod = otlp_bridge.import_exporter()
    except otlp_bridge.OtlpBridgeError as exc:
        return Check(
            name="cot-extractor.exporter",
            status=CheckStatus.WARN,
            message=str(exc),
            hint=(
                "set AGENT_COT_EXTRACTOR_SRC, or set "
                "cot_extractor_repo in config.toml, or `pip install` cot-extractor."
            ),
        )

    presets = getattr(mod, "BACKEND_PRESETS", None)
    if not presets:
        return Check(
            name="cot-extractor.exporter",
            status=CheckStatus.WARN,
            message="cot_otlp_exporter imports but BACKEND_PRESETS is missing",
            hint="upgrade cot-extractor to v0.12+ which ships BACKEND_PRESETS.",
        )

    return Check(
        name="cot-extractor.exporter",
        status=CheckStatus.OK,
        message=f"{len(presets)} OTLP presets available",
        extra={"presets": [p.get("id") for p in presets]},
    )


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


@_safe
def check_pid_file() -> Check:
    pf = PidFile.default()
    record = pf.read()
    if record is None:
        return Check(
            name="runtime.pid_file",
            status=CheckStatus.OK,
            message="no backend currently registered (clean state)",
        )

    alive = is_pid_running(record.pid, cmdline_marker=record.cmdline_marker)
    if alive:
        return Check(
            name="runtime.pid_file",
            status=CheckStatus.OK,
            message=f"backend running (pid={record.pid}, port={record.port})",
            extra={
                "pid": record.pid,
                "port": record.port,
                "started_at": record.started_at,
            },
        )
    return Check(
        name="runtime.pid_file",
        status=CheckStatus.WARN,
        message=f"stale PID file (pid={record.pid} no longer running)",
        hint="run `agent-cot stop` to clean it up.",
    )


@_safe
def check_default_port(port: int = 8765) -> Check:
    """Is the conventional dashboard port available, *or* held by us?

    "Held by us" is fine — that's our running backend. "Held by some
    random thing" should warn the user so they understand why
    ``start`` will pick a different port.
    """
    pf = PidFile.default()
    record = pf.read()
    if record is not None and record.port == port and is_pid_running(record.pid):
        return Check(
            name=f"runtime.port.{port}",
            status=CheckStatus.OK,
            message=f"port {port} held by our own backend (pid={record.pid})",
        )

    free = _is_port_free(port)
    if free:
        return Check(
            name=f"runtime.port.{port}",
            status=CheckStatus.OK,
            message=f"port {port} free",
        )
    return Check(
        name=f"runtime.port.{port}",
        status=CheckStatus.WARN,
        message=f"port {port} is in use by another process",
        hint=(
            "harmless — `agent-cot start` will auto-pick a different port. "
            "If you want this exact port, free it first."
        ),
    )


# ---------------------------------------------------------------------------
# Deep checks (v0.18.15) — detect the EXACT failure modes that make a
# colleague's UI render less rich than the maintainer's. Run only when the
# user passes ``--deep``; they're a hair slower (file reads + regex).
# ---------------------------------------------------------------------------


# Match BOTH single-quoted (unpatched template) and double-quoted (patched
# via json.dumps) literals; both spellings of the variable name (RAW_*
# v0.18.15+, bare COT_ROOT/PYTHON v0.18.14 and earlier).
_HOOK_LITERAL_RE = __import__("re").compile(
    r"const\s+(?:RAW_COT_ROOT|COT_ROOT)\s*=\s*process\.env\.COT_EXTRACTOR_ROOT\s*\|\|\s*['\"]([^'\"]*)['\"]"
)
_PYTHON_LITERAL_RE = __import__("re").compile(
    r"const\s+(?:RAW_PYTHON|PYTHON)\s*=\s*process\.env\.COT_PYTHON\s*\|\|\s*['\"]([^'\"]*)['\"]"
)


def _read_hook_text(name: str) -> str | None:
    p = cursor_root() / "hooks" / name
    if not p.is_file():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return None


@_safe
def check_runtime_state() -> Check:
    """``~/.agent-cot/runtime.json`` should point at the current install."""
    state = read_runtime_state()
    if state is None:
        return Check(
            name="deep.runtime_state",
            status=CheckStatus.WARN,
            message=f"runtime.json missing: {runtime_state_path()}",
            hint=(
                "run `agent-cot start` (writes runtime.json automatically). "
                "Without it, hook scripts can only self-heal via Python probe (slower)."
            ),
        )
    pkg_ver = state.get("agent_cot_version")
    if pkg_ver != __version__:
        return Check(
            name="deep.runtime_state",
            status=CheckStatus.WARN,
            message=(
                f"runtime.json was written by agent-cot {pkg_ver}, "
                f"current install is {__version__}"
            ),
            hint="run `agent-cot start` to refresh runtime.json.",
            extra={"path": str(runtime_state_path()), **state},
        )
    extractor = state.get("cot_extractor_root")
    if not extractor or not (Path(extractor) / "scripts" / "extract_cot.py").is_file():
        return Check(
            name="deep.runtime_state",
            status=CheckStatus.FAIL,
            message=(
                f"runtime.json points at extractor root that no longer has "
                f"scripts/extract_cot.py: {extractor}"
            ),
            hint=(
                "run `agent-cot start` (rewrites runtime.json from the "
                "current install)."
            ),
            extra={"path": str(runtime_state_path()), **state},
        )
    return Check(
        name="deep.runtime_state",
        status=CheckStatus.OK,
        message=f"runtime.json current ({__version__}); extractor → {extractor}",
        extra={"path": str(runtime_state_path()), **state},
    )


@_safe
def check_hook_scripts_alive() -> Check:
    """The on-disk Cursor hook scripts should resolve a real ``extract_cot.py``."""
    bridge_text = _read_hook_text("cot-bridge.js")
    stream_text = _read_hook_text("cot-stream.js")
    if bridge_text is None and stream_text is None:
        return Check(
            name="deep.hook_scripts",
            status=CheckStatus.WARN,
            message=(
                f"no hook scripts at {cursor_root() / 'hooks'} — Cursor hooks "
                "won't fire."
            ),
            hint="run `agent-cot init --apply` to install them.",
        )

    findings: dict[str, Any] = {}
    fails: list[str] = []
    warns: list[str] = []

    for name, text in (("cot-bridge.js", bridge_text), ("cot-stream.js", stream_text)):
        if text is None:
            warns.append(f"{name} missing")
            continue
        m = _HOOK_LITERAL_RE.search(text)
        literal = m.group(1) if m else None
        findings[f"{name}.cot_root_literal"] = literal
        if not literal or literal.startswith("__AGENT_COT_EXTRACTOR_ROOT_UNCONFIGURED__"):
            warns.append(f"{name}: COT_ROOT literal unconfigured (will rely on runtime.json or env)")
        else:
            extract_py = Path(literal) / "scripts" / "extract_cot.py"
            if not extract_py.is_file():
                fails.append(
                    f"{name}: COT_ROOT literal points at vanished path "
                    f"{literal!r} (no scripts/extract_cot.py)"
                )

    if name == "cot-bridge.js" and stream_text is not None:
        # check PYTHON literal in cot-bridge.js too
        pmatch = _PYTHON_LITERAL_RE.search(bridge_text or "")
        py_literal = pmatch.group(1) if pmatch else None
        findings["cot-bridge.js.python_literal"] = py_literal
        if not py_literal or py_literal.startswith("__AGENT_COT_PYTHON_UNCONFIGURED__"):
            warns.append(
                "cot-bridge.js: PYTHON literal unconfigured (will fall back "
                "to PATH probe)"
            )
        elif not Path(py_literal).is_file():
            fails.append(
                f"cot-bridge.js: PYTHON literal points at vanished interpreter "
                f"{py_literal!r}"
            )

    if fails:
        return Check(
            name="deep.hook_scripts",
            status=CheckStatus.FAIL,
            message="; ".join(fails),
            hint=(
                "run `agent-cot start` (auto self-heals on v0.18.15+) "
                "or `agent-cot upgrade --apply` (rewrites the literals)."
            ),
            extra=findings,
        )
    if warns:
        return Check(
            name="deep.hook_scripts",
            status=CheckStatus.WARN,
            message="; ".join(warns),
            hint=(
                "harmless if runtime.json is current (`agent-cot doctor "
                "--deep` checks that separately)."
            ),
            extra=findings,
        )
    return Check(
        name="deep.hook_scripts",
        status=CheckStatus.OK,
        message="hook scripts on disk resolve to a real extract_cot.py",
        extra=findings,
    )


@_safe
def check_codebuddy_hook_alive() -> Check:
    """The on-disk CodeBuddy hook script should write events.jsonl to a sane root.

    Codebuddy's ``cot-stream-codebuddy.js`` is literal-free (no ``COT_ROOT``
    patch, no ``PYTHON`` patch — its only job is to append events.jsonl to
    a per-session dir under ``AGENT_COT_DATA_ROOT``). What CAN go wrong is:

    1. ~/.codebuddy/ exists but the hook was never installed (user added
       ``hooks`` to settings.json manually with a stale path).
    2. The hook on disk is from an *older* wheel that pre-dates the
       ``AGENT_COT_DATA_ROOT`` defaulting fix (v0.18.5) — it would still
       write to a site-packages-relative path, lose data silently.
    3. The hook on disk is current but ``AGENT_COT_DATA_ROOT`` env +
       runtime.json + homedir-default all evaluate to the *same* writable
       directory (the happy path).

    We confirm by reading the file and grepping for the v0.18.5+ marker
    (``AGENT_COT_DATA_ROOT``) and the v0.19.0+ marker (``runtime.json``
    fallback). Mismatches WARN (not FAIL — codebuddy is optional).
    """
    cb_dir = Path.home() / ".codebuddy"
    if not cb_dir.is_dir():
        return Check(
            name="deep.codebuddy_hook",
            status=CheckStatus.SKIP,
            message="CodeBuddy not installed on this machine (~/.codebuddy missing).",
        )

    hook_path = cb_dir / "hooks" / "cot-stream-codebuddy.js"
    if not hook_path.is_file():
        return Check(
            name="deep.codebuddy_hook",
            status=CheckStatus.WARN,
            message=(
                f"CodeBuddy detected but hook not installed at {hook_path}."
            ),
            hint=(
                "run `agent-cot init --agent codebuddy --apply` to install "
                "the hook + register it in ~/.codebuddy/settings.json."
            ),
        )

    try:
        body = hook_path.read_text(encoding="utf-8")
    except OSError as exc:
        return Check(
            name="deep.codebuddy_hook",
            status=CheckStatus.FAIL,
            message=f"cannot read {hook_path}: {exc}",
        )

    findings: dict[str, Any] = {
        "hook_path": str(hook_path),
        "size_bytes": len(body),
        "has_data_root_env_check": "AGENT_COT_DATA_ROOT" in body,
        "has_runtime_state_fallback": "runtime.json" in body
            and "readRuntimeState" in body,
        "has_writable_default": ".agent-cot" in body and "data" in body,
    }
    warns: list[str] = []

    if not findings["has_data_root_env_check"]:
        warns.append(
            "hook does not honor AGENT_COT_DATA_ROOT — pre-v0.18.5 build, "
            "events.jsonl will land in a site-packages dir (silent data loss)"
        )
    if not findings["has_writable_default"]:
        warns.append(
            "hook has no ~/.agent-cot/data fallback — old build, will fail "
            "when AGENT_COT_DATA_ROOT is unset"
        )
    if not findings["has_runtime_state_fallback"]:
        # v0.19.0 marker; older but functional installs still work, just
        # without the cross-process self-heal layer.
        warns.append(
            "hook lacks runtime.json fallback (pre-v0.19.0) — works as long "
            "as AGENT_COT_DATA_ROOT or homedir resolves correctly"
        )

    # Independently confirm the resolved DATA_ROOT matches what backend reads.
    expected_root = agent_cot_root() / "data"
    findings["expected_data_root"] = str(expected_root)
    findings["expected_data_root_writable"] = expected_root.is_dir() or (
        expected_root.parent.is_dir()
    )

    if warns:
        return Check(
            name="deep.codebuddy_hook",
            status=CheckStatus.WARN,
            message="; ".join(warns),
            hint=(
                "run `agent-cot start` (auto self-heals on v0.19.0+) "
                "or `agent-cot upgrade --apply` to refresh the hook bytes."
            ),
            extra=findings,
        )
    return Check(
        name="deep.codebuddy_hook",
        status=CheckStatus.OK,
        message=(
            f"codebuddy hook installed, env+runtime.json+default fallbacks "
            f"all wired (writes to {expected_root})"
        ),
        extra=findings,
    )


@_safe
def check_claude_hook_alive() -> Check:
    """The on-disk Claude Internal hook script + settings.json registration.

    v0.19.1 引入。Claude 这条链路特殊：

    * hook 不是 .js 而是 ``~/.claude/hooks/claude_stream_hook.py``，里面有
      ``_maybe_trigger_extract`` 函数 + 4 层 fallback 路径解析 +
      30 秒 debounce —— 这三件事是 v0.19.1 的关键标记，缺任何一个就说明
      hook 是旧版（或被 sync 倒退过），cot.json 不会自动出现在前端。
    * Claude Code 的 hook 配置在 ``~/.claude/settings.json``（嵌套
      ``hooks.<Event>[].hooks[]`` 结构），而不是 ``~/.cursor/hooks.json``
      那种扁平 schema。

    我们检查：

    1. ``~/.claude/`` 目录存在（不在 → SKIP，跟 codebuddy 一致）
    2. ``~/.claude/hooks/claude_stream_hook.py`` 存在
    3. hook 文件含 v0.19.1 三个关键 marker（_maybe_trigger_extract,
       _resolve_extractor_root, _debounce_should_run）
    4. ``~/.claude/settings.json`` 至少有 1 个 event 注册了我们的 hook
       （检查 command 含 ``claude_stream_hook.py``）
    """
    claude_dir = Path.home() / ".claude"
    if not claude_dir.is_dir():
        return Check(
            name="deep.claude_hook",
            status=CheckStatus.SKIP,
            message="Claude Internal not installed on this machine (~/.claude missing).",
        )

    hook_path = claude_dir / "hooks" / "claude_stream_hook.py"
    if not hook_path.is_file():
        return Check(
            name="deep.claude_hook",
            status=CheckStatus.WARN,
            message=(
                f"Claude detected but hook not installed at {hook_path}."
            ),
            hint=(
                "run `agent-cot init --agent claude --apply` to install "
                "the hook + register it in ~/.claude/settings.json."
            ),
        )

    try:
        body = hook_path.read_text(encoding="utf-8")
    except OSError as exc:
        return Check(
            name="deep.claude_hook",
            status=CheckStatus.FAIL,
            message=f"cannot read {hook_path}: {exc}",
        )

    findings: dict[str, Any] = {
        "hook_path": str(hook_path),
        "size_bytes": len(body),
        "has_extract_trigger": "_maybe_trigger_extract" in body,
        "has_4layer_resolver": "_resolve_extractor_root" in body
            and "_resolve_python" in body,
        "has_debounce": "_debounce_should_run" in body,
    }
    warns: list[str] = []

    if not findings["has_extract_trigger"]:
        warns.append(
            "hook is pre-v0.19.1 (no _maybe_trigger_extract) — events.jsonl "
            "will be written but no cot.json gets generated, so Claude "
            "sessions will NOT appear in the dashboard"
        )
    if not findings["has_4layer_resolver"]:
        warns.append(
            "hook lacks 4-layer resolver (env > runtime.json > probe) — "
            "may fail to find cot-extractor / python on a fresh machine"
        )
    if not findings["has_debounce"]:
        warns.append(
            "hook lacks debounce (pre-v0.19.1) — repeated Stop events in a "
            "short window will spawn extract_cot multiple times, costly but "
            "not dangerous"
        )

    # settings.json 注册检查
    settings_path = claude_dir / "settings.json"
    findings["settings_path"] = str(settings_path)
    if not settings_path.is_file():
        warns.append(
            f"~/.claude/settings.json missing — Claude Code won't ever "
            f"trigger our hook even though it's installed at {hook_path}. "
            f"Run `agent-cot init --agent claude --apply` to register."
        )
    else:
        try:
            import json as _json
            settings = _json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            warns.append(f"settings.json unreadable / invalid JSON: {exc}")
            settings = {}
        registered_events: list[str] = []
        hooks_block = settings.get("hooks") if isinstance(settings, dict) else None
        if isinstance(hooks_block, dict):
            for event, groups in hooks_block.items():
                if not isinstance(groups, list):
                    continue
                for group in groups:
                    if not isinstance(group, dict):
                        continue
                    cmds = group.get("hooks") or []
                    if not isinstance(cmds, list):
                        continue
                    for cmd_obj in cmds:
                        if not isinstance(cmd_obj, dict):
                            continue
                        cmd = str(cmd_obj.get("command") or "")
                        if "claude_stream_hook.py" in cmd:
                            registered_events.append(str(event))
                            break
        findings["registered_event_count"] = len(registered_events)
        findings["registered_events_sample"] = registered_events[:5]
        if not registered_events:
            warns.append(
                "settings.json exists but NO event registers our hook — "
                "run `agent-cot init --agent claude --apply` to register."
            )
        elif len(registered_events) < 5:
            warns.append(
                f"only {len(registered_events)} event(s) register our hook "
                "(expected 27 for full coverage). Re-run init to refresh."
            )

    # 4 层 fallback 链能否真的找到 extractor？
    state = read_runtime_state()
    if state and isinstance(state, dict):
        extractor = state.get("cot_extractor_root")
        if extractor and (Path(extractor) / "scripts" / "extract_cot.py").is_file():
            findings["fallback_chain"] = "agent-cot/runtime.json OK"
        else:
            findings["fallback_chain"] = "agent-cot/runtime.json points at vanished path"
            warns.append(
                "agent-cot runtime.json points at non-existent extractor — "
                "Claude hook would still fall back to .cursor-cot/runtime.json "
                "or sys.executable probe, but explicitly running `agent-cot start` "
                "would refresh runtime.json."
            )
    else:
        # Try cursor-cot fallback
        cc_runtime = Path.home() / ".cursor-cot" / "runtime.json"
        if cc_runtime.is_file():
            findings["fallback_chain"] = "cursor-cot/runtime.json (legacy fallback)"
        else:
            findings["fallback_chain"] = "no runtime.json available"
            warns.append(
                "neither agent-cot nor cursor-cot runtime.json present — "
                "hook will rely on COT_EXTRACTOR_ROOT env or sys.executable "
                "probe at spawn time. Run `agent-cot start` to write runtime.json."
            )

    if warns:
        return Check(
            name="deep.claude_hook",
            status=CheckStatus.WARN,
            message="; ".join(warns),
            hint=(
                "run `agent-cot start` (auto self-heals on v0.19.1+) "
                "or `agent-cot upgrade --apply` to refresh the hook bytes; "
                "for the very first install run `agent-cot init --agent claude --apply`."
            ),
            extra=findings,
        )
    return Check(
        name="deep.claude_hook",
        status=CheckStatus.OK,
        message=(
            f"claude hook current (v0.19.1+ markers all present); "
            f"{findings.get('registered_event_count', 0)} events registered "
            f"in settings.json; fallback={findings.get('fallback_chain', '?')}"
        ),
        extra=findings,
    )


@_safe
def check_recent_data() -> Check:
    """Are recent ``cot.json`` files actually being written?"""
    cot_dir = agent_cot_root() / "data" / "cot"
    if not cot_dir.is_dir():
        return Check(
            name="deep.recent_data",
            status=CheckStatus.WARN,
            message=f"data dir missing: {cot_dir}",
            hint="run `agent-cot start` and trigger one Cursor turn to populate.",
        )
    import time as _time

    files = sorted(cot_dir.glob("*_cot.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return Check(
            name="deep.recent_data",
            status=CheckStatus.WARN,
            message="no cot.json files yet — hooks may have never fired.",
            hint=(
                "run a Cursor turn (any prompt), wait 5s, then re-run "
                "`agent-cot doctor --deep`. If still empty, see the "
                "deep.hook_scripts check."
            ),
        )
    newest = files[0]
    age_s = _time.time() - newest.stat().st_mtime
    extra = {
        "newest_file": newest.name,
        "newest_size": newest.stat().st_size,
        "age_seconds": int(age_s),
        "total_files": len(files),
    }
    if age_s > 7 * 86400:
        return Check(
            name="deep.recent_data",
            status=CheckStatus.WARN,
            message=(
                f"newest cot.json is {int(age_s/3600)}h old: {newest.name}. "
                "If you've been using Cursor recently, hooks may not be firing."
            ),
            hint=(
                "check `~/.cursor/cot-bridge.log` for spawn errors; "
                "verify hooks.json contains our entries (agent-cot doctor --verbose)."
            ),
            extra=extra,
        )
    return Check(
        name="deep.recent_data",
        status=CheckStatus.OK,
        message=(
            f"{len(files)} session(s); newest is {newest.name} "
            f"({int(age_s)}s old)"
        ),
        extra=extra,
    )


@_safe
def check_recent_data_richness() -> Check:
    """Spot-check that the newest cot.json actually contains TOOL_DECISION steps.

    This is the marquee deep-check: if the colleague says "I ran a complex
    prompt with TodoWrite and got nothing in the UI", but cot.json has 0
    TOOL_DECISION steps, we know the extractor ran but its transcript-tool_use
    parsing missed everything (most commonly: the transcript path was wrong
    so only events.jsonl-derived synthesised executions made it through).
    """
    import json as _json
    import time as _time

    cot_dir = agent_cot_root() / "data" / "cot"
    if not cot_dir.is_dir():
        return Check(
            name="deep.data_richness",
            status=CheckStatus.SKIP,
            message="no data dir yet — see deep.recent_data.",
        )
    files = sorted(cot_dir.glob("*_cot.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return Check(
            name="deep.data_richness",
            status=CheckStatus.SKIP,
            message="no cot.json files — see deep.recent_data.",
        )
    newest = files[0]
    age_s = _time.time() - newest.stat().st_mtime
    if newest.stat().st_size > 80 * 1024 * 1024:
        return Check(
            name="deep.data_richness",
            status=CheckStatus.SKIP,
            message=f"newest cot.json too large ({newest.stat().st_size//1024//1024} MB) "
                    "to inspect; run on a smaller session manually.",
        )
    try:
        data = _json.loads(newest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return Check(
            name="deep.data_richness",
            status=CheckStatus.WARN,
            message=f"newest cot.json unreadable: {exc}",
            hint="extractor may have crashed mid-write; check ~/.cursor/cot-bridge.log.",
        )

    decision_count = 0
    execution_count = 0
    todo_count = 0
    for turn in data.get("turns", []):
        for s in (turn.get("steps") or []):
            stype = s.get("step_type")
            if stype == "tool_decision":
                decision_count += 1
                if (s.get("tool_name") or "") == "TodoWrite":
                    todo_count += 1
            elif stype == "tool_execution":
                execution_count += 1

    extra = {
        "session_id": newest.stem.removesuffix("_cot"),
        "age_seconds": int(age_s),
        "tool_decision_count": decision_count,
        "tool_execution_count": execution_count,
        "todo_write_count": todo_count,
        "size_bytes": newest.stat().st_size,
    }

    if decision_count == 0 and execution_count == 0:
        return Check(
            name="deep.data_richness",
            status=CheckStatus.FAIL,
            message=(
                f"newest cot.json {newest.name} has 0 tool_decision AND 0 tool_execution steps. "
                "Extractor likely couldn't find the transcript or events.jsonl."
            ),
            hint=(
                "check `~/.cursor/cot-bridge.log` for 'extract_cot.py missing' / "
                "'Located by ...' lines; verify that "
                "~/.cursor/projects/<slug>/agent-transcripts/<sid>/<sid>.jsonl exists."
            ),
            extra=extra,
        )
    if decision_count == 0 and execution_count > 0:
        return Check(
            name="deep.data_richness",
            status=CheckStatus.WARN,
            message=(
                f"newest cot.json {newest.name} has 0 tool_decision but "
                f"{execution_count} tool_execution steps. Plan/file_op badges "
                "and the right-side LLM-call panel will be EMPTY in the UI."
            ),
            hint=(
                "transcript may have no `tool_use` blocks (Cursor v2.6+ moved "
                "them to events.jsonl in some builds). Run a *new* Cursor turn "
                "with TodoWrite + StrReplace/Write and re-check; if still 0, the "
                "transcript jsonl needs investigation (open it manually, look for "
                "'tool_use' entries)."
            ),
            extra=extra,
        )
    return Check(
        name="deep.data_richness",
        status=CheckStatus.OK,
        message=(
            f"newest cot.json: {decision_count} tool_decision + "
            f"{execution_count} tool_execution + {todo_count} TodoWrite calls"
        ),
        extra=extra,
    )


@_safe
def check_bundled_extractor_present() -> Check:
    """The wheel must ship a usable cot-extractor for the bridge to spawn."""
    if has_bundled_extractor():
        return Check(
            name="deep.bundled_extractor",
            status=CheckStatus.OK,
            message=str(bundled_extractor_root().resolve()),
        )
    return Check(
        name="deep.bundled_extractor",
        status=CheckStatus.FAIL,
        message="wheel-bundled cot-extractor missing scripts/extract_cot.py",
        hint=(
            "your wheel install is incomplete; reinstall: "
            "`pip install --force-reinstall agent-cot`."
        ),
    )


# ---------------------------------------------------------------------------
# Entry point used by runner.py
# ---------------------------------------------------------------------------


def all_checks(*, deep: bool = False) -> list[Check]:
    """Run every diagnostic in the canonical order.

    When ``deep`` is True, also include v0.18.15+ deep checks that read
    on-disk hook scripts, runtime.json, and recent cot.json files. These
    cost a few extra disk reads + small JSON parses; cheap on a healthy
    machine, slower on a session corpus of GBs.
    """
    out: list[Check] = []
    out.append(check_python_version())
    out.append(check_agent_cot_version())
    out.append(check_distribution_conflict())
    out.append(check_cursor_renderer_logs())

    out.append(
        check_required_dep(
            "click",
            hint="`pip install click>=8.1` (should be a transitive dep).",
        )
    )
    out.append(
        check_required_dep(
            "tomli_w",
            hint="`pip install tomli_w` (transitive dep; reinstall agent-cot).",
        )
    )

    out.append(check_optional_dep("fastapi", "dashboard"))
    out.append(check_optional_dep("uvicorn", "dashboard"))
    out.append(check_optional_dep("opentelemetry.sdk", "otlp"))
    out.append(
        check_optional_dep(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter",
            "otlp",
        )
    )
    out.append(check_optional_dep("psutil", "dashboard"))

    out.append(check_agent_cot_root())
    out.append(check_config_file())
    out.append(check_hooks_json())
    # v0.19.4 (P-12): symmetric default checks for CodeBuddy + Claude
    # settings.json. These remain ``warn`` (not ``fail``) when missing
    # because most users won't have all three IDEs installed —— but at
    # least they now show up in the report instead of silently being
    # "all green".
    out.append(check_codebuddy_settings_json())
    out.append(check_claude_settings_json())
    # v0.20.4: native Claude OTel env wiring (port-aware).
    out.append(check_claude_otel_env())
    # v0.20.6: OSS vs Internal vs Dual mode detection.
    out.append(check_claude_variants())

    out.append(check_bundled_frontend())
    out.append(check_bundled_backend())

    for name in list_agents():
        out.append(check_agent(name))

    out.append(check_cot_extractor())

    out.append(check_pid_file())
    out.append(check_default_port(8765))

    # v0.18.15: deep checks — these are what tells the user "your hook
    # script is patched at a stale path" or "your cot.json has 0 tool
    # decisions". Worth running by default on `agent-cot doctor`?
    # Decision: opt-in via --deep. Cheap enough that we could promote
    # later, but for now keep `doctor` itself snappy.
    if deep:
        out.append(check_bundled_extractor_present())
        out.append(check_runtime_state())
        out.append(check_hook_scripts_alive())
        # v0.19.0: codebuddy hook health (literal-free hook, but needs the
        # AGENT_COT_DATA_ROOT + runtime.json fallbacks to be in the bytes).
        out.append(check_codebuddy_hook_alive())
        # v0.19.1: claude hook health (Python hook with extract trigger +
        # 4-layer resolver; checks both file content markers and
        # ~/.claude/settings.json registration).
        out.append(check_claude_hook_alive())
        out.append(check_recent_data())
        out.append(check_recent_data_richness())

    return out


__all__ = ["Check", "CheckStatus", "all_checks"]
