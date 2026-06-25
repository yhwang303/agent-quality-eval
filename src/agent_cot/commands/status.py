"""``agent-cot status`` — report on the local dashboard runtime.

Read-only. Never spawns, kills, or writes anything; safe to invoke
from cron / CI without authorisation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from ..installer.config import config_path, load_config
from ..installer.platform_paths import agent_cot_root, cursor_root
from ..runtime import PidFile, is_pid_running
from ..runtime.health import probe_once


@dataclass
class StatusReport:
    """Machine-readable snapshot of the runtime, used by ``--json``."""

    backend_running: bool
    backend_pid: int | None
    backend_port: int | None
    backend_health_ok: bool
    backend_last_error: str | None
    started_at_iso: str | None
    log_path: str | None
    pid_file: str
    config_file: str
    agent_cot_root: str
    hooks_config: str
    hooks_config_present: bool
    installed_agents: list[str]
    cot_extractor_repo: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def collect_status() -> StatusReport:
    """Build a :class:`StatusReport` without performing any side effects.

    Exists separately from the click command so callers (tests, future
    ``doctor``) can consume the structured form.
    """
    config = load_config()
    pid_file = PidFile.default()
    record = pid_file.read()

    running = False
    health_ok = False
    last_error: str | None = "no PID file"
    pid: int | None = None
    port: int | None = config.backend_port
    started_at_iso: str | None = None
    log_path: str | None = None

    if record is not None:
        pid = record.pid
        port = record.port
        log_path = record.log_path
        if record.started_at:
            started_at_iso = (
                datetime.fromtimestamp(record.started_at, tz=timezone.utc)
                .isoformat(timespec="seconds")
            )
        if is_pid_running(record.pid, cmdline_marker=record.cmdline_marker):
            running = True
            health_ok, probe_err = probe_once(record.port, timeout=1.0)
            last_error = None if health_ok else probe_err
        else:
            last_error = "stale PID file (process exited)"

    hooks_path = cursor_root() / "hooks.json"

    return StatusReport(
        backend_running=running,
        backend_pid=pid,
        backend_port=port,
        backend_health_ok=health_ok,
        backend_last_error=last_error,
        started_at_iso=started_at_iso,
        log_path=log_path,
        pid_file=str(pid_file.path),
        config_file=str(config_path()),
        agent_cot_root=str(agent_cot_root()),
        hooks_config=str(hooks_path),
        hooks_config_present=hooks_path.is_file(),
        installed_agents=list(config.installed_agents),
        cot_extractor_repo=config.cot_extractor_repo,
    )


def render_human(report: StatusReport) -> str:
    """Pretty-print ``report`` for the default (non-JSON) CLI output."""
    lines: list[str] = []

    badge_color = "green" if report.backend_running and report.backend_health_ok else (
        "yellow" if report.backend_running else "red"
    )
    badge = (
        "● running & healthy"
        if report.backend_running and report.backend_health_ok
        else "● running but unhealthy"
        if report.backend_running
        else "○ not running"
    )
    lines.append(f"backend         : {badge}")
    if report.backend_pid:
        lines.append(f"  pid           : {report.backend_pid}")
    if report.backend_port:
        lines.append(f"  port          : {report.backend_port}")
    if report.started_at_iso:
        lines.append(f"  started at    : {report.started_at_iso}")
    if report.log_path:
        lines.append(f"  log           : {report.log_path}")
    if report.backend_last_error:
        lines.append(f"  last error    : {report.backend_last_error}")
    lines.append("")
    lines.append(f"pid file        : {report.pid_file}")
    lines.append(f"config file     : {report.config_file}")
    lines.append(f"data root       : {report.agent_cot_root}")
    lines.append("")
    hooks_state = "present" if report.hooks_config_present else "absent"
    lines.append(f"hooks.json      : {report.hooks_config} ({hooks_state})")
    lines.append(
        f"installed agents: {', '.join(report.installed_agents) or '(none)'}"
    )
    cot = report.cot_extractor_repo or "(not detected — run `agent-cot init`)"
    lines.append(f"cot-extractor   : {cot}")

    # Suppress unused-variable warning for badge_color; it's available
    # for callers that want colour output.
    del badge_color
    return "\n".join(lines)


__all__ = ["StatusReport", "collect_status", "render_human"]
