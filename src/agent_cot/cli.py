"""Top-level CLI entry point — wired up via ``[project.scripts]``.

The ``agent-cot`` command tree at v0.13.0a0 is intentionally thin: only
``--version``, ``--help``, ``agents``, and one stub per planned command.

Each stub command:
* parses its own flags so end-users can ``--help`` it today,
* exits with a clear message naming which roadmap phase implements it,
* never silently no-ops.

This shape lets us merge the package onto PyPI early and iterate on
behaviour without breaking users' muscle memory.
"""

from __future__ import annotations

import sys
from typing import NoReturn

import click

from . import __version__
from .agents import (
    AgentAdapter,
    AgentNotImplementedError,
    UnknownAgentError,
    get_adapter,
    list_agents,
)
from .agents.base import CursorCotError

# ---------------------------------------------------------------------------
# Error formatting helpers
# ---------------------------------------------------------------------------


def _die(message: str, exit_code: int = 1) -> NoReturn:
    """Print a red error line and exit. Avoids stack traces for known cases."""
    click.secho(f"error: {message}", fg="red", err=True)
    sys.exit(exit_code)


def _resolve_agent(name: str) -> AgentAdapter:
    """Look up an adapter or die with a friendly message."""
    try:
        return get_adapter(name)
    except UnknownAgentError as exc:
        _die(str(exc))


def _stub_message(phase: str) -> str:
    return (
        f"this command is scheduled for {phase} (see SETUP_PLAN.md). "
        "v0.13.0a0 ships only the CLI skeleton."
    )


# ---------------------------------------------------------------------------
# Shared option decorators
# ---------------------------------------------------------------------------


def _agent_option(default: str = "cursor"):
    """Common ``--agent`` flag used by every command that touches an agent."""
    return click.option(
        "--agent",
        "agent_name",
        default=default,
        show_default=True,
        metavar="NAME",
        help=f"Target agent ({', '.join(list_agents())}, or 'all').",
    )


# ---------------------------------------------------------------------------
# Root command group
# ---------------------------------------------------------------------------


@click.group(
    help=(
        "agent-cot · Local-first Chain-of-Thought observability "
        "for Cursor (Claude Code support coming in v0.14).\n\n"
        "Run `agent-cot init` once, then `agent-cot start` whenever you "
        "want the local dashboard. See SETUP_PLAN.md for the full roadmap."
    ),
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(
    __version__,
    "-V",
    "--version",
    prog_name="agent-cot",
    message="%(prog)s %(version)s",
)
def main() -> None:
    """Root command. Subcommands are mounted below."""


# ---------------------------------------------------------------------------
# `agents` — works today, useful for sanity-checking installation
# ---------------------------------------------------------------------------


@main.command("agents", help="List supported agent adapters and their status.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def cmd_agents(as_json: bool) -> None:
    rows: list[dict[str, str | bool]] = []
    for name in list_agents():
        adapter = get_adapter(name)
        rows.append(
            {
                "name": adapter.name,
                "display_name": adapter.display_name,
                "minimum_version": adapter.minimum_version,
                "installed_locally": adapter.detect_installed(),
            }
        )

    if as_json:
        import json

        click.echo(json.dumps(rows, indent=2))
        return

    width = max(len(str(r["display_name"])) for r in rows)
    click.echo(click.style("Supported agents:", bold=True))
    for r in rows:
        bullet = "✓" if r["installed_locally"] else "·"
        color = "green" if r["installed_locally"] else "yellow"
        click.echo(
            "  "
            + click.style(bullet, fg=color)
            + f"  {r['display_name']!s:<{width}}  "
            + click.style(f"(supported from {r['minimum_version']})", dim=True)
        )
    click.echo("")
    click.echo(
        click.style("legend: ", dim=True)
        + click.style("✓", fg="green")
        + click.style(" detected on this machine, ", dim=True)
        + click.style("·", fg="yellow")
        + click.style(" not detected", dim=True)
    )


# ---------------------------------------------------------------------------
# Stub commands (P1+)
# ---------------------------------------------------------------------------


@main.command(
    "init",
    help="One-time setup: register hooks, pick ports, write config.",
)
@_agent_option()
@click.option(
    "--dry-run/--apply",
    default=True,
    show_default=True,
    help="Plan-only mode (default) vs. actually writing to disk.",
)
@click.option(
    "--force",
    is_flag=True,
    help="With --apply: proceed even if our hooks already look installed.",
)
@click.option(
    "--no-cursor-hooks",
    is_flag=True,
    help="Skip writing the agent's hook config (CI / server use).",
)
@click.option(
    "--port-backend",
    type=int,
    default=None,
    help="Pin the backend port (default: auto-pick).",
)
@click.option(
    "--no-otel-env",
    is_flag=True,
    help=(
        "Claude only: skip auto-merging the OTel env block into "
        "~/.claude/settings.json. Useful when you point Claude Code at "
        "a different OTel collector (corporate / Langfuse / Phoenix). "
        "Default behaviour is fill-missing-only — pre-existing env keys "
        "are NEVER overwritten."
    ),
)
def cmd_init(
    agent_name: str,
    dry_run: bool,
    force: bool,
    no_cursor_hooks: bool,
    port_backend: int | None,
    no_otel_env: bool,
) -> None:
    from .commands import init as init_cmd

    _resolve_agent(agent_name)  # surfaces UnknownAgentError early

    try:
        plan = init_cmd.build_plan(
            agent_name=agent_name,
            port_backend=port_backend,
            write_otel_env=not no_otel_env,
        )
    except AgentNotImplementedError as exc:
        _die(str(exc))

    click.echo(click.style("agent-cot init — plan", bold=True))
    click.echo(plan.render_summary())

    if dry_run:
        click.echo("")
        click.echo(
            click.style(
                "(dry-run; nothing written. Re-run with --apply to commit.)",
                fg="yellow",
            )
        )
        return

    if no_cursor_hooks:
        click.echo("")
        click.echo(
            click.style(
                "--no-cursor-hooks given: skipping hooks.json / hook script writes.",
                fg="yellow",
            )
        )
        # Still persist config so `start` knows which port to use.
        from .installer.config import save_config

        save_config(plan.config)
        click.echo(
            click.style(
                f"  ✓ wrote {(plan.config and 'config') or ''}",
                fg="green",
            )
        )
        return

    if not force:
        # hooks.json 已合并时 diff 为空 —— 以前这里直接 return，导致 **从不**
        # 重新拷贝 cot-bridge.js / cot-stream.js。用户 pip 升到 0.18.2 后若不再改
        # hooks.json，磁盘上的 JS 仍保留 0.18.1 时写进去的 ``D:/ai-ide-langfuse/...``
        # 默认字面量，status 虽能解析出 wheel 内 cot-extractor，hook 却永远指错路径。
        #
        # v0.20.4 新增条件：env_additions 也得是空 —— 否则同学从 0.20.3 升 0.20.4
        # 时虽然 27 个 hook 已经齐了（diff.has_changes=False），但 settings.json
        # 还缺 OTel env block，必须走 apply_plan 才能把 env 写进去；走 refresh
        # 路径只会刷 hook 脚本字节，不会动 settings.json。
        if not plan.diff.has_changes and not plan.env_additions:
            click.echo("")
            click.echo(
                click.style(
                    "hooks.json unchanged — refreshing hook scripts on disk "
                    "(cot-bridge.js / cot-stream.js) so COT_ROOT matches this install.",
                    fg="yellow",
                )
            )
            result = init_cmd.refresh_bundled_hook_scripts(plan)
            click.echo("")
            click.secho("agent-cot init — done", bold=True, fg="green")
            for p in result.scripts_installed:
                click.echo(f"  ✓ refreshed    {p}")
            if result.config_written:
                click.echo(f"  ✓ saved config {result.config_written}")
            click.echo("")
            click.echo(
                click.style(
                    "Fully quit Cursor (all Cursor.exe), reopen, then chat once so hooks run.",
                    dim=True,
                )
            )
            return

    result = init_cmd.apply_plan(plan, force=force)

    click.echo("")
    click.secho("agent-cot init — done", bold=True, fg="green")
    if result.hooks_backup:
        click.echo(f"  ✓ backed up    {result.hooks_backup}")
    if result.hooks_written:
        click.echo(f"  ✓ wrote        {result.hooks_written}")
    for p in result.scripts_installed:
        click.echo(f"  ✓ installed    {p}")
    if result.config_written:
        click.echo(f"  ✓ saved config {result.config_written}")
    # v0.20.4: surface the OTel env merge so the user sees exactly which
    # keys we added and which we preserved (their value, not our recommendation).
    if plan.otel_env_enabled:
        if plan.env_additions:
            for k, v in plan.env_additions:
                click.echo(click.style(f"  ✓ env added    {k} = {v}", fg="cyan"))
            click.echo(
                click.style(
                    "  ! Claude Code reads env at startup — fully quit Claude Code "
                    "and reopen for the new OTel env to take effect.",
                    fg="yellow",
                )
            )
        if plan.env_preserved:
            for k, v in plan.env_preserved:
                click.echo(
                    click.style(
                        f"  · env kept     {k} = {v}  (your value, never overwritten)",
                        dim=True,
                    )
                )
    click.echo("")
    click.echo("Next: `agent-cot start` to launch the local dashboard.")


@main.command("start", help="Start the local dashboard backend.")
@_agent_option()
@click.option(
    "--foreground",
    is_flag=True,
    help="Run uvicorn in this terminal (Ctrl-C to stop) instead of detaching.",
)
@click.option(
    "--no-browser",
    is_flag=True,
    help="Do not open a browser tab on success.",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Override the configured backend port (default: from config.toml).",
)
@click.option(
    "--health-timeout",
    type=float,
    default=15.0,
    show_default=True,
    help="Seconds to wait for the backend to answer /api/sessions before giving up.",
)
def cmd_start(
    agent_name: str,
    foreground: bool,
    no_browser: bool,
    port: int | None,
    health_timeout: float,
) -> None:
    from .commands import start as start_cmd

    _resolve_agent(agent_name)

    try:
        result = start_cmd.start_backend(
            port_override=port,
            foreground=foreground,
            open_browser=not no_browser,
            health_timeout=health_timeout,
        )
    except start_cmd.StartError as exc:
        _die(str(exc))

    click.secho("agent-cot start — running", bold=True, fg="green")
    click.echo(f"  pid          : {result.pid}")
    click.echo(f"  port         : {result.port}  (http://127.0.0.1:{result.port})")
    click.echo(f"  backend dir  : {result.backend_dir}")
    click.echo(f"  log          : {result.log_path}")
    click.echo(f"  pid file     : {result.pid_file}")
    # v0.18.5: 把 transcript_watcher 状态显式打出来 —— 同事老问"为啥没 tool"，
    # 这一行直接告诉他 watcher 是不是起来了 / 起不来的原因。
    if result.watcher_pid is not None:
        click.echo(
            click.style(
                f"  watcher      : pid={result.watcher_pid}  log={result.watcher_log_path}",
                fg="cyan",
            )
        )
    elif result.watcher_skip_reason:
        click.echo(
            click.style(
                f"  watcher      : SKIPPED — {result.watcher_skip_reason}",
                fg="yellow",
            )
        )
        click.echo(
            click.style(
                "                 (gap-tool capture disabled: Glob/Grep/Delete/Web*/"
                "Task/SemanticSearch/TodoWrite 等不会出现在 trace 里)",
                dim=True,
            )
        )
    # v0.19.0: report the self-heal pass so the user can confirm "yes,
    # the on-disk hooks are in sync with my freshly upgraded wheel".
    # Stays quiet when nothing happened (the common case after the very
    # first install, or every subsequent ``start`` on a healthy machine).
    if result.self_heal_scripts_refreshed is not None:
        n = result.self_heal_scripts_refreshed
        per_ide = result.self_heal_per_ide or {}
        per_ide_str = (
            " (" + ", ".join(f"{k}={v}" for k, v in per_ide.items()) + ")"
            if per_ide else ""
        )
        if n > 0:
            click.echo(
                click.style(
                    f"  self-heal    : refreshed {n} hook script(s){per_ide_str} "
                    f"(stale literals or missing files were healed)",
                    fg="cyan",
                )
            )
        else:
            click.echo(
                click.style(
                    f"  self-heal    : 0 hook script(s) needed refresh{per_ide_str} "
                    "(on-disk hooks already match installed wheel)",
                    dim=True,
                )
            )
    if result.self_heal_runtime_state is not None:
        click.echo(
            click.style(
                f"  runtime.json : {result.self_heal_runtime_state}",
                dim=True,
            )
        )
    if result.self_heal_skip_reason:
        click.echo(
            click.style(
                f"  self-heal    : PARTIAL — {result.self_heal_skip_reason}",
                fg="yellow",
            )
        )
    if result.auto_bootstrap_agents:
        click.echo(
            click.style(
                "  bootstrap    : hooks registered for "
                + ", ".join(result.auto_bootstrap_agents),
                fg="cyan",
            )
        )
    if result.auto_bootstrap_skip_reason:
        click.echo(
            click.style(
                f"  bootstrap    : PARTIAL - {result.auto_bootstrap_skip_reason}",
                fg="yellow",
            )
        )
    # v0.20.4: surface the Claude OTel endpoint self-heal so users
    # see (a) when we rewrote the URL after port eviction, and
    # (b) when the user's foreign endpoint is intentionally preserved.
    if result.otel_endpoint_self_heal == "updated":
        click.echo(
            click.style(
                f"  claude OTel  : endpoint rewritten "
                f"({result.otel_endpoint_self_heal_detail}). "
                "Fully quit Claude Code and reopen so the new env takes effect.",
                fg="cyan",
            )
        )
    elif result.otel_endpoint_self_heal == "foreign":
        click.echo(
            click.style(
                "  claude OTel  : user-managed endpoint detected — left untouched "
                "(your config wins; dashboard's ClaudeOtelPanel will stay empty).",
                dim=True,
            )
        )
    elif result.otel_endpoint_self_heal == "error":
        click.echo(
            click.style(
                f"  claude OTel  : self-heal FAILED — {result.otel_endpoint_self_heal_detail}",
                fg="yellow",
            )
        )
    if result.opened_browser:
        click.echo("  browser      : opened")
    else:
        click.echo(
            click.style(
                "  browser      : not opened (use the URL above)",
                dim=True,
            )
        )
    click.echo("")
    click.echo("Stop with `agent-cot stop`.")


@main.command("stop", help="Stop the local dashboard backend.")
@click.option(
    "--force",
    is_flag=True,
    help="Skip the cmdline-marker safety check before terminating.",
)
def cmd_stop(force: bool) -> None:
    from .commands import stop as stop_cmd

    try:
        result = stop_cmd.stop_backend(force=force)
    except stop_cmd.StopError as exc:
        _die(str(exc))

    if not result.found_pid_file:
        click.echo("backend not running (no PID file).")
        return
    if not result.was_running:
        if result.cleaned_pid_file:
            click.echo(
                click.style(
                    "removed stale PID file (process was already gone).",
                    fg="yellow",
                )
            )
        else:
            click.echo("nothing to stop.")
        return
    if result.terminated:
        click.secho(f"stopped pid={result.pid}", fg="green")
    else:
        _die(
            f"failed to stop pid={result.pid}; the process is still running. "
            "Try again with --force, or kill it manually."
        )

    # v0.18.5: 把 transcript_watcher daemon 的下场也打出来，避免用户对"watcher 还活着吗"产生疑惑。
    if result.watcher_found_pid_file:
        if result.watcher_was_running and result.watcher_terminated:
            click.secho(f"stopped watcher pid={result.watcher_pid}", fg="green")
        elif result.watcher_was_running and not result.watcher_terminated:
            click.echo(
                click.style(
                    f"warning: watcher pid={result.watcher_pid} did not exit cleanly "
                    "(see ~/.agent-cot/logs/watcher.log)",
                    fg="yellow",
                )
            )
        elif not result.watcher_was_running and result.watcher_cleaned_pid_file:
            click.echo(
                click.style(
                    "removed stale watcher PID file (process was already gone).",
                    fg="yellow",
                )
            )


@main.command("status", help="Show whether the dashboard is running.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def cmd_status(as_json: bool) -> None:
    from .commands import status as status_cmd

    report = status_cmd.collect_status()

    if as_json:
        import json as _json

        click.echo(_json.dumps(report.to_dict(), indent=2))
        return

    click.echo(status_cmd.render_human(report))


@main.command("doctor", help="Self-check: ports, dependencies, hook health.")
@click.option("--verbose", is_flag=True, help="Print every check, not just failures.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
@click.option(
    "--deep",
    is_flag=True,
    help=(
        "Run extra checks: ~/.agent-cot/runtime.json freshness, on-disk "
        "Cursor + CodeBuddy hook script staleness, recent cot.json "
        "tool-decision richness. Use this to debug 'pip install -U "
        "agent-cot didn't help' / 'colleague's UI is sparse' scenarios."
    ),
)
def cmd_doctor(verbose: bool, as_json: bool, deep: bool) -> None:
    from .commands import doctor as doctor_cmd

    sys.exit(doctor_cmd.run_doctor(verbose=verbose, as_json=as_json, deep=deep))


@main.group("otlp", help="Forward CoT data to any OTLP/HTTP backend.")
def cmd_otlp() -> None:
    """OTLP exporter — bridges to ``cot_otlp_exporter`` from cot-extractor."""


@cmd_otlp.command("list-presets", help="List built-in OTLP backend presets.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def cmd_otlp_list_presets(as_json: bool) -> None:
    from .commands import otlp as otlp_cmd

    sys.exit(otlp_cmd.run_list_presets(as_json=as_json))


@cmd_otlp.command("send", help="Export one session to an OTLP backend.")
@click.argument("session_id", required=False, default=None)
@click.option(
    "--cot-path",
    "cot_path",
    default=None,
    type=click.Path(dir_okay=False),
    help="Direct path to a cot.json (overrides SESSION_ID lookup).",
)
@click.option(
    "--preset",
    default=None,
    metavar="ID",
    help="Backend preset id (run `otlp list-presets`).",
)
@click.option("--endpoint", default=None, help="OTLP/HTTP traces URL.")
@click.option(
    "--header",
    "headers",
    multiple=True,
    metavar="K=V",
    help="HTTP header for the OTLP request (repeatable).",
)
@click.option(
    "--service-name",
    default="agent-cot",
    show_default=True,
    help="resource attribute service.name",
)
@click.option(
    "--service-version",
    default=None,
    help="resource attribute service.version (default: agent-cot version).",
)
@click.option(
    "--env",
    "environment",
    default=None,
    help="resource attribute deployment.environment",
)
@click.option(
    "--timeout",
    type=float,
    default=10.0,
    show_default=True,
    help="OTLP HTTP timeout (seconds).",
)
@click.option("--dry-run", is_flag=True, help="Render the span tree without sending.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def cmd_otlp_send(
    session_id: str | None,
    cot_path: str | None,
    preset: str | None,
    endpoint: str | None,
    headers: tuple[str, ...],
    service_name: str,
    service_version: str | None,
    environment: str | None,
    timeout: float,
    dry_run: bool,
    as_json: bool,
) -> None:
    from .commands import otlp as otlp_cmd

    if not session_id and not cot_path:
        _die("either SESSION_ID positional argument or --cot-path is required.")

    sys.exit(
        otlp_cmd.run_send(
            session_id=session_id,
            cot_path=cot_path,
            preset=preset,
            endpoint=endpoint,
            headers=list(headers),
            service_name=service_name,
            service_version=service_version,
            environment=environment,
            timeout=timeout,
            dry_run=dry_run,
            as_json=as_json,
        )
    )


@main.command(
    "export-trace",
    help="Export one session's full trace (thinking / tools / plan / permissions).",
)
@click.argument("session_id", required=False, default=None)
@click.option(
    "--cot-path",
    "cot_path",
    default=None,
    type=click.Path(dir_okay=False),
    help="Direct path to a cot.json (overrides SESSION_ID lookup).",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["jsonl", "json", "md"]),
    default="jsonl",
    show_default=True,
    help="jsonl is the canonical format; md is the human/LLM-readable view.",
)
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(),
    help="Output file or directory. Omit (or pass '-') to write to stdout.",
)
@click.option("--quiet", is_flag=True, help="Suppress the summary footer.")
def cmd_export_trace(
    session_id: str | None,
    cot_path: str | None,
    fmt: str,
    output: str | None,
    quiet: bool,
) -> None:
    from .commands import export_trace as export_cmd

    if not session_id and not cot_path:
        _die("either SESSION_ID positional argument or --cot-path is required.")

    sys.exit(
        export_cmd.run_export(
            session_id=session_id,
            cot_path=cot_path,
            fmt=fmt,
            output=output,
            quiet=quiet,
        )
    )


@main.command(
    "dedupe-thinking",
    help=(
        "Clean duplicate thinking steps out of already-captured cot.json files "
        "(Cursor's hook double-writes the same reasoning)."
    ),
)
@click.argument("session_id", required=False, default=None)
@click.option(
    "--cot-dir",
    "cot_dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Directory holding <sid>_cot.json. Defaults to the usual data root.",
)
@click.option(
    "--apply/--dry-run",
    "apply",
    default=False,
    show_default=True,
    help="Dry run by default — it rewrites captured data, so opt in explicitly.",
)
@click.option("--quiet", is_flag=True, help="Suppress per-file output and the footer.")
def cmd_dedupe_thinking(
    session_id: str | None,
    cot_dir: str | None,
    apply: bool,
    quiet: bool,
) -> None:
    from .commands import dedupe_thinking as dedupe_cmd

    sys.exit(
        dedupe_cmd.run_dedupe(
            session_id=session_id,
            cot_dir=cot_dir,
            apply=apply,
            quiet=quiet,
        )
    )


@main.command(
    "uninstall",
    help="Remove our hooks. Backs up hooks.json and keeps captured data by default.",
)
@_agent_option()
@click.option(
    "--dry-run/--apply",
    default=True,
    show_default=True,
    help="Plan-only mode (default) vs. actually writing to disk.",
)
@click.option(
    "--purge-data",
    is_flag=True,
    help="Also delete ~/.agent-cot (PID files, logs, captured CoT). "
    "Default keeps the data root intact.",
)
@click.option(
    "--restore-backup",
    is_flag=True,
    help="Restore the most recent hooks.json.bak.<ts> backup instead of "
    "running uninstall. Useful if init went wrong.",
)
def cmd_uninstall(
    agent_name: str,
    dry_run: bool,
    purge_data: bool,
    restore_backup: bool,
) -> None:
    from .commands import uninstall as uninstall_cmd

    adapter = _resolve_agent(agent_name)

    if restore_backup:
        listing = uninstall_cmd.list_available_backups(agent_name=agent_name)
        if not listing.backups:
            _die(
                f"no backups found next to {listing.target}. "
                "Backups are created by `agent-cot init --apply`."
            )

        click.echo(click.style("agent-cot uninstall — restore backup", bold=True))
        click.echo(f"target         : {listing.target}")
        click.echo(f"latest backup  : {listing.backups[0]}")
        if len(listing.backups) > 1:
            click.echo(
                click.style(
                    f"  (also available: {len(listing.backups) - 1} older backup(s))",
                    dim=True,
                )
            )

        if dry_run:
            click.echo("")
            click.echo(
                click.style(
                    "(dry-run; nothing written. Re-run with --apply to commit.)",
                    fg="yellow",
                )
            )
            return

        try:
            res = uninstall_cmd.restore(agent_name=agent_name)
        except uninstall_cmd.UninstallError as exc:
            _die(str(exc))

        click.secho("restore — done", bold=True, fg="green")
        click.echo(f"  ✓ restored from   {res.restored_from}")
        if res.pre_restore_backup:
            click.echo(f"  ✓ pre-restore bak {res.pre_restore_backup}")
        return

    try:
        ctx = uninstall_cmd.build_context(
            agent_name=agent_name,
            keep_data=not purge_data,
        )
    except AgentNotImplementedError as exc:
        _die(str(exc))

    click.echo(click.style("agent-cot uninstall — plan", bold=True))
    click.echo(f"agent          : {adapter.name}")
    click.echo(ctx.render_summary())

    if ctx.plan.is_noop:
        click.echo("")
        click.echo(
            click.style(
                "no changes to apply — our hooks are not installed.",
                fg="green",
            )
        )
        return

    if dry_run:
        click.echo("")
        click.echo(
            click.style(
                "(dry-run; nothing written. Re-run with --apply to commit.)",
                fg="yellow",
            )
        )
        return

    res = uninstall_cmd.apply(ctx)

    click.echo("")
    click.secho("agent-cot uninstall — done", bold=True, fg="green")
    if res.hooks_backup:
        click.echo(f"  ✓ backed up    {res.hooks_backup}")
    if res.hooks_written:
        click.echo(f"  ✓ rewrote      {res.hooks_written}")
    for p in res.scripts_deleted:
        click.echo(f"  ✓ deleted      {p}")
    if res.config_updated:
        click.echo(f"  ✓ updated cfg  {res.config_updated}")
    if res.data_root_deleted:
        click.echo(
            click.style(f"  ✗ purged       {res.data_root_deleted}", fg="yellow")
        )
    click.echo("")
    if not purge_data:
        click.echo(
            click.style(
                "  (~/.agent-cot kept — pass --purge-data to remove it too.)",
                dim=True,
            )
        )
    click.echo("Cursor will start without our hooks on its next launch.")


@main.command(
    "upgrade",
    help="Refresh bundled hook scripts in place (does not touch hooks.json).",
)
@_agent_option(default="all")  # v0.19.4 (P-13): default to all agents so
                               # that ``agent-cot upgrade --apply`` after
                               # a ``pip install -U`` actually refreshes
                               # CodeBuddy + Claude hooks too. Before this
                               # the silent default of ``cursor`` made the
                               # other two IDEs drift on every upgrade.
@click.option(
    "--dry-run/--apply",
    default=True,
    show_default=True,
    help="Plan-only mode (default) vs. actually writing to disk.",
)
def cmd_upgrade(agent_name: str, dry_run: bool) -> None:
    from .commands import upgrade as upgrade_cmd

    # v0.19.4 (P-13): ``all`` fans out across every installed agent.
    # Each agent gets an independent plan + apply pass; if any one
    # bombs (e.g. Claude not present yet), we keep going with the
    # rest and surface a per-agent footer at the end.
    if agent_name == "all":
        agents_to_run = list(list_agents())
    else:
        _resolve_agent(agent_name)
        agents_to_run = [agent_name]

    any_failure = False
    for idx, an in enumerate(agents_to_run):
        if len(agents_to_run) > 1:
            click.echo("")
            click.secho(f"=== upgrade · {an} ===", bold=True, fg="cyan")
        try:
            ctx = upgrade_cmd.build_context(agent_name=an)
        except AgentNotImplementedError as exc:
            click.secho(f"  skipped ({an}): {exc}", fg="yellow")
            continue
        except Exception as exc:
            click.secho(f"  failed ({an}): {exc}", fg="red")
            any_failure = True
            continue

        click.echo(click.style("agent-cot upgrade — plan", bold=True))
        click.echo(ctx.render_summary())

        if ctx.plan.is_noop:
            click.echo("")
            click.echo(
                click.style(
                    f"all hook scripts already match the bundled version. "
                    f"Nothing to upgrade for {an}.",
                    fg="green",
                )
            )
            continue

        if dry_run:
            click.echo("")
            click.echo(
                click.style(
                    "(dry-run; nothing written. Re-run with --apply to commit.)",
                    fg="yellow",
                )
            )
            continue

        _emit_upgrade_result(upgrade_cmd.apply(ctx))

    # Skip the legacy single-agent rendering below — we already
    # rendered everything inside the loop.
    if any_failure:
        raise SystemExit(1)


def _emit_upgrade_result(res) -> None:
    """v0.19.4 (P-13): shared upgrade-success renderer (used by every
    iteration of the per-agent loop above)."""
    click.echo("")
    click.secho("agent-cot upgrade — done", bold=True, fg="green")
    for p in res.scripts_replaced:
        click.echo(f"  ✓ replaced   {p}")
    for p in res.scripts_installed:
        click.echo(f"  ✓ installed  {p}")
    for b in res.backups:
        click.echo(click.style(f"  ↳ backup     {b}", dim=True))


# ---------------------------------------------------------------------------
# Final guard: turn unexpected adapter errors into clean exits
# ---------------------------------------------------------------------------


def _entrypoint() -> None:
    try:
        main()
    except CursorCotError as exc:
        _die(str(exc))


if __name__ == "__main__":
    _entrypoint()
