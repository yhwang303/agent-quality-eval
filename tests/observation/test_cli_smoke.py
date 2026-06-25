"""P0 smoke tests: the CLI is importable, --help / --version work, and
every stub command exits non-zero with a friendly message rather than a
stack trace.
"""

from __future__ import annotations

from click.testing import CliRunner

from agent_cot import __version__
from agent_cot.cli import main


def test_version_flag() -> None:
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_flag_lists_subcommands() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ("init", "start", "stop", "status", "doctor", "otlp", "agents"):
        assert cmd in result.output, f"missing subcommand in --help: {cmd}"


def test_agents_subcommand_lists_cursor_and_claude() -> None:
    result = CliRunner().invoke(main, ["agents"])
    assert result.exit_code == 0
    assert "Cursor" in result.output
    # v0.19.1+: claude adapter is rebranded "Claude Internal" (the
    # internal-only Claude Code build) — accept either label so a
    # downstream rename back to "Claude Code" stays green too.
    assert ("Claude Internal" in result.output) or ("Claude Code" in result.output)


def test_agents_subcommand_json() -> None:
    result = CliRunner().invoke(main, ["agents", "--json"])
    assert result.exit_code == 0
    assert '"name": "cursor"' in result.output
    assert '"name": "claude"' in result.output


def test_init_with_unknown_agent_dies_cleanly() -> None:
    # CliRunner defaults to ``mix_stderr=True``, which means stderr is
    # captured into ``.output``. We rely on that here so the assertions
    # work uniformly across click 8.x and forward.
    result = CliRunner().invoke(main, ["init", "--agent", "nope"])
    assert result.exit_code != 0
    assert "Unknown agent" in result.output
    assert "Traceback" not in result.output


def test_init_with_claude_runs_dry_run_successfully() -> None:
    """v0.19.1+: Claude Internal is no longer a stub.

    ``agent-cot init --agent claude`` should produce a dry-run plan
    summary (exit 0, no traceback) just like the cursor path —
    ``settings.json`` is *not* touched without ``--apply``.
    """
    result = CliRunner().invoke(main, ["init", "--agent", "claude"])
    assert result.exit_code == 0, result.output
    assert "agent" in result.output.lower()
    assert "Traceback" not in result.output


def test_init_with_cursor_dry_run_succeeds() -> None:
    # P1: cursor init now runs in dry-run mode by default and exits 0
    # with a plan summary; nothing on disk should be modified.
    result = CliRunner().invoke(main, ["init", "--agent", "cursor"])
    assert result.exit_code == 0, result.output
    assert "agent           : cursor" in result.output
    assert "backend port" in result.output
    assert "hooks.json changes:" in result.output
    assert "dry-run" in result.output
    assert "Traceback" not in result.output


def test_start_help_mentions_lifecycle_flags() -> None:
    """P2: --help should advertise the new lifecycle controls."""
    result = CliRunner().invoke(main, ["start", "--help"])
    assert result.exit_code == 0, result.output
    for flag in ("--foreground", "--no-browser", "--port", "--health-timeout"):
        assert flag in result.output, f"missing flag in start --help: {flag}"


def test_stop_help_mentions_force() -> None:
    result = CliRunner().invoke(main, ["stop", "--help"])
    assert result.exit_code == 0, result.output
    assert "--force" in result.output


def test_status_help_mentions_json() -> None:
    result = CliRunner().invoke(main, ["status", "--help"])
    assert result.exit_code == 0, result.output
    assert "--json" in result.output


def test_doctor_help_mentions_verbose_and_json() -> None:
    """P4: doctor advertises --verbose and --json, no stub message."""
    result = CliRunner().invoke(main, ["doctor", "--help"])
    assert result.exit_code == 0, result.output
    assert "--verbose" in result.output
    assert "--json" in result.output
    assert "stub" not in result.output


def test_otlp_help_lists_subcommands() -> None:
    result = CliRunner().invoke(main, ["otlp", "--help"])
    assert result.exit_code == 0, result.output
    assert "list-presets" in result.output
    assert "send" in result.output


def test_uninstall_help_mentions_dry_run_and_restore() -> None:
    """P5: uninstall is no longer a stub; --dry-run / --apply / --restore-backup."""
    result = CliRunner().invoke(main, ["uninstall", "--help"])
    assert result.exit_code == 0, result.output
    for flag in ("--dry-run", "--apply", "--purge-data", "--restore-backup"):
        assert flag in result.output, f"missing flag in uninstall --help: {flag}"
    assert "stub" not in result.output


def test_upgrade_help_mentions_dry_run() -> None:
    result = CliRunner().invoke(main, ["upgrade", "--help"])
    assert result.exit_code == 0, result.output
    for flag in ("--dry-run", "--apply"):
        assert flag in result.output, f"missing flag in upgrade --help: {flag}"
    assert "stub" not in result.output


def test_otlp_send_help_advertises_flags() -> None:
    result = CliRunner().invoke(main, ["otlp", "send", "--help"])
    assert result.exit_code == 0, result.output
    for flag in (
        "--preset",
        "--endpoint",
        "--header",
        "--service-name",
        "--dry-run",
        "--cot-path",
        "--timeout",
    ):
        assert flag in result.output, f"missing flag in otlp send --help: {flag}"
