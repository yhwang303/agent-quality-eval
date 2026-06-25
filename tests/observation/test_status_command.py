"""Smoke-ish tests for ``commands.status``.

We isolate ``Path.home`` to a tmp dir so the test doesn't pick up a
real PID file on the developer machine.
"""

from __future__ import annotations

import json as _json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_cot.cli import main
from agent_cot.commands import status as status_cmd
from agent_cot.runtime.pid_file import PidFile, PidRecord


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    return fake_home


def test_status_clean_machine_says_not_running(
    isolated_home: Path,
) -> None:
    report = status_cmd.collect_status()
    assert report.backend_running is False
    assert report.backend_pid is None
    assert report.backend_health_ok is False
    assert report.installed_agents == []
    assert report.cot_extractor_repo is None


def test_status_cli_human_output(isolated_home: Path) -> None:
    result = CliRunner().invoke(main, ["status"])
    assert result.exit_code == 0, result.output
    assert "backend" in result.output.lower()
    assert "not running" in result.output.lower()


def test_status_cli_json_output(isolated_home: Path) -> None:
    result = CliRunner().invoke(main, ["status", "--json"])
    assert result.exit_code == 0, result.output
    payload = _json.loads(result.output)
    assert payload["backend_running"] is False
    assert "pid_file" in payload
    assert "config_file" in payload


def test_status_with_stale_pid_file(isolated_home: Path) -> None:
    """A PID file pointing at a dead process should report 'stale'."""
    pid_file = PidFile.default()
    pid_file.write(
        PidRecord(
            pid=2**31 - 1,
            port=9999,
            cmdline_marker="agent-cot-backend",
        )
    )

    report = status_cmd.collect_status()
    assert report.backend_running is False
    assert report.backend_last_error is not None
    assert "stale" in report.backend_last_error.lower()
