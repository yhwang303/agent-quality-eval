"""Tests for ``commands.stop`` covering each branch of the decision matrix."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_cot.cli import main
from agent_cot.commands import stop as stop_cmd
from agent_cot.runtime.pid_file import PidFile, PidRecord


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    return fake_home


def test_stop_when_no_pid_file(isolated_home: Path) -> None:
    result = stop_cmd.stop_backend()
    assert result.found_pid_file is False
    assert result.terminated is True
    assert result.was_running is False


def test_stop_with_stale_pid_file(isolated_home: Path) -> None:
    """Stale PID file gets cleaned up; we don't kill anything."""
    pid_file = PidFile.default()
    pid_file.write(
        PidRecord(
            pid=2**31 - 1,
            port=9999,
            cmdline_marker="agent-cot-backend",
        )
    )
    assert pid_file.exists()

    result = stop_cmd.stop_backend()
    assert result.found_pid_file is True
    assert result.was_running is False
    assert result.terminated is True
    assert result.cleaned_pid_file is True
    assert not pid_file.exists()


def test_stop_with_corrupt_pid_file(isolated_home: Path) -> None:
    """Corrupt JSON should be wiped without crashing."""
    pid_file = PidFile.default()
    pid_file.path.parent.mkdir(parents=True, exist_ok=True)
    pid_file.path.write_text("{not json", encoding="utf-8")
    assert pid_file.exists()

    result = stop_cmd.stop_backend()
    assert result.cleaned_pid_file is True
    assert not pid_file.exists()


def test_cli_stop_smoke(isolated_home: Path) -> None:
    result = CliRunner().invoke(main, ["stop"])
    assert result.exit_code == 0, result.output
    assert "not running" in result.output.lower()
