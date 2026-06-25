"""Unit tests for runtime/pid_file.py."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from agent_cot.runtime.pid_file import (
    PID_RECORD_VERSION,
    PidFile,
    PidRecord,
    is_pid_running,
    read_pid_record,
    remove_pid_file,
    write_pid_record,
)

# ---------------------------------------------------------------------------
# PidRecord serialisation
# ---------------------------------------------------------------------------


def test_record_round_trip(tmp_path: Path) -> None:
    rec = PidRecord(
        pid=1234,
        port=8765,
        started_at=1234567890.0,
        log_path=str(tmp_path / "log"),
        cmdline_marker="agent-cot-backend",
    )
    p = write_pid_record(rec, tmp_path / "backend.pid")
    assert p.is_file()

    loaded = read_pid_record(p)
    assert loaded is not None
    assert loaded.pid == 1234
    assert loaded.port == 8765
    assert loaded.started_at == 1234567890.0
    assert loaded.cmdline_marker == "agent-cot-backend"
    assert loaded.schema == PID_RECORD_VERSION


def test_read_missing_file_returns_none(tmp_path: Path) -> None:
    assert read_pid_record(tmp_path / "does-not-exist.pid") is None


def test_read_corrupt_file_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "backend.pid"
    p.write_text("{not valid json", encoding="utf-8")
    assert read_pid_record(p) is None


def test_read_partial_schema_returns_none(tmp_path: Path) -> None:
    """Files missing required fields should not blow up."""
    p = tmp_path / "backend.pid"
    p.write_text('{"pid": "not-an-int"}', encoding="utf-8")
    assert read_pid_record(p) is None


def test_remove_pid_file_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "backend.pid"
    assert remove_pid_file(p) is False
    p.write_text("{}")
    assert remove_pid_file(p) is True
    assert remove_pid_file(p) is False


# ---------------------------------------------------------------------------
# PidFile facade
# ---------------------------------------------------------------------------


def test_pid_file_facade(tmp_path: Path) -> None:
    pid_path = tmp_path / "run" / "backend.pid"
    pf = PidFile(path=pid_path)
    assert not pf.exists()
    assert pf.read() is None

    rec = PidRecord(pid=4242, port=9000)
    pf.write(rec)
    assert pf.exists()

    loaded = pf.read()
    assert loaded is not None and loaded.pid == 4242

    assert pf.remove() is True
    assert not pf.exists()


# ---------------------------------------------------------------------------
# is_pid_running
# ---------------------------------------------------------------------------


def test_is_pid_running_rejects_invalid_pids() -> None:
    assert is_pid_running(0) is False
    assert is_pid_running(-1) is False
    assert is_pid_running(None) is False  # type: ignore[arg-type]


def test_is_pid_running_finds_self() -> None:
    """Our own PID is always running."""
    assert is_pid_running(os.getpid()) is True


def test_is_pid_running_rejects_unlikely_pid() -> None:
    """A very high PID is unlikely to exist."""
    # 2**31 - 1 is the kernel max on most systems, way above anything
    # typical CI / dev machines actually allocate.
    assert is_pid_running(2**31 - 1) is False


def test_is_pid_running_rejects_when_marker_missing() -> None:
    """Marker mismatch should treat the PID as 'not ours'."""
    # Self has python somewhere in cmdline but definitely not this
    # synthetic marker.
    assert (
        is_pid_running(
            os.getpid(),
            cmdline_marker="this-marker-does-not-exist-anywhere-12345",
        )
        is False
        # But on systems without psutil we can't verify the cmdline,
        # in which case is_pid_running returns True permissively.
        # That's checked separately via mocking.
        or _psutil_available() is False
    )


def test_is_pid_running_accepts_when_marker_present() -> None:
    """When marker is in cmdline, it's ours."""
    if not _psutil_available():
        pytest.skip("psutil not installed; cmdline matching not exercised")
    # ``python`` definitely appears in pytest's own cmdline.
    assert is_pid_running(os.getpid(), cmdline_marker="python") is True


def _psutil_available() -> bool:
    try:
        import psutil  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Stale-detection: a PID file pointing at a long-dead process
# ---------------------------------------------------------------------------


def test_stale_pid_detection(tmp_path: Path) -> None:
    """Even a fresh PID file is 'stale' if the PID is dead."""
    rec = PidRecord(
        pid=2**31 - 1,  # nonexistent
        port=8765,
        started_at=time.time(),
        cmdline_marker="agent-cot-backend",
    )
    write_pid_record(rec, tmp_path / "backend.pid")
    loaded = read_pid_record(tmp_path / "backend.pid")
    assert loaded is not None
    assert is_pid_running(loaded.pid, cmdline_marker=loaded.cmdline_marker) is False
