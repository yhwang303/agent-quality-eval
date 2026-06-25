"""Tests for ``agent_cot.doctor``.

We focus on the *contract* of doctor:

* Every individual check is wrapped in ``_safe`` and never raises.
* ``run_all`` returns a stable, non-empty report.
* ``DoctorReport.overall_status`` is the worst of (FAIL, WARN, OK).
* SKIP never affects the overall verdict.
* ``commands.doctor`` exit code maps to overall status.

We do **not** assert specific check-by-check status here, because
that depends on the developer machine. Instead we assert *shape*.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_cot.cli import main
from agent_cot.commands import doctor as doctor_cmd
from agent_cot.doctor import CheckStatus, run_all
from agent_cot.doctor.checks import (
    Check,
    _safe,
    check_python_version,
)
from agent_cot.doctor.runner import DoctorReport


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pin ``Path.home`` to a tmp dir so we don't probe the dev's real
    ``~/.agent-cot`` / ``~/.cursor`` and don't write anything either."""
    fake = tmp_path / "home"
    fake.mkdir()
    monkeypatch.setenv("HOME", str(fake))
    monkeypatch.setenv("USERPROFILE", str(fake))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake))
    return fake


# ---------------------------------------------------------------------------
# Per-check shape
# ---------------------------------------------------------------------------


def test_safe_decorator_swallows_exceptions() -> None:
    @_safe
    def boom() -> Check:
        raise RuntimeError("kapow")

    out = boom()
    assert isinstance(out, Check)
    assert out.status is CheckStatus.FAIL
    assert "kapow" in out.message


def test_python_version_check_is_ok_on_test_runner() -> None:
    c = check_python_version()
    assert c.status is CheckStatus.OK
    assert c.message.startswith("Python ")


# ---------------------------------------------------------------------------
# run_all / DoctorReport
# ---------------------------------------------------------------------------


def test_run_all_returns_non_empty_report(isolated_home: Path) -> None:
    rep = run_all()
    assert isinstance(rep, DoctorReport)
    assert len(rep.checks) >= 8  # python + agent-cot + at least a handful of others
    # No check ever raises.
    for c in rep.checks:
        assert isinstance(c, Check)
        assert c.status in CheckStatus


def test_overall_status_is_worst_of_all(isolated_home: Path) -> None:
    rep = DoctorReport(
        checks=[
            Check("a", CheckStatus.OK, "fine"),
            Check("b", CheckStatus.WARN, "meh"),
            Check("c", CheckStatus.OK, "fine"),
        ]
    )
    assert rep.overall_status is CheckStatus.WARN

    rep2 = DoctorReport(
        checks=[
            Check("a", CheckStatus.OK, "fine"),
            Check("b", CheckStatus.WARN, "meh"),
            Check("c", CheckStatus.FAIL, "ouch"),
        ]
    )
    assert rep2.overall_status is CheckStatus.FAIL


def test_skip_does_not_affect_overall_status() -> None:
    rep = DoctorReport(
        checks=[
            Check("a", CheckStatus.OK, "fine"),
            Check("b", CheckStatus.SKIP, "stub"),
        ]
    )
    assert rep.overall_status is CheckStatus.OK


def test_to_dict_roundtrip(isolated_home: Path) -> None:
    rep = run_all()
    d = rep.to_dict()
    assert d["overall_status"] in {s.value for s in CheckStatus}
    assert "counts" in d and "checks" in d
    # counts sum equals number of checks.
    assert sum(d["counts"].values()) == len(d["checks"])


# ---------------------------------------------------------------------------
# CLI mapping
# ---------------------------------------------------------------------------


def test_run_doctor_emits_overall_line_and_exits_correctly(
    isolated_home: Path,
) -> None:
    runner = CliRunner()
    res = runner.invoke(main, ["doctor"])  # default: human, non-verbose
    # Exit code 0/1/2 — never crash.
    assert res.exit_code in (0, 1, 2), res.output
    assert "overall" in res.output


def test_run_doctor_json(isolated_home: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(main, ["doctor", "--json"])
    assert res.exit_code in (0, 1, 2)
    import json

    parsed = json.loads(res.output)
    assert parsed["overall_status"] in {s.value for s in CheckStatus}
    assert isinstance(parsed["checks"], list) and parsed["checks"]


def test_exit_code_map() -> None:
    fail = DoctorReport(checks=[Check("x", CheckStatus.FAIL, "bad")])
    warn = DoctorReport(checks=[Check("x", CheckStatus.WARN, "meh")])
    ok = DoctorReport(checks=[Check("x", CheckStatus.OK, "fine")])
    assert doctor_cmd._exit_code_for(fail.overall_status) == 2
    assert doctor_cmd._exit_code_for(warn.overall_status) == 1
    assert doctor_cmd._exit_code_for(ok.overall_status) == 0


# ---------------------------------------------------------------------------
# v0.20.4: check_claude_otel_env state coverage
# ---------------------------------------------------------------------------


def test_claude_otel_env_check_skips_when_no_claude(isolated_home: Path) -> None:
    from agent_cot.doctor.checks import check_claude_otel_env

    check = check_claude_otel_env()
    assert check.status == CheckStatus.SKIP


def test_claude_otel_env_check_warns_when_no_env_block(isolated_home: Path) -> None:
    import json as _json

    from agent_cot.doctor.checks import check_claude_otel_env

    settings = isolated_home / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(_json.dumps({"hooks": {"Stop": []}}), encoding="utf-8")

    check = check_claude_otel_env()
    assert check.status == CheckStatus.WARN
    assert "env" in check.message.lower()


def test_claude_otel_env_check_warns_when_protocol_is_protobuf(
    isolated_home: Path,
) -> None:
    """Our receiver only speaks http/json; protobuf payloads get rejected."""
    import json as _json

    from agent_cot.doctor.checks import check_claude_otel_env

    settings = isolated_home / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        _json.dumps(
            {
                "env": {
                    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
                    "OTEL_LOGS_EXPORTER": "otlp",
                    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
                    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:8765",
                }
            }
        ),
        encoding="utf-8",
    )

    check = check_claude_otel_env()
    assert check.status == CheckStatus.WARN
    assert "protobuf" in check.message.lower() or "json" in check.message.lower()


def test_claude_otel_env_check_warns_on_port_mismatch(
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json as _json

    from agent_cot.doctor import checks as checks_mod

    # Pin config.backend_port via fake load_config.
    from agent_cot.installer.config import CursorCotConfig

    monkeypatch.setattr(
        checks_mod, "load_config", lambda: CursorCotConfig(backend_port=8766)
    )

    settings = isolated_home / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        _json.dumps(
            {
                "env": {
                    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
                    "OTEL_LOGS_EXPORTER": "otlp",
                    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/json",
                    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:8765",
                }
            }
        ),
        encoding="utf-8",
    )

    check = checks_mod.check_claude_otel_env()
    assert check.status == CheckStatus.WARN
    assert "port" in check.message.lower()


def test_claude_otel_env_check_ok_on_match(
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json as _json

    from agent_cot.doctor import checks as checks_mod

    from agent_cot.installer.config import CursorCotConfig

    monkeypatch.setattr(
        checks_mod, "load_config", lambda: CursorCotConfig(backend_port=8765)
    )

    settings = isolated_home / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        _json.dumps(
            {
                "env": {
                    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
                    "OTEL_LOGS_EXPORTER": "otlp",
                    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/json",
                    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:8765",
                }
            }
        ),
        encoding="utf-8",
    )

    check = checks_mod.check_claude_otel_env()
    assert check.status == CheckStatus.OK


def test_claude_otel_env_check_ok_when_foreign_endpoint(
    isolated_home: Path,
) -> None:
    """User explicitly aimed at corp / langfuse / phoenix — we acknowledge, never warn."""
    import json as _json

    from agent_cot.doctor.checks import check_claude_otel_env

    settings = isolated_home / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        _json.dumps(
            {
                "env": {
                    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
                    "OTEL_LOGS_EXPORTER": "otlp",
                    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/json",
                    "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otel.corp.example.com:4318",
                }
            }
        ),
        encoding="utf-8",
    )

    check = check_claude_otel_env()
    assert check.status == CheckStatus.OK
    assert "user-managed" in check.message.lower()
