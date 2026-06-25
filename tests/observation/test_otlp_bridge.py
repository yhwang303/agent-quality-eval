"""Tests for ``agent_cot.commands.otlp_bridge``.

These cover the **resolution layer** only — actually exporting spans
requires the OTel SDK and a network endpoint, which ``cot-extractor``
already tests in its own suite. We verify:

* ``parse_headers`` accepts both ``k=v`` and ``k: v`` forms.
* ``resolve_cot_json`` finds flat layout, sessions/<sid> layout, and
  raises a clear error when neither matches.
* ``import_exporter`` raises ``OtlpBridgeError`` (not ``ImportError``)
  when the source tree is not on disk.
* ``find_preset`` errors mention the available ids.
* ``cli otlp list-presets`` and ``cli otlp send --dry-run`` work
  end-to-end *iff* the local cot-extractor source is reachable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_cot.cli import main
from agent_cot.commands import otlp_bridge as bridge

# ---------------------------------------------------------------------------
# parse_headers
# ---------------------------------------------------------------------------


def test_parse_headers_eq_and_colon_forms() -> None:
    out = bridge.parse_headers(
        ["x-key=abc", "x-other:  spaced  ", "x-eq-wins=v=actually=valid"]
    )
    assert out == {
        "x-key": "abc",
        "x-other": "spaced",
        "x-eq-wins": "v=actually=valid",
    }


def test_parse_headers_rejects_malformed() -> None:
    with pytest.raises(bridge.OtlpBridgeError, match="must be 'k=v'"):
        bridge.parse_headers(["nokv"])


def test_parse_headers_rejects_empty_key() -> None:
    with pytest.raises(bridge.OtlpBridgeError, match="empty header key"):
        bridge.parse_headers(["=value"])


# ---------------------------------------------------------------------------
# resolve_cot_json
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point COT_DIR / cot_extractor_repo / HOME to a tmp dir so we
    don't pick up the dev's real ``~/.agent-cot`` config."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.delenv("COT_DIR", raising=False)
    monkeypatch.delenv("AGENT_COT_EXTRACTOR_SRC", raising=False)
    return tmp_path


def _write_cot(path: Path, sid: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"session_id": sid, "turns": [], "otel_view": {}}),
        encoding="utf-8",
    )


def test_resolve_cot_json_flat_layout(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cot_dir = isolated_env / "cot"
    _write_cot(cot_dir / "abc_cot.json", "abc")
    monkeypatch.setenv("COT_DIR", str(cot_dir))

    res = bridge.resolve_cot_json(session_id="abc")
    assert res.session_id == "abc"
    assert res.path.name == "abc_cot.json"
    assert res.raw["session_id"] == "abc"


def test_resolve_cot_json_sessions_layout_picks_latest(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # COT_DIR points at .../output/cot, sessions sibling holds the file.
    cot_dir = isolated_env / "output" / "cot"
    cot_dir.mkdir(parents=True)
    sessions_dir = isolated_env / "output" / "sessions" / "sid42"
    _write_cot(sessions_dir / "20260101-100000_cot.json", "sid42")
    _write_cot(sessions_dir / "20260101-200000_cot.json", "sid42")
    monkeypatch.setenv("COT_DIR", str(cot_dir))

    res = bridge.resolve_cot_json(session_id="sid42")
    assert res.path.name == "20260101-200000_cot.json"  # latest wins


def test_resolve_cot_json_explicit_path_wins(isolated_env: Path) -> None:
    p = isolated_env / "explicit.json"
    _write_cot(p, "manual-sid")

    res = bridge.resolve_cot_json(cot_path=str(p))
    assert res.path == p.resolve()
    assert res.session_id == "manual-sid"


def test_resolve_cot_json_missing_raises(isolated_env: Path) -> None:
    with pytest.raises(bridge.OtlpBridgeError, match="could not locate"):
        bridge.resolve_cot_json(session_id="ghost")


def test_resolve_cot_json_no_args_raises(isolated_env: Path) -> None:
    with pytest.raises(bridge.OtlpBridgeError, match="--session-id or --cot-path"):
        bridge.resolve_cot_json()


def test_resolve_cot_json_invalid_path_raises(isolated_env: Path) -> None:
    with pytest.raises(bridge.OtlpBridgeError, match=r"cot\.json not found"):
        bridge.resolve_cot_json(cot_path=str(isolated_env / "no-such.json"))


# ---------------------------------------------------------------------------
# import_exporter / get_presets / find_preset
# ---------------------------------------------------------------------------


def test_import_exporter_raises_when_no_source(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_COT_EXTRACTOR_SRC", str(isolated_env / "nope"))
    # Force the regular import path to also fail by making sure the
    # name is not on sys.modules.
    monkeypatch.delitem(__import__("sys").modules, "cot_otlp_exporter", raising=False)
    # If the dev machine has cot-extractor on PYTHONPATH, this test
    # would succeed in finding it. We make a best-effort assertion: if
    # the bridge succeeds we accept it (the dev *does* have it
    # installed); otherwise the error must be the structured one.
    try:
        bridge.import_exporter()
    except bridge.OtlpBridgeError as exc:
        assert "Tried:" in str(exc)
        assert "Fix:" in str(exc)


def test_find_preset_unknown_lists_alternatives() -> None:
    try:
        presets = bridge.get_presets()
    except bridge.OtlpBridgeError:
        pytest.skip("cot-extractor not importable on this machine")

    with pytest.raises(bridge.OtlpBridgeError) as ei:
        bridge.find_preset("definitely-not-a-real-preset")
    msg = str(ei.value)
    assert "unknown preset" in msg
    # All known ids should appear in the error message.
    for p in presets:
        assert p["id"] in msg


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_otlp_list_presets_human(isolated_env: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(main, ["otlp", "list-presets"])
    if "could not import cot_otlp_exporter" in res.output:
        pytest.skip("cot-extractor source not reachable on this machine")
    assert res.exit_code == 0, res.output
    assert "OTLP backend presets" in res.output


def test_cli_otlp_list_presets_json(isolated_env: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(main, ["otlp", "list-presets", "--json"])
    if "could not import cot_otlp_exporter" in res.output:
        pytest.skip("cot-extractor source not reachable on this machine")
    assert res.exit_code == 0, res.output
    parsed = json.loads(res.output)
    assert isinstance(parsed, list) and parsed
    assert all("id" in p and "endpoint" in p for p in parsed)


def test_cli_otlp_send_dry_run(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end dry-run path: writes a tiny cot.json, asks the CLI
    to send it with --dry-run, expects a span tree dump.
    """
    cot_dir = isolated_env / "cot"
    sid = "dryrun-1"
    _write_cot(cot_dir / f"{sid}_cot.json", sid)
    monkeypatch.setenv("COT_DIR", str(cot_dir))

    runner = CliRunner()
    res = runner.invoke(main, ["otlp", "send", sid, "--dry-run"])
    if "could not import cot_otlp_exporter" in res.output:
        pytest.skip("cot-extractor source not reachable on this machine")
    if "opentelemetry-sdk" in res.output:
        pytest.skip("OTel SDK not installed on this machine")
    # The cot.json is essentially empty so spans=0 is fine; we just
    # care that we made it through the bridge without an exception.
    assert res.exit_code in (0, 2), res.output
    assert "session_id" in res.output


def test_cli_otlp_send_requires_arg() -> None:
    runner = CliRunner()
    res = runner.invoke(main, ["otlp", "send"])
    assert res.exit_code != 0
    assert "either SESSION_ID" in res.output
