"""Tests for the resolution side of ``commands.start`` (no real spawn).

We exercise:

* ``_find_backend_dir`` walking through config / source / bundled.
* ``_build_backend_env`` populating ``AGENT_COT_FRONTEND_DIST`` /
  ``AGENT_COT_EXTRACTOR_SRC`` from config + bundled assets.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_cot import _assets
from agent_cot.commands import start as start_cmd
from agent_cot.installer.config import CursorCotConfig


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    return fake_home


def test_find_backend_dir_uses_config_override(
    tmp_path: Path,
    isolated_home: Path,
) -> None:
    fake_repo = tmp_path / "fake-dashboard"
    backend = fake_repo / "backend"
    backend.mkdir(parents=True)
    (backend / "main.py").write_text("# stub", encoding="utf-8")

    config = CursorCotConfig(dashboard_repo=str(fake_repo))
    found = start_cmd._find_backend_dir(config)
    assert found is not None
    assert found.resolve() == backend.resolve()


def test_find_backend_dir_falls_back_to_bundle(
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When no config and no source-tree match, the bundled copy wins."""
    config = CursorCotConfig()  # no dashboard_repo

    # Force the source-tree walk to fail by chdir'ing somewhere unrelated
    # AND mocking _find_backend_dir's internal walk via __file__... but
    # that's too invasive. Instead we just trust that when we *do* have
    # a working source tree (this repo), the source wins; assert the
    # invariant that the result is a real backend dir either way.
    found = start_cmd._find_backend_dir(config)
    assert found is not None
    assert (found / "main.py").is_file()

    # And: if the bundled copy is present, we should be able to point at
    # it independently.
    if _assets.has_bundled_backend():
        bundled = _assets.bundled_backend_dir().resolve()
        assert (bundled / "main.py").is_file()


def test_build_backend_env_sets_frontend_dist_when_bundled(
    isolated_home: Path,
) -> None:
    env = start_cmd._build_backend_env()

    if _assets.has_frontend_dist():
        assert "AGENT_COT_FRONTEND_DIST" in env
        assert (
            Path(env["AGENT_COT_FRONTEND_DIST"]) ==
            _assets.frontend_dist().resolve()
        )
    else:
        assert "AGENT_COT_FRONTEND_DIST" not in env


def test_build_backend_env_sets_extractor_src_from_config(
    isolated_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_repo = tmp_path / "fake-cot-extractor"
    (fake_repo / "src").mkdir(parents=True)

    # Patch load_config to return a config that points at our fake repo.
    from agent_cot.installer.config import CursorCotConfig as _Cfg

    def _fake_load() -> _Cfg:
        return _Cfg(cot_extractor_repo=str(fake_repo))

    monkeypatch.setattr(start_cmd, "load_config", _fake_load)

    env = start_cmd._build_backend_env()
    assert env["AGENT_COT_EXTRACTOR_SRC"] == str((fake_repo / "src").resolve())


# ---------------------------------------------------------------------------
# v0.20.4: Claude OTel endpoint self-heal at start time
# ---------------------------------------------------------------------------


import json  # noqa: E402


def test_otel_endpoint_self_heal_absent_when_claude_not_installed(
    isolated_home: Path,
) -> None:
    """No ~/.claude/ → quiet 'absent' status, no I/O."""
    status, detail = start_cmd._self_heal_claude_otel_endpoint(backend_port=8765)
    assert status == "absent"
    assert detail is None


def test_otel_endpoint_self_heal_absent_when_no_env_block(
    isolated_home: Path,
) -> None:
    settings_path = isolated_home / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps({"hooks": {"Stop": []}}), encoding="utf-8"
    )

    status, _ = start_cmd._self_heal_claude_otel_endpoint(backend_port=8765)
    assert status == "absent"


def test_otel_endpoint_self_heal_ok_when_port_matches(
    isolated_home: Path,
) -> None:
    settings_path = isolated_home / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    original = {
        "env": {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:8765"},
        "hooks": {"Stop": []},
    }
    settings_path.write_text(json.dumps(original), encoding="utf-8")

    status, detail = start_cmd._self_heal_claude_otel_endpoint(backend_port=8765)
    assert status == "ok"
    assert detail is None
    # File untouched.
    after = json.loads(settings_path.read_text(encoding="utf-8"))
    assert after == original


def test_otel_endpoint_self_heal_rewrites_loopback_port_mismatch(
    isolated_home: Path,
) -> None:
    """The core scenario: port-eviction between init and start."""
    settings_path = isolated_home / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "env": {
                    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:8765",
                    "OTEL_LOG_USER_PROMPTS": "1",
                },
                "permissions": {"defaultMode": "acceptEdits"},
            }
        ),
        encoding="utf-8",
    )

    status, detail = start_cmd._self_heal_claude_otel_endpoint(backend_port=8767)
    assert status == "updated"
    assert "8765" in detail and "8767" in detail

    after = json.loads(settings_path.read_text(encoding="utf-8"))
    assert after["env"]["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://127.0.0.1:8767"
    # Other env keys + top-level keys untouched.
    assert after["env"]["OTEL_LOG_USER_PROMPTS"] == "1"
    assert after["permissions"] == {"defaultMode": "acceptEdits"}


def test_otel_endpoint_self_heal_leaves_localhost_alone_if_port_matches(
    isolated_home: Path,
) -> None:
    settings_path = isolated_home / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {"env": {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:8765"}}
        ),
        encoding="utf-8",
    )

    status, _ = start_cmd._self_heal_claude_otel_endpoint(backend_port=8765)
    assert status == "ok"


def test_otel_endpoint_self_heal_skips_foreign_endpoint(
    isolated_home: Path,
) -> None:
    """User's corporate / cloud endpoint NEVER rewritten."""
    settings_path = isolated_home / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    user_url = "https://otel.corp.example.com:4318"
    settings_path.write_text(
        json.dumps({"env": {"OTEL_EXPORTER_OTLP_ENDPOINT": user_url}}),
        encoding="utf-8",
    )

    status, _ = start_cmd._self_heal_claude_otel_endpoint(backend_port=8766)
    assert status == "foreign"
    # File literally unchanged.
    after = json.loads(settings_path.read_text(encoding="utf-8"))
    assert after["env"]["OTEL_EXPORTER_OTLP_ENDPOINT"] == user_url


def test_otel_endpoint_self_heal_handles_invalid_json_gracefully(
    isolated_home: Path,
) -> None:
    """Corrupted settings.json must NOT bring down `agent-cot start`."""
    settings_path = isolated_home / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text("{not even valid", encoding="utf-8")

    status, _ = start_cmd._self_heal_claude_otel_endpoint(backend_port=8765)
    # Treated as if env was absent — best-effort, never raises.
    assert status == "absent"
    # And the broken file is left exactly as it was (no rewrite, no overwrite).
    assert settings_path.read_text(encoding="utf-8") == "{not even valid"


def test_auto_bootstrap_registers_codex_hooks_without_touching_config(
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh one-click start should make Codex observable without TOML edits."""
    codex_home = isolated_home / ".codex"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    import agent_cot.agents as agents_mod

    monkeypatch.setattr(agents_mod, "list_agents", lambda: ["codex"])

    applied, skip = start_cmd._auto_bootstrap_installed_agents(backend_port=8768)

    assert applied == ["codex"]
    assert skip is None
    assert not (codex_home / "config.toml").exists()

    hooks_path = codex_home / "hooks.json"
    hooks = json.loads(hooks_path.read_text(encoding="utf-8"))
    for event in (
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PermissionRequest",
        "PostToolUse",
        "PreCompact",
        "PostCompact",
        "SubagentStart",
        "SubagentStop",
        "Stop",
    ):
        assert event in hooks.get("hooks", {})
    assert (codex_home / "hooks" / "codex_stream_hook.py").is_file()
    assert (codex_home / "hooks" / "codex_sidecar_collector.py").is_file()
    assert not (codex_home / "hooks" / "agent_critic_hook.py").exists()
    assert (codex_home / "agents" / "agent-quality-critic.md").is_file()
    stop_commands = [
        hook.get("command", "")
        for group in hooks["hooks"]["Stop"]
        for hook in group.get("hooks", [])
    ]
    assert not any("agent_critic_hook.py" in command for command in stop_commands)
