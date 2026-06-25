"""Tests for the v0.20.6 Claude dual-variant (OSS + Internal) write path.

Covers:

1. ClaudeAdapter.hooks_config_path() picks ``~/.claude-internal/`` when only
   it exists.
2. ClaudeAdapter.hooks_config_path() picks ``~/.claude/`` when both or
   only OSS exist (backward compatible).
3. ClaudeAdapter.additional_hooks_targets() returns the OTHER variant when
   both exist (and empty list otherwise).
4. ClaudeAdapter.detect_installed() returns True on either variant.
5. apply_plan() mirror writes — both settings.json get the same hooks +
   env block when dual install present.

These tests pin a contract that, if broken, means司内 Cursor-embedded Claude
Code users silently lose hooks again. Treat regressions here as P0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_cot.agents.claude import ClaudeAdapter
from agent_cot.commands.init import apply_plan, build_plan


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect Path.home() so the test never touches the real ~/.

    ClaudeAdapter's ``_CLAUDE_OSS_HOME`` / ``_CLAUDE_INTERNAL_HOME`` are
    properties that re-call Path.home() per-invocation, so a single
    monkeypatch on Path.home is sufficient (no need to also patch the
    adapter's class attrs).
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_primary_is_internal_when_only_internal_exists(isolated_home: Path) -> None:
    """Internal-only users (most司内 cases) get .claude-internal as primary."""
    _mkdir(isolated_home / ".claude-internal")
    adapter = ClaudeAdapter()
    assert adapter.hooks_config_path() == isolated_home / ".claude-internal" / "settings.json"
    assert adapter.hooks_assets_dir() == isolated_home / ".claude-internal" / "hooks"
    # No mirror — only one variant exists.
    assert adapter.additional_hooks_targets() == []


def test_primary_accepts_internal_compat_spelling(isolated_home: Path) -> None:
    """Some enterprise wrappers use ~/.claude-inertnal; keep it working."""
    _mkdir(isolated_home / ".claude-inertnal")
    adapter = ClaudeAdapter()
    assert adapter.hooks_config_path() == isolated_home / ".claude-inertnal" / "settings.json"
    assert adapter.hooks_assets_dir() == isolated_home / ".claude-inertnal" / "hooks"
    assert adapter.detect_installed() is True


def test_primary_is_oss_when_only_oss_exists(isolated_home: Path) -> None:
    """OSS-only users keep ~/.claude/ as primary (backward compatible)."""
    _mkdir(isolated_home / ".claude")
    adapter = ClaudeAdapter()
    assert adapter.hooks_config_path() == isolated_home / ".claude" / "settings.json"
    assert adapter.additional_hooks_targets() == []


def test_dual_install_mirrors_to_internal(isolated_home: Path) -> None:
    """When both exist, primary stays OSS for backward compat AND mirror
    targets .claude-internal so both get our hooks."""
    _mkdir(isolated_home / ".claude")
    _mkdir(isolated_home / ".claude-internal")
    adapter = ClaudeAdapter()
    assert adapter.hooks_config_path() == isolated_home / ".claude" / "settings.json"
    mirrors = adapter.additional_hooks_targets()
    assert len(mirrors) == 1
    settings, assets = mirrors[0]
    assert settings == isolated_home / ".claude-internal" / "settings.json"
    assert assets == isolated_home / ".claude-internal" / "hooks"


def test_detect_installed_accepts_either_variant(isolated_home: Path) -> None:
    """``--agent all`` must NOT skip Internal-only machines (pre-0.20.6 it did)."""
    adapter = ClaudeAdapter()
    # Neither installed → False.
    assert adapter.detect_installed() is False
    # OSS only → True.
    _mkdir(isolated_home / ".claude")
    assert adapter.detect_installed() is True
    # Reset and try Internal only.
    (isolated_home / ".claude").rmdir()
    _mkdir(isolated_home / ".claude-internal")
    assert adapter.detect_installed() is True


def test_apply_plan_mirror_writes_settings_json(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Full integration: build_plan + apply_plan dual-write contract.

    Set up both Claude variants, run init --apply --agent claude, then
    verify BOTH settings.json files contain our hooks + the OTel env block.
    """
    # We need a real ``hooks_dir()`` source for the bundled hook copy. Point
    # ``_assets.hooks_dir()`` at a tmp mirror containing a non-stub
    # claude_stream_hook.py so the install-time stub guard doesn't reject.
    from agent_cot import _assets

    fake_hooks = tmp_path / "fake_assets" / "hooks"
    (fake_hooks / "claude").mkdir(parents=True)
    # 2 KB of real-looking Python so the stub guard (< 1 KB) passes.
    for name in ("claude_stream_hook.py",):
        (fake_hooks / "claude" / name).write_text(
            "# fake hook for unit test\n" + ("# pad line\n" * 200),
            encoding="utf-8",
        )
    monkeypatch.setattr(_assets, "hooks_dir", lambda: fake_hooks)

    # Set up both Claude home dirs.
    _mkdir(isolated_home / ".claude")
    _mkdir(isolated_home / ".claude-internal")
    # Pre-existing entries in .claude-internal that we must not clobber
    # (simulates codebuddy-mem hooks the user already installed).
    pre_existing_internal = {
        "hooks": {
            "Stop": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "node /opt/codebuddy-mem/stop.js",
                        }
                    ]
                }
            ]
        },
        "env": {
            "MY_PRESET_KEY": "user_value",
        },
    }
    (isolated_home / ".claude-internal" / "settings.json").write_text(
        json.dumps(pre_existing_internal, indent=2),
        encoding="utf-8",
    )

    plan = build_plan(agent_name="claude", port_backend=8765, write_otel_env=True)
    assert len(plan.additional_targets) == 1, "dual install must yield 1 mirror target"

    apply_plan(plan)

    primary = json.loads(
        (isolated_home / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    mirror = json.loads(
        (isolated_home / ".claude-internal" / "settings.json").read_text(encoding="utf-8")
    )

    # 1. Both have our hooks.
    assert "Stop" in primary.get("hooks", {})
    assert "Stop" in mirror.get("hooks", {})
    primary_stop_cmds = [
        h.get("command", "")
        for group in primary["hooks"]["Stop"]
        for h in (group.get("hooks") or [])
    ]
    mirror_stop_cmds = [
        h.get("command", "")
        for group in mirror["hooks"]["Stop"]
        for h in (group.get("hooks") or [])
    ]
    assert any("claude_stream_hook.py" in c for c in primary_stop_cmds)
    assert any("claude_stream_hook.py" in c for c in mirror_stop_cmds)
    assert not any("agent_critic_hook.py" in c for c in primary_stop_cmds)
    assert not any("agent_critic_hook.py" in c for c in mirror_stop_cmds)
    assert any(".claude/hooks/claude_stream_hook.py" in c.replace("\\", "/") for c in primary_stop_cmds)
    assert any(".claude-internal/hooks/claude_stream_hook.py" in c.replace("\\", "/") for c in mirror_stop_cmds)

    # 2. Mirror PRESERVED the pre-existing codebuddy-mem entry.
    assert any("codebuddy-mem" in c for c in mirror_stop_cmds), (
        "mirror write must merge into existing settings, not overwrite — "
        "codebuddy-mem entry was wiped"
    )

    # 3. Both have the OTel env block.
    assert "CLAUDE_CODE_ENABLE_TELEMETRY" in primary.get("env", {})
    assert "CLAUDE_CODE_ENABLE_TELEMETRY" in mirror.get("env", {})
    # Mirror preserved user's pre-existing env key.
    assert mirror["env"].get("MY_PRESET_KEY") == "user_value"

    # 4. Hook scripts copied to BOTH hooks/ dirs.
    assert (isolated_home / ".claude" / "hooks" / "claude_stream_hook.py").is_file()
    assert (isolated_home / ".claude-internal" / "hooks" / "claude_stream_hook.py").is_file()
    assert not (isolated_home / ".claude" / "hooks" / "agent_critic_hook.py").exists()
    assert not (isolated_home / ".claude-internal" / "hooks" / "agent_critic_hook.py").exists()
