"""Integration-ish tests for ``commands.init`` — the orchestration layer.

These tests use ``tmp_path`` to redirect every disk-touching primitive,
so that running ``pytest`` on a real developer machine NEVER writes to
``~/.cursor/hooks.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_cot.cli import main
from agent_cot.commands import init as init_cmd

# Fake hook body used by every test fixture below. Padded past the 1 KB
# stub-guard threshold inside ``apply_plan`` (v0.20.5) so install-time
# refuses real stubs but accepts these test fakes. Real Cursor hooks
# ship at 16-18 KB; CodeBuddy / Claude at 18-26 KB; this 2 KB lorem is
# fine for fixture purposes and keeps the COT_ROOT default token intact
# so ``_patch_cot_root_default`` still has something to substitute.
_FAKE_HOOK_BODY = (
    "// agent-cot test fixture - padded above the 1 KB stub-guard\n"
    "const COT_ROOT = process.env.COT_EXTRACTOR_ROOT || 'OLD';\n"
    "// "
    + ("padding " * 400)
    + "\n"
)
assert len(_FAKE_HOOK_BODY) >= 1024, "fixture must clear the stub-guard"


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect both ``Path.home()`` and the assets resolver into ``tmp_path``.

    This is what makes init tests safe to run on a real dev box.

    v0.20.5 critical fix
    --------------------
    Prior versions monkey-patched ``Path.home()`` but left
    ``_assets.hooks_dir()`` resolving against the *real source tree* —
    ``src/agent_cot/assets/hooks/``. Tests downstream then did::

        asset_root = _assets.hooks_dir() / "cursor"
        if not (asset_root / "cot-stream.js").is_file():
            (asset_root / "cot-stream.js").write_text("...stub...")

    The first time that test ran on a freshly checked-out tree (or after
    a maintainer accidentally cleaned the build cache), it would land
    a 59-byte stub inside the **source tree**, which then propagated
    straight into the next wheel build — which is exactly what bricked
    Cursor adapter in 0.20.4 (stub hook → Cursor fires hook → node runs
    a one-line script → events.jsonl never written).

    The fix: re-point ``_assets.hooks_dir()`` itself at a per-test
    isolated location and pre-create the empty agent sub-dirs there.
    Tests can keep writing stubs to ``_assets.hooks_dir() / <agent>``
    exactly as before — but those writes now land in tmp_path, never
    in src/.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    # Re-point assets/hooks/ at a tmp_path mirror so test fixtures
    # writing stub hooks can't reach the real source tree.
    from agent_cot import _assets

    isolated_hooks = tmp_path / "assets" / "hooks"
    isolated_hooks.mkdir(parents=True, exist_ok=True)
    for agent_name in ("cursor", "codebuddy", "claude", "vscode"):
        (isolated_hooks / agent_name).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_assets, "hooks_dir", lambda: isolated_hooks)
    return fake_home


def test_build_plan_on_empty_machine_has_warnings(
    isolated_home: Path,
) -> None:
    plan = init_cmd.build_plan(agent_name="cursor")
    assert plan.agent_name == "cursor"
    assert plan.backend_port > 0
    # No hooks.json on a clean machine; diff is pure additions.
    assert plan.diff.has_changes
    assert len(plan.diff.added) >= 1
    assert plan.diff.removed == []


def test_dry_run_does_not_write_anything(
    isolated_home: Path,
) -> None:
    result = CliRunner().invoke(
        main, ["init", "--agent", "cursor", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    # Hooks file must NOT be created.
    assert not (isolated_home / ".cursor" / "hooks.json").exists()
    # Config dir must NOT be created.
    assert not (isolated_home / ".agent-cot").exists()
    assert "dry-run" in result.output


def test_apply_writes_hooks_and_config(
    isolated_home: Path,
) -> None:
    """Full apply path: from nothing to fully installed."""
    # First, plant a few real hook scripts inside our package's assets dir
    # so apply_plan has something to copy.
    from agent_cot import _assets

    asset_root = _assets.hooks_dir() / "cursor"
    asset_root.mkdir(parents=True, exist_ok=True)
    for name in ("cot-bridge.js", "cot-stream.js"):
        src = asset_root / name
        if not src.is_file():
            src.write_text(_FAKE_HOOK_BODY, encoding="utf-8")

    result = CliRunner().invoke(
        main, ["init", "--agent", "cursor", "--apply"]
    )
    assert result.exit_code == 0, result.output

    hooks_json = isolated_home / ".cursor" / "hooks.json"
    assert hooks_json.is_file()
    blob = json.loads(hooks_json.read_text(encoding="utf-8"))
    assert blob["version"] == 1
    assert any(
        "cot-bridge.js" in e["command"] for e in blob["hooks"]["stop"]
    )

    # Hook scripts copied into ~/.cursor/hooks/
    hooks_dir = isolated_home / ".cursor" / "hooks"
    assert (hooks_dir / "cot-bridge.js").is_file()
    assert (hooks_dir / "cot-stream.js").is_file()
    assert not (hooks_dir / "agent_critic_hook.py").exists()

    # config.toml written
    cfg = isolated_home / ".agent-cot" / "config.toml"
    assert cfg.is_file()
    text = cfg.read_text(encoding="utf-8")
    assert "backend_port" in text


def test_apply_preserves_existing_third_party_hooks(
    isolated_home: Path,
) -> None:
    """Re-running init never wipes out codebuddy-mem etc."""
    from agent_cot import _assets

    asset_root = _assets.hooks_dir() / "cursor"
    asset_root.mkdir(parents=True, exist_ok=True)
    for name in ("cot-bridge.js", "cot-stream.js"):
        p = asset_root / name
        if not p.is_file():
            p.write_text(_FAKE_HOOK_BODY, encoding="utf-8")

    # Plant a pre-existing hooks.json with a third-party hook.
    hooks_json = isolated_home / ".cursor" / "hooks.json"
    hooks_json.parent.mkdir(parents=True, exist_ok=True)
    pre_existing = {
        "version": 1,
        "hooks": {
            "beforeSubmitPrompt": [
                {
                    "command": 'cmd /c node "{home}/.cursor/hooks/codebuddy-mem.js"'.format(
                        home=str(isolated_home).replace("\\", "/")
                    ),
                    "timeout": 10,
                }
            ]
        },
    }
    hooks_json.write_text(json.dumps(pre_existing), encoding="utf-8")

    result = CliRunner().invoke(
        main, ["init", "--agent", "cursor", "--apply"]
    )
    assert result.exit_code == 0, result.output

    after = json.loads(hooks_json.read_text(encoding="utf-8"))
    # Third-party entry must still be there.
    pre_cmd = pre_existing["hooks"]["beforeSubmitPrompt"][0]["command"]
    after_cmds = [e["command"] for e in after["hooks"]["beforeSubmitPrompt"]]
    assert pre_cmd in after_cmds, (
        f"third-party hook was dropped: {pre_cmd} not in {after_cmds}"
    )
    # And our entry must be there.
    assert any(
        "cot-bridge.js" in e["command"] for e in after["hooks"]["stop"]
    )

    # Backup file should also have been created.
    backups = list(hooks_json.parent.glob("hooks.json.bak.*"))
    assert backups, "backup file was not produced before write"


def test_second_init_refreshes_hook_scripts_when_hooks_json_unchanged(
    isolated_home: Path,
) -> None:
    """Regression: hooks.json already merged must still re-copy patched JS.

    v0.18.2 bug: ``init --apply`` returned early when ``diff.has_changes`` was
    false, leaving stale ``cot-bridge.js`` (e.g. ``D:/ai-ide-langfuse/...``).
    """
    from agent_cot import _assets

    asset_root = _assets.hooks_dir() / "cursor"
    asset_root.mkdir(parents=True, exist_ok=True)
    for name in ("cot-bridge.js", "cot-stream.js"):
        p = asset_root / name
        if not p.is_file():
            p.write_text(_FAKE_HOOK_BODY, encoding="utf-8")

    r1 = CliRunner().invoke(main, ["init", "--agent", "cursor", "--apply"])
    assert r1.exit_code == 0, r1.output

    hooks_dir = isolated_home / ".cursor" / "hooks"
    # 用绝不会出现在真实 bundle 里的占位串，避免与开发机 sibling 路径撞字面值。
    stale = (
        "const COT_ROOT = process.env.COT_EXTRACTOR_ROOT || '__STALE_INIT_TEST__';\n"
    )
    (hooks_dir / "cot-bridge.js").write_text(stale, encoding="utf-8")
    (hooks_dir / "cot-stream.js").write_text(stale, encoding="utf-8")

    r2 = CliRunner().invoke(main, ["init", "--agent", "cursor", "--apply"])
    assert r2.exit_code == 0, r2.output
    assert "refreshing hook scripts" in r2.output.lower()
    bridge = (hooks_dir / "cot-bridge.js").read_text(encoding="utf-8")
    # 第二次 init 必须从 wheel 资产重新拷完整脚本，而不是保留我们故意写的一行 stub。
    assert len(bridge) > 2000
    assert "__STALE_INIT_TEST__" not in bridge
    assert "process.env.COT_EXTRACTOR_ROOT" in bridge


def test_patch_cot_root_default_swaps_only_default() -> None:
    src = "const COT_ROOT = process.env.COT_EXTRACTOR_ROOT || 'd:/old/path';\n"
    out = init_cmd._patch_cot_root_default(src, r"D:\new\where")
    assert "process.env.COT_EXTRACTOR_ROOT" in out  # env override survives
    assert "'d:/old/path'" not in out
    assert "'D:/new/where'" in out


def test_patch_python_default_swaps_only_default() -> None:
    src = "const PYTHON = process.env.COT_PYTHON || 'python';\n"
    out = init_cmd._patch_python_default(src, r"C:\Python314\python.exe")
    assert "process.env.COT_PYTHON" in out
    assert "'python'" not in out
    assert "C:/Python314/python.exe" in out


# ---------------------------------------------------------------------------
# v0.20.4: Claude OTel env auto-injection via init --apply --agent claude
# ---------------------------------------------------------------------------


def _plant_claude_hook_asset() -> None:
    """Plant the bundled claude_stream_hook.py so apply_plan has something
    to copy without depending on the developer's actual sync state."""
    from agent_cot import _assets

    asset_root = _assets.hooks_dir() / "claude"
    asset_root.mkdir(parents=True, exist_ok=True)
    for name in ("claude_stream_hook.py",):
        target = asset_root / name
        if target.is_file():
            continue
        # 1 KB+ so it clears the v0.20.5 stub-guard inside apply_plan.
        body = (
            "# agent-cot test fixture - padded above the 1 KB stub-guard\n"
            "# _maybe_trigger_extract: real hook spawns extract_cot.py here\n"
            "# "
            + ("padding " * 200)
            + "\n"
        )
        target.write_text(body, encoding="utf-8")


def test_claude_init_apply_writes_otel_env_by_default(
    isolated_home: Path,
) -> None:
    """One-click: `agent-cot init --apply --agent claude` should land
    the OTel env in ~/.claude/settings.json with no extra flags."""
    _plant_claude_hook_asset()

    result = CliRunner().invoke(
        main, ["init", "--agent", "claude", "--apply"]
    )
    assert result.exit_code == 0, result.output

    settings_path = isolated_home / ".claude" / "settings.json"
    assert settings_path.is_file(), result.output
    settings = json.loads(settings_path.read_text(encoding="utf-8"))

    env = settings.get("env")
    assert isinstance(env, dict), settings
    # All four required keys present.
    assert env.get("CLAUDE_CODE_ENABLE_TELEMETRY") == "1"
    assert env.get("OTEL_EXPORTER_OTLP_PROTOCOL") == "http/json"
    assert env.get("OTEL_LOGS_EXPORTER") == "otlp"
    # Endpoint is 127.0.0.1 (NOT localhost) and dynamic to picked port.
    endpoint = env.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    assert endpoint.startswith("http://127.0.0.1:")
    assert "localhost" not in endpoint


def test_claude_init_apply_with_no_otel_env_flag_skips_env(
    isolated_home: Path,
) -> None:
    _plant_claude_hook_asset()

    result = CliRunner().invoke(
        main, ["init", "--agent", "claude", "--apply", "--no-otel-env"]
    )
    assert result.exit_code == 0, result.output

    settings_path = isolated_home / ".claude" / "settings.json"
    assert settings_path.is_file()
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    # Hooks still installed, but env block is absent / empty.
    assert "hooks" in settings
    assert not settings.get("env")


def test_claude_init_apply_preserves_user_endpoint(
    isolated_home: Path,
) -> None:
    """Fill-missing contract: user's corp / langfuse endpoint NEVER overwritten."""
    _plant_claude_hook_asset()

    settings_path = isolated_home / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    user_endpoint = "https://otel.corp.example.com:4318"
    settings_path.write_text(
        json.dumps(
            {
                "env": {
                    "OTEL_EXPORTER_OTLP_ENDPOINT": user_endpoint,
                    "OTEL_RESOURCE_ATTRIBUTES": "team=infra",
                }
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main, ["init", "--agent", "claude", "--apply"]
    )
    assert result.exit_code == 0, result.output

    after = json.loads(settings_path.read_text(encoding="utf-8"))
    env = after["env"]
    # User's endpoint kept verbatim.
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == user_endpoint
    # User's resource attrs kept verbatim.
    assert env["OTEL_RESOURCE_ATTRIBUTES"] == "team=infra"
    # Missing keys filled in.
    assert env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert env["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/json"
    # CLI output names what we preserved (so the user sees we noticed).
    assert "kept" in result.output.lower() or "preserved" in result.output.lower()


def test_claude_second_init_with_only_env_changes_still_writes(
    isolated_home: Path,
) -> None:
    """Regression: upgrading 0.20.3 -> 0.20.4 means hooks diff is empty
    but env is missing; init must NOT short-circuit into refresh-only mode."""
    _plant_claude_hook_asset()

    # First init writes hooks AND env in one go.
    r1 = CliRunner().invoke(main, ["init", "--agent", "claude", "--apply"])
    assert r1.exit_code == 0, r1.output

    settings_path = isolated_home / ".claude" / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    # Simulate the 0.20.3 state: hooks already there, env stripped out
    # (the keys 0.20.4 wants didn't exist in 0.20.3).
    settings.pop("env", None)
    settings_path.write_text(json.dumps(settings), encoding="utf-8")

    r2 = CliRunner().invoke(main, ["init", "--agent", "claude", "--apply"])
    assert r2.exit_code == 0, r2.output

    # Env block back, hooks block survives.
    after = json.loads(settings_path.read_text(encoding="utf-8"))
    assert "env" in after and after["env"].get("CLAUDE_CODE_ENABLE_TELEMETRY") == "1"
    assert "hooks" in after


def test_claude_dry_run_renders_env_diff(
    isolated_home: Path,
) -> None:
    _plant_claude_hook_asset()

    result = CliRunner().invoke(
        main, ["init", "--agent", "claude", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Claude OTel env" in out
    assert "CLAUDE_CODE_ENABLE_TELEMETRY" in out
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" in out
    # Nothing got written on dry-run.
    assert not (isolated_home / ".claude" / "settings.json").exists()


def test_cursor_init_does_not_touch_env(
    isolated_home: Path,
) -> None:
    """Sanity: only Claude adapter publishes recommended_env(); cursor must not
    write an env block into hooks.json."""
    from agent_cot import _assets

    asset_root = _assets.hooks_dir() / "cursor"
    asset_root.mkdir(parents=True, exist_ok=True)
    for n in ("cot-bridge.js", "cot-stream.js"):
        (asset_root / n).write_text(_FAKE_HOOK_BODY, encoding="utf-8")

    result = CliRunner().invoke(main, ["init", "--agent", "cursor", "--apply"])
    assert result.exit_code == 0, result.output

    hooks_json = isolated_home / ".cursor" / "hooks.json"
    blob = json.loads(hooks_json.read_text(encoding="utf-8"))
    # No env block injected into cursor's hooks.json — that field is
    # Claude-specific (settings.json schema).
    assert "env" not in blob
    # Plan side: build_plan reports otel_env_enabled=False for cursor.
    plan = init_cmd.build_plan(agent_name="cursor")
    assert plan.otel_env_enabled is False
    assert plan.env_additions == []
