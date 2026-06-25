"""Unit tests for ``installer/upgrader.py`` + ``commands/upgrade.py``.

Coverage focus:

* Plan correctly classifies missing / stale / up-to-date scripts.
* Apply is idempotent (running a second time is a no-op).
* hooks.json is NEVER touched by upgrade — the safety net users rely on.
* COT_ROOT default is patched per the user's stored config.
* Per-file backups have a distinct prefix from hooks.json backups
  (otherwise they'd pollute uninstall's restore list).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_cot import _assets
from agent_cot.cli import main
from agent_cot.commands import upgrade as upgrade_cmd
from agent_cot.installer.config import CursorCotConfig, save_config
from agent_cot.installer.upgrader import (
    apply_upgrade_plan,
    build_upgrade_plan,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake = tmp_path / "home"
    fake.mkdir()
    monkeypatch.setenv("HOME", str(fake))
    monkeypatch.setenv("USERPROFILE", str(fake))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake))
    return fake


def _seed_with_old_scripts(home: Path) -> tuple[Path, Path, dict]:
    """Plant placeholder hook scripts that visibly differ from the
    bundled ones, plus a hooks.json that we'll later assert is
    unchanged."""
    hooks_dir = home / ".cursor" / "hooks"
    hooks_dir.mkdir(parents=True)
    (hooks_dir / "cot-bridge.js").write_text(
        "// OLD-BRIDGE-PLACEHOLDER", encoding="utf-8"
    )
    (hooks_dir / "cot-stream.js").write_text(
        "// OLD-STREAM-PLACEHOLDER", encoding="utf-8"
    )
    (hooks_dir / "agent_critic_hook.py").write_text(
        "# OLD-CRITIC-PLACEHOLDER", encoding="utf-8"
    )

    hooks_json = home / ".cursor" / "hooks.json"
    snapshot = {"version": 1, "hooks": {"Stop": [{"command": "x", "timeout": 30}]}}
    hooks_json.write_text(json.dumps(snapshot), encoding="utf-8")
    return hooks_dir, hooks_json, snapshot


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def test_build_plan_marks_old_scripts_as_stale(isolated_home: Path) -> None:
    hooks_dir, _, _ = _seed_with_old_scripts(isolated_home)
    plan = build_upgrade_plan(
        agent_name="cursor",
        hooks_assets_dir=hooks_dir,
        cot_extractor_root=None,
    )
    assert len(plan.deltas) == 2
    assert all(d.target_exists for d in plan.deltas)
    assert all(d.needs_update for d in plan.deltas)
    assert plan.is_noop is False


def test_build_plan_when_scripts_missing(isolated_home: Path) -> None:
    hooks_dir = isolated_home / ".cursor" / "hooks"
    hooks_dir.mkdir(parents=True)
    plan = build_upgrade_plan(
        agent_name="cursor",
        hooks_assets_dir=hooks_dir,
        cot_extractor_root=None,
    )
    assert len(plan.missing) == len(plan.deltas)
    assert plan.is_noop is False  # missing files still need writing


def test_build_plan_warns_when_no_cot_root(isolated_home: Path) -> None:
    hooks_dir, _, _ = _seed_with_old_scripts(isolated_home)
    plan = build_upgrade_plan(
        agent_name="cursor",
        hooks_assets_dir=hooks_dir,
        cot_extractor_root=None,
    )
    assert any("cot_extractor_repo" in w for w in plan.warnings)


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def test_apply_replaces_stale_scripts_and_writes_backup(isolated_home: Path) -> None:
    hooks_dir, hooks_json, snapshot = _seed_with_old_scripts(isolated_home)
    plan = build_upgrade_plan(
        agent_name="cursor",
        hooks_assets_dir=hooks_dir,
        cot_extractor_root=None,
    )
    res = apply_upgrade_plan(plan)

    # Replaced bundled hook files + has a backup for each.
    assert {p.name for p in res.scripts_replaced} == {
        "cot-bridge.js",
        "cot-stream.js",
    }
    assert len(res.backups) == 2
    for b in res.backups:
        assert ".upgrade-bak." in b.name

    # The new scripts no longer contain the placeholder marker.
    bridge_text = (hooks_dir / "cot-bridge.js").read_text(encoding="utf-8")
    assert "OLD-BRIDGE-PLACEHOLDER" not in bridge_text

    # hooks.json is untouched — the whole point of upgrade.
    assert json.loads(hooks_json.read_text(encoding="utf-8")) == snapshot


def test_apply_is_idempotent(isolated_home: Path) -> None:
    hooks_dir, _, _ = _seed_with_old_scripts(isolated_home)

    plan1 = build_upgrade_plan(
        agent_name="cursor",
        hooks_assets_dir=hooks_dir,
        cot_extractor_root=None,
    )
    apply_upgrade_plan(plan1)

    plan2 = build_upgrade_plan(
        agent_name="cursor",
        hooks_assets_dir=hooks_dir,
        cot_extractor_root=None,
    )
    assert plan2.is_noop is True
    res2 = apply_upgrade_plan(plan2)
    assert res2.scripts_replaced == []
    assert res2.scripts_installed == []
    assert res2.backups == []


def test_apply_installs_when_target_missing(isolated_home: Path) -> None:
    hooks_dir = isolated_home / ".cursor" / "hooks"
    hooks_dir.mkdir(parents=True)

    plan = build_upgrade_plan(
        agent_name="cursor",
        hooks_assets_dir=hooks_dir,
        cot_extractor_root=None,
    )
    res = apply_upgrade_plan(plan)
    # All bundled hook files are *installed*, not replaced; no backups exist
    # because there was nothing to back up.
    assert res.scripts_replaced == []
    assert {p.name for p in res.scripts_installed} == {
        "cot-bridge.js",
        "cot-stream.js",
    }
    assert res.backups == []


def test_apply_patches_cot_root_default(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Verify the COT_ROOT literal in the *bundled* script gets
    rewritten when we pass cot_extractor_root."""
    bundled = _assets.hooks_dir() / "cursor" / "cot-bridge.js"
    if not bundled.is_file():
        pytest.skip("bundled cursor cot-bridge.js not present in this build")

    raw = bundled.read_text(encoding="utf-8")
    if "COT_EXTRACTOR_ROOT" not in raw:
        pytest.skip("bundled bridge does not have a COT_EXTRACTOR_ROOT default")

    hooks_dir = isolated_home / ".cursor" / "hooks"
    hooks_dir.mkdir(parents=True)

    cot_root = tmp_path / "cot-extractor"
    plan = build_upgrade_plan(
        agent_name="cursor",
        hooks_assets_dir=hooks_dir,
        cot_extractor_root=cot_root,
    )
    apply_upgrade_plan(plan)

    written = (hooks_dir / "cot-bridge.js").read_text(encoding="utf-8")
    posix = str(cot_root).replace("\\", "/")
    assert posix in written


def test_upgrade_backups_do_not_pollute_hooks_json_restore(
    isolated_home: Path,
) -> None:
    """Sanity check: `find_backups` should NOT return upgrade-bak files."""
    from agent_cot.installer.uninstaller import find_backups

    hooks_dir, hooks_json, _ = _seed_with_old_scripts(isolated_home)
    plan = build_upgrade_plan(
        agent_name="cursor",
        hooks_assets_dir=hooks_dir,
        cot_extractor_root=None,
    )
    apply_upgrade_plan(plan)

    # hooks.json sits next to the .upgrade-bak files; uninstall's
    # restore list must ignore them.
    assert find_backups(hooks_json) == []
    upgrade_bak_count = len(
        [p for p in hooks_dir.iterdir() if ".upgrade-bak." in p.name]
    )
    assert upgrade_bak_count == 2


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_upgrade_default_is_dry_run(isolated_home: Path) -> None:
    hooks_dir, _, _ = _seed_with_old_scripts(isolated_home)
    save_config(CursorCotConfig(installed_agents=["cursor"]))
    runner = CliRunner()
    res = runner.invoke(main, ["upgrade"])
    assert res.exit_code == 0, res.output
    assert "dry-run" in res.output
    # nothing on disk changed:
    assert "OLD-BRIDGE-PLACEHOLDER" in (hooks_dir / "cot-bridge.js").read_text(
        encoding="utf-8"
    )


def test_cli_upgrade_apply_changes_disk(isolated_home: Path) -> None:
    hooks_dir, hooks_json, snapshot = _seed_with_old_scripts(isolated_home)
    save_config(CursorCotConfig(installed_agents=["cursor"]))

    runner = CliRunner()
    res = runner.invoke(main, ["upgrade", "--apply"])
    assert res.exit_code == 0, res.output
    assert "upgrade — done" in res.output
    assert "OLD-BRIDGE-PLACEHOLDER" not in (hooks_dir / "cot-bridge.js").read_text(
        encoding="utf-8"
    )
    # hooks.json still untouched
    assert json.loads(hooks_json.read_text(encoding="utf-8")) == snapshot


def test_cli_upgrade_idempotent_run_says_nothing_to_upgrade(
    isolated_home: Path,
) -> None:
    _seed_with_old_scripts(isolated_home)
    save_config(CursorCotConfig(installed_agents=["cursor"]))

    runner = CliRunner()
    runner.invoke(main, ["upgrade", "--apply"])  # first run

    res = runner.invoke(main, ["upgrade"])  # dry-run should say no-op
    assert res.exit_code == 0
    assert re.search(r"already match", res.output)


def test_build_context_picks_up_config_cot_root(
    isolated_home: Path, tmp_path: Path
) -> None:
    fake_cot = tmp_path / "cot-extractor"
    fake_cot.mkdir()
    save_config(
        CursorCotConfig(installed_agents=["cursor"], cot_extractor_repo=str(fake_cot))
    )
    ctx = upgrade_cmd.build_context(agent_name="cursor")
    assert ctx.plan.cot_extractor_root == fake_cot
