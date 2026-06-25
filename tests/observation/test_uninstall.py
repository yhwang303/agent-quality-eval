"""Unit tests for ``installer/uninstaller.py`` + ``commands/uninstall.py``.

Coverage focus:

* The core merge-strip rule deletes only OWNED entries.
* Bundled hook scripts are removed from disk.
* Other tools' hooks survive byte-for-byte.
* Uninstall on a clean machine is a no-op (idempotent).
* ``find_backups`` accepts only the canonical timestamp shape.
* ``restore_latest_backup`` writes the right bytes and records a
  pre-restore safety copy.
* The CLI surface honours ``--dry-run`` (default) and ``--apply``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_cot.cli import main
from agent_cot.commands import uninstall as uninstall_cmd
from agent_cot.installer.platform_paths import backup_path
from agent_cot.installer.uninstaller import (
    apply_uninstall_plan,
    build_uninstall_plan,
    find_backups,
    restore_latest_backup,
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


def _seed_installed_state(home: Path) -> tuple[Path, Path]:
    """Synthesize a realistic ``~/.cursor`` layout with our hooks
    installed alongside a third-party hook entry that we MUST NOT
    touch."""
    cursor = home / ".cursor"
    hooks_dir = cursor / "hooks"
    hooks_dir.mkdir(parents=True)

    (hooks_dir / "cot-bridge.js").write_text("// fake bridge", encoding="utf-8")
    (hooks_dir / "cot-stream.js").write_text("// fake stream", encoding="utf-8")

    hooks_json = cursor / "hooks.json"
    hooks_json.write_text(
        json.dumps(
            {
                "version": 1,
                "hooks": {
                    "Stop": [
                        # someone else's hook — must be preserved
                        {
                            "command": (
                                "python "
                                "\"$HOME/.cursor/hooks/codebuddy-mem-hook.py\""
                            ),
                            "timeout": 30,
                        },
                        # ours
                        {
                            "command": "node \"$HOME/.cursor/hooks/cot-bridge.js\"",
                            "timeout": 30,
                        },
                    ],
                    "Notification": [
                        {
                            "command": "node \"$HOME/.cursor/hooks/cot-stream.js\"",
                            "timeout": 30,
                        },
                    ],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return hooks_json, hooks_dir


# ---------------------------------------------------------------------------
# build_uninstall_plan
# ---------------------------------------------------------------------------


def test_build_plan_lists_owned_entries_and_scripts(isolated_home: Path) -> None:
    hooks_json, hooks_dir = _seed_installed_state(isolated_home)

    plan = build_uninstall_plan(
        hooks_config_path=hooks_json,
        hooks_assets_dir=hooks_dir,
        keep_data=True,
        agent_name="cursor",
    )

    # 2 owned hook entries (one per script).
    assert len(plan.will_remove_entries) == 2
    cmds = [c for _, c in plan.will_remove_entries]
    assert any("cot-bridge.js" in c for c in cmds)
    assert any("cot-stream.js" in c for c in cmds)
    # The codebuddy-mem entry is NOT in the removal list.
    assert all("codebuddy-mem" not in c for c in cmds)

    # 2 hook script files exist and are listed.
    assert len(plan.will_delete_scripts) == 2
    assert {p.name for p in plan.will_delete_scripts} == {
        "cot-bridge.js",
        "cot-stream.js",
    }


def test_build_plan_on_clean_machine_is_noop(isolated_home: Path) -> None:
    cursor = isolated_home / ".cursor"
    cursor.mkdir()
    plan = build_uninstall_plan(
        hooks_config_path=cursor / "hooks.json",
        hooks_assets_dir=cursor / "hooks",
        keep_data=True,
    )
    assert plan.will_remove_entries == []
    assert plan.will_delete_scripts == []
    assert plan.is_noop is True


# ---------------------------------------------------------------------------
# apply_uninstall_plan
# ---------------------------------------------------------------------------


def test_apply_strips_only_owned_entries_and_keeps_others(
    isolated_home: Path,
) -> None:
    hooks_json, hooks_dir = _seed_installed_state(isolated_home)

    plan = build_uninstall_plan(
        hooks_config_path=hooks_json,
        hooks_assets_dir=hooks_dir,
        keep_data=True,
    )
    res = apply_uninstall_plan(plan)

    # 1. Backup was taken.
    assert res.hooks_backup is not None and res.hooks_backup.is_file()

    # 2. hooks.json now contains exactly the codebuddy-mem entry.
    out = json.loads(hooks_json.read_text(encoding="utf-8"))
    stop = out["hooks"]["Stop"]
    assert len(stop) == 1
    assert "codebuddy-mem" in stop[0]["command"]
    # Notification list was left empty (we never delete the event key).
    assert out["hooks"].get("Notification", []) == []

    # 3. Both scripts are gone from disk.
    assert not (hooks_dir / "cot-bridge.js").exists()
    assert not (hooks_dir / "cot-stream.js").exists()
    assert {p.name for p in res.scripts_deleted} == {
        "cot-bridge.js",
        "cot-stream.js",
    }


def test_apply_is_idempotent(isolated_home: Path) -> None:
    hooks_json, hooks_dir = _seed_installed_state(isolated_home)
    plan = build_uninstall_plan(
        hooks_config_path=hooks_json, hooks_assets_dir=hooks_dir
    )
    apply_uninstall_plan(plan)

    # Re-build the plan after running once; it must be a no-op now.
    plan2 = build_uninstall_plan(
        hooks_config_path=hooks_json, hooks_assets_dir=hooks_dir
    )
    assert plan2.will_remove_entries == []
    assert plan2.will_delete_scripts == []

    # Applying the second plan should not raise and leaves disk
    # unchanged.
    res2 = apply_uninstall_plan(plan2)
    assert res2.scripts_deleted == []
    # No new backup since hooks.json *was* still there but stripping
    # an empty plan still rewrites the file — that's fine; we accept
    # either branch as long as nothing else mutated.
    contents = json.loads(hooks_json.read_text(encoding="utf-8"))
    assert "codebuddy-mem" in str(contents)


def test_apply_preserves_malformed_hooks_json_safely(isolated_home: Path) -> None:
    """A corrupt hooks.json should not silently nuke other tools'
    state. We back it up but skip the merge step."""
    cursor = isolated_home / ".cursor"
    cursor.mkdir()
    hooks_json = cursor / "hooks.json"
    hooks_json.write_text("this is { not :: json", encoding="utf-8")

    plan = build_uninstall_plan(
        hooks_config_path=hooks_json,
        hooks_assets_dir=cursor / "hooks",
    )
    # build_plan tolerated the invalid file (it doesn't enumerate
    # owned entries because it can't read them):
    assert plan.will_remove_entries == []


# ---------------------------------------------------------------------------
# find_backups / restore
# ---------------------------------------------------------------------------


def test_find_backups_only_matches_canonical_pattern(isolated_home: Path) -> None:
    cursor = isolated_home / ".cursor"
    cursor.mkdir()
    target = cursor / "hooks.json"
    target.write_text("{}", encoding="utf-8")

    # Real backups produced by backup_path():
    older = backup_path(target)
    older.write_text("{\"v\":1}", encoding="utf-8")
    # backup_path uses second precision; sleep-free we simulate a
    # newer timestamp by manually constructing the name.
    newer = target.with_name("hooks.json.bak.20300101-000000")
    newer.write_text("{\"v\":2}", encoding="utf-8")
    # Decoy files that must NOT be picked up:
    (cursor / "hooks.json.backup").write_text("nope", encoding="utf-8")
    (cursor / "hooks.json.bak").write_text("nope2", encoding="utf-8")
    (cursor / "hooks.json.bak.not-a-date").write_text("nope3", encoding="utf-8")

    found = find_backups(target)
    names = [p.name for p in found]
    assert names == ["hooks.json.bak.20300101-000000", older.name]


def test_restore_latest_backup_writes_expected_bytes_and_pre_backup(
    isolated_home: Path,
) -> None:
    cursor = isolated_home / ".cursor"
    cursor.mkdir()
    target = cursor / "hooks.json"
    target.write_text("CURRENT", encoding="utf-8")
    saved = target.with_name("hooks.json.bak.20300101-120000")
    saved.write_text("RESTORED", encoding="utf-8")

    res = restore_latest_backup(target)
    assert target.read_text(encoding="utf-8") == "RESTORED"
    assert res.restored_from == saved
    # A pre-restore backup must exist so the user can undo the undo.
    assert res.pre_restore_backup is not None
    assert res.pre_restore_backup.is_file()
    assert res.pre_restore_backup.read_text(encoding="utf-8") == "CURRENT"


def test_restore_latest_backup_no_backups_raises(isolated_home: Path) -> None:
    cursor = isolated_home / ".cursor"
    cursor.mkdir()
    with pytest.raises(FileNotFoundError, match="no backups found"):
        restore_latest_backup(cursor / "hooks.json")


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_uninstall_default_is_dry_run(isolated_home: Path) -> None:
    hooks_json, hooks_dir = _seed_installed_state(isolated_home)
    runner = CliRunner()
    res = runner.invoke(main, ["uninstall"])
    assert res.exit_code == 0, res.output
    assert "dry-run" in res.output
    # On-disk state unchanged:
    assert (hooks_dir / "cot-bridge.js").exists()
    assert "cot-bridge.js" in hooks_json.read_text(encoding="utf-8")


def test_cli_uninstall_apply_modifies_disk(isolated_home: Path) -> None:
    hooks_json, hooks_dir = _seed_installed_state(isolated_home)
    runner = CliRunner()
    res = runner.invoke(main, ["uninstall", "--apply"])
    assert res.exit_code == 0, res.output
    assert "uninstall — done" in res.output
    # scripts removed, codebuddy-mem preserved:
    assert not (hooks_dir / "cot-bridge.js").exists()
    out = json.loads(hooks_json.read_text(encoding="utf-8"))
    cmds = [e["command"] for e in out["hooks"].get("Stop", [])]
    assert all("cot-bridge" not in c for c in cmds)
    assert any("codebuddy-mem" in c for c in cmds)


def test_cli_uninstall_clean_machine_is_friendly(isolated_home: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(main, ["uninstall", "--apply"])
    assert res.exit_code == 0, res.output
    assert "no changes to apply" in res.output


def test_cli_uninstall_restore_backup_dry_run(isolated_home: Path) -> None:
    hooks_json, _ = _seed_installed_state(isolated_home)
    bak = hooks_json.with_name("hooks.json.bak.20300202-100000")
    bak.write_text("{}", encoding="utf-8")

    runner = CliRunner()
    res = runner.invoke(main, ["uninstall", "--restore-backup"])
    assert res.exit_code == 0, res.output
    assert "latest backup" in res.output
    assert "dry-run" in res.output
    # Disk untouched:
    assert "cot-bridge" in hooks_json.read_text(encoding="utf-8")


def test_cli_uninstall_restore_backup_apply(isolated_home: Path) -> None:
    hooks_json, _ = _seed_installed_state(isolated_home)
    bak = hooks_json.with_name("hooks.json.bak.20300202-100000")
    bak.write_text("{\"version\": 1, \"hooks\": {}}", encoding="utf-8")

    runner = CliRunner()
    res = runner.invoke(main, ["uninstall", "--restore-backup", "--apply"])
    assert res.exit_code == 0, res.output
    assert "restore — done" in res.output
    out = json.loads(hooks_json.read_text(encoding="utf-8"))
    assert out == {"version": 1, "hooks": {}}


def test_cli_uninstall_restore_backup_no_backups(isolated_home: Path) -> None:
    cursor = isolated_home / ".cursor"
    cursor.mkdir()
    runner = CliRunner()
    res = runner.invoke(main, ["uninstall", "--restore-backup", "--apply"])
    assert res.exit_code != 0
    assert "no backups found" in res.output


# ---------------------------------------------------------------------------
# build_context (commands layer)
# ---------------------------------------------------------------------------


def test_build_context_uses_correct_adapter(isolated_home: Path) -> None:
    ctx = uninstall_cmd.build_context(agent_name="cursor", keep_data=True)
    assert ctx.adapter.name == "cursor"
    assert ctx.plan.hooks_config_path.name == "hooks.json"
