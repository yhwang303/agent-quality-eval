"""Unit tests for installer/platform_paths.py."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from agent_cot.installer.platform_paths import (
    backup_path,
    agent_cot_root,
    cursor_root,
    ensure_dir,
)


def test_cursor_root_under_home() -> None:
    p = cursor_root()
    assert p.name == ".cursor"
    assert Path.home() in p.parents or Path.home() == p.parent


def test_agent_cot_root_distinct_from_cursor_root() -> None:
    """We absolutely must NOT collide with Cursor's own data dir."""
    assert agent_cot_root() != cursor_root()
    assert agent_cot_root().name == ".agent-cot"


def test_ensure_dir_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    out = ensure_dir(target)
    assert out == target
    assert target.is_dir()
    # Second call must not raise.
    ensure_dir(target)


def test_backup_path_format(tmp_path: Path) -> None:
    src = tmp_path / "hooks.json"
    src.write_text("{}")
    when = datetime(2026, 4, 27, 21, 30, 5)
    bak = backup_path(src, when=when)
    assert bak.name == "hooks.json.bak.20260427-213005"
    assert bak.parent == src.parent


def test_backup_path_unique_at_second_resolution(tmp_path: Path) -> None:
    src = tmp_path / "x"
    when_a = datetime(2026, 4, 27, 21, 30, 5)
    when_b = datetime(2026, 4, 27, 21, 30, 6)
    assert backup_path(src, when=when_a) != backup_path(src, when=when_b)
