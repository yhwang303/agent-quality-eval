"""Unit tests for ``_assets`` (P3 frontend / backend bundling)."""

from __future__ import annotations

from pathlib import Path

from agent_cot import _assets


def test_assets_root_is_a_directory() -> None:
    root = Path(str(_assets.assets_root()))
    assert root.is_dir()
    assert (root / "hooks").is_dir()


def test_hooks_dir_resolves() -> None:
    p = _assets.hooks_dir()
    assert p.is_dir()
    # Cursor and Claude folders ship as subdirectories from P0.
    assert (p / "cursor").is_dir()
    assert (p / "claude").is_dir()


def test_frontend_dist_path_is_consistent() -> None:
    """``frontend_dist`` always returns a path; presence determines bundle."""
    p = _assets.frontend_dist()
    assert isinstance(p, Path)
    # The path is *under* the package root.
    assert "frontend-dist" in p.parts


def test_has_frontend_dist_matches_disk(tmp_path: Path) -> None:
    """If index.html exists, ``has_frontend_dist`` must be True."""
    expected = (_assets.frontend_dist() / "index.html").is_file()
    assert _assets.has_frontend_dist() is expected


def test_bundled_backend_path_under_assets() -> None:
    p = _assets.bundled_backend_dir()
    assert "assets" in p.parts and "backend" in p.parts


def test_has_bundled_backend_matches_disk() -> None:
    expected = (_assets.bundled_backend_dir() / "main.py").is_file()
    assert _assets.has_bundled_backend() is expected


def test_p3_assets_actually_present() -> None:
    """After P3 the package must ship the bundle. Dev checkouts that
    forget ``python -m agent_cot._build_assets sync`` should fail
    here so we don't accidentally publish a hollow wheel.
    """
    assert _assets.has_bundled_backend(), (
        "assets/backend/main.py missing; run "
        "`python -m agent_cot._build_assets sync`"
    )
    assert _assets.has_frontend_dist(), (
        "assets/frontend-dist/index.html missing; run "
        "`npm run build` in agent-dashboard/frontend then "
        "`python -m agent_cot._build_assets sync`"
    )
