"""Cross-OS path conventions used by every installer step.

We deliberately keep these as plain functions returning :class:`pathlib.Path`,
not constants, because:

* tests need to monkey-patch ``Path.home()`` per-test;
* CI on Linux / macOS / Windows must all import this module without
  side effects.

All paths are created lazily via :func:`ensure_dir` — importing this
module never touches the filesystem.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def cursor_root() -> Path:
    """Return the per-user Cursor configuration root.

    On every supported OS this resolves to ``~/.cursor`` (Cursor itself
    creates this directory on first launch).
    """
    return Path.home() / ".cursor"


def agent_cot_root() -> Path:
    """Return *our* per-user data root, ``~/.agent-cot``.

    Distinct from ``~/.cursor`` so that a clean ``agent-cot uninstall``
    can wipe just our data without touching anything Cursor owns.
    """
    return Path.home() / ".agent-cot"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing; return it for chaining."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def backup_path(original: Path, *, when: datetime | None = None) -> Path:
    """Return a timestamped sibling backup path for ``original``.

    Format: ``<name>.bak.YYYYMMDD-HHMMSS<suffix>``. Reusing this helper
    everywhere keeps backup discovery (e.g. ``uninstall --restore``)
    trivial.
    """
    when = when or datetime.now()
    stamp = when.strftime("%Y%m%d-%H%M%S")
    return original.with_name(f"{original.name}.bak.{stamp}")


__all__ = [
    "backup_path",
    "agent_cot_root",
    "cursor_root",
    "ensure_dir",
]
