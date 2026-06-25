"""Resolve on-disk paths to bundled assets.

The ``assets/`` directory ships *with* the wheel (see ``pyproject.toml``
``package-data``) and is the source-of-truth location for:

* hook scripts (``assets/hooks/cursor/*.js``, ``assets/hooks/claude/*.js``)
* the pre-built frontend (``assets/frontend-dist/``)

Pulling these via :mod:`importlib.resources` makes them work the same way
in editable installs, wheels, and zipped wheels.
"""

from __future__ import annotations

import os
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path


def assets_root() -> Traversable:
    """The package-relative ``assets/`` root."""
    override = os.environ.get("AGENT_COT_ASSETS_ROOT")
    if override:
        return Path(override).expanduser()
    return resources.files("agent_cot") / "assets"


def hooks_dir() -> Path:
    """Filesystem path to ``assets/hooks/``.

    Falls back to the source layout when running from an editable install
    where the assets directory may not exist yet (pre-P3).
    """
    root = assets_root() / "hooks"
    try:
        return Path(str(root))
    except (FileNotFoundError, NotADirectoryError):
        # Editable install before assets are populated — point at the
        # expected source layout so error messages are still useful.
        return Path(__file__).resolve().parent / "assets" / "hooks"


def frontend_dist() -> Path:
    """Filesystem path to ``assets/frontend-dist/``.

    Returns the package-bundled location even when the directory hasn't
    been populated yet (the caller should check ``has_frontend_dist()``
    before using the result).
    """
    root = assets_root() / "frontend-dist"
    try:
        return Path(str(root))
    except (FileNotFoundError, NotADirectoryError):
        return Path(__file__).resolve().parent / "assets" / "frontend-dist"


def has_frontend_dist() -> bool:
    """Return True iff a usable ``index.html`` lives in the bundle."""
    p = frontend_dist()
    return p.is_dir() and (p / "index.html").is_file()


def bundled_backend_dir() -> Path:
    """Filesystem path to the bundled FastAPI backend.

    P3 copies ``agent-dashboard/backend/`` into ``assets/backend/`` so
    the wheel ships a ready-to-spawn ``main:app``. Source-tree dev
    flows still take precedence (see ``commands/start._find_backend_dir``);
    this helper is the strict-bundle fallback.
    """
    root = assets_root() / "backend"
    try:
        return Path(str(root))
    except (FileNotFoundError, NotADirectoryError):
        return Path(__file__).resolve().parent / "assets" / "backend"


def has_bundled_backend() -> bool:
    """Return True iff a usable ``main.py`` lives in the bundle."""
    p = bundled_backend_dir()
    return p.is_dir() and (p / "main.py").is_file()


def bundled_extractor_root() -> Path:
    """Filesystem path to the bundled ``cot-extractor/`` checkout-equivalent.

    v0.18.2: ``assets/cot-extractor/`` ships the full ``scripts/ + src/``
    layout so that:

    * ``cot-bridge.js`` can resolve ``<root>/scripts/extract_cot.py``
      and spawn it after a Cursor stop event.
    * ``commands/init._find_cot_extractor_root`` can fall back here when
      no source checkout sits next to the wheel — i.e. when the user
      installed us via ``pip`` and not via ``git clone + pip install -e .``.

    Returns the path even when the directory hasn't been populated yet;
    callers should check ``has_bundled_extractor()`` first.
    """
    root = assets_root() / "cot-extractor"
    try:
        return Path(str(root))
    except (FileNotFoundError, NotADirectoryError):
        return Path(__file__).resolve().parent / "assets" / "cot-extractor"


def has_bundled_extractor() -> bool:
    """Return True iff a usable ``scripts/extract_cot.py`` lives in the bundle.

    Mirror of ``has_bundled_backend`` / ``has_frontend_dist``: hides the
    "wheel might be old / partially populated" probe from every caller.
    """
    p = bundled_extractor_root()
    return p.is_dir() and (p / "scripts" / "extract_cot.py").is_file()


__all__ = [
    "assets_root",
    "bundled_backend_dir",
    "bundled_extractor_root",
    "frontend_dist",
    "has_bundled_backend",
    "has_bundled_extractor",
    "has_frontend_dist",
    "hooks_dir",
]
