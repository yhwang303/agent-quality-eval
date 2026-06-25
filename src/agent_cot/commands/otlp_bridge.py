"""Bridge between agent-cot CLI and the cot-extractor OTLP exporter.

We do **not** vendor ``cot_otlp_exporter`` into the wheel: it is a
substantial amount of OTel SDK code and depends on the user's choice
of backend extras. Instead, we locate the cot-extractor source tree
at runtime, in this priority order:

1. ``AGENT_COT_EXTRACTOR_SRC`` env (overrides everything; intended
   for CI / containerised setups).
2. ``cot_extractor_repo`` field in ``~/.agent-cot/config.toml`` (set
   by ``init`` when it auto-detects the source checkout).
3. A walk up from this file looking for ``cot-extractor/src`` — the
   developer-checkout fastpath.
4. A regular ``import cot_otlp_exporter`` — succeeds when the user
   ``pip install``-ed cot-extractor as a real package.

If none of these work we raise :class:`OtlpBridgeError` with a message
the CLI can print verbatim. We intentionally keep the import path
lookup *here*, not inlined in each command, so doctor can probe it
with the same logic.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agents.base import CursorCotError
from ..installer.config import load_config
from ..installer.runtime_state import read_runtime_state


class OtlpBridgeError(CursorCotError):
    """Raised when we cannot import / use cot_otlp_exporter."""


# ---------------------------------------------------------------------------
# v0.19.4: shared data_root / extractor_root resolution helpers
# ---------------------------------------------------------------------------
#
# Before 0.19.4 this module had its own bespoke "walk parents looking for
# cot-extractor/" logic and never consulted ``runtime.json``. The result:
# after a normal ``agent-cot init`` install (no source repo in sight),
# every ``agent-cot otlp send <sid>`` failed with "could not locate
# cot.json" — because hooks wrote ``<sid>_cot.json`` to
# ``~/.agent-cot/data/cot/`` and we only scanned ``<repo>/cot-extractor/
# output/cot/``. Fix: read ``runtime.json`` first (same order hooks use:
# env → runtime.json → default), THEN keep the repo-walk for developer
# checkouts.
#
# ``_normalize_data_root`` mirrors the hook self-healing logic: if
# someone (or an older init) wrote ``data_root = ~/.agent-cot`` we still
# resolve the actual data dir at ``~/.agent-cot/data``.


def _normalize_data_root(p: Path) -> Path:
    """If ``data_root`` ends at ``.agent-cot`` / ``.cursor-cot`` (no
    ``/data`` suffix), append it. Mirrors hook ``_normalizeDataRoot``.
    """
    name = p.name
    if name == "data":
        return p
    if name in (".agent-cot", ".cursor-cot"):
        return p / "data"
    return p


def _resolve_data_root() -> Path:
    """env > runtime.json > default — identical chain to the hooks.

    Returns the user-level data root (i.e. parent of ``cot/``,
    ``events/``, ``reports/``).
    """
    env_v = os.environ.get("AGENT_COT_DATA_ROOT", "").strip()
    if env_v:
        return _normalize_data_root(Path(env_v).expanduser())
    rt = read_runtime_state() or {}
    rt_root = rt.get("data_root")
    if isinstance(rt_root, str) and rt_root.strip():
        return _normalize_data_root(Path(rt_root).expanduser())
    return Path.home() / ".agent-cot" / "data"


def _resolve_extractor_root() -> Path | None:
    """env > runtime.json.cot_extractor_root > config.toml.cot_extractor_repo.

    Returns ``None`` if every layer is empty — caller falls back to
    repo-walk.
    """
    env_v = os.environ.get("AGENT_COT_EXTRACTOR_SRC", "").strip()
    if env_v:
        return Path(env_v).expanduser()
    rt = read_runtime_state() or {}
    rt_root = rt.get("cot_extractor_root")
    if isinstance(rt_root, str) and rt_root.strip():
        return Path(rt_root).expanduser()
    cfg = load_config()
    if cfg.cot_extractor_repo:
        return Path(cfg.cot_extractor_repo)
    return None


# ---------------------------------------------------------------------------
# Source-tree resolution
# ---------------------------------------------------------------------------


_MAX_PARENT_WALK = 5
"""How many ancestors of this file to scan when looking for a
sibling ``cot-extractor/src``. Five lets us escape
``agent-cot/src/agent_cot/commands/`` (4 levels deep) plus one
more so we land at the mono-repo root that hosts ``cot-extractor/``
as a sibling — while staying short enough that error output is
still readable."""


def _dedup_paths(paths: list[Path]) -> list[Path]:
    """Drop duplicates while preserving order. Resolves are skipped
    because some entries refer to non-existent paths and we want the
    error message to surface them verbatim.
    """
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _candidate_src_dirs() -> list[Path]:
    """Ordered list of directories likely to contain ``cot_otlp_exporter.py``.

    Empty entries are filtered out; non-existent paths are kept (so
    error messages can show what we tried).

    v0.19.4 (P-4): now also reads ``runtime.json.cot_extractor_root``
    (set by ``init``/``start`` to the bundled-wheel location), and
    finally checks the bundled ``assets/cot-extractor-src/`` shipped
    with this wheel. Previously, neither was consulted — meaning a
    wheel-only install (no source repo in sight) had to fall back to
    plain ``import cot_otlp_exporter`` which only works if a separate
    cot-extractor package is pip-installed.
    """
    out: list[Path] = []

    # 1. env var override
    env = os.environ.get("AGENT_COT_EXTRACTOR_SRC", "").strip()
    if env:
        out.append(Path(env))

    # 2. runtime.json (the cross-process source of truth — same chain
    #    every hook uses)
    rt = read_runtime_state() or {}
    rt_root = rt.get("cot_extractor_root")
    if isinstance(rt_root, str) and rt_root.strip():
        root = Path(rt_root).expanduser()
        out.append(root / "src")
        out.append(root)

    # 3. config.toml ``cot_extractor_repo`` (legacy field; init still
    #    writes it for back-compat)
    cfg = load_config()
    if cfg.cot_extractor_repo:
        repo = Path(cfg.cot_extractor_repo)
        out.append(repo / "src")
        out.append(repo)

    # 4. bundled vendor copy in this wheel
    here = Path(__file__).resolve()
    bundled_extractor_src = here.parent.parent / "assets" / "cot-extractor-src"
    out.append(bundled_extractor_src)
    bundled_extractor_full_src = here.parent.parent / "assets" / "cot-extractor" / "src"
    out.append(bundled_extractor_full_src)

    # 5. developer-checkout fallback: walk up looking for a sibling
    for parent in list(here.parents)[:_MAX_PARENT_WALK]:
        out.append(parent / "cot-extractor" / "src")

    return _dedup_paths([p for p in out if p])


def _ensure_on_sys_path(p: Path) -> None:
    sp = str(p.resolve())
    if sp not in sys.path:
        sys.path.insert(0, sp)


def import_exporter():
    """Return the imported ``cot_otlp_exporter`` module or raise.

    We try every candidate dir in turn, then fall back to a plain
    ``import`` (for ``pip install`` users). On failure we raise an
    :class:`OtlpBridgeError` listing every path we attempted, so the
    user has a precise diagnostic.
    """
    tried: list[str] = []

    # 1-3: source-tree lookups
    for cand in _candidate_src_dirs():
        if cand.is_dir() and (cand / "cot_otlp_exporter.py").is_file():
            _ensure_on_sys_path(cand)
            try:
                return importlib.import_module("cot_otlp_exporter")
            except ImportError as exc:
                tried.append(f"{cand} (import failed: {exc})")
                continue
        else:
            tried.append(f"{cand} (no cot_otlp_exporter.py here)")

    # 4: pip-installed
    try:
        return importlib.import_module("cot_otlp_exporter")
    except ImportError as exc:
        tried.append(f"`pip install cot-extractor` (import failed: {exc})")

    raise OtlpBridgeError(
        "could not import cot_otlp_exporter from any known location.\n"
        "Tried:\n  - " + "\n  - ".join(tried) + "\n\n"
        "Fix: set AGENT_COT_EXTRACTOR_SRC=/path/to/cot-extractor/src,\n"
        "or run `agent-cot init --apply` from inside a repo that contains\n"
        "the cot-extractor/ subdirectory."
    )


# ---------------------------------------------------------------------------
# cot.json resolution
# ---------------------------------------------------------------------------


@dataclass
class ResolvedCotJson:
    path: Path
    session_id: str
    raw: dict[str, Any] = field(default_factory=dict)


def _candidate_cot_dirs() -> list[Path]:
    """Where on disk should we look for ``<sid>_cot.json`` files?

    v0.19.4 (P-1): now also reads ``AGENT_COT_DATA_ROOT`` env and
    ``runtime.json.data_root`` first, then the user default
    ``~/.agent-cot/data/cot``. Without this, ``agent-cot otlp send
    <sid>`` could never find ``cot.json`` files written by hooks on a
    normal wheel install — the previous logic only scanned the
    in-repo ``cot-extractor/output/`` layout used by developer
    checkouts.

    Order:

    1. ``COT_DIR`` env (explicit override; preserves legacy semantics).
    2. ``AGENT_COT_DATA_ROOT`` env / ``runtime.json.data_root`` /
       default → ``<data_root>/cot`` (matches the hook + backend chain).
    3. Legacy user-level roots (``~/.agent-cot/cot`` etc) — covers
       0.19.0~0.19.2 installs that wrote to the wrong dir before the
       data-root divergence was fixed.
    4. ``config.toml.cot_extractor_repo`` & repo-walk —
       developer-checkout fastpaths (unchanged, kept last so they
       don't shadow user data on real installs).
    """
    out: list[Path] = []

    # 1. explicit env override (highest precedence; preserves old API).
    env = os.environ.get("COT_DIR", "").strip()
    if env:
        out.append(Path(env))

    # 2. user data root via same chain hooks/backend use.
    data_root = _resolve_data_root()
    out.append(data_root / "cot")
    out.append(data_root / "sessions")

    # 3. legacy locations left by 0.19.0~0.19.2 broken init (the
    #    ``data_root = ~/.agent-cot`` (no ``/data``) case). The
    #    SessionScanner already covers these; covering them here too
    #    keeps ``otlp send`` symmetric with the dashboard.
    home = Path.home()
    out.append(home / ".agent-cot" / "cot")
    out.append(home / ".cursor-cot" / "data" / "cot")
    out.append(home / ".cursor-cot" / "cot")

    # 4. developer-checkout fastpaths.
    cfg = load_config()
    if cfg.cot_extractor_repo:
        repo = Path(cfg.cot_extractor_repo)
        out.append(repo / "output" / "cot")
        out.append(repo / "output" / "sessions")

    here = Path(__file__).resolve()
    for parent in list(here.parents)[:_MAX_PARENT_WALK]:
        out.append(parent / "cot-extractor" / "output" / "cot")
        out.append(parent / "cot-extractor" / "output" / "sessions")

    return _dedup_paths([p for p in out if p])


def resolve_cot_json(
    *,
    session_id: str | None = None,
    cot_path: str | None = None,
) -> ResolvedCotJson:
    """Resolve a session id (or explicit path) to a parsed cot.json.

    Search strategy mirrors cot-extractor's ``_resolve_cot_path``:

    * ``<COT_DIR>/<sid>_cot.json`` (flat layout)
    * ``<COT_DIR>/../sessions/<sid>/*_cot.json`` — pick most recent.
    * ``<sessions_dir>/<sid>/*_cot.json`` — same, when COT_DIR points
      directly at ``sessions/``.
    """
    if cot_path:
        p = Path(cot_path).expanduser().resolve()
        if not p.is_file():
            raise OtlpBridgeError(f"cot.json not found: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise OtlpBridgeError(f"failed to parse {p}: {exc}") from exc
        sid = data.get("session_id") or session_id or p.stem.replace("_cot", "")
        return ResolvedCotJson(path=p, session_id=sid, raw=data)

    if not session_id:
        raise OtlpBridgeError("either --session-id or --cot-path is required.")

    tried: list[Path] = []
    for cot_dir in _candidate_cot_dirs():
        flat = cot_dir / f"{session_id}_cot.json"
        tried.append(flat)
        if flat.is_file():
            return _load_cot(flat, session_id)

        # `<cot_dir>/../sessions/<sid>/*_cot.json`
        sessions_dir_a = cot_dir.parent / "sessions" / session_id
        tried.append(sessions_dir_a / "*_cot.json")
        if sessions_dir_a.is_dir():
            picked = _pick_latest(sessions_dir_a)
            if picked is not None:
                return _load_cot(picked, session_id)

        # `<cot_dir>/<sid>/*_cot.json` (when COT_DIR = .../sessions)
        sessions_dir_b = cot_dir / session_id
        tried.append(sessions_dir_b / "*_cot.json")
        if sessions_dir_b.is_dir():
            picked = _pick_latest(sessions_dir_b)
            if picked is not None:
                return _load_cot(picked, session_id)

    seen: set[str] = set()
    unique_tried: list[Path] = []
    for p in tried:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        unique_tried.append(p)

    msg = (
        f"could not locate cot.json for session_id={session_id}. Tried:\n  - "
        + "\n  - ".join(str(p) for p in unique_tried)
        + "\nFix: pass --cot-path explicitly, or set COT_DIR / "
        "cot_extractor_repo in ~/.agent-cot/config.toml."
    )
    raise OtlpBridgeError(msg)


def _pick_latest(dir_: Path) -> Path | None:
    candidates = sorted(dir_.glob("*_cot.json"))
    return candidates[-1] if candidates else None


def _load_cot(path: Path, fallback_sid: str) -> ResolvedCotJson:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise OtlpBridgeError(f"failed to parse {path}: {exc}") from exc
    sid = data.get("session_id") or fallback_sid
    return ResolvedCotJson(path=path, session_id=sid, raw=data)


# ---------------------------------------------------------------------------
# Preset resolution
# ---------------------------------------------------------------------------


def get_presets() -> list[dict[str, Any]]:
    """Return the list of backend presets, or raise."""
    mod = import_exporter()
    presets = getattr(mod, "BACKEND_PRESETS", None)
    if not presets:
        raise OtlpBridgeError(
            "cot_otlp_exporter is importable but BACKEND_PRESETS is empty. "
            "Upgrade cot-extractor to v0.12+."
        )
    return list(presets)


def find_preset(preset_id: str) -> dict[str, Any]:
    presets = get_presets()
    for p in presets:
        if p.get("id") == preset_id:
            return p
    ids = ", ".join(p.get("id", "?") for p in presets)
    raise OtlpBridgeError(
        f"unknown preset '{preset_id}'. Available: {ids}\n"
        "Run `agent-cot otlp list-presets` to see full descriptions."
    )


# ---------------------------------------------------------------------------
# Header parsing (shared between CLI and tests)
# ---------------------------------------------------------------------------


def parse_headers(items: list[str]) -> dict[str, str]:
    """Parse ``["k=v", "k2: v2"]`` into a dict.

    Mirrors cot-extractor's ``scripts/export_otlp.py::_parse_headers``
    so users can copy/paste invocations between the two CLIs.
    """
    out: dict[str, str] = {}
    for raw in items or []:
        if "=" in raw:
            k, _, v = raw.partition("=")
        elif ":" in raw:
            k, _, v = raw.partition(":")
        else:
            raise OtlpBridgeError(
                f"--header must be 'k=v' or 'k: v'; got: {raw!r}"
            )
        k = k.strip()
        v = v.strip()
        if not k:
            raise OtlpBridgeError(f"empty header key in: {raw!r}")
        out[k] = v
    return out


__all__ = [
    "OtlpBridgeError",
    "ResolvedCotJson",
    "find_preset",
    "get_presets",
    "import_exporter",
    "parse_headers",
    "resolve_cot_json",
]
