"""Cross-process runtime state file at ``~/.agent-cot/runtime.json``.

Background
----------
The on-disk hook scripts (``~/.cursor/hooks/cot-bridge.js`` /
``cot-stream.js`` and ``~/.codebuddy/hooks/cot-stream-codebuddy.js``)
are install-time-patched copies of the bundled
``assets/hooks/{cursor,codebuddy}/*.js`` from this wheel. Two literals
are baked into the cursor scripts:

* ``COT_ROOT`` — directory containing ``scripts/extract_cot.py``
* ``PYTHON``   — interpreter that should run ``extract_cot.py``

Both default to env-var overrides at runtime; the literal fallback is
only used when no env var is present (the common case for hook spawns,
since neither Cursor nor CodeBuddy carries the user's shell env into
hook subprocesses on Windows).

The codebuddy stream hook is simpler — it only needs to know the data
root (``~/.agent-cot/data``) where ``events.jsonl`` is written. v0.18.6
already made that path user-writable. v0.19 layers an extra
``runtime.json`` fallback so the codebuddy hook can also recover from
weird env states (e.g. ``HOME`` unset under some service-launched
shells).

Failure mode this module fixes
------------------------------
After the user runs ``pip install -U agent-cot`` *without*
re-running ``agent-cot init --apply`` / ``upgrade --apply``, the
on-disk hook scripts still hold the **previous** install's literals.
If the previous wheel lived at a path that no longer resolves
(uninstall of an older wheel, virtualenv recreation, Python upgrade,
...), every Cursor ``stop`` event silently no-ops:

    spawn ENOENT __AGENT_COT_EXTRACTOR_ROOT_UNCONFIGURED__/scripts/extract_cot.py

→ no ``cot.json`` is ever written → the dashboard shows old / partial
data forever, with no obvious error to the user.

Design
------
We additionally write a JSON state file at ``~/.agent-cot/runtime.json``
on every ``agent-cot init / upgrade / start`` invocation:

    {
      "schema": 1,
      "updated_at": "2026-05-12T10:00:00Z",
      "agent_cot_version": "0.19.0",
      "cot_extractor_root": "C:/.../site-packages/agent_cot/assets/cot-extractor",
      "python_executable":  "C:/.../python.exe",
      "data_root":          "C:/Users/<u>/.agent-cot/data",
      "frontend_dist":      "C:/.../assets/frontend-dist",
      "backend_dir":        "C:/.../assets/backend",
      "cursor_hooks_dir":   "C:/Users/<u>/.cursor/hooks",
      "codebuddy_hooks_dir":"C:/Users/<u>/.codebuddy/hooks"
    }

Hook scripts read this file as a *secondary* fallback when both the
env-var override **and** the install-time literal don't resolve to a
real ``scripts/extract_cot.py`` (cursor) or a writable data root
(codebuddy). This makes ``pip install -U`` self-healing:
``agent-cot start`` (which the user runs anyway to refresh the
dashboard) writes a fresh ``runtime.json``, and the next Cursor /
CodeBuddy event picks up the new path automatically — even if the user
forgot ``upgrade --apply``.

Schema is intentionally tiny and forward-compatible (add fields
freely; ``schema`` bump only on incompatible removals). The file lives
in ``~/.agent-cot/`` so a clean ``agent-cot uninstall --purge-data``
wipes it along with the rest of our state.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import __version__, _assets
from .platform_paths import agent_cot_root, ensure_dir

SCHEMA_VERSION = 1


def _pipeline_log(event: str, **note: Any) -> None:
    """One-line breadcrumb to ~/.agent-cot/logs/pipeline.log. Never raises.

    Mirror of the JS / extractor / claude hook loggers — same format so a
    single ``tail -f`` shows the full lifecycle (CLI → hook → extract →
    backend) when something goes wrong on a colleague's machine.
    """
    try:
        from .. import diag
        diag.log("installer", ide="-", sid="-", event=event, **note)
    except Exception:
        pass


def runtime_state_path() -> Path:
    """Canonical path: ``~/.agent-cot/runtime.json``."""
    return agent_cot_root() / "runtime.json"


def _safe_str_path(p: Path | str | None) -> str | None:
    if p is None:
        return None
    try:
        return str(Path(p).resolve())
    except Exception:
        return str(p)


def build_state(
    *,
    cot_extractor_root: Path | str | None = None,
    python_executable: str | None = None,
    data_root: Path | str | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Snapshot the canonical paths a hook spawn needs to succeed.

    All arguments are optional; missing ones are auto-detected from the
    currently-installed package via :mod:`agent_cot._assets`. Callers
    who already know a value (e.g. ``init`` has resolved
    ``cot_extractor_root`` once) should pass it through to keep the
    state file consistent across writes.
    """
    extractor: Path | None = None
    if cot_extractor_root is not None:
        extractor = Path(cot_extractor_root)
    elif _assets.has_bundled_extractor():
        extractor = _assets.bundled_extractor_root().resolve()

    frontend: Path | None = None
    if _assets.has_frontend_dist():
        frontend = _assets.frontend_dist().resolve()

    backend: Path | None = None
    if _assets.has_bundled_backend():
        backend = _assets.bundled_backend_dir().resolve()

    state: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent_cot_version": __version__,
        "cot_extractor_root": _safe_str_path(extractor),
        "python_executable": _safe_str_path(python_executable or sys.executable),
        "data_root": _safe_str_path(data_root or agent_cot_root() / "data"),
        "frontend_dist": _safe_str_path(frontend),
        "backend_dir": _safe_str_path(backend),
        # v0.19: per-IDE hooks dirs so that doctor --deep can locate the
        # on-disk hook scripts without re-resolving each agent adapter.
        # Hook scripts themselves don't read these — they live under
        # ``~/.cursor/hooks/`` and ``~/.codebuddy/hooks/`` by convention.
        "cursor_hooks_dir": _safe_str_path(Path.home() / ".cursor" / "hooks"),
        "codebuddy_hooks_dir": _safe_str_path(Path.home() / ".codebuddy" / "hooks"),
    }
    if extras:
        state.update(extras)
    return state


def write_runtime_state(
    *,
    cot_extractor_root: Path | str | None = None,
    python_executable: str | None = None,
    data_root: Path | str | None = None,
    extras: dict[str, Any] | None = None,
) -> Path:
    """Atomically write ``~/.agent-cot/runtime.json`` with the current paths.

    Returns the path written. Safe to call multiple times — every call
    overwrites with the freshest snapshot.
    """
    ensure_dir(agent_cot_root())
    state = build_state(
        cot_extractor_root=cot_extractor_root,
        python_executable=python_executable,
        data_root=data_root,
        extras=extras,
    )
    target = runtime_state_path()
    tmp = target.with_suffix(".tmp")
    payload = json.dumps(state, indent=2, ensure_ascii=False) + "\n"
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, target)
    _pipeline_log(
        "runtime_state_written",
        path=str(target),
        data_root=state.get("data_root"),
        cot_extractor_root=state.get("cot_extractor_root"),
        python_executable=state.get("python_executable"),
    )
    return target


def read_runtime_state() -> dict[str, Any] | None:
    """Read the runtime state if present and well-formed; else ``None``.

    Hook scripts have their own JS-side reader (we duplicate this in
    ``cot-bridge.js`` / ``cot-stream.js`` / ``cot-stream-codebuddy.js``
    for the no-Python-required path); this helper is for Python-side
    consumers (doctor, tests).
    """
    p = runtime_state_path()
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


__all__ = [
    "SCHEMA_VERSION",
    "build_state",
    "read_runtime_state",
    "runtime_state_path",
    "write_runtime_state",
]
