"""Read / write ``~/.agent-cot/config.toml``.

This is the *only* file we own that persists state between CLI
invocations. It's intentionally small: ports, the user's last init
choices, and the location of the cot-extractor / dashboard repos.

We use TOML rather than JSON so that:

* humans can hand-edit it without worrying about trailing commas;
* the file format is stable across Python versions
  (3.11+ has ``tomllib`` built-in; 3.10 falls back to ``tomli``).
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w

from .platform_paths import agent_cot_root, ensure_dir

CONFIG_FILENAME = "config.toml"
CONFIG_VERSION = 1


@dataclass
class CursorCotConfig:
    """In-memory representation of ``~/.agent-cot/config.toml``.

    Field semantics:

    ``schema_version``
        Always increments by exactly 1 when we change the layout. Used
        by ``agent-cot upgrade`` to apply migrations.
    ``backend_port``
        Picked at ``init`` time. Overridable by ``--port`` at ``start``.
    ``data_root``
        Where we keep PID files, server logs, and capture archives.
        Defaults to ``~/.agent-cot``; users with constrained home dirs
        can repoint this without re-running init.
    ``installed_agents``
        Names of agents whose hooks we currently own (think
        idempotent set on disk). Updated by ``init`` / ``uninstall``.
    ``cot_extractor_repo`` / ``dashboard_repo``
        Absolute paths to the source-of-truth checkouts. Filled in at
        init time; in a wheel install these point at locations inside
        the wheel itself; in editable installs they point at the user's
        repo. ``None`` means "not yet detected".
    """

    schema_version: int = CONFIG_VERSION
    backend_port: int | None = None
    data_root: str = ""
    installed_agents: list[str] = field(default_factory=list)
    cot_extractor_repo: str | None = None
    dashboard_repo: str | None = None

    def to_toml_dict(self) -> dict[str, Any]:
        # tomli_w is strict about None — drop empties so the file is
        # readable rather than dumping ``key = ""`` everywhere.
        out: dict[str, Any] = {}
        for k, v in asdict(self).items():
            if v is None or v == "":
                continue
            out[k] = v
        return out


def data_root() -> Path:
    """The ``~/.agent-cot`` directory; created on demand.

    *This call DOES create the directory* — only invoke from inside an
    ``apply``-phase code path. Read-only callers should use
    :func:`agent_cot.installer.platform_paths.agent_cot_root` instead.
    """
    return ensure_dir(agent_cot_root())


def config_path() -> Path:
    """Absolute path to ``~/.agent-cot/config.toml`` (no side effects)."""
    return agent_cot_root() / CONFIG_FILENAME


def load_config(*, path: Path | None = None) -> CursorCotConfig:
    """Load config from disk, returning defaults when the file is absent.

    Pure read; never creates directories.
    """
    p = path or config_path()
    default_root = str(agent_cot_root())
    if not p.is_file():
        return CursorCotConfig(data_root=default_root)

    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    return CursorCotConfig(
        schema_version=int(raw.get("schema_version", CONFIG_VERSION)),
        backend_port=raw.get("backend_port"),
        data_root=raw.get("data_root", default_root),
        installed_agents=list(raw.get("installed_agents", [])),
        cot_extractor_repo=raw.get("cot_extractor_repo"),
        dashboard_repo=raw.get("dashboard_repo"),
    )


def save_config(cfg: CursorCotConfig, *, path: Path | None = None) -> Path:
    """Persist ``cfg`` atomically (write-temp + rename). Creates the
    parent directory on demand."""
    p = path or config_path()
    ensure_dir(p.parent)
    payload = tomli_w.dumps(cfg.to_toml_dict())
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(p)
    return p


__all__ = [
    "CONFIG_FILENAME",
    "CONFIG_VERSION",
    "CursorCotConfig",
    "config_path",
    "data_root",
    "load_config",
    "save_config",
]
