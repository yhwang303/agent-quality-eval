"""installer/ — site-modifying primitives used by ``agent-cot init``.

Modules in this package are deliberately small and side-effect-light:
each one does one thing (path resolution, port picking, hook merging,
config IO) so that the high-level ``commands/init.py`` orchestration
remains easy to read and unit-test.

Nothing in this package writes outside ``agent-cot``'s own data dirs
without an explicit caller decision; see ``commands/init.py`` for the
state machine that gates real disk writes behind ``--dry-run`` /
explicit user consent.
"""

from __future__ import annotations

from .config import (
    CursorCotConfig,
    config_path,
    data_root,
    load_config,
    save_config,
)
from .hooks_merger import (
    HookDiff,
    diff_hooks,
    is_owned_command,
    merge_cursor_hooks,
)
from .platform_paths import backup_path, agent_cot_root, cursor_root, ensure_dir
from .port_picker import PortNotFoundError, is_port_free, pick_port

__all__ = [
    "CursorCotConfig",
    "HookDiff",
    "PortNotFoundError",
    "backup_path",
    "config_path",
    "agent_cot_root",
    "cursor_root",
    "data_root",
    "diff_hooks",
    "ensure_dir",
    "is_owned_command",
    "is_port_free",
    "load_config",
    "merge_cursor_hooks",
    "pick_port",
    "save_config",
]
