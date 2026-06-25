"""``agent-cot uninstall`` — orchestration layer over the installer
helpers.

Responsibilities:

* Resolve the agent adapter (cursor today; claude in v0.14).
* Build an :class:`UninstallPlan` for that adapter.
* In dry-run mode (the default), print what *would* happen.
* In apply mode, run :func:`apply_uninstall_plan` and update
  ``config.toml``.
* For ``--restore-backup``, defer to
  :func:`installer.uninstaller.restore_latest_backup` and bail out.

Why we still go through ``--apply``: same reason ``init`` does —
nobody should lose hooks just because they fat-fingered a flag.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..agents import AgentNotImplementedError, get_adapter
from ..agents.base import AgentAdapter, CursorCotError
from ..installer.config import (
    CursorCotConfig,
    config_path,
    load_config,
    save_config,
)
from ..installer.uninstaller import (
    RestoreResult,
    UninstallPlan,
    UninstallResult,
    apply_uninstall_plan,
    build_uninstall_plan,
    find_backups,
    restore_latest_backup,
)


class UninstallError(CursorCotError):
    """Raised for uninstall-specific failures the CLI can render verbatim."""


@dataclass
class UninstallContext:
    """Glues the per-adapter plan to the global config write step."""

    plan: UninstallPlan
    adapter: AgentAdapter
    config_before: CursorCotConfig

    def render_summary(self) -> str:
        return self.plan.render()


# ---------------------------------------------------------------------------
# build / apply
# ---------------------------------------------------------------------------


def build_context(
    *,
    agent_name: str = "cursor",
    keep_data: bool = True,
) -> UninstallContext:
    """Pure planning step — no disk writes."""
    adapter = get_adapter(agent_name)
    try:
        # We don't actually need the entries (we strip ours by marker),
        # but exercising this method validates the adapter is usable.
        adapter.hook_entries()
    except AgentNotImplementedError:
        raise

    cfg = load_config()
    # Only flag a config edit when there's actually something to drop;
    # otherwise the "clean machine" noop branch would be falsely tripped.
    config_agent = agent_name if agent_name in cfg.installed_agents else None

    plan = build_uninstall_plan(
        hooks_config_path=adapter.hooks_config_path(),
        hooks_assets_dir=adapter.hooks_assets_dir(),
        keep_data=keep_data,
        agent_name=config_agent,
    )
    return UninstallContext(plan=plan, adapter=adapter, config_before=cfg)


def apply(ctx: UninstallContext) -> UninstallResult:
    """Carry out the planned uninstall + persist config changes.

    The ``config.toml`` rewrite is wired in via a callback so the
    installer module stays import-cycle-free.
    """
    cfg = ctx.config_before

    def save_config_callback() -> Path:
        if ctx.adapter.name in cfg.installed_agents:
            cfg.installed_agents = [
                n for n in cfg.installed_agents if n != ctx.adapter.name
            ]
        return save_config(cfg)

    # Only update config if there is something to update; otherwise
    # we'd silently re-write the file with no change.
    callback: Callable[[], Path] | None = (
        save_config_callback
        if ctx.adapter.name in cfg.installed_agents
        else None
    )

    return apply_uninstall_plan(ctx.plan, config_save_callback=callback)


# ---------------------------------------------------------------------------
# Restore-from-backup path
# ---------------------------------------------------------------------------


@dataclass
class BackupListing:
    target: Path
    backups: list[Path]


def list_available_backups(*, agent_name: str = "cursor") -> BackupListing:
    """Inspect the agent's ``hooks.json`` directory for backups.

    Used by ``--restore-backup --dry-run`` so users can see exactly
    which file would be restored and when it was taken.
    """
    adapter = get_adapter(agent_name)
    target = adapter.hooks_config_path()
    return BackupListing(target=target, backups=find_backups(target))


def restore(*, agent_name: str = "cursor") -> RestoreResult:
    """Apply :func:`restore_latest_backup` for the given adapter."""
    adapter = get_adapter(agent_name)
    target = adapter.hooks_config_path()
    try:
        return restore_latest_backup(target)
    except FileNotFoundError as exc:
        raise UninstallError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Read-only helpers used by the CLI for messaging
# ---------------------------------------------------------------------------


def config_will_disappear(ctx: UninstallContext) -> bool:
    """``config.toml`` survives uninstall; we only edit it. The user
    is told this so they don't expect a clean slate."""
    return False


def config_summary_line(ctx: UninstallContext) -> str:
    cp = config_path()
    return f"config.toml    : kept at {cp}"


__all__ = [
    "BackupListing",
    "UninstallContext",
    "UninstallError",
    "UninstallResult",
    "apply",
    "build_context",
    "config_summary_line",
    "config_will_disappear",
    "list_available_backups",
    "restore",
]
