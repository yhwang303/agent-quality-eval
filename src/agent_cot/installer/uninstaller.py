"""Reverse of :mod:`agent_cot.commands.init` — purely on the
filesystem layer.

This module never touches an :class:`AgentAdapter`; it operates on
the *filenames* we own (see :data:`OWNED_HOOK_SCRIPTS`) plus the
single ``hooks.json`` that lives at a path the caller picks. That
separation is deliberate: the CLI command in
``commands.uninstall`` is the only piece that resolves agents and
loops over them, so this file stays unit-testable on a tmp dir.

Two operations are exposed:

* :func:`build_uninstall_plan` / :func:`apply_uninstall_plan` —
  remove our merged hooks + delete bundled scripts, mirroring init.
* :func:`restore_latest_backup` — atomically swap the most recent
  ``hooks.json.bak.<ts>`` back into place. Used by
  ``agent-cot uninstall --restore-backup``.

Everything is dry-run-by-default in spirit: the ``apply_*`` helpers
are the only writers, and they only run when the CLI explicitly
calls them. The ``build_*`` helpers are pure.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .claude_hooks_merger import merge_claude_hooks
from .codebuddy_hooks_merger import merge_codebuddy_hooks
from .copilot_hooks_merger import merge_copilot_hooks
from .hooks_merger import OWNED_HOOK_SCRIPTS, is_owned_command, merge_cursor_hooks
from .platform_paths import backup_path


# ---------------------------------------------------------------------------
# v0.19.4 (P-10): per-agent merger dispatch
# ---------------------------------------------------------------------------
#
# Cursor has a flat hook schema (``hooks.<event> = [{type, command}, ...]``)
# while CodeBuddy and Claude store hooks in a nested
# ``hooks.<event> = [{"matcher": "...", "hooks": [{type, command}]}]`` shape.
# Copilot has yet another (close to cursor but with extras). Each shape
# has its own merger module that knows how to strip *only* our own
# commands while leaving every third-party hook intact.
#
# Before 0.19.4 ``apply_uninstall_plan`` unconditionally invoked
# ``merge_cursor_hooks(existing, additions=[])``. On a CodeBuddy /
# Claude / Copilot ``settings.json`` that meant the file passed through
# unchanged — we deleted our hook *scripts* on disk but left their
# absolute paths inside ``settings.json``, leaving stale references that
# now point at missing files. ``agent-cot uninstall`` looked like it
# succeeded but the IDE next launch would emit "hook command not
# found" errors.
#
# We now select the merger by ``agent_name``. Unknown / None falls back
# to cursor (the conservative default — cursor is the only shape ``init``
# would ever produce on a fresh install for the "cursor" agent).


def _resolve_merger(agent_name: str | None):
    name = (agent_name or "").lower()
    if name == "codebuddy":
        return merge_codebuddy_hooks
    if name in ("claude", "claude-internal", "claude_internal", "claude-code", "claude_code"):
        return merge_claude_hooks
    if name == "copilot":
        return merge_copilot_hooks
    # Cursor / VSCode / unknown — use the flat schema merger.
    return merge_cursor_hooks

# ---------------------------------------------------------------------------
# Plan / Result types
# ---------------------------------------------------------------------------


@dataclass
class UninstallPlan:
    """Describes everything an uninstall would do, before any write."""

    hooks_config_path: Path
    """``~/.cursor/hooks.json`` (or whatever the adapter said)."""

    hooks_assets_dir: Path
    """Directory holding our bundled JS hook scripts."""

    will_remove_entries: list[tuple[str, str]] = field(default_factory=list)
    """``(event, command)`` pairs that match :func:`is_owned_command`."""

    will_delete_scripts: list[Path] = field(default_factory=list)
    """Absolute paths of bundled hook scripts that exist and will be deleted."""

    keep_data: bool = True
    """If False, ``apply`` will additionally wipe ``~/.agent-cot``."""

    will_clear_config_agent: str | None = None
    """If non-None, drop this agent from ``config.toml`` on apply."""

    @property
    def is_noop(self) -> bool:
        return (
            not self.will_remove_entries
            and not self.will_delete_scripts
            and not self.will_clear_config_agent
            and self.keep_data
        )

    def render(self) -> str:
        lines: list[str] = []
        lines.append(f"hooks.json     : {self.hooks_config_path}")
        lines.append(f"hooks dir      : {self.hooks_assets_dir}")
        lines.append("")
        lines.append("hooks.json removals:")
        if self.will_remove_entries:
            for ev, cmd in self.will_remove_entries:
                lines.append(f"  - [{ev}] {cmd}")
        else:
            lines.append("  (none — nothing of ours is registered)")
        lines.append("")
        lines.append("script files to delete:")
        if self.will_delete_scripts:
            for p in self.will_delete_scripts:
                lines.append(f"  ✗ {p}")
        else:
            lines.append("  (none — already absent)")
        if self.will_clear_config_agent:
            lines.append("")
            lines.append(
                f"config.toml    : drop agent '{self.will_clear_config_agent}' "
                "from installed_agents"
            )
        if not self.keep_data:
            lines.append("")
            lines.append("data           : --purge-data was given; "
                         "~/.agent-cot will be removed")
        return "\n".join(lines)


@dataclass
class UninstallResult:
    """What apply_uninstall_plan actually did on disk."""

    hooks_backup: Path | None = None
    hooks_written: Path | None = None
    """``None`` if hooks.json no longer exists after uninstall (we
    don't synthesise an empty file)."""

    scripts_deleted: list[Path] = field(default_factory=list)
    config_updated: Path | None = None
    data_root_deleted: Path | None = None


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def build_uninstall_plan(
    *,
    hooks_config_path: Path,
    hooks_assets_dir: Path,
    keep_data: bool = True,
    agent_name: str | None = None,
) -> UninstallPlan:
    """Build a :class:`UninstallPlan` without writing anything.

    The plan is safe to print, log, or hand to a unit test. The
    detection rule is the same one ``init`` uses to find duplicates,
    which guarantees uninstall is the inverse of init.
    """
    plan = UninstallPlan(
        hooks_config_path=hooks_config_path,
        hooks_assets_dir=hooks_assets_dir,
        keep_data=keep_data,
        will_clear_config_agent=agent_name,
    )

    # 1. Owned entries currently in hooks.json.
    #
    # v0.19.4 (P-10): handle BOTH the flat (cursor) and the nested
    # (codebuddy / claude) shapes. Before this, nested entries went
    # undetected and the plan reported "nothing to remove" for
    # CodeBuddy / Claude even when our hooks were clearly registered.
    if hooks_config_path.is_file():
        try:
            existing = json.loads(
                hooks_config_path.read_text(encoding="utf-8") or "{}"
            )
        except json.JSONDecodeError:
            # Treat as "nothing to remove" — uninstall on a malformed
            # file would be too dangerous; the CLI will warn loudly.
            existing = {}
        hooks_block = existing.get("hooks") if isinstance(existing, dict) else {}
        if isinstance(hooks_block, dict):
            for event, entries in hooks_block.items():
                if not isinstance(entries, list):
                    continue
                for e in entries:
                    if not isinstance(e, dict):
                        continue
                    # flat shape: {type, command}
                    cmd = e.get("command")
                    if isinstance(cmd, str) and is_owned_command(cmd):
                        plan.will_remove_entries.append((event, cmd))
                        continue
                    # nested shape: {matcher, hooks: [{type, command}]}
                    nested = e.get("hooks")
                    if isinstance(nested, list):
                        for inner in nested:
                            if not isinstance(inner, dict):
                                continue
                            inner_cmd = inner.get("command")
                            if isinstance(inner_cmd, str) and is_owned_command(inner_cmd):
                                plan.will_remove_entries.append((event, inner_cmd))

    # 2. Hook scripts that exist on disk.
    if hooks_assets_dir.is_dir():
        for name in sorted(OWNED_HOOK_SCRIPTS):
            target = hooks_assets_dir / name
            if target.is_file():
                plan.will_delete_scripts.append(target)

    return plan


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def apply_uninstall_plan(
    plan: UninstallPlan,
    *,
    config_save_callback=None,
) -> UninstallResult:
    """Carry out an :class:`UninstallPlan`.

    Order of operations (pick to keep the user recoverable on a
    crash mid-way):

    1. Back up ``hooks.json`` so accidental data loss is reversible.
    2. Rewrite ``hooks.json`` with our entries stripped (atomic).
       If, after stripping, no other hook events remain *and* the
       file was empty other than ``version: 1``, we still leave the
       skeleton — Cursor's UI is happier with an empty file than
       with a missing one. We **never** delete ``hooks.json`` itself.
    3. Delete our bundled hook scripts.
    4. (optional) Update ``config.toml`` via ``config_save_callback``.
    5. (optional) Wipe ``~/.agent-cot`` if ``keep_data=False``.

    ``config_save_callback`` is an injection point so this module
    doesn't import :mod:`agent_cot.installer.config` (avoids
    circular imports). The CLI passes ``functools.partial`` of
    ``save_config(cfg)`` and gets the resulting Path back.
    """
    result = UninstallResult()

    # --- 1+2. hooks.json ---------------------------------------------------
    if plan.hooks_config_path.is_file():
        backup = backup_path(plan.hooks_config_path)
        shutil.copy2(plan.hooks_config_path, backup)
        result.hooks_backup = backup

        existing = json.loads(plan.hooks_config_path.read_text(encoding="utf-8") or "{}")
        # v0.19.4 (P-10): dispatch to the merger matching the agent's
        # hook schema. Passing additions=[] re-uses the merger as the
        # "strip ours, add nothing" cleaner.
        merger = _resolve_merger(plan.will_clear_config_agent)
        new_blob = merger(existing, additions=[])

        payload = json.dumps(new_blob, indent=2, ensure_ascii=False)
        tmp = plan.hooks_config_path.with_suffix(
            plan.hooks_config_path.suffix + ".tmp"
        )
        tmp.write_text(payload + "\n", encoding="utf-8")
        tmp.replace(plan.hooks_config_path)
        result.hooks_written = plan.hooks_config_path

    # --- 3. delete bundled hook scripts -----------------------------------
    for target in plan.will_delete_scripts:
        try:
            target.unlink()
        except FileNotFoundError:
            # Race / second invocation — fine.
            continue
        result.scripts_deleted.append(target)

    # --- 4. config.toml ---------------------------------------------------
    if config_save_callback is not None and plan.will_clear_config_agent:
        result.config_updated = config_save_callback()

    # --- 5. wipe data root (only if explicitly requested) -----------------
    if not plan.keep_data:
        from .platform_paths import agent_cot_root

        root = agent_cot_root()
        if root.is_dir():
            shutil.rmtree(root, ignore_errors=False)
            result.data_root_deleted = root

    return result


# ---------------------------------------------------------------------------
# Backup discovery / restore
# ---------------------------------------------------------------------------


_BACKUP_RE = re.compile(r"\.bak\.(\d{8}-\d{6})$")


def find_backups(hooks_config_path: Path) -> list[Path]:
    """Return every ``hooks.json.bak.<ts>`` next to ``hooks_config_path``,
    sorted newest-first.

    The discovery rule is intentionally narrow: we only accept the
    exact pattern :func:`backup_path` produces. That way we will
    never mistake an arbitrary user file for one of ours.
    """
    parent = hooks_config_path.parent
    if not parent.is_dir():
        return []

    name = hooks_config_path.name
    found: list[tuple[datetime, Path]] = []
    for child in parent.iterdir():
        if not child.is_file():
            continue
        if not child.name.startswith(name + ".bak."):
            continue
        m = _BACKUP_RE.search(child.name)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group(1), "%Y%m%d-%H%M%S")
        except ValueError:
            continue
        found.append((ts, child))
    found.sort(reverse=True)
    return [p for _, p in found]


@dataclass
class RestoreResult:
    """What :func:`restore_latest_backup` actually did."""

    restored_from: Path
    target: Path
    pre_restore_backup: Path | None
    """A safety copy of whatever was at ``target`` before we overwrote
    it. Always written when ``target`` already existed."""


def restore_latest_backup(hooks_config_path: Path) -> RestoreResult:
    """Atomically swap the newest backup back into place.

    Raises ``FileNotFoundError`` if no recognisable backup exists.
    The pre-restore safety copy is *also* a ``hooks.json.bak.<ts>``
    so it shows up in :func:`find_backups` going forward — handy
    when the user wanted to undo the undo.
    """
    backups = find_backups(hooks_config_path)
    if not backups:
        raise FileNotFoundError(
            f"no backups found next to {hooks_config_path} "
            "(searched for files named '<name>.bak.YYYYMMDD-HHMMSS')."
        )
    src = backups[0]

    pre_backup: Path | None = None
    if hooks_config_path.is_file():
        pre_backup = backup_path(hooks_config_path)
        shutil.copy2(hooks_config_path, pre_backup)

    tmp = hooks_config_path.with_suffix(hooks_config_path.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(hooks_config_path)

    return RestoreResult(
        restored_from=src,
        target=hooks_config_path,
        pre_restore_backup=pre_backup,
    )


__all__ = [
    "RestoreResult",
    "UninstallPlan",
    "UninstallResult",
    "apply_uninstall_plan",
    "build_uninstall_plan",
    "find_backups",
    "restore_latest_backup",
]
