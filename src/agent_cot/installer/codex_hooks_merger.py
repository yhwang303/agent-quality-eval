"""Idempotent merge helpers for Codex ``hooks.json``.

Codex uses the same three-level hook shape as Claude Code:
``hooks.<Event>[].hooks[]``.  We keep a separate merger so Codex-owned
commands can be stripped by filename without touching existing user hooks such
as codebuddy-mem.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..agents.base import HookEntry
from .hooks_merger import _commands_match

_CODEX_OWNED_NAMES: frozenset[str] = frozenset(
    {"codex_stream_hook.py", "codex_sidecar_collector.py", "agent_critic_hook.py"}
)


def is_codex_owned_command(command: str) -> bool:
    if not isinstance(command, str) or not command:
        return False
    haystack = command.replace("\\", "/").lower()
    return any(name in haystack for name in _CODEX_OWNED_NAMES)


@dataclass
class CodexHookDiff:
    added: list[tuple[str, str]] = field(default_factory=list)
    removed: list[tuple[str, str]] = field(default_factory=list)
    untouched_other_owners: int = 0

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed)

    def render(self) -> str:
        lines: list[str] = []
        if not self.has_changes:
            lines.append("(no changes - hooks.json already in sync)")
        for ev, cmd in self.removed:
            lines.append(f"  - [{ev}] {cmd}")
        for ev, cmd in self.added:
            lines.append(f"  + [{ev}] {cmd}")
        if self.untouched_other_owners:
            lines.append(
                f"  (preserving {self.untouched_other_owners} entries from other tools)"
            )
        return "\n".join(lines)


def _entry_from_hookentry(entry: HookEntry) -> dict[str, Any]:
    hook: dict[str, Any] = {
        "type": entry.type or "command",
        "command": entry.command,
    }
    for key in ("timeout", "statusMessage", "commandWindows"):
        value = (entry.extra or {}).get(key)
        if value is not None:
            hook[key] = value
    group: dict[str, Any] = {"hooks": [hook]}
    matcher = (entry.extra or {}).get("matcher")
    if matcher:
        group["matcher"] = matcher
    return group


def _group_commands(group: Any) -> list[str]:
    if not isinstance(group, dict):
        return []
    hooks = group.get("hooks")
    if not isinstance(hooks, list):
        return []
    return [
        h.get("command", "")
        for h in hooks
        if isinstance(h, dict) and isinstance(h.get("command"), str)
    ]


def _strip_owned_from_group(group: dict[str, Any]) -> dict[str, Any] | None:
    hooks = group.get("hooks")
    if not isinstance(hooks, list):
        return group
    kept = [
        h
        for h in hooks
        if not (isinstance(h, dict) and is_codex_owned_command(h.get("command", "")))
    ]
    if not kept:
        return None
    group["hooks"] = kept
    return group


def merge_codex_hooks(
    existing: dict[str, Any] | None,
    additions: list[HookEntry],
) -> dict[str, Any]:
    base: dict[str, Any] = copy.deepcopy(existing) if existing else {}
    hooks_block = base.setdefault("hooks", {})
    if not isinstance(hooks_block, dict):
        raise ValueError("Codex hooks.json has unexpected shape: 'hooks' must be an object")

    for event, groups in list(hooks_block.items()):
        if not isinstance(groups, list):
            continue
        kept_groups: list[Any] = []
        for group in groups:
            if not isinstance(group, dict):
                kept_groups.append(group)
                continue
            stripped = _strip_owned_from_group(group)
            if stripped is not None:
                kept_groups.append(stripped)
        hooks_block[event] = kept_groups

    for entry in additions:
        bucket = hooks_block.setdefault(entry.event, [])
        if not isinstance(bucket, list):
            raise ValueError(
                f"Codex hooks.json has unexpected shape at hooks.{entry.event}: expected list"
            )
        bucket.append(_entry_from_hookentry(entry))

    return base


def diff_codex_hooks(
    existing: dict[str, Any] | None,
    additions: list[HookEntry],
) -> CodexHookDiff:
    diff = CodexHookDiff()
    base_hooks = (existing or {}).get("hooks", {}) if existing else {}

    existing_commands: dict[str, list[str]] = {}
    if isinstance(base_hooks, dict):
        for event, groups in base_hooks.items():
            if not isinstance(groups, list):
                continue
            event_commands: list[str] = []
            for group in groups:
                commands = _group_commands(group)
                event_commands.extend(commands)
                if commands:
                    for command in commands:
                        if is_codex_owned_command(command):
                            diff.removed.append((event, command))
                        else:
                            diff.untouched_other_owners += 1
                else:
                    diff.untouched_other_owners += 1
            existing_commands[event] = event_commands

    for entry in additions:
        present = any(
            _commands_match(command, entry.command)
            for command in existing_commands.get(entry.event, [])
        )
        if present:
            try:
                diff.removed.remove((entry.event, entry.command))
            except ValueError:
                pass
            continue
        diff.added.append((entry.event, entry.command))

    return diff


__all__ = [
    "CodexHookDiff",
    "diff_codex_hooks",
    "is_codex_owned_command",
    "merge_codex_hooks",
]
