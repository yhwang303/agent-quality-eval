"""Idempotent merge for CodeBuddy user hooks in ``~/.codebuddy/settings.json``.

CodeBuddy user settings preserve normal editor/plugin settings at the top
level and keep hook configuration under ``settings["hooks"]``. Each event
contains groups shaped like ``{"matcher": "...", "hooks": [{type, command}]}``.

The merge contract mirrors the Cursor merger: remove only agent-cot owned
commands, append fresh entries, and leave all third-party hooks byte-for-byte
unchanged.
"""
from __future__ import annotations

import copy
from typing import Any

from ..agents.base import HookEntry
from .hooks_merger import HookDiff, _commands_match, is_owned_command


def _entry_from_hookentry(entry: HookEntry) -> dict[str, Any]:
    hook: dict[str, Any] = {
        "type": entry.type or "command",
        "command": entry.command,
    }
    timeout = (entry.extra or {}).get("timeout", 10000)
    if timeout is not None:
        hook["timeout"] = timeout

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
        if not (isinstance(h, dict) and is_owned_command(h.get("command", "")))
    ]
    if not kept:
        return None
    group["hooks"] = kept
    return group


def merge_codebuddy_hooks(
    existing: dict[str, Any] | None,
    additions: list[HookEntry],
) -> dict[str, Any]:
    """Merge agent-cot CodeBuddy hooks into ``settings.json`` safely."""
    base: dict[str, Any] = copy.deepcopy(existing) if existing else {}
    hooks_block: dict[str, Any] = base.setdefault("hooks", {})
    if not isinstance(hooks_block, dict):
        raise ValueError(
            "CodeBuddy settings.json has unexpected shape: 'hooks' must be an object"
        )

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
                f"CodeBuddy settings.json has unexpected shape at hooks.{entry.event}: "
                "expected list"
            )
        bucket.append(_entry_from_hookentry(entry))

    return base


def diff_codebuddy_hooks(
    existing: dict[str, Any] | None,
    additions: list[HookEntry],
) -> HookDiff:
    """Compute the human-facing diff for a CodeBuddy settings merge."""
    diff = HookDiff()
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
                        if is_owned_command(command):
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
    "diff_codebuddy_hooks",
    "merge_codebuddy_hooks",
]
