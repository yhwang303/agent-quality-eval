"""Idempotent merge for Claude Code (Internal) user hooks in
``~/.claude/settings.json``.

v0.19.1 引入。Claude Code 的 user settings 顶层保留所有非 hook 配置，
hook 配置位于 ``settings["hooks"]`` 下，按事件名分组：

.. code-block:: json

    {
      "hooks": {
        "Stop": [
          {"hooks": [{"type": "command", "command": "..."}]}
        ],
        "PreToolUse": [
          {"matcher": "Bash", "hooks": [{"type": "command", "command": "..."}]}
        ]
      }
    }

shape 跟 CodeBuddy 完全一致（嵌套 ``hooks.<Event>[].hooks[]`` 结构 +
可选 ``matcher``），逻辑上完全可以复用 codebuddy 的 merger。但是为了
**符合用户底线"不要碰 codebuddy 的部分"** + 让 Claude / CodeBuddy 各自的
ownership-detection 完全解耦，这里独立维护一份。

ownership 判定：所有 ``command`` 字段含 ``claude_stream_hook.py``（任何
路径变体 / 引号 / 斜杠方向）的条目即视为 agent-cot owned，再次 init
时会被先 strip 再 append，保证幂等。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..agents.base import HookEntry
from .hooks_merger import _commands_match


# ---------------------------------------------------------------------------
# Owner detection — 跟 hooks_merger.is_owned_command 同思路，但只看 claude
# 自己的 hook 文件名，避免把 cursor / codebuddy 的命令误删。
# ---------------------------------------------------------------------------


_CLAUDE_OWNED_NAMES: frozenset[str] = frozenset(
    {"claude_stream_hook.py", "agent_critic_hook.py"}
)


def is_claude_owned_command(command: str) -> bool:
    """``True`` iff ``command`` invokes one of OUR Claude hook scripts.

    Tolerant of forward / backward slashes, surrounding quotes and any
    ``python`` / absolute path prefix. Matches by *filename only* — never
    by a substring like ``agent-cot`` that the user might use elsewhere.
    """
    if not isinstance(command, str) or not command:
        return False
    haystack = command.replace("\\", "/").lower()
    return any(name in haystack for name in (n.lower() for n in _CLAUDE_OWNED_NAMES))


# ---------------------------------------------------------------------------
# Diff (mirrors :class:`agent_cot.installer.hooks_merger.HookDiff` shape so
# CLI / tests can render either uniformly).
# ---------------------------------------------------------------------------


@dataclass
class ClaudeHookDiff:
    added: list[tuple[str, str]] = field(default_factory=list)
    removed: list[tuple[str, str]] = field(default_factory=list)
    untouched_other_owners: int = 0

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed)

    def render(self) -> str:
        lines: list[str] = []
        if not self.has_changes:
            lines.append("(no changes — settings.json already in sync)")
        for ev, cmd in self.removed:
            lines.append(f"  - [{ev}] {cmd}")
        for ev, cmd in self.added:
            lines.append(f"  + [{ev}] {cmd}")
        if self.untouched_other_owners:
            lines.append(
                f"  (preserving {self.untouched_other_owners} entries from other tools)"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry_from_hookentry(entry: HookEntry) -> dict[str, Any]:
    """Render one ``HookEntry`` into Claude's nested group shape."""
    hook: dict[str, Any] = {
        "type": entry.type or "command",
        "command": entry.command,
    }
    timeout = (entry.extra or {}).get("timeout")
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
    """Drop OUR commands from one group; return None if the group becomes empty."""
    hooks = group.get("hooks")
    if not isinstance(hooks, list):
        return group
    kept = [
        h
        for h in hooks
        if not (
            isinstance(h, dict) and is_claude_owned_command(h.get("command", ""))
        )
    ]
    if not kept:
        return None
    group["hooks"] = kept
    return group


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_claude_hooks(
    existing: dict[str, Any] | None,
    additions: list[HookEntry],
) -> dict[str, Any]:
    """Idempotently inject our entries into Claude ``settings.json``.

    Contract:

    1. Strip every command whose filename matches :func:`is_claude_owned_command`.
       This wipes any prior agent-cot install (regardless of the absolute
       path baked in earlier) so re-running ``init --apply`` after
       ``pip install -U agent-cot`` produces the same final shape.
    2. Append every entry in ``additions`` under its event bucket.
    3. NEVER touch entries owned by other tools (codebuddy-mem,
       langfuse_hook, user's own bash one-liners, etc).
    """
    base: dict[str, Any] = copy.deepcopy(existing) if existing else {}
    hooks_block = base.setdefault("hooks", {})
    if not isinstance(hooks_block, dict):
        raise ValueError(
            "Claude settings.json has unexpected shape: 'hooks' must be an object"
        )

    # --- 1. strip our commands from every existing group ----------------
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

    # --- 2. append fresh entries ---------------------------------------
    for entry in additions:
        bucket = hooks_block.setdefault(entry.event, [])
        if not isinstance(bucket, list):
            raise ValueError(
                f"Claude settings.json has unexpected shape at hooks.{entry.event}: "
                "expected list"
            )
        bucket.append(_entry_from_hookentry(entry))

    return base


def diff_claude_hooks(
    existing: dict[str, Any] | None,
    additions: list[HookEntry],
) -> ClaudeHookDiff:
    """Compute the human-facing diff (what merge would do)."""
    diff = ClaudeHookDiff()
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
                        if is_claude_owned_command(command):
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


# ---------------------------------------------------------------------------
# v0.20.4: env block fill-missing merge for ~/.claude/settings.json
#
# Claude Code 的 native OTel pipeline 需要在 settings.json 顶层放一段 ``env``
# 字典（``CLAUDE_CODE_ENABLE_TELEMETRY=1`` + 一组 ``OTEL_*``）。我们想让
# ``agent-cot init --apply --agent claude`` 一键把这段配上，但绝对不能
# 覆盖同学已有的 env —— 公司内部 collector、langfuse 密钥、自定义
# resource attribute 都是同学预先配好的。所以语义是 **fill-missing only**：
# - 同学没设过的 key：补；
# - 同学设过的 key：原样保留，agent-cot 不动；
# - diff_claude_env 把这两类分别归到 .added / .preserved，方便 init dry-run
#   渲染 + doctor 检查 endpoint 是否指向当前 backend 端口。
# ---------------------------------------------------------------------------


@dataclass
class ClaudeEnvDiff:
    """What :func:`merge_claude_env` would do to ``settings.env``.

    ``added`` — env keys we would inject (currently absent from the user's
    settings.json).
    ``preserved`` — env keys we recommend but the user already set; we
    keep their value, never overwrite.
    """

    added: list[tuple[str, str]] = field(default_factory=list)
    preserved: list[tuple[str, str]] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.added)

    def render(self) -> str:
        lines: list[str] = []
        if not self.added and not self.preserved:
            lines.append("(no env changes — Claude OTel env not requested)")
            return "\n".join(lines)
        for k, v in self.added:
            lines.append(f"  + env.{k} = {v}")
        for k, v in self.preserved:
            lines.append(f"  · env.{k} = {v}  (kept — already set by you)")
        return "\n".join(lines)


def merge_claude_env(
    existing: dict[str, Any] | None,
    additions: dict[str, str],
) -> tuple[dict[str, Any], list[tuple[str, str]], list[tuple[str, str]]]:
    """Fill-missing-only merge of ``additions`` into ``existing["env"]``.

    Returns ``(new_settings_dict, added_pairs, preserved_pairs)``:

    * ``new_settings_dict`` — the full settings object with the env block
      patched (deep-copied; ``existing`` is never mutated).
    * ``added_pairs`` — ``[(key, value), ...]`` we just added.
    * ``preserved_pairs`` — ``[(key, value), ...]`` we left alone because
      the user had already set them. The value is the **user's** value,
      not our recommendation — this is what doctor needs to know whether
      e.g. ``OTEL_EXPORTER_OTLP_ENDPOINT`` points at our backend port.

    Never overwrites. Never raises on missing ``env`` block — we'll create
    it. Bails out only when ``settings["env"]`` exists but isn't a dict
    (rare; some users put a list there by accident).
    """
    base: dict[str, Any] = copy.deepcopy(existing) if existing else {}
    env_block = base.setdefault("env", {})
    if not isinstance(env_block, dict):
        raise ValueError(
            "Claude settings.json has unexpected shape: 'env' must be an object"
        )

    added: list[tuple[str, str]] = []
    preserved: list[tuple[str, str]] = []
    for key, recommended in additions.items():
        if key in env_block:
            preserved.append((key, str(env_block[key])))
            continue
        env_block[key] = recommended
        added.append((key, recommended))
    return base, added, preserved


def diff_claude_env(
    existing: dict[str, Any] | None,
    additions: dict[str, str],
) -> ClaudeEnvDiff:
    """Pure (no mutation) diff for ``--dry-run`` rendering."""
    diff = ClaudeEnvDiff()
    existing_env = {}
    if isinstance(existing, dict):
        env_block = existing.get("env")
        if isinstance(env_block, dict):
            existing_env = env_block
    for key, recommended in additions.items():
        if key in existing_env:
            diff.preserved.append((key, str(existing_env[key])))
        else:
            diff.added.append((key, recommended))
    return diff


__all__ = [
    "ClaudeEnvDiff",
    "ClaudeHookDiff",
    "diff_claude_env",
    "diff_claude_hooks",
    "is_claude_owned_command",
    "merge_claude_env",
    "merge_claude_hooks",
]
