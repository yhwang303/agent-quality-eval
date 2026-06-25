"""Idempotent merge of agent-cot hooks into VSCode Copilot Chat ``hooks.json``.

Target file: ``~/.copilot/hooks.json`` (用户级) 或 ``.github/hooks/<name>.json``
（工作区级；本期只处理用户级）。

Schema差异 vs Cursor：
    Cursor:  {"version":1,"hooks":{"<event>":[{"command":..,"timeout":..}]}}
    Copilot: {"hooks":{"<Event>":[{"type":"command","command":".."}]}}

差别两处：
1. 顶层无 ``version`` 字段
2. 每个 entry 必须带 ``"type":"command"``（VSCode Agent hooks Preview 文档要求）

Owned 识别 / 幂等性合约 / 兜底逻辑全部跟 :mod:`hooks_merger` 一致——这里
只复用 ``is_owned_command`` 和 ``HookEntry`` 数据载体，不重写 detection。
"""
from __future__ import annotations

import copy
from typing import Any

from ..agents.base import HookEntry
from .hooks_merger import HookDiff, _commands_match, is_owned_command


def _entry_from_hookentry(entry: HookEntry) -> dict[str, Any]:
    """Render :class:`HookEntry` into VSCode Copilot 的 ``{type, command}`` shape。

    与 Cursor merger 不同的两点：
    * 必须显式设置 ``type=command``（GitHub Agent hooks Preview 强制要求）；
    * 不写 ``timeout``（VSCode 还没文档化这个字段，设了可能被拒）。
    """
    out: dict[str, Any] = {
        "type": entry.type or "command",
        "command": entry.command,
    }
    # 把 adapter 自定义的 extra 透传过去（例如 ``matcher``），但显式禁止 timeout
    # 等 Cursor 专属字段污染 VSCode schema。
    for k, v in (entry.extra or {}).items():
        if k in {"timeout"}:
            continue
        out[k] = v
    return out


def merge_copilot_hooks(
    existing: dict[str, Any] | None,
    additions: list[HookEntry],
) -> dict[str, Any]:
    """合并 ``additions`` 进 VSCode Copilot ``hooks.json``，幂等且不破坏第三方条目。

    步骤：

    1. 复制 existing → base，确保 ``base["hooks"]`` 是 dict；
    2. 对每个 event array 移除所有 ``is_owned_command`` 命中的条目；
    3. 按 additions 顺序追加。

    与 :func:`merge_cursor_hooks` 唯一逻辑差别就是 entry shape（多一个 ``type`` 字段）
    和不写 ``version``。
    """
    base: dict[str, Any] = copy.deepcopy(existing) if existing else {}
    hooks_block: dict[str, Any] = base.setdefault("hooks", {})
    if not isinstance(hooks_block, dict):
        raise ValueError(
            "copilot hooks.json has unexpected shape: 'hooks' must be an object"
        )

    # Step 1: prune owned
    for event, entries in list(hooks_block.items()):
        if not isinstance(entries, list):
            continue
        kept = [
            e
            for e in entries
            if not (isinstance(e, dict) and is_owned_command(e.get("command", "")))
        ]
        hooks_block[event] = kept

    # Step 2: append
    for entry in additions:
        rendered = _entry_from_hookentry(entry)
        bucket = hooks_block.setdefault(entry.event, [])
        if not isinstance(bucket, list):
            raise ValueError(
                f"copilot hooks.json has unexpected shape at hooks.{entry.event}: "
                "expected list"
            )
        bucket.append(rendered)

    return base


def diff_copilot_hooks(
    existing: dict[str, Any] | None,
    additions: list[HookEntry],
) -> HookDiff:
    """跟 :func:`diff_hooks` 同语义，只是从 Copilot schema 里读 hooks block。"""
    diff = HookDiff()
    base_hooks = (existing or {}).get("hooks", {}) if existing else {}

    if isinstance(base_hooks, dict):
        for event, entries in base_hooks.items():
            if not isinstance(entries, list):
                continue
            for e in entries:
                if isinstance(e, dict) and is_owned_command(e.get("command", "")):
                    diff.removed.append((event, e.get("command", "")))
                else:
                    diff.untouched_other_owners += 1

    if isinstance(base_hooks, dict):
        existing_owned: dict[str, list[str]] = {
            ev: [e["command"] for e in es if isinstance(e, dict) and "command" in e]
            for ev, es in base_hooks.items()
            if isinstance(es, list)
        }
    else:
        existing_owned = {}

    for entry in additions:
        present = any(
            _commands_match(c, entry.command)
            for c in existing_owned.get(entry.event, [])
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
    "diff_copilot_hooks",
    "merge_copilot_hooks",
]
