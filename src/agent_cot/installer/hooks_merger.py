"""Idempotent merge of agent-cot hooks into ``~/.cursor/hooks.json``.

Key contract (the **only** invariant we promise to users):

    Re-running ``agent-cot init`` any number of times produces the same
    final ``hooks.json``, AND every hook entry NOT owned by us
    (codebuddy-mem, hook-handler, the user's own scripts, …) is byte-
    for-byte unchanged.

How we identify "ours" without polluting Cursor's schema:

* Cursor's real schema is ``{"version":1,"hooks":{<event>:[{"command","timeout"}]}}``.
* We do NOT add any custom keys (``owner`` / ``$agent_cot``) at the
  entry level — Cursor might tighten its parser later.
* Instead we recognise our entries by the **command string** — the
  ``cot-bridge.js`` and ``cot-stream.js`` filenames are stable handles
  that we 100% control. Any entry whose command path ends in one of
  those filenames is considered owned by us.

This file is the single source of truth for that detection rule;
``commands/init.py`` and ``commands/uninstall.py`` both go through it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..agents.base import HookEntry

# Filenames we own. Stable across versions — changing these is a
# breaking change requiring an upgrade migration path.
#
# v0.17.0：扩展支持多 IDE。所有 agent-cot 注入的 hook 脚本统一以 cot-* 前缀
# 命名，再 owned 检测时只看尾部 basename（is_owned_command）。
OWNED_HOOK_SCRIPTS: frozenset[str] = frozenset(
    {
        # Cursor
        "cot-bridge.js",
        "cot-stream.js",
        # VSCode (Copilot Chat Agent hooks, Preview)
        "cot-stream-vscode.js",
        "cot-bridge-vscode.js",
        # CodeBuddy（Phase 3 用）
        "cot-stream-codebuddy.js",
        "cot-bridge-codebuddy.js",
        # Claude Internal (v0.19.1) — 唯一一个 .py hook（其它都是 .js）
        "claude_stream_hook.py",
        "agent_critic_hook.py",
    }
)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def is_owned_command(command: str) -> bool:
    """Return ``True`` if ``command`` invokes one of our hook scripts.

    Tolerant of:

    * forward / backward slashes (Windows quotes paths in either form);
    * surrounding quotes;
    * ``node`` / absolute path / ``cmd /c`` prefixes.

    Detection is intentionally narrow: we only match by *filename*,
    never by a substring like ``agent-cot`` that the user might use
    elsewhere.
    """
    if not isinstance(command, str) or not command:
        return False
    haystack = command.replace("\\", "/").lower()
    return any(f"/{name}".lower() in haystack for name in OWNED_HOOK_SCRIPTS)


# ---------------------------------------------------------------------------
# Diff for human-readable dry-run output
# ---------------------------------------------------------------------------


@dataclass
class HookDiff:
    """Summary of what a merge will change, for dry-run rendering."""

    added: list[tuple[str, str]] = field(default_factory=list)
    """Pairs of (event, command) we'll insert."""

    removed: list[tuple[str, str]] = field(default_factory=list)
    """Pairs of (event, command) we'll drop (only owned ones)."""

    untouched_other_owners: int = 0
    """How many entries belonging to other tools we kept as-is."""

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed)

    def render(self) -> str:
        lines: list[str] = []
        if not self.has_changes:
            lines.append("(no changes — hooks.json already in sync)")
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
# Internal helpers
# ---------------------------------------------------------------------------


def _entry_from_hookentry(entry: HookEntry) -> dict[str, Any]:
    """Render one :class:`HookEntry` into Cursor's flat schema.

    We deliberately *do not* emit a ``description`` / ``owner`` key
    — see module docstring. The ``extra`` dict is the only escape hatch
    for adapter-specific keys (``timeout``, future ``matcher``, ...).
    """
    out: dict[str, Any] = {"command": entry.command}
    out.update(entry.extra)
    out.setdefault("timeout", 30)
    return out


def _commands_match(a: str, b: str) -> bool:
    """Loose equality used to detect "this entry is the same as ours"."""
    norm_a = a.replace("\\", "/").strip().lower()
    norm_b = b.replace("\\", "/").strip().lower()
    return norm_a == norm_b


# ---------------------------------------------------------------------------
# Public merge / diff API
# ---------------------------------------------------------------------------


def merge_cursor_hooks(
    existing: dict[str, Any] | None,
    additions: list[HookEntry],
) -> dict[str, Any]:
    """Return a new ``hooks.json``-shaped dict with our hooks merged in.

    ``existing`` may be:

    * ``None`` / ``{}`` — fresh install, we synthesise the skeleton;
    * an already-populated dict in Cursor's real schema.

    Mutation rules (in order):

    1. Strip every entry whose ``command`` matches :func:`is_owned_command`.
       This guarantees re-running init doesn't pile up duplicate rows.
    2. For each addition, append a fresh entry to the appropriate event
       array (preserving insertion order — last items, so ours run after
       codebuddy-mem / hook-handler, matching the user's manual layout).
    3. NEVER touch ``existing["version"]`` or any ``hooks.<event>`` entry
       not owned by us.

    The function is pure — it does not mutate ``existing``.
    """
    base: dict[str, Any] = copy.deepcopy(existing) if existing else {}

    # Make sure the skeleton exists. Default version=1 matches every
    # Cursor build observed so far.
    base.setdefault("version", 1)
    hooks_block: dict[str, Any] = base.setdefault("hooks", {})
    if not isinstance(hooks_block, dict):
        # User had a malformed config; refuse to clobber arbitrarily and
        # let the caller surface the issue.
        raise ValueError(
            "hooks.json has unexpected shape: 'hooks' must be an object"
        )

    # --- Step 1: prune owned entries from every event array. ---------------
    for event, entries in list(hooks_block.items()):
        if not isinstance(entries, list):
            continue
        kept = [
            e
            for e in entries
            if not (isinstance(e, dict) and is_owned_command(e.get("command", "")))
        ]
        hooks_block[event] = kept

    # --- Step 2: append our additions in the order given. -----------------
    for entry in additions:
        rendered = _entry_from_hookentry(entry)
        bucket = hooks_block.setdefault(entry.event, [])
        if not isinstance(bucket, list):
            raise ValueError(
                f"hooks.json has unexpected shape at hooks.{entry.event}: "
                "expected list"
            )
        bucket.append(rendered)

    return base


def diff_hooks(
    existing: dict[str, Any] | None,
    additions: list[HookEntry],
) -> HookDiff:
    """Compute the human-facing diff for a merge — without performing it.

    Used by ``agent-cot init --dry-run`` to print what *would* happen.
    """
    diff = HookDiff()
    base_hooks = (existing or {}).get("hooks", {}) if existing else {}

    # Removals: count *all* owned entries currently in the file. They
    # might or might not be re-added (if `additions` no longer needs
    # the event), but we list them so the user sees them in dry-run.
    if isinstance(base_hooks, dict):
        for event, entries in base_hooks.items():
            if not isinstance(entries, list):
                continue
            for e in entries:
                if isinstance(e, dict) and is_owned_command(e.get("command", "")):
                    diff.removed.append((event, e.get("command", "")))
                else:
                    diff.untouched_other_owners += 1

    # Additions: only count the ones that are NOT already present
    # verbatim, so re-running init looks idempotent in the diff too.
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
            # We're going to remove + re-add the same line, but for the
            # human-readable diff that's a no-op; collapse it.
            try:
                diff.removed.remove((entry.event, entry.command))
            except ValueError:
                pass
            continue
        diff.added.append((entry.event, entry.command))

    return diff


__all__ = [
    "OWNED_HOOK_SCRIPTS",
    "HookDiff",
    "diff_hooks",
    "is_owned_command",
    "merge_cursor_hooks",
]
