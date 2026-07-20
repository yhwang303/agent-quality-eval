"""Agent adapter registry.

Each supported AI coding agent (Cursor, Claude Code, Codex CLI, ...) ships an
:class:`AgentAdapter` implementation. The registry below is the single
look-up point used by every CLI command, so adding a new agent in the future
means *only* dropping a new file under this package.

Design rationale: see ``SETUP_PLAN.md`` §1.4.
"""

from __future__ import annotations

from collections.abc import Iterable

from .base import (
    AgentAdapter,
    AgentNotImplementedError,
    CursorCotError,
    HookEntry,
    UnknownAgentError,
)
from .claude import ClaudeAdapter
from .codex import CodexAdapter
from .codebuddy import CodeBuddyAdapter
from .cursor import CursorAdapter

_REGISTRY: dict[str, type[AgentAdapter]] = {
    "cursor": CursorAdapter,
    "claude": ClaudeAdapter,
    "codex": CodexAdapter,
    "codebuddy": CodeBuddyAdapter,
}


def list_agents() -> list[str]:
    """Return the names of every registered agent adapter.

    Even agents whose adapter raises :class:`NotImplementedError` are listed
    here; we want CLI users to *see* the planned support in ``--help``.
    """
    return sorted(_REGISTRY.keys())


def get_adapter(name: str) -> AgentAdapter:
    """Resolve an agent adapter by name.

    The adapter is instantiated lazily so that listing agents (e.g. for
    ``--help``) never triggers expensive setup or import-time side effects.
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise UnknownAgentError(name=name, available=list_agents())
    return _REGISTRY[key]()


def iter_adapters(names: Iterable[str]) -> Iterable[AgentAdapter]:
    """Resolve multiple adapter names at once. ``"all"`` expands to everything."""
    out: list[str] = []
    for n in names:
        if n == "all":
            out.extend(list_agents())
        else:
            out.append(n)
    seen: set[str] = set()
    for n in out:
        if n in seen:
            continue
        seen.add(n)
        yield get_adapter(n)


__all__ = [
    "AgentAdapter",
    "AgentNotImplementedError",
    "ClaudeAdapter",
    "CodexAdapter",
    "CodeBuddyAdapter",
    "CursorAdapter",
    "CursorCotError",
    "HookEntry",
    "UnknownAgentError",
    "get_adapter",
    "iter_adapters",
    "list_agents",
]
