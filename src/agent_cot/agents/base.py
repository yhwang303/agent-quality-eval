"""Abstract contract every agent adapter must satisfy.

Why the indirection: Cursor and Claude Code (and later Codex / Aider) each
have *their own* hook config schema, transcript shape and on-disk path
conventions. We isolate these differences here so that the rest of the CLI
(``commands/init.py``, ``commands/start.py``, ``installer/hooks_merger.py``,
…) can stay agent-agnostic.

This file MUST be backward-compatible: once published, removing or renaming
methods will break third-party adapters. New methods should be added as
``@property`` defaults or with a sensible base implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CursorCotError(Exception):
    """Base class so callers can ``except CursorCotError`` everywhere."""


class UnknownAgentError(CursorCotError):
    """Raised when ``--agent <name>`` does not match any registered adapter."""

    def __init__(self, name: str, available: list[str]):
        self.name = name
        self.available = available
        super().__init__(
            f"Unknown agent '{name}'. Available: {', '.join(available)}"
        )


class AgentNotImplementedError(CursorCotError, NotImplementedError):
    """Raised by adapter stubs that are scheduled for a future release.

    We deliberately subclass *both* :class:`NotImplementedError` (so generic
    error handling still works) and :class:`CursorCotError` (so the CLI can
    print a friendly multi-line message instead of a stack trace).
    """

    def __init__(self, agent: str, since_version: str, tracking: str | None = None):
        self.agent = agent
        self.since_version = since_version
        self.tracking = tracking
        msg = (
            f"Agent '{agent}' is not yet supported (planned for {since_version}). "
            f"See SETUP_PLAN.md §5 for the roadmap."
        )
        if tracking:
            msg += f" Tracking: {tracking}"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HookEntry:
    """One row to inject into the agent's hook config file.

    The ``owner`` field is what makes our merge idempotent: re-running
    ``agent-cot init`` strips every entry whose owner is ``"agent-cot"``
    and re-adds them, while leaving every other hook untouched.
    """

    event: str
    """Which hook event this fires on (e.g. ``Stop``, ``SessionEnd``)."""

    command: str
    """Shell command to run when the event fires."""

    type: str = "command"
    """Hook type. Almost always ``command``; reserved for future expansion."""

    description: str = ""
    """Human-readable label embedded into the JSON entry, useful when
    debugging an unfamiliar machine's ``hooks.json``."""

    owner: str = "agent-cot"
    """Stable marker we use to find *our own* entries on re-install."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Adapter-specific extra fields (e.g. Claude's ``matcher`` / ``timeout``)."""


# ---------------------------------------------------------------------------
# The adapter protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AgentAdapter(Protocol):
    """Interface every concrete agent adapter implements.

    Concrete adapters live next to this file (``cursor.py``, ``claude.py``).
    Use :class:`typing.Protocol` rather than ABC so duck-typed third-party
    adapters work without forced inheritance.
    """

    name: str
    """Lower-case canonical name (matches the ``--agent <name>`` CLI flag)."""

    display_name: str
    """Human-readable label used in CLI output."""

    minimum_version: str
    """Earliest agent-cot release that fully supports this adapter."""

    # -- Filesystem layout ----------------------------------------------------

    def hooks_config_path(self) -> Path:
        """Where the agent stores its hook configuration JSON."""
        ...

    def hooks_assets_dir(self) -> Path:
        """Directory the agent loads hook scripts from."""
        ...

    def additional_hooks_targets(self) -> list[tuple[Path, Path]]:
        """**Optional** v0.20.6+. Extra ``(settings_path, assets_dir)`` pairs
        that should receive a *mirror copy* of the primary write.

        Why this exists:
          Claude Code ships in two flavors that share the same hook + env
          schema but read from different home directories:

          * ``~/.claude/``           — Anthropic OSS ``claude`` CLI
          * ``~/.claude-internal/``  — Tencent ``@tencent/claude-code-internal``
            (the one Cursor embeds)

          A user can have **either or both** installed. When both exist,
          one ``agent-cot init --apply --agent claude`` should wire up
          both — otherwise the user's "primary" choice silently loses
          hooks + OTel env on the other variant.

        Return an empty list (the default) when this agent has nothing to
        mirror. Adapters that *do* mirror MUST return targets where the
        ``settings_path`` accepts the same ``new_hooks_blob`` shape and
        ``assets_dir`` accepts the same hook script filenames as the
        primary :meth:`hooks_config_path` / :meth:`hooks_assets_dir` —
        otherwise the mirror write will produce a broken config.
        """
        return []

    def bridge_files(self) -> list[str]:
        """Filenames of bundled hook scripts to install on init.

        Returning *filenames* (not absolute paths) keeps the abstraction
        clean: callers know to look up the source under
        ``agent_cot/assets/hooks/<agent>/<name>`` and to write the
        target at ``self.hooks_assets_dir() / <name>``.
        """
        ...

    # -- Detection ------------------------------------------------------------

    def detect_installed(self) -> bool:
        """Return ``True`` iff this agent appears to be installed locally.

        Used by ``agent-cot doctor`` and by ``--agent all`` to skip absent
        agents without erroring.
        """
        ...

    # -- Hook config manipulation --------------------------------------------

    def hook_entries(self) -> list[HookEntry]:
        """List of hook rows we want injected into the config file."""
        ...

    def merge_hook_entries(
        self,
        existing: dict[str, Any],
        additions: list[HookEntry],
    ) -> dict[str, Any]:
        """Merge ``additions`` into ``existing`` *idempotently*.

        Implementations MUST:

        1. First strip every entry whose ``owner == "agent-cot"``.
        2. Then append the rows in ``additions``.
        3. NEVER touch entries owned by other tools (e.g. ``codebuddy-mem``,
           ``langfuse_hook``).
        """
        ...

    def diff_hook_entries(
        self,
        existing: dict[str, Any] | None,
        additions: list[HookEntry],
    ) -> Any:
        """Compute a human-facing diff of what :meth:`merge_hook_entries` would do.

        Returns a :class:`agent_cot.installer.hooks_merger.HookDiff`-shaped object
        (any object with ``.added``, ``.removed``, ``.untouched_other_owners`` and
        ``.render()``). v0.17.0 引入；老 adapter 没实现时调用方应回退到 cursor 的
        ``diff_hooks``。
        """
        ...

    # -- Transcript ----------------------------------------------------------

    def transcript_glob(self) -> str:
        """Glob pattern locating raw transcript files for this agent.

        Used by ``cot-extractor`` so that one extractor binary can consume
        transcripts from any supported agent.
        """
        ...


__all__ = [
    "AgentAdapter",
    "AgentNotImplementedError",
    "CursorCotError",
    "HookEntry",
    "UnknownAgentError",
]
