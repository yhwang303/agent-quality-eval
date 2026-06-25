"""Cursor IDE agent adapter.

This is the *only* fully-implemented adapter for v0.13. Mirrors the layout
already documented in ``cot-extractor/CURSOR_GLOBAL_INTEGRATION.md``:

- Hook config:  ``~/.cursor/hooks.json``
- Hook scripts: ``~/.cursor/hooks/cot-bridge.js`` and ``cot-stream.js``

Note: this file currently provides the *interface surface* (paths, hook
rows, detection). Real merge logic lives in :mod:`agent_cot.installer`
and is wired up in P1 of ``SETUP_PLAN.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import AgentAdapter, HookEntry


class CursorAdapter(AgentAdapter):
    name = "cursor"
    display_name = "Cursor"
    minimum_version = "0.13.0"

    # -- Filesystem layout ----------------------------------------------------

    def hooks_config_path(self) -> Path:
        return Path.home() / ".cursor" / "hooks.json"

    def hooks_assets_dir(self) -> Path:
        return Path.home() / ".cursor" / "hooks"

    def bridge_files(self) -> list[str]:
        return ["cot-bridge.js", "cot-stream.js"]

    # -- Detection ------------------------------------------------------------

    def detect_installed(self) -> bool:
        # Cursor's per-user dir is created on first launch. Existence of
        # the directory is enough to say "this user runs Cursor".
        return (Path.home() / ".cursor").is_dir()

    # -- Hook config manipulation --------------------------------------------

    # The full set of "high-frequency" event hooks we want cot-stream to
    # listen on. These mirror what the user already has working manually
    # (cot-stream is a fast, non-blocking event tap), and what
    # cot-extractor's stream parser knows how to consume.
    _STREAM_EVENTS: tuple[str, ...] = (
        "afterAgentResponse",
        "afterAgentThought",
        "beforeShellExecution",
        "afterShellExecution",
        "beforeMCPExecution",
        "afterMCPExecution",
        "beforeReadFile",
        "afterFileEdit",
    )

    # The "session boundary" event: at turn stop we run the heavy
    # cot-extractor pass that materialises the cot.json snapshot.
    _BRIDGE_EVENT: str = "stop"

    def hook_entries(self) -> list[HookEntry]:
        # Real Cursor schema (verified on user's machine 2026-04-27):
        #   { "version": 1, "hooks": { "<eventCamelCase>": [
        #       { "command": "<shell>", "timeout": <int> }, ... ] } }
        # Note: events are lower-camelCase, entries carry no "type" field.
        hooks_dir = self.hooks_assets_dir().as_posix()
        bridge_cmd = f'node "{hooks_dir}/cot-bridge.js"'
        stream_cmd = f'node "{hooks_dir}/cot-stream.js"'

        rows: list[HookEntry] = [
            HookEntry(
                event=self._BRIDGE_EVENT,
                command=bridge_cmd,
                description="agent-cot: extract transcript on turn stop",
                extra={"timeout": 10},
            )
        ]
        for ev in self._STREAM_EVENTS:
            rows.append(
                HookEntry(
                    event=ev,
                    command=stream_cmd,
                    description="agent-cot: stream event tap",
                    extra={"timeout": 5},
                )
            )
        return rows

    def merge_hook_entries(
        self,
        existing: dict[str, Any],
        additions: list[HookEntry],
    ) -> dict[str, Any]:
        # Delegate to the central merger so every adapter shares one
        # idempotent, well-tested implementation. Imported lazily to
        # avoid a circular import at module-load time.
        from ..installer.hooks_merger import merge_cursor_hooks

        return merge_cursor_hooks(existing, additions)

    def diff_hook_entries(
        self,
        existing: dict[str, Any] | None,
        additions: list[HookEntry],
    ) -> Any:
        from ..installer.hooks_merger import diff_hooks

        return diff_hooks(existing, additions)

    # -- Transcript ----------------------------------------------------------

    def transcript_glob(self) -> str:
        # Cursor stores transcripts under ~/.cursor/agent-transcripts/<uuid>/<uuid>.jsonl
        return str(Path.home() / ".cursor" / "agent-transcripts" / "*" / "*.jsonl")


__all__ = ["CursorAdapter"]
