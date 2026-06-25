"""CodeBuddy IDE adapter (Tencent Cloud Code Assistant).

The real user-level hook config lives at ``~/.codebuddy/settings.json`` and
uses a Claude-like nested hook schema:

    {"hooks": {"PostToolUse": [{"matcher": "...", "hooks": [
        {"type": "command", "command": "...", "timeout": 10000}
    ]}]}}

This adapter keeps CodeBuddy separate from the Cursor lower-camel hook schema
so installing agent-cot never rewrites the user's existing CodeBuddy plugins
or third-party hooks.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import AgentAdapter, HookEntry


class CodeBuddyAdapter(AgentAdapter):
    name = "codebuddy"
    display_name = "CodeBuddy"
    minimum_version = "0.17.0"

    # -- Filesystem layout ----------------------------------------------------

    def hooks_config_path(self) -> Path:
        return Path.home() / ".codebuddy" / "settings.json"

    def hooks_assets_dir(self) -> Path:
        return Path.home() / ".codebuddy" / "hooks"

    def bridge_files(self) -> list[str]:
        return ["cot-stream-codebuddy.js"]

    # -- Detection ------------------------------------------------------------

    def detect_installed(self) -> bool:
        return (Path.home() / ".codebuddy").is_dir()

    # -- Hook config manipulation --------------------------------------------

    # CodeBuddy's documented user hook events use PascalCase. We subscribe to
    # lifecycle/tool/subagent/compaction events that can be emitted without
    # relying on Cursor-only afterAgentThought hooks.
    _STREAM_EVENTS: tuple[str, ...] = (
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "PermissionRequest",
        "PermissionDenied",
        "Notification",
        "SubagentStart",
        "SubagentStop",
        "TaskCreated",
        "TaskCompleted",
        "Stop",
        "StopFailure",
        "PreCompact",
        "PostCompact",
        "ConfigChange",
        "CwdChanged",
        "SessionEnd",
    )

    _EXPERIMENTAL_THOUGHT_EVENTS: tuple[str, ...] = (
        "AfterAgentThought",
        "AgentThought",
        "AfterAgentResponse",
    )

    def hook_entries(self) -> list[HookEntry]:
        hooks_dir = self.hooks_assets_dir().as_posix()
        stream_cmd_template = f'node "{hooks_dir}/cot-stream-codebuddy.js"'
        events = list(self._STREAM_EVENTS)
        if os.environ.get("AGENT_COT_CODEBUDDY_EXPERIMENTAL_THOUGHT_HOOKS", "").lower() in {
            "1", "true", "yes", "on",
        }:
            events.extend(self._EXPERIMENTAL_THOUGHT_EVENTS)

        rows: list[HookEntry] = []
        for ev in events:
            rows.append(
                HookEntry(
                    event=ev,
                    command=f"{stream_cmd_template} {ev}",
                    description="agent-cot: codebuddy stream event tap",
                    extra={"timeout": 10000},
                )
            )
        return rows

    def merge_hook_entries(
        self,
        existing: dict[str, Any],
        additions: list[HookEntry],
    ) -> dict[str, Any]:
        from ..installer.codebuddy_hooks_merger import merge_codebuddy_hooks

        return merge_codebuddy_hooks(existing, additions)

    def diff_hook_entries(
        self,
        existing: dict[str, Any] | None,
        additions: list[HookEntry],
    ) -> Any:
        from ..installer.codebuddy_hooks_merger import diff_codebuddy_hooks

        return diff_codebuddy_hooks(existing, additions)

    # -- Transcript ----------------------------------------------------------

    def transcript_glob(self) -> str:
        """Where CodeBuddy writes its native conversation history.

        Verified empirically on Windows with CodeBuddyIDE 4.9.8:
        ``%LOCALAPPDATA%/CodeBuddyExtension/Data/<machine>/CodeBuddyIDE/
        <machine>/history/<workspace>/<session_id>/index.json``
        plus a sibling ``messages/<msg_id>.json`` per message.

        We point the glob at ``index.json`` because that is what the
        ``cot_extractor`` CodeBuddy parser keys off of. The fallback
        ``~/.codebuddy/sessions/*.jsonl`` is kept only as a no-op for
        very old / unknown CodeBuddy variants — it just won't match.
        """
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            return str(
                Path(local_appdata)
                / "CodeBuddyExtension" / "Data"
                / "*" / "CodeBuddyIDE" / "*"
                / "history" / "*" / "*" / "index.json"
            )
        return str(Path.home() / ".codebuddy" / "sessions" / "*.jsonl")


__all__ = ["CodeBuddyAdapter"]
