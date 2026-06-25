"""Codex agent adapter.

Codex exposes official lifecycle hooks through ``$CODEX_HOME/hooks.json`` and
native OpenTelemetry through ``$CODEX_HOME/config.toml``.  This adapter wires
the hook side; ``agent_cot.installer.codex_config`` enables the native OTel
exporter during ``init --apply --agent codex``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

from ..installer.codex_config import codex_home
from .base import AgentAdapter, HookEntry


class CodexAdapter(AgentAdapter):
    name = "codex"
    display_name = "Codex"
    minimum_version = "1.0.5"

    _OFFICIAL_HOOK_EVENTS: tuple[str, ...] = (
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PermissionRequest",
        "PostToolUse",
        "PreCompact",
        "PostCompact",
        "SubagentStart",
        "SubagentStop",
        "Stop",
    )

    def hooks_config_path(self) -> Path:
        return codex_home() / "hooks.json"

    def hooks_assets_dir(self) -> Path:
        return codex_home() / "hooks"

    def bridge_files(self) -> list[str]:
        return ["codex_stream_hook.py", "codex_sidecar_collector.py"]

    def detect_installed(self) -> bool:
        home = codex_home()
        return home.is_dir() or shutil.which("codex") is not None

    def hook_entries(self) -> list[HookEntry]:
        hooks_dir = self.hooks_assets_dir().as_posix()
        script_path = f"{hooks_dir}/codex_stream_hook.py"
        python = Path(sys.executable or "python").as_posix()

        rows: list[HookEntry] = []
        for event in self._OFFICIAL_HOOK_EVENTS:
            extra: dict[str, Any] = {
                "timeout": 30,
                "statusMessage": "Recording Codex trace",
            }
            rows.append(
                HookEntry(
                    event=event,
                    command=f'"{python}" "{script_path}" {event}',
                    description="agent-cot: codex hook stream + sidecar collector",
                    extra=extra,
                )
            )
        return rows

    def merge_hook_entries(
        self,
        existing: dict[str, Any],
        additions: list[HookEntry],
    ) -> dict[str, Any]:
        from ..installer.codex_hooks_merger import merge_codex_hooks

        return merge_codex_hooks(existing, additions)

    def diff_hook_entries(
        self,
        existing: dict[str, Any] | None,
        additions: list[HookEntry],
    ) -> Any:
        from ..installer.codex_hooks_merger import diff_codex_hooks

        return diff_codex_hooks(existing, additions)

    def transcript_glob(self) -> str:
        return str(codex_home() / "sessions" / "*" / "*" / "*" / "*.jsonl")


__all__ = ["CodexAdapter"]
