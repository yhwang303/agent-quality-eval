"""Claude Code (Internal) agent adapter.

v0.19.1 把这个 adapter 从 stub 升级成 **完整实装**：

1. ``hook_entries()`` 返回 27 个 hook event 的注册项，每条命令形如
   ``python "<hooks_dir>/claude_stream_hook.py" <Event>``。``Event``
   名作为 argv[1] 显式传入，防止某些 Claude Code 版本不在 stdin payload
   里写 ``hook_event_name`` 时 hook 把所有事件混淆为 ``Unknown``。

2. ``merge_hook_entries()`` / ``diff_hook_entries()`` 委托给新的
   :mod:`agent_cot.installer.claude_hooks_merger` —— 它跟 codebuddy 的
   merger 实现独立但 schema 一致（嵌套 ``hooks.<Event>[].hooks[]``）。

3. ``hooks_assets_dir()`` = ``~/.claude/hooks`` —— 即 settings.json 引用的
   绝对路径起点；``bridge_files()`` 只列 ``claude_stream_hook.py``，
   ``langfuse_hook.py`` 是用户自选 / 跟 langfuse 关联的额外组件，agent-cot
   不接管（避免无意中改用户的 langfuse 配置）。

4. ``transcript_glob()`` 同时覆盖：
   * Claude **Internal**: ``~/.claude-internal/projects/<slug>/<sid>.jsonl``
   * Claude **OSS**: ``~/.claude/projects/<slug>/<sid>.jsonl``

   这两条是 cot_extractor 在 ``_candidate_projects_dirs()`` 里就已经枚举
   过的真实路径，跟现场用户的 Claude 安装态对齐。

设计约束：
* **不修改 Cursor / CodeBuddy 任何代码** —— Claude 拿自己独立的 merger /
  ownership 检测器 / hook 资产目录；与已有两个 IDE 完全解耦。
* **不硬编码任何机器路径** —— hook 脚本内部用 4 层 fallback
  （env > ``.agent-cot/runtime.json`` > ``.cursor-cot/runtime.json`` >
  探测）找 cot-extractor + python；adapter 这边只把 hook 脚本投放到
  ``~/.claude/hooks/`` 并写命令字符串，绝不在命令里写死 extractor 路径。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import AgentAdapter, HookEntry


class ClaudeAdapter(AgentAdapter):
    name = "claude"
    display_name = "Claude Internal"
    minimum_version = "0.19.1"

    # -- Filesystem layout (public; tests pin against these) -----------------
    #
    # v0.20.6: Claude Code ships in two flavors:
    #   * ``~/.claude/``           — Anthropic OSS ``claude`` CLI
    #   * ``~/.claude-internal/``  — Tencent ``@tencent/claude-code-internal``
    #     (the variant Cursor embeds — see :func:`transcript_glob`)
    # Both read settings.json from their own home dir and never look at the
    # other. Pre-0.20.6 we only wrote to ``~/.claude/`` which silently meant
    # Cursor-embedded Claude Code never got our hooks / OTel env. From 0.20.6
    # on, ``hooks_config_path`` picks the variant that's actually installed:
    # internal-only users get .claude-internal as primary; OSS-only users get
    # .claude (unchanged); users who have **both** get one as primary and the
    # other as a mirror target via :meth:`additional_hooks_targets`.

    # Computed lazily via properties so test fixtures that monkeypatch
    # ``Path.home`` (the common pattern in tests/test_init_command.py) get
    # the redirected home rather than the home that was current at class
    # definition time. Class-level constants would freeze the real home
    # at import time and break every test.
    @property
    def _CLAUDE_OSS_HOME(self) -> Path:  # noqa: N802 (kept upper-case for test compat)
        return Path.home() / ".claude"

    @property
    def _CLAUDE_INTERNAL_HOME(self) -> Path:  # noqa: N802
        return Path.home() / ".claude-internal"

    @property
    def _CLAUDE_INTERNAL_COMPAT_HOME(self) -> Path:  # noqa: N802
        # Some internal wrappers have shipped the misspelled directory name.
        # Treat it as an alias, but never create it unless it already exists.
        return Path.home() / ".claude-inertnal"

    def _internal_homes(self) -> tuple[Path, ...]:
        return (self._CLAUDE_INTERNAL_HOME, self._CLAUDE_INTERNAL_COMPAT_HOME)

    def _primary_home(self) -> Path:
        """Pick which Claude home owns the *primary* write.

        Preference order:

        1. If only ``~/.claude-internal/`` exists → use it (Cursor users).
        2. Otherwise → use ``~/.claude/`` (OSS default; also the path that
           pre-0.20.6 always used, so this stays backward-compatible for
           single-OSS-install machines).

        When **both** exist we still return ``~/.claude/`` as primary, and
        ``additional_hooks_targets()`` will mirror the write to
        ``~/.claude-internal/``. The "primary" choice is only cosmetic
        (it's what gets shown in ``init`` summary headlines) — the mirror
        ensures both variants are functionally identical regardless.
        """
        existing_internal = [home for home in self._internal_homes() if home.is_dir()]
        if existing_internal and not self._CLAUDE_OSS_HOME.is_dir():
            return existing_internal[0]
        return self._CLAUDE_OSS_HOME

    def hooks_config_path(self) -> Path:
        # Claude Code stores hooks under settings.json (NOT a dedicated hooks.json).
        return self._primary_home() / "settings.json"

    def hooks_assets_dir(self) -> Path:
        return self._primary_home() / "hooks"

    def additional_hooks_targets(self) -> list[tuple[Path, Path]]:
        """Return the *other* Claude home as a mirror target when both exist.

        Empty list when only one variant is installed (no mirroring needed).
        Non-empty only when BOTH ``~/.claude/`` and ``~/.claude-internal/``
        are real directories on disk.
        """
        primary = self._primary_home()
        targets: list[tuple[Path, Path]] = []
        for home in (self._CLAUDE_OSS_HOME, *self._internal_homes()):
            if home == primary:
                continue
            if home.is_dir():
                targets.append((home / "settings.json", home / "hooks"))
        return targets

    def bridge_files(self) -> list[str]:
        # claude_stream_hook.py is the v0.19.1 Python hook that BOTH
        #   1) appends to ~/.claude/state/events/<sid>/events.jsonl, AND
        #   2) on Stop / SubagentStop / SessionEnd / StopFailure events spawns
        #      extract_cot.py in the background to materialise cot.json.
        # langfuse_hook.py is intentionally NOT listed here — it's an
        # opt-in third-party companion (see assets/hooks/claude/langfuse_hook.py)
        # that users wire up themselves if they want Langfuse upload; agent-cot
        # neither requires nor manages it, so we don't touch their config.
        return ["claude_stream_hook.py"]

    # -- Detection (cheap & safe) --------------------------------------------

    def detect_installed(self) -> bool:
        # v0.20.6: either Claude variant counts as "installed". Pre-0.20.6
        # only checked ``~/.claude/`` which made ``--agent all`` silently
        # skip Cursor-only users (who have ``~/.claude-internal/`` but no
        # ``~/.claude/`` because they never ran the OSS CLI).
        return self._CLAUDE_OSS_HOME.is_dir() or any(home.is_dir() for home in self._internal_homes())

    # -- Hook entries ---------------------------------------------------------

    # All 27 events that Claude Code currently fires. Worth subscribing to
    # ALL of them because:
    # * The "session boundary" events (Stop / SubagentStop / SessionEnd /
    #   StopFailure) trigger extract_cot.py, which is what makes cot.json
    #   show up on the dashboard.
    # * The other events feed events.jsonl — used by cot_extractor's
    #   ``_attach_claude_hook_events`` to enrich SessionCoT with the
    #   5 Claude-only timelines (compact / subagent / permission /
    #   notification / environment events) that you can see in the UI as
    #   the right-side chips ("Subagent×N", "PermissionRequest", etc.).
    #
    # Order matches what's documented in claude_stream_hook.py::ALL_EVENTS
    # so install / uninstall produce identical diffs regardless of caller.
    _ALL_HOOK_EVENTS: tuple[str, ...] = (
        "SessionStart",
        "SessionEnd",
        "Setup",
        "UserPromptSubmit",
        "Stop",
        "StopFailure",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "SubagentStart",
        "SubagentStop",
        "TaskCreated",
        "TaskCompleted",
        "TeammateIdle",
        "PreCompact",
        "PostCompact",
        "PermissionRequest",
        "PermissionDenied",
        "Notification",
        "Elicitation",
        "ElicitationResult",
        "ConfigChange",
        "InstructionsLoaded",
        "CwdChanged",
        "FileChanged",
        "WorktreeCreate",
        "WorktreeRemove",
    )

    def _hook_entries_for_assets_dir(self, assets_dir: Path) -> list[HookEntry]:
        """Build the 27 ``HookEntry`` rows we want injected into settings.json.

        Command shape: ``python "<hooks_dir>/claude_stream_hook.py" <Event>``.

        * ``python`` (no absolute path) — Claude Code resolves it via PATH
          on the user's machine. We deliberately DO NOT bake an absolute
          ``sys.executable`` into the command, because:
            - Claude Code processes inherit the user's full PATH (unlike
              Cursor's ``cot-bridge.js`` which Cursor spawns with a thin
              env, where we DO need a literal python).
            - Allowing PATH-based resolution makes the Claude hook resilient
              to venv switches: rebuild the venv, the hook just keeps working
              because the hook script ITSELF resolves the runtime python via
              4-layer fallback (env > .agent-cot/runtime.json > .cursor-cot/
              runtime.json > sys.executable probe).
        * Event name as argv[1] — backstop for Claude versions that don't
          inject ``hook_event_name`` into the JSON payload.
        * Forward slashes in the path — works on both Windows (cmd.exe
          accepts them) and POSIX (the only path separator).
        """
        hooks_dir = Path(assets_dir).as_posix()
        script_path = f"{hooks_dir}/claude_stream_hook.py"

        rows: list[HookEntry] = []
        for ev in self._ALL_HOOK_EVENTS:
            rows.append(
                HookEntry(
                    event=ev,
                    command=f'python "{script_path}" {ev}',
                    description="agent-cot: claude stream hook + extract_cot trigger",
                )
            )
        return rows

    def hook_entries(self) -> list[HookEntry]:
        return self._hook_entries_for_assets_dir(self.hooks_assets_dir())

    def hook_entries_for_assets_dir(self, assets_dir: Path) -> list[HookEntry]:
        return self._hook_entries_for_assets_dir(Path(assets_dir))

    # -- OTel env recommendations (v0.20.4) -----------------------------------

    # The full set of env vars we want present in ~/.claude/settings.json so
    # Claude Code's native OpenTelemetry pipeline ships logs + metrics + (opt-in)
    # raw bodies to OUR local FastAPI backend, where ``claude_otel_receiver``
    # buckets them under ``~/.claude/state/otel/<sid>/`` and the dashboard's
    # ClaudeOtelPanel renders the prompt timeline + API-true cost / tokens.
    #
    # The endpoint value is supplied at call-time by :meth:`recommended_env`
    # because the actual backend port is discovered by ``agent-cot init`` (or
    # ``start``) at runtime, so we MUST avoid the historical hardcoded
    # ``http://localhost:8765`` that silently dropped events when port 8765
    # was occupied and ``pick_port`` fell back to 8766.
    _OTEL_ENV_TEMPLATE: tuple[tuple[str, str], ...] = (
        ("CLAUDE_CODE_ENABLE_TELEMETRY", "1"),
        ("OTEL_LOGS_EXPORTER", "otlp"),
        ("OTEL_METRICS_EXPORTER", "otlp"),
        # http/json (not http/protobuf) — claude_otel_receiver.py parses JSON only.
        ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
        # OTEL_EXPORTER_OTLP_ENDPOINT injected dynamically per backend_port.
        ("OTEL_LOG_USER_PROMPTS", "1"),
        ("OTEL_LOG_TOOL_DETAILS", "1"),
        # 10s / 5s instead of the default 60s — makes the dashboard refresh
        # mid-conversation instead of feeling broken until the next minute.
        ("OTEL_METRIC_EXPORT_INTERVAL", "10000"),
        ("OTEL_LOGS_EXPORT_INTERVAL", "5000"),
    )

    OTEL_ENDPOINT_KEY: str = "OTEL_EXPORTER_OTLP_ENDPOINT"
    """Public so :mod:`agent_cot.commands.start` can self-heal the URL when
    the actual backend port differs from what ``init`` originally wrote."""

    @classmethod
    def otel_endpoint_for_port(cls, backend_port: int) -> str:
        """Canonical OTLP/HTTP endpoint URL pointing at our local backend.

        Uses ``127.0.0.1`` rather than ``localhost`` deliberately —
        Windows Defender and some corporate proxies block IPv4 loopback
        via the ``localhost`` hostname but allow the literal IP. This
        matches the troubleshooting recommendation in ``CLAUDE_OTEL.md``.
        """
        return f"http://127.0.0.1:{int(backend_port)}"

    def recommended_env(self, *, backend_port: int) -> dict[str, str]:
        """Env block we want present in ``~/.claude/settings.json``.

        Plugging this into Claude Code lights up the **native OTel channel**
        (the third independent signal source alongside transcripts + hooks),
        which gives the dashboard's right-panel ``ClaudeOtelPanel`` the
        official API-true ``input_tokens / output_tokens / cost_usd`` and
        the ``prompt.id`` linkage that ties one user prompt to its api_request
        and tool_result events.

        The exact key set is documented in ``assets/backend/CLAUDE_OTEL.md``.
        Callers (``init``, ``start`` self-heal) feed the returned dict to
        :func:`agent_cot.installer.claude_hooks_merger.merge_claude_env`
        which does fill-missing-only merging — pre-existing keys configured
        by the user (corporate collector, langfuse, custom resource attrs)
        are NEVER overwritten.
        """
        env = {k: v for k, v in self._OTEL_ENV_TEMPLATE}
        env[self.OTEL_ENDPOINT_KEY] = self.otel_endpoint_for_port(backend_port)
        return env

    # -- Merge / diff ---------------------------------------------------------

    def merge_hook_entries(
        self,
        existing: dict[str, Any],
        additions: list[HookEntry],
    ) -> dict[str, Any]:
        from ..installer.claude_hooks_merger import merge_claude_hooks

        return merge_claude_hooks(existing, additions)

    def diff_hook_entries(
        self,
        existing: dict[str, Any] | None,
        additions: list[HookEntry],
    ) -> Any:
        from ..installer.claude_hooks_merger import diff_claude_hooks

        return diff_claude_hooks(existing, additions)

    # -- Transcript ----------------------------------------------------------

    def transcript_glob(self) -> str:
        """Where Claude writes raw transcripts on disk.

        Both layouts are real and live in the field:

        * **Claude Internal** (the version users typically have via Tencent
          deployment) writes to ``~/.claude-internal/projects/<slug>/<sid>.jsonl``.
        * **Claude OSS** (claude-code package on npm) writes to
          ``~/.claude/projects/<slug>/<sid>.jsonl``.

        cot_extractor's ``_candidate_projects_dirs()`` already enumerates
        BOTH so ``extract_cot.py`` finds the transcript regardless of
        which Claude variant the user has. We return the Internal path
        here because that's the canonical case for Tencent-internal users
        (and the one cot-extractor will hit first); doctor can fall back
        to OSS via the same enumeration.
        """
        return str(
            self._CLAUDE_INTERNAL_HOME / "projects" / "*" / "*.jsonl"
        )


__all__ = ["ClaudeAdapter"]
