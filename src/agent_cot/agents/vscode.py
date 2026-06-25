"""VSCode (GitHub Copilot Chat Agent hooks, Preview) adapter.

v0.17.0 引入。VSCode 1.95+ 提供了和 Cursor / Claude 类似的 Agent hooks 机制
（详见 https://code.visualstudio.com/docs/copilot/customization/hooks ），允许
我们在 ``~/.copilot/hooks.json`` 注册外部命令，IDE 在生命周期事件触发时通过
stdin JSON 投递 payload。

跟其它 adapter 的对比：

==============  ===============================  ==============================
                Cursor (CursorAdapter)            VSCode (VSCodeAdapter, 本文件)
==============  ===============================  ==============================
配置文件        ~/.cursor/hooks.json              ~/.copilot/hooks.json
schema          {version,hooks:{ev:[{cmd,to}]}}   {hooks:{Ev:[{type,command}]}}
事件命名        lowerCamelCase（afterFileEdit）   PascalCase（AfterToolUse）
hook 脚本       node cot-stream.js                node cot-stream-vscode.js
session 隔离    无前缀                            会话目录前缀 vscode-（多 IDE 共存）
==============  ===============================  ==============================

⚠️ Preview：GitHub Agent hooks 的事件名集合和 stdin schema 仍可能小幅调整。
本文件维护一个保守的事件名列表，覆盖最常用的生命周期点；hook 脚本本身
通过广泛的字段名 fallback 适配后续可能的变更（详见 cot-stream-vscode.js）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import AgentAdapter, HookEntry


class VSCodeAdapter(AgentAdapter):
    name = "vscode"
    display_name = "VSCode (Copilot Chat)"
    minimum_version = "0.17.0"

    # -- Filesystem layout ----------------------------------------------------

    def hooks_config_path(self) -> Path:
        # GitHub Copilot Chat Agent hooks 用户级配置位置
        # （workspace 级在 .github/hooks/*.json，本期不处理）
        return Path.home() / ".copilot" / "hooks.json"

    def hooks_assets_dir(self) -> Path:
        # 跟 hooks.json 同目录，方便 VSCode 自身加载、也方便用户排查
        return Path.home() / ".copilot" / "hooks"

    def bridge_files(self) -> list[str]:
        # v0.17.0：只装 stream，不带 bridge——因为 VSCode 的 transcript 落盘格式
        # 还在 state.vscdb（二进制，不稳定），暂时不做 stop 时的全量提取，
        # 主要靠 OTel 通道补齐丰富数据。
        return ["cot-stream-vscode.js"]

    # -- Detection ------------------------------------------------------------

    def detect_installed(self) -> bool:
        """VSCode 自身用户级目录（settings.json 所在）的常见路径任一存在即视为已装。

        注意：detect_installed 是给 ``--agent all`` 用的"装了才装我"的兜底，
        所以不需要严格——只要有 VSCode 痕迹就行。
        """
        # 1. macOS / Linux 用户配置
        if (Path.home() / ".config" / "Code" / "User" / "settings.json").exists():
            return True
        # 2. Windows 用户配置（%APPDATA%\Code\User\settings.json）
        appdata = Path.home() / "AppData" / "Roaming" / "Code" / "User" / "settings.json"
        if appdata.exists():
            return True
        # 3. ~/.copilot 已经存在（用户已经手动改过 settings 启用 Copilot Chat）
        if (Path.home() / ".copilot").is_dir():
            return True
        return False

    # -- Hook config manipulation --------------------------------------------

    # GitHub Copilot Agent hooks Preview 阶段的事件集合（基于公开文档 +
    # Claude 27 hooks 的子集对照推测，每个事件未来可能改名）。
    # 我们只挂常用 9 个：覆盖 SessionStart/End、tool 调用前后、shell、prompt 提交。
    _STREAM_EVENTS: tuple[str, ...] = (
        "SessionStart",
        "SessionEnd",
        "UserPromptSubmit",
        "AfterAgentResponse",
        "BeforeToolUse",
        "AfterToolUse",
        "BeforeShellExecution",
        "AfterShellExecution",
        "Stop",
    )

    def hook_entries(self) -> list[HookEntry]:
        hooks_dir = self.hooks_assets_dir().as_posix()
        stream_cmd_template = f'node "{hooks_dir}/cot-stream-vscode.js"'

        rows: list[HookEntry] = []
        for ev in self._STREAM_EVENTS:
            # 每个事件追加事件名作为 argv，方便 cot-stream-vscode.js 当 stdin 缺
            # ``hook_event_name`` 时回退到命令行参数（跟 claude_stream_hook.py 一样）
            rows.append(
                HookEntry(
                    event=ev,
                    command=f"{stream_cmd_template} {ev}",
                    description="agent-cot: vscode/copilot stream event tap",
                    type="command",
                )
            )
        return rows

    def merge_hook_entries(
        self,
        existing: dict[str, Any],
        additions: list[HookEntry],
    ) -> dict[str, Any]:
        # 用 VSCode 专用 merger（schema 跟 Cursor 不同：嵌套 + type 字段）
        from ..installer.copilot_hooks_merger import merge_copilot_hooks

        return merge_copilot_hooks(existing, additions)

    def diff_hook_entries(
        self,
        existing: dict[str, Any] | None,
        additions: list[HookEntry],
    ) -> Any:
        from ..installer.copilot_hooks_merger import diff_copilot_hooks

        return diff_copilot_hooks(existing, additions)

    # -- Transcript ----------------------------------------------------------

    def transcript_glob(self) -> str:
        """VSCode Copilot Chat 的 chat session 落在 state.vscdb（SQLite 二进制），
        不是文件级 jsonl。这里返回一个不存在的 glob，让 cot_extractor 知道
        ``vscode`` agent 不走 transcript 通道（数据来自 OTel + events.jsonl）。
        """
        return str(Path.home() / ".copilot" / "transcripts" / "*.jsonl")


__all__ = ["VSCodeAdapter"]
