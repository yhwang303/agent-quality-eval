"""Best-effort native critic definitions for supported IDEs.

Trace/live hooks are the authoritative observation path. Native IDE
definitions are installed as a product affordance so users can inspect or
manually invoke the same read-only critic persona where the IDE supports it.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

MARKER = "<!-- agent-cot:agent-quality-critic:v1 -->"
DEFINITION_NAME = "agent-quality-critic"

DEFINITION_MD = f"""{MARKER}
---
name: agent-quality-critic
description: Read-only sidecar critic for evaluating an agent turn after it completes.
tools: []
---

You are Agent Quality Critic, a read-only evaluator for agent interactions.

Your job is to review the just-completed user-agent turn using only the
provided transcript, tool results, hook events, and runtime metrics. Do not
modify files, run shell commands, call external services, or ask the primary
agent to do more work. If evidence is insufficient, state the uncertainty and
the risk clearly instead of inventing missing facts.

Return a JSON object with these keys:

- `summary_conclusion`: one natural-language paragraph beginning with
  `结论：`. Cover the overall judgment, the user request, key agent actions,
  delivered value, and main risks.
- `overall_verdict`: `resolved`, `partial`, or `unresolved`.
- `task_completion`, `tool_use`, `reasoning`, `instruction_following`,
  `faithfulness`, `efficiency`, `reliability`: each is an object with
  `verdict` and `review`.
- `review_markdown`: a concise human-readable review whose first paragraph is
  `**结论**：...`.

Evaluation policy:

- Judge only the current turn.
- A local failure does not imply an overall failure when the agent recovered or
  provided a useful alternative.
- A claim in the final answer is strong only when supported by transcript or
  tool-result evidence.
- Prefer precise, product-facing language over generic praise or blame.
"""


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _targets(agent_name: str) -> list[Path]:
    home = Path.home()
    if agent_name == "codex":
        codex_home = Path(os.environ.get("CODEX_HOME") or home / ".codex").expanduser()
        return [codex_home / "agents" / f"{DEFINITION_NAME}.md"]
    if agent_name == "claude":
        homes = [home / ".claude", home / ".claude-internal", home / ".claude-inertnal"]
        return [p / "agents" / f"{DEFINITION_NAME}.md" for p in homes if p.is_dir()]
    if agent_name == "cursor":
        return [home / ".cursor" / "rules" / f"{DEFINITION_NAME}.mdc"]
    if agent_name == "codebuddy":
        return [home / ".codebuddy" / "agents" / f"{DEFINITION_NAME}.md"]
    return []


def _write_status(agent_name: str, rows: list[dict[str, str]]) -> None:
    try:
        root = Path.home() / ".agent-cot" / "critic"
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{agent_name}-definition-status.json").write_text(
            json.dumps({"agent": agent_name, "items": rows}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def register_critic_definition(agent_name: str) -> list[dict[str, str]]:
    """Install or refresh the named critic definition without clobbering users.

    If a same-name definition already exists and is not ours, we keep it in
    place, write our desired content to a sibling ``.agent-cot-new`` file, and
    create a timestamped backup. Trace/live hooks remain registered, and final
    eval can still be run from the frontend Agent Critic button.
    """
    rows: list[dict[str, str]] = []
    for target in _targets(agent_name):
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                current = target.read_text(encoding="utf-8", errors="replace")
                if MARKER not in current and current.strip() != DEFINITION_MD.strip():
                    backup = target.with_suffix(target.suffix + f".bak.{_timestamp()}")
                    shutil.copy2(target, backup)
                    pending = target.with_suffix(target.suffix + ".agent-cot-new")
                    pending.write_text(DEFINITION_MD + "\n", encoding="utf-8")
                    rows.append(
                        {
                            "path": str(target),
                            "status": "conflict_preserved",
                            "backup": str(backup),
                            "pending": str(pending),
                        }
                    )
                    continue
            target.write_text(DEFINITION_MD + "\n", encoding="utf-8")
            rows.append({"path": str(target), "status": "written"})
        except Exception as exc:
            rows.append({"path": str(target), "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    _write_status(agent_name, rows)
    return rows


__all__ = ["DEFINITION_NAME", "MARKER", "register_critic_definition"]
