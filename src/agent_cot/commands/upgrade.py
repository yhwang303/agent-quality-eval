"""``agent-cot upgrade`` — refresh bundled hook scripts in place.

Thin wrapper over :mod:`agent_cot.installer.upgrader`. Like
``init`` / ``uninstall`` we keep the agent-resolution dance here
so the installer module remains a pure data layer.

The upgrade command is **only** about hook scripts. ``hooks.json``
already holds correct command paths after ``init``; ``config.toml``
schema migrations are out of scope for v0.13. Rerunning ``upgrade``
when nothing changed is a no-op and exits 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..agents import AgentNotImplementedError, get_adapter
from ..agents.base import AgentAdapter, CursorCotError
from ..installer.config import load_config
from ..installer.upgrader import (
    UpgradePlan,
    UpgradeResult,
    apply_upgrade_plan,
    build_upgrade_plan,
)


class UpgradeError(CursorCotError):
    """Raised for upgrade-specific failures the CLI can render verbatim."""


@dataclass
class UpgradeContext:
    plan: UpgradePlan
    adapter: AgentAdapter

    def render_summary(self) -> str:
        return self.plan.render()


def build_context(*, agent_name: str = "cursor") -> UpgradeContext:
    """Pure planning step — no disk writes."""
    adapter = get_adapter(agent_name)
    try:
        adapter.hook_entries()
    except AgentNotImplementedError:
        raise

    cfg = load_config()
    cot_root: Path | None = (
        Path(cfg.cot_extractor_repo) if cfg.cot_extractor_repo else None
    )

    plan = build_upgrade_plan(
        agent_name=agent_name,
        hooks_assets_dir=adapter.hooks_assets_dir(),
        cot_extractor_root=cot_root,
    )
    return UpgradeContext(plan=plan, adapter=adapter)


def apply(ctx: UpgradeContext) -> UpgradeResult:
    """Run the side-effects described by ``ctx.plan``."""
    return apply_upgrade_plan(ctx.plan)


__all__ = [
    "UpgradeContext",
    "UpgradeError",
    "UpgradeResult",
    "apply",
    "build_context",
]
