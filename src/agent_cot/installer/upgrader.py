"""In-place refresh of the bundled hook scripts.

Scope is intentionally narrow:

* **Replaces**  ``~/.cursor/hooks/cot-bridge.js`` and
  ``~/.cursor/hooks/cot-stream.js`` (the OWNED_HOOK_SCRIPTS) with
  the latest copies shipped inside our wheel.
* **Patches**   the ``COT_ROOT`` default literal so the upgraded
  script keeps pointing at the user's existing cot-extractor
  checkout (read from ``config.toml``).
* **Does NOT** touch ``~/.cursor/hooks.json`` /
  ``~/.codebuddy/settings.json`` / ``~/.claude/settings.json``
  (``init`` / ``uninstall`` own those), nor
  ``~/.agent-cot/config.toml`` (schema migrations are a separate,
  future concern).
* **v0.20.4 explicit contract**: also does NOT touch the ``env``
  block inside ``~/.claude/settings.json``. Writing the Claude OTel
  env is an ``init --apply`` responsibility (one-time injection
  with fill-missing semantics). Upgrades stay env-blind so a user
  who has tuned their OTel keys post-install (changed endpoint to
  a corp collector, flipped ``OTEL_LOG_USER_PROMPTS`` off for
  privacy, etc.) keeps their tuning after every wheel upgrade.
  Endpoint port-eviction self-heal lives in ``commands/start.py``
  instead, which is where the live backend port is known.

The contract: an upgrade is safe to run any time, idempotent, and
carries a per-file backup that ``uninstall --restore-backup``
ignores (we use a different naming convention for upgrade
backups so they don't pollute the hooks.json restore list).
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .. import _assets
from ..agents import get_adapter
from ..commands.init import _patch_cot_root_default  # internal but stable enough
from .hooks_merger import OWNED_HOOK_SCRIPTS
# v0.19.0: refresh ~/.agent-cot/runtime.json after each upgrade --apply so the
# on-disk hook scripts get a fresh secondary fallback even when the user
# never re-runs ``init --apply``.
from .runtime_state import read_runtime_state, write_runtime_state

# ---------------------------------------------------------------------------
# Plan / Result types
# ---------------------------------------------------------------------------


@dataclass
class ScriptDelta:
    """Per-file diff between the on-disk hook script and the bundled one."""

    name: str
    target: Path
    bundled_source: Path
    target_exists: bool
    needs_update: bool
    """``False`` when the on-disk SHA-256 matches the would-be patched bundle."""

    target_sha256: str | None
    new_sha256: str

    @property
    def status(self) -> str:
        if not self.target_exists:
            return "missing"
        return "stale" if self.needs_update else "up-to-date"


@dataclass
class UpgradePlan:
    """Everything :func:`apply_upgrade_plan` would do, computed up-front."""

    agent_name: str
    hooks_assets_dir: Path
    cot_extractor_root: Path | None
    deltas: list[ScriptDelta] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def changed(self) -> list[ScriptDelta]:
        return [d for d in self.deltas if d.target_exists and d.needs_update]

    @property
    def missing(self) -> list[ScriptDelta]:
        return [d for d in self.deltas if not d.target_exists]

    @property
    def is_noop(self) -> bool:
        return not self.changed and not self.missing

    def render(self) -> str:
        lines: list[str] = []
        lines.append(f"agent          : {self.agent_name}")
        lines.append(f"hooks dir      : {self.hooks_assets_dir}")
        lines.append(
            f"cot-extractor  : {self.cot_extractor_root or '(unset — using bundled default)'}"
        )
        lines.append("")
        lines.append("hook scripts:")
        if not self.deltas:
            lines.append("  (no bundled scripts found — packaging bug?)")
        for d in self.deltas:
            glyph = {
                "missing": "?",
                "stale": "↻",
                "up-to-date": "✓",
            }[d.status]
            lines.append(f"  {glyph} {d.name:<14} {d.status:<11} {d.target}")
        if self.warnings:
            lines.append("")
            lines.append("warnings:")
            for w in self.warnings:
                lines.append(f"  ! {w}")
        return "\n".join(lines)


@dataclass
class UpgradeResult:
    scripts_replaced: list[Path] = field(default_factory=list)
    scripts_installed: list[Path] = field(default_factory=list)
    """Files that didn't exist before but now do."""

    backups: list[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(p: Path) -> str | None:
    if not p.is_file():
        return None
    return _sha256_text(p.read_text(encoding="utf-8"))


def _upgrade_backup_path(target: Path) -> Path:
    """Distinct from ``hooks.json.bak.<ts>`` so we don't crowd the
    uninstall-restore list. Format: ``<name>.upgrade-bak.<ts>``."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return target.with_name(f"{target.name}.upgrade-bak.{stamp}")


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def build_upgrade_plan(
    *,
    agent_name: str,
    hooks_assets_dir: Path,
    cot_extractor_root: Path | None,
) -> UpgradePlan:
    """Compute, without writing, what an upgrade would do.

    Picks up the bundle from ``agent_cot/assets/hooks/<agent>/`` and
    per-file compares the would-be content against what's on disk.
    """
    plan = UpgradePlan(
        agent_name=agent_name,
        hooks_assets_dir=hooks_assets_dir,
        cot_extractor_root=cot_extractor_root,
    )

    src_root = _assets.hooks_dir() / agent_name
    if not src_root.is_dir():
        plan.warnings.append(
            f"bundled hooks for agent='{agent_name}' not found at {src_root}; "
            "this is a packaging bug — re-install agent-cot."
        )
        return plan

    # v0.19.1: iterate the SPECIFIC adapter's bridge_files() rather than the
    # global OWNED_HOOK_SCRIPTS — otherwise upgrading agent='cursor' would
    # warn about missing cot-stream-codebuddy.js (codebuddy's file in
    # cursor's src dir), claude_stream_hook.py, etc. Keeps warnings honest
    # and prevents the per-agent dir contract from leaking across IDEs.
    try:
        adapter = get_adapter(agent_name)
        bridge_names = list(adapter.bridge_files())
    except Exception:
        # Fallback: use OWNED list (preserves pre-v0.19.1 behaviour)
        bridge_names = sorted(OWNED_HOOK_SCRIPTS)

    for name in bridge_names:
        src = src_root / name
        if not src.is_file():
            plan.warnings.append(
                f"bundled hook script missing: {src} "
                "(skipping; remaining files will still upgrade)"
            )
            continue

        target = hooks_assets_dir / name
        bundled_text = src.read_text(encoding="utf-8")
        # Only patch JS literals — Python hooks self-resolve at runtime
        # via 4-layer fallback (env > runtime.json > probe).
        if name.endswith(".js") and cot_extractor_root is not None:
            bundled_text = _patch_cot_root_default(
                bundled_text, str(cot_extractor_root)
            )
        new_hash = _sha256_text(bundled_text)
        old_hash = _sha256_file(target)

        plan.deltas.append(
            ScriptDelta(
                name=name,
                target=target,
                bundled_source=src,
                target_exists=target.is_file(),
                needs_update=(old_hash != new_hash),
                target_sha256=old_hash,
                new_sha256=new_hash,
            )
        )

    if cot_extractor_root is None:
        plan.warnings.append(
            "config.toml has no cot_extractor_repo set; the upgraded "
            "scripts will fall back to their bundled default. Run "
            "`agent-cot init --apply` once to record the path."
        )

    return plan


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def apply_upgrade_plan(plan: UpgradePlan) -> UpgradeResult:
    """Replace each hook script in ``plan.changed`` (and install any
    missing ones), atomically, with a per-file backup.

    Files marked ``up-to-date`` are skipped — that's what makes
    upgrade idempotent.
    """
    result = UpgradeResult()

    # ensure parent dir exists (rare but possible: user nuked .cursor/hooks/)
    plan.hooks_assets_dir.mkdir(parents=True, exist_ok=True)

    for delta in plan.deltas:
        if delta.target_exists and not delta.needs_update:
            continue

        bundled_text = delta.bundled_source.read_text(encoding="utf-8")
        # Same JS-only patching as in build_upgrade_plan above. Python
        # hooks (claude_stream_hook.py) are byte-for-byte copied; their
        # 4-layer runtime fallback handles path resolution at spawn.
        if delta.name.endswith(".js") and plan.cot_extractor_root is not None:
            bundled_text = _patch_cot_root_default(
                bundled_text, str(plan.cot_extractor_root)
            )

        if delta.target_exists:
            backup = _upgrade_backup_path(delta.target)
            shutil.copy2(delta.target, backup)
            result.backups.append(backup)

        tmp = delta.target.with_suffix(delta.target.suffix + ".tmp")
        tmp.write_text(bundled_text, encoding="utf-8")
        tmp.replace(delta.target)

        if delta.target_exists:
            result.scripts_replaced.append(delta.target)
        else:
            result.scripts_installed.append(delta.target)

    # v0.19.0: snapshot the currently-installed paths into runtime.json so
    # the JS hooks (cursor + codebuddy) can self-heal mid-session via the
    # secondary fallback chain (env > literal > runtime.json).
    #
    # v0.19.4 (P-9): also forward the previously-stored ``data_root`` and
    # ``python_executable``, so an upgrade never silently rewrites a
    # user's custom data root back to the package default. Before this,
    # an ``agent-cot upgrade`` run on a setup with a non-standard
    # ``AGENT_COT_DATA_ROOT`` would reset ``runtime.json.data_root`` to
    # ``~/.agent-cot/data``, breaking every hook on the next save.
    try:
        prev = read_runtime_state() or {}
        prev_data_root = prev.get("data_root")
        prev_python = prev.get("python_executable")
        write_runtime_state(
            cot_extractor_root=plan.cot_extractor_root,
            python_executable=prev_python if isinstance(prev_python, str) else None,
            data_root=prev_data_root if isinstance(prev_data_root, str) else None,
        )
    except Exception:
        # Best-effort: never fail an upgrade because we couldn't write a
        # state file — the patched script literals are already on disk.
        pass

    return result


__all__ = [
    "ScriptDelta",
    "UpgradePlan",
    "UpgradeResult",
    "apply_upgrade_plan",
    "build_upgrade_plan",
]
