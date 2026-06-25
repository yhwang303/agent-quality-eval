"""``agent-cot init`` — one-time setup orchestration.

Six steps, in order:

1. Resolve agent adapter (``cursor`` only in v0.13).
2. Detect a free backend port (or use the user-pinned one).
3. Locate the user's ``cot-extractor`` checkout (so hook scripts can
   call into it). Fail loudly when missing rather than silently
   dropping captures.
4. Render the new ``hooks.json`` *in memory*.
5. (dry-run aside): print the diff and the planned filesystem
   operations, then return without writing.
6. (real run): atomic backup + write of ``hooks.json``, copy hook
   scripts, write ``~/.agent-cot/config.toml``.

Every disk-touching step is gated behind ``apply=True``. The default
shape — ``apply=False`` — means even a buggy call site can't trash a
user's hook config without an explicit "yes".
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .. import _assets, diag
from ..agents import AgentNotImplementedError, get_adapter
from ..agents.base import AgentAdapter, HookEntry
from ..installer.config import (
    CursorCotConfig,
    load_config,
    save_config,
)
from ..installer.hooks_merger import HookDiff
# v0.17.0: 不再 hardcode merge_cursor_hooks/diff_hooks。改为通过 adapter
# 的 merge_hook_entries / diff_hook_entries 分派，让每个 IDE 用自己的 schema。
from ..installer.platform_paths import backup_path, agent_cot_root, ensure_dir
from ..installer.port_picker import pick_port
# v0.19.0: write ~/.agent-cot/runtime.json so on-disk hook scripts have a
# stable secondary fallback that always points at the *currently installed*
# extractor / python / data root — even if the user later forgets to re-run
# ``agent-cot upgrade --apply`` after ``pip install -U agent-cot``.
from ..installer.runtime_state import write_runtime_state

# ---------------------------------------------------------------------------
# Plan / Result types — what init computes, what init reports
# ---------------------------------------------------------------------------


@dataclass
class InitPlan:
    """Everything we *would* do — captured before any disk write."""

    agent_name: str
    backend_port: int
    cot_extractor_root: Path | None
    hooks_config_path: Path
    hooks_assets_dir: Path
    bridge_filenames: list[str]
    """Just the basenames; absolute source/target paths are derived in apply."""
    new_hooks_blob: dict
    diff: HookDiff
    config: CursorCotConfig
    # sys.executable — embedded into cot-bridge.js when PATH has no ``python`` (common under Cursor).
    hook_python_executable: str
    warnings: list[str] = field(default_factory=list)
    # v0.20.4: Claude OTel env auto-injection — only populated when
    # ``agent_name == 'claude'`` AND ``write_otel_env=True`` was passed to
    # build_plan. ``env_additions`` = recommended env keys we plan to add
    # (user didn't have them); ``env_preserved`` = recommended keys the
    # user already set, kept as-is (we never overwrite). Both empty when
    # the OTel env path is off or doesn't apply (Cursor / CodeBuddy /
    # VSCode adapters don't expose recommended_env()).
    env_additions: list[tuple[str, str]] = field(default_factory=list)
    env_preserved: list[tuple[str, str]] = field(default_factory=list)
    otel_env_enabled: bool = False
    """``True`` iff this plan will merge an ``env`` block (Claude only,
    requested via ``write_otel_env=True``). Tells :func:`apply_plan` to
    actually call ``merge_claude_env`` on the new hooks blob."""
    additional_targets: list[tuple[Path, Path]] = field(default_factory=list)
    """v0.20.6: Extra ``(settings_path, assets_dir)`` pairs that should
    receive a mirror write of the primary ``new_hooks_blob`` + bridge
    scripts. Only populated when the adapter is Claude AND the user has
    both ``~/.claude/`` and ``~/.claude-internal/`` installed (i.e. they
    run the OSS CLI *and* the Tencent Cursor-embedded variant). Empty
    list otherwise — single-variant users keep the pre-0.20.6 behavior."""

    def render_summary(self) -> str:
        lines: list[str] = []
        lines.append(f"agent           : {self.agent_name}")
        lines.append(f"backend port    : {self.backend_port}")
        lines.append(
            f"cot-extractor   : {self.cot_extractor_root or '(not found — hooks will be skipped at runtime)'}"
        )
        lines.append(f"hook python     : {self.hook_python_executable}")
        lines.append(f"hooks.json      : {self.hooks_config_path}")
        lines.append(f"hooks dir       : {self.hooks_assets_dir}")
        if self.additional_targets:
            # v0.20.6: surface mirror targets so the user can see Claude
            # internal/OSS dual-write before --apply runs.
            lines.append("mirror writes   :")
            for settings, assets in self.additional_targets:
                lines.append(f"  → {settings}")
                lines.append(f"    + hooks dir: {assets}")
        lines.append("")
        lines.append("hooks.json changes:")
        lines.append(self.diff.render() or "  (none)")
        if self.bridge_filenames:
            lines.append("")
            lines.append("hook scripts to install:")
            for name in self.bridge_filenames:
                lines.append(f"  → {self.hooks_assets_dir / name}")
                for _, assets_dir in self.additional_targets:
                    lines.append(f"  → {assets_dir / name}  (mirror)")
        if self.otel_env_enabled:
            lines.append("")
            lines.append("Claude OTel env (fill-missing only — your values never overwritten):")
            if not self.env_additions and not self.env_preserved:
                lines.append("  (nothing to do — env block already has every recommended key)")
            for k, v in self.env_additions:
                lines.append(f"  + env.{k} = {v}")
            for k, v in self.env_preserved:
                lines.append(f"  · env.{k} = {v}  (kept — already set by you)")
        if self.warnings:
            lines.append("")
            lines.append("warnings:")
            for w in self.warnings:
                lines.append(f"  ! {w}")
        return "\n".join(lines)


@dataclass
class InitResult:
    """Side-effects actually performed (only populated when ``apply=True``)."""

    hooks_backup: Path | None = None
    hooks_written: Path | None = None
    scripts_installed: list[Path] = field(default_factory=list)
    config_written: Path | None = None


# ---------------------------------------------------------------------------
# cot-extractor location detection
# ---------------------------------------------------------------------------


def _find_cot_extractor_root(start: Path | None = None) -> Path | None:
    """Locate a usable ``cot-extractor`` root for hook script wiring.

    Resolution order (v0.20.6, *reversed* from earlier releases):

    1. **Bundled wheel asset** — ``agent_cot/assets/cot-extractor/``
       (v0.18.2+). This is the path the *currently installed* agent-cot
       ships and is guaranteed to be in sync with this package's
       extract_cot.py. End users on a fresh ``pip install agent-cot``
       hit this branch.
    2. **Source-tree sibling** — walk upward from ``start`` looking
       for a sibling ``cot-extractor/scripts/extract_cot.py``. Only
       contributors actively iterating on the extractor in a git
       checkout hit this, AND only when the bundled asset is somehow
       missing (e.g. ``pip install -e .`` before ``python -m
       agent_cot._build_assets sync`` has ever run).

    Why the reordering: pre-0.20.6 the sibling search ran FIRST, which
    means a maintainer with a stale ``cot-extractor/`` git checkout
    next to ``agent-cot/`` would have that stale path baked into
    ``~/.cursor/hooks/cot-bridge.js`` as the patched literal. Every
    subsequent Cursor session would spawn the stale extractor (which
    writes to a different output dir), so the dashboard backend never
    saw the new sessions. The bundled copy is always in sync with the
    installed package, so making it the trusted source heals this
    silent-failure mode without affecting fresh-install users.

    A directory of the right name **without** ``scripts/extract_cot.py``
    is treated as not-found — the bridge script needs that exact path.
    """
    # v0.20.6: bundled first — this is what the installed wheel ships.
    from .. import _assets

    if _assets.has_bundled_extractor():
        return _assets.bundled_extractor_root().resolve()

    # Fallback: sibling git checkout (for contributors hacking on the
    # extractor source with no synced bundle).
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        repo = candidate / "cot-extractor"
        if (repo / "scripts" / "extract_cot.py").is_file():
            return repo
    return None


# ---------------------------------------------------------------------------
# hook-script content patching
# ---------------------------------------------------------------------------


_COT_ROOT_LINE = re.compile(
    r"""(const\s+(?:RAW_)?COT_ROOT\s*=\s*process\.env\.COT_EXTRACTOR_ROOT\s*\|\|\s*)['"][^'"]*['"]"""
)

_PYTHON_LINE = re.compile(
    r"""(const\s+(?:RAW_)?PYTHON\s*=\s*process\.env\.COT_PYTHON\s*\|\|\s*)['"][^'"]*['"]"""
)


def _patch_cot_root_default(source: str, new_default: str) -> str:
    """Rewrite the hard-coded ``COT_ROOT`` fallback in our hook scripts.

    We never strip the ``process.env`` clause — environment overrides
    must keep working for power users. We only swap the literal default.
    """
    posix = new_default.replace("\\", "/")
    return _COT_ROOT_LINE.sub(rf"\1'{posix}'", source, count=1)


def _patch_python_default(source: str, exe: str) -> str:
    """Embed ``sys.executable`` so hooks work when Cursor's PATH has no ``python``."""
    posix = str(Path(exe).resolve()).replace("\\", "/")
    lit = json.dumps(posix)
    return _PYTHON_LINE.sub(rf"\1{lit}", source, count=1)


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def build_plan(
    *,
    agent_name: str = "cursor",
    port_backend: int | None = None,
    cot_extractor_root: Path | None = None,
    write_otel_env: bool = True,
) -> InitPlan:
    """Compute (without writing) what ``init`` would do.

    ``write_otel_env`` (default True): when the target adapter exposes a
    ``recommended_env(backend_port=...)`` method (currently only
    :class:`ClaudeAdapter`), the plan additionally merges that env
    dict into ``settings.json`` with **fill-missing-only** semantics —
    every key the user already set keeps its value. Pass False (via
    CLI ``--no-otel-env``) to skip the env merge entirely; useful when
    a user wants only the hooks and manages their OTel config out-of-band.
    """
    adapter: AgentAdapter = get_adapter(agent_name)
    try:
        additions: list[HookEntry] = adapter.hook_entries()
    except AgentNotImplementedError:
        # Bubble up; the CLI knows how to render this nicely.
        raise

    hooks_config = adapter.hooks_config_path()
    hooks_dir = adapter.hooks_assets_dir()

    # Existing config (if any). We do NOT mutate the file — only read.
    existing: dict | None = None
    if hooks_config.is_file():
        existing = json.loads(hooks_config.read_text(encoding="utf-8") or "{}")

    new_blob = adapter.merge_hook_entries(existing or {}, additions)
    diff = adapter.diff_hook_entries(existing, additions)

    port = pick_port(prefer=port_backend or 8765)

    cot_root = cot_extractor_root or _find_cot_extractor_root()

    # Loading config is pure read; apply_plan is the only thing
    # allowed to mkdir.
    cfg = load_config()
    cfg.backend_port = port
    cfg.data_root = str(agent_cot_root())
    if cot_root is not None:
        cfg.cot_extractor_repo = str(cot_root)
    if agent_name not in cfg.installed_agents:
        cfg.installed_agents = sorted({*cfg.installed_agents, agent_name})

    warnings: list[str] = []
    if cot_root is None:
        # v0.18.2: 这条 warning 现在很难触发 —— ``_find_cot_extractor_root``
        # 的第二档兜底（``assets/cot-extractor/``）在 wheel 安装态总会命中。
        # 只有当 wheel 损坏 / 未跑过 ``_build_assets sync`` 时才会到这里，
        # 此时硬编码到 hook 文件里的 ``D:/ai-ide-langfuse/cot-extractor``
        # 默认值在用户机上永远不存在，hook 必然 silently no-op。
        warnings.append(
            "cot-extractor checkout not found AND wheel asset bundle is "
            "missing scripts/extract_cot.py — your install is broken. "
            "Reinstall agent-cot (>=0.18.2) or set "
            "COT_EXTRACTOR_ROOT to a valid checkout path."
        )

    # v0.20.4: Claude OTel env one-shot enablement. ``recommended_env``
    # is duck-typed — only the Claude adapter implements it today; the
    # other adapters (cursor / codebuddy / vscode) skip this branch
    # silently. Pass ``backend_port=port`` so the endpoint URL points
    # at the **actual** port init just locked in (NOT the historical
    # hardcoded 8765 that silently dropped events on machines with the
    # default port already taken).
    env_additions: list[tuple[str, str]] = []
    env_preserved: list[tuple[str, str]] = []
    otel_env_enabled = False
    recommended_env_fn = getattr(adapter, "recommended_env", None)
    if write_otel_env and callable(recommended_env_fn):
        try:
            recommended = recommended_env_fn(backend_port=port)
        except TypeError:
            # Defensive: future adapters might publish recommended_env
            # without the keyword we expect. Fall through quietly.
            recommended = None
        if recommended:
            from ..installer.claude_hooks_merger import (
                diff_claude_env,
                merge_claude_env,
            )

            otel_env_enabled = True
            env_diff = diff_claude_env(existing, recommended)
            env_additions = list(env_diff.added)
            env_preserved = list(env_diff.preserved)
            # Apply the env merge into our in-memory hook blob so apply_plan
            # writes both the hooks AND the env in a single atomic save.
            new_blob, _added, _preserved = merge_claude_env(new_blob, recommended)

    # v0.20.6: collect mirror targets from the adapter (Claude uses this to
    # write to BOTH ~/.claude/settings.json and ~/.claude-internal/settings.json
    # when the user has both variants). Adapters that don't implement
    # ``additional_hooks_targets`` (cursor / codebuddy / vscode) inherit the
    # base.py default that returns an empty list.
    additional_targets: list[tuple[Path, Path]] = []
    try:
        raw_targets = adapter.additional_hooks_targets()
    except AttributeError:
        raw_targets = []
    if raw_targets:
        seen = {(hooks_config.resolve(), hooks_dir.resolve())}
        for settings_path, assets_dir in raw_targets:
            try:
                key = (Path(settings_path).resolve(), Path(assets_dir).resolve())
            except OSError:
                # Path can't be resolved (e.g. parent dir missing) — still
                # accept it, apply_plan will mkdir as needed.
                key = (Path(settings_path), Path(assets_dir))
            if key in seen:
                continue
            seen.add(key)
            additional_targets.append((Path(settings_path), Path(assets_dir)))

    return InitPlan(
        agent_name=agent_name,
        backend_port=port,
        cot_extractor_root=cot_root,
        hooks_config_path=hooks_config,
        hooks_assets_dir=hooks_dir,
        bridge_filenames=list(adapter.bridge_files()),
        new_hooks_blob=new_blob,
        diff=diff,
        config=cfg,
        hook_python_executable=sys.executable,
        warnings=warnings,
        env_additions=env_additions,
        env_preserved=env_preserved,
        otel_env_enabled=otel_env_enabled,
        additional_targets=additional_targets,
    )


# ---------------------------------------------------------------------------
# Apply (real disk writes) — only called from the CLI when --dry-run is off
# ---------------------------------------------------------------------------


def _install_bridge_scripts_from_bundle(plan: InitPlan) -> list[Path]:
    """Copy bundled hook scripts + patch JS literals (where applicable).

    Shared by :func:`apply_plan` and :func:`refresh_bundled_hook_scripts`.

    Per-script handling:

    * ``cot-bridge.js`` (Cursor): patch ``COT_ROOT`` literal +
      ``PYTHON`` literal (so Cursor's thin spawn env can still find python).
    * ``cot-stream.js`` (Cursor): patch ``COT_ROOT`` literal only.
    * ``cot-stream-codebuddy.js`` (CodeBuddy): no JS literal patches —
      reads everything from env / runtime.json.
    * ``claude_stream_hook.py`` (Claude Internal, v0.19.1+): a Python
      script — the JS regexes don't match anyway, but we explicitly
      skip patching to make the intent obvious. The Python hook resolves
      cot-extractor / python via its own 4-layer fallback chain
      (env > .agent-cot/runtime.json > .cursor-cot/runtime.json > probe),
      so writing a literal here would be redundant.

    v0.20.6: When ``plan.additional_targets`` is non-empty, each script is
    also mirrored to every target's ``assets_dir`` so dual-Claude users
    (both ``~/.claude/`` and ``~/.claude-internal/`` installed) get
    functional hooks regardless of which variant their next session runs.
    """
    installed: list[Path] = []
    src_root = _assets.hooks_dir() / plan.agent_name

    # Build the list of (assets_dir, label) destinations: primary + mirrors.
    # Label is purely for the install log so a user grepping pipeline.log
    # can see which copy is the "primary" vs a mirror.
    destinations: list[tuple[Path, str]] = [(plan.hooks_assets_dir, "primary")]
    for _, mirror_dir in plan.additional_targets:
        destinations.append((mirror_dir, "mirror"))

    for assets_dir, _label in destinations:
        ensure_dir(assets_dir)
        for name in plan.bridge_filenames:
            src = src_root / name
            target = assets_dir / name
            if not src.is_file():
                raise FileNotFoundError(
                    f"bundled hook asset missing: {src} "
                    "(this is a packaging bug; please re-install agent-cot)"
                )
            # v0.20.5 stub-guard: refuse to install one-line placeholders.
            #
            # 0.20.4 shipped a wheel whose ``cursor/cot-stream.js`` and
            # ``cursor/cot-bridge.js`` were the 59-byte test-fixture stub
            # (``const COT_ROOT = process.env.COT_EXTRACTOR_ROOT || 'OLD';``).
            # Cursor still fired hooks, node still ran the file, but the
            # file did nothing — events.jsonl was never written, extractor
            # was never spawned, dashboard reported zero Cursor sessions.
            # Detecting this at install-time (and refusing to wire it up)
            # turns a silent invisible failure into a loud one a user
            # can actually report. Threshold: every real shipping hook
            # is >2 KB; we set the cutoff at 1 KB to leave headroom and
            # never trigger on a future minimal-but-real hook.
            size = src.stat().st_size
            if size < 1024 and name.endswith((".js", ".py")):
                raise RuntimeError(
                    f"bundled hook asset looks like a stub: {src} "
                    f"({size} bytes). This is a packaging bug — reinstall "
                    "agent-cot from a newer wheel."
                )
            content = src.read_text(encoding="utf-8")
            # Patch JS literals only — Python hooks self-resolve at runtime.
            is_js = name.endswith(".js")
            if is_js and plan.cot_extractor_root is not None:
                content = _patch_cot_root_default(
                    content, str(plan.cot_extractor_root)
                )
            if name == "cot-bridge.js":
                content = _patch_python_default(content, plan.hook_python_executable)
            target.write_text(content, encoding="utf-8")
            installed.append(target)
    return installed


def refresh_bundled_hook_scripts(plan: InitPlan) -> InitResult:
    """Re-copy hook JS from the package and patch ``COT_ROOT`` — **without** touching ``hooks.json``.

    ``agent-cot init --apply`` used to return early when ``hooks.json`` was
    already merged (``plan.diff.has_changes`` is false). In that case
    :func:`apply_plan` never ran, so ``~/.cursor/hooks/cot-bridge.js`` stayed
    at whatever bytes were written by an **older** ``init`` —— e.g. the
    literal ``D:/ai-ide-langfuse/cot-extractor`` baked into the template before
    :func:`_patch_cot_root_default` ran with ``cot_extractor_root=None``.
    Users who ``pip install`` 0.18.2 then ran ``init`` saw ``cot-extractor``
    detected in ``status`` but **dashboard still empty** because the on-disk
    hook never pointed at the wheel bundle.

    Call this path whenever ``hooks.json`` is unchanged so the JS on disk
    always matches the currently installed package + resolved extractor root.
    """
    result = InitResult()
    result.scripts_installed = _install_bridge_scripts_from_bundle(plan)
    try:
        from ..installer.critic_registration import register_critic_definition

        register_critic_definition(plan.agent_name)
    except Exception:
        pass
    result.config_written = save_config(plan.config)
    # v0.19.0: refresh runtime.json so codebuddy hook + cursor hooks both
    # have a fresh secondary fallback for AGENT_COT_DATA_ROOT / extractor.
    #
    # v0.19.3: runtime.json 里 ``data_root`` 的语义跟 cfg 不一样 —— cfg.data_root
    # 是 "agent-cot 根目录"（默认 ~/.agent-cot）；runtime.json.data_root 是
    # backend/hook/extractor 都读的 "数据子目录"（必须等于 ~/.agent-cot/data）。
    # 早期版本直接把 cfg 值传过来，导致 hook 解析到 ~/.agent-cot 写 cot.json，
    # 但 backend 默认扫 ~/.agent-cot/data，两端永远对不上 → SessionList 缺
    # codebuddy / claude 会话。这里显式拼上 /data，让两边语义对齐。
    write_runtime_state(
        cot_extractor_root=plan.cot_extractor_root,
        python_executable=plan.hook_python_executable,
        data_root=Path(plan.config.data_root) / "data",
    )
    return result


def apply_plan(plan: InitPlan, *, force: bool = False) -> InitResult:
    """Perform the side effects described by ``plan``.

    Order of writes is chosen so that *any* failure leaves the user's
    machine in a recoverable state:

    1. Back up ``hooks.json`` (no-op if absent).
    2. Install hook scripts (idempotent, additive).
    3. Atomically write the new ``hooks.json``.
    4. Persist ``~/.agent-cot/config.toml``.

    If step 3 fails we still have the backup *and* the (newer) hook
    scripts in place; uninstall can recover.
    """
    diag.log(
        "cli.init",
        ide=plan.agent_name,
        event="apply_plan",
        cot_extractor_root=str(plan.cot_extractor_root or "-"),
        hook_python=plan.hook_python_executable,
        backend_port=plan.backend_port,
    )
    result = InitResult()

    # --- 1. backup --------------------------------------------------------
    if plan.hooks_config_path.is_file():
        backup = backup_path(plan.hooks_config_path)
        shutil.copy2(plan.hooks_config_path, backup)
        result.hooks_backup = backup

    # --- 2. install hook scripts -----------------------------------------
    result.scripts_installed = _install_bridge_scripts_from_bundle(plan)

    # --- 2.b best-effort critic definition -------------------------------
    #
    # The hook sidecar is the guaranteed automatic path. Native definitions
    # are advisory and IDE-specific, so registration failures must never block
    # hook installation or dashboard startup.
    try:
        from ..installer.critic_registration import register_critic_definition

        register_critic_definition(plan.agent_name)
    except Exception:
        pass

    # --- 3. atomic hooks.json write --------------------------------------
    #
    # v0.20.6: When ``plan.additional_targets`` is non-empty we also write a
    # *mirror copy* of the same blob to every listed settings.json. We DO
    # backup-then-merge each mirror separately so any hooks the user added
    # to their Claude-internal settings.json (or vice versa) via codebuddy-mem
    # or hand-edited entries are preserved — we never overwrite blindly,
    # we go through the same adapter.merge_hook_entries() pipeline.
    payload = json.dumps(plan.new_hooks_blob, indent=2, ensure_ascii=False)

    def _atomic_write(target_path: Path, blob: str) -> Path:
        """Backup + atomic write of one settings.json. Returns the path written."""
        ensure_dir(target_path.parent)
        if target_path.is_file():
            shutil.copy2(target_path, backup_path(target_path))
        tmp = target_path.with_suffix(target_path.suffix + ".tmp")
        tmp.write_text(blob + "\n", encoding="utf-8")
        tmp.replace(target_path)
        return target_path

    result.hooks_written = _atomic_write(plan.hooks_config_path, payload)

    # 3.b — mirror writes. For each (settings_path, _assets_dir) target,
    # re-merge our additions into THAT file's existing blob so we don't
    # clobber the user's pre-existing entries (e.g. codebuddy-mem hooks
    # on the .claude-internal settings.json). We re-run merge per-target
    # because each settings.json starts from a *different* existing state.
    if plan.additional_targets:
        from ..agents import get_adapter
        adapter = get_adapter(plan.agent_name)
        for settings_path, _assets_dir in plan.additional_targets:
            mirror_entries_fn = getattr(adapter, "hook_entries_for_assets_dir", None)
            if callable(mirror_entries_fn):
                additions = list(mirror_entries_fn(Path(_assets_dir)))
            else:
                additions = adapter.hook_entries()
            existing_blob: dict | None = None
            if settings_path.is_file():
                try:
                    existing_blob = json.loads(
                        settings_path.read_text(encoding="utf-8") or "{}"
                    )
                except Exception:
                    # Corrupted target settings.json — fall through to fresh
                    # write. We've already backed it up above.
                    existing_blob = None
            mirror_blob = adapter.merge_hook_entries(existing_blob or {}, additions)
            # Mirror the env block too if the primary plan injected env.
            if plan.otel_env_enabled and getattr(adapter, "recommended_env", None):
                try:
                    from ..installer.claude_hooks_merger import merge_claude_env
                    recommended = adapter.recommended_env(
                        backend_port=plan.backend_port
                    )
                    mirror_blob, _a, _p = merge_claude_env(mirror_blob, recommended)
                except Exception:
                    # Env merge failure on the mirror must not abort the
                    # whole apply — primary write already succeeded.
                    pass
            mirror_payload = json.dumps(mirror_blob, indent=2, ensure_ascii=False)
            _atomic_write(settings_path, mirror_payload)

    # --- 4. config.toml ---------------------------------------------------
    result.config_written = save_config(plan.config)

    # --- 5. runtime.json (v0.19.0 self-heal) ------------------------------
    # Snapshot the *currently installed* paths so that the on-disk hook
    # scripts (cursor's cot-bridge.js / cot-stream.js + codebuddy's
    # cot-stream-codebuddy.js) have a stable JSON fallback to consult when
    # their baked-in literal points at a no-longer-existing path. This is
    # what makes ``pip install -U agent-cot`` work without forcing the
    # user to re-run ``init --apply`` afterward.
    #
    # v0.19.3: ``data_root`` 显式拼 ``/data`` — 见 refresh_disk_after_idempotent_init
    # 同名注释。修复 codebuddy / claude 写 cot.json 跟 backend 扫盘目录对不上的
    # 历史 bug。
    write_runtime_state(
        cot_extractor_root=plan.cot_extractor_root,
        python_executable=plan.hook_python_executable,
        data_root=Path(plan.config.data_root) / "data",
    )

    return result


__all__ = [
    "InitPlan",
    "InitResult",
    "apply_plan",
    "build_plan",
    "refresh_bundled_hook_scripts",
]
