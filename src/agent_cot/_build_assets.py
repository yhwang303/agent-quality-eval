"""Sync the source-tree backend + frontend build into ``assets/``.

Called by maintainers before ``python -m build`` so the wheel ships
the latest copy. End users never touch this — they install a pre-built
wheel where ``assets/backend/`` and ``assets/frontend-dist/`` are
already populated.

Usage
-----
    # Refresh both
    python -m agent_cot._build_assets sync

    # Show paths only (no copy)
    python -m agent_cot._build_assets info

The script intentionally avoids running ``npm run build`` for you: a
maintainer is the right person to decide when frontend assets are
release-ready, and CI / local environments differ in node version.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def _here() -> Path:
    return Path(__file__).resolve().parent


def _project_repo_root() -> Path | None:
    """Walk up from this file looking for ``agent-dashboard/`` sibling.

    Returns ``None`` when called from an installed wheel (not editable),
    where the source tree isn't reachable — in that mode ``sync`` is a
    no-op. ``info`` still works (paths are package-relative).
    """
    for parent in _here().parents:
        if (parent / "agent-dashboard").is_dir():
            return parent
    return None


def _bundled_backend_target() -> Path:
    return _here() / "assets" / "backend"


def _bundled_frontend_target() -> Path:
    return _here() / "assets" / "frontend-dist"


def _bundled_extractor_target() -> Path:
    """v0.17: cot-extractor/src/*.py vendored into the wheel.

    Why: backend/main.py imports cot_otlp_exporter / cot_otel_enricher /
    cot_extractor at runtime via sys.path. End users who pip install us
    don't have the source repo, so we ship the extractor next to the
    backend and the wrapper resolves it at spawn time.
    """
    return _here() / "assets" / "cot-extractor-src"


def _bundled_extractor_full_target() -> Path:
    """v0.18.2: 完整 ``cot-extractor/`` 仓库布局（含 ``scripts/`` 子目录）。

    历史背景：``assets/cot-extractor-src/`` 只 vendor 了 ``cot-extractor/src/``
    下的库代码（cot_extractor / cot_otlp_exporter / …），完全没带
    ``cot-extractor/scripts/extract_cot.py`` —— 即 Cursor hook 在 stop
    事件触发后跑的那个 CLI 入口。结果：装了 wheel 的同事，cot-bridge.js
    去找 ``<COT_ROOT>/scripts/extract_cot.py`` 永远是 404，整条采集链
    在 wheel 安装态下从 v0.17 起就是断的（dashboard 永远 SESSIONS 0）。

    v0.18.2 新增这条 vendor 路径，把整个 cot-extractor/ 的目录结构
    （scripts/ + src/ + 顶层 .env.example 等）按原样镜像到
    ``assets/cot-extractor/`` 下。``_find_cot_extractor_root`` 找不到
    源 checkout 时回退到这里，cot-bridge.js 就能拼出正确的
    ``<bundled>/scripts/extract_cot.py`` 路径。

    保留 ``cot-extractor-src/`` 不动是为了 backend wrapper 的 sys.path
    注入路径稳定（一改全炸），新功能用新目录。
    """
    return _here() / "assets" / "cot-extractor"


def _sync_backend(repo: Path) -> tuple[int, list[Path]]:
    src = repo / "agent-dashboard" / "backend"
    dst = _bundled_backend_target()
    if not src.is_dir():
        return 0, []

    if dst.is_dir():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for py_file in src.rglob("*"):
        if py_file.is_dir():
            continue
        # Skip caches and dot-files.
        if "__pycache__" in py_file.parts:
            continue
        if any(p.startswith(".") for p in py_file.parts[len(src.parts):]):
            continue
        rel = py_file.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(py_file, out)
        copied.append(out)

    # NB: we deliberately do NOT plant __init__.py inside the bundled
    # backend tree. Doing so would make setuptools treat it as a
    # ``agent_cot.assets.backend`` sub-package and try to import it,
    # which collides with the spawn-time semantics (we run main:app
    # with cwd=backend, importing services.* relative to cwd).
    return len(copied), copied


def _sync_frontend(repo: Path) -> tuple[int, list[Path]]:
    src = repo / "agent-dashboard" / "frontend" / "dist"
    dst = _bundled_frontend_target()
    if not src.is_dir():
        return 0, []

    if dst.is_dir():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for f in src.rglob("*"):
        if f.is_dir():
            continue
        rel = f.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        copied.append(out)
    return len(copied), copied


def _sync_extractor(repo: Path) -> tuple[int, list[Path]]:
    """Copy cot-extractor/src/*.py (top-level only, no __pycache__) into
    assets/cot-extractor-src/. Backend wrapper picks it up via sys.path
    when the user installs us as a wheel and the source repo is not at
    a sibling directory.
    """
    src = repo / "cot-extractor" / "src"
    dst = _bundled_extractor_target()
    if not src.is_dir():
        return 0, []

    if dst.is_dir():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for f in src.iterdir():
        # Top-level .py files only — extractor module is flat by design.
        # Skip __pycache__, hidden dot-files, anything not a .py module.
        if not f.is_file():
            continue
        if f.name.startswith(".") or f.name.startswith("_"):
            # Note: we keep modules whose names start with "cot_" / etc;
            # the underscore filter is just for "_pycache_" style dirs,
            # which we already excluded by `is_file()`.
            if not f.name.endswith(".py"):
                continue
        if not f.name.endswith(".py"):
            continue
        out = dst / f.name
        shutil.copy2(f, out)
        copied.append(out)

    # Same anti-namespace-package trick as backend: do NOT plant an
    # __init__.py here. Modules are reachable via sys.path injection
    # from the backend wrapper, not via `agent_cot.assets.cot_extractor_src.*`
    # imports.
    return len(copied), copied


def _sync_extractor_full(repo: Path) -> tuple[int, list[Path]]:
    """v0.18.2: 完整镜像 ``cot-extractor/`` 到 ``assets/cot-extractor/``.

    与 ``_sync_extractor`` 不同：
    * 保留**目录结构**（``scripts/extract_cot.py`` / ``src/cot_*.py``）
    * 不止 ``src/``，把 ``scripts/`` 一起带上 —— cot-bridge.js 需要
      ``<root>/scripts/extract_cot.py`` 这条路径

    剔除：
    * ``output/``、``__pycache__/``、隐藏文件、``.env``（防止把开发机的
      Langfuse / 智谱密钥打进 wheel）
    * 跟 hook / extractor / watcher 链路无关的辅助脚本（test_* / mcp_* /
      verify_* / analyze_* / mcp_proxy / transcript_watcher.cmd / .sh 等）

    **v0.18.5 关键修正**：白名单恢复 ``transcript_watcher.py`` /
    ``tool_reproducer.py`` / ``backfill_results.py`` / ``cot_hook.py``。
    之前 v0.18.2 的注释把它们标成"hook 不需要的辅助脚本"是错的 ——
    没有 watcher daemon，Cursor 那 10+ 个 gap 工具（Glob / Grep / Delete /
    WebFetch / WebSearch / Task / SemanticSearch / TodoWrite / AskQuestion /
    ReadLints / AwaitShell / EditNotebook / GenerateImage 等）的 tool_use
    与 tool_result 一行都不会被采到，前端就只剩 thinking bullet（同事在
    0.18.4 上反馈的"毛坯房" trace 就是这个根因）。
    """
    src = repo / "cot-extractor"
    dst = _bundled_extractor_full_target()
    if not src.is_dir():
        return 0, []

    # v0.19.1: tolerate the case where dst already exists but the parent
    # rmtree fails because Cursor/VSCode are watching the dir handle
    # (Windows: file watchers hold handles; rmdir fails with EBUSY).
    # In that case, remove file CONTENTS in-place (rmtree per child),
    # then re-populate. The empty top dir staying around is fine —
    # we just refill it.
    if dst.is_dir():
        try:
            shutil.rmtree(dst)
        except (PermissionError, OSError) as exc:
            # Fallback: clear contents, leave dir
            print(f"  note: cannot rmtree {dst} ({exc.__class__.__name__}); "
                  "clearing contents in-place instead.")
            for child in list(dst.iterdir()):
                try:
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink()
                except OSError:
                    pass
    dst.mkdir(parents=True, exist_ok=True)

    # 只保留 hook + extractor 运行时真的会用到的子目录
    keep_subdirs = {"src", "scripts"}
    # v0.18.5: scripts/ 白名单 —— 只放真正会被 hook 触发或被 agent-cot start
    # 拉起的脚本。test_* / mcp_proxy / verify_* / analyze_* / wrapper .cmd .sh
    # 故意排除掉，避免把 wheel 体积撑大。
    keep_scripts = {
        "extract_cot.py",       # cot-bridge.js stop event 触发
        "export_otlp.py",       # agent-cot otlp send 走的入口
        "transcript_watcher.py",  # agent-cot start 拉起的 daemon（gap-tool 补全核心）
        "tool_reproducer.py",   # transcript_watcher.py 同目录依赖（Glob/Grep/Delete 重放）
        "backfill_results.py",  # 历史 session 一次性补 agentToolResult
        "cot_hook.py",          # Claude Code stop hook（同样会调 uplink）
    }

    copied: list[Path] = []
    for f in src.rglob("*"):
        if f.is_dir():
            continue
        rel = f.relative_to(src)
        parts = rel.parts
        # 跳过非目标顶级目录
        if parts[0] not in keep_subdirs:
            continue
        # 跳过缓存/隐藏
        if "__pycache__" in parts:
            continue
        if any(p.startswith(".") for p in parts):
            continue
        # scripts/ 子目录白名单
        if parts[0] == "scripts" and (len(parts) != 2 or parts[1] not in keep_scripts):
            continue
        # 只要 .py
        if not f.name.endswith(".py"):
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        copied.append(out)

    # 同 backend / cot-extractor-src：不放 __init__.py，保持 hook 子进程
    # 用 ``python <wheel>/.../scripts/extract_cot.py`` 这种独立脚本语义。
    return len(copied), copied


def _sync_claude_hooks(repo: Path) -> tuple[int, list[Path]]:
    """v0.17：把 ``claude-code/hooks/*.py`` 拷进 ``assets/hooks/claude/``。

    Why: 用户在自己机器上 ``~/.claude/hooks/`` 安装的脚本来源是仓库根目录
    ``claude-code/hooks/`` —— 之前打 wheel 时没人 sync 这一段，导致
    ``agent-cot install-claude-hooks`` 之类的命令找不到 source。这里把
    ``claude_stream_hook.py`` / ``langfuse_hook.py`` 一并 vendor，让 wheel
    安装的用户能复现一模一样的 Claude 27-hook 接入体验。

    **v0.19.1 重要例外**：``claude_stream_hook.py`` 由 agent-cot **自主维护**
    （含 4 层路径解析 + 后台 spawn extract_cot 触发），而仓库根的
    ``claude-code/hooks/claude_stream_hook.py`` 是上游 cursor-cot-observer
    的旧版本（不带这些能力）。如果允许 sync 覆盖，每次 ``_build_assets sync``
    都会把 0.19.1 的关键改动倒退回旧版（前端再次看不到 Claude session）。
    所以这里**显式跳过**这一个文件，其余 .py（如 langfuse_hook.py）正常 sync。
    """
    src = repo / "claude-code" / "hooks"
    dst = _here() / "assets" / "hooks" / "claude"
    if not src.is_dir():
        return 0, []

    dst.mkdir(parents=True, exist_ok=True)
    # v0.19.1: 不删 claude_stream_hook.py（agent-cot 自主维护的版本）
    _AGENT_OWNED_PY = {"claude_stream_hook.py"}
    for old in dst.glob("*.py"):
        if old.name in _AGENT_OWNED_PY:
            continue
        old.unlink()

    copied: list[Path] = []
    for f in src.iterdir():
        if not f.is_file() or not f.name.endswith(".py"):
            continue
        if f.name in _AGENT_OWNED_PY:
            # 不允许上游旧版覆盖 agent-cot 维护的 hook
            continue
        out = dst / f.name
        shutil.copy2(f, out)
        copied.append(out)
    return len(copied), copied


def _cmd_info() -> int:
    print("agent_cot._build_assets")
    print(f"  package root       : {_here()}")
    print(f"  backend target     : {_bundled_backend_target()}")
    print(f"  frontend target    : {_bundled_frontend_target()}")
    print(f"  extractor target   : {_bundled_extractor_target()}")
    repo = _project_repo_root()
    if repo is None:
        print("  source repo        : (not editable; sync would be a no-op)")
    else:
        print(f"  source repo        : {repo}")
        print(f"    backend  source  : {repo / 'agent-dashboard' / 'backend'}")
        print(f"    frontend source  : {repo / 'agent-dashboard' / 'frontend' / 'dist'}")
        print(f"    extractor source : {repo / 'cot-extractor' / 'src'}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# v0.20.0 — post-sync pipeline.log injection
# ─────────────────────────────────────────────────────────────────────────────
#
# Why this exists:
#   - We do NOT modify ``cot-extractor/`` or ``agent-dashboard/backend/`` in
#     the source tree (user rule: "do not modify cursor-cot-observer source").
#   - But we DO want the wheel's vendored copies of those files to carry
#     unified pipeline.log breadcrumbs (so colleagues can ``tail -f`` one
#     file and see the full trace lifecycle when something goes wrong).
#
# Solution: every time ``_cmd_sync`` runs, after copying assets, we walk
# the vendored extractor / backend copies and inject the pipeline-log
# helpers in well-defined locations. The injection is idempotent (the
# helper checks for a marker before inserting).
#
# Markers used:
#   ``# AGENT_COT_PIPELINE_LOG_INJECT_v1``
#
# Touchpoints:
#   * ``assets/cot-extractor/scripts/extract_cot.py``  — _pipeline_log()
#     wired into start / fatal / cot_written breadcrumbs.
#   * ``assets/backend/config.py``                       — boot breadcrumb
#     + COT_SCAN_DIRS (legacy root multi-scan; the 0.19.3 release notes
#     claimed it but the helper was never coded).
#   * ``assets/backend/services/session_scanner.py``     — iterate over
#     COT_SCAN_DIRS in scan_sessions() / get_session_cot().

_INJECT_MARKER = "AGENT_COT_PIPELINE_LOG_INJECT_v1"


def _inject_extract_cot_pipeline_log(path: Path) -> bool:
    """Patch ``extract_cot.py`` (vendored) with pipeline.log breadcrumbs."""
    if not path.is_file():
        return False
    src = path.read_text(encoding="utf-8", errors="replace")
    if _INJECT_MARKER in src:
        return False  # already patched

    helper_block = (
        "\n# " + _INJECT_MARKER + "\n"
        "import os as _agent_cot_os\n"
        "from pathlib import Path as _AgentCotPath\n"
        "_PIPELINE_LOG = _AgentCotPath(\n"
        "    _agent_cot_os.environ.get(\"AGENT_COT_PIPELINE_LOG\")\n"
        "    or str(_AgentCotPath.home() / \".agent-cot\" / \"logs\" / \"pipeline.log\")\n"
        ").expanduser()\n"
        "\n"
        "def _pipeline_log(event, sid='-', ok=True, **note):\n"
        "    try:\n"
        "        _PIPELINE_LOG.parent.mkdir(parents=True, exist_ok=True)\n"
        "        from datetime import datetime as _dt, timezone as _tz\n"
        "        ts = _dt.now(_tz.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')\n"
        "        parts = []\n"
        "        for k, v in note.items():\n"
        "            if v is None:\n"
        "                continue\n"
        "            s = str(v)\n"
        "            if ' ' in s or '\"' in s:\n"
        "                s = '\"' + s.replace('\"', '\\\\\"') + '\"'\n"
        "            parts.append(f'{k}={s}')\n"
        "        line = (f'[{ts}] [extractor] [-] [sid={sid}] '\n"
        "                f'event={event} status={\"ok\" if ok else \"FAIL\"}'\n"
        "                + (' ' + ' '.join(parts) if parts else '')\n"
        "                + '\\n')\n"
        "        with open(_PIPELINE_LOG, 'a', encoding='utf-8', errors='replace') as f:\n"
        "            f.write(line)\n"
        "    except Exception:\n"
        "        pass\n"
        "\n"
    )

    # Anchor: insert helpers right before "def main():"
    anchor = "def main():"
    if anchor not in src:
        return False
    src = src.replace(anchor, helper_block + anchor, 1)

    # Wrap the dispatch block inside main() with start / fatal logs.
    #
    # 不能直接 anchor 在 ``args = parser.parse_args()`` 后面 —— main() 后面紧跟
    # 着一行 ``global _OUTPUT_DIR``，Python 要求 ``global X`` 必须出现在
    # **第一次对 X 的引用之前**，否则会抛 SyntaxError 让整个脚本不可用。
    # 0.20.0 第一次切 wheel 就因为这个 bug 让 cot-bridge 把 extract_cot.py spawn 起来后直接
    # 死在 import 阶段（pipeline.log 里能看到 ``event=extract_spawn``，但没有
    # 后续 ``event=session_done``）。0.20.1 改成 anchor 在 ``if args.output:``
    # 紧跟的那行之后，确保 ``global`` 声明已经生效再去读 ``_OUTPUT_DIR``。
    dispatch_old = (
        "    global _OUTPUT_DIR\n"
        "    if args.output:\n"
        "        _OUTPUT_DIR = Path(args.output)\n"
    )
    dispatch_new = (
        dispatch_old
        + "    _pipeline_log('start', sid=(args.session_id or '-'), "
        + "transcript=args.transcript, output_dir=str(_OUTPUT_DIR))\n"
    )
    if dispatch_old in src and "_pipeline_log('start'" not in src:
        src = src.replace(dispatch_old, dispatch_new, 1)

    # Append a fatal-catcher at the end of main() if not already there.
    if "_pipeline_log('fatal'" not in src:
        # heuristic: add after the last "success += 1" / before "if __name__"
        if 'if __name__ == "__main__":' in src:
            src = src.replace(
                'if __name__ == "__main__":',
                (
                    "def _agent_cot_run_main():\n"
                    "    try:\n"
                    "        return main()\n"
                    "    except SystemExit:\n"
                    "        raise\n"
                    "    except Exception as _exc:\n"
                    "        _pipeline_log('fatal', ok=False, "
                    "error=type(_exc).__name__ + ': ' + str(_exc)[:200])\n"
                    "        raise\n\n"
                    'if __name__ == "__main__":'
                ),
                1,
            )
            src = src.replace(
                "if __name__ == \"__main__\":\n    main()",
                "if __name__ == \"__main__\":\n    _agent_cot_run_main()",
                1,
            )

    path.write_text(src, encoding="utf-8")
    return True


def _inject_backend_config_scan_dirs(path: Path) -> bool:
    """Patch ``backend/config.py`` (vendored) with COT_SCAN_DIRS + boot breadcrumb."""
    if not path.is_file():
        return False
    src = path.read_text(encoding="utf-8", errors="replace")
    if _INJECT_MARKER in src:
        return False

    # Ensure ``datetime`` import is present.
    if "from datetime import datetime, timezone" not in src:
        src = src.replace(
            "import os\nfrom pathlib import Path",
            "import json\nimport os\nfrom datetime import datetime, timezone\nfrom pathlib import Path",
            1,
        )

    # Inject helper + COT_SCAN_DIRS just before the per-dir mkdir block.
    anchor = "# v0.18.2: 启动时把数据目录 mkdir 出来"
    if anchor not in src:
        return False

    inject_block = (
        "# " + _INJECT_MARKER + "\n"
        "def _read_runtime_state():\n"
        "    try:\n"
        "        p = Path.home() / '.agent-cot' / 'runtime.json'\n"
        "        if p.is_file():\n"
        "            data = json.loads(p.read_text(encoding='utf-8'))\n"
        "            return data if isinstance(data, dict) else {}\n"
        "    except Exception:\n"
        "        pass\n"
        "    return {}\n"
        "\n"
        "def _normalize_data_root(p: Path) -> Path:\n"
        "    name = p.name\n"
        "    if name == 'data':\n"
        "        return p\n"
        "    if name in ('.agent-cot', '.cursor-cot'):\n"
        "        return p / 'data'\n"
        "    return p\n"
        "\n"
        "def _legacy_scan_roots():\n"
        "    primary = _user_data_root().resolve()\n"
        "    home = Path.home()\n"
        "    cands = [home / '.agent-cot', home / '.agent-cot' / 'data',\n"
        "             home / '.cursor-cot', home / '.cursor-cot' / 'data']\n"
        "    extra = os.environ.get('AGENT_COT_EXTRA_SCAN_ROOTS', '').strip()\n"
        "    if extra:\n"
        "        for chunk in extra.replace(';', os.pathsep).split(os.pathsep):\n"
        "            c = chunk.strip()\n"
        "            if c:\n"
        "                cands.append(Path(c).expanduser())\n"
        "    out, seen = [], set()\n"
        "    for c in cands:\n"
        "        try:\n"
        "            r = c.resolve()\n"
        "        except Exception:\n"
        "            continue\n"
        "        if r == primary or str(r) in seen:\n"
        "            continue\n"
        "        seen.add(str(r))\n"
        "        if r.is_dir():\n"
        "            out.append(r)\n"
        "    return out\n"
        "\n"
        "COT_SCAN_DIRS = [COT_DIR]\n"
        "for _legacy_root in _legacy_scan_roots():\n"
        "    _legacy_cot = _legacy_root / 'cot'\n"
        "    if _legacy_cot.is_dir() and _legacy_cot.resolve() != COT_DIR.resolve():\n"
        "        COT_SCAN_DIRS.append(_legacy_cot)\n"
        "\n"
        "def _config_pipeline_breadcrumb():\n"
        "    try:\n"
        "        lp = Path(os.environ.get('AGENT_COT_PIPELINE_LOG')\n"
        "                  or str(Path.home() / '.agent-cot' / 'logs' / 'pipeline.log')\n"
        "                  ).expanduser()\n"
        "        lp.parent.mkdir(parents=True, exist_ok=True)\n"
        "        ts = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')\n"
        "        sd = ';'.join(str(p) for p in COT_SCAN_DIRS)\n"
        "        with open(lp, 'a', encoding='utf-8', errors='replace') as f:\n"
        "            f.write(f'[{ts}] [backend.config] [-] [sid=-] event=boot status=ok '\n"
        "                    f'cot_dir={COT_DIR} scan_dirs=\"{sd}\"\\n')\n"
        "    except Exception:\n"
        "        pass\n"
        "\n"
        "_config_pipeline_breadcrumb()\n"
        "\n"
    )

    # Inject just BEFORE the comment block that does the mkdir.
    src = src.replace(anchor, inject_block + anchor, 1)
    path.write_text(src, encoding="utf-8")
    return True


def _inject_session_scanner_multi_root(path: Path) -> bool:
    """Patch ``backend/services/session_scanner.py`` to consume ``COT_SCAN_DIRS``."""
    if not path.is_file():
        return False
    src = path.read_text(encoding="utf-8", errors="replace")
    if _INJECT_MARKER in src:
        return False

    # Replace the import line to also pull in COT_SCAN_DIRS.
    old_import = (
        "from config import COT_DIR, COT_REPORTS_DIR, RESPONSE_REPORTS_DIR, "
        "TRANSCRIPTS_DIR, LANGFUSE_CACHE_DIR"
    )
    new_import = (
        "from config import (\n"
        "    COT_DIR,\n"
        "    COT_REPORTS_DIR,\n"
        "    COT_SCAN_DIRS,  # " + _INJECT_MARKER + "\n"
        "    RESPONSE_REPORTS_DIR,\n"
        "    TRANSCRIPTS_DIR,\n"
        "    LANGFUSE_CACHE_DIR,\n"
        ")"
    )
    if old_import in src:
        src = src.replace(old_import, new_import, 1)

    # Patch scan_sessions(): iterate COT_SCAN_DIRS, dedupe by sid.
    old_scan = (
        "    # ── 1) 本机 session ──\n"
        "    if COT_DIR.exists():\n"
        "        for cot_file in sorted(COT_DIR.glob(\"*_cot.json\"), reverse=True):\n"
        "            filename = cot_file.name\n"
        "            if filename.startswith(\".\"):\n"
        "                continue\n"
        "            cot_data = _read_json(cot_file)\n"
        "            if not cot_data:\n"
        "                continue\n"
        "            session_id = cot_data.get(\"session_id\", cot_file.stem.replace(\"_cot\", \"\"))\n"
        "            sessions.append(_build_session_overview(cot_data, session_id))"
    )
    new_scan = (
        "    # ── 1) 本机 session（COT_SCAN_DIRS 多目录扫描 + sid 去重）──\n"
        "    seen_sids = set()  # " + _INJECT_MARKER + "\n"
        "    for _scan_dir in COT_SCAN_DIRS:\n"
        "        if not _scan_dir.exists():\n"
        "            continue\n"
        "        try:\n"
        "            cot_files = sorted(_scan_dir.glob(\"*_cot.json\"), reverse=True)\n"
        "        except OSError:\n"
        "            continue\n"
        "        for cot_file in cot_files:\n"
        "            filename = cot_file.name\n"
        "            if filename.startswith(\".\"):\n"
        "                continue\n"
        "            cot_data = _read_json(cot_file)\n"
        "            if not cot_data:\n"
        "                continue\n"
        "            session_id = cot_data.get(\"session_id\", cot_file.stem.replace(\"_cot\", \"\"))\n"
        "            if session_id in seen_sids:\n"
        "                continue\n"
        "            seen_sids.add(session_id)\n"
        "            sessions.append(_build_session_overview(cot_data, session_id))"
    )
    if old_scan in src:
        src = src.replace(old_scan, new_scan, 1)

    # Patch get_session_cot(): try every scan dir.
    old_get = (
        "    cot_file = COT_DIR / f\"{session_id}_cot.json\"\n"
        "    if cot_file.exists():\n"
        "        return _read_json(cot_file)\n"
        "    return None"
    )
    new_get = (
        "    # " + _INJECT_MARKER + " — try every scan dir.\n"
        "    for _scan_dir in COT_SCAN_DIRS:\n"
        "        cot_file = _scan_dir / f\"{session_id}_cot.json\"\n"
        "        if cot_file.exists():\n"
        "            data = _read_json(cot_file)\n"
        "            if data is not None:\n"
        "                return data\n"
        "    return None"
    )
    if old_get in src:
        src = src.replace(old_get, new_get, 1)

    path.write_text(src, encoding="utf-8")
    return True


def _post_sync_inject_pipeline_log() -> tuple[int, list[str]]:
    """Run all post-sync patchers. Returns (count, summary lines).

    v0.20.1: after every patcher, run ``python -m py_compile`` on the
    target so a malformed inject (e.g. ``global X`` placed after ``X`` is
    referenced) explodes at *build* time, not silently at first hook fire
    on a colleague's machine. 0.20.0 shipped exactly that bug —— the
    ``_pipeline_log('start', …)`` line referenced ``_OUTPUT_DIR`` before
    ``global _OUTPUT_DIR`` could take effect, and the hook chain seemed to
    work (events.jsonl filled in, cot-bridge ran, extract_cot.py was
    spawned) but the child python crashed at import with ``SyntaxError:
    name '_OUTPUT_DIR' is used prior to global declaration`` and no
    cot.json ever appeared. The only place that knew was the orphan
    stderr that goes to /dev/null. Compiling here closes that hole.
    """
    import py_compile

    n = 0
    log_lines: list[str] = []
    targets = [
        (
            _bundled_extractor_full_target() / "scripts" / "extract_cot.py",
            _inject_extract_cot_pipeline_log,
        ),
        (
            _bundled_backend_target() / "config.py",
            _inject_backend_config_scan_dirs,
        ),
        (
            _bundled_backend_target() / "services" / "session_scanner.py",
            _inject_session_scanner_multi_root,
        ),
    ]
    for path, patcher in targets:
        ok = patcher(path)
        if ok:
            n += 1
            log_lines.append(f"  [OK] {patcher.__name__}: patched {path.name}")
        else:
            log_lines.append(
                f"  [..] {patcher.__name__}: skipped {path.name} "
                "(already patched or missing)"
            )
        if path.suffix == ".py" and path.exists():
            try:
                py_compile.compile(str(path), doraise=True)
                log_lines.append(f"       py_compile ok: {path.name}")
            except py_compile.PyCompileError as exc:
                raise RuntimeError(
                    f"post-sync patcher produced invalid python in "
                    f"{path}: {exc}"
                ) from exc
    return n, log_lines


# ─────────────────────────────────────────────────────────────────────────────
# v0.18.6 (agent-cot fork) post-sync rebrand
# ─────────────────────────────────────────────────────────────────────────────
#
# 这个 sync 流程会把 ``cot-extractor/src/*.py`` / ``cot-extractor/scripts/*.py`` /
# ``agent-dashboard/backend/*.py`` 从 *源代码目录* 拉进来当 vendor。源代码目录
# 是 cursor-cot-observer 那边维护的（用 ``cursor_cot`` / ``.cursor-cot`` /
# ``CURSOR_COT_*`` 命名约定）—— agent-cot 不能去改源代码（用户底线："不要
# 修改 cursor-cot-observer 任何东西"），所以 sync 完后我们必须做一遍 *本地
# rebrand*，把刚 vendor 进来的 .py 里 ``cursor_cot`` → ``agent_cot``、
# ``CURSOR_COT_*`` → ``AGENT_COT_*``、``.cursor-cot`` → ``.agent-cot`` 全部
# 替换掉，让 wheel 内的 vendor 跟 agent_cot package 的命名约定完全对齐。
#
# 不替换：
#   * 单独出现的 ``cursor`` 单词（IDE 名）
#   * frontend bundle (.js)（Vite 已 minified，不该再动）
#   * hooks/cursor / hooks/codebuddy 下的 .js（直接编辑就好，
#     不走 sync 链路；它们的 DATA_ROOT 写盘逻辑已经在 agent-cot 的源里改好）
_REBRAND_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("cursor-cot-observer", "agent-cot"),
    ("cursor_cot_observer", "agent_cot"),
    ("cursor_cot", "agent_cot"),
    ("CURSOR_COT_", "AGENT_COT_"),
    (".cursor-cot", ".agent-cot"),
    ("cursor-cot", "agent-cot"),
)

# 只 rebrand 这些被 sync 进来的 vendor 目录里的 .py 文件。其它（assets/hooks/cursor
# 等）由 agent-cot 直接编辑维护，不需要 post-sync rebrand。
_REBRAND_TARGET_DIRS: tuple[Path, ...] = ()  # initialised in _rebrand_synced_assets


def _rebrand_synced_assets() -> tuple[int, int]:
    """Run a global string replace on every vendor .py we just sync'd in.

    Returns ``(files_touched, total_substitutions)``.

    **v0.19.1 重要例外**：``assets/hooks/claude/claude_stream_hook.py`` 故意
    保留 ``.cursor-cot`` / ``cursor_cot`` 字面量 —— 它们是 4 层 fallback 链
    （env > .agent-cot/runtime.json > **.cursor-cot/runtime.json** > 探测）
    的关键一环，让同事即便已经装了 cursor-cot 也能让 Claude 集成立刻工作。
    rebrand 把 ``.cursor-cot`` 改成 ``.agent-cot`` 会让这个 fallback 失效。
    所以 SKIP_REBRAND_FILES 里把它显式排除。
    """
    targets: list[Path] = [
        _bundled_backend_target(),       # assets/backend/*.py
        _bundled_extractor_target(),     # assets/cot-extractor-src/*.py
        _bundled_extractor_full_target(),  # assets/cot-extractor/src+scripts/*.py
        _here() / "assets" / "hooks" / "claude",  # langfuse_hook.py 等（claude_stream_hook 见下）
    ]
    # v0.19.1: agent-cot 维护的 hook，rebrand 必跳过 —— 否则它内部
    # 4 层 fallback 链里对 .cursor-cot 的兼容会被替换掉。
    SKIP_REBRAND_FILES = {"claude_stream_hook.py"}
    files_touched = 0
    total_subs = 0
    for root in targets:
        if not root.is_dir():
            continue
        for f in root.rglob("*.py"):
            if f.name in SKIP_REBRAND_FILES:
                continue
            try:
                body = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue  # unlikely for .py, but skip rather than crash
            new = body
            subs = 0
            for old_str, new_str in _REBRAND_REPLACEMENTS:
                count = new.count(old_str)
                if count:
                    new = new.replace(old_str, new_str)
                    subs += count
            if subs and new != body:
                f.write_text(new, encoding="utf-8")
                files_touched += 1
                total_subs += subs
    return files_touched, total_subs


def _cmd_sync() -> int:
    repo = _project_repo_root()
    if repo is None:
        print("error: cannot find source repo (run from an editable install).")
        return 2

    n_backend, _b = _sync_backend(repo)
    n_frontend, _f = _sync_frontend(repo)
    n_extractor, _e = _sync_extractor(repo)
    n_extractor_full, _ef = _sync_extractor_full(repo)
    n_claude_hooks, _ch = _sync_claude_hooks(repo)

    # v0.18.6 (agent-cot): 把 vendor 进来的 cursor_cot/.cursor-cot/CURSOR_COT_*
    # 全部 rebrand 成 agent_cot / .agent-cot / AGENT_COT_*，跟本包命名对齐。
    rebrand_files, rebrand_subs = _rebrand_synced_assets()

    # v0.20.0: post-sync 注入统一 pipeline.log + COT_SCAN_DIRS 兜底
    inject_n, inject_lines = _post_sync_inject_pipeline_log()

    print(f"backend         : {n_backend} file(s) → {_bundled_backend_target()}")
    print(f"frontend        : {n_frontend} file(s) → {_bundled_frontend_target()}")
    print(f"extractor (lib) : {n_extractor} file(s) → {_bundled_extractor_target()}")
    print(f"extractor (full): {n_extractor_full} file(s) → {_bundled_extractor_full_target()}")
    print(f"claude hooks    : {n_claude_hooks} file(s) → assets/hooks/claude/")
    print(
        f"rebrand pass    : {rebrand_files} file(s) touched, "
        f"{rebrand_subs} substitution(s) (cursor_cot → agent_cot etc.)"
    )
    print(f"pipeline.log inject: {inject_n} file(s) patched")
    for line in inject_lines:
        print(line)
    if n_frontend == 0:
        print(
            "warning: frontend dist is empty — did you forget "
            "`npm run build` in agent-dashboard/frontend/?"
        )
    if n_extractor == 0:
        print(
            "warning: cot-extractor/src/ is empty or missing — backend "
            "OTLP / RAG features will fall back to host-machine sys.path."
        )
    if n_claude_hooks == 0:
        print(
            "warning: claude-code/hooks/ is empty — Claude 27-hook "
            "stream capture won't be installable from the wheel."
        )
    return 0


def _cmd_check() -> int:
    """v0.20.5: pre-build sanity check for ``assets/hooks/*``.

    0.20.4 shipped with ``cursor/cot-stream.js`` and ``cursor/cot-bridge.js``
    replaced by 59-byte fixture stubs (test_init_command.py used to write
    a placeholder into the real source tree when the file was missing). The
    install path silently accepted the stub, so colleagues who ran ``pip
    install observation-agent==0.20.4`` got a working CodeBuddy adapter but
    a completely silent Cursor adapter — events.jsonl was never written.

    This check runs over every hook asset, refuses any ``.js`` / ``.py``
    file under ``assets/hooks/`` that's smaller than 1 KB, and prints a
    summary table. Exit code 0 = ok to build, non-zero = abort. Wire it
    into your release script::

        python -m agent_cot._build_assets check && python -m build --wheel
    """
    hooks_root = _here() / "assets" / "hooks"
    if not hooks_root.is_dir():
        print(f"error: hooks dir missing: {hooks_root}")
        return 2

    THRESHOLD = 1024  # see docstring above
    rows = []
    failed = []
    for f in sorted(hooks_root.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix not in (".js", ".py"):
            continue
        size = f.stat().st_size
        ok = size >= THRESHOLD
        rows.append((f.relative_to(hooks_root), size, ok))
        if not ok:
            failed.append((f, size))

    width = max((len(str(r[0])) for r in rows), default=20)
    print(f"{'relative path':{width}}  {'size':>8}  status")
    print("-" * (width + 22))
    for rel, size, ok in rows:
        mark = "ok" if ok else "STUB!"
        print(f"{str(rel):{width}}  {size:>8}  {mark}")

    if failed:
        print()
        print(f"error: {len(failed)} stub hook(s) detected (< {THRESHOLD} bytes):")
        for f, size in failed:
            print(f"  - {f} ({size} bytes)")
        print()
        print(
            "Fix: restore the real hook bytes "
            "(e.g. from the previous published wheel, or re-export from "
            "the maintained source) before building. Test fixtures must "
            "never write into src/agent_cot/assets/."
        )
        return 1
    print(f"\nall {len(rows)} hook asset(s) look healthy.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "info":
        return _cmd_info()
    if cmd == "sync":
        return _cmd_sync()
    if cmd == "check":
        return _cmd_check()
    print(f"unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
