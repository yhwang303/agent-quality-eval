"""Deterministic re-execution of Cursor's gap tools.

Why this exists
---------------
Cursor's hooks expose results for Shell / MCP / Read / Edit, but several
tools have **no result anywhere on disk** — verified empirically against
this repo on 2026-04-28 by scanning ``%APPDATA%\\Cursor\\logs`` and the
agent transcripts. In particular:

  * Glob, Grep, Delete : no Cursor-side log keeps their results.
  * `Cursor Grep Service.log` only records *queries*, not results.

The good news: these are **deterministic file-system operations**.
Re-running the same call against the same workspace gives THE same result
the agent saw — provided the filesystem hasn't drifted in the few seconds
between Cursor's call and our replay. That is not inference: it is the
identical function over identical inputs. We mark every replayed result
with ``source="reproduced"`` and ``reproduction_lag_ms`` so consumers can
trust it or downgrade it accordingly.

Tools handled
-------------
- ``Glob``        :: re-run via ``pathlib.Path.glob`` mirroring Cursor's
                     auto-prepend rule for ``"**/"``
- ``Grep`` / ``rg`` :: run via the **same rg binary Cursor uses**, found
                       under Cursor's bundled ``@vscode/ripgrep``. Falls
                       back to system ``rg`` then to a Python regex sweep.
- ``Delete``      :: probe-only — never deletes anything; just records
                     whether the path still exists at replay time.

Tools that **cannot** be safely reproduced (and why)
----------------------------------------------------
- ``WebFetch`` / ``WebSearch`` :: depend on external state, results may
  drift. (And they're already covered by the ``agent-tools/<uuid>.txt``
  artifact watcher.)
- ``Task``                    :: a subagent run; result is the subagent's
  full transcript, not reproducible.
- ``GenerateImage``           :: stochastic LLM output.
- ``SemanticSearch``          :: depends on Cursor's embedding index.
- ``ReadLints``               :: depends on which language servers are
  active in Cursor's process (we'd need to drive VS Code's diagnostics).
- ``AskQuestion`` / ``TodoWrite`` :: input *is* the action — no extra
  result to capture beyond what the next user message / state diff shows.

Public API
----------
``reproduce(tool_name: str, input_: dict) -> Optional[dict]``

Returns ``None`` for unsupported tools (caller should fall back to "input
captured, result unknown"). Otherwise returns a dict with at least
``ok: bool`` and tool-specific fields documented per function.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


LOG = logging.getLogger("tool_reproducer")

# Hard caps: results larger than this risk crowding events.jsonl.
GLOB_MAX_RETURN = 200
GLOB_HARD_STOP = 5000          # stop iterating after this many matches
GREP_MAX_RETURN = 200
GREP_TIMEOUT_S = 15
PYREGEX_MAX_FILES = 2000       # fallback only — normally rg handles it


# ─────────────────────────────────────────────────────────────────────────────
# rg binary discovery (use the SAME one Cursor uses when possible)
# ─────────────────────────────────────────────────────────────────────────────

def _find_rg_binary() -> Optional[str]:
    candidates: List[str] = []
    if sys.platform == "win32":
        # Cursor's bundled rg — verified path on this machine 2026-04-28
        bundled = (Path.home()
                   / "AppData/Local/Programs/cursor/resources/app"
                   / "node_modules/@vscode/ripgrep/bin/rg.exe")
        if bundled.exists():
            candidates.append(str(bundled))
    else:
        # macOS .app bundle
        if sys.platform == "darwin":
            mac_bundled = Path("/Applications/Cursor.app/Contents/Resources"
                               "/app/node_modules/@vscode/ripgrep/bin/rg")
            if mac_bundled.exists():
                candidates.append(str(mac_bundled))
        # Linux installed paths vary; try common ones plus PATH
        linux_candidates = [
            "/opt/cursor/resources/app/node_modules/@vscode/ripgrep/bin/rg",
            "/usr/share/cursor/resources/app/node_modules/@vscode/ripgrep/bin/rg",
        ]
        for c in linux_candidates:
            if Path(c).exists():
                candidates.append(c)
    sysrg = shutil.which("rg")
    if sysrg:
        candidates.append(sysrg)
    return candidates[0] if candidates else None


_RG_BIN_CACHE: Optional[str] = None


def rg_binary() -> Optional[str]:
    global _RG_BIN_CACHE
    if _RG_BIN_CACHE is None:
        _RG_BIN_CACHE = _find_rg_binary()
        if _RG_BIN_CACHE:
            LOG.info("rg binary: %s", _RG_BIN_CACHE)
        else:
            LOG.info("rg binary not found; Grep will use Python regex fallback")
    return _RG_BIN_CACHE


# ─────────────────────────────────────────────────────────────────────────────
# Glob — Cursor's spec mirrored
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_glob_pattern(raw: str) -> str:
    """Mirror Cursor's documented behavior:
        > Patterns not starting with "**/" are automatically prepended
        > with "**/" to enable recursive searching.

    Examples:
      "*.py"      -> "**/*.py"
      "src/**"    -> "src/**"   (already has separator semantics)
      "**/x.ts"   -> "**/x.ts"
    """
    p = raw.strip()
    if not p:
        return "**/*"
    if p.startswith("**/"):
        return p
    # Pattern is "anchored" if it starts with a literal directory segment
    # (contains "/" before any glob meta). Heuristic: if first segment has
    # no glob char, treat as anchored.
    head = p.split("/", 1)[0]
    if "/" in p and not any(c in head for c in "*?["):
        return p
    return "**/" + p


def reproduce_glob(input_: Dict[str, Any]) -> Dict[str, Any]:
    """Re-run a Glob call.

    Returns:
      {
        "ok": True,
        "tool": "Glob",
        "matches": [<paths sorted>, ...],   # capped to GLOB_MAX_RETURN
        "match_count": <int>,               # full count up to GLOB_HARD_STOP
        "truncated": bool,
        "target_directory": <resolved abs path>,
        "pattern_normalized": <effective glob>,
      }
    """
    pattern = _normalize_glob_pattern(str(input_.get("glob_pattern") or ""))
    target_raw = input_.get("target_directory")
    target = Path(target_raw).resolve() if target_raw else Path.cwd()
    if not target.exists():
        return {"ok": False, "tool": "Glob",
                "error": f"target_directory does not exist: {target}"}
    matches: List[str] = []
    try:
        for p in target.glob(pattern):
            try:
                if p.is_file():
                    matches.append(str(p))
            except OSError:
                continue
            if len(matches) >= GLOB_HARD_STOP:
                break
    except (OSError, ValueError) as e:
        return {"ok": False, "tool": "Glob", "error": str(e)}
    matches.sort()
    return {
        "ok": True,
        "tool": "Glob",
        "matches": matches[:GLOB_MAX_RETURN],
        "match_count": len(matches),
        "truncated": len(matches) > GLOB_MAX_RETURN,
        "target_directory": str(target),
        "pattern_normalized": pattern,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Grep — prefer Cursor's bundled rg; fall back to Python regex
# ─────────────────────────────────────────────────────────────────────────────

def _build_rg_args(input_: Dict[str, Any], rg: str) -> List[str]:
    args = [rg, "--json"]
    output_mode = (input_.get("output_mode") or "content").lower()
    if output_mode == "files_with_matches":
        args.append("-l")
    elif output_mode == "count":
        args.append("-c")
    if input_.get("-i") or input_.get("ignore_case"):
        args.append("-i")
    if input_.get("multiline"):
        args.extend(["-U", "--multiline-dotall"])
    for ctx_flag, ctx_key in (("-A", "-A"), ("-B", "-B"), ("-C", "-C")):
        v = input_.get(ctx_key)
        if isinstance(v, (int, float)) and int(v) > 0:
            args.extend([ctx_flag, str(int(v))])
    if input_.get("type"):
        args.extend(["--type", str(input_["type"])])
    if input_.get("glob"):
        args.extend(["--glob", str(input_["glob"])])
    head_limit = input_.get("head_limit")
    if isinstance(head_limit, (int, float)) and int(head_limit) > 0:
        args.extend(["--max-count", str(int(head_limit))])
    args.append(str(input_.get("pattern") or ""))
    path = input_.get("path")
    if path and Path(path).exists():
        args.append(str(path))
    return args


def _parse_rg_json(raw: str) -> Dict[str, Any]:
    """Parse rg --json output. rg streams one JSON event per line:
       {"type":"begin","data":{"path":{"text":"..."}}}
       {"type":"match","data":{"path":...,"line_number":N,"lines":{"text":"..."}}}
       {"type":"end",...}
       {"type":"summary",...}
    """
    matches: List[Dict[str, Any]] = []
    file_count = 0
    summary: Optional[Dict[str, Any]] = None
    for ln in raw.splitlines():
        if not ln.strip():
            continue
        try:
            ev = json.loads(ln)
        except Exception:
            continue
        # Defensive: very rarely rg streams a JSON literal that is not an
        # object (seen empirically when passed pathological input). Skip.
        if not isinstance(ev, dict):
            continue
        et = ev.get("type")
        if et == "match":
            d = ev.get("data") or {}
            text = ((d.get("lines") or {}).get("text") or "").rstrip("\n")
            if len(text) > 1000:
                text = text[:1000] + f"…[+{len(text)-1000}ch]"
            matches.append({
                "path": (d.get("path") or {}).get("text"),
                "line_number": d.get("line_number"),
                "line": text,
            })
            if len(matches) >= GREP_MAX_RETURN * 2:  # parse a bit past cap
                break
        elif et == "begin":
            file_count += 1
        elif et == "summary":
            summary = ev.get("data")
    return {"matches": matches, "file_count": file_count, "summary": summary}


def reproduce_grep(input_: Dict[str, Any]) -> Dict[str, Any]:
    """Re-run a Grep call. Tries Cursor's bundled rg first; on failure or
    absence falls back to a small Python regex sweep over the target path.
    """
    pattern = input_.get("pattern")
    if not pattern:
        return {"ok": False, "tool": "Grep", "error": "no pattern"}

    rg = rg_binary()
    if rg:
        args = _build_rg_args(input_, rg)
        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=GREP_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "tool": "Grep",
                    "error": f"rg timeout after {GREP_TIMEOUT_S}s",
                    "rg_binary": rg}
        except OSError as e:
            return {"ok": False, "tool": "Grep",
                    "error": f"rg spawn failed: {e}",
                    "rg_binary": rg}
        parsed = _parse_rg_json(proc.stdout)
        return {
            "ok": True,
            "tool": "Grep",
            "via": "rg",
            "rg_binary": rg,
            "exit_code": proc.returncode,
            "match_count": len(parsed["matches"]),
            "matches": parsed["matches"][:GREP_MAX_RETURN],
            "file_count": parsed["file_count"],
            "truncated": len(parsed["matches"]) > GREP_MAX_RETURN,
            "stderr_head": proc.stderr[:500] if proc.stderr else "",
        }

    return _grep_python_fallback(input_)


def _grep_python_fallback(input_: Dict[str, Any]) -> Dict[str, Any]:
    pattern = str(input_.get("pattern") or "")
    flags = re.IGNORECASE if input_.get("-i") or input_.get("ignore_case") else 0
    if input_.get("multiline"):
        flags |= re.DOTALL | re.MULTILINE
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return {"ok": False, "tool": "Grep", "error": f"regex compile: {e}"}
    target = Path(input_.get("path") or os.getcwd())
    matches: List[Dict[str, Any]] = []
    files_visited = 0
    if target.is_file():
        candidate_files = [target]
    else:
        candidate_files = []
        for p in target.rglob("*"):
            if p.is_file() and not _is_skipworthy(p):
                candidate_files.append(p)
                if len(candidate_files) >= PYREGEX_MAX_FILES:
                    break
    for f in candidate_files:
        files_visited += 1
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                for n, line in enumerate(fh, start=1):
                    if regex.search(line):
                        matches.append({
                            "path": str(f),
                            "line_number": n,
                            "line": line.rstrip("\n")[:1000],
                        })
                        if len(matches) >= GREP_MAX_RETURN:
                            break
        except OSError:
            continue
        if len(matches) >= GREP_MAX_RETURN:
            break
    return {
        "ok": True,
        "tool": "Grep",
        "via": "python_regex_fallback",
        "match_count": len(matches),
        "matches": matches,
        "file_count": files_visited,
        "truncated": len(matches) >= GREP_MAX_RETURN,
    }


_SKIP_DIRS = {".git", "node_modules", "dist", "build", ".venv", "venv",
              "__pycache__", ".idea", ".vscode", ".cursor"}


def _is_skipworthy(p: Path) -> bool:
    return any(part in _SKIP_DIRS for part in p.parts)


# ─────────────────────────────────────────────────────────────────────────────
# Delete — probe only, never actually deletes
# ─────────────────────────────────────────────────────────────────────────────

def reproduce_delete(input_: Dict[str, Any]) -> Dict[str, Any]:
    """Observe whether the target still exists, **without deleting**.

    Semantics:
      - still_exists=False  -> Cursor's delete probably succeeded
                               (or path never existed)
      - still_exists=True   -> delete failed, was deferred, or only
                               applied to a different path
    """
    path = (input_.get("path") or input_.get("file_path")
            or input_.get("filepath"))
    if not path:
        return {"ok": False, "tool": "Delete", "error": "no path"}
    p = Path(path)
    return {
        "ok": True,
        "tool": "Delete",
        "checked_path": str(p),
        "still_exists": p.exists(),
        "is_file": p.is_file() if p.exists() else False,
        "is_dir": p.is_dir() if p.exists() else False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

# Map Cursor's tool name to a reproduction function. Aliases (e.g. "rg"
# alongside "Grep") map to the same reproducer because cot-extractor and
# transcript both use both names interchangeably depending on era.
_REGISTRY = {
    "Glob": reproduce_glob,
    "Grep": reproduce_grep,
    "rg": reproduce_grep,
    "Delete": reproduce_delete,
}


def reproduce(tool_name: str, input_: Dict[str, Any],
              *, max_runtime_s: float = GREP_TIMEOUT_S + 2) -> Optional[Dict[str, Any]]:
    """Dispatcher. Returns None for unsupported tools.

    The result dict, when non-None, always carries:
      - ``ok``         : bool
      - ``tool``       : canonical tool name
      - ``elapsed_ms`` : reproduction wall-clock cost
      - ``reproduced_at_ms`` : timestamp when reproduction completed
    """
    fn = _REGISTRY.get(tool_name)
    if fn is None:
        return None
    if not isinstance(input_, dict):
        return {"ok": False, "tool": tool_name, "error": "non-dict input"}
    started = time.monotonic()
    try:
        out = fn(input_)
    except Exception as e:  # never let reproduction crash the watcher
        LOG.exception("reproducer crashed for tool=%s", tool_name)
        return {"ok": False, "tool": tool_name, "error": f"{type(e).__name__}: {e}"}
    elapsed_ms = int((time.monotonic() - started) * 1000)
    if not isinstance(out, dict):
        return {"ok": False, "tool": tool_name, "error": "reproducer returned non-dict"}
    out.setdefault("ok", True)
    out.setdefault("tool", tool_name)
    out["elapsed_ms"] = elapsed_ms
    out["reproduced_at_ms"] = int(time.time() * 1000)
    if elapsed_ms > max_runtime_s * 1000:
        out["slow_warning"] = True
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI for ad-hoc testing
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Reproduce a single tool call.")
    p.add_argument("tool", choices=sorted(_REGISTRY.keys()))
    p.add_argument("--input", required=True,
                   help="JSON literal of the tool input")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        inp = json.loads(args.input)
    except Exception as e:
        print(f"bad --input JSON: {e}", file=sys.stderr)
        return 2
    out = reproduce(args.tool, inp)
    if out is None:
        print(f"tool {args.tool} is not reproducible")
        return 3
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
