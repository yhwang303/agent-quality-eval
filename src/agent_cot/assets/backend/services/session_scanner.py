"""
Session 扫描器 — 扫描本地 JSON 文件，加载所有 session 数据
简单直接：每个 *_cot.json 文件就是一个独立的 session

v0.18.0：新增 central uplink 来源——
扫 ``~/.agent-cot-central/users/<owner>/cot/<sid>_cot.json``，
session_id 在前端以 ``<owner>::<sid>`` 形式呈现，避免与本机 session 撞 id。
"""
import hashlib
import json
import os
import re
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def _hidden_subprocess_kwargs() -> Dict[str, Any]:
    """Avoid a one-frame python.exe console window when refreshing on Windows."""
    if sys.platform != "win32":
        return {}
    kwargs: Dict[str, Any] = {}
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    kwargs["startupinfo"] = startupinfo
    return kwargs
from config import (
    COT_DIR,
    COT_REPORTS_DIR,
    COT_SCAN_DIRS,  # AGENT_COT_PIPELINE_LOG_INJECT_v1
    RESPONSE_REPORTS_DIR,
    TRANSCRIPTS_DIR,
    LANGFUSE_CACHE_DIR,
)

# v0.18.0: central uplink 来源
try:
    from services.uplink_receiver import iter_central_cot_files  # type: ignore
except Exception:
    def iter_central_cot_files():  # type: ignore
        return []

# 中央 session_id 命名空间分隔符。前端以此切割出 owner / 真实 sid
CENTRAL_SID_SEP = "::"


def _clean_project_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.replace("\\", "/").rstrip("/")


def _project_name_from_path(path_text: str) -> str:
    clean = _clean_project_path(path_text)
    if not clean:
        return ""
    return clean.split("/")[-1] or clean


def _project_info_from_path(path_text: Any, source: str) -> Optional[Dict[str, str]]:
    clean = _clean_project_path(path_text)
    if not clean:
        return None
    return {
        "project_name": _project_name_from_path(clean) or "Unknown Project",
        "project_path": clean,
        "project_id": hashlib.sha1(clean.lower().encode("utf-8", errors="replace")).hexdigest()[:12],
        "project_source": source,
    }


def _looks_like_project_path(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return bool(
        re.match(r"^[A-Za-z]:[\\/][^:*?\"<>|]+", text)
        or text.startswith("/")
        or text.startswith("~")
    )


def _candidate_project_paths(payload: Any) -> List[str]:
    keys = {
        "workspace_roots",
        "workspace_root",
        "workspace",
        "workspace_dir",
        "workspace_path",
        "cwd",
        "project_dir",
        "project_path",
        "root",
        "repo",
        "repository",
    }
    found: List[str] = []

    def visit(value: Any, key_hint: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                k = str(key)
                if k in keys or any(part in k.lower() for part in ("workspace", "project", "cwd")):
                    visit(child, k)
            return
        if isinstance(value, list):
            for child in value[:8]:
                visit(child, key_hint)
            return
        if key_hint and _looks_like_project_path(value):
            found.append(str(value))

    visit(payload)
    return found


def _first_project_info_from_payload(payload: Any, source: str) -> Optional[Dict[str, str]]:
    for path_text in _candidate_project_paths(payload):
        info = _project_info_from_path(path_text, source)
        if info:
            return info
    return None


def _is_git_root(path: Path) -> bool:
    try:
        return path.is_dir() and (path / ".git").exists()
    except OSError:
        return False


def _nearest_project_root(path_text: Any) -> str:
    clean = _clean_project_path(path_text)
    if not clean:
        return ""
    path = Path(clean)
    if path.suffix:
        path = path.parent
    try:
        resolved = path.expanduser()
        cur = resolved if resolved.exists() else path
        for candidate in [cur, *cur.parents]:
            if _is_git_root(candidate):
                return _clean_project_path(candidate)
    except Exception:
        pass
    return _clean_project_path(path)


def _project_info_from_events_file(path_text: str) -> Optional[Dict[str, str]]:
    path = Path(path_text)
    if not path.is_file() or path.name.lower() != "events.jsonl":
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                if idx >= 40:
                    break
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                info = (
                    _first_project_info_from_payload(ev.get("payload"), "event_payload")
                    or _first_project_info_from_payload(ev.get("brief_input"), "event_brief_input")
                )
                if info:
                    return info
    except OSError:
        return None
    return None


def _project_info_from_observed_events(cot_data: Dict) -> Optional[Dict[str, str]]:
    observed = cot_data.get("observed_events") if isinstance(cot_data, dict) else None
    if not isinstance(observed, dict):
        return None
    paths = observed.get("events_paths")
    if not isinstance(paths, list):
        return None
    for path_text in paths[:4]:
        info = _project_info_from_events_file(_clean_project_path(path_text))
        if info:
            return info
    return None


_CODEBUDDY_HISTORY_RE = re.compile(r"/history/([0-9a-f]{32})(?:/([0-9a-f]{24,64}))?/", re.IGNORECASE)


def _codebuddy_workspace_hash(transcript_path: str) -> tuple[str, str]:
    clean = _clean_project_path(transcript_path)
    match = _CODEBUDDY_HISTORY_RE.search(f"{clean}/")
    if not match:
        return "", ""
    return match.group(1).lower(), (match.group(2) or "").lower()


def _workspace_hash_variants(path: Path) -> set[str]:
    clean = _clean_project_path(path)
    variants = {
        clean,
        clean.lower(),
        clean.replace("/", "\\"),
        clean.lower().replace("/", "\\"),
        clean.replace("\\", "/"),
        clean.lower().replace("\\", "/"),
    }
    return {hashlib.md5(v.encode("utf-8", errors="replace")).hexdigest() for v in variants if v}


@lru_cache(maxsize=1)
def _windows_available_drives() -> List[str]:
    """Return only drives that Windows reports as *present*, avoiding hangs.

    ``Path("A:/").exists()`` will spin for seconds when a floppy or a
    disconnected network drive is mapped; users see the backend hang on
    first request. ``GetLogicalDrives`` returns a bit-mask sourced from the
    kernel drive table without touching the media, so we can filter first
    and only ``exists()`` on real drives.
    """
    if sys.platform != "win32":
        return []
    try:
        import ctypes

        mask = ctypes.windll.kernel32.GetLogicalDrives()
    except Exception:
        # Fall back to the full A-Z sweep if ctypes is unavailable.
        return list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    letters: List[str] = []
    for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        if mask & (1 << i):
            letters.append(letter)
    return letters


def _local_project_hash_index() -> Dict[str, str]:
    roots: List[Path] = []
    raw_roots = os.environ.get("AGENT_COT_PROJECT_ROOTS") or ""
    for item in raw_roots.replace(";", os.pathsep).split(os.pathsep):
        item = item.strip()
        if item:
            roots.append(Path(item).expanduser())

    if sys.platform == "win32":
        for drive in _windows_available_drives():
            root = Path(f"{drive}:/")
            if root.exists():
                roots.append(root)
    else:
        roots.extend([Path.home(), Path("/workspace"), Path("/workspaces")])
    roots.extend([Path.home(), Path.home() / "Desktop", Path.home() / "Documents"])

    paths: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            if not root.exists():
                continue
            root_key = str(root.resolve()).lower()
        except Exception:
            continue
        if root_key in seen:
            continue
        seen.add(root_key)
        paths.append(root)
        try:
            for child in root.iterdir():
                if child.is_dir():
                    paths.append(child)
        except OSError:
            continue

    out: Dict[str, str] = {}
    for path in paths:
        try:
            clean = _clean_project_path(path.resolve())
        except Exception:
            clean = _clean_project_path(path)
        if not clean:
            continue
        for h in _workspace_hash_variants(Path(clean)):
            out.setdefault(h, clean)
    return out


def _codebuddy_storage_root(transcript_path: str, workspace_hash: str) -> Optional[Path]:
    clean = _clean_project_path(transcript_path)
    marker = f"/history/{workspace_hash}"
    idx = clean.lower().find(marker.lower())
    if idx < 0:
        return None
    return Path(clean[:idx])


def _project_info_from_codebuddy_file_tree(transcript_path: str, workspace_hash: str, session_hash: str) -> Optional[Dict[str, str]]:
    storage_root = _codebuddy_storage_root(transcript_path, workspace_hash)
    if not storage_root:
        return None
    candidates = []
    if session_hash:
        candidates.append(storage_root / "file-tree" / workspace_hash / session_hash / "file-tree.json")
    candidates.extend((storage_root / "file-tree" / workspace_hash).glob("*/file-tree.json") if (storage_root / "file-tree" / workspace_hash).is_dir() else [])

    for file_tree in candidates[:12]:
        data = _read_json(file_tree)
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            paths = [item.get("filePath")]
            for version in item.get("versions") or []:
                if isinstance(version, dict):
                    paths.extend(version.get("createdEntries") or [])
            for path_text in paths:
                if _looks_like_project_path(path_text):
                    root = _nearest_project_root(path_text)
                    info = _project_info_from_path(root, "codebuddy_file_tree")
                    if info:
                        return info
    return None


def _project_info_from_codebuddy_history(transcript_path: str) -> Optional[Dict[str, str]]:
    workspace_hash, session_hash = _codebuddy_workspace_hash(transcript_path)
    if not workspace_hash:
        return None

    path_text = _local_project_hash_index().get(workspace_hash)
    if path_text:
        return _project_info_from_path(path_text, "codebuddy_workspace_hash")

    return _project_info_from_codebuddy_file_tree(transcript_path, workspace_hash, session_hash)


def _slug_project_name(value: str) -> str:
    text = str(value or "").strip()
    text = text.lstrip("-")
    if len(text) > 2 and text[1] == "-" and text[0].isalpha():
        drive = text[0]
        text = (text[2:] or text).lstrip("-")
        if not text:
            return drive.upper()
    if "--" in text:
        parts = [part for part in text.split("--") if part]
        if parts:
            return parts[-1]
    return text


def _extract_project_info(cot_data: Dict) -> Dict[str, str]:
    # User-uploaded traces are intentionally NOT attributed to any real
    # project. They live in a dedicated virtual project so the sidebar can
    # render them as a separate section without polluting CodeBuddy/Cursor/
    # Codex/Claude project trees.
    if isinstance(cot_data, dict) and cot_data.get("_uploaded"):
        return {
            "project_name": "Uploaded Traces",
            "project_path": "",
            "project_id": "__uploaded__",
            "project_source": "upload",
        }
    meta = cot_data.get("session_meta") if isinstance(cot_data.get("session_meta"), dict) else {}
    roots = meta.get("workspace_roots") if isinstance(meta, dict) else None
    if isinstance(roots, list):
        for root in roots:
            info = _project_info_from_path(root, "workspace_roots")
            if info:
                return info

    for payload, source in (
        (meta, "session_meta"),
        (cot_data, "cot_payload"),
    ):
        info = _first_project_info_from_payload(payload, source)
        if info:
            return info

    transcript_path = ""
    if isinstance(meta, dict):
        transcript_path = _clean_project_path(meta.get("transcript_path"))
    transcript_path = transcript_path or _clean_project_path(cot_data.get("transcript_path"))

    if transcript_path:
        info = _project_info_from_codebuddy_history(transcript_path)
        if info:
            return info

    info = _project_info_from_observed_events(cot_data)
    if info:
        return info

    if transcript_path:
        info = _project_info_from_events_file(transcript_path)
        if info:
            return info

        parts = [part for part in transcript_path.split("/") if part]
        for marker in ("projects", "workspaces"):
            if marker in parts:
                idx = parts.index(marker)
                if idx + 1 < len(parts):
                    raw_name = parts[idx + 1]
                    return {
                        "project_name": _slug_project_name(raw_name) or "Unknown Project",
                        "project_path": "",
                        "project_id": hashlib.sha1(f"{marker}:{raw_name}".lower().encode("utf-8", errors="replace")).hexdigest()[:12],
                        "project_source": "transcript_path",
                    }

    return {
        "project_name": "Unknown Project",
        "project_path": "",
        "project_id": "unknown-project",
        "project_source": "fallback",
    }


def _extract_topic(cot_data: Dict) -> str:
    """
    从 CoT 数据中提取 session 主题：
    取第一个有 user_query 的 turn，截取前 40 个字符作为主题。
    """
    turns = cot_data.get("turns", [])
    for turn in turns:
        query = turn.get("user_query", "").strip()
        if query:
            clean = query.replace("\r", "").replace("\n", " ").strip()
            return clean[:40] + ("..." if len(clean) > 40 else "")
    dist = cot_data.get("tool_call_distribution", {})
    if dist:
        tools = ", ".join(f"{k}×{v}" for k, v in dist.items())
        return f"工具调用: {tools}"
    return "未知主题"


def _read_json(path: Path) -> Optional[Dict]:
    """安全读取 JSON 文件"""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _is_subagent_sid(session_id: str) -> bool:
    return str(session_id or "").startswith("agent-")


def _merge_subagent_timeline(parent_cot: Dict, parent_sid: str) -> Dict:
    timeline = list(parent_cot.get("subagent_timeline") or [])
    seen = {
        str(item.get("sub_agent_id"))
        for item in timeline
        if isinstance(item, dict) and item.get("sub_agent_id")
    }
    marker = f"/{parent_sid}/subagents/"

    for scan_dir in COT_SCAN_DIRS:
        if not scan_dir.exists():
            continue
        try:
            candidates = sorted(scan_dir.glob("agent-*_cot.json"))
        except OSError:
            continue
        for cot_file in candidates:
            sub = _read_json(cot_file)
            if not sub:
                continue
            sub_sid = str(sub.get("session_id") or cot_file.stem.replace("_cot", ""))
            if sub_sid in seen:
                continue
            transcript_path = str(sub.get("transcript_path") or "").replace("\\", "/")
            if marker not in transcript_path:
                continue
            summary = _build_subagent_summary(sub, sub_sid)
            if summary:
                timeline.append(summary)
                seen.add(sub_sid)

    if timeline:
        parent_cot["subagent_timeline"] = timeline
    return parent_cot


def _build_subagent_summary(sub_cot: Dict, sub_sid: str) -> Optional[Dict[str, Any]]:
    turns = sub_cot.get("turns") or []
    if not turns:
        return None

    prompt_preview = ""
    steps: list[Dict[str, Any]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        if not prompt_preview:
            prompt_preview = str(turn.get("user_query") or turn.get("question") or "")[:240]
        for step in turn.get("steps", []) or []:
            if isinstance(step, dict):
                steps.append(step)

    tool_distribution: Dict[str, int] = {}
    model = None
    for step in steps:
        tool_name = step.get("tool_name") or step.get("name")
        if tool_name:
            tool_distribution[str(tool_name)] = tool_distribution.get(str(tool_name), 0) + 1
        if model is None:
            otel = step.get("otel") if isinstance(step.get("otel"), dict) else {}
            model = otel.get("model")

    otel_view = sub_cot.get("otel_view") if isinstance(sub_cot.get("otel_view"), dict) else {}
    actual = otel_view.get("actual_token_usage") if isinstance(otel_view.get("actual_token_usage"), dict) else {}
    totals = otel_view.get("totals") if isinstance(otel_view.get("totals"), dict) else {}
    model = model or otel_view.get("model")

    return {
        "sub_agent_id": sub_sid,
        "agent_type": sub_cot.get("agent_type") or "claude-task",
        "prompt_preview": prompt_preview,
        "phase": "merged_from_disk",
        "model": model,
        "input_tokens": actual.get("input_tokens", totals.get("input_tokens", 0)),
        "output_tokens": actual.get("output_tokens", totals.get("output_tokens", 0)),
        "cost_usd": totals.get("cost_usd", 0),
        "total_steps": len(steps),
        "tool_call_distribution": tool_distribution,
    }


def _iso_from_ms(value: Any) -> Optional[str]:
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return None
    if ms <= 0:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _ms_from_iso(value: Any) -> Optional[int]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return None


def _session_occurred_at(cot_data: Dict) -> str:
    """Return the stable latest real activity time for SessionList ordering.

    Bug fix: for a session whose *latest* turn is still in progress (no
    turn_end_ms_observed / step timestamps yet — common for harnesses that
    don't emit per-step observed timestamps), the turn/step candidates below
    can only ever resolve to that turn's *start* time, so the list kept
    showing "session start" instead of "most recent interaction" while the
    turn was still running (explicit user report: "时间显示的是session开始的
    时间，正确的应该是显示最近一次交互会话的时间"). The top-level `extracted_at`
    field is rewritten every time the extractor re-processes the file (which
    happens continuously while a session is active), so it's a much fresher
    proxy for "we saw activity as recently as X" — always include it as a
    candidate (not just a last-resort fallback) so it wins whenever it's more
    recent than any completed-turn timestamp.
    """
    candidates: list[tuple[int, str]] = []

    top_level_extracted_at = cot_data.get("extracted_at")
    if isinstance(top_level_extracted_at, str) and top_level_extracted_at.strip():
        ms = _ms_from_iso(top_level_extracted_at)
        if ms:
            candidates.append((ms, top_level_extracted_at))

    for key in ("session_last_activity_at", "last_activity_at", "updated_at", "session_started_at", "started_at", "created_at"):
        value = cot_data.get(key)
        if isinstance(value, str) and value.strip():
            ms = _ms_from_iso(value)
            if ms:
                candidates.append((ms, value))

    meta = cot_data.get("session_meta")
    if isinstance(meta, dict):
        for key in ("session_end_ms_observed", "session_start_ms_observed"):
            observed = _iso_from_ms(meta.get(key))
            if observed:
                ms = _ms_from_iso(observed)
                if ms:
                    candidates.append((ms, observed))

    for turn in cot_data.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        for key in ("turn_end_ms_observed", "turn_start_ms_observed"):
            observed = _iso_from_ms(turn.get(key))
            if observed:
                ms = _ms_from_iso(observed)
                if ms:
                    candidates.append((ms, observed))
        for key in ("turn_start_time", "_interaction_time"):
            value = turn.get(key)
            if isinstance(value, str) and value.strip():
                ms = _ms_from_iso(value)
                if ms:
                    candidates.append((ms, value))
        for step in turn.get("steps") or []:
            if isinstance(step, dict):
                ts = step.get("timestamp")
                if isinstance(ts, str) and ts.strip():
                    ms = _ms_from_iso(ts)
                    if ms:
                        candidates.append((ms, ts))

    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return str(cot_data.get("extracted_at") or "")


_CODEX_REFRESH_LAST = 0.0


def _codex_rollout_files(session_id: Optional[str] = None) -> List[Path]:
    root = Path.home() / ".codex" / "sessions"
    if not root.exists():
        return []
    try:
        files = list(root.glob("*/*/*/*.jsonl"))
    except OSError:
        return []
    if session_id:
        needle = session_id.replace("codex-", "").lower()
        files = [p for p in files if needle in p.name.lower()]
    return files


def _rollout_sid(path: Path) -> Optional[str]:
    parts = path.stem.split("-")
    if len(parts) >= 6 and parts[0] == "rollout":
        return "-".join(parts[-5:])
    return None


def _maybe_refresh_codex_cot(session_id: Optional[str] = None) -> None:
    """Refresh Codex COT if native rollout JSONL is newer than *_cot.json."""
    global _CODEX_REFRESH_LAST
    now = time.time()
    if now - _CODEX_REFRESH_LAST < 2.0:
        return

    rollouts = _codex_rollout_files(session_id)
    if not rollouts:
        return

    stale = False
    for rollout in rollouts:
        try:
            sid = _rollout_sid(rollout)
            if not sid:
                continue
            target = COT_DIR / f"codex-{sid}_cot.json"
            if not target.exists() or rollout.stat().st_mtime > target.stat().st_mtime:
                stale = True
                break
        except OSError:
            stale = True
            break
    if not stale:
        return

    collector_candidates = [
        Path.home() / ".codex" / "hooks" / "codex_sidecar_collector.py",
        Path(__file__).resolve().parents[2] / "hooks" / "codex" / "codex_sidecar_collector.py",
    ]
    collector = next((p for p in collector_candidates if p.is_file()), None)
    if collector is None:
        return

    cmd = [sys.executable, str(collector), "--recent-seconds", "172800"]
    if session_id:
        cmd.extend(["--session-id", session_id.replace("codex-", "")])
    try:
        _CODEX_REFRESH_LAST = now
        subprocess.run(
            cmd,
            cwd=str(Path.home() / ".codex"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
            check=False,
            **_hidden_subprocess_kwargs(),
        )
    except Exception:
        pass


def _extract_otel_kpi(cot_data: Dict) -> Optional[Dict[str, Any]]:
    """v0.11.2：从 cot.otel_view 抽 model / cost / cache hit 等关键 KPI 给 SessionList 用。

    仅抽『常驻列表行能用上的几项』，避免 SessionList 接口被吹成完整 OTel view；
    不存在 otel_view（老 session）时返回 None。
    """
    otel = cot_data.get("otel_view") or {}
    if not otel:
        return None

    actual = otel.get("actual_token_usage") or {}
    runtime = otel.get("client_runtime") or {}
    totals = otel.get("totals") or {}

    in_tok = actual.get("input_tokens", 0) or 0
    out_tok = actual.get("output_tokens", 0) or 0
    cache_read = actual.get("cache_read_tokens", 0) or 0
    cache_write = actual.get("cache_write_tokens", 0) or 0
    non_cache_in = max(0, in_tok - cache_read - cache_write)
    cache_hit_rate: Optional[float] = None
    denom = non_cache_in + cache_read + cache_write
    if denom > 0:
        cache_hit_rate = round(cache_read / denom, 4)

    cost_usd = otel.get("actual_cost_usd")
    if cost_usd is None:
        cost_usd = totals.get("cost_usd")

    return {
        "model": otel.get("model") or "unknown",
        "model_source": otel.get("model_source"),
        "agent_name": otel.get("agent_name"),
        "provider": otel.get("provider"),
        "cost_usd": cost_usd,
        "full_price_cost_usd": (actual.get("full_price_cost_usd")
                                 if isinstance(actual, dict) else None),
        "input_tokens": in_tok or totals.get("input_tokens", 0),
        "output_tokens": out_tok or totals.get("output_tokens", 0),
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "cache_hit_rate": cache_hit_rate,
        "cursor_version": runtime.get("cursor_version"),
        "events_count": runtime.get("events_count"),
        "has_actual_usage": bool(actual),
    }


def _guess_agent_type(session_id: str, cot_data: Dict) -> str:
    """v0.20.7: cot.json 缺 ``agent_type`` 字段时的启发式判定。

    判定顺序（高置信度在前）：
      1. session_id 以 ``codebuddy-`` 前缀 → codebuddy
      2. session_id 以 ``vscode-`` 前缀 → vscode (Copilot)
      3. otel.model 以 ``claude-`` 开头 / provider=anthropic → claude
      4. 兜底 → cursor（历史最常见）
    """
    sid = (session_id or "").lower()
    if sid.startswith("codebuddy-"):
        return "codebuddy"
    if sid.startswith("vscode-"):
        return "vscode"
    otel = (cot_data.get("session_otel") or {}) if isinstance(cot_data, dict) else {}
    model = (otel.get("model") or "").lower() if isinstance(otel, dict) else ""
    provider = (otel.get("provider") or "").lower() if isinstance(otel, dict) else ""
    if model.startswith("claude-") or provider == "anthropic":
        return "claude"
    # cursor 用 hex uuid（无前缀），多数老 session 都是它
    return "cursor"


def _build_session_overview(
    cot_data: Dict,
    session_id: str,
    *,
    owner: Optional[str] = None,
    host: Optional[str] = None,
    received_at: Optional[str] = None,
) -> Dict[str, Any]:
    """把一份 cot.json 转成 SessionList 行需要的 overview 字典。

    抽取的核心动作：取 topic / 时间 / 统计 / OTel KPI / response report。
    本机 session 与 central session 完全共用这个函数 —— owner 仅当来自 central
    时才会传，本机 session owner = None（前端理解为『我自己』）。
    """
    response_report = None
    has_transcript = False
    if owner is None:
        # 本机 session 才需要查附属 report / transcript（central 只搬 cot.json）
        response_report = _read_json(RESPONSE_REPORTS_DIR / f"{session_id}_report.json")
        has_transcript = (TRANSCRIPTS_DIR / f"{session_id}_full.json").exists()

    response_score = None
    if response_report:
        summary = response_report.get("summary", {})
        avg_ocs = summary.get("avg_ocs")
        if avg_ocs is not None and isinstance(avg_ocs, (int, float)):
            response_score = round(float(avg_ocs), 3)

    otel_kpi = _extract_otel_kpi(cot_data)

    # v0.20.7: 暴露 agent_type（cot_extractor._detect_agent_type 写的真值）
    # 给前端 SessionList 标 IDE 来源徽章。cot.json 未带这个字段（极老 session）
    # 或值为 "unknown"（cot_extractor 自报检测失败）时做启发式 fallback：
    # session_id 前缀 + otel.model 形态。
    raw_agent_type = cot_data.get("agent_type")
    if not raw_agent_type or str(raw_agent_type).lower() == "unknown":
        agent_type = _guess_agent_type(session_id, cot_data)
    else:
        agent_type = raw_agent_type

    project_info = _extract_project_info(cot_data)

    info = {
        "session_id": session_id,
        "topic": _extract_topic(cot_data),
        "extracted_at": _session_occurred_at(cot_data),
        "transcript_path": cot_data.get("transcript_path", ""),
        "total_turns": len(cot_data.get("turns", [])),
        "total_tool_calls": cot_data.get("total_tool_calls", 0),
        "total_thinking_steps": cot_data.get("total_thinking_steps", 0),
        "total_strategy_shifts": cot_data.get("total_strategy_shifts", 0),
        "avg_complexity": cot_data.get("avg_complexity", 0),
        "avg_steps_per_turn": cot_data.get("avg_steps_per_turn", 0),
        "tool_call_distribution": cot_data.get("tool_call_distribution", {}),
        "has_response_report": response_report is not None,
        "has_transcript": has_transcript,
        "response_score": response_score,
        "otel": otel_kpi,
        "agent_type": agent_type,

        **project_info,
    }
    if owner:
        # central 来源 —— 把 owner / host / received_at 暴露给前端筛选用
        info["owner"] = owner
        info["source"] = "uplink"
        if host:
            info["host"] = host
        if received_at:
            info["received_at"] = received_at
    else:
        info["owner"] = None  # 显式 null，前端区分『本机』
        info["source"] = "local"
    return info


def scan_sessions() -> List[Dict[str, Any]]:
    """
    扫描 cot/ 目录 + central uplink 目录，发现所有 session。
    每个 *_cot.json 文件就是一个独立的 session。
    返回 session 概览列表，按时间倒序排列。

    v0.18.0：合并本机 + 中央两种来源。中央 session 的 ``session_id`` 字段会被
    重写为 ``<owner>::<原sid>``，避免与本机 session 撞 id；前端筛选时以
    ``owner`` 字段为准。
    """
    _maybe_refresh_codex_cot()

    sessions: List[Dict[str, Any]] = []

    # ── 1) 本机 session（COT_SCAN_DIRS 多目录扫描 + sid 去重）──
    seen_sids = set()  # AGENT_COT_PIPELINE_LOG_INJECT_v1
    for _scan_dir in COT_SCAN_DIRS:
        if not _scan_dir.exists():
            continue
        try:
            cot_files = sorted(_scan_dir.glob("*_cot.json"), reverse=True)
        except OSError:
            continue
        for cot_file in cot_files:
            filename = cot_file.name
            if filename.startswith("."):
                continue
            cot_data = _read_json(cot_file)
            if not cot_data:
                continue
            session_id = cot_data.get("session_id", cot_file.stem.replace("_cot", ""))
            if _is_subagent_sid(session_id):

                continue

            if session_id in seen_sids:
                continue
            seen_sids.add(session_id)
            sessions.append(_build_session_overview(cot_data, session_id))

    # ── 2) Central uplink session（来自小组同事） ──
    try:
        for owner, cot_file in iter_central_cot_files():
            cot_data = _read_json(cot_file)
            if not cot_data:
                continue
            raw_sid = cot_data.get("session_id") or cot_file.stem.replace("_cot", "")
            uplink_meta = cot_data.get("_uplink") or {}
            host = uplink_meta.get("host") if isinstance(uplink_meta, dict) else None
            received_at = uplink_meta.get("received_at") if isinstance(uplink_meta, dict) else None
            # 命名空间化避免与本机撞 id
            namespaced_sid = f"{owner}{CENTRAL_SID_SEP}{raw_sid}"
            sessions.append(_build_session_overview(
                cot_data, namespaced_sid,
                owner=owner, host=host, received_at=received_at,
            ))
    except Exception:
        # 中央扫描失败完全不影响本机 list
        import traceback
        traceback.print_exc()

    sessions.sort(key=lambda s: s.get("extracted_at", ""), reverse=True)
    return sessions


def _split_central_sid(session_id: str) -> tuple[Optional[str], str]:
    """``<owner>::<bare_sid>`` → (owner, bare_sid)。无前缀返回 (None, sid)。"""
    if CENTRAL_SID_SEP in session_id:
        owner, bare = session_id.split(CENTRAL_SID_SEP, 1)
        return owner, bare
    return None, session_id


def _resolve_central_cot_path(owner: str, bare_sid: str) -> Optional[Path]:
    """根据 owner / bare sid 定位中央落盘文件。"""
    try:
        from services.uplink_receiver import users_root  # type: ignore
    except Exception:
        return None
    p = users_root() / owner / "cot" / f"{bare_sid}_cot.json"
    return p if p.exists() else None


def get_session_cot(session_id: str) -> Optional[Dict]:
    """
    获取指定 session 的完整 CoT 数据。
    优先在本机 cot/ 找；找不到时若 session_id 形如 ``<owner>::<sid>``
    就去中央 ``~/.agent-cot-central/users/<owner>/cot/`` 找。

    v0.20.11：本机 cot 命中时检查 ``otel_view.enricher_version``，比当前
    版本低就 lazy re-enrich + 写回磁盘 — 让历史 session 立即吃到关键修复
    （events.jsonl legacy 路径补全、tool/user_input 强制 0/0、per-call
    真值匹配等），无需用户手动重跑 hook。central 来源不动（原始落盘者
    那侧负责 enrich 版本，本地不擅自重写）。
    """
    owner, bare_sid = _split_central_sid(session_id)
    if owner:
        path = _resolve_central_cot_path(owner, bare_sid)
        if path:
            return _read_json(path)
        return None

    if session_id.startswith("codex-"):
        _maybe_refresh_codex_cot(session_id)

    # AGENT_COT_PIPELINE_LOG_INJECT_v1 — try every scan dir.
    for _scan_dir in COT_SCAN_DIRS:
        cot_file = _scan_dir / f"{session_id}_cot.json"
        if cot_file.exists():
            data = _read_json(cot_file)
            if data is not None:
                data = _maybe_lazy_re_enrich(data, cot_file)
                return _merge_subagent_timeline(data, session_id)
    return None


# v0.20.11: enricher 行为版本号 — 跟 cot_otel_enricher.py 里 otel_view
# 写入的 ``enricher_version`` 同步。低于这个值的 cot.json 在被 GET 时
# 会触发 lazy re-enrich + 落盘。每次 enricher 行为变化（路径补全 /
# 真值匹配规则 / 默认填值口径）都要把这个数字 +1。
# v7（v0.20.11 二次修正）：Cursor / CodeBuddy 改成 turn-level 真值累加 +
# char 比例分摊；删掉 char/4 启发式兜底。
# v8（v1.0.1）：thinking step 的 in_chars fallback 从 1 升级到
# len(step.content)，避免 Cursor/CodeBuddy thinking 步骤显示一片相同
# input 数（如 941）的视觉伪估算。
# v9（v1.0.2）：Cursor/CodeBuddy 的 input 分摊改为「均值 ± 20% 阻尼」，
# 模拟 Claude Anthropic transcript 真值实测的 max/min ≈ 1.20x 模式。
# 1.0.1 的 full content-weight 让 thinking 单步从 9K 飙到 750K 量级
# （80x 极差），跟 CC 的"近似一致"形态完全不像；1.0.2 后 turn 内
# input 序列收敛到 1.5x 内、output 仍按内容长度分摊（output 是
# per-step 真信号，不阻尼）。Claude 真值路径
# （transcript_per_message）完全不走分摊，行为不变。
# v10（v1.0.3）：Cursor/CodeBuddy **完全停止 per-step token 显示** ——
# step 级 input/output 不再分摊，统一显示 0/0 + source=missing_turn_real，
# 前端不画 step 级 token chip。理由：hook 只暴露 turn 级真值，分摊出来
# 的 step 数字本质都是估算，会让用户怀疑数据可信度。session/turn 总额
# 仍是 hook 真值（不动）。Claude 走 transcript_per_message 真值通道，
# step 级真值显示完全保留（这是 Anthropic SDK 字段级真值）。
_CURRENT_ENRICHER_VERSION = 11


def _maybe_lazy_re_enrich(data: Dict, cot_file: Path) -> Dict:
    """检查 cot.json 的 enricher_version，必要时重跑 enrich 并写回。

    Lazy 策略避免阻塞首次 / SessionList 拉取：
      * 失败任何一步都返回原 data（保证可用性 > 修复及时性）
      * 写回失败也不抛错（磁盘只读 / 权限问题不影响响应）
      * 一次性升级：写回后该 cot.json 后续访问不再重跑

    重跑代价：单 session 通常 < 100ms（in-memory 操作；不重读 transcript
    会再读一次 events.jsonl / index.json，但都已经在磁盘 cache）。
    """
    try:
        ov = data.get("otel_view") if isinstance(data, dict) else None
        if isinstance(ov, dict):
            cur = int(ov.get("enricher_version") or 0)
            if cur >= _CURRENT_ENRICHER_VERSION:
                return data  # 已经是新版本
        # 走重 enrich
        return _re_enrich_cot_inplace(data, cot_file)
    except Exception:
        return data


def _re_enrich_cot_inplace(data: Dict, cot_file: Path) -> Dict:
    """重新跑 enrich：把 dict 反序列化回 SessionCoT-like，调 enricher，
    然后把结果写回 cot.json。

    实现路径：直接修改 dict 上 ``otel_view`` 字段需要重跑完整 enrich，
    跨模块复杂。这里采用最小侵入方案 ——
    通过 ``cot_extractor.SessionCoT.from_dict`` 反序列化 + ``enrich_session_with_otel``
    + 重新 ``to_dict``。两个函数都是项目自有 API，无外部依赖。

    若反序列化或 enrich 抛错（schema 不兼容 / 老格式 cot.json），
    返回原 data，让用户至少看到老视图，不阻塞页面。
    """
    try:
        # 直接 import enricher 和 extractor 的 dataclass
        # （sys.path 已在文件顶部插入，可以直接 from cot_extractor import ...）
        try:
            import sys as _sys
            from pathlib import Path as _Path
            _ext_src = _Path(__file__).resolve().parent.parent.parent / "cot-extractor" / "src"
            if str(_ext_src) not in _sys.path:
                _sys.path.insert(0, str(_ext_src))
            from cot_extractor import SessionCoT  # type: ignore
            from cot_otel_enricher import enrich_session_with_otel  # type: ignore
        except Exception:
            return data

        if not hasattr(SessionCoT, "from_dict") or not hasattr(SessionCoT, "to_dict"):
            return data

        session_cot = SessionCoT.from_dict(data)
        enrich_session_with_otel(session_cot)
        new_data = session_cot.to_dict()

        # 写回（best-effort）
        try:
            with open(cot_file, "w", encoding="utf-8") as f:
                json.dump(new_data, f, ensure_ascii=False)
        except OSError:
            pass

        return new_data
    except Exception:
        return data


def get_session_response_report(session_id: str) -> Optional[Dict]:
    """获取指定 session 的 Response 准确度报告（central session 不带 report）"""
    owner, _ = _split_central_sid(session_id)
    if owner:
        return None
    report_file = RESPONSE_REPORTS_DIR / f"{session_id}_report.json"
    return _read_json(report_file)


def get_session_transcript(session_id: str) -> Optional[Dict]:
    """获取指定 session 的 transcript 数据（central session 不带 transcript）"""
    owner, _ = _split_central_sid(session_id)
    if owner:
        return None
    transcript_file = TRANSCRIPTS_DIR / f"{session_id}_full.json"
    return _read_json(transcript_file)


def get_session_langfuse_cache(session_id: str) -> Optional[Dict]:
    """获取指定 session 的 Langfuse 缓存数据（central session 不带）"""
    owner, _ = _split_central_sid(session_id)
    if owner:
        return None
    cache_file = LANGFUSE_CACHE_DIR / f"{session_id}.json"
    return _read_json(cache_file)


def delete_session(session_id: str) -> bool:
    """
    删除指定 session 的所有相关文件。
    支持本机 + central 两种来源；central 删除只动 ~/.agent-cot-central 那一份。
    返回 True 表示至少删除了一个文件。
    """
    owner, bare_sid = _split_central_sid(session_id)
    if owner:
        path = _resolve_central_cot_path(owner, bare_sid)
        if path:
            try:
                path.unlink()
                return True
            except Exception:
                pass
        return False

    deleted = False
    files_to_delete = [
        COT_DIR / f"{session_id}_cot.json",
        COT_REPORTS_DIR / f"{session_id}_cot.json",
        COT_REPORTS_DIR / f"{session_id}_cot.md",
        RESPONSE_REPORTS_DIR / f"{session_id}_report.json",
        RESPONSE_REPORTS_DIR / f"{session_id}_report.md",
        TRANSCRIPTS_DIR / f"{session_id}_full.json",
        LANGFUSE_CACHE_DIR / f"{session_id}.json",
    ]
    for f in files_to_delete:
        try:
            if f.exists():
                f.unlink()
                deleted = True
        except Exception:
            pass
    return deleted
