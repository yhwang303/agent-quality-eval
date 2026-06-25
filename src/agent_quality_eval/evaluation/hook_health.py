"""Read-only IDE hook health checks for the dashboard."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHECKED_AGENTS = ("codex", "claude", "cursor", "codebuddy")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _runtime_status() -> dict[str, Any]:
    path = Path.home() / ".agent-cot" / "runtime.json"
    data = _read_json(path) if path.is_file() else None
    python_executable = str((data or {}).get("python_executable") or "")
    cot_extractor_root = str((data or {}).get("cot_extractor_root") or "")
    data_root = str((data or {}).get("data_root") or "")
    return {
        "path": str(path),
        "exists": path.is_file(),
        "python_executable": python_executable,
        "python_exists": bool(python_executable and Path(python_executable).exists()),
        "cot_extractor_root": cot_extractor_root,
        "cot_extractor_exists": bool(cot_extractor_root and Path(cot_extractor_root).exists()),
        "data_root": data_root,
        "data_root_exists": bool(data_root and Path(data_root).exists()),
    }


def _event_name_from_line(line: str) -> str:
    match = re.search(r"\bevent=([^\s]+)", line)
    if match:
        return match.group(1)
    try:
        data = json.loads(line)
        if isinstance(data, dict):
            return str(data.get("event") or "")
    except Exception:
        pass
    return ""


def _timestamp_from_line(line: str) -> str:
    match = re.match(r"\[([^\]]+)\]", line)
    if match:
        return match.group(1)
    try:
        data = json.loads(line)
        if isinstance(data, dict):
            return str(data.get("ts") or "")
    except Exception:
        pass
    return ""


def _line_matches_agent(line: str, agent_name: str) -> bool:
    lower = line.lower()
    if agent_name == "claude":
        return "[claude]" in lower or "claude-stream" in lower or "agent_type\": \"claude" in lower
    if agent_name == "cursor":
        return "[cursor]" in lower or "cursor-stream" in lower or "agent_type\": \"cursor" in lower
    if agent_name == "codebuddy":
        return "[codebuddy]" in lower or "codebuddy-stream" in lower or "agent_type\": \"codebuddy" in lower
    if agent_name == "codex":
        return "[codex]" in lower or "codex-stream" in lower or "agent_type\": \"codex" in lower
    return agent_name in lower


def _recent_activity(agent_name: str) -> dict[str, Any]:
    paths = [
        Path.home() / ".agent-cot" / "logs" / "pipeline.log",
        Path.home() / ".agent-cot" / "logs" / "critic-runner.log",
    ]
    matches: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-1500:]
        except Exception:
            continue
        for line in reversed(lines):
            if not _line_matches_agent(line, agent_name):
                continue
            matches.append(
                {
                    "path": str(path),
                    "ts": _timestamp_from_line(line),
                    "event": _event_name_from_line(line),
                    "line": line[:600],
                }
            )
            break
    return {
        "available": bool(matches),
        "latest": matches[0] if matches else None,
        "matches": matches,
    }


def _script_name(command: str) -> str:
    match = re.search(r"([A-Za-z0-9_.-]+\.(?:py|js))", command)
    return match.group(1) if match else command


def _target_entries(adapter: Any, assets_dir: Path) -> list[Any]:
    fn = getattr(adapter, "hook_entries_for_assets_dir", None)
    if callable(fn):
        return list(fn(assets_dir))
    return list(adapter.hook_entries())


def _target_status(adapter: Any, config_path: Path, assets_dir: Path) -> dict[str, Any]:
    entries = _target_entries(adapter, assets_dir)
    bridge_files = list(adapter.bridge_files())
    config = _read_json(config_path) if config_path.is_file() else None
    config_text = json.dumps(config or {}, ensure_ascii=False)
    missing_entries = []
    for entry in entries:
        if entry.event not in config_text or _script_name(entry.command) not in config_text:
            missing_entries.append({"event": entry.event, "script": _script_name(entry.command)})
    missing_assets = [name for name in bridge_files if not (assets_dir / name).is_file()]
    config_ok = config_path.is_file() and isinstance(config, dict)
    entries_ok = config_ok and not missing_entries
    assets_ok = not missing_assets
    return {
        "config_path": str(config_path),
        "assets_dir": str(assets_dir),
        "config_exists": config_path.is_file(),
        "config_valid_json": isinstance(config, dict),
        "assets_written": assets_ok,
        "missing_assets": missing_assets,
        "expected_entry_count": len(entries),
        "missing_entry_count": len(missing_entries),
        "missing_entries": missing_entries[:12],
        "entries_active": entries_ok,
        "status": "ok" if config_ok and entries_ok and assets_ok else "warning",
    }


def _agent_health(agent_name: str, runtime: dict[str, Any]) -> dict[str, Any]:
    from agent_cot.agents import get_adapter

    adapter = get_adapter(agent_name)
    try:
        installed = bool(adapter.detect_installed())
    except Exception as exc:
        return {
            "agent": agent_name,
            "display_name": getattr(adapter, "display_name", agent_name),
            "installed": False,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not installed:
        return {
            "agent": agent_name,
            "display_name": getattr(adapter, "display_name", agent_name),
            "installed": False,
            "status": "skipped",
            "reason": "IDE not detected on this machine.",
        }

    targets: list[dict[str, Any]] = []
    primary_config = Path(adapter.hooks_config_path())
    primary_assets = Path(adapter.hooks_assets_dir())
    targets.append(_target_status(adapter, primary_config, primary_assets))
    try:
        additional = list(adapter.additional_hooks_targets())
    except Exception:
        additional = []
    for config_path, assets_dir in additional:
        targets.append(_target_status(adapter, Path(config_path), Path(assets_dir)))

    runtime_ok = bool(runtime.get("exists") and runtime.get("python_exists") and runtime.get("cot_extractor_exists"))
    activated = all(t.get("entries_active") and t.get("assets_written") for t in targets)
    status = "ok" if activated and runtime_ok else "warning"
    return {
        "agent": agent_name,
        "display_name": getattr(adapter, "display_name", agent_name),
        "installed": True,
        "status": status,
        "activated": activated,
        "runtime_ok": runtime_ok,
        "targets": targets,
        "recent_activity": _recent_activity(agent_name),
    }


def build_hook_health_report() -> dict[str, Any]:
    runtime = _runtime_status()
    agents = []
    for agent_name in CHECKED_AGENTS:
        try:
            agents.append(_agent_health(agent_name, runtime))
        except Exception as exc:
            agents.append(
                {
                    "agent": agent_name,
                    "display_name": agent_name,
                    "installed": False,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    installed_agents = [item for item in agents if item.get("installed")]
    failing = [item for item in installed_agents if item.get("status") != "ok"]
    overall = "ok" if installed_agents and not failing else ("warning" if installed_agents else "skipped")
    return {
        "schema_version": "hook-health-v1",
        "generated_at": _utc_now(),
        "runtime": runtime,
        "current_python": sys.executable,
        "pid": os.getpid(),
        "overall_status": overall,
        "agents": agents,
    }

