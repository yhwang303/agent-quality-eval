"""离线读取 Claude 原生 OTel 事件。

后端进程有 ``services.claude_otel_receiver.load_session_otel()`` 可用，但 CLI
不在 backend 的 sys.path 上。这里提供一个最小读取器，只取 ``flatten_session``
真正需要的东西——事件列表——让 CLI 导出和后端导出拿到同一批 subagent 内部
工具调用，不会出现「UI 里有、命令行导出没有」的差异。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Claude Code 的 OTLP 接收器落盘根目录，与 backend 的 OTEL_ROOT 保持一致
_OTEL_ROOT = Path.home() / ".claude" / "state" / "otel"

# 同一个会话在不同 provider 下会带前缀落盘
_PROVIDER_PREFIXES = ("claude-", "codex-", "codebuddy-")

_MAX_EVENTS = 5000


def _session_dir(session_id: str) -> Optional[Path]:
    root = Path(os.environ.get("AGENT_COT_OTEL_ROOT") or _OTEL_ROOT)
    direct = root / session_id
    if direct.is_dir():
        return direct
    for prefix in _PROVIDER_PREFIXES:
        candidate = root / f"{prefix}{session_id}"
        if candidate.is_dir():
            return candidate
    return None


def load_otel_events(session_id: str) -> Optional[Dict[str, Any]]:
    """返回 ``{"session_id", "events"}``，没有 OTel 数据时返回 None。

    读不动就当没有——OTel 是锦上添花的补充数据源，不能让它的缺失阻断导出。
    """
    if not session_id:
        return None
    sess_dir = _session_dir(session_id)
    if sess_dir is None:
        return None
    target = sess_dir / "events.jsonl"
    if not target.is_file():
        return None

    events: List[Dict[str, Any]] = []
    try:
        with open(target, "r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if i >= _MAX_EVENTS:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return None
    if not events:
        return None
    events.sort(key=lambda r: str(r.get("ts") or "9999"))
    return {"session_id": session_id, "events": events}
