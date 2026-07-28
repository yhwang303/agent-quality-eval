#!/usr/bin/env python3
"""
CoT Trace Exporter — 把拍平后的会话事件流渲染成 jsonl / json / md。

三种格式都从 ``flattener.flatten_session()`` 的同一份输出渲染，
所以顺序天然一致，不存在「md 和 jsonl 对不上」的问题。

**格式怎么选**

    jsonl  规范格式。一行一个事件，顺序即行号，可流式读、可按行截断、可 grep。
           长会话动辄几千个事件、单个工具结果就 2KB，下游 agent 能先读骨架
           再按 seq 展开正文，而不是被迫一次吞下整个文件。
    json   整包。就是 jsonl 的数组包装，适合一次性 load 进程序做统计。
    md     人和 LLM 直接读的视图。有损（长内容会截断），不要拿它做程序化
           diff 或回归比对——那是 jsonl 的活。

**会话隔离**：每种格式的产物都带 ``session_id``（jsonl 甚至每行都带），文件名
也以 sanitize 后的 session_id 命名。调用方必须自己保证传进来的 cot 就是目标
会话的——本模块不做任何跨会话查找或兜底。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from agent_cot.trace.flattener import TRACE_SCHEMA, flatten_session

SUPPORTED_FORMATS = ("jsonl", "json", "md")

_MEDIA_TYPES = {
    "jsonl": "application/x-ndjson; charset=utf-8",
    "json": "application/json; charset=utf-8",
    "md": "text/markdown; charset=utf-8",
}

# md 视图里各类内容的截断长度。jsonl / json 不截断（除了提取阶段就已经截掉的）。
_MD_LIMITS = {
    "thinking": 1200,
    "content": 600,
    "payload": 400,
}

_EVENT_ICONS = {
    "turn_start": "💬",
    "user_input": "💬",
    "tool_result_input": "📥",
    "thinking": "🧠",
    "pre_tool_reasoning": "💡",
    "tool_call": "🔧",
    "tool_result": "⚡",
    "strategy_shift": "🔄",
    "error_recovery": "🚨",
    "final_response": "✅",
    "subagent": "🧬",
    "permission": "🔐",
    "notification": "🔔",
    "compact": "🗜️",
    "environment": "⚙️",
}


def sanitize_session_id(session_id: str) -> str:
    """把 session_id 变成安全的文件名片段。

    中央上行模式下 session_id 形如 ``{owner}::{sid}``，``::`` 在 Windows 文件名
    里非法；此外也要防住路径穿越（``../``）。
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", session_id or "")
    cleaned = cleaned.strip("-.") or "session"
    return cleaned[:120]


# ════════════════════════════════════════════════════════════
#  渲染
# ════════════════════════════════════════════════════════════

def render_jsonl(flat: Dict[str, Any]) -> str:
    """第一行是 session header，其后一行一个事件（JSONL 惯例）。"""
    lines = [json.dumps(flat["header"], ensure_ascii=False)]
    for event in flat["events"]:
        lines.append(json.dumps(event, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def render_json(flat: Dict[str, Any]) -> str:
    return json.dumps(flat, ensure_ascii=False, indent=2) + "\n"


def _truncate(text: str, limit: int) -> str:
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"…（共 {len(text)} 字符，md 视图已截断）"


def render_markdown(flat: Dict[str, Any]) -> str:
    header = flat["header"]
    events: List[Dict[str, Any]] = flat["events"]
    out: List[str] = [
        f"# Agent Trace — `{header.get('session_id')}`",
        "",
        f"- **Agent**: {header.get('agent_type') or 'unknown'}",
        f"- **Turns**: {header.get('turns')} · **Events**: {header.get('total_events')}",
        f"- **Extracted**: {header.get('extracted_at')}",
        f"- **Exported**: {header.get('exported_at')}",
        f"- **Schema**: `{header.get('schema')}`",
    ]
    if header.get("truncated_events"):
        out.append(
            f"- ⚠️ **{header['truncated_events']} 条工具结果在采集阶段即被截断**，"
            "本文件无法还原完整输出"
        )
    out += ["", "---", ""]

    for event in events:
        if event.get("type") == "turn_start":
            out += [
                "",
                f"## Turn {event.get('turn')} — {event.get('title') or '（无标题）'}",
                "",
            ]
            if event.get("content"):
                out += ["> " + _truncate(event["content"], _MD_LIMITS["content"]).replace("\n", "\n> "), ""]
            continue
        out += _render_md_event(event)
    return "\n".join(out) + "\n"


def _render_md_event(event: Dict[str, Any]) -> List[str]:
    etype = str(event.get("type"))
    icon = _EVENT_ICONS.get(etype, "•")
    indent = "  " if event.get("role") == "child" else ""
    parts = [f"{indent}**#{event.get('seq')}** {icon} `{etype}`"]
    if event.get("tool"):
        parts.append(f"→ **{event['tool']}**")
    if event.get("is_error"):
        parts.append("⚠️ **error**")
    if event.get("duration_ms"):
        parts.append(f"· {int(event['duration_ms'])}ms")
    if event.get("in_ui") is False:
        parts.append("· _(UI 未展示)_")
    lines = [" ".join(parts), ""]

    if etype == "thinking":
        lines += ["```", _truncate(event.get("content", ""), _MD_LIMITS["thinking"]), "```", ""]
    elif etype == "tool_call" and event.get("payload") is not None:
        payload = json.dumps(event["payload"], ensure_ascii=False, indent=2) \
            if not isinstance(event["payload"], str) else event["payload"]
        lines += ["```json", _truncate(payload, _MD_LIMITS["payload"]), "```", ""]
    elif etype in ("subagent", "permission", "notification", "compact", "environment"):
        payload = json.dumps(event.get("payload") or {}, ensure_ascii=False)
        lines += [f"{indent}> {_truncate(payload, _MD_LIMITS['payload'])}", ""]
    elif event.get("content"):
        body = _truncate(event["content"], _MD_LIMITS["content"])
        lines += [f"{indent}> " + body.replace("\n", f"\n{indent}> "), ""]

    if event.get("truncated"):
        lines += [f"{indent}> *(原始结果 {event.get('original_len')} 字符，采集时已截断)*", ""]
    if event.get("plan"):
        plan = event["plan"]
        lines += [
            f"{indent}> 🗺️ plan {plan.get('completed', 0)}/{plan.get('total', 0)} 完成",
            "",
        ]
    return lines


_RENDERERS = {
    "jsonl": render_jsonl,
    "json": render_json,
    "md": render_markdown,
}


# ════════════════════════════════════════════════════════════
#  公开 API
# ════════════════════════════════════════════════════════════

def export_turn_trace(
    cot: Dict[str, Any],
    turn_index: int,
    otel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """只导出一轮交互（一次「用户提问 → agent 答完」）的事件流。

    这是界面上每张交互卡片旁那个导出按钮走的路径。整个 session 的导出留给
    CLI 和 HTTP 接口——闭环 harness 常常要整条会话，但人看 trace 时想要的
    几乎总是「这一轮到底干了什么」。

    为什么先拍平整个 session 再筛这一轮，而不是只拍平这一轮：事件顺序依赖
    跨 turn 的时间线归属（并行时间线事件挂哪一轮由窗口决定），OTel 补齐的
    subagent 工具调用也是 session 级的输入。先整体拍平再筛，得到的顺序与
    界面上看到的完全一致；反过来单独拍一轮则会丢掉这些关联。

    ``seq`` 会重新从 1 编号，这样导出的文件自己就是一条完整可读的流；原始的
    session 级序号保留在 ``session_seq`` 里，方便回到整条会话里定位。

    只出 jsonl：一轮的量级本来就适合一行一个事件，再给三种格式只会让下游多
    一层「该信哪个」的判断。
    """
    if turn_index is None:
        raise ValueError("turn_index 不能为空")
    flat = flatten_session(cot, otel=otel)
    session_id = str(cot.get("session_id") or "")

    events = [e for e in flat["events"] if e.get("turn") == turn_index]
    if not events:
        raise ValueError(
            f"会话 {session_id!r} 里没有 turn {turn_index} 的事件"
        )

    scoped: List[Dict[str, Any]] = []
    for seq, event in enumerate(events, start=1):
        item = dict(event)
        item["session_seq"] = item.get("seq")
        item["seq"] = seq
        scoped.append(item)

    turn_meta = next(
        (t for t in (cot.get("turns") or []) if t.get("turn_index") == turn_index),
        {},
    )
    header = dict(flat["header"])
    header.update({
        "type": "turn_header",
        "scope": "turn",
        "turn_index": turn_index,
        "user_query": turn_meta.get("user_query") or "",
        "interaction_summary": turn_meta.get("interaction_summary") or "",
        "total_events": len(scoped),
        "ordered_events": sum(1 for e in scoped if e.get("ordered") is not False),
        "truncated_events": sum(1 for e in scoped if e.get("truncated")),
        # 整条会话的规模一并带上，免得只看单轮文件会误判「这就是全部」
        "session_total_events": flat["header"].get("total_events"),
        "session_turns": flat["header"].get("turns"),
    })
    header.pop("unordered_events", None)

    turn_flat = {"schema": TRACE_SCHEMA, "header": header, "events": scoped}
    return {
        "content": render_jsonl(turn_flat),
        "media_type": _MEDIA_TYPES["jsonl"],
        "filename": (
            f"trace-{sanitize_session_id(session_id)}-turn{turn_index}.jsonl"
        ),
        "session_id": session_id,
        "turn_index": turn_index,
        "event_count": len(scoped),
        "schema": TRACE_SCHEMA,
    }


def export_session_trace(
    cot: Dict[str, Any],
    fmt: str = "jsonl",
    otel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """把单个会话的 cot.json 导出成指定格式。

    Args:
        cot: 目标会话的 ``{sid}_cot.json`` 内容。
        fmt: ``jsonl`` / ``json`` / ``md``。
        otel: 可选的 Claude 原生 OTel 数据，用于补齐 subagent 内部工具调用。

    Returns:
        ``{"content", "media_type", "filename", "session_id", "event_count"}``
    """
    fmt = (fmt or "jsonl").lower().lstrip(".")
    if fmt not in _RENDERERS:
        raise ValueError(
            f"不支持的导出格式 {fmt!r}，可选：{', '.join(SUPPORTED_FORMATS)}"
        )
    flat = flatten_session(cot, otel=otel)
    session_id = str(cot.get("session_id") or "")
    return {
        "content": _RENDERERS[fmt](flat),
        "media_type": _MEDIA_TYPES[fmt],
        "filename": f"trace-{sanitize_session_id(session_id)}.{fmt}",
        "session_id": session_id,
        "event_count": len(flat["events"]),
        "schema": TRACE_SCHEMA,
    }
