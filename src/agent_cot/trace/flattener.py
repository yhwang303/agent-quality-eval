#!/usr/bin/env python3
"""
CoT Trace Flattener — 把 SessionCoT 拍平成一条按执行顺序排列的事件流。

**为什么需要它**

``cot.json`` 存的是「一条主步流 + 若干条并行时间线」的分离结构：

    turns[].steps[]      thinking / tool_decision / tool_execution / ...
    permission_events    权限请求
    subagent_timeline    子代理
    notification_events  通知
    compact_events       上下文压缩
    environment_events   环境变更
    user_activity        用户手动操作

它们各自带 ``t_ms``，但「谁在谁前面」这个顺序此前只在前端 ``SpanTree.tsx``
渲染时现算（buildRenderGroups → mergeThinkingPhases → interleaveClaudeEvents）。
后端不存顺序，导致：

  1. trace 导出只能给下游一堆碎片桶，下游得自己重新实现排序（且极易实现错）；
  2. 同一份数据有两套排序实现，UI 和导出会漂移。

本模块把那套排序逻辑下沉为**唯一真值**：前端改成消费它的输出，导出也用它。

**输出契约**

``flatten_session()`` 返回 ``{"schema", "header", "events"}``。``events`` 是
全局有序的扁平列表，每个事件除了内容字段外还带三个「重建 UI 树」用的字段：

    group_id   同一个渲染组（一个 leader + 紧随其后的工具链）共享
    role       "leader" | "child"
    phase_id   连续 thinking 折叠段的编号；不属于折叠段时为 None

前端按 ``group_id`` 一次 reduce 就能还原出原来的 RenderGroup / ThinkingPhase
结构，不损失任何折叠体验。导出侧则直接忽略这三个字段，按顺序读即可。

**顺序的确切含义**：``seq`` 就是树上从上往下的阅读顺序。它基本等同于时间顺序，
但有一个刻意的例外——工具并发时，结果会被提到它自己那次调用的正后方，于是
``t_ms`` 会回跳。transcript 里的 ``D1 D2 E1 E2`` 会被排成 ``D1 E1 D2 E2``。
下游**不要**按 ``t_ms`` 重排，那会把调用和结果拆散；要还原物理并发时序，
按 ``t_ms`` 单独排一份即可，两种视图并不冲突。

**保真度上限**：``cot_extractor`` 在提取阶段就把工具结果截断到 2000 字符
（见 cot_extractor.py 的 ``result_text[:2000]``），本模块无法还原原文，只能在
事件上如实标注 ``truncated`` / ``original_len``，让下游知道这里有信息缺口。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

TRACE_SCHEMA = "agent-cot.trace/v1"

# 最后一轮没有结束信号时的窗口上界。会话尾部的事件都算最后一轮。
_OPEN_END_MS = 1 << 53

# 主步流里被视为「工具链」的 step_type —— 它们挂在前一个 leader 之下，
# 与前端 SpanTree.buildRenderGroups 的 isToolish 保持一致。
_TOOLISH_STEP_TYPES = frozenset({
    "tool_decision",
    "tool_execution",
    "strategy_shift",
    "error_recovery",
})

# thinking 的三种历史写法（不同 IDE / 不同版本的 extractor 产出过不同值）
_THINKING_STEP_TYPES = frozenset({
    "thinking_explicit",
    "thinking_inter",
    "thinking_intermediate",
})

# step_type → 导出事件类型。导出用中性命名（tool_call / tool_result），
# 原始 step_type 仍原样保留在事件的 ``step_type`` 字段里，不丢信息。
_STEP_TYPE_TO_EVENT_TYPE = {
    "user_input": "user_input",
    "tool_result_input": "tool_result_input",
    "thinking_explicit": "thinking",
    "thinking_inter": "thinking",
    "thinking_intermediate": "thinking",
    "pre_tool_reasoning": "pre_tool_reasoning",
    "tool_decision": "tool_call",
    "tool_execution": "tool_result",
    "strategy_shift": "strategy_shift",
    "error_recovery": "error_recovery",
    "final_response": "final_response",
}

# 并行时间线 → 事件类型。key 是 SessionCoT 上的字段名。
_TIMELINE_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("subagent_timeline", "subagent"),
    ("permission_events", "permission"),
    ("notification_events", "notification"),
    ("compact_events", "compact"),
    ("environment_events", "environment"),
)

# 带 plan 语义的工具：它们的 metadata 上被后端回灌了 plan_* 字段。
_PLAN_TOOLS = frozenset({"TodoWrite", "TaskCreate", "TaskUpdate"})


# ════════════════════════════════════════════════════════════
#  时间处理
# ════════════════════════════════════════════════════════════

def _parse_iso_ms(value: Any) -> int:
    """ISO 时间戳 → epoch 毫秒；解析不了返回 0。"""
    if not isinstance(value, str) or not value:
        return 0
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_iso(ms: Any) -> Optional[str]:
    if not isinstance(ms, (int, float)) or ms <= 0:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _step_ts_ms(step: Dict[str, Any]) -> int:
    """取 step 的真实发生时刻。

    优先 ``metadata.observed_at_ms``（hook 真值）而不是 ``timestamp``：
    tool_execution 的 timestamp 往往是它所在 user message 的时间，可能比实际
    执行晚很多，直接拿来排序会把工具结果甩到时间线末尾。
    """
    md = step.get("metadata") or {}
    for key in ("observed_at_ms", "_t_ms"):
        val = md.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
    return _parse_iso_ms(step.get("timestamp"))


def _event_ts_ms(event: Dict[str, Any]) -> int:
    for key in ("t_ms", "t", "observed_at_ms"):
        val = event.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
    return _parse_iso_ms(event.get("ts") or event.get("timestamp"))


def _sort_steps_by_time(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按真实时间排序，缺时间戳的步骤保持贴在原邻居旁边。

    前端旧实现的比较器在「一边有 ts、一边没有」时回退比 step_index，这在
    JS 里是个不自洽的比较器，结果依赖 TimSort 的归并次序。这里改成显式的
    全序：给缺 ts 的步骤沿用前一个已知 ts（carry-forward），再以 step_index
    做次级键。效果是缺 ts 的步骤留在它原本的邻居旁，而不是被甩到最前面。
    """
    indexed: List[Tuple[int, int, Dict[str, Any]]] = []
    carried = 0
    for step in sorted(steps, key=lambda s: _as_int(s.get("step_index"))):
        ts = _step_ts_ms(step)
        if ts > 0:
            carried = ts
        indexed.append((carried, _as_int(step.get("step_index")), step))
    indexed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in indexed]


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    return default


# ════════════════════════════════════════════════════════════
#  渲染分组（对齐 SpanTree.tsx）
# ════════════════════════════════════════════════════════════

class _Group:
    """一个渲染组：一个 leader + 紧随其后的工具链。

    ``phase`` 为 True 时表示这是「连续 thinking 折叠段」，此时 leader 为 None、
    thoughts 装该段里的全部 thinking 步骤。
    """

    __slots__ = ("leader", "tools", "phase", "thoughts")

    def __init__(self, leader: Optional[Dict[str, Any]] = None, phase: bool = False):
        self.leader = leader
        self.tools: List[Dict[str, Any]] = []
        self.phase = phase
        self.thoughts: List[Dict[str, Any]] = []


def _pair_tool_decision_execution(
    tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """把 tool_execution 移到它对应的 tool_decision 后面。

    模型一次回复里并发触发多个 tool_use 时，transcript 的真实顺序是
    D1 D2 E1 E2（assistant 一条消息多个 tool_use，user 一条消息多个
    tool_result）。原样输出会变成「决策堆一起、结果堆一起」，读 trace 的人
    和 agent 都很难把结果对回是哪次调用。这里按 tool_use_id 配成 D1 E1 D2 E2。
    """
    if len(tools) < 2:
        return tools

    decision_idx_by_use_id: Dict[str, int] = {}
    for i, step in enumerate(tools):
        if step.get("step_type") != "tool_decision":
            continue
        use_id = _tool_use_id(step)
        if use_id and use_id not in decision_idx_by_use_id:
            decision_idx_by_use_id[use_id] = i
    if not decision_idx_by_use_id:
        return tools

    inline_after: Dict[int, Dict[str, Any]] = {}
    consumed: set = set()
    for i, step in enumerate(tools):
        if step.get("step_type") != "tool_execution":
            continue
        use_id = _tool_use_id(step)
        if not use_id:
            continue
        dec_idx = decision_idx_by_use_id.get(use_id)
        # 同一个 decision 只吸附一个 execution，多余的保持原位
        if dec_idx is None or dec_idx in inline_after:
            continue
        inline_after[dec_idx] = step
        consumed.add(i)
    if not inline_after:
        return tools

    out: List[Dict[str, Any]] = []
    for i, step in enumerate(tools):
        if i in consumed:
            continue
        out.append(step)
        if step.get("step_type") == "tool_decision" and i in inline_after:
            out.append(inline_after[i])
    return out


def _tool_use_id(step: Dict[str, Any]) -> str:
    val = step.get("tool_use_id")
    if isinstance(val, str) and val:
        return val
    md = step.get("metadata") or {}
    val = md.get("tool_use_id")
    return val if isinstance(val, str) else ""


def _tool_name(step: Dict[str, Any]) -> str:
    val = step.get("tool_name")
    if isinstance(val, str) and val:
        return val
    md = step.get("metadata") or {}
    val = md.get("tool_name")
    return val if isinstance(val, str) else ""


def _build_render_groups(steps: List[Dict[str, Any]]) -> List[_Group]:
    groups: List[_Group] = []
    current: Optional[_Group] = None
    for step in _sort_steps_by_time(steps):
        if step.get("step_type") in _TOOLISH_STEP_TYPES:
            if current is None:
                # 没有前导 thinking，开一个「隐式」组
                current = _Group(leader=None)
                groups.append(current)
            current.tools.append(step)
        else:
            current = _Group(leader=step)
            groups.append(current)
    for group in groups:
        group.tools = _pair_tool_decision_execution(group.tools)
    return groups


def _should_treat_pre_tool_as_thinking(agent_type: str, steps: List[Dict[str, Any]]) -> bool:
    """CodeBuddy 恒为真；Claude 只在没开 extended thinking 时为真。

    这两家的 ``pre_tool_reasoning`` 装的是模型在调工具前的自然语言推理，
    语义上就是思考。Cursor 的同名步骤是简短的决策说明，不参与折叠。
    """
    if agent_type == "codebuddy":
        return True
    if agent_type != "claude":
        return False
    return not any(s.get("step_type") in _THINKING_STEP_TYPES for s in steps)


def _merge_thinking_phases(
    groups: List[_Group],
    treat_pre_tool_as_thinking: bool,
) -> List[_Group]:
    """把连续 ≥2 个「只有 thinking、没跟工具」的组折叠成一个 phase。

    带工具的 thinking 保持原样——那种思考是紧接着的工具决策的理由说明，
    拆开就看不出因果了。
    """
    out: List[_Group] = []
    buf: List[Dict[str, Any]] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        if len(buf) == 1:
            out.append(_Group(leader=buf[0]))
        else:
            phase = _Group(phase=True)
            phase.thoughts = buf
            out.append(phase)
        buf = []

    for group in groups:
        leader = group.leader
        standalone_thinking = (
            leader is not None
            and not group.tools
            and (
                leader.get("step_type") == "thinking_explicit"
                or (
                    treat_pre_tool_as_thinking
                    and leader.get("step_type") == "pre_tool_reasoning"
                )
            )
        )
        if standalone_thinking and leader is not None:
            buf.append(leader)
        else:
            flush()
            out.append(group)
    flush()
    return out


def _group_start_ts(group: _Group) -> int:
    if group.phase:
        return _step_ts_ms(group.thoughts[0]) if group.thoughts else 0
    if group.leader is not None:
        ts = _step_ts_ms(group.leader)
        if ts > 0:
            return ts
    for tool in group.tools:
        ts = _step_ts_ms(tool)
        if ts > 0:
            return ts
    return 0


def _interleave_timeline_events(
    groups: List[_Group],
    events: List[Dict[str, Any]],
) -> List[_Group]:
    """把并行时间线事件按 t_ms 插回主步流。

    落点规则：找 startTs ≤ 事件时刻的最靠右的非 phase 组，再在该组的工具链里
    按时间插到合适位置。phase 内部不插事件——hook 事件基本都伴随工具调用，
    落进纯思考段的概率极低，落在两个 phase 之间时挂到前一个普通组。
    """
    if not events:
        return groups
    start_ts = [_group_start_ts(g) for g in groups]
    for event in sorted(events, key=lambda e: e["_t_ms"]):
        t_ms = event["_t_ms"]
        target = -1
        for i, group in enumerate(groups):
            ts = start_ts[i]
            if ts <= 0:
                continue
            if ts > t_ms:
                break
            if not group.phase:
                target = i
        if target < 0:
            for i, group in enumerate(groups):
                if not group.phase:
                    target = i
                    break
        if target < 0:
            continue
        tools = groups[target].tools
        ins_idx = len(tools)
        for i, tool in enumerate(tools):
            tool_ts = _step_ts_ms(tool)
            if tool_ts > 0 and tool_ts > t_ms:
                ins_idx = i
                break
        tools.insert(ins_idx, event)
    return groups


# ════════════════════════════════════════════════════════════
#  OTel 孤儿工具调用补齐
# ════════════════════════════════════════════════════════════

def _augment_with_otel_orphans(
    turns: List[Dict[str, Any]],
    otel: Optional[Dict[str, Any]],
) -> None:
    """把只出现在 Claude 原生 OTel 通道里的工具调用补进对应 turn 的 steps。

    Claude 的 subagent 在自己的 sidechain 里调用工具，这些调用不会写进主
    transcript，只有 OTel events 通道能看到。前端此前在 SpanTree 里现场
    合并（augmentCotWithOtelOrphans），后端 cot.json 里没有——不补的话导出
    会整块丢掉 subagent 内部的全部工具调用。就地修改 ``turns``。
    """
    if not otel:
        return
    events = otel.get("events") or []
    if not events:
        return

    known_use_ids = {
        _tool_use_id(step)
        for turn in turns
        for step in (turn.get("steps") or [])
        if _tool_use_id(step)
    }

    orphans: Dict[str, Dict[str, Any]] = {}
    for event in events:
        name = event.get("event_name")
        if name not in ("tool_decision", "tool_result"):
            continue
        attrs = event.get("attributes") or {}
        use_id = attrs.get("tool_use_id")
        if not isinstance(use_id, str) or not use_id or use_id in known_use_ids:
            continue
        slot = orphans.setdefault(use_id, {"use_id": use_id})
        slot["decision" if name == "tool_decision" else "result"] = event
    if not orphans:
        return

    windows = _turn_windows(turns)
    virt_seq = 0
    for slot in orphans.values():
        t_dec = _otel_event_ts(slot.get("decision"))
        t_res = _otel_event_ts(slot.get("result"))
        t_ev = t_dec or t_res
        if not t_ev:
            continue
        target = _locate_turn(windows, t_ev)
        if target < 0:
            continue
        turn = turns[target]
        dec_attrs = (slot.get("decision") or {}).get("attributes") or {}
        res_attrs = (slot.get("result") or {}).get("attributes") or {}
        tool_name = dec_attrs.get("tool_name") or res_attrs.get("tool_name") or "tool"
        tool_input = dec_attrs.get("tool_input") or res_attrs.get("tool_input") or ""
        steps = turn.setdefault("steps", [])

        if slot.get("decision"):
            steps.append({
                # 用足够大的 step_index 避免和真步号冲突，同时保持唯一
                "step_index": 1_000_000 + virt_seq,
                "turn_index": turn.get("turn_index", 0),
                "step_type": "tool_decision",
                "content": tool_input if isinstance(tool_input, str) else json.dumps(
                    tool_input, ensure_ascii=False),
                "tool_name": tool_name,
                "tool_use_id": slot["use_id"],
                "metadata": {
                    "observed_at_ms": t_dec or t_ev,
                    "tool_name": tool_name,
                    "tool_use_id": slot["use_id"],
                    "tool_input": tool_input,
                    "_otel_orphan": True,
                },
                "timestamp": dec_attrs.get("event.timestamp") or slot["decision"].get("ts") or "",
            })
            virt_seq += 1
        if slot.get("result"):
            steps.append({
                "step_index": 1_000_000 + virt_seq,
                "turn_index": turn.get("turn_index", 0),
                "step_type": "tool_execution",
                "content": "",
                "tool_name": tool_name,
                "tool_use_id": slot["use_id"],
                "metadata": {
                    "observed_at_ms": t_res or t_ev,
                    "tool_name": tool_name,
                    "tool_use_id": slot["use_id"],
                    "success": res_attrs.get("success") in ("true", True),
                    "error_type": res_attrs.get("error_type"),
                    "_otel_orphan": True,
                },
                "timestamp": res_attrs.get("event.timestamp") or slot["result"].get("ts") or "",
            })
            virt_seq += 1


def _otel_event_ts(event: Optional[Dict[str, Any]]) -> int:
    if not event:
        return 0
    attrs = event.get("attributes") or {}
    return _parse_iso_ms(attrs.get("event.timestamp")) or _parse_iso_ms(event.get("ts"))


def _turn_window(turn: Dict[str, Any]) -> Tuple[int, int]:
    t0 = _as_int(turn.get("turn_start_ms_observed"))
    t1 = _as_int(turn.get("turn_end_ms_observed"))
    if not t0 or not t1:
        stamps = [ts for ts in (_step_ts_ms(s) for s in (turn.get("steps") or [])) if ts > 0]
        if stamps:
            t0 = t0 or min(stamps)
            t1 = t1 or max(stamps)
    return t0, t1


def _turn_windows(turns: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """逐 turn 的时间窗，上界补到下一轮开始之前。

    ``_turn_window`` 在没有 ``turn_end_ms_observed`` 时只能拿最后一个 step 的时间
    当上界，于是「本轮最后一步之后、下一轮开始之前」发生的事（收尾的
    notification、subagent 回收）会掉进两窗之间的缝里。这类事件属于前一轮，
    所以只要该轮没有显式的结束信号，就把上界延到下一轮开始前；最后一轮延到无穷。
    有显式结束时间的轮次保持原样，避免把下一轮的等待期算进上一轮。
    """
    windows = [_turn_window(turn) for turn in turns]
    for i, turn in enumerate(turns):
        t0, t1 = windows[i]
        if not t0:
            continue
        explicit_end = _as_int(turn.get("turn_end_ms_observed"))
        if not explicit_end:
            duration = _as_int(turn.get("turn_duration_ms"))
            if duration:
                explicit_end = t0 + duration
        next_t0 = next((windows[j][0] for j in range(i + 1, len(windows)) if windows[j][0]), 0)
        end = max(t1, explicit_end) if explicit_end else (
            next_t0 - 1 if next_t0 else _OPEN_END_MS
        )
        if next_t0:
            # 窗口之间不重叠，否则一个事件会被归给更早的那一轮。
            # 但本轮 step 自己的时间戳永远算本轮，即使它越过了下一轮的起点。
            end = max(t1, min(end, next_t0 - 1))
        windows[i] = (t0, end)
    return windows


def _locate_turn(windows: List[Tuple[int, int]], t_ms: int) -> int:
    """事件时刻 → turn 下标。落在窗口内取该窗口，落在两窗之间归前一个 turn。"""
    target = -1
    for i, (t0, t1) in enumerate(windows):
        if not t0:
            continue
        if t1 and t0 <= t_ms <= t1:
            return i
        if t_ms >= t0:
            target = i
    if target < 0 and windows:
        # 早于所有 turn（例如 sessionStart 阶段的环境事件）→ 归第一个 turn
        target = 0
    return target


# ════════════════════════════════════════════════════════════
#  事件构造
# ════════════════════════════════════════════════════════════

def _first_sentence(text: str, max_len: int = 100) -> str:
    """抽首句做 title，供下游先读骨架再按需展开正文。"""
    if not text:
        return ""
    cleaned = text.lstrip().replace("\r", "")
    for i, ch in enumerate(cleaned):
        if ch in ".!?。！？\n" and i >= 6:
            cleaned = cleaned[:i]
            break
    cleaned = cleaned.strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip() + "…"
    return cleaned


def _plan_payload(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not any(k.startswith("plan_") for k in metadata):
        return None
    payload = {
        "total": metadata.get("plan_total"),
        "completed": metadata.get("plan_completed_count"),
        "in_progress": metadata.get("plan_in_progress_count"),
        "snapshot_index": metadata.get("plan_snapshot_idx"),
        "todos": metadata.get("plan_full_todos"),
        "diff": metadata.get("plan_diff"),
    }
    return {k: v for k, v in payload.items() if v is not None}


def _step_title(step: Dict[str, Any], event_type: str) -> str:
    tool = _tool_name(step)
    if event_type == "tool_call":
        return f"{tool}" if tool else "tool call"
    if event_type == "tool_result":
        md = step.get("metadata") or {}
        status = "error" if md.get("is_error") or md.get("success") is False else "ok"
        return f"{tool or 'tool'} → {status}"
    return _first_sentence(step.get("content") or "")


def _build_step_event(step: Dict[str, Any]) -> Dict[str, Any]:
    step_type = step.get("step_type") or "unknown"
    event_type = _STEP_TYPE_TO_EVENT_TYPE.get(step_type, step_type)
    metadata = step.get("metadata") or {}
    t_ms = _step_ts_ms(step)

    event: Dict[str, Any] = {
        "type": event_type,
        "step_type": step_type,
        "step_index": step.get("step_index"),
        "t_ms": t_ms or None,
        "ts": _ms_to_iso(t_ms),
        "duration_ms": step.get("duration_ms"),
        "title": _step_title(step, event_type),
        "content": step.get("content") or "",
    }

    tool = _tool_name(step)
    if tool:
        event["tool"] = tool
    use_id = _tool_use_id(step)
    if use_id:
        event["tool_use_id"] = use_id
    if event_type == "tool_call":
        event["payload"] = metadata.get("tool_input")
    if event_type == "tool_result":
        # 提取阶段已经截断过，如实标注缺口，避免下游误以为看到了全量输出
        if metadata.get("truncated"):
            event["truncated"] = True
            event["original_len"] = metadata.get("result_len")
        if metadata.get("is_error") or metadata.get("success") is False:
            event["is_error"] = True
    if metadata.get("_otel_orphan"):
        event["otel_orphan"] = True
    if tool in _PLAN_TOOLS:
        plan = _plan_payload(metadata)
        if plan:
            event["plan"] = plan
    tokens = step.get("tokens")
    if isinstance(tokens, (int, float)) and tokens:
        event["tokens"] = int(tokens)
    otel = step.get("otel") or {}
    usage = otel.get("token_usage") if isinstance(otel, dict) else None
    if usage:
        event["token_usage"] = usage
    return event


def _renders_in_ui(
    kind: str,
    payload: Dict[str, Any],
    agent_type: str,
    t_ms: int,
    window: Tuple[int, int],
) -> bool:
    """当前 SpanTree 是否会把这条并行时间线事件画进树里。

    导出永远是完整的；这个标记只是告诉前端「渲染时跳过哪些」，让「后端定顺序、
    前端照着画」这条规则里所有的例外都集中在这一个函数里，而不是散在 UI 代码。

    四类不渲染的情况：
      * agent 不是 claude / codex —— 其它 IDE 的树上从来没有这些节点
      * environment 事件 —— 属于 IDE 环境层，与 agent 行为不直接相关
      * PermissionMode —— transcript 里的权限档位切换记录，噪声大
      * 早于所属 turn 的起点 —— 只可能是首轮之前的会话初始化事件

    这里只查下界。上界由 turn 归属保证：窗口互不重叠，事件挂到哪一轮，就说明它
    发生在下一轮开始之前。subagent 常常在主线程最后一步之后好几分钟才收尾，
    卡死上界会把这类事件从树上抹掉。
    """
    if agent_type not in ("claude", "codex"):
        return False
    if kind == "environment":
        return False
    if kind == "permission" and payload.get("kind") == "PermissionMode":
        return False
    t0, _ = window
    return bool(t0 and t_ms >= t0)


def _build_timeline_event(kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    t_ms = _event_ts_ms(payload)
    title = (
        payload.get("kind")
        or payload.get("mode")
        or payload.get("message")
        or payload.get("tool_name")
        or kind
    )
    event: Dict[str, Any] = {
        "type": kind,
        "t_ms": t_ms or None,
        "ts": _ms_to_iso(t_ms),
        "title": _first_sentence(str(title)) if title else kind,
        "payload": payload,
    }
    duration = payload.get("duration_ms")
    if isinstance(duration, (int, float)):
        event["duration_ms"] = duration
    return event


# ════════════════════════════════════════════════════════════
#  公开 API
# ════════════════════════════════════════════════════════════

def flatten_session(
    cot: Dict[str, Any],
    otel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """把一个 session 的 cot.json 拍平成有序事件流。

    Args:
        cot: ``SessionCoT.to_dict()`` 的产物（即 ``{sid}_cot.json`` 的内容）。
        otel: 可选，``load_session_otel(sid)`` 的产物。只对 Claude 会话有意义，
            用于补齐 subagent 内部那些主 transcript 看不见的工具调用。

    Returns:
        ``{"schema", "header", "events"}``。``events`` 全局有序，``seq`` 从 1
        开始连续递增。每条事件都冗余带 ``session_id``——JSONL 太容易被 ``cat``
        拼到一起，带上它下游能直接检出串会话。
    """
    session_id = str(cot.get("session_id") or "")
    agent_type = str(cot.get("agent_type") or "")
    turns: List[Dict[str, Any]] = list(cot.get("turns") or [])

    _augment_with_otel_orphans(turns, otel)

    # 并行时间线事件按 turn 归属分桶。归不进任何 turn 窗口的事件不丢弃，而是
    # 挂到时间上最近的前一个 turn 并标 in_ui=False —— 导出要完整，UI 要保持原样。
    windows = _turn_windows(turns)
    buckets: List[List[Dict[str, Any]]] = [[] for _ in turns]
    # 没有可用时间戳的事件（例如 transcript 里的 permission-mode 行，t_ms 恒为 0）
    # 无法放进执行顺序。它们不被丢弃，而是追加到有序流之后并标 ordered=False，
    # 这样「按 seq 读就是执行顺序」这条语义不会被污染，数据也不丢。
    unordered: List[Dict[str, Any]] = []
    for field_name, kind in _TIMELINE_FIELDS:
        for raw in (cot.get(field_name) or []):
            if not isinstance(raw, dict):
                continue
            t_ms = _event_ts_ms(raw)
            if t_ms <= 0:
                entry = _build_timeline_event(kind, raw)
                entry["ordered"] = False
                entry["in_ui"] = False
                unordered.append(entry)
                continue
            idx = _locate_turn(windows, t_ms)
            if idx < 0:
                # 会话没有任何 turn（极少见的空壳 cot.json），无处可挂
                entry = _build_timeline_event(kind, raw)
                entry["ordered"] = False
                unordered.append(entry)
                continue
            entry = _build_timeline_event(kind, raw)
            entry["_t_ms"] = t_ms
            t0, t1 = windows[idx]
            if not _renders_in_ui(kind, raw, agent_type, t_ms, windows[idx]):
                entry["in_ui"] = False
            buckets[idx].append(entry)

    events: List[Dict[str, Any]] = []
    seq = 0
    group_seq = 0
    phase_seq = 0

    def emit(event: Dict[str, Any], turn_index: Optional[int], group_id: Optional[int],
             role: Optional[str], phase_id: Optional[int]) -> None:
        nonlocal seq
        seq += 1
        event["seq"] = seq
        event["session_id"] = session_id
        if turn_index is not None:
            event["turn"] = turn_index
        if group_id is not None:
            event["group_id"] = group_id
        if role:
            event["role"] = role
        if phase_id is not None:
            event["phase_id"] = phase_id
        event.pop("_t_ms", None)
        events.append({k: v for k, v in event.items() if v is not None and v != ""})

    for turn_idx, turn in enumerate(turns):
        turn_index = _as_int(turn.get("turn_index"), turn_idx)
        steps: List[Dict[str, Any]] = list(turn.get("steps") or [])

        t0, _t1 = windows[turn_idx]
        emit({
            "type": "turn_start",
            "t_ms": t0 or None,
            "ts": _ms_to_iso(t0),
            "title": _first_sentence(turn.get("user_query") or ""),
            "content": turn.get("user_query") or "",
            "duration_ms": turn.get("turn_duration_ms_observed") or turn.get("turn_duration_ms"),
            "usage": turn.get("usage") or None,
            "complexity_score": turn.get("complexity_score"),
        }, turn_index, None, None, None)

        groups = _build_render_groups(steps)
        groups = _merge_thinking_phases(
            groups,
            _should_treat_pre_tool_as_thinking(agent_type, steps),
        )
        groups = _interleave_timeline_events(groups, buckets[turn_idx])

        for group in groups:
            group_seq += 1
            if group.phase:
                phase_seq += 1
                for thought in group.thoughts:
                    emit(_build_step_event(thought), turn_index,
                         group_seq, "leader", phase_seq)
                continue
            if group.leader is not None:
                emit(_build_step_event(group.leader), turn_index,
                     group_seq, "leader", None)
            for child in group.tools:
                if "_t_ms" in child or child.get("type") in {
                    "subagent", "permission", "notification", "compact", "environment",
                }:
                    emit(dict(child), turn_index, group_seq, "child", None)
                else:
                    emit(_build_step_event(child), turn_index,
                         group_seq, "child", None)

    ordered_count = len(events)
    for entry in unordered:
        emit(entry, None, None, None, None)

    header = _build_header(cot, events, otel)
    header["ordered_events"] = ordered_count
    header["unordered_events"] = len(unordered)
    return {"schema": TRACE_SCHEMA, "header": header, "events": events}


def _build_header(
    cot: Dict[str, Any],
    events: List[Dict[str, Any]],
    otel: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    type_counts: Dict[str, int] = {}
    for event in events:
        key = str(event.get("type"))
        type_counts[key] = type_counts.get(key, 0) + 1
    truncated = sum(1 for e in events if e.get("truncated"))
    return {
        "schema": TRACE_SCHEMA,
        "type": "session_header",
        "session_id": cot.get("session_id"),
        "agent_type": cot.get("agent_type"),
        "transcript_path": cot.get("transcript_path"),
        "extracted_at": cot.get("extracted_at"),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "turns": len(cot.get("turns") or []),
        "total_events": len(events),
        "event_type_counts": type_counts,
        "total_tool_calls": cot.get("total_tool_calls"),
        "tool_call_distribution": cot.get("tool_call_distribution"),
        "otel_merged": bool(otel and (otel.get("events") or [])),
        # 下游拿到 trace 后能一眼知道有多少条工具结果在提取阶段就被截断了
        "truncated_events": truncated,
    }
