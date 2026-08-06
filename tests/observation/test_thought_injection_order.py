"""Cursor thinking/tool 注入的时序守卫。

回归背景（会话 6b0c4dd8 turn1 实测）：
- ``agentToolResult`` 重放事件把**重放时刻**当成 ``observed_at_ms`` 写进
  step metadata，污染 ``_inject_agent_thoughts`` 的时间锚——全部 thinking
  被判成「早于第一个锚点」，17 条 thinking 扎堆排到 turn 开头；
- 即便没有重放污染，无锚点场景下注入循环也会把 thought 倒到 step 0
  （user_input）前面，形成「用户还没问就先想了一屏」；
- 通道 5 的重排把无时间戳 step 一律排key=0，会把 tool_decision→
  tool_execution 相邻对拆散、沉到带锚步骤之前。

这里守三条：user_input 永远第一、无锚步骤继承邻近锚点、合成步骤按真实
wall-clock 交错而不是简单 append 或沉底。
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXTRACTOR_SRC = (
    Path(__file__).resolve().parents[2]
    / "src" / "agent_cot" / "assets" / "cot-extractor-src"
)
if str(_EXTRACTOR_SRC) not in sys.path:
    sys.path.insert(0, str(_EXTRACTOR_SRC))

from cot_extractor import (  # noqa: E402
    StepType,
    ThoughtStep,
    TurnCoT,
    _attach_cursor_events,
    _inject_agent_thoughts,
    _parse_user_ts_ms,
    _user_ts_turn_windows,
)


def _step(i, step_type, content="", *, gid=None, t_ms=None,
          tool_name="", tool_use_id=""):
    md = {}
    if gid:
        md["generation_id"] = gid
    if t_ms is not None:
        md["observed_at_ms"] = t_ms
    return ThoughtStep(
        step_index=i, turn_index=1, step_type=step_type, content=content,
        metadata=md, tool_name=tool_name, tool_use_id=tool_use_id,
    )


def _turn(steps):
    turn = TurnCoT(turn_index=1, user_query="q")
    turn.steps = steps
    turn.total_steps = len(steps)
    return turn


def _thought(t_ms, gid, text):
    return {
        "t": t_ms,
        "event": "afterAgentThought",
        "payload": {"generation_id": gid, "text": text, "duration_ms": 5},
    }


_LONG_THOUGHT = "先把任务拆清楚再动手，这条思考的真实时间远早于第一个工具锚点。"


def test_thought_never_injected_before_user_input():
    """锚点全部在 user_input 之后时，thinking 也得排在用户提问之后。"""
    turn = _turn([
        _step(1, StepType.USER_INPUT, "用户：做 XX 任务"),
        _step(2, StepType.TOOL_DECISION, "调用 Shell", gid="g1",
              tool_name="Shell", tool_use_id="c1"),
        _step(3, StepType.TOOL_EXECUTION, "", gid="g1", t_ms=100_000,
              tool_name="Shell", tool_use_id="c1"),
    ])
    events = [_thought(50_000, "g1", _LONG_THOUGHT)]

    stats = _inject_agent_thoughts([turn], events)

    types = [s.step_type for s in turn.steps]
    assert stats["thought_injected"] == 1
    assert types[0] == StepType.USER_INPUT
    # 早于锚点的 thought 落在 user_input 之后、tool_decision 之前
    assert types == [
        StepType.USER_INPUT,
        StepType.THINKING_EXPLICIT,
        StepType.TOOL_DECISION,
        StepType.TOOL_EXECUTION,
    ]


def test_thought_injection_with_no_anchors_keeps_user_input_first():
    """turn 内完全无锚点时，thought 不能倒到 user_input 前面。"""
    turn = _turn([
        _step(1, StepType.USER_INPUT, "用户：只做只读扫描"),
        _step(2, StepType.TOOL_DECISION, "调用 Glob", gid="g1",
              tool_name="Glob", tool_use_id="c1"),
    ])
    events = [_thought(50_000, "g1", _LONG_THOUGHT)]

    stats = _inject_agent_thoughts([turn], events)

    types = [s.step_type for s in turn.steps]
    assert stats["thought_injected"] == 1
    assert types[0] == StepType.USER_INPUT
    assert types[1] == StepType.THINKING_EXPLICIT


def test_channel5_synth_step_interleaves_by_wall_clock():
    """合成步骤按真实时间插入，而不是 append 到队尾或沉到队首。"""
    turn = _turn([
        _step(1, StepType.USER_INPUT, "q"),
        _step(2, StepType.THINKING_EXPLICIT, _LONG_THOUGHT, t_ms=2000),
        _step(3, StepType.FINAL_RESPONSE, "done"),
    ])
    events = [{
        "t": 1000,
        "event": "afterShellExecution",
        "brief_input": {"command": "pytest -q"},
        "brief_output": {"exit_code": 0, "stdout": "ok"},
        "payload": {},
    }]

    _attach_cursor_events([turn], events)

    types = [s.step_type for s in turn.steps]
    assert types == [
        StepType.USER_INPUT,
        StepType.TOOL_EXECUTION,      # 合成的 shell 步骤 t=1000
        StepType.THINKING_EXPLICIT,   # t=2000
        StepType.FINAL_RESPONSE,
    ]


def test_channel5_resort_preserves_decision_execution_adjacency():
    """无锚的 tool_decision→tool_execution 对不能被重排拆散。"""
    turn = _turn([
        _step(1, StepType.USER_INPUT, "q"),
        _step(2, StepType.TOOL_DECISION, "调用 Shell",
              tool_name="Shell", tool_use_id="c1"),
        _step(3, StepType.TOOL_EXECUTION, "",
              tool_name="Shell", tool_use_id="c1"),
        _step(4, StepType.THINKING_EXPLICIT, _LONG_THOUGHT, t_ms=2000),
        _step(5, StepType.FINAL_RESPONSE, "done"),
    ])
    # afterFileEdit 没有对应 Edit exec step → orphan → 通道 5 合成并触发重排
    events = [{
        "t": 1000,
        "event": "afterFileEdit",
        "brief_input": {"file_path": "a.py"},
        "brief_output": {"lines_added": 3},
        "payload": {},
    }]

    _attach_cursor_events([turn], events)

    types = [s.step_type for s in turn.steps]
    assert types[0] == StepType.USER_INPUT
    assert types[-1] == StepType.FINAL_RESPONSE
    # Shell 的 decision→exec 对必须仍然相邻（不许被合成步骤或重排拆开）
    dec = types.index(StepType.TOOL_DECISION)
    assert types[dec + 1] == StepType.TOOL_EXECUTION
    # 合成的 Edit 步骤（t=1000）按 wall-clock 排在 thinking（t=2000）之前
    execs = [i for i, t in enumerate(types) if t == StepType.TOOL_EXECUTION]
    edit = next(i for i in execs if i != dec + 1)
    assert edit < types.index(StepType.THINKING_EXPLICIT)


# ─── v0.19.6：压缩场景——用户消息 <timestamp> 全覆盖窗口路由 ───
#
# 回归背景（会话 83811387 实测）：Cursor 压缩重写 transcript 后，晚近
# turn 的 tool_use 块被剥掉，turn 失去全部 observed_at_ms 锚点；事件
# 路由（thinking 注入 + orphan 合成）只剩锚点窗口可用，把压缩点之后
# 数百条事件全部塌陷进最后一个有锚的 turn，晚近 turn 只剩
# user_input + final_response。

_TS_T1 = "<timestamp>Friday, Jul 31, 2026, 7:33 PM (UTC+8)</timestamp>"
_TS_T2 = "<timestamp>Monday, Aug 3, 2026, 11:09 AM (UTC+8)</timestamp>"


def _ts_turn(idx, ts_text, anchored=False):
    turn = TurnCoT(turn_index=idx, user_query=f"q{idx}")
    steps = [
        ThoughtStep(
            step_index=1, turn_index=idx, step_type=StepType.USER_INPUT,
            content=f"{ts_text}\n<user_query>\nq{idx}\n</user_query>",
            metadata={}, tool_name="", tool_use_id="",
        )
    ]
    if anchored:
        steps.append(ThoughtStep(
            step_index=2, turn_index=idx, step_type=StepType.TOOL_EXECUTION,
            content="Shell", metadata={"generation_id": f"g{idx}",
                                       "observed_at_ms": _parse_user_ts_ms(ts_text) + 1000},
            tool_name="Shell", tool_use_id=f"c{idx}",
        ))
    steps.append(ThoughtStep(
        step_index=9, turn_index=idx, step_type=StepType.FINAL_RESPONSE,
        content="done", metadata={}, tool_name="", tool_use_id="",
    ))
    turn.steps = steps
    turn.total_steps = len(steps)
    return turn


def test_parse_user_ts_ms_basic():
    """IDE 注入的 <timestamp> 能解析成正确 epoch ms，垃圾输入返回 None。"""
    ms = _parse_user_ts_ms(_TS_T1)
    assert ms is not None
    from datetime import datetime, timezone
    expect = int(datetime(2026, 7, 31, 11, 33, tzinfo=timezone.utc).timestamp() * 1000)
    assert ms == expect
    assert _parse_user_ts_ms("no timestamp here") is None
    assert _parse_user_ts_ms("") is None
    assert _parse_user_ts_ms(None) is None
    assert _parse_user_ts_ms(
        "<timestamp>Sunday, Foo 99, 2026, 7:33 PM (UTC+8)</timestamp>") is None


def test_user_ts_windows_cover_all_turns():
    """窗口半开 [start_i, start_i+1)：任何时刻都有唯一归属。"""
    t1 = _ts_turn(1, _TS_T1, anchored=True)
    t2 = _ts_turn(2, _TS_T2)
    windows = _user_ts_turn_windows([t1, t2])
    assert windows is not None
    s1, s2 = _parse_user_ts_ms(_TS_T1), _parse_user_ts_ms(_TS_T2)
    assert s1 and s2 and s2 > s1
    assert windows[0][0] == s1 and windows[0][1] == s2
    assert windows[1][0] == s2 and windows[1][1] is None  # 末 turn +∞


def test_orphan_event_routes_to_later_turn_after_compaction():
    """无锚 turn 的 orphan 事件按用户时间戳归位，不再塌陷进有锚 turn。"""
    t1 = _ts_turn(1, _TS_T1, anchored=True)
    t2 = _ts_turn(2, _TS_T2)
    s2 = _parse_user_ts_ms(_TS_T2)
    events = [{
        # 压缩后的 turn 没有任何 Edit tool_execution 占位，afterFileEdit
        # 事件无处 zip → orphan → 走 _pick_turn_for 合成
        "t": s2 + 60_000,  # 落在 turn2 窗口内，远离 turn1 的锚点范围
        "event": "afterFileEdit",
        "brief_input": {"file_path": "a.py", "edits_count": 1},
        "brief_output": {"lines_added": 3},
        "payload": {},
    }]
    t1_steps_before = len(t1.steps)

    _attach_cursor_events([t1, t2], events)

    assert len(t1.steps) == t1_steps_before  # turn1 不再被倒垃圾
    synth = [s for s in t2.steps
             if s.step_type == StepType.TOOL_EXECUTION
             and (s.metadata or {}).get("synthesised_from_events")]
    assert len(synth) == 1
    assert synth[0].turn_index == 2


def test_thought_routes_to_anchorless_turn_by_user_timestamp():
    """无锚 turn 的 thinking 不再被判 orphan 丢弃。"""
    t1 = _ts_turn(1, _TS_T1, anchored=True)
    t2 = _ts_turn(2, _TS_T2)
    s2 = _parse_user_ts_ms(_TS_T2)
    events = [_thought(s2 + 5000, "gid-not-on-map", _LONG_THOUGHT)]

    stats = _inject_agent_thoughts([t1, t2], events)

    assert stats["thought_injected"] == 1
    assert stats["thought_orphan"] == 0
    types2 = [s.step_type for s in t2.steps]
    assert types2[0] == StepType.USER_INPUT
    assert StepType.THINKING_EXPLICIT in types2
    # user_input 仍然第一，final_response 仍然最后
    assert types2[-1] == StepType.FINAL_RESPONSE
