"""Trace 导出：顺序、完整性、会话隔离、三种格式。

这套测试守的是闭环 harness 的地基——下游 agent 靠导出的 trace 判断优化方向，
所以「顺序对不对」「有没有悄悄丢事件」「会不会串会话」比覆盖率更重要。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_cot.trace import (
    export_session_trace,
    export_turn_trace,
    flatten_session,
    render_jsonl,
    sanitize_session_id,
)

_GOLD = Path(__file__).resolve().parents[2] / "agent-regression-gold-dataset"


def _step(index, step_type, *, t_ms=0, tool=None, use_id=None, content="", metadata=None):
    md = dict(metadata or {})
    if t_ms:
        md["observed_at_ms"] = t_ms
    return {
        "step_index": index,
        "turn_index": 0,
        "step_type": step_type,
        "content": content,
        "tool_name": tool or "",
        "tool_use_id": use_id or "",
        "metadata": md,
    }


def _cot(steps, *, session_id="sess-a", agent_type="claude", **timelines):
    stamps = [s["metadata"]["observed_at_ms"] for s in steps if s["metadata"].get("observed_at_ms")]
    turn = {
        "turn_index": 0,
        "user_query": "帮我修一下这个 bug",
        "steps": steps,
        "turn_start_ms_observed": min(stamps) if stamps else 0,
        "turn_end_ms_observed": max(stamps) if stamps else 0,
    }
    base = {
        "session_id": session_id,
        "agent_type": agent_type,
        "transcript_path": "/tmp/x.jsonl",
        "extracted_at": "2026-07-01T00:00:00+00:00",
        "turns": [turn],
    }
    base.update(timelines)
    return base


# ── 顺序 ────────────────────────────────────────────────────

def test_permission_event_lands_between_the_tools_it_happened_between():
    """截图里的核心诉求：权限请求要按真实时刻卡在两次工具调用之间。"""
    cot = _cot(
        [
            _step(1, "thinking_explicit", t_ms=1000, content="先看看代码"),
            _step(2, "tool_decision", t_ms=1100, tool="Read", use_id="u1"),
            _step(3, "tool_execution", t_ms=1200, tool="Read", use_id="u1"),
            _step(4, "tool_decision", t_ms=1500, tool="Bash", use_id="u2"),
            _step(5, "tool_execution", t_ms=1600, tool="Bash", use_id="u2"),
        ],
        permission_events=[{"t_ms": 1300, "kind": "PermissionRequest", "tool_name": "Bash"}],
    )
    types = [e["type"] for e in flatten_session(cot)["events"]]
    assert types == [
        "turn_start", "thinking", "tool_call", "tool_result",
        "permission",
        "tool_call", "tool_result",
    ]


def test_concurrent_tool_calls_are_paired_decision_with_its_own_result():
    """并发 tool_use 在 transcript 里是 D1 D2 E1 E2，导出必须配成 D1 E1 D2 E2。"""
    cot = _cot([
        _step(1, "thinking_explicit", t_ms=1000),
        _step(2, "tool_decision", t_ms=1100, tool="Read", use_id="u1"),
        _step(3, "tool_decision", t_ms=1101, tool="Grep", use_id="u2"),
        _step(4, "tool_execution", t_ms=1200, tool="Read", use_id="u1"),
        _step(5, "tool_execution", t_ms=1201, tool="Grep", use_id="u2"),
    ])
    events = [e for e in flatten_session(cot)["events"] if e["type"].startswith("tool_")]
    assert [(e["type"], e["tool_use_id"]) for e in events] == [
        ("tool_call", "u1"), ("tool_result", "u1"),
        ("tool_call", "u2"), ("tool_result", "u2"),
    ]
    # 配对的代价：u1 的结果(1200)排在了 u2 的调用(1101)之前，时间上是回跳的。
    # 这是刻意的，和树上看到的一致，下游按 t_ms 重排反而会拆散调用与结果。
    stamps = [e["t_ms"] for e in events]
    assert stamps == [1100, 1200, 1101, 1201]


def test_seq_is_dense_and_sequential_work_stays_chronological():
    """工具串行执行时，seq 顺序就是时间顺序。

    并发调用是唯一的例外，见 test_concurrent_tool_calls_are_paired_*：
    结果会被提到自己那次调用的后面，此时 t_ms 允许回跳。
    """
    cot = _cot(
        [
            _step(1, "thinking_explicit", t_ms=1000),
            _step(2, "tool_decision", t_ms=1400, tool="Bash", use_id="u1"),
            _step(3, "tool_execution", t_ms=1600, tool="Bash", use_id="u1"),
        ],
        notification_events=[{"t_ms": 1500, "kind": "Notification", "message": "需要确认"}],
    )
    events = flatten_session(cot)["events"]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    stamps = [e["t_ms"] for e in events if e.get("t_ms")]
    assert stamps == sorted(stamps)


def test_steps_without_timestamp_stay_next_to_their_neighbours():
    """缺时间戳的步骤应沿用前一步的时刻，而不是被甩到时间线最前面。"""
    cot = _cot([
        _step(1, "user_input", t_ms=1000, content="做这个"),
        _step(2, "thinking_explicit", content="没有时间戳的思考"),
        _step(3, "tool_decision", t_ms=2000, tool="Bash", use_id="u1"),
    ])
    types = [e["type"] for e in flatten_session(cot)["events"]]
    assert types == ["turn_start", "user_input", "thinking", "tool_call"]


# ── 完整性 ──────────────────────────────────────────────────

def test_every_source_event_survives_the_export():
    cot = _cot(
        [_step(1, "user_input", t_ms=1000), _step(2, "final_response", t_ms=3000)],
        permission_events=[{"t_ms": 1100, "kind": "PermissionRequest"}],
        subagent_timeline=[{"t_ms": 1200, "sub_agent_id": "a1", "phase": "start"}],
        notification_events=[{"t_ms": 1300, "kind": "Notification", "message": "hi"}],
        compact_events=[{"t_ms": 1400, "phase": "before", "before_tokens": 100}],
        environment_events=[{"t_ms": 1500, "kind": "CwdChanged"}],
    )
    events = flatten_session(cot)["events"]
    counts = {}
    for event in events:
        counts[event["type"]] = counts.get(event["type"], 0) + 1
    for kind in ("permission", "subagent", "notification", "compact", "environment"):
        assert counts.get(kind) == 1, f"{kind} 事件在导出中丢失"
    assert counts["user_input"] == 1 and counts["final_response"] == 1


def _two_turn_cot(**timelines):
    """两轮会话，中间留一段空档：第 0 轮 2000 结束，第 1 轮 5000 才开始。"""
    base = {
        "session_id": "sess-a",
        "agent_type": "claude",
        "transcript_path": "/tmp/x.jsonl",
        "extracted_at": "2026-07-01T00:00:00+00:00",
        "turns": [
            {
                "turn_index": 0,
                "user_query": "第一轮",
                "steps": [_step(1, "user_input", t_ms=1000),
                          _step(2, "final_response", t_ms=2000)],
                "turn_start_ms_observed": 1000,
                "turn_end_ms_observed": 2000,
            },
            {
                "turn_index": 1,
                "user_query": "第二轮",
                "steps": [_step(3, "user_input", t_ms=5000),
                          _step(4, "final_response", t_ms=6000)],
                "turn_start_ms_observed": 5000,
                "turn_end_ms_observed": 6000,
            },
        ],
    }
    base.update(timelines)
    return base


def test_event_in_the_gap_after_a_turn_belongs_to_that_turn():
    """subagent 常在主线程最后一步之后才收尾，这类事件属于它跟着的那一轮。

    卡死「必须落在 turn_end 之前」会让它们从树上消失。
    """
    cot = _two_turn_cot(
        subagent_timeline=[{"t_ms": 3000, "sub_agent_id": "a1", "phase": "end"}],
    )
    events = [e for e in flatten_session(cot)["events"] if e["type"] == "subagent"]
    assert len(events) == 1
    assert events[0]["turn"] == 0, "落在空档里的事件应归前一轮，而不是被丢弃或推给下一轮"
    assert events[0].get("in_ui") is not False, "归属明确的事件必须照常渲染"


def test_a_timeline_event_is_attributed_to_exactly_one_turn():
    """turn 窗口必须互不重叠，否则同一个事件会在多轮里各画一遍。"""
    cot = _two_turn_cot(
        permission_events=[
            {"t_ms": 1500, "kind": "PermissionRequest"},
            {"t_ms": 3000, "kind": "PermissionRequest"},
            {"t_ms": 5500, "kind": "PermissionRequest"},
            {"t_ms": 9000, "kind": "PermissionRequest"},
        ],
    )
    events = [e for e in flatten_session(cot)["events"] if e["type"] == "permission"]
    assert len(events) == 4, "事件既不能重复也不能丢"
    assert [e["turn"] for e in events] == [0, 0, 1, 1]


def test_events_without_timestamp_are_kept_but_marked_unordered():
    """没有时间戳的事件插不进执行顺序，但不能丢——追加到末尾并标记。"""
    cot = _cot(
        [_step(1, "user_input", t_ms=1000)],
        permission_events=[
            {"t_ms": 0, "kind": "PermissionMode", "mode": "acceptEdits"},
            {"t_ms": 1100, "kind": "PermissionRequest"},
        ],
    )
    flat = flatten_session(cot)
    assert flat["header"]["unordered_events"] == 1
    unordered = [e for e in flat["events"] if e.get("ordered") is False]
    assert len(unordered) == 1
    assert unordered[0]["payload"]["kind"] == "PermissionMode"
    # 无序事件必须排在有序流之后，否则「按 seq 读即执行顺序」就不成立了
    assert unordered[0]["seq"] == max(e["seq"] for e in flat["events"])


def test_truncated_tool_result_is_flagged_with_original_length():
    cot = _cot([
        _step(1, "tool_decision", t_ms=1000, tool="Bash", use_id="u1"),
        _step(2, "tool_execution", t_ms=1100, tool="Bash", use_id="u1",
              content="x" * 2000, metadata={"truncated": True, "result_len": 18432}),
    ])
    result = next(e for e in flatten_session(cot)["events"] if e["type"] == "tool_result")
    assert result["truncated"] is True
    assert result["original_len"] == 18432


def test_thinking_content_is_exported_in_full():
    """思考内容是导出的核心价值，不允许在 jsonl 里被截断。"""
    long_thought = "推理" * 5000
    cot = _cot([_step(1, "thinking_explicit", t_ms=1000, content=long_thought)])
    event = next(e for e in flatten_session(cot)["events"] if e["type"] == "thinking")
    assert event["content"] == long_thought


def test_plan_metadata_rides_along_with_the_plan_tool_call():
    cot = _cot([
        _step(1, "tool_decision", t_ms=1000, tool="TaskCreate", use_id="u1",
              metadata={"plan_total": 3, "plan_completed_count": 1, "plan_snapshot_idx": 0}),
    ])
    event = next(e for e in flatten_session(cot)["events"] if e["type"] == "tool_call")
    assert event["plan"]["total"] == 3
    assert event["plan"]["completed"] == 1


# ── UI 树重建 ───────────────────────────────────────────────

def test_group_and_role_let_the_ui_rebuild_its_tree():
    cot = _cot([
        _step(1, "thinking_explicit", t_ms=1000, content="想一下"),
        _step(2, "tool_decision", t_ms=1100, tool="Read", use_id="u1"),
        _step(3, "tool_execution", t_ms=1200, tool="Read", use_id="u1"),
    ])
    events = [e for e in flatten_session(cot)["events"] if e["type"] != "turn_start"]
    leader = events[0]
    assert leader["role"] == "leader"
    assert all(e["role"] == "child" for e in events[1:])
    assert len({e["group_id"] for e in events}) == 1


def test_consecutive_standalone_thinking_shares_a_phase_id():
    cot = _cot([
        _step(1, "thinking_explicit", t_ms=1000, content="第一段"),
        _step(2, "thinking_explicit", t_ms=1100, content="第二段"),
        _step(3, "thinking_explicit", t_ms=1200, content="第三段"),
    ])
    events = [e for e in flatten_session(cot)["events"] if e["type"] == "thinking"]
    assert len({e["phase_id"] for e in events}) == 1


# ── 会话隔离 ────────────────────────────────────────────────

def test_every_event_carries_its_own_session_id():
    """JSONL 极易被 cat 拼接，每行带 session_id 才能检出串会话。"""
    events = flatten_session(_cot([_step(1, "user_input", t_ms=1000)], session_id="sess-a"))["events"]
    assert events and all(e["session_id"] == "sess-a" for e in events)


def test_two_sessions_do_not_bleed_into_each_other():
    a = flatten_session(_cot([_step(1, "user_input", t_ms=1000, content="A")], session_id="sess-a"))
    b = flatten_session(_cot([_step(1, "user_input", t_ms=2000, content="B")], session_id="sess-b"))
    assert {e["session_id"] for e in a["events"]} == {"sess-a"}
    assert {e["session_id"] for e in b["events"]} == {"sess-b"}
    assert a["header"]["session_id"] != b["header"]["session_id"]


@pytest.mark.parametrize("raw,expected_safe", [
    ("zhangsan::abc-123", "zhangsan-abc-123"),
    ("../../etc/passwd", "etc-passwd"),
    ("", "session"),
])
def test_session_id_is_sanitised_before_it_reaches_a_filename(raw, expected_safe):
    safe = sanitize_session_id(raw)
    assert safe == expected_safe
    assert "/" not in safe and "\\" not in safe and ":" not in safe


def test_cli_writes_the_same_bytes_the_http_export_returns(tmp_path):
    """CLI 落盘与 HTTP 返回必须字节一致。

    Windows 上文本模式会把 \\n 转成 \\r\\n，一个 242 行的 trace 就凭空多出 242
    字节，下游拿哈希或 diff 比对两条路径的产物时会看到满屏假差异。
    """
    from agent_cot.commands.export_trace import run_export

    cot = _cot([
        _step(1, "thinking_explicit", t_ms=1000, content="想一下"),
        _step(2, "tool_decision", t_ms=1100, tool="Read", use_id="u1"),
    ])
    cot_file = tmp_path / "sess-a_cot.json"
    cot_file.write_text(json.dumps(cot, ensure_ascii=False), encoding="utf-8")

    out = tmp_path / "trace.jsonl"
    rc = run_export(
        session_id=None, cot_path=str(cot_file), fmt="jsonl",
        output=str(out), quiet=True,
    )
    assert rc == 0

    on_disk = out.read_bytes()
    assert b"\r\n" not in on_disk, "jsonl 不该出现 CRLF"
    # 逐行比对而不是整体比，因为 header 里的 exported_at 每次不同
    served = export_session_trace(cot, fmt="jsonl")["content"].encode("utf-8")
    assert on_disk.count(b"\n") == served.count(b"\n")
    assert on_disk.split(b"\n")[1:] == served.split(b"\n")[1:]


# ── 三种格式 ────────────────────────────────────────────────

def test_jsonl_starts_with_a_header_line_then_one_event_per_line():
    cot = _cot([
        _step(1, "thinking_explicit", t_ms=1000, content="想"),
        _step(2, "tool_decision", t_ms=1100, tool="Read", use_id="u1"),
    ])
    flat = flatten_session(cot)
    lines = render_jsonl(flat).strip().split("\n")
    header = json.loads(lines[0])
    assert header["type"] == "session_header"
    parsed = [json.loads(line) for line in lines[1:]]
    assert [e["seq"] for e in parsed] == list(range(1, len(parsed) + 1))
    assert len(parsed) == flat["header"]["total_events"]


@pytest.mark.parametrize("fmt", ["jsonl", "json", "md"])
def test_all_three_formats_agree_on_event_order(fmt):
    cot = _cot(
        [
            _step(1, "thinking_explicit", t_ms=1000, content="想一下这个问题"),
            _step(2, "tool_decision", t_ms=1100, tool="Read", use_id="u1"),
            _step(3, "tool_execution", t_ms=1200, tool="Read", use_id="u1"),
        ],
        permission_events=[{"t_ms": 1150, "kind": "PermissionRequest"}],
    )
    result = export_session_trace(cot, fmt=fmt)
    assert result["filename"] == f"trace-sess-a.{fmt}"
    assert result["session_id"] == "sess-a"
    content = result["content"]
    # 三种格式都必须让 permission 出现在两个工具事件之间
    if fmt == "md":
        positions = [content.index(f"**#{seq}**") for seq in (2, 3, 4, 5)]
        assert positions == sorted(positions)
    else:
        assert result["event_count"] == 5


def test_unsupported_format_is_rejected_loudly():
    with pytest.raises(ValueError, match="不支持的导出格式"):
        export_session_trace(_cot([_step(1, "user_input", t_ms=1)]), fmt="yaml")


# ── 单轮导出 ────────────────────────────────────────────────
#
# 界面上每张交互卡片旁的导出按钮走这条路径。用户要的是「这一轮到底干了什么」，
# 所以最关键的两件事：别把别的轮次的事件带进来，也别漏掉本轮的任何一个。

def _turn_export_events(cot, turn_index):
    """取单轮导出里的事件（跳过第一行 header）。"""
    body = export_turn_trace(cot, turn_index)["content"].strip().split("\n")[1:]
    return [json.loads(line) for line in body]


def test_turn_export_contains_only_that_turn():
    cot = _two_turn_cot(
        permission_events=[
            {"t_ms": 1500, "kind": "PermissionRequest"},
            {"t_ms": 5500, "kind": "PermissionRequest"},
        ],
    )
    turn0 = _turn_export_events(cot, 0)
    turn1 = _turn_export_events(cot, 1)
    assert turn0 and turn1, "两轮的事件都不应为空"
    assert {e["turn"] for e in turn0} == {0}
    assert {e["turn"] for e in turn1} == {1}

    # 两轮之和要等于整条会话的有序事件数：既不重复也不遗漏
    session_ordered = sum(
        1 for e in flatten_session(cot)["events"] if e.get("ordered") is not False
    )
    assert len(turn0) + len(turn1) == session_ordered


def test_turn_export_renumbers_seq_but_keeps_the_session_position():
    """单轮文件自己要能从 1 读到底；回到整条会话里的位置留在 session_seq。"""
    events = _turn_export_events(_two_turn_cot(), 1)
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert all(e["session_seq"] > e["seq"] for e in events), \
        "第二轮在会话里的原始序号必然大于它在单轮文件里的序号"


def test_turn_export_header_declares_its_scope_and_the_whole_session_size():
    """只看单轮文件也要知道「这不是全部」，否则下游会把一轮当成整条会话。"""
    cot = _two_turn_cot()
    result = export_turn_trace(cot, 0)
    header = json.loads(result["content"].strip().split("\n")[0])
    assert header["type"] == "turn_header"
    assert header["scope"] == "turn"
    assert header["turn_index"] == 0
    assert header["user_query"] == "第一轮"
    assert header["session_turns"] == 2
    assert header["session_total_events"] > header["total_events"]


def test_turn_export_filename_pins_session_and_turn():
    result = export_turn_trace(_two_turn_cot(), 1)
    assert result["filename"] == "trace-sess-a-turn1.jsonl"
    assert result["turn_index"] == 1


def test_turn_export_rejects_a_turn_that_does_not_exist():
    """轮次号不存在要报错，不能回退成「导出整条会话」这种张冠李戴。"""
    with pytest.raises(ValueError, match="turn 7"):
        export_turn_trace(_two_turn_cot(), 7)


def test_turn_export_keeps_every_event_of_that_turn_verbatim():
    """单轮导出只是筛选，不做二次加工——内容必须与整会话导出逐字一致。"""
    long_thought = "推理" * 3000
    cot = _two_turn_cot()
    cot["turns"][0]["steps"].append(
        _step(9, "thinking_explicit", t_ms=1500, content=long_thought)
    )
    thinking = next(e for e in _turn_export_events(cot, 0) if e["type"] == "thinking")
    assert thinking["content"] == long_thought


# ── 真实数据 ────────────────────────────────────────────────

@pytest.mark.parametrize("case", [
    "case-01-ide-benchmark-strategy",
    "case-02-opencli-vs-browser-skill",
    "case-03-hook-restart-guidance",
    "case-04-ab-evaluation-method",
    "case-05-iwiki-report-rewrite",
    "case-06-product-brainstorm",
])
def test_real_gold_traces_flatten_across_all_agent_types(case):
    """六个真实 trace 覆盖 claude / cursor / codex / codebuddy 四种来源。"""
    path = _GOLD / case / "trace.json"
    if not path.exists():
        pytest.skip(f"gold dataset 缺失：{path}")
    cot = json.loads(path.read_text(encoding="utf-8"))
    flat = flatten_session(cot)
    events = flat["events"]
    assert events, "真实 trace 拍平后不应为空"
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert {e["session_id"] for e in events} == {cot["session_id"]}
    source_steps = sum(len(t.get("steps") or []) for t in cot.get("turns") or [])
    exported_steps = sum(1 for e in events if "step_index" in e)
    assert exported_steps == source_steps, "有 step 在导出过程中丢失"
