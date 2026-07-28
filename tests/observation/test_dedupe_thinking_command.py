"""``agent-cot dedupe-thinking``：存量 cot.json 的重复 thinking 清理。

extractor 里的修复只对之后提取的数据生效，已经写完的 cot.json 还带着旧的重复。
这套测试守的是「清理之后文件依然是一份可用的 cot.json」——步骤号连续、计数与
实际步骤对得上、默认不写盘。
"""

from __future__ import annotations

import json

import pytest

from agent_cot.commands.dedupe_thinking import dedupe_cot_dict, run_dedupe

_LONG_A = "I need to examine how the frontend renders the timeline visualization."
_LONG_B = "Now I will check the backend export API before touching the exporter."


def _step(index, step_type, content, t_ms=None, gid=None):
    md = {}
    if t_ms is not None:
        md["observed_at_ms"] = t_ms
    if gid is not None:
        md["generation_id"] = gid
    return {
        "step_index": index,
        "turn_index": 1,
        "step_type": step_type,
        "content": content,
        "tool_name": "",
        "tool_use_id": "",
        "metadata": md,
    }


def _cot_with_duplicates():
    """一轮里同时存在 hook 双写和 transcript 副本。"""
    steps = [
        # hook 双写：正文一致，gid 一个裸一个带后缀，相隔 163ms
        _step(1, "thinking_explicit", _LONG_A, t_ms=1000, gid="uuid-1"),
        _step(2, "thinking_explicit", _LONG_A, t_ms=1163, gid="uuid-1-0-0q2m"),
        _step(3, "tool_execution", "", t_ms=1200),
        # transcript 副本：与下面那条 explicit 撞车，且隔得很远
        _step(4, "thinking_inter", _LONG_B),
        _step(5, "tool_execution", "", t_ms=1400),
        _step(6, "thinking_explicit", _LONG_B, t_ms=1500, gid="uuid-2"),
    ]
    return {
        "session_id": "sess-a",
        "agent_type": "cursor",
        "total_thinking_steps": 4,
        "avg_steps_per_turn": 6.0,
        "turns": [{
            "turn_index": 1,
            "user_query": "看看前端怎么渲染的",
            "steps": steps,
            "total_steps": len(steps),
            "thinking_depth": 4,
        }],
    }


def test_both_kinds_of_duplicate_are_removed():
    cot = _cot_with_duplicates()
    stats = dedupe_cot_dict(cot)
    assert stats["double_written"] == 1
    assert stats["inter_copies"] == 1
    assert stats["steps_before"] - stats["steps_after"] == 2

    kinds = [s["step_type"] for s in cot["turns"][0]["steps"]]
    assert kinds.count("thinking_explicit") == 2
    assert "thinking_inter" not in kinds


def test_step_index_stays_dense_after_cleaning():
    """留下空号会让人以为步骤被吞了——extractor 去重后也会重排，这里跟上。"""
    cot = _cot_with_duplicates()
    dedupe_cot_dict(cot)
    indexes = [s["step_index"] for s in cot["turns"][0]["steps"]]
    assert indexes == list(range(1, len(indexes) + 1))


def test_counters_match_the_steps_that_remain():
    """计数不重算的话，前端 KPI 会跟树上数出来的数量对不上。"""
    cot = _cot_with_duplicates()
    dedupe_cot_dict(cot)
    turn = cot["turns"][0]
    assert turn["total_steps"] == len(turn["steps"])
    assert turn["thinking_depth"] == 2
    assert cot["total_thinking_steps"] == 2


def test_dropped_generation_id_is_kept_on_the_survivor():
    cot = _cot_with_duplicates()
    dedupe_cot_dict(cot)
    survivor = cot["turns"][0]["steps"][0]
    assert survivor["metadata"]["duplicate_generation_ids"] == ["uuid-1-0-0q2m"]


def test_running_twice_changes_nothing_the_second_time():
    """幂等：清理命令可能被反复执行，不能越删越多。"""
    cot = _cot_with_duplicates()
    dedupe_cot_dict(cot)
    after_first = json.dumps(cot, sort_keys=True, ensure_ascii=False)
    stats = dedupe_cot_dict(cot)
    assert stats["steps_before"] == stats["steps_after"]
    assert json.dumps(cot, sort_keys=True, ensure_ascii=False) == after_first


def test_a_clean_session_is_left_untouched():
    cot = {
        "session_id": "sess-a",
        "turns": [{
            "turn_index": 1,
            "steps": [
                _step(1, "thinking_explicit", _LONG_A, t_ms=1000, gid="uuid-1"),
                _step(2, "thinking_explicit", _LONG_B, t_ms=2000, gid="uuid-2"),
            ],
            "total_steps": 2,
            "thinking_depth": 2,
        }],
    }
    before = json.dumps(cot, sort_keys=True, ensure_ascii=False)
    stats = dedupe_cot_dict(cot)
    assert stats["steps_before"] == stats["steps_after"]
    assert json.dumps(cot, sort_keys=True, ensure_ascii=False) == before


# ── CLI 行为 ────────────────────────────────────────────────

def _write_cot(tmp_path, sid="sess-a"):
    path = tmp_path / f"{sid}_cot.json"
    path.write_text(
        json.dumps(_cot_with_duplicates(), ensure_ascii=False), encoding="utf-8"
    )
    return path


def test_dry_run_is_the_default_and_writes_nothing(tmp_path):
    """这条命令改的是用户已采集的数据，必须显式 --apply 才落盘。"""
    path = _write_cot(tmp_path)
    before = path.read_bytes()
    rc = run_dedupe(session_id=None, cot_dir=str(tmp_path), apply=False, quiet=True)
    assert rc == 0
    assert path.read_bytes() == before


def test_apply_writes_a_still_loadable_cot_json(tmp_path):
    path = _write_cot(tmp_path)
    rc = run_dedupe(session_id=None, cot_dir=str(tmp_path), apply=True, quiet=True)
    assert rc == 0
    cot = json.loads(path.read_text(encoding="utf-8"))
    steps = cot["turns"][0]["steps"]
    assert len(steps) == 4
    assert [s["step_index"] for s in steps] == [1, 2, 3, 4]
    assert cot["session_id"] == "sess-a", "身份字段不能在清理中丢失"


def test_a_session_id_prefix_is_enough_to_target_one_session(tmp_path):
    """会话 id 是 uuid，要求手打全长不现实。"""
    _write_cot(tmp_path, sid="3206a7e7-f591-430a-bb68-ecfc61d42787")
    rc = run_dedupe(session_id="3206a7e7", cot_dir=str(tmp_path), apply=False, quiet=True)
    assert rc == 0


def test_missing_session_fails_loudly(tmp_path):
    rc = run_dedupe(session_id="nope", cot_dir=str(tmp_path), apply=False, quiet=True)
    assert rc == 1


def test_exported_trace_stops_showing_the_duplicate_after_cleaning():
    """终点验收：清理之后导出的 trace 里同一条思考只出现一次。"""
    from agent_cot.trace import flatten_session

    cot = _cot_with_duplicates()
    cot["turns"][0]["turn_start_ms_observed"] = 1000
    cot["turns"][0]["turn_end_ms_observed"] = 1500

    def thinking_texts(data):
        return [
            e["content"] for e in flatten_session(data)["events"]
            if e["type"] == "thinking"
        ]

    assert thinking_texts(cot).count(_LONG_A) == 2
    dedupe_cot_dict(cot)
    assert thinking_texts(cot).count(_LONG_A) == 1
    assert thinking_texts(cot).count(_LONG_B) == 1
