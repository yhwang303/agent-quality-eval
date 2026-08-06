"""Cursor thinking 去重：hook 双写、跨通道重复、以及不能误删的占位符。

背景：Cursor 会把同一段 reasoning 推两次 afterAgentThought hook（一次带裸
generation_id，一次带加了后缀的），前端 Thinking Phase 于是每条思考显示两遍。
同时 transcript 的 text block 又会产出一份 thinking_inter 副本。这套测试守的是
「重复要清掉」和「真实步骤不能被当成重复删掉」这两条同时成立。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EXTRACTOR_SRC = (
    Path(__file__).resolve().parents[2]
    / "src" / "agent_cot" / "assets" / "cot-extractor-src"
)
if str(_EXTRACTOR_SRC) not in sys.path:
    sys.path.insert(0, str(_EXTRACTOR_SRC))

from cot_extractor import (  # noqa: E402
    _dedupe_double_written_thoughts,
    _dedupe_redundant_thinking_inter,
    _is_dedupe_safe_thought,
)


class _Step:
    """喂给去重函数的最小步骤壳。

    去重只读 step_type / content / metadata，用真的 ThoughtStep 需要一堆
    无关字段，反而让测试意图变模糊。
    """

    def __init__(self, index, step_type, content, t_ms=None, gid=None):
        self.step_index = index
        self.step_type = step_type
        self.content = content
        self.metadata = {}
        if t_ms is not None:
            self.metadata["observed_at_ms"] = t_ms
        if gid is not None:
            self.metadata["generation_id"] = gid


class _Turn:
    def __init__(self, steps):
        self.turn_index = 1
        self.steps = steps
        self.total_steps = len(steps)
        self.thinking_depth = sum(
            1 for s in steps if s.step_type in ("thinking_explicit", "thinking_inter")
        )


_LONG_A = "I need to examine how the frontend renders the timeline visualization."
_LONG_B = "Now I will check the backend export API before touching the exporter."


# ── hook 双写 ────────────────────────────────────────────────

def test_hook_double_write_keeps_one_copy():
    """裸 gid 与带后缀 gid 的同一条思考，只留一条。"""
    turn = _Turn([
        _Step(1, "thinking_explicit", _LONG_A, t_ms=1000, gid="uuid-1"),
        _Step(2, "thinking_explicit", _LONG_A, t_ms=1163, gid="uuid-1-0-0q2m"),
    ])
    removed = _dedupe_double_written_thoughts([turn])
    assert removed == 1
    assert len(turn.steps) == 1
    assert turn.total_steps == 1
    assert turn.thinking_depth == 1


def test_dropped_generation_id_is_recorded_not_silently_lost():
    """被删那条的 gid 要留在保留条上，hook 双写是上游行为，得留凭据。"""
    turn = _Turn([
        _Step(1, "thinking_explicit", _LONG_A, t_ms=1000, gid="uuid-1"),
        _Step(2, "thinking_explicit", _LONG_A, t_ms=1163, gid="uuid-1-0-0q2m"),
    ])
    _dedupe_double_written_thoughts([turn])
    assert turn.steps[0].metadata["duplicate_generation_ids"] == ["uuid-1-0-0q2m"]


def test_same_text_far_apart_in_time_is_kept():
    """间隔远的同文本是模型真的又想了一遍，不能合并。"""
    turn = _Turn([
        _Step(1, "thinking_explicit", _LONG_A, t_ms=1000, gid="uuid-1"),
        _Step(2, "thinking_explicit", _LONG_A, t_ms=1000 + 120_000, gid="uuid-9"),
    ])
    removed = _dedupe_double_written_thoughts([turn])
    assert removed == 0
    assert len(turn.steps) == 2


def test_different_thoughts_are_never_merged():
    turn = _Turn([
        _Step(1, "thinking_explicit", _LONG_A, t_ms=1000, gid="uuid-1"),
        _Step(2, "thinking_explicit", _LONG_B, t_ms=1100, gid="uuid-1-1-abcd"),
    ])
    assert _dedupe_double_written_thoughts([turn]) == 0
    assert len(turn.steps) == 2


# ── 占位符不能被当成重复 ──────────────────────────────────────

@pytest.mark.parametrize("placeholder", ["[REDACTED]", "[truncated]", "<omitted>"])
def test_placeholder_thoughts_survive_dedup(placeholder):
    """脱敏后的思考正文都长一样，但它们是不同的步骤，删了就是数据丢失。

    实测一条真实会话里有 48 组这种「重复」，全部必须保留。
    """
    turn = _Turn([
        _Step(i, "thinking_explicit", placeholder, t_ms=1000 + i * 10, gid=f"g{i}")
        for i in range(1, 6)
    ])
    assert _dedupe_double_written_thoughts([turn]) == 0
    assert len(turn.steps) == 5


def test_short_thoughts_are_not_used_as_identity():
    assert _is_dedupe_safe_thought("好的") is False
    assert _is_dedupe_safe_thought("[REDACTED]") is False
    assert _is_dedupe_safe_thought(_LONG_A) is True


# ── 跨通道重复（inter vs explicit）─────────────────────────────

def test_inter_copy_is_removed_even_when_far_from_its_explicit():
    """两条通道的注入位置由各自时间戳决定，常常相隔上百个 step。

    早先只扫 ±2 窗口，实测一条会话里 782 组重复因此全漏。
    """
    steps = [_Step(1, "thinking_inter", _LONG_A)]
    steps += [_Step(i, "tool_execution", "") for i in range(2, 60)]
    steps.append(_Step(60, "thinking_explicit", _LONG_A, t_ms=5000, gid="uuid-1"))
    turn = _Turn(steps)

    removed = _dedupe_redundant_thinking_inter([turn])
    assert removed == 1
    kinds = [s.step_type for s in turn.steps]
    assert "thinking_inter" not in kinds
    # 保留 explicit，因为它带 hook 元数据（observed_at_ms / generation_id）
    assert kinds.count("thinking_explicit") == 1


def test_inter_without_explicit_counterpart_is_kept():
    turn = _Turn([
        _Step(1, "thinking_inter", _LONG_A),
        _Step(2, "thinking_explicit", _LONG_B, t_ms=1000, gid="uuid-1"),
    ])
    assert _dedupe_redundant_thinking_inter([turn]) == 0
    assert len(turn.steps) == 2


def test_trailing_punctuation_difference_still_counts_as_duplicate():
    """两条通道落盘常差行尾几个字符，不能因此漏判。"""
    turn = _Turn([
        _Step(1, "thinking_inter", _LONG_A + "\n"),
        _Step(2, "thinking_explicit", _LONG_A, t_ms=1000, gid="uuid-1"),
    ])
    assert _dedupe_redundant_thinking_inter([turn]) == 1


# ── 乱码兜底：ASCII 骨架判重 + 干净文本升级 ────────────────────
#
# 背景（会话 6b0c4dd8 实测）：hook 落盘 events.jsonl 发生编码事故时，
# explicit 正文里的 CJK 被烤成乱码（"选择器"→"鈥?"，含 U+FFFD），norm /
# 前缀规则全部失效，同一条思考的 inter 与 explicit 双双存活。

_GARBLED_A = (
    "I need to examine how the frontend renders the 鈥?timeline鈥? "
    "visualization. 瀹屾垚鍚庡啀鐪嬪悗鍙? Next check the export pipeline."
)
_CLEAN_A = (
    "I need to examine how the frontend renders the “timeline” "
    "visualization. 完成后再看后端。 Next check the export pipeline."
)
_GARBLED_B = (
    "Now I will check the 鈥?backend鈥? export API 鉁? before touching "
    "the exporter implementation details."
)


def test_garbled_explicit_pair_deduped_by_ascii_skeleton():
    """乱码 explicit 与干净 inter 的 ASCII 骨架一致 → inter 删除，explicit 留下。"""
    turn = _Turn([
        _Step(1, "thinking_inter", _CLEAN_A),
        _Step(2, "thinking_explicit", _GARBLED_A, t_ms=1000, gid="uuid-1"),
    ])
    removed = _dedupe_redundant_thinking_inter([turn])
    assert removed == 1
    kinds = [s.step_type for s in turn.steps]
    assert kinds == ["thinking_explicit"]


def test_garbled_explicit_content_upgraded_from_clean_inter():
    """explicit 乱码、inter 干净时，explicit 正文换成干净副本，metadata 不动。"""
    explicit = _Step(2, "thinking_explicit", _GARBLED_A, t_ms=1000, gid="uuid-1")
    turn = _Turn([
        _Step(1, "thinking_inter", _CLEAN_A),
        explicit,
    ])
    removed = _dedupe_redundant_thinking_inter([turn])
    assert removed == 1
    assert explicit.content == _CLEAN_A
    # hook 元数据留在 explicit 上
    assert explicit.metadata["observed_at_ms"] == 1000
    assert explicit.metadata["generation_id"] == "uuid-1"


def test_clean_explicit_is_not_overwritten_by_inter():
    """explicit 本身干净时不做正文替换，只删 inter 副本。"""
    turn = _Turn([
        _Step(1, "thinking_inter", _CLEAN_A),
        _Step(2, "thinking_explicit", _CLEAN_A, t_ms=1000, gid="uuid-1"),
    ])
    removed = _dedupe_redundant_thinking_inter([turn])
    assert removed == 1
    assert turn.steps[0].content == _CLEAN_A


def test_garbled_explicit_without_inter_counterpart_is_kept():
    """乱码 explicit 找不到骨架一致的 inter 时原样保留——宁漏勿删。"""
    turn = _Turn([
        _Step(1, "thinking_explicit", _GARBLED_A, t_ms=1000, gid="uuid-1"),
        _Step(2, "thinking_inter", _LONG_B),
    ])
    removed = _dedupe_redundant_thinking_inter([turn])
    assert removed == 0
    assert len(turn.steps) == 2
    assert turn.steps[0].content == _GARBLED_A


def test_skeleton_shorter_than_floor_never_merges():
    """正文不足 24 字符时不具备区分度，即使完全相同也不判重——宁可保留。"""
    turn = _Turn([
        _Step(1, "thinking_inter", "用 rg 搜索 error 相关代码路径"),
        _Step(2, "thinking_explicit", "用 rg 搜索 error 相关代码路径",
              t_ms=1000, gid="uuid-1"),
    ])
    assert _dedupe_redundant_thinking_inter([turn]) == 0
    assert len(turn.steps) == 2


def test_substring_skeleton_matches_asymmetric_garble():
    """CJK 前导语乱码不对称时，严格相等的骨架会漏——子串规则兜底。"""
    inter_text = "先总结已完成的部分。 " + _LONG_A
    turn = _Turn([
        _Step(1, "thinking_inter", inter_text),
        _Step(2, "thinking_explicit", _LONG_A, t_ms=1000, gid="uuid-1"),
    ])
    removed = _dedupe_redundant_thinking_inter([turn])
    assert removed == 1
    assert [s.step_type for s in turn.steps] == ["thinking_explicit"]
