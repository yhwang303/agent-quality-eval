"""单轮 token 用量的取数口径。

Cursor 常常拿不到 per-turn 的 hook 真值（``turn.usage`` 全 0），这时候必须
**累加本轮每个 LLM step**，而不是从 payload 里挑用量最大的那一份。挑单个的
后果是 eval 面板报几百几千、turn 卡片报几万，同一轮交互两个答案。
"""

from __future__ import annotations

from agent_quality_eval.evaluation.session_eval import _extract_turn_usage


def _llm_step(index, inp, out, cache_read=0, cache_write=0):
    return {
        "step_index": index,
        "step_type": "thinking_explicit",
        "otel": {
            "step_kind": "llm_call",
            "token_usage": {
                "input_tokens": inp,
                "output_tokens": out,
                "cache_read_tokens": cache_read,
                "cache_write_tokens": cache_write,
                "total_tokens": inp + out,
                "cost_reason": "unknown_model",
            },
        },
    }


def _tool_step(index, inp, out):
    """工具执行步骤：enricher 按字符估了个数，但模型没产出过它。"""
    return {
        "step_index": index,
        "step_type": "tool_execution",
        "otel": {
            "step_kind": "host_tool",
            "token_usage": {
                "input_tokens": inp,
                "output_tokens": out,
                "total_tokens": inp + out,
                "cost_usd": None,
                "cost_reason": "non_llm_step",
            },
        },
    }


_ZERO_USAGE = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}


def test_hook_truth_wins_when_it_exists():
    turn = {
        "usage": {"input_tokens": 5000, "output_tokens": 900},
        "steps": [_llm_step(1, 10, 20)],
    }
    usage = _extract_turn_usage(turn, turn)
    assert usage["input_tokens"] == 5000
    assert usage["output_tokens"] == 900


def test_llm_steps_are_summed_when_the_turn_has_no_hook_truth():
    """而不是挑其中最大的一个。"""
    turn = {
        "usage": dict(_ZERO_USAGE),
        "steps": [_llm_step(1, 100, 200), _llm_step(2, 300, 900), _llm_step(3, 50, 78)],
    }
    usage = _extract_turn_usage(turn, turn)
    assert usage["input_tokens"] == 450
    assert usage["output_tokens"] == 1178
    assert usage["total_tokens"] == 1628


def test_the_biggest_single_step_is_not_mistaken_for_the_whole_turn():
    """回归守卫：这正是面板显示 0/1178 而卡片显示 9K/33.7K 的原因。"""
    turn = {
        "usage": dict(_ZERO_USAGE),
        "steps": [_llm_step(i, 24, 143) for i in range(1, 182)],
    }
    usage = _extract_turn_usage(turn, turn)
    assert usage["output_tokens"] == 181 * 143
    assert usage["output_tokens"] != 143, "挑了单个 step 就说明口径又退回去了"


def test_tool_steps_do_not_inflate_the_turn():
    """non_llm_step 的 token 是工具入参/结果的字符估算，不是模型用量。"""
    turn = {
        "usage": dict(_ZERO_USAGE),
        "steps": [
            _llm_step(1, 4404, 26017),
            _tool_step(2, 4435, 7683),
            _tool_step(3, 223, 0),
        ],
    }
    usage = _extract_turn_usage(turn, turn)
    assert usage["input_tokens"] == 4404
    assert usage["output_tokens"] == 26017


def test_cache_tokens_are_summed_too():
    turn = {
        "usage": dict(_ZERO_USAGE),
        "steps": [_llm_step(1, 10, 20, cache_read=7, cache_write=3),
                  _llm_step(2, 10, 20, cache_read=5, cache_write=1)],
    }
    usage = _extract_turn_usage(turn, turn)
    assert usage["cache_read_tokens"] == 12
    assert usage["cache_write_tokens"] == 4
    assert usage["total_tokens"] == 20 + 40 + 12 + 4


def test_turn_with_no_step_usage_at_all_still_falls_back():
    """老数据没经过 enricher，兜底路径要保留，聊胜于无。"""
    turn = {"usage": dict(_ZERO_USAGE), "steps": [{"step_index": 1}]}
    payload = {"usage": {"input_tokens": 11, "output_tokens": 22, "total_tokens": 33}}
    usage = _extract_turn_usage(turn, payload)
    assert usage["total_tokens"] == 33


def test_a_turn_with_zero_everything_reports_zero_not_garbage():
    turn = {"usage": dict(_ZERO_USAGE), "steps": []}
    usage = _extract_turn_usage(turn, turn)
    assert usage["total_tokens"] == 0
    assert usage["input_tokens"] == 0
