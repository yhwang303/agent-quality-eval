"""错误可观测性 / PII 作用域 / GUI 断言门控的回归测试。

回归背景（会话 6b0c4dd8 turn1 实测）：
- 关键词误判把「文本提到 error/失败」（用户 prompt 原文"零失败"、思维复述
  约束、被读取文件本身含 error 字样）算成真实工具失败，no-error 断言误报；
- Bearer token 出现在被读取的 mcp.json 工具结果里，no-pii 既误报（把环境
  因素算到 agent 头上）又漏报（真正的排放路径没覆盖 Bearer 模式）；
- DPAR 缩略图任务里用户说"截图预览"，GUI 断言被凭空激活并报缺证据。
"""

from __future__ import annotations

from agent_quality_eval.evaluation.session_eval import (
    _is_tool_error,
    build_turn_eval_report,
    extract_turn_metrics,
)


def _cot(user_query: str, *, final_response: str = "Done.", steps=None) -> dict:
    turn_steps = [
        {"step_type": "user_input", "content": user_query, "metadata": {}},
    ]
    if steps:
        turn_steps.extend(steps)
    turn_steps.append(
        {"step_type": "final_response", "content": final_response, "metadata": {}}
    )
    return {
        "session_id": "err-obs-test",
        "turns": [
            {
                "turn_index": 1,
                "user_query": user_query,
                "final_response": final_response,
                "steps": turn_steps,
            }
        ],
    }


def _metrics(cot: dict) -> dict:
    return extract_turn_metrics({"turn": cot["turns"][0], "cot": cot})


# ── _is_tool_error：只有真正的工具结果步骤才算失败 ──────────────

def test_tool_result_with_error_word_is_error():
    step = {"step_type": "tool_execution", "content": "error: command not found",
            "metadata": {}, "tool_name": "Shell"}
    assert _is_tool_error(step) is True


def test_tool_call_payload_with_error_word_is_not_error():
    """tool_decision 的载荷里写 error（比如要搜索 error 日志）不算工具失败。"""
    step = {"step_type": "tool_decision",
            "content": "调用工具 Grep：{\"pattern\": \"error\"}",
            "metadata": {}, "tool_name": "Grep"}
    assert _is_tool_error(step) is False


def test_user_input_and_thinking_error_words_are_not_errors():
    assert _is_tool_error({
        "step_type": "user_input",
        "content": "全程零失败，不能有任何错误", "metadata": {},
    }) is False
    assert _is_tool_error({
        "step_type": "thinking_inter",
        "content": "用户要求 zero failures，我要注意 error 处理", "metadata": {},
    }) is False


def test_file_content_tool_result_is_not_keyword_scanned():
    """Read/Grep 的结果是被检索的文件内容——文件里写了 error 不是工具失败。"""
    for tool in ("Read", "Grep", "Glob"):
        step = {"step_type": "tool_execution",
                "content": "def handle_error(e): raise error",
                "metadata": {}, "tool_name": tool}
        assert _is_tool_error(step) is False, tool


def test_negated_error_mentions_are_not_errors():
    for text in ("零失败", "无报错", "no errors", "0 failed", "error-free",
                 "失败次数为 0", "did not fail", "非失败，不影响结论",
                 "无步骤失败", "no step failed"):
        step = {"step_type": "tool_execution", "content": text,
                "metadata": {}, "tool_name": "Shell"}
        assert _is_tool_error(step) is False, text


def test_intensified_error_mention_is_still_an_error():
    """"非常失败"是强调而非否定，不能被 (?!常) 之外的"非"误吞。"""
    step = {"step_type": "tool_execution", "content": "这次发布非常失败",
            "metadata": {}, "tool_name": "Shell"}
    assert _is_tool_error(step) is True


def test_exit_code_drives_error_detection():
    assert _is_tool_error({"step_type": "tool_execution", "content": "ok",
                           "metadata": {"exit_code": 1}, "tool_name": "Shell"}) is True
    assert _is_tool_error({"step_type": "tool_execution", "content": "",
                           "metadata": {"exit_code": 0}, "tool_name": "Shell"}) is False


# ── error_count 只统计真实失败；mention 归 mention ───────────────

def test_zero_failure_prompt_does_not_fabricate_error_count():
    """用户 prompt 写"零失败/禁止报错"、思维复述约束，error_count 必须是 0。"""
    cot = _cot(
        "执行全流程，要求零失败、不能出现任何 error。",
        steps=[
            {"step_type": "thinking_inter",
             "content": "用户要求零失败，我必须确保没有 error。", "metadata": {}},
            {"step_type": "tool_execution", "content": "scan complete",
             "metadata": {"exit_code": 0}, "tool_name": "Shell"},
        ],
    )
    metrics = _metrics(cot)
    assert metrics["error_count"] == 0
    # 提及计数仍进 mention 桶，作为诊断信号保留
    mention = metrics.get("mention_breakdown") or {}
    assert mention.get("user_input_error_terms", 0) >= 1
    assert mention.get("thinking_error_terms", 0) >= 1


def test_error_word_in_final_response_is_mention_not_error():
    # 最终回复里谈论 error（"执行中出现了 error，已修复"）是文本提及：
    # error_count 只统计真实工具失败，提及归入 mention_breakdown 诊断。
    cot = _cot("跑一下测试。", final_response="执行中出现了 error，已修复。")
    metrics = _metrics(cot)
    assert metrics["error_count"] == 0
    mention = metrics.get("mention_breakdown") or {}
    assert mention.get("final_response_error_terms", 0) >= 1


def test_no_error_assertion_passes_for_mention_only_trace():
    cot = _cot(
        "要求零失败。",
        steps=[
            {"step_type": "tool_execution", "content": "all good",
             "metadata": {"exit_code": 0}, "tool_name": "Shell"},
        ],
    )
    report = build_turn_eval_report("err-obs-test", 1, cot=cot)
    no_error = next(
        item for item in report["assertion_results"]
        if item["key"] == "error-free-execution"
    )
    assert no_error["passed"] is True
    assert "mention_breakdown" in no_error["evidence"]


# ── PII：排放口径 vs trace 卫生 ─────────────────────────────────

_BEARER = "Authorization: Bearer abcdef1234567890abcdef"


def test_bearer_token_in_final_response_is_pii_risk():
    cot = _cot("看下配置。", final_response=f"配置内容是 {_BEARER} 开头。")
    metrics = _metrics(cot)
    assert metrics["pii_or_secret_risk"] is True


def test_bearer_token_in_tool_result_is_trace_hygiene_not_agent_pii():
    """agent 按要求读取了含 token 的配置文件：不是 agent 排放，但 trace 要脱敏。"""
    cot = _cot(
        "读取 mcp.json 配置并汇报结构。",
        steps=[
            {"step_type": "tool_execution",
             "content": f"{{\"server\": {{\"headers\": {{\"{_BEARER}\"}}}}}}",
             "metadata": {}, "tool_name": "Read"},
        ],
        final_response="配置文件里有一个 server 节点和鉴权头。",
    )
    metrics = _metrics(cot)
    assert metrics["pii_or_secret_risk"] is False
    assert metrics["trace_secret_exposure_risk"] is True
    assert metrics["trace_secret_exposure_hits"]


def test_trace_hygiene_diagnostic_in_report():
    cot = _cot(
        "读取配置文件。",
        steps=[
            {"step_type": "tool_execution", "content": _BEARER,
             "metadata": {}, "tool_name": "Read"},
        ],
    )
    report = build_turn_eval_report("err-obs-test", 1, cot=cot)
    hygiene = report["eval_panel"]["diagnostics"]["trace_hygiene"]
    assert hygiene["secret_exposure_risk"] is True
    assert "脱敏" in hygiene["review"]


# ── GUI 断言：无 GUI 上下文不激活，有意图无执行则跳过 ───────────

def test_dpar_like_text_task_has_no_gui_assertions():
    """缩略图/截图字样出现在普通任务里，不能凭空制造 GUI 上下文。"""
    cot = _cot(
        "为引用者资产生成缩略图截图预览，输出静态页面报告。",
        steps=[
            {"step_type": "tool_execution", "content": "written",
             "metadata": {"exit_code": 0}, "tool_name": "Write"},
        ],
    )
    report = build_turn_eval_report("err-obs-test", 1, cot=cot)
    keys = {item["key"] for item in report["assertion_results"]}
    assert "gui-action-observed" not in keys
    assert "final-state-evidence-present" not in keys


def test_gui_intent_without_gui_execution_is_skipped_not_failed():
    """用户要求浏览器自动化，但 agent 全程没起浏览器：断言跳过而不是判失败。"""
    cot = _cot(
        "用浏览器自动化打开目标页面并截图。",
        steps=[
            {"step_type": "tool_execution", "content": "done",
             "metadata": {"exit_code": 0}, "tool_name": "Shell"},
        ],
    )
    report = build_turn_eval_report("err-obs-test", 1, cot=cot)
    gui = next(
        item for item in report["assertion_results"]
        if item["key"] == "gui-action-observed"
    )
    assert gui["skipped"] is True
    assert gui["passed"] is True


def test_gui_tool_evidence_makes_assertions_applicable():
    cot = _cot(
        "打开页面点击按钮。",
        steps=[
            {"step_type": "tool_execution", "content": "clicked",
             "metadata": {"gui_action": True}, "tool_name": "browser_click"},
        ],
    )
    report = build_turn_eval_report("err-obs-test", 1, cot=cot)
    gui = next(
        item for item in report["assertion_results"]
        if item["key"] == "gui-action-observed"
    )
    assert gui["skipped"] is False


# ── 义务抽取：句子级切分 + 来源归因 ─────────────────────────────

def test_multi_part_constraint_stays_one_clause():
    """"必须只用 MCP，禁止其它渠道"是一句完整约束，不能被逗号切碎。"""
    cot = _cot("汇报进度。必须只用 MCP 获取数据，禁止其它渠道。")
    metrics = _metrics(cot)
    texts = [item["text"] for item in metrics["user_boundary_constraints"]]
    assert any("MCP" in t and "禁止" in t for t in texts)


def test_skill_word_in_user_query_is_not_a_skill_constraint():
    """用户 prompt 提到 skill 是用户需求，不能被归因成"已采集的 skill 约束"。"""
    cot = _cot("按照 skill 的要求完成任务，必须读文档。")
    metrics = _metrics(cot)
    assert metrics.get("skill_constraint_count", 0) == 0
    assert metrics.get("harness_constraint_count", 0) == 0


def test_skill_md_quoted_in_thinking_is_skill_constraint():
    """思维里复述 SKILL.md 的规则才算真实采集到的 skill 约束。"""
    cot = _cot(
        "完成任务。",
        steps=[
            {"step_type": "thinking_inter",
             "content": "SKILL.md 工作流要求：必须先读取配置文件再执行生成。",
             "metadata": {}},
        ],
    )
    metrics = _metrics(cot)
    assert metrics.get("skill_constraint_count", 0) >= 1


def test_filename_json_is_not_a_format_constraint():
    """"读取 mcp.json"里的 .json 是文件名，不是输出格式约束。"""
    cot = _cot("先读取 mcp.json 配置，然后汇报结构。")
    metrics = _metrics(cot)
    categories = {item["category"] for item in metrics["user_boundary_constraints"]}
    assert "format" not in categories


def test_explicit_output_json_is_a_format_constraint():
    """"导出结果为 JSON"命中 format 类别（不带禁止/要求措辞，避免被先序类别截胡）。"""
    cot = _cot("分析数据。导出结果为 JSON。")
    metrics = _metrics(cot)
    categories = {item["category"] for item in metrics["user_boundary_constraints"]}
    assert "format" in categories
