#!/usr/bin/env python3
"""
CoT Reporter — 生成 CoT 报告（Markdown + JSON）

将 SessionCoT 对象转换为人类可读的 Markdown 报告和机器可读的 JSON 报告。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from cot_extractor import SessionCoT, TurnCoT, ThoughtStep, StepType


# ─── 步骤类型的显示配置 ───────────────────────────────────

STEP_ICONS = {
    StepType.USER_INPUT:        "💬",
    StepType.TOOL_RESULT_INPUT: "📥",
    StepType.THINKING_INTER:    "🧠",
    StepType.THINKING_EXPLICIT: "💭",
    StepType.TOOL_DECISION:     "🔧",
    StepType.TOOL_EXECUTION:    "⚡",
    StepType.STRATEGY_SHIFT:    "🔄",
    StepType.ERROR_RECOVERY:    "🚨",
    StepType.FINAL_RESPONSE:    "✅",
}

STEP_LABELS = {
    StepType.USER_INPUT:        "用户输入",
    StepType.TOOL_RESULT_INPUT: "工具结果输入",
    StepType.THINKING_INTER:    "中间思考",
    StepType.THINKING_EXPLICIT: "显式思考（Extended Thinking）",
    StepType.TOOL_DECISION:     "工具调用决策",
    StepType.TOOL_EXECUTION:    "工具执行结果",
    StepType.STRATEGY_SHIFT:    "策略转换",
    StepType.ERROR_RECOVERY:    "错误恢复",
    StepType.FINAL_RESPONSE:    "最终回复",
}

COMPLEXITY_LABELS = {
    (0.0, 1.0):  "🟢 极简",
    (1.0, 2.0):  "🟡 简单",
    (2.0, 4.0):  "🟠 中等",
    (4.0, 7.0):  "🔴 复杂",
    (7.0, 999.): "🔥 高度复杂",
}


def _complexity_label(score: float) -> str:
    for (lo, hi), label in COMPLEXITY_LABELS.items():
        if lo <= score < hi:
            return label
    return "🔥 高度复杂"


def _truncate(text: str, max_len: int = 300) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"...(共{len(text)}字符)"


# ─── Markdown 报告生成 ────────────────────────────────────

def generate_markdown_report(session_cot: SessionCoT) -> str:
    lines = []

    # 标题
    lines += [
        "# CoT Report — Claude 思维链分析报告",
        "",
        f"- **Session ID**: `{session_cot.session_id}`",
        f"- **Transcript**: `{session_cot.transcript_path}`",
        f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **总 Turn 数**: {len(session_cot.turns)}",
        "",
        "---",
        "",
    ]

    # Session 汇总
    lines += [
        "## 📊 Session 汇总",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| 总工具调用次数 | **{session_cot.total_tool_calls}** |",
        f"| 总策略转换次数 | **{session_cot.total_strategy_shifts}** |",
        f"| 总思考步骤数 | **{session_cot.total_thinking_steps}** |",
        f"| 平均每 Turn 步骤数 | **{session_cot.avg_steps_per_turn}** |",
        f"| 平均任务复杂度 | **{session_cot.avg_complexity}** |",
        "",
    ]

    # 工具调用分布
    if session_cot.tool_call_distribution:
        lines += ["### 🔧 工具调用分布", ""]
        sorted_tools = sorted(
            session_cot.tool_call_distribution.items(),
            key=lambda x: x[1], reverse=True
        )
        lines += ["| 工具 | 调用次数 |", "|------|---------|"]
        for tool, count in sorted_tools:
            lines.append(f"| `{tool}` | {count} |")
        lines.append("")

    lines += ["---", ""]

    # 每个 Turn 的详细 CoT
    for turn in session_cot.turns:
        lines += _generate_turn_section(turn)

    return "\n".join(lines)


def _generate_turn_section(turn: TurnCoT) -> list:
    lines = []

    # Turn 标题
    query_preview = _truncate(turn.user_query, 80) if turn.user_query else "（工具结果输入）"
    complexity_label = _complexity_label(turn.complexity_score)
    lines += [
        f"## Turn {turn.turn_index}: \"{query_preview}\"",
        "",
        f"**任务复杂度**: {complexity_label} (score={turn.complexity_score})",
        "",
    ]

    # Turn 统计
    lines += [
        "### 📈 Turn 统计",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| 总步骤数 | {turn.total_steps} |",
        f"| 工具调用 | {len(turn.tool_calls)} 次 ({', '.join(f'`{t}`' for t in turn.tool_calls) if turn.tool_calls else '无'}) |",
        f"| 策略转换 | {turn.strategy_shifts} 次 |",
        f"| 思考深度 | {turn.thinking_depth} 层 |",
        f"| 错误恢复 | {'是' if turn.has_error_recovery else '否'} |",
        f"| Input Tokens | {turn.usage.get('input_tokens', 0)} |",
        f"| Output Tokens | {turn.usage.get('output_tokens', 0)} |",
        f"| Cache Read | {turn.usage.get('cache_read_input_tokens', 0)} |",
        "",
    ]

    # 思维链步骤
    if turn.steps:
        lines += ["### 🧠 思维链步骤", ""]
        for step in turn.steps:
            icon = STEP_ICONS.get(step.step_type, "•")
            label = STEP_LABELS.get(step.step_type, step.step_type)

            # 步骤标题行
            if step.step_type == StepType.TOOL_DECISION and step.tool_name:
                header = f"**Step {step.step_index}** {icon} `{label}` → **{step.tool_name}**"
            elif step.step_type == StepType.TOOL_EXECUTION and step.tool_use_id:
                # 找对应的工具名
                header = f"**Step {step.step_index}** {icon} `{label}`"
                if step.metadata.get("is_error"):
                    header += " ⚠️ **[执行出错]**"
            else:
                header = f"**Step {step.step_index}** {icon} `{label}`"

            lines.append(header)
            lines.append("")

            # 步骤内容
            content = _truncate(step.content, 500)
            if step.step_type in (StepType.TOOL_DECISION,):
                # 工具调用：显示工具输入
                tool_input = step.metadata.get("tool_input", {})
                if tool_input:
                    input_str = json.dumps(tool_input, ensure_ascii=False, indent=2)
                    if len(input_str) > 400:
                        input_str = input_str[:400] + "\n...(截断)"
                    lines.append(f"```json\n{input_str}\n```")
                else:
                    lines.append(f"> {content}")
            elif step.step_type in (StepType.THINKING_INTER, StepType.THINKING_EXPLICIT):
                # 思考内容：用引用块显示
                lines.append(f"```\n{_truncate(step.content, 800)}\n```")
            elif step.step_type == StepType.FINAL_RESPONSE:
                # 最终回复：显示前 500 字符
                lines.append(_truncate(step.content, 500))
            elif step.step_type == StepType.TOOL_EXECUTION:
                # 工具结果
                result_len = step.metadata.get("result_len", len(step.content))
                truncated = step.metadata.get("truncated", False)
                lines.append(f"> {_truncate(step.content, 300)}")
                if truncated:
                    lines.append(f"> *(原始结果 {result_len} 字符，已截断)*")
            elif step.step_type == StepType.STRATEGY_SHIFT:
                from_tool = step.metadata.get("from_tool", "?")
                to_tool = step.metadata.get("to_tool", "?")
                lines.append(f"> ⚠️ 检测到策略转换：`{from_tool}` → `{to_tool}`")
            elif step.step_type == StepType.ERROR_RECOVERY:
                lines.append(f"> 🚨 {_truncate(step.content, 200)}")
            else:
                lines.append(f"> {content}")

            # 摘要式推理（如果有）
            if hasattr(step, 'reasoning_digest') and step.reasoning_digest:
                rd = step.reasoning_digest
                lines.append("")
                lines.append(f"  > 💡 **理由**: {rd.why}")
                lines.append(f"  > 📋 **证据**: {rd.evidence}")
                lines.append(f"  > ⚖️ **依据**: {rd.basis}")
                lines.append(f"  > ➡️ **下一步**: {rd.next_plan}")

            lines.append("")

    # 最终回复预览
    if turn.final_response:
        lines += [
            "### 💬 最终回复",
            "",
            _truncate(turn.final_response, 600),
            "",
        ]

    lines += ["---", ""]
    return lines


# ─── 报告保存 ─────────────────────────────────────────────

def save_reports(
    session_cot: SessionCoT,
    output_dir: Path,
) -> dict:
    """
    保存 CoT 报告到指定目录。

    Returns:
        {"md": str, "json": str}  文件路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = session_cot.session_id

    # JSON 报告
    json_path = output_dir / f"{session_id}_cot.json"
    json_path.write_text(
        json.dumps(session_cot.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Markdown 报告
    md_path = output_dir / f"{session_id}_cot.md"
    md_content = generate_markdown_report(session_cot)
    md_path.write_text(md_content, encoding="utf-8")

    return {"md": str(md_path), "json": str(json_path)}
