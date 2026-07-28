from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from agent_quality_eval.evaluation.reference_eval import preview_reference_upload


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COT = Path.home() / ".agent-cot" / "data" / "cot"
OUTPUT = ROOT / "agent-regression-gold-dataset"


CASES: list[dict[str, Any]] = [
    {
        "id": "case-01-ide-benchmark-strategy",
        "session_id": "030ce5ad-1b06-4eee-bad9-3bab70aea0f2",
        "turn_index": 19,
        "agent_type": "claude",
        "complexity": "simple",
        "gold_missingness_level": 6,
        "gold_missingness": "critical-final-answer-only",
        "question": "如何用约 50 条相同任务比较 Cursor、Claude、Codex、CodeBuddy？Provider 能否自动分发到这些 IDE？",
        "expected_answer": (
            "Provider 不能直接把测试集分发到 GUI IDE。Provider 面向 question→answer 的 HTTP API，"
            "而 IDE 编程任务的真值是 workspace diff、测试和命令结果。应把任务升级为包含初始 workspace、"
            "prompt 与 verifier 的任务卡，在干净 workspace 中分别执行，采集 trace，并按通过率、稳定性、"
            "工具次数、token 和耗时聚合。"
        ),
        "rubric": "必须否定直接分发，解释 API 与 IDE/workspace 的差异，并给出任务卡、trace、verifier 路径。",
        "keywords": ["Provider", "不能", "HTTP API", "workspace", "verifier", "trace"],
        "assertions": [
            {"type": "keywords", "value": ["不能", "Provider", "workspace", "verifier"], "all_required": True},
            {"type": "not-contains", "value": "可以自动分发到四个 IDE"},
        ],
        "process_requirements": {},
    },
    {
        "id": "case-02-opencli-vs-browser-skill",
        "session_id": "426f9d3b-9913-44b8-8bb0-e04c2b1c6740",
        "turn_index": 3,
        "agent_type": "cursor",
        "complexity": "simple",
        "gold_missingness_level": 5,
        "gold_missingness": "very-high-answer-with-casual-intro",
        "question": "OpenCLI 和浏览器 skill 有什么区别？MVP 和规模化阶段分别应该怎样选择？",
        "expected_answer": (
            "OpenCLI 是稳定、结构化、可重复的站点采集 CLI；浏览器 skill 是通过 CDP 动态探索和补救的 AI 操作层。"
            "MVP 应以 OpenCLI 为主、skill 兜底。规模化后两者都不应成为线上实时核心数据源，应离线采集并写入"
            "结构化数据库，用户请求只查询数据库。"
        ),
        "rubric": "区分 CLI 适配器与 AI 浏览器操作；给出 MVP 主辅组合；规模化采用离线采集和数据库。",
        "keywords": ["OpenCLI", "skill", "CDP", "MVP", "离线", "数据库", "兜底"],
        "assertions": [],
        "process_requirements": {},
    },
    {
        "id": "case-03-hook-restart-guidance",
        "session_id": "codex-019ef798-b92f-7eb2-b787-f4b08b24bbb7",
        "turn_index": 5,
        "agent_type": "codex",
        "complexity": "simple",
        "gold_missingness_level": 4,
        "gold_missingness": "high-titled-answer-note",
        "question": "安装 agent-quality-eval / agent-cot hook 后需要重启 IDE 吗？",
        "expected_answer": (
            "需要，建议重启对应 IDE。hook 配置、runtime 路径和脚本资产通常在 IDE 进程启动时加载。"
            "推荐关闭 IDE，安装并重新 init hook，重启 IDE，再开启新会话验证 trace/critic 是否生成。"
        ),
        "rubric": "必须明确建议重启，并给出关闭、安装/init、重启、新会话验证的顺序。",
        "keywords": ["重启", "IDE", "hook", "runtime", "新会话", "验证"],
        "assertions": [{"type": "not-contains", "value": "完全不需要重启"}],
        "process_requirements": {},
    },
    {
        "id": "case-04-ab-evaluation-method",
        "session_id": "codebuddy-46d404cd26524755885527dc186d0530",
        "turn_index": 3,
        "agent_type": "codebuddy",
        "complexity": "simple",
        "gold_missingness_level": 3,
        "gold_missingness": "medium-question-and-answer-note",
        "question": "Skill A/B 是否全程 AI 自迭代？base 可互换是否消除了偏见？指标和盲化方案应该如何设计？",
        "expected_answer": (
            "该流程是 human-in-the-loop：AI 运行、评分和分析，人类审阅、反馈并决定停止。base/candidate 可互换只减少"
            "身份偏见，不能消除位置偏见和随机性。指标应分为少量 primary 决策指标与 diagnostic 指标。"
            "deterministic diff 应作为 primary，盲化或双向 LLM 比较只用于边界 case；Regression 保留确定性 gate。"
        ),
        "rubric": "必须解释人类参与、三类偏见、指标分层，并把 deterministic 放在 primary。",
        "keywords": ["human-in-the-loop", "deterministic", "primary", "位置偏见", "regression"],
        "assertions": [{"type": "not-contains", "value": "完全 AI 自迭代"}],
        "process_requirements": {},
    },
    {
        "id": "case-05-iwiki-report-rewrite",
        "session_id": "030ce5ad-1b06-4eee-bad9-3bab70aea0f2",
        "turn_index": 1,
        "agent_type": "claude",
        "complexity": "complex",
        "gold_missingness_level": 2,
        "gold_missingness": "low-question-answer-and-rough-expectation",
        "question": "读取一份内部 agent eval 技术报告，重写为只保留核心架构、实现路径和典型工程问题的新报告并保存。",
        "expected_answer": (
            "应先读取源文档并核对项目结构，再写入目标文档。新报告覆盖本地优先的 IDE adapter→hook runtime→critic"
            "→前端/ingest 架构，live pulse 与 turn final critic 的实现路径，以及 frozen runner、stale placeholder、"
            "JSON 解析、hook evidence 归因等典型工程问题；不得复制逐版本 changelog，最终确认保存成功。"
        ),
        "rubric": "必须读取源文档、核对代码结构、按架构/路径/工程问题重写，并通过 iWiki 保存。",
        "keywords": ["架构", "hook", "critic", "live", "Regression", "Gold"],
        "assertions": [],
        "process_requirements": {
            "required_tools": ["mcp__iWiki__getDocument", "mcp__iWiki__saveDocument"],
            "must_not_include": ["未读取源文档就声称完成"],
        },
    },
    {
        "id": "case-06-product-brainstorm",
        "session_id": "3239f078-7456-42e0-8aa9-5d624020c479",
        "turn_index": 1,
        "agent_type": "cursor",
        "complexity": "complex",
        "gold_missingness_level": 1,
        "gold_missingness": "light-natural-acceptance-note",
        "question": "为 solo developer 头脑风暴一个两周可做、可开源或订阅、能写入简历的产品，并通过交互和调研收敛方向。",
        "expected_answer": (
            "应先提问确认用户、形态和周期约束，再用 WebSearch 验证候选方向的竞品，否定过度拥挤方向，最后收敛到"
            "一个主推荐。推荐 SkillLab：测试 SKILL.md 的触发、漏触发和误触发，支持正负样本、批量运行、版本对比、"
            "报告导出和本地优先。给出两周 MVP、差异化、简历表述和下一步，并说明有竞品能验证需求。"
        ),
        "rubric": "必须有交互澄清、竞品调研、单一主推荐、两周 MVP 和简历价值，不能只列空泛点子。",
        "keywords": ["SkillLab", "SKILL.md", "触发", "误触发", "MVP", "竞品", "两周"],
        "assertions": [{"type": "not-contains", "value": "完全没有竞品"}],
        "process_requirements": {
            "required_tools": ["AskQuestion", "WebSearch"],
            "must_not_include": ["未做调研就声称没有竞品"],
        },
    },
]


def sanitize(value: str) -> str:
    text = re.sub(r"(?i)https?://[^\s)>\]]+", "[URL]", value)
    text = re.sub(r"(?i)[A-Z]:\\Users\\[^\\\s]+", r"C:\\Users\\[USER]", text)
    text = re.sub(r"(?i)/Users/[^/\s]+", "/Users/[USER]", text)
    text = re.sub(r"(?i)(token|api[_-]?key|authorization)(\s*[:=]\s*)[^\s,;]+", r"\1\2[REDACTED]", text)
    return text


def find_turn(cot: dict[str, Any], turn_index: int) -> dict[str, Any]:
    turns = cot.get("turns") if isinstance(cot.get("turns"), list) else []
    for turn in turns:
        if isinstance(turn, dict) and int(turn.get("turn_index") or 0) == turn_index:
            return turn
    raise KeyError(f"turn {turn_index} not found")


def minimal_trace(case: dict[str, Any]) -> dict[str, Any]:
    source = SOURCE_COT / f"{case['session_id']}_cot.json"
    cot = json.loads(source.read_text(encoding="utf-8"))
    turn = find_turn(cot, int(case["turn_index"]))
    steps = []
    for index, step in enumerate(turn.get("steps") or [], start=1):
        if not isinstance(step, dict):
            continue
        metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        tool_name = str(step.get("tool_name") or metadata.get("tool_name") or metadata.get("name") or "")
        content = sanitize(str(step.get("content") or ""))
        steps.append(
            {
                "step_index": int(step.get("step_index") or index),
                "turn_index": int(case["turn_index"]),
                "step_type": str(step.get("step_type") or "unknown"),
                "tool_name": tool_name,
                "content_excerpt": content[:500],
            }
        )
    return {
        "schema_version": "aqe-regression-trace-v1",
        "provenance": {
            "source": "local observed trace",
            "source_session_id": case["session_id"],
            "source_turn_index": case["turn_index"],
            "agent_type": case["agent_type"],
            "complexity": case["complexity"],
            "minimized": True,
        },
        "session_id": f"dataset-{case['id']}",
        "agent_type": case["agent_type"],
        "turns": [
            {
                "turn_index": case["turn_index"],
                "user_query": sanitize(str(turn.get("user_query") or case["question"])),
                "final_response": sanitize(str(turn.get("final_response") or "")),
                "steps": steps,
                "total_steps": len(steps),
            }
        ],
    }


def raw_gold(case: dict[str, Any]) -> tuple[str, str]:
    level = int(case["gold_missingness_level"])
    answer = case["expected_answer"]
    question = case["question"]
    rubric = case["rubric"]

    if level == 6:
        # Worst missingness: this is literally all the new user has.
        return "gold.raw.txt", answer
    if level == 5:
        return "gold.raw.txt", f"这是我手头整理出的参考结果，可能不完整：\n\n{answer}"
    if level == 4:
        title = question.rstrip("？?。").split("，", 1)[0]
        return "gold.raw.md", f"# {title}\n\n{answer}\n"
    if level == 3:
        return (
            "gold.raw.md",
            f"# 我保存的一份回答\n\n用户当时大概问的是：{question}\n\n{answer}\n",
        )
    if level == 2:
        return (
            "gold.raw.txt",
            f"用户的问题大概是：{question}\n\n"
            f"我认为比较合理的回答是：\n{answer}\n\n"
            f"我主要会看这一点：{rubric}",
        )
    return (
        "gold.raw.md",
        f"# 我对这次任务的预期\n\n"
        f"用户想解决的问题\n\n{question}\n\n"
        f"我认为合理的结果\n\n{answer}\n\n"
        f"验收时我主要看\n\n{rubric}\n",
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_cases = []
    for case in CASES:
        case_dir = OUTPUT / case["id"]
        trace = minimal_trace(case)
        raw_name, raw_content = raw_gold(case)
        shutil.rmtree(case_dir / "raw", ignore_errors=True)
        raw_path = case_dir / "raw" / raw_name
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw_content, encoding="utf-8")

        preview = preview_reference_upload(raw_name, raw_content, issue_token=False)
        canonical = preview["canonical"]
        write_json(case_dir / "canonical" / "gold.canonical.json", canonical)
        write_json(case_dir / "normalization-report.json", preview["normalization_report"])
        write_json(case_dir / "trace.json", trace)

        negative = json.loads(json.dumps(trace, ensure_ascii=False))
        negative["synthetic_control"] = {
            "type": "negative",
            "purpose": "Regression gate validation only; not a real agent output.",
        }
        negative["turns"][0]["final_response"] = (
            "[SYNTHETIC NEGATIVE CONTROL] Required outcome and process evidence intentionally omitted."
        )
        negative["turns"][0]["steps"] = [
            step
            for step in negative["turns"][0]["steps"]
            if step.get("step_type") in {"user_input", "final_response"}
        ]
        negative["turns"][0]["total_steps"] = len(negative["turns"][0]["steps"])
        write_json(case_dir / "negative-control" / "trace.json", negative)

        manifest_cases.append(
            {
                "case_id": case["id"],
                "agent_type": case["agent_type"],
                "complexity": case["complexity"],
                "gold_missingness_level": case["gold_missingness_level"],
                "gold_missingness": case["gold_missingness"],
                "source_session_id": case["session_id"],
                "source_turn_index": case["turn_index"],
                "trace": f"{case['id']}/trace.json",
                "raw_gold": f"{case['id']}/raw/{raw_name}",
                "canonical_gold": f"{case['id']}/canonical/gold.canonical.json",
                "normalization_report": f"{case['id']}/normalization-report.json",
                "negative_control": f"{case['id']}/negative-control/trace.json",
                "canonical_hash": preview["normalization_report"]["canonical_hash"],
            }
        )

    manifest = {
        "schema_version": "aqe-regression-gold-suite-v1",
        "name": "AQE real-trace Gold normalization and regression suite",
        "case_count": len(manifest_cases),
        "source_policy": "Minimized copies of local observed traces; raw Gold files simulate incomplete novice uploads.",
        "cases": manifest_cases,
    }
    write_json(OUTPUT / "manifest.json", manifest)
    (OUTPUT / "README.md").write_text(
        """# AQE Regression Gold Dataset

This suite contains six one-to-one Trace/Gold cases selected from real local
observations: four simple turns and two multi-step tool turns across Cursor,
Claude, Codex, and CodeBuddy.

Each case contains:

- `trace.json`: minimized observed trace.
- `raw/`: simulated incomplete Gold written naturally by a new user.
- `canonical/gold.canonical.json`: deterministic normalized Gold.
- `normalization-report.json`: source-to-canonical field mapping and warnings.
- `negative-control/trace.json`: explicitly synthetic degraded candidate used
  only to prove that the regression gate detects a loss.

The six raw uploads are ordered from missingness level 6 (only a final answer)
to level 1 (question, answer, and a rough natural-language acceptance note).
Even level 1 has no internal IDs, keyword list, assertions, field mapping, or
structured process requirements. The manifest binds every trace to one Gold
and records its missingness level.

Run `scripts/build_regression_gold_dataset.py` with `PYTHONPATH=src` to rebuild
the suite from the local source traces.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
