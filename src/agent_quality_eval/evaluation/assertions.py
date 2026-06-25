"""Deterministic and LLM assertion scoring."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from .models import PerformanceMetrics, ScoreResult

JudgeFunc = Callable[[str, str, str, str | None], tuple[float, str]]


ERROR_KEYWORDS = [
    "error",
    "错误",
    "失败",
    "异常",
    "exception",
    "无法",
    "不能",
    "找不到",
    "not found",
    "404",
    "抱歉",
    "sorry",
    "出错了",
]

PII_PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone": r"(?<!\d)(?:\+?\d[\d -]{8,}\d)(?!\d)",
    "credit_card": r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)",
}


def run_assertions(
    answer: str,
    assertions: list[dict[str, Any] | str],
    *,
    question: str = "",
    expected_answer: str | None = None,
    response_time: float | None = None,
    performance: PerformanceMetrics | None = None,
    trace: dict[str, Any] | None = None,
    llm_judge_func: JudgeFunc | None = None,
) -> list[ScoreResult]:
    results = []
    for assertion in assertions:
        if isinstance(assertion, str):
            assertion = _parse_short_assertion(assertion)
        if _should_skip_assertion(assertion, llm_judge_func):
            continue
        results.append(
            run_single_assertion(
                assertion,
                answer=answer,
                question=question,
                expected_answer=expected_answer,
                response_time=response_time,
                performance=performance,
                trace=trace,
                llm_judge_func=llm_judge_func,
            )
        )
    return results


def _should_skip_assertion(assertion: dict[str, Any], llm_judge_func: JudgeFunc | None) -> bool:
    """Skip explicitly optional LLM assertions when no judge is configured."""
    atype = str(assertion.get("type", "")).lower().replace("_", "-")
    optional = bool(assertion.get("optional") or assertion.get("skip_without_judge"))
    return optional and llm_judge_func is None and atype in {"llm-rubric", "llm", "task-completion", "plan-quality", "plan-adherence"}


def _parse_short_assertion(assertion: str) -> dict[str, Any]:
    if ":" in assertion:
        atype, value = assertion.split(":", 1)
        return {"type": atype.strip(), "value": value.strip()}
    return {"type": assertion}


def run_single_assertion(
    assertion: dict[str, Any],
    *,
    answer: str,
    question: str,
    expected_answer: str | None,
    response_time: float | None,
    performance: PerformanceMetrics | None,
    trace: dict[str, Any] | None,
    llm_judge_func: JudgeFunc | None,
) -> ScoreResult:
    atype = str(assertion.get("type", "")).lower().replace("_", "-")
    value = assertion.get("value")
    threshold = float(assertion.get("threshold", 0.5))
    answer = answer or ""
    performance = performance or PerformanceMetrics()

    if atype == "contains":
        return _contains(answer, value, assertion.get("case_sensitive", False), threshold)
    if atype in {"not-contains", "notcontains"}:
        return _not_contains(answer, value, assertion.get("case_sensitive", False), threshold)
    if atype == "equals":
        return _binary(atype, answer.strip() == str(value).strip(), "回答完全等于期望值", "回答不等于期望值", threshold, value)
    if atype == "regex":
        return _regex(answer, str(value), threshold)
    if atype in {"starts-with", "startswith"}:
        return _binary("starts-with", answer.strip().startswith(str(value)), f"回答以 {value!r} 开头", f"回答未以 {value!r} 开头", threshold, value)
    if atype in {"ends-with", "endswith"}:
        return _binary("ends-with", answer.strip().endswith(str(value)), f"回答以 {value!r} 结尾", f"回答未以 {value!r} 结尾", threshold, value)
    if atype in {"contains-json", "containsjson"}:
        return _contains_json(answer, threshold)
    if atype in {"min-length", "minlength"}:
        min_len = int(value or 1)
        score = min(1.0, len(answer.strip()) / min_len) if min_len else 1.0
        return _score("min-length", score, threshold, f"回答长度 {len(answer.strip())}，最小要求 {min_len}", value)
    if atype in {"max-length", "maxlength"}:
        max_len = int(value or 10000)
        score = 1.0 if len(answer.strip()) <= max_len else max(0.0, max_len / max(1, len(answer.strip())))
        return _score("max-length", score, threshold, f"回答长度 {len(answer.strip())}，最大限制 {max_len}", value)
    if atype in {"non-empty", "nonempty"}:
        return _binary("non-empty", bool(answer.strip()), f"回答非空，长度 {len(answer)}", "回答为空", threshold, value)
    if atype in {"no-error", "noerror"}:
        found = [kw for kw in ERROR_KEYWORDS if kw in answer.lower()]
        return _binary("no-error", not found, "未检测到错误信息", f"检测到错误关键词: {found}", threshold, value)
    if atype in {"response-time", "responsetime"}:
        max_time = float(value or 60.0)
        if response_time is None:
            return _score("response-time", 1.0, threshold, "未记录响应时间，默认通过", value)
        score = 1.0 if response_time <= max_time else max(0.0, max_time / max(response_time, 0.001))
        return _score("response-time", score, threshold, f"响应时间 {response_time:.2f}s，限制 {max_time:.2f}s", value)
    if atype in {"keyword", "keywords"}:
        keywords = value if isinstance(value, list) else [value]
        return _keywords(answer, keywords, bool(assertion.get("all_required", True)), threshold)
    if atype == "python":
        return _python_assert(answer, str(value), question, response_time, threshold)
    if atype in {"contains-pii", "pii"}:
        return _contains_pii(answer, threshold)
    if atype in {"no-pii", "pii-leakage"}:
        pii = _find_pii(answer)
        return _binary("no-pii", not pii, "未检测到 PII", f"检测到 PII: {sorted(pii)}", threshold, pii)
    if atype in {"token-budget", "max-tokens"}:
        total = float(performance.token_usage.get("total_tokens", 0) or 0)
        limit = float(value or assertion.get("max", 0) or 0)
        score = 1.0 if not limit or total <= limit else max(0.0, limit / total)
        return _score("token-budget", score, threshold, f"Token {total:g}，预算 {limit:g}", value)
    if atype in {"cost-budget", "max-cost"}:
        cost = float(performance.cost or performance.token_usage.get("cost", 0) or 0)
        limit = float(value or assertion.get("max", 0) or 0)
        score = 1.0 if not limit or cost <= limit else max(0.0, limit / cost)
        return _score("cost-budget", score, threshold, f"成本 {cost:g}，预算 {limit:g}", value)
    if atype in {"tool-called", "tool-use", "tool-correctness"}:
        expected = value if isinstance(value, list) else [value]
        return _tool_called(expected, performance, trace, threshold)
    if atype in {"max-tool-calls", "step-efficiency"}:
        limit = int(value or assertion.get("max", 0) or 0)
        count = len(_extract_tools(performance, trace))
        score = 1.0 if not limit or count <= limit else max(0.0, limit / count)
        return _score(atype, score, threshold, f"工具调用 {count} 次，限制 {limit}", value, {"tool_count": count})
    if atype in {"llm-rubric", "llm", "task-completion", "plan-quality", "plan-adherence"}:
        criteria = str(value or assertion.get("criteria") or "回答是否正确、完整、相关")
        return _llm_rubric(atype, question, answer, criteria, threshold, expected_answer, llm_judge_func)

    return ScoreResult(
        name=atype or "unknown",
        type=atype or "unknown",
        score=0.0,
        passed=False,
        reason=f"未知断言类型: {atype}",
        threshold=threshold,
        metadata={"value": value},
    )


def _score(name: str, score: float, threshold: float, reason: str, value: Any = None, metadata: dict[str, Any] | None = None) -> ScoreResult:
    score = max(0.0, min(1.0, float(score)))
    md = dict(metadata or {})
    if value is not None:
        md["value"] = value
    return ScoreResult(name=name, type=name, score=score, passed=score >= threshold, reason=reason, threshold=threshold, metadata=md)


def _binary(name: str, passed: bool, yes: str, no: str, threshold: float, value: Any = None) -> ScoreResult:
    return _score(name, 1.0 if passed else 0.0, threshold, yes if passed else no, value)


def _contains(answer: str, value: Any, case_sensitive: bool, threshold: float) -> ScoreResult:
    haystack = answer if case_sensitive else answer.lower()
    needle = str(value) if case_sensitive else str(value).lower()
    return _binary("contains", needle in haystack, f"回答包含文本: {value!r}", f"回答不包含文本: {value!r}", threshold, value)


def _not_contains(answer: str, value: Any, case_sensitive: bool, threshold: float) -> ScoreResult:
    haystack = answer if case_sensitive else answer.lower()
    needle = str(value) if case_sensitive else str(value).lower()
    return _binary("not-contains", needle not in haystack, f"回答不包含文本: {value!r}", f"回答包含不应出现文本: {value!r}", threshold, value)


def _regex(answer: str, pattern: str, threshold: float) -> ScoreResult:
    try:
        matched = bool(re.search(pattern, answer, re.IGNORECASE))
        return _binary("regex", matched, f"回答匹配正则: {pattern}", f"回答不匹配正则: {pattern}", threshold, pattern)
    except re.error as exc:
        return _score("regex", 0.0, threshold, f"正则表达式错误: {exc}", pattern)


def _contains_json(answer: str, threshold: float) -> ScoreResult:
    candidates = []
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", answer, flags=re.IGNORECASE)
    candidates.extend(fenced)
    candidates.extend(re.findall(r"\{[\s\S]*?\}", answer))
    candidates.extend(re.findall(r"\[[\s\S]*?\]", answer))
    for candidate in candidates:
        try:
            json.loads(candidate.strip())
            return _score("contains-json", 1.0, threshold, "回答包含有效 JSON")
        except json.JSONDecodeError:
            continue
    return _score("contains-json", 0.0, threshold, "回答不包含有效 JSON")


def _keywords(answer: str, keywords: list[Any], all_required: bool, threshold: float) -> ScoreResult:
    words = [str(k) for k in keywords if k is not None]
    lower = answer.lower()
    found = [w for w in words if w.lower() in lower]
    if all_required:
        score = len(found) / len(words) if words else 1.0
        missing = [w for w in words if w not in found]
        reason = f"找到 {len(found)}/{len(words)} 个关键词" + (f"，缺失: {missing}" if missing else "")
    else:
        score = 1.0 if found else 0.0
        reason = f"找到关键词: {found}" if found else f"未找到关键词，期望: {words}"
    return _score("keyword", score, threshold, reason, words)


def _python_assert(answer: str, expression: str, question: str, response_time: float | None, threshold: float) -> ScoreResult:
    safe_builtins = {
        "len": len,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "round": round,
        "sorted": sorted,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "any": any,
        "all": all,
        "enumerate": enumerate,
        "range": range,
        "zip": zip,
        "isinstance": isinstance,
        "True": True,
        "False": False,
        "None": None,
    }
    try:
        value = eval(
            expression,
            {"__builtins__": safe_builtins},
            {"output": answer, "answer": answer, "question": question, "response_time": response_time},
        )
        return _score("python", 1.0 if value else 0.0, threshold, f"Python 表达式结果: {value}", expression)
    except Exception as exc:
        return _score("python", 0.0, threshold, f"Python 表达式执行失败: {exc}", expression)


def _find_pii(answer: str) -> set[str]:
    found = set()
    for label, pattern in PII_PATTERNS.items():
        if re.search(pattern, answer):
            found.add(label)
    return found


def _contains_pii(answer: str, threshold: float) -> ScoreResult:
    pii = _find_pii(answer)
    return _binary("contains-pii", bool(pii), f"检测到 PII: {sorted(pii)}", "未检测到 PII", threshold, sorted(pii))


def _extract_tools(performance: PerformanceMetrics, trace: dict[str, Any] | None) -> list[dict[str, Any]]:
    tools = list(performance.tool_calls or [])
    if trace:
        for key in ("tool_calls", "tools"):
            if isinstance(trace.get(key), list):
                tools.extend(trace[key])
        for turn in trace.get("turns", []) if isinstance(trace.get("turns"), list) else []:
            for key in ("tool_calls", "steps"):
                items = turn.get(key, []) if isinstance(turn, dict) else []
                for item in items:
                    if isinstance(item, dict) and (item.get("name") or item.get("tool") or item.get("tool_name")):
                        tools.append(item)
    return tools


def _tool_called(expected: list[Any], performance: PerformanceMetrics, trace: dict[str, Any] | None, threshold: float) -> ScoreResult:
    expected_names = [str(item) for item in expected if item is not None]
    tools = _extract_tools(performance, trace)
    names = {
        str(tool.get("name") or tool.get("tool") or tool.get("tool_name") or "").lower()
        for tool in tools
    }
    found = [name for name in expected_names if name.lower() in names]
    score = len(found) / len(expected_names) if expected_names else 1.0
    reason = f"工具命中 {len(found)}/{len(expected_names)}，实际工具: {sorted(names)}"
    return _score("tool-called", score, threshold, reason, expected_names, {"actual_tools": sorted(names)})


def _llm_rubric(
    name: str,
    question: str,
    answer: str,
    criteria: str,
    threshold: float,
    expected_answer: str | None,
    llm_judge_func: JudgeFunc | None,
) -> ScoreResult:
    if llm_judge_func is None:
        return _score(name, 0.0, threshold, "LLM judge 未配置，无法执行 LLM 断言", criteria)
    try:
        score, reason = llm_judge_func(question, answer, criteria, expected_answer)
        return _score(name, score, threshold, reason, criteria)
    except Exception as exc:
        return _score(name, 0.0, threshold, f"LLM judge 调用失败: {type(exc).__name__}: {exc}", criteria)
