"""Reference-answer datasets and optional turn evaluation."""

from __future__ import annotations

import difflib
import hashlib
import html
import json
import re
import time
from pathlib import Path
from typing import Any

from .assertions import run_assertions
from .models import PerformanceMetrics, utc_now
from .store import default_home


SUPPORTED_EXTENSIONS = {".json", ".yaml", ".yml", ".md", ".markdown", ".txt"}
_REFERENCE_LLM_CACHE: dict[str, dict[str, Any]] = {}


def reference_eval_home() -> Path:
    root = default_home() / "reference-evals"
    root.mkdir(parents=True, exist_ok=True)
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    (root / "turn-answers").mkdir(parents=True, exist_ok=True)
    return root


def turn_reference_answer_path(session_id: str, turn_index: int) -> Path:
    safe_session = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(session_id or "unknown"))[:180]
    return reference_eval_home() / "turn-answers" / safe_session / f"turn_{int(turn_index)}.json"


def save_turn_reference_answer(session_id: str, turn_index: int, filename: str, content: str) -> dict[str, Any]:
    parsed = parse_reference_dataset(filename, content)
    case = _select_first_reference_case(parsed)
    payload = {
        "schema_version": "turn-reference-answer-v1",
        "session_id": session_id,
        "turn_index": int(turn_index),
        "created_at": utc_now(),
        "source_filename": filename,
        "dataset": dataset_summary(parsed),
        "case": case,
        "reference_answer": _reference_case_payload(case, parsed),
    }
    path = turn_reference_answer_path(session_id, turn_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    payload["storage_path"] = str(path.resolve())
    return payload


def load_turn_reference_answer(session_id: str, turn_index: int) -> dict[str, Any] | None:
    path = turn_reference_answer_path(session_id, turn_index)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    data["storage_path"] = str(path.resolve())
    return data


def delete_turn_reference_answer(session_id: str, turn_index: int) -> bool:
    path = turn_reference_answer_path(session_id, turn_index)
    if not path.is_file():
        return False
    path.unlink()
    return True


def normalize_reference_upload(filename: str, content: str) -> dict[str, Any]:
    parsed = parse_reference_dataset(filename, content)
    case = _select_first_reference_case(parsed)
    return {
        "dataset": dataset_summary(parsed),
        "case": case,
        "reference_answer": _reference_case_payload(case, parsed),
    }


def upload_reference_dataset(filename: str, content: str) -> dict[str, Any]:
    parsed = parse_reference_dataset(filename, content)
    dataset_id = parsed["dataset_id"]
    target = reference_eval_home() / "datasets" / dataset_id
    target.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower() or ".txt"
    (target / f"original{suffix}").write_text(content, encoding="utf-8")
    (target / "normalized.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dataset_summary(parsed, dataset_dir=target)


def list_reference_datasets() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted((reference_eval_home() / "datasets").iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_dir():
            continue
        data = _read_dataset(path.name)
        if data:
            out.append(dataset_summary(data, dataset_dir=path))
    return out


def get_reference_dataset(dataset_id: str) -> dict[str, Any]:
    data = _read_dataset(dataset_id)
    if not data:
        raise KeyError(f"reference dataset not found: {dataset_id}")
    data["storage_dir"] = str((reference_eval_home() / "datasets" / dataset_id).resolve())
    return data


def _select_first_reference_case(parsed: dict[str, Any]) -> dict[str, Any]:
    cases = parsed.get("cases") if isinstance(parsed.get("cases"), list) else []
    if not cases:
        raise ValueError("reference answer must contain at least one case")
    first = cases[0]
    if not isinstance(first, dict):
        raise ValueError("reference answer case is invalid")
    return first


def _reference_case_payload(case: dict[str, Any], dataset: dict[str, Any] | None = None) -> dict[str, Any]:
    expected = str(case.get("expected_answer") or "")
    return {
        "case_id": case.get("id"),
        "title": case.get("title"),
        "question": case.get("question"),
        "standard_answer": expected,
        "expected_answer": expected,
        "original_expected_answer": case.get("original_expected_answer") or expected,
        "rubric": case.get("rubric"),
        "keywords": case.get("keywords") if isinstance(case.get("keywords"), list) else [],
        "assertions": case.get("assertions") if isinstance(case.get("assertions"), list) else [],
        "process_requirements": case.get("process_requirements") if isinstance(case.get("process_requirements"), dict) else {},
        "dataset_name": (dataset or {}).get("name"),
        "dataset_version": (dataset or {}).get("version"),
        "normalization": case.get("normalization") or {"mode": "schema_only"},
    }


def parse_reference_dataset(filename: str, content: str) -> dict[str, Any]:
    ext = Path(filename).suffix.lower()
    if ext and ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported reference dataset format: {ext}")
    raw = _load_raw_reference_content(ext, content)
    normalized = _normalize_raw_dataset(filename, raw)
    if not normalized["cases"]:
        raise ValueError("reference dataset must contain at least one case")
    material = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    content_hash = hashlib.sha256(material.encode("utf-8", errors="replace")).hexdigest()[:12]
    name = normalized["name"]
    version = normalized["version"]
    normalized["content_hash"] = content_hash
    normalized["dataset_id"] = _slug(f"{name}-{version}-{content_hash[:8]}")
    normalized["source_filename"] = filename
    normalized["uploaded_at"] = utc_now()
    return normalized


def evaluate_turn_against_reference(
    *,
    session_id: str,
    turn_index: int,
    dataset_id: str,
    case_id: str | None,
    turn_eval: dict[str, Any],
    turn_context: dict[str, Any],
) -> dict[str, Any]:
    dataset = get_reference_dataset(dataset_id)
    case = _select_case(dataset.get("cases", []), case_id, turn_eval, turn_context)
    if not case:
        raise KeyError("reference case not found")
    answer = _turn_final_answer(turn_eval, turn_context)
    expected = str(case.get("expected_answer") or "")
    metrics = turn_eval.get("metrics") if isinstance(turn_eval.get("metrics"), dict) else {}
    performance = PerformanceMetrics.from_dict(
        {
            "total_duration": (float(metrics.get("duration_ms") or 0.0) / 1000.0),
            "token_usage": {
                "input_tokens": metrics.get("input_tokens") or 0,
                "output_tokens": metrics.get("output_tokens") or 0,
                "total_tokens": metrics.get("total_tokens") or 0,
            },
            "cost": metrics.get("cost_usd"),
        }
    )
    deterministic = _deterministic_reference_eval(
        question=_turn_user_query(turn_eval, turn_context),
        answer=answer,
        expected_answer=expected,
        case=case,
        performance=performance,
        trace=turn_context.get("turn") if isinstance(turn_context.get("turn"), dict) else {},
    )
    llm_compare = _run_reference_llm_compare(
        case=case,
        answer=answer,
        turn_eval=turn_eval,
        turn_context=turn_context,
        deterministic=deterministic,
    )
    final_score = deterministic["overall_score"]
    if llm_compare.get("status") == "completed" and isinstance(llm_compare.get("score"), (int, float)):
        final_score = round((final_score * 0.65) + (float(llm_compare["score"]) * 0.35), 4)
    verdict = _reference_verdict(final_score, deterministic, llm_compare)
    result = {
        "run_id": _short_hash(f"{session_id}:{turn_index}:{dataset_id}:{case.get('id')}:{time.time()}"),
        "created_at": utc_now(),
        "session_id": session_id,
        "turn_index": turn_index,
        "dataset": dataset_summary(dataset, dataset_dir=reference_eval_home() / "datasets" / dataset_id),
        "case": case,
        "matched_case_id": case.get("id"),
        "match_confidence": case.get("_match_confidence"),
        "final_score": final_score,
        "verdict": verdict,
        "deterministic": deterministic,
        "llm_compare": llm_compare,
        "trace_metrics": _trace_metrics_summary(metrics, turn_context),
        "answer_excerpt": _compact(answer, 1600),
        "expected_excerpt": _compact(expected, 1600),
    }
    artifact_path = _write_reference_run_artifacts(result)
    result["artifact_path"] = str(artifact_path)
    return result


def dataset_summary(data: dict[str, Any], *, dataset_dir: Path | None = None) -> dict[str, Any]:
    cases = data.get("cases") if isinstance(data.get("cases"), list) else []
    with_expected = sum(1 for item in cases if str(item.get("expected_answer") or "").strip())
    with_rubric = sum(1 for item in cases if str(item.get("rubric") or "").strip())
    with_assertions = sum(1 for item in cases if item.get("assertions"))
    with_process_requirements = sum(1 for item in cases if item.get("process_requirements"))
    return {
        "dataset_id": data.get("dataset_id"),
        "name": data.get("name"),
        "version": data.get("version"),
        "description": data.get("description"),
        "source_filename": data.get("source_filename"),
        "uploaded_at": data.get("uploaded_at"),
        "case_count": len(cases),
        "with_expected_answer": with_expected,
        "with_rubric": with_rubric,
        "with_assertions": with_assertions,
        "with_process_requirements": with_process_requirements,
        "content_hash": data.get("content_hash"),
        "storage_dir": str(dataset_dir.resolve()) if dataset_dir else data.get("storage_dir"),
    }


def _load_raw_reference_content(ext: str, content: str) -> Any:
    if ext == ".json":
        return json.loads(content)
    if ext in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(content)
    if ext in {".md", ".markdown", ".txt", ""}:
        return _parse_markdown_reference(content)
    return content


def _normalize_raw_dataset(filename: str, raw: Any) -> dict[str, Any]:
    if isinstance(raw, list):
        raw = {"name": Path(filename).stem, "version": "v1", "cases": raw}
    if not isinstance(raw, dict):
        raw = {"name": Path(filename).stem, "version": "v1", "cases": [{"id": "case-1", "expected_answer": str(raw or "")}]}
    cases_raw = raw.get("cases", raw.get("tests", raw.get("items", [])))
    if not cases_raw and _looks_like_single_reference_case(raw):
        cases_raw = [raw]
    if isinstance(cases_raw, dict):
        cases_raw = [cases_raw]
    cases: list[dict[str, Any]] = []
    for index, item in enumerate(cases_raw or [], start=1):
        case = _normalize_case(item, index)
        if case:
            cases.append(case)
    return {
        "name": str(raw.get("name") or raw.get("id") or Path(filename).stem or "reference-dataset"),
        "version": str(raw.get("version") or "v1"),
        "description": str(raw.get("description") or ""),
        "metadata": raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
        "cases": cases,
    }


def _normalize_case(item: Any, index: int) -> dict[str, Any] | None:
    if isinstance(item, str):
        item = {"id": f"case-{index}", "expected_answer": item}
    if not isinstance(item, dict):
        return None
    expected = (
        item.get("expected_answer")
        or item.get("reference_answer")
        or item.get("gold_answer")
        or item.get("standard_answer")
        or item.get("reference")
        or item.get("answer")
        or item.get("expected")
        or item.get("output")
        or ""
    )
    assertions = item.get("assertions", item.get("assert", [])) or []
    if isinstance(assertions, dict):
        assertions = [assertions]
    keywords = item.get("keywords") or item.get("key_points") or item.get("expected_keywords") or []
    if isinstance(keywords, str):
        keywords = [part.strip() for part in re.split(r"[,;\n]", keywords) if part.strip()]
    process_requirements = _normalize_process_requirements(item)
    return {
        "id": str(item.get("id") or item.get("name") or f"case-{index}"),
        "title": str(item.get("title") or item.get("name") or item.get("id") or f"Case {index}"),
        "question": str(item.get("question") or item.get("input") or item.get("prompt") or ""),
        "expected_answer": str(expected or ""),
        "original_expected_answer": str(expected or ""),
        "rubric": str(item.get("rubric") or item.get("criteria") or ""),
        "keywords": [str(x) for x in keywords if str(x).strip()],
        "assertions": list(assertions) if isinstance(assertions, list) else [],
        "process_requirements": process_requirements,
        "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
        "normalization": {
            "mode": "schema_only",
            "note": "Input was normalized into eval fields without changing the expected answer semantics.",
        },
    }


def _normalize_process_requirements(item: dict[str, Any]) -> dict[str, Any]:
    buckets: dict[str, list[str]] = {
        "must_include": [],
        "must_not_include": [],
        "required_tools": [],
        "forbidden_tools": [],
        "steps": [],
    }

    def add(bucket: str, value: Any) -> None:
        for text in _requirement_texts(value):
            if text and text not in buckets[bucket]:
                buckets[bucket].append(text)

    def visit(value: Any, *, default_bucket: str = "must_include") -> None:
        if isinstance(value, dict):
            key_map = {
                "must_include": "must_include",
                "include": "must_include",
                "required": "must_include",
                "must": "must_include",
                "must_not_include": "must_not_include",
                "must_not": "must_not_include",
                "forbidden": "must_not_include",
                "avoid": "must_not_include",
                "required_tools": "required_tools",
                "tools": "required_tools",
                "tool_names": "required_tools",
                "must_use_tools": "required_tools",
                "forbidden_tools": "forbidden_tools",
                "must_not_use_tools": "forbidden_tools",
                "steps": "steps",
                "expected_steps": "steps",
                "workflow": "steps",
            }
            consumed = False
            for key, raw in value.items():
                normalized_key = str(key or "").strip().lower().replace("-", "_").replace(" ", "_")
                bucket = key_map.get(normalized_key)
                if bucket:
                    add(bucket, raw)
                    consumed = True
                elif normalized_key in {"process_requirements", "trace_expectations"}:
                    visit(raw, default_bucket=default_bucket)
                    consumed = True
            if not consumed:
                add(default_bucket, value)
            return
        add(default_bucket, value)

    for key in ("process_requirements", "trace_expectations"):
        if key in item:
            visit(item.get(key))
    for key, bucket in (
        ("steps", "steps"),
        ("expected_steps", "steps"),
        ("must_include", "must_include"),
        ("required", "must_include"),
        ("required_tools", "required_tools"),
        ("tools", "required_tools"),
        ("must_not_include", "must_not_include"),
        ("forbidden", "must_not_include"),
        ("forbidden_tools", "forbidden_tools"),
    ):
        if key in item:
            add(bucket, item.get(key))

    out = {key: values for key, values in buckets.items() if values}
    if out:
        out["schema_version"] = "process-requirements-v1"
    return out


def _requirement_texts(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(_requirement_texts(item))
        return out
    if isinstance(value, dict):
        preferred = (
            value.get("text")
            or value.get("description")
            or value.get("requirement")
            or value.get("expectation")
            or value.get("name")
            or value.get("tool")
            or value.get("tool_name")
        )
        if preferred:
            return _requirement_texts(preferred)
        return [
            f"{key}: {val}"
            for key, val in value.items()
            if str(key).strip() and str(val).strip()
        ]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"[,;\n]", text) if part.strip()]


def _looks_like_single_reference_case(raw: dict[str, Any]) -> bool:
    answer_keys = {
        "expected_answer",
        "reference_answer",
        "gold_answer",
        "standard_answer",
        "reference",
        "answer",
        "expected",
        "output",
    }
    prompt_keys = {"question", "input", "prompt"}
    rubric_keys = {
        "rubric",
        "criteria",
        "keywords",
        "key_points",
        "expected_keywords",
        "assertions",
        "assert",
        "process_requirements",
        "trace_expectations",
        "steps",
        "must_include",
        "required_tools",
    }
    return bool((answer_keys | prompt_keys | rubric_keys).intersection(raw.keys()))


def _parse_markdown_reference(content: str) -> dict[str, Any]:
    blocks = _markdown_case_blocks(content)
    cases = []
    if blocks:
        for idx, (title, body) in enumerate(blocks, start=1):
            sections = _markdown_sections(body)
            cases.append(
                {
                    "id": _slug(title) or f"case-{idx}",
                    "title": title,
                    "question": sections.get("question") or sections.get("input") or sections.get("prompt") or "",
                    "expected_answer": sections.get("expected") or sections.get("expected_answer") or sections.get("answer") or body.strip(),
                    "rubric": sections.get("rubric") or sections.get("criteria") or "",
                    "keywords": sections.get("keywords") or "",
                }
            )
    else:
        cases.append({"id": "case-1", "title": "Reference Answer", "expected_answer": content.strip()})
    return {"name": "reference-dataset", "version": "v1", "cases": cases}


def _markdown_case_blocks(content: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)^#{1,3}\s+(.+?)\s*$", content))
    if not matches:
        return []
    out = []
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        body = content[start:end].strip()
        if body:
            out.append((title, body))
    return out


def _markdown_sections(body: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = "expected"
    aliases = {
        "question": "question",
        "input": "input",
        "prompt": "prompt",
        "expected": "expected",
        "expected answer": "expected",
        "reference": "expected",
        "answer": "answer",
        "rubric": "rubric",
        "criteria": "criteria",
        "keywords": "keywords",
    }
    for line in body.splitlines():
        stripped = line.strip()
        label_match = re.match(r"^(?:#{2,5}\s*)?([A-Za-z ]{3,24}|问题|标准答案|答案|评分标准|关键词)\s*[:：]\s*(.*)$", stripped)
        if label_match:
            raw_label = label_match.group(1).strip().lower()
            mapped = aliases.get(raw_label)
            if raw_label == "问题":
                mapped = "question"
            elif raw_label in {"标准答案", "答案"}:
                mapped = "expected"
            elif raw_label == "评分标准":
                mapped = "rubric"
            elif raw_label == "关键词":
                mapped = "keywords"
            if mapped:
                current = mapped
                rest = label_match.group(2).strip()
                sections.setdefault(current, [])
                if rest:
                    sections[current].append(rest)
                continue
        sections.setdefault(current, []).append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items()}


def _select_case(cases: list[dict[str, Any]], case_id: str | None, turn_eval: dict[str, Any], turn_context: dict[str, Any]) -> dict[str, Any] | None:
    if case_id:
        for item in cases:
            if str(item.get("id")) == str(case_id):
                return dict(item)
        return None
    query = _turn_user_query(turn_eval, turn_context)
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in cases:
        question = str(item.get("question") or item.get("title") or "")
        score = _text_similarity(query, question) if question.strip() else 0.0
        scored.append((score, item))
    scored.sort(key=lambda row: row[0], reverse=True)
    if not scored:
        return None
    selected = dict(scored[0][1])
    selected["_match_confidence"] = round(scored[0][0], 4)
    return selected


def _deterministic_reference_eval(
    *,
    question: str,
    answer: str,
    expected_answer: str,
    case: dict[str, Any],
    performance: PerformanceMetrics,
    trace: dict[str, Any],
) -> dict[str, Any]:
    seq = difflib.SequenceMatcher(None, _normalize_text(answer), _normalize_text(expected_answer)).ratio() if expected_answer else 0.0
    token_overlap = _text_similarity(answer, expected_answer) if expected_answer else 0.0
    keywords = list(case.get("keywords") or [])
    keyword_score, missing_keywords = _keyword_coverage(answer, keywords)
    assertion_results = [
        item.to_dict()
        for item in run_assertions(
            answer,
            case.get("assertions") or [],
            question=question,
            expected_answer=expected_answer,
            response_time=performance.total_duration,
            performance=performance,
            trace=trace,
            llm_judge_func=None,
        )
    ]
    assertion_score = (
        sum(float(item.get("score") or 0.0) for item in assertion_results) / len(assertion_results)
        if assertion_results
        else 1.0
    )
    has_keywords = bool(keywords)
    if has_keywords:
        overall = (seq * 0.25) + (token_overlap * 0.30) + (keyword_score * 0.30) + (assertion_score * 0.15)
    else:
        overall = (seq * 0.40) + (token_overlap * 0.45) + (assertion_score * 0.15)
    process_eval = _evaluate_process_requirements(case.get("process_requirements"), trace)
    if process_eval.get("applicable"):
        overall = (overall * 0.80) + (float(process_eval.get("score") or 0.0) * 0.20)
    return {
        "overall_score": round(max(0.0, min(1.0, overall)), 4),
        "sequence_similarity": round(seq, 4),
        "token_overlap": round(token_overlap, 4),
        "keyword_coverage": round(keyword_score, 4),
        "missing_keywords": missing_keywords,
        "assertion_score": round(assertion_score, 4),
        "assertion_results": assertion_results,
        "process_requirements": process_eval,
        "method": "sequence_similarity + token_overlap + keyword_coverage + deterministic_assertions",
    }


def _evaluate_process_requirements(requirements: Any, trace: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(requirements, dict) or not any(requirements.get(key) for key in ("must_include", "must_not_include", "required_tools", "forbidden_tools", "steps")):
        return {"applicable": False, "score": 1.0, "results": []}
    trace_text, tool_text = _trace_requirement_text(trace)
    results: list[dict[str, Any]] = []

    def check(kind: str, values: Any, haystack: str, should_exist: bool) -> None:
        for value in values if isinstance(values, list) else []:
            needle = str(value or "").strip()
            if not needle:
                continue
            found = _requirement_found(needle, haystack)
            passed = found if should_exist else not found
            results.append(
                {
                    "kind": kind,
                    "requirement": needle,
                    "passed": passed,
                    "evidence": "found" if found else "not_found",
                }
            )

    check("must_include", requirements.get("must_include"), trace_text, True)
    check("step", requirements.get("steps"), trace_text, True)
    check("required_tool", requirements.get("required_tools"), tool_text + "\n" + trace_text, True)
    check("must_not_include", requirements.get("must_not_include"), trace_text, False)
    check("forbidden_tool", requirements.get("forbidden_tools"), tool_text + "\n" + trace_text, False)
    passed = sum(1 for item in results if item["passed"])
    score = passed / len(results) if results else 1.0
    return {
        "applicable": True,
        "score": round(score, 4),
        "passed": passed,
        "total": len(results),
        "results": results,
    }


def _trace_requirement_text(trace: dict[str, Any]) -> tuple[str, str]:
    parts: list[str] = []
    tools: list[str] = []
    if isinstance(trace, dict):
        parts.append(str(trace.get("user_query") or ""))
        parts.append(str(trace.get("final_response") or trace.get("assistant_response") or trace.get("response") or ""))
        for step in trace.get("steps") or []:
            if not isinstance(step, dict):
                continue
            metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
            tool = metadata.get("tool_name") or metadata.get("name") or step.get("tool_name")
            if tool:
                tools.append(str(tool))
            parts.extend(
                str(value or "")
                for value in (
                    step.get("step_type"),
                    step.get("type"),
                    step.get("content"),
                    tool,
                    metadata.get("observed_input"),
                    metadata.get("observed_output"),
                )
            )
    return "\n".join(parts).lower(), "\n".join(tools).lower()


def _requirement_found(needle: str, haystack: str) -> bool:
    expected = str(needle or "").strip().lower()
    if not expected:
        return True
    if expected in haystack:
        return True
    tokens = [token for token in _tokens(expected) if len(token) > 1]
    return bool(tokens) and all(token in haystack for token in tokens)


def _run_reference_llm_compare(
    *,
    case: dict[str, Any],
    answer: str,
    turn_eval: dict[str, Any],
    turn_context: dict[str, Any],
    deterministic: dict[str, Any],
) -> dict[str, Any]:
    try:
        from .providers import load_provider
        from .settings import load_critic_settings
    except Exception as exc:
        return _reference_llm_fallback("unavailable", f"critic provider unavailable: {type(exc).__name__}: {exc}")
    settings = load_critic_settings()
    if not settings.enabled:
        return _reference_llm_fallback("disabled", "Critic model is disabled.")
    provider_config = settings.to_provider_config()
    if not provider_config:
        return _reference_llm_fallback("unconfigured", "Critic model is not configured.")
    prompt = _build_reference_llm_prompt(case, answer, turn_eval, turn_context, deterministic)
    cache_key = _short_hash(json.dumps({"provider": settings.provider, "model": settings.model, "prompt": prompt}, ensure_ascii=False, sort_keys=True))
    if cache_key in _REFERENCE_LLM_CACHE:
        cached = dict(_REFERENCE_LLM_CACHE[cache_key])
        cached["cache_hit"] = True
        return cached
    try:
        provider = load_provider(provider_config)
        started = time.time()
        response = provider.call(prompt)
        latency_ms = int((time.time() - started) * 1000)
        if response.error:
            return _reference_llm_fallback("error", response.error, provider=settings.provider, model=settings.model, latency_ms=latency_ms)
        parsed = _extract_json_object(response.output)
        if not isinstance(parsed, dict):
            data = _reference_llm_fallback("error", "Critic model did not return valid JSON.", provider=settings.provider, model=settings.model, latency_ms=latency_ms)
            data["raw_output"] = str(response.output or "")[:2000]
            return data
        result = _normalize_reference_llm_result(parsed)
        result.update(
            {
                "provider": settings.provider,
                "model": settings.model,
                "latency_ms": latency_ms,
                "token_usage": response.performance.token_usage if response.performance else {},
                "cache_hit": False,
            }
        )
        _REFERENCE_LLM_CACHE[cache_key] = dict(result)
        return result
    except Exception as exc:
        return _reference_llm_fallback("error", f"{type(exc).__name__}: {exc}", provider=settings.provider, model=settings.model)


def _build_reference_llm_prompt(
    case: dict[str, Any],
    answer: str,
    turn_eval: dict[str, Any],
    turn_context: dict[str, Any],
    deterministic: dict[str, Any],
) -> str:
    schema = {
        "verdict": "pass | partial | fail",
        "score": "0.0-1.0",
        "summary_conclusion": "中文，必须以'标准答案评测结论：'开头。",
        "semantic_equivalence": {"verdict": "equivalent | partial | different", "review": "中文"},
        "missing_key_points": ["中文数组"],
        "extra_or_wrong_claims": ["中文数组"],
        "reasoning_and_trace": {"verdict": "solid | weak | risky | unclear", "review": "中文"},
        "token_and_efficiency": {"verdict": "reasonable | high | excessive | unclear", "review": "中文"},
        "manual_review_notes": ["中文数组"],
    }
    metrics = turn_eval.get("metrics") if isinstance(turn_eval.get("metrics"), dict) else {}
    turn = turn_context.get("turn") if isinstance(turn_context.get("turn"), dict) else {}
    trace_steps = []
    for idx, step in enumerate(turn.get("steps") or [], start=1):
        if not isinstance(step, dict) or len(trace_steps) >= 28:
            continue
        metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        trace_steps.append(
            {
                "idx": idx,
                "type": step.get("step_type") or step.get("type"),
                "tool": metadata.get("tool_name") or metadata.get("name") or step.get("tool_name"),
                "is_error": bool(metadata.get("is_error")),
                "content": _compact(step.get("content"), 500),
            }
        )
    payload = {
        "case": {
            "id": case.get("id"),
            "question": case.get("question"),
            "expected_answer": _compact(case.get("expected_answer"), 4000),
            "rubric": case.get("rubric"),
            "keywords": case.get("keywords"),
            "process_requirements": case.get("process_requirements") or {},
        },
        "actual_answer": _compact(answer, 4000),
        "deterministic_scores": deterministic,
        "trace": {
            "metrics": {
                "total_tokens": metrics.get("total_tokens"),
                "input_tokens": metrics.get("input_tokens"),
                "output_tokens": metrics.get("output_tokens"),
                "duration_ms": metrics.get("duration_ms"),
                "tool_count": metrics.get("tool_count"),
                "tool_error_count": metrics.get("tool_error_count"),
            },
            "steps": trace_steps,
            "eval_panel": turn_eval.get("eval_panel"),
            "critical_failures": turn_eval.get("critical_failures"),
        },
    }
    return (
        "你是一名 Agent 标准答案评测员。请判断 actual_answer 是否满足 reference case 的标准答案与评分标准。\n"
        "重点比较语义等价、关键点覆盖、额外错误声称，并结合 trace 中的工具、推理路径、token、耗时说明执行质量。\n"
        "只返回 JSON，不要 Markdown，不要代码块。所有自然语言字段必须使用中文。\n"
        f"输出 schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        f"评测输入:\n{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}"
    )


def _normalize_reference_llm_result(data: dict[str, Any]) -> dict[str, Any]:
    verdict = str(data.get("verdict") or "partial").lower()
    if verdict not in {"pass", "partial", "fail"}:
        verdict = "partial"
    try:
        score = float(data.get("score"))
    except (TypeError, ValueError):
        score = {"pass": 0.85, "partial": 0.55, "fail": 0.25}[verdict]
    score = max(0.0, min(1.0, score))
    out = {
        "status": "completed",
        "verdict": verdict,
        "score": round(score, 4),
        "summary_conclusion": str(data.get("summary_conclusion") or "").strip()[:1400],
        "missing_key_points": _text_list(data.get("missing_key_points")),
        "extra_or_wrong_claims": _text_list(data.get("extra_or_wrong_claims")),
        "manual_review_notes": _text_list(data.get("manual_review_notes")),
        "created_at": utc_now(),
    }
    if not out["summary_conclusion"]:
        out["summary_conclusion"] = f"标准答案评测结论：{verdict}。LLM 评审未给出完整结论，请结合确定性相似度复核。"
    for key in ("semantic_equivalence", "reasoning_and_trace", "token_and_efficiency"):
        value = data.get(key) if isinstance(data.get(key), dict) else {}
        out[key] = {
            "verdict": str(value.get("verdict") or "unclear")[:80],
            "review": str(value.get("review") or value.get("reason") or "")[:1400],
        }
    return out


def _reference_llm_fallback(
    status: str,
    reason: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    latency_ms: int | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "verdict": "partial",
        "score": None,
        "summary_conclusion": f"标准答案评测结论：LLM 评审暂不可用，当前结果仅基于确定性相似度、关键词和断言。原因：{reason}",
        "semantic_equivalence": {"verdict": "unclear", "review": "LLM 评审未完成。"},
        "reasoning_and_trace": {"verdict": "unclear", "review": "LLM 评审未完成。"},
        "token_and_efficiency": {"verdict": "unclear", "review": "LLM 评审未完成。"},
        "missing_key_points": [],
        "extra_or_wrong_claims": [],
        "manual_review_notes": ["请优先查看确定性相似度、关键词覆盖和断言结果。"],
        "provider": provider,
        "model": model,
        "latency_ms": latency_ms,
        "reason": reason,
        "created_at": utc_now(),
        "cache_hit": False,
    }


def _reference_verdict(score: float, deterministic: dict[str, Any], llm_compare: dict[str, Any]) -> str:
    llm_verdict = str(llm_compare.get("verdict") or "").lower()
    if llm_verdict == "fail" and score < 0.68:
        return "fail"
    if score >= 0.78 and not deterministic.get("missing_keywords"):
        return "pass"
    if score >= 0.52:
        return "partial"
    return "fail"


def _write_reference_run_artifacts(result: dict[str, Any]) -> Path:
    run_dir = reference_eval_home() / "runs" / str(result["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "reference-eval.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path = run_dir / "reference-eval.html"
    html_path.write_text(_render_reference_html(result), encoding="utf-8")
    return json_path.resolve()


def _render_reference_html(result: dict[str, Any]) -> str:
    det = result.get("deterministic") or {}
    llm = result.get("llm_compare") or {}
    return f"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<title>Reference Eval {html.escape(str(result.get("run_id")))} </title>
<style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;background:#0f172a;color:#e5e7eb;margin:0;padding:32px}}
.wrap{{max-width:1080px;margin:auto}}.card{{background:#111827;border:1px solid #334155;border-radius:10px;padding:18px;margin:14px 0}}
h1,h2{{margin:0 0 10px}}.kpi{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.kpi div{{background:#0b1220;border:1px solid #263244;border-radius:8px;padding:12px}}span{{color:#94a3b8}}pre{{white-space:pre-wrap;word-break:break-word;background:#020617;border-radius:8px;padding:12px;color:#dbeafe}}
.pass{{color:#22c55e}}.partial{{color:#f59e0b}}.fail{{color:#ef4444}}
</style><div class="wrap">
<h1>Reference Eval <b class="{html.escape(str(result.get("verdict")))}">{html.escape(str(result.get("verdict")).upper())}</b></h1>
<div class="card kpi">
<div><span>Final Score</span><h2>{float(result.get("final_score") or 0):.1%}</h2></div>
<div><span>Similarity</span><h2>{float(det.get("token_overlap") or 0):.1%}</h2></div>
<div><span>Keyword Coverage</span><h2>{float(det.get("keyword_coverage") or 0):.1%}</h2></div>
<div><span>LLM Status</span><h2>{html.escape(str(llm.get("status") or "unknown"))}</h2></div>
</div>
<div class="card"><h2>Conclusion</h2><p>{html.escape(str(llm.get("summary_conclusion") or ""))}</p></div>
<div class="card"><h2>Expected Answer</h2><pre>{html.escape(str(result.get("expected_excerpt") or ""))}</pre></div>
<div class="card"><h2>Actual Answer</h2><pre>{html.escape(str(result.get("answer_excerpt") or ""))}</pre></div>
</div></html>"""


def _read_dataset(dataset_id: str) -> dict[str, Any] | None:
    path = reference_eval_home() / "datasets" / dataset_id / "normalized.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _turn_user_query(turn_eval: dict[str, Any], turn_context: dict[str, Any]) -> str:
    metrics = turn_eval.get("metrics") if isinstance(turn_eval.get("metrics"), dict) else {}
    turn = turn_context.get("turn") if isinstance(turn_context.get("turn"), dict) else {}
    return str(metrics.get("user_query") or turn.get("user_query") or turn.get("question") or "")


def _turn_final_answer(turn_eval: dict[str, Any], turn_context: dict[str, Any]) -> str:
    metrics = turn_eval.get("metrics") if isinstance(turn_eval.get("metrics"), dict) else {}
    turn = turn_context.get("turn") if isinstance(turn_context.get("turn"), dict) else {}
    candidates = (
        metrics.get("final_response"),
        metrics.get("assistant_response"),
        turn.get("final_response"),
        turn.get("assistant_response"),
        turn.get("response"),
    )
    for value in candidates:
        text = str(value or "").strip()
        if text:
            return text
    steps = turn.get("steps") if isinstance(turn.get("steps"), list) else []
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        if str(step.get("step_type") or step.get("type") or "") == "final_response":
            return str(step.get("content") or "")
    return ""


def _trace_metrics_summary(metrics: dict[str, Any], turn_context: dict[str, Any]) -> dict[str, Any]:
    turn = turn_context.get("turn") if isinstance(turn_context.get("turn"), dict) else {}
    return {
        "total_tokens": metrics.get("total_tokens"),
        "input_tokens": metrics.get("input_tokens"),
        "output_tokens": metrics.get("output_tokens"),
        "duration_ms": metrics.get("duration_ms") or turn.get("duration_ms"),
        "tool_count": metrics.get("tool_count"),
        "tool_error_count": metrics.get("tool_error_count"),
        "step_count": metrics.get("step_count") or len(turn.get("steps") or []),
        "assertion_pass_rate": metrics.get("assertion_pass_rate"),
    }


def _keyword_coverage(answer: str, keywords: list[str]) -> tuple[float, list[str]]:
    if not keywords:
        return 1.0, []
    lower = answer.lower()
    missing = [kw for kw in keywords if kw.lower() not in lower]
    return (len(keywords) - len(missing)) / len(keywords), missing


def _text_similarity(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[\w]+|[\u4e00-\u9fff]", text or "", flags=re.UNICODE) if token.strip()]


def _normalize_text(text: str) -> str:
    return " ".join(_tokens(text))


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _text_list(value: Any, *, limit: int = 10) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip()[:800] for item in value if str(item).strip()][:limit]


def _compact(value: Any, max_chars: int = 500) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _short_hash(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()[:12]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip().lower()).strip("-")
    return slug[:90] or "reference-dataset"
