"""Build a real-case eval dataset from CodeBuddy Memory (CBM).

The builder reads the user's CBM SQLite database in read-only mode, extracts
real development/debugging cases, redacts sensitive details, and writes:

- a YAML eval config that can be loaded by the existing eval pipeline
- a JSONL evidence file that preserves source row references and redacted proof
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CBM_DB = Path.home() / ".codebuddy-mem" / "codebuddy-mem.db"
DEFAULT_PROJECTS = {
    "agent-memory",
    "ai-ide-langfuse",
    "agent-quality-eval",
    "letsgoagenteval",
    "shadow-folk",
}
CASE_QUOTAS = {
    "bugfix": 10,
    "tool_trace": 8,
    "eval_ab": 5,
    "build_release": 5,
    "negative": 5,
    "research_or_diagnosis": 5,
}


@dataclass
class CbmCandidate:
    source_table: str
    source_row_id: int
    memory_session_id: str
    project: str
    category: str
    request: str
    investigated: str
    completed: str
    next_steps: str
    evidence: str
    created_at: str
    raw_text: str


def build_cbm_real_dataset(
    db_path: str | Path = DEFAULT_CBM_DB,
    *,
    yaml_path: str | Path,
    evidence_path: str | Path,
    max_cases: int = 40,
) -> dict[str, Any]:
    """Create the standard real-case eval dataset from a CBM database."""
    db_path = Path(db_path)
    yaml_path = Path(yaml_path)
    evidence_path = Path(evidence_path)
    candidates = _load_candidates(db_path)
    selected = _select_candidates(candidates, max_cases=max_cases)
    tests = [_candidate_to_test_case(item, idx + 1) for idx, item in enumerate(selected)]

    config = {
        "name": "cbm-real-dev-cases-v3",
        "providers": [
            {
                "type": "mock",
                "name": "replace-with-agent-under-test",
                "default_response": (
                    "This is a placeholder provider. Replace it with a real agent provider "
                    "before using this dataset for release gating."
                ),
            }
        ],
        "settings": {
            "num_trials": 1,
            "pass_threshold": 0.8,
            "output_dir": "./results",
        },
        "dataset": {
            "name": "cbm-real-dev-cases",
            "version": "v1",
            "description": "Redacted real development/debugging cases extracted from CodeBuddy Memory.",
        },
        "defaultAssertions": [
            {"type": "non-empty"},
            {"type": "no-error"},
            {"type": "no-pii"},
            {"type": "min-length", "value": 80},
        ],
        "tests": tests,
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    with evidence_path.open("w", encoding="utf-8") as fh:
        for test, item in zip(tests, selected):
            raw_hash = hashlib.sha256(item.raw_text.encode("utf-8", errors="ignore")).hexdigest()
            redacted = _redact(item.raw_text)
            redacted_hash = hashlib.sha256(redacted.encode("utf-8", errors="ignore")).hexdigest()
            fh.write(
                json.dumps(
                    {
                        "case_id": test["id"],
                        "source_table": item.source_table,
                        "source_row_id": item.source_row_id,
                        "memory_session_id": _redact_session_id(item.memory_session_id),
                        "source_project": _project_slug(item.project),
                        "category": item.category,
                        "raw_sha256": raw_hash,
                        "redacted_sha256": redacted_hash,
                        "selection_reason": _selection_reason(item),
                        "evidence": {
                            "request": _redact(item.request),
                            "investigated": _redact(item.investigated),
                            "completed": _redact(item.completed),
                            "next_steps": _redact(item.next_steps),
                            "supporting_observation": _redact(item.evidence),
                        },
                        "created_at": item.created_at,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {
        "yaml_path": str(yaml_path),
        "evidence_path": str(evidence_path),
        "cases": len(tests),
        "category_counts": _category_counts(selected),
    }


def _load_candidates(db_path: Path) -> list[CbmCandidate]:
    if not db_path.exists():
        raise FileNotFoundError(f"CBM database not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        summaries = conn.execute(
            """
            SELECT id, memory_session_id, project, request, investigated, completed,
                   next_steps, notes, files_read, files_edited, created_at, created_at_epoch
            FROM session_summaries
            ORDER BY created_at_epoch DESC
            """
        ).fetchall()
        observations = _load_observation_index(conn)
    finally:
        conn.close()

    candidates: list[CbmCandidate] = []
    for row in summaries:
        project = str(row["project"] or "")
        if _project_slug(project) not in DEFAULT_PROJECTS:
            continue
        raw = "\n".join(
            str(row[key] or "")
            for key in ("request", "investigated", "completed", "next_steps", "notes", "files_read", "files_edited")
        )
        category = _categorize(raw)
        if not category:
            continue
        evidence = observations.get(str(row["memory_session_id"]), "")
        if not _has_enough_signal(raw, evidence):
            continue
        candidates.append(
            CbmCandidate(
                source_table="session_summaries",
                source_row_id=int(row["id"]),
                memory_session_id=str(row["memory_session_id"] or ""),
                project=project,
                category=category,
                request=str(row["request"] or ""),
                investigated=str(row["investigated"] or ""),
                completed=str(row["completed"] or ""),
                next_steps=str(row["next_steps"] or ""),
                evidence=evidence,
                created_at=str(row["created_at"] or ""),
                raw_text=raw + "\n" + evidence,
            )
        )
    return candidates


def _load_observation_index(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute(
        """
        SELECT memory_session_id, title, subtitle, text, facts, evidence, created_at_epoch
        FROM observations
        WHERE type IN ('bugfix', 'refactor', 'learning', 'analysis')
        ORDER BY created_at_epoch DESC
        LIMIT 5000
        """
    ).fetchall()
    out: dict[str, list[str]] = {}
    for row in rows:
        sid = str(row["memory_session_id"] or "")
        if not sid:
            continue
        text = "\n".join(str(row[key] or "") for key in ("title", "subtitle", "text", "facts", "evidence"))
        if _has_enough_signal(text, ""):
            out.setdefault(sid, []).append(text[:1200])
    return {sid: "\n".join(items[:3]) for sid, items in out.items()}


def _select_candidates(candidates: list[CbmCandidate], max_cases: int) -> list[CbmCandidate]:
    selected: list[CbmCandidate] = []
    seen_sessions: set[str] = set()
    for category, quota in CASE_QUOTAS.items():
        for item in candidates:
            if len([x for x in selected if x.category == category]) >= quota:
                break
            if item.category != category or item.memory_session_id in seen_sessions:
                continue
            selected.append(item)
            seen_sessions.add(item.memory_session_id)
    for item in candidates:
        if len(selected) >= max_cases:
            break
        if item.memory_session_id in seen_sessions:
            continue
        selected.append(item)
        seen_sessions.add(item.memory_session_id)
    return selected[:max_cases]


def _candidate_to_test_case(item: CbmCandidate, index: int) -> dict[str, Any]:
    case_id = f"cbm-{item.category}-{index:03d}"
    brief = _compact(" ".join([item.request, item.investigated, item.completed]))
    expected = _expected_answer(item)
    keywords = _keywords_for(item)
    priority = "critical" if item.category in {"bugfix", "negative"} else "high"
    return {
        "id": case_id,
        "question": (
            "Analyze this redacted real development case and produce a release-quality answer. "
            "Your answer must explain the user problem, likely root cause, fix or evaluation approach, "
            "validation evidence, and remaining regression risk.\n\n"
            f"Case brief: {brief}"
        ),
        "description": f"Real CBM case from {_project_slug(item.project)} / {item.category}.",
        "expected_answer": expected,
        "priority": priority,
        "metadata": {
            "category": item.category,
            "source_project": _project_slug(item.project),
            "source_table": item.source_table,
            "source_row_id": item.source_row_id,
            "memory_session_id": _redact_session_id(item.memory_session_id),
        },
        "assert": [
            {"type": "keywords", "value": keywords, "all_required": False, "threshold": 0.5},
            {"type": "not-contains", "value": "直接发布即可"},
            {"type": "not-contains", "value": "无需验证"},
            {
                "type": "llm-rubric",
                "value": (
                    "The answer should be grounded in the case brief, include problem/root-cause/fix-or-plan/"
                    "validation/regression-risk, avoid unsupported claims, and avoid exposing secrets."
                ),
                "threshold": 0.7,
                "optional": True,
            },
        ],
    }


def _expected_answer(item: CbmCandidate) -> str:
    pieces = [
        "Must identify the concrete problem or user request.",
        "Must explain the investigated evidence or likely root cause.",
        "Must describe the completed fix, evaluation approach, or why it remains blocked.",
        "Must mention validation or regression checks.",
        "Must preserve privacy and not invent unobserved evidence.",
    ]
    if item.category == "negative":
        pieces.append("Must not claim the issue is fully fixed when the case says it is unresolved, cached, blocked, or missing validation.")
    return " ".join(pieces) + " Case evidence: " + _compact(" ".join([item.completed, item.next_steps, item.evidence]))


def _keywords_for(item: CbmCandidate) -> list[str]:
    base = ["problem", "root cause", "validation", "regression", "risk"]
    by_category = {
        "bugfix": ["bug", "fix", "test"],
        "tool_trace": ["trace", "tool", "evidence"],
        "eval_ab": ["eval", "baseline", "candidate"],
        "build_release": ["build", "package", "release"],
        "negative": ["not", "blocked", "risk"],
        "research_or_diagnosis": ["diagnosis", "evidence", "next step"],
    }
    return base + by_category.get(item.category, [])


def _categorize(text: str) -> str | None:
    low = text.lower()
    if re.search(r"不应|不能|不要|未修复|仍未|卡点|缺少|误判|幻觉|泄露|blocked|missing|unavailable|not fixed", low):
        return "negative"
    if re.search(r"eval|a/b|ab test|baseline|candidate|评估|对比评估|评分|断言", low):
        return "eval_ab"
    if re.search(r"hook|transcript|tool|mcp|trace|summary|obs|观测|工具|会话|记忆", low):
        return "tool_trace"
    if re.search(r"打包|安装包|exe|构建|build|release|缓存|旧页面|安装|发布", low):
        return "build_release"
    if re.search(r"bug|fix|failed|failure|error|修复|报错|错误|失败|异常|崩溃", low):
        return "bugfix"
    if re.search(r"排查|诊断|分析|调研|梳理|定位|investigate|diagnos", low):
        return "research_or_diagnosis"
    return None


def _has_enough_signal(text: str, evidence: str) -> bool:
    combined = f"{text}\n{evidence}".strip()
    if len(combined) < 80:
        return False
    return bool(re.search(r"修复|验证|测试|回归|定位|原因|bug|fix|test|build|trace|eval|risk", combined, re.I))


def _selection_reason(item: CbmCandidate) -> str:
    return f"Selected as {item.category} because the redacted summary contains concrete request, investigation, and completion/next-step evidence."


def _category_counts(items: list[CbmCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.category] = counts.get(item.category, 0) + 1
    return counts


def _project_slug(project: str) -> str:
    normalized = project.replace("\\", "/").strip("/").lower()
    if not normalized:
        return "unknown"
    slug = normalized.split("/")[-1]
    return re.sub(r"[^a-z0-9_.-]+", "-", slug) or "unknown"


def _redact_session_id(value: str) -> str:
    if not value:
        return "session-redacted"
    digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"session-{digest}"


def _compact(text: str, limit: int = 900) -> str:
    text = _redact(re.sub(r"\s+", " ", text or "").strip())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _redact(text: str) -> str:
    text = text or ""
    replacements = [
        (r"(?i)(api[_-]?key|token|secret|password|passwd)\s*[:=]\s*['\"]?[^'\"\s,;}]{6,}", r"\1=[REDACTED_SECRET]"),
        (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]"),
        (r"https?://[^\s)>\"]+", "[REDACTED_URL]"),
        (r"(?i)\b[A-Z]:[\\/][^\s,;\"')]+", "[REDACTED_PATH]"),
        (r"(?i)(?:^|\s)/(?:users|home|var|tmp|mnt)/[^\s,;\"')]+", " [REDACTED_PATH]"),
        (r"(?i)\b(?:[\w.-]+[\\/]){1,}[\w.-]+", "[REDACTED_PATH]"),
        (r"(?i)\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED_SECRET]"),
        (r"\b[0-9a-f]{32,}\b", "[REDACTED_ID]"),
        (r"(?i)\b[\w.-]+\.(?:exe|db)\b", "[REDACTED_ARTIFACT]"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    return text
