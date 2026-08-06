"""SQLite-backed local dataset and experiment store."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from .models import ExperimentResult, ScoreResult, TestCase, TrialResult, utc_now

CURRENT_TURN_EVAL_VERSION = "v3"
CURRENT_TURN_EVAL_ASSERTION_SET_VERSION = "turn-v3.9"


def default_home() -> Path:
    return Path(os.environ.get("AGENT_QUALITY_EVAL_HOME", Path.home() / ".agent-quality-eval"))


def default_db_path() -> Path:
    return default_home() / "data" / "eval.db"


class DatasetStore:
    def __init__(self, db_path: str | os.PathLike[str] | None = None):
        self.db_path = Path(db_path) if db_path else default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _migrate(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    description TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    UNIQUE(name, version)
                );

                CREATE TABLE IF NOT EXISTS test_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
                    case_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    expected_answer TEXT,
                    context TEXT,
                    priority TEXT NOT NULL DEFAULT 'normal',
                    assertions_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    trace_json TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(dataset_id, case_id)
                );

                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    dataset_version TEXT NOT NULL,
                    providers_json TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT NOT NULL,
                    overall_pass_rate REAL NOT NULL DEFAULT 0,
                    average_score REAL NOT NULL DEFAULT 0,
                    average_response_time REAL NOT NULL DEFAULT 0,
                    total_trials INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                    provider_name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trials (
                    id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                    case_id TEXT NOT NULL,
                    provider_name TEXT NOT NULL,
                    trial_number INTEGER NOT NULL,
                    answer TEXT NOT NULL,
                    response_time REAL NOT NULL,
                    pass_rate REAL NOT NULL,
                    passed INTEGER NOT NULL,
                    conversation_id TEXT,
                    timestamp TEXT NOT NULL,
                    performance_json TEXT NOT NULL DEFAULT '{}',
                    trace_json TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trial_id TEXT NOT NULL REFERENCES trials(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    score REAL NOT NULL,
                    passed INTEGER NOT NULL,
                    threshold REAL NOT NULL,
                    reason TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS baselines (
                    dataset_name TEXT NOT NULL,
                    dataset_version TEXT NOT NULL,
                    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                    promoted_at TEXT NOT NULL,
                    PRIMARY KEY(dataset_name, dataset_version)
                );

                CREATE TABLE IF NOT EXISTS trace_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                    trial_id TEXT,
                    session_id TEXT,
                    trace_path TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS human_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trial_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    scorer_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    passed INTEGER NOT NULL,
                    dimension_scores_json TEXT NOT NULL DEFAULT '{}',
                    comments TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_evals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    overall_score REAL NOT NULL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL,
                    latency_ms REAL,
                    tool_count INTEGER NOT NULL DEFAULT 0,
                    error_count INTEGER NOT NULL DEFAULT 0,
                    report_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS turn_evals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    overall_score REAL NOT NULL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    tokens_per_second REAL,
                    tool_count INTEGER NOT NULL DEFAULT 0,
                    error_count INTEGER NOT NULL DEFAULT 0,
                    report_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_turn_evals_session_turn
                ON turn_evals(session_id, turn_index, created_at DESC, id DESC);

                CREATE TABLE IF NOT EXISTS eval_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    project_id TEXT,
                    project_name TEXT,
                    project_path TEXT,
                    session_id TEXT,
                    turn_index INTEGER,
                    baseline_session_id TEXT,
                    baseline_turn_index INTEGER,
                    candidate_session_id TEXT,
                    candidate_turn_index INTEGER,
                    has_gold INTEGER NOT NULL DEFAULT 0,
                    gold_hash TEXT,
                    verdict TEXT,
                    winner TEXT,
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    target_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_eval_events_created
                ON eval_events(created_at DESC, id DESC);

                CREATE INDEX IF NOT EXISTS idx_eval_events_type_project
                ON eval_events(event_type, project_id, has_gold, created_at DESC);
                """
            )

    def upsert_dataset(
        self,
        name: str,
        version: str,
        test_cases: list[TestCase],
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO datasets(name, version, description, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name, version) DO UPDATE SET
                  description=excluded.description,
                  metadata_json=excluded.metadata_json
                """,
                (name, version, description, json.dumps(metadata or {}, ensure_ascii=False), now),
            )
            dataset_id = conn.execute(
                "SELECT id FROM datasets WHERE name=? AND version=?", (name, version)
            ).fetchone()["id"]
            for case in test_cases:
                conn.execute(
                    """
                    INSERT INTO test_cases(
                        dataset_id, case_id, question, expected_answer, context, priority,
                        assertions_json, metadata_json, trace_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(dataset_id, case_id) DO UPDATE SET
                      question=excluded.question,
                      expected_answer=excluded.expected_answer,
                      context=excluded.context,
                      priority=excluded.priority,
                      assertions_json=excluded.assertions_json,
                      metadata_json=excluded.metadata_json,
                      trace_json=excluded.trace_json
                    """,
                    (
                        dataset_id,
                        case.id,
                        case.question,
                        case.expected_answer,
                        case.context,
                        case.priority,
                        json.dumps(case.assertions, ensure_ascii=False),
                        json.dumps(case.metadata, ensure_ascii=False),
                        json.dumps(case.trace, ensure_ascii=False) if case.trace else None,
                        now,
                    ),
                )
            return int(dataset_id)

    def save_experiment(self, result: ExperimentResult) -> None:
        data = result.to_dict()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments(
                    id, name, dataset_name, dataset_version, providers_json, started_at,
                    ended_at, status, overall_pass_rate, average_score, average_response_time,
                    total_trials, metadata_json, result_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.experiment_id,
                    result.name,
                    result.dataset_name,
                    result.dataset_version,
                    json.dumps(result.providers, ensure_ascii=False),
                    result.started_at,
                    result.ended_at,
                    result.status,
                    result.overall_pass_rate,
                    result.average_score,
                    result.average_response_time,
                    result.total_trials,
                    json.dumps(result.metadata, ensure_ascii=False),
                    json.dumps(data, ensure_ascii=False),
                ),
            )
            for provider in result.providers:
                conn.execute(
                    "INSERT INTO runs(experiment_id, provider_name, created_at) VALUES (?, ?, ?)",
                    (result.experiment_id, provider, utc_now()),
                )
            for case in result.case_results:
                for trial in case.trials:
                    self._insert_trial(conn, result.experiment_id, trial)

    def _insert_trial(self, conn: sqlite3.Connection, experiment_id: str, trial: TrialResult) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO trials(
                id, experiment_id, case_id, provider_name, trial_number, answer,
                response_time, pass_rate, passed, conversation_id, timestamp,
                performance_json, trace_json, error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trial.trial_id,
                experiment_id,
                trial.test_case_id,
                trial.provider_name,
                trial.trial_number,
                trial.answer,
                trial.response_time,
                trial.pass_rate,
                1 if trial.passed else 0,
                trial.conversation_id,
                trial.timestamp,
                json.dumps(trial.performance.to_dict(), ensure_ascii=False),
                json.dumps(trial.trace, ensure_ascii=False) if trial.trace else None,
                trial.error,
            ),
        )
        conn.execute("DELETE FROM scores WHERE trial_id=?", (trial.trial_id,))
        for score in trial.scores:
            self._insert_score(conn, trial.trial_id, score)

    def _insert_score(self, conn: sqlite3.Connection, trial_id: str, score: ScoreResult) -> None:
        conn.execute(
            """
            INSERT INTO scores(trial_id, name, type, score, passed, threshold, reason, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trial_id,
                score.name,
                score.type,
                score.score,
                1 if score.passed else 0,
                score.threshold,
                score.reason,
                json.dumps(score.metadata, ensure_ascii=False),
            ),
        )

    def get_experiment_dict(self, experiment_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT result_json FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            if not row:
                raise KeyError(f"Experiment not found: {experiment_id}")
            return json.loads(row["result_json"])

    def list_experiments(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, name, dataset_name, dataset_version, providers_json, started_at,
                       ended_at, status, overall_pass_rate, average_score,
                       average_response_time, total_trials
                FROM experiments
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [
                {
                    **dict(row),
                    "providers": json.loads(row["providers_json"]),
                }
                for row in rows
            ]

    def promote_baseline(self, experiment_id: str) -> None:
        exp = self.get_experiment_dict(experiment_id)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO baselines(dataset_name, dataset_version, experiment_id, promoted_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(dataset_name, dataset_version) DO UPDATE SET
                  experiment_id=excluded.experiment_id,
                  promoted_at=excluded.promoted_at
                """,
                (exp["dataset_name"], exp["dataset_version"], experiment_id, utc_now()),
            )

    def get_baseline(self, dataset_name: str, dataset_version: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT experiment_id FROM baselines WHERE dataset_name=? AND dataset_version=?",
                (dataset_name, dataset_version),
            ).fetchone()
            return row["experiment_id"] if row else None

    def save_session_eval(self, report: dict[str, Any]) -> int:
        metrics = report.get("metrics") or {}
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO session_evals(
                    session_id, created_at, passed, overall_score, input_tokens,
                    output_tokens, total_tokens, cost_usd, latency_ms, tool_count,
                    error_count, report_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report["session_id"],
                    report.get("created_at") or utc_now(),
                    1 if report.get("passed") else 0,
                    float(report.get("overall_score", 0)),
                    int(metrics.get("input_tokens", 0) or 0),
                    int(metrics.get("output_tokens", 0) or 0),
                    int(metrics.get("total_tokens", 0) or 0),
                    metrics.get("cost_usd"),
                    metrics.get("latency_ms"),
                    int(metrics.get("tool_count", 0) or 0),
                    int(metrics.get("error_count", 0) or 0),
                    json.dumps(report, ensure_ascii=False),
                ),
            )
            report_id = int(cursor.lastrowid)
        report["id"] = report_id
        return report_id

    def list_session_evals(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, created_at, passed, overall_score,
                       input_tokens, output_tokens, total_tokens, cost_usd,
                       latency_ms, tool_count, error_count
                FROM session_evals
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_latest_session_eval(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT report_json
                FROM session_evals
                WHERE session_id=?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            return json.loads(row["report_json"]) if row else None

    def save_turn_eval(self, report: dict[str, Any]) -> int:
        metrics = report.get("metrics") or {}
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO turn_evals(
                    session_id, turn_index, created_at, passed, overall_score,
                    input_tokens, output_tokens, total_tokens, tokens_per_second,
                    tool_count, error_count, report_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report["session_id"],
                    int(report["turn_index"]),
                    report.get("created_at") or utc_now(),
                    1 if report.get("passed") else 0,
                    float(report.get("overall_score", 0)),
                    int(metrics.get("input_tokens", 0) or 0),
                    int(metrics.get("output_tokens", 0) or 0),
                    int(metrics.get("total_tokens", 0) or 0),
                    metrics.get("tokens_per_second"),
                    int(metrics.get("tool_count", 0) or 0),
                    int(metrics.get("error_count", 0) or 0),
                    json.dumps(report, ensure_ascii=False),
                ),
            )
            report_id = int(cursor.lastrowid)
        report["id"] = report_id
        return report_id

    def get_latest_turn_eval(self, session_id: str, turn_index: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT report_json
                FROM turn_evals
                WHERE session_id=? AND turn_index=?
                ORDER BY created_at DESC, id DESC
                """,
                (session_id, int(turn_index)),
            ).fetchall()
            for row in rows:
                report = json.loads(row["report_json"])
                if _is_current_turn_eval_report(report):
                    return report
            return None

    def list_turn_evals(self, session_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if session_id:
                rows = conn.execute(
                    """
                    SELECT id, session_id, turn_index, created_at, passed, overall_score,
                           input_tokens, output_tokens, total_tokens, tokens_per_second,
                           tool_count, error_count, report_json
                    FROM turn_evals
                    WHERE session_id=?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (session_id, max(limit * 10, limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, session_id, turn_index, created_at, passed, overall_score,
                           input_tokens, output_tokens, total_tokens, tokens_per_second,
                           tool_count, error_count, report_json
                    FROM turn_evals
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (max(limit * 10, limit),),
                ).fetchall()
            reports: list[dict[str, Any]] = []
            seen: set[tuple[str, int]] = set()
            for row in rows:
                row_dict = dict(row)
                try:
                    report = json.loads(str(row_dict.pop("report_json") or "{}"))
                except json.JSONDecodeError:
                    continue
                if not _is_current_turn_eval_report(report):
                    continue
                key = (
                    str(row_dict.get("session_id") or report.get("session_id") or ""),
                    int(row_dict.get("turn_index") or report.get("turn_index") or 0),
                )
                if key in seen:
                    continue
                seen.add(key)
                reports.append({**report, **{k: v for k, v in row_dict.items() if k not in report}})
                if len(reports) >= limit:
                    break
            return reports

    def save_eval_event(self, event: dict[str, Any]) -> int:
        now = event.get("created_at") or utc_now()
        summary = event.get("summary") if isinstance(event.get("summary"), dict) else {}
        target = event.get("target") if isinstance(event.get("target"), dict) else {}
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO eval_events(
                    created_at, event_type, project_id, project_name, project_path,
                    session_id, turn_index, baseline_session_id, baseline_turn_index,
                    candidate_session_id, candidate_turn_index, has_gold, gold_hash,
                    verdict, winner, summary_json, target_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    str(event.get("event_type") or "trace"),
                    event.get("project_id"),
                    event.get("project_name"),
                    event.get("project_path"),
                    event.get("session_id"),
                    event.get("turn_index"),
                    event.get("baseline_session_id"),
                    event.get("baseline_turn_index"),
                    event.get("candidate_session_id"),
                    event.get("candidate_turn_index"),
                    1 if event.get("has_gold") else 0,
                    event.get("gold_hash"),
                    event.get("verdict"),
                    event.get("winner"),
                    json.dumps(summary, ensure_ascii=False, default=str),
                    json.dumps(target, ensure_ascii=False, default=str),
                ),
            )
            return int(cursor.lastrowid)

    def list_eval_events(
        self,
        *,
        limit: int = 100,
        event_type: str | None = None,
        project_id: str | None = None,
        has_gold: bool | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if project_id:
            clauses.append("project_id = ?")
            params.append(project_id)
        if has_gold is not None:
            clauses.append("has_gold = ?")
            params.append(1 if has_gold else 0)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit or 100), 500)))
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, created_at, event_type, project_id, project_name, project_path,
                       session_id, turn_index, baseline_session_id, baseline_turn_index,
                       candidate_session_id, candidate_turn_index, has_gold, gold_hash,
                       verdict, winner, summary_json, target_json
                FROM eval_events
                {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["summary"] = json.loads(str(item.pop("summary_json") or "{}"))
            except json.JSONDecodeError:
                item["summary"] = {}
            try:
                item["target"] = json.loads(str(item.pop("target_json") or "{}"))
            except json.JSONDecodeError:
                item["target"] = {}
            item["has_gold"] = bool(item.get("has_gold"))
            events.append(item)
        return events


def _is_current_turn_eval_report(report: dict[str, Any]) -> bool:
    assertion_set = report.get("assertion_set") if isinstance(report, dict) else {}
    return (
        isinstance(report, dict)
        and report.get("eval_version") == CURRENT_TURN_EVAL_VERSION
        and isinstance(report.get("assertion_results"), list)
        and isinstance(assertion_set, dict)
        and assertion_set.get("version") == CURRENT_TURN_EVAL_ASSERTION_SET_VERSION
    )
