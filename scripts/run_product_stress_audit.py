from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / ".audit-workspace"
RESULT_PATH = ROOT / "product-stress-results.json"


def timed(callable_):
    started = time.perf_counter()
    value = callable_()
    return value, round((time.perf_counter() - started) * 1000, 2)


def write_cot(path: Path, index: int) -> None:
    payload = {
        "session_id": f"stress-{index:05d}",
        "agent_type": "cursor",
        "extracted_at": f"2026-07-17T08:{index % 60:02d}:00Z",
        "turns": [
            {
                "turn_index": 1,
                "user_query": "stress fixture",
                "final_response": "ok",
                "steps": [],
            }
        ],
    }
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def scan_benchmark() -> list[dict[str, Any]]:
    backend = ROOT / "src" / "agent_cot" / "assets" / "backend"
    sys.path.insert(0, str(backend))
    import services.session_scanner as scanner  # type: ignore

    cot_dir = Path(os.environ["COT_DIR"])
    results = []
    created = 0
    process = psutil.Process()
    for count in (100, 1000, 5000):
        for index in range(created, count):
            write_cot(cot_dir / f"stress-{index:05d}_cot.json", index)
        created = count
        scanner._OVERVIEW_CACHE.clear()
        rss_before = process.memory_info().rss
        cold_sessions, cold_ms = timed(scanner.scan_sessions)
        rss_after = process.memory_info().rss
        warm_sessions, warm_ms = timed(scanner.scan_sessions)
        results.append(
            {
                "trace_count": count,
                "cold_ms": cold_ms,
                "warm_ms": warm_ms,
                "sessions_returned": len(cold_sessions),
                "warm_sessions_returned": len(warm_sessions),
                "rss_delta_mb": round((rss_after - rss_before) / 1024 / 1024, 2),
            }
        )
    return results


def sqlite_concurrency_benchmark() -> dict[str, Any]:
    from agent_quality_eval.evaluation.store import DatasetStore

    db_path = WORKSPACE / "eval.db"
    DatasetStore(db_path)
    errors: list[str] = []

    def writer(index: int) -> None:
        try:
            local = DatasetStore(db_path)
            for turn in range(10):
                local.save_turn_eval(
                    {
                        "session_id": f"concurrent-{index}",
                        "turn_index": turn,
                        "created_at": f"2026-07-17T08:00:{index:02d}Z",
                        "passed": True,
                        "overall_score": 1.0,
                        "metrics": {"total_tokens": 10},
                        "eval_version": "v3",
                        "assertion_results": [],
                        "assertion_set": {"version": "turn-v3.7"},
                    }
                )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(writer, range(32)))
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    with sqlite3.connect(db_path) as conn:
        journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0])
        row_count = int(conn.execute("SELECT COUNT(*) FROM turn_evals").fetchone()[0])
    return {
        "writers": 32,
        "writes_expected": 320,
        "writes_persisted": row_count,
        "elapsed_ms": elapsed_ms,
        "journal_mode": journal_mode,
        "error_count": len(errors),
        "errors": errors[:10],
    }


def path_traversal_probe() -> dict[str, Any]:
    import services.session_scanner as scanner  # type: ignore

    cot_dir = Path(os.environ["COT_DIR"])
    outside = cot_dir.parent / "outside_cot.json"
    outside.write_text(
        json.dumps({"session_id": "outside", "turns": [{"turn_index": 1}]}),
        encoding="utf-8",
    )
    loaded = scanner.get_session_cot("../outside")
    return {
        "probe": "../outside",
        "outside_file": str(outside),
        "escaped_read_succeeded": isinstance(loaded, dict)
        and loaded.get("session_id") == "outside",
    }


def oversized_gold_probe() -> dict[str, Any]:
    from agent_quality_eval.evaluation.reference_eval import preview_reference_upload

    content = json.dumps(
        {"question": "large", "expected_answer": "x" * (5 * 1024 * 1024)}
    )
    preview, elapsed_ms = timed(
        lambda: preview_reference_upload("large.json", content, issue_token=False)
    )
    return {
        "input_mb": round(len(content.encode("utf-8")) / 1024 / 1024, 2),
        "accepted": bool(preview.get("canonical")),
        "elapsed_ms": elapsed_ms,
        "warning_count": len(preview.get("warnings") or []),
    }


def main() -> None:
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    cot_dir = WORKSPACE / "data" / "cot"
    fake_home = WORKSPACE / "home"
    cot_dir.mkdir(parents=True)
    fake_home.mkdir(parents=True)
    os.environ.update(
        {
            "HOME": str(fake_home),
            "USERPROFILE": str(fake_home),
            "AGENT_COT_DATA_ROOT": str(WORKSPACE / "data"),
            "COT_DIR": str(cot_dir),
            "CODEX_HOME": str(WORKSPACE / "codex"),
            "AGENT_QUALITY_EVAL_HOME": str(WORKSPACE / "eval-home"),
        }
    )
    result = {
        "schema_version": "aqe-product-stress-v1",
        "environment": {
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
            "isolated_workspace": str(WORKSPACE),
        },
        "session_scanner": scan_benchmark(),
        "sqlite_concurrency": sqlite_concurrency_benchmark(),
        "path_traversal": path_traversal_probe(),
        "oversized_gold": oversized_gold_probe(),
    }
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
