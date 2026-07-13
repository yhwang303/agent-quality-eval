import json
import sys
import types
from pathlib import Path

from agent_quality_eval.evaluation.critic import CRITIC_REPORT_SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[2]


def _read_asset(*parts: str) -> str:
    return (ROOT / "src" / "agent_cot" / "assets" / "hooks" / Path(*parts)).read_text(
        encoding="utf-8"
    )


def test_manual_eval_uses_agent_critic_fallback_when_hook_report_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_COT_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_QUALITY_EVAL_HOME", str(tmp_path / "eval-home"))

    scanner = types.ModuleType("services.session_scanner")
    scanner.get_session_cot = lambda session_id: {
        "session_id": session_id,
        "agent_type": "codex",
        "turns": [{"turn_index": 0, "user_query": "hello", "final_response": "hi", "steps": []}],
    }
    scanner.get_session_transcript = lambda session_id: None
    scanner.scan_sessions = lambda: []

    otel = types.ModuleType("services.claude_otel_receiver")
    otel.load_session_otel = lambda session_id: {}

    services = types.ModuleType("services")
    monkeypatch.setitem(sys.modules, "services", services)
    monkeypatch.setitem(sys.modules, "services.session_scanner", scanner)
    monkeypatch.setitem(sys.modules, "services.claude_otel_receiver", otel)

    from agent_quality_eval.evaluation.api import eval_observed_turn

    report = eval_observed_turn("codex-hook-required", 0, db_path=str(tmp_path / "eval.db"))

    assert report["judge"]["source_event"] == "manual-agent-critic-fallback"
    assert report["judge"]["eval_method"] == "agent_critic_v1"
    assert report["judge"]["status"] in {"completed", "unconfigured", "disabled", "error"}


def test_api_rerun_reports_are_not_valid_hook_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_COT_DATA_ROOT", str(tmp_path / "data"))

    from agent_quality_eval.evaluation.critic import load_critic_report, write_critic_report

    write_critic_report(
        {
            "schema_version": CRITIC_REPORT_SCHEMA_VERSION,
            "eval_method": "agent_critic_v1",
            "session_id": "legacy-api-rerun",
            "turn_index": 0,
            "status": "completed",
            "source_event": "api-rerun",
            "created_at": "2026-01-01T00:00:00Z",
        }
    )

    assert load_critic_report("legacy-api-rerun", 0) is None


def test_all_ide_hooks_spawn_final_agent_critic_from_hook_assets():
    codebuddy = _read_asset("codebuddy", "cot-stream-codebuddy.js")
    assert "maybeTriggerCritic(event, cid);" in codebuddy
    assert "`codebuddy-stream:${event}`" in codebuddy
    assert "'--no-persist-eval'" in codebuddy
    assert "'--agent-quality-eval-runner', kind" in codebuddy

    codex = _read_asset("codex", "codex_stream_hook.py")
    assert "if event in _CRITIC_TRIGGER_EVENTS:" in codex
    assert "_trigger_critic(sid, event)" in codex
    assert '"--no-persist-eval"' in codex
    assert '"--agent-quality-eval-runner", runner' in codex
    codex_sidecar = _read_asset("codex", "codex_sidecar_collector.py")
    assert "_agent_quality_eval_runner_cmd(py, \"critic\"" in codex_sidecar
    assert '"--agent-quality-eval-runner", runner' in codex_sidecar

    claude = _read_asset("claude", "claude_stream_hook.py")
    assert "_maybe_trigger_critic(record)" in claude
    assert 'f"claude-stream:{ev}"' in claude
    assert '"--no-persist-eval"' in claude
    assert '"--agent-quality-eval-runner", runner' in claude

    cursor_stream = _read_asset("cursor", "cot-stream.js")
    assert "maybeTriggerCritic(event, cid);" in cursor_stream
    assert "`cursor-stream:${event}`" in cursor_stream
    assert "'--no-persist-eval'" in cursor_stream
    assert "'--agent-quality-eval-runner', kind" in cursor_stream

    cursor_bridge = _read_asset("cursor", "cot-bridge.js")
    assert "`cursor-bridge:${event}`" in cursor_bridge
    assert "'--no-persist-eval'" in cursor_bridge
    assert "'--agent-quality-eval-runner', kind" in cursor_bridge

    for agent in ("codebuddy", "codex", "cursor", "claude"):
        hook = _read_asset(agent, "agent_critic_hook.py")
        assert "def _agent_quality_eval_runner_cmd" in hook
        assert '"--agent-quality-eval-runner", "critic"' in hook
        assert "cmd = _agent_quality_eval_runner_cmd(py," in hook


def test_hook_health_reports_codex_config_assets_runtime_and_recent_activity(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from agent_cot.agents.codex import CodexAdapter
    from agent_quality_eval.evaluation.hook_health import build_hook_health_report

    codex_home = tmp_path / ".codex"
    hooks_dir = codex_home / "hooks"
    hooks_dir.mkdir(parents=True)
    adapter = CodexAdapter()
    entries = adapter.hook_entries()
    config = adapter.merge_hook_entries({}, entries)
    adapter.hooks_config_path().write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    for name in adapter.bridge_files():
        (hooks_dir / name).write_text("# test hook\n", encoding="utf-8")

    runtime_root = tmp_path / ".agent-cot"
    cot_extractor = tmp_path / "cot-extractor"
    data_root = runtime_root / "data"
    cot_extractor.mkdir(parents=True)
    data_root.mkdir(parents=True)
    runtime_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "runtime.json").write_text(
        json.dumps(
            {
                "python_executable": sys.executable,
                "cot_extractor_root": str(cot_extractor),
                "data_root": str(data_root),
            }
        ),
        encoding="utf-8",
    )
    logs_dir = runtime_root / "logs"
    logs_dir.mkdir()
    (logs_dir / "pipeline.log").write_text(
        "[2026-06-23T12:00:00Z] [hook.codex] [codex] [sid=codex-test] event=Stop status=ok\n",
        encoding="utf-8",
    )

    report = build_hook_health_report()
    codex = next(item for item in report["agents"] if item["agent"] == "codex")
    assert codex["status"] == "ok"
    assert codex["activated"] is True
    assert codex["runtime_ok"] is True
    assert codex["targets"][0]["entries_active"] is True
    assert codex["recent_activity"]["available"] is True


def test_hook_report_can_be_ingested_without_model_rerun(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_COT_DATA_ROOT", str(tmp_path / "data"))

    from agent_quality_eval.evaluation.critic import write_critic_report
    from agent_quality_eval.evaluation.session_eval import _run_optional_turn_judge

    write_critic_report(
        {
            "schema_version": CRITIC_REPORT_SCHEMA_VERSION,
            "eval_method": "agent_critic_v1",
            "session_id": "hook-ingest",
            "turn_index": 1,
            "status": "completed",
            "source_event": "codex-stream:Stop",
            "created_at": "2026-01-01T00:00:00Z",
            "provider": "timiai",
            "model": "gpt-4o-mini",
            "structured": {
                "summary_conclusion": "结论：hook report 已生成。",
                "overall_verdict": "resolved",
                "review_markdown": "**结论**：hook report 已生成。",
            },
        }
    )

    judge = _run_optional_turn_judge(
        {},
        {},
        {},
        {"session_id": "hook-ingest", "turn_index": 1, "sources": []},
    )

    assert judge["status"] == "completed"
    assert judge["source_event"] == "codex-stream:Stop"
    assert judge["model"] == "gpt-4o-mini"


def test_running_hook_report_is_not_rendered_as_completed_judge(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_COT_DATA_ROOT", str(tmp_path / "data"))

    from agent_quality_eval.evaluation.critic import (
        critic_report_path,
        load_best_live_critic_state,
        write_critic_report,
    )
    from agent_quality_eval.evaluation.live_critic import live_critic_state_path, live_critic_turn_path
    from agent_quality_eval.evaluation.session_eval import _run_optional_turn_judge

    write_critic_report(
        {
            "schema_version": CRITIC_REPORT_SCHEMA_VERSION,
            "eval_method": "agent_critic_v1",
            "session_id": "stale-running",
            "turn_index": 1,
            "status": "running",
            "source_event": "codebuddy-stream:Stop",
            "created_at": "2026-01-01T00:00:00Z",
            "structured": {"review_markdown": "must not be shown"},
        }
    )
    live_critic_turn_path("stale-running", 1).parent.mkdir(parents=True, exist_ok=True)
    live_critic_turn_path("stale-running", 1).write_text(
        json.dumps(
            {
                "schema_version": "agent-critic-live-v1",
                "session_id": "stale-running",
                "status": "running",
                "updated_at": "2026-01-01T00:00:01Z",
                "event_count": 1,
                "turn_index_approx": 1,
            }
        ),
        encoding="utf-8",
    )
    live_critic_state_path("stale-running").write_text(
        json.dumps(
            {
                "schema_version": "agent-critic-live-v1",
                "session_id": "stale-running",
                "status": "completed",
                "updated_at": "2026-01-01T00:00:10Z",
                "event_count": 13,
                "turn_index_approx": 2,
            }
        ),
        encoding="utf-8",
    )

    judge = _run_optional_turn_judge(
        {},
        {},
        {},
        {"session_id": "stale-running", "turn_index": 1, "sources": []},
    )

    assert critic_report_path("stale-running", 1).is_file()
    assert load_best_live_critic_state("stale-running", 1)["status"] == "completed"
    assert judge["status"] == "interrupted"
    assert judge["structured"] == {}
    assert judge["live_supervisor"]["status"] == "completed"


def test_manual_eval_retries_stale_running_hook_report(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_COT_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_QUALITY_EVAL_HOME", str(tmp_path / "eval-home"))

    from agent_quality_eval.evaluation.critic import write_critic_report

    write_critic_report(
        {
            "schema_version": CRITIC_REPORT_SCHEMA_VERSION,
            "eval_method": "agent_critic_v1",
            "session_id": "retry-stale-running",
            "turn_index": 0,
            "status": "running",
            "source_event": "codebuddy-stream:Stop",
            "created_at": "2026-01-01T00:00:00Z",
        }
    )

    scanner = types.ModuleType("services.session_scanner")
    scanner.get_session_cot = lambda session_id: {
        "session_id": session_id,
        "agent_type": "codebuddy",
        "turns": [{"turn_index": 0, "user_query": "hello", "final_response": "hi", "steps": []}],
    }
    scanner.get_session_transcript = lambda session_id: None
    scanner.scan_sessions = lambda: []

    otel = types.ModuleType("services.claude_otel_receiver")
    otel.load_session_otel = lambda session_id: {}

    services = types.ModuleType("services")
    monkeypatch.setitem(sys.modules, "services", services)
    monkeypatch.setitem(sys.modules, "services.session_scanner", scanner)
    monkeypatch.setitem(sys.modules, "services.claude_otel_receiver", otel)

    from agent_quality_eval.evaluation.api import eval_observed_turn

    report = eval_observed_turn("retry-stale-running", 0, db_path=str(tmp_path / "eval.db"))

    assert report["judge"]["source_event"] == "manual-hook-report-recovery"
    assert report["judge"]["status"] != "running"


def test_manual_eval_recovers_completed_manual_fallback_when_live_hook_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_COT_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_QUALITY_EVAL_HOME", str(tmp_path / "eval-home"))

    from agent_quality_eval.evaluation.critic import write_critic_report
    from agent_quality_eval.evaluation.live_critic import live_critic_turn_path

    write_critic_report(
        {
            "schema_version": CRITIC_REPORT_SCHEMA_VERSION,
            "eval_method": "agent_critic_v1",
            "session_id": "recover-manual-fallback",
            "turn_index": 0,
            "status": "completed",
            "source_event": "manual-agent-critic-fallback",
            "created_at": "2026-01-01T00:00:00Z",
            "structured": {"summary_conclusion": "old manual fallback"},
        }
    )
    live_path = live_critic_turn_path("recover-manual-fallback", 0)
    live_path.parent.mkdir(parents=True, exist_ok=True)
    live_path.write_text(
        json.dumps(
            {
                "schema_version": "agent-critic-live-v1",
                "session_id": "recover-manual-fallback",
                "status": "completed",
                "last_event": "codex-stream:Stop",
                "updated_at": "2026-01-01T00:00:10Z",
                "event_count": 3,
                "turn_index_approx": 0,
            }
        ),
        encoding="utf-8",
    )

    scanner = types.ModuleType("services.session_scanner")
    scanner.get_session_cot = lambda session_id: {
        "session_id": session_id,
        "agent_type": "codex",
        "turns": [{"turn_index": 0, "user_query": "hello", "final_response": "hi", "steps": []}],
    }
    scanner.get_session_transcript = lambda session_id: None
    scanner.scan_sessions = lambda: []

    otel = types.ModuleType("services.claude_otel_receiver")
    otel.load_session_otel = lambda session_id: {}

    services = types.ModuleType("services")
    monkeypatch.setitem(sys.modules, "services", services)
    monkeypatch.setitem(sys.modules, "services.session_scanner", scanner)
    monkeypatch.setitem(sys.modules, "services.claude_otel_receiver", otel)

    from agent_quality_eval.evaluation.api import eval_observed_turn

    report = eval_observed_turn("recover-manual-fallback", 0, db_path=str(tmp_path / "eval.db"))

    assert report["judge"]["source_event"] == "manual-hook-report-recovery"
    assert report["judge"]["status"] != "running"


def test_manual_eval_recovers_completed_manual_fallback_when_turn_has_hook_timestamps(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("AGENT_COT_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_QUALITY_EVAL_HOME", str(tmp_path / "eval-home"))

    from agent_quality_eval.evaluation.critic import write_critic_report

    write_critic_report(
        {
            "schema_version": CRITIC_REPORT_SCHEMA_VERSION,
            "eval_method": "agent_critic_v1",
            "session_id": "recover-turn-hook-evidence",
            "turn_index": 4,
            "status": "completed",
            "source_event": "manual-agent-critic-fallback",
            "created_at": "2026-01-01T00:00:00Z",
            "structured": {"summary_conclusion": "old manual fallback"},
        }
    )

    scanner = types.ModuleType("services.session_scanner")
    scanner.get_session_cot = lambda session_id: {
        "session_id": session_id,
        "agent_type": "codex",
        "turns": [
            {
                "turn_index": 4,
                "user_query": "hello",
                "final_response": "hi",
                "turn_start_ms_observed": 1782272265102,
                "turn_end_ms_observed": 1782272689850,
                "turn_duration_ms_observed": 424748,
                "steps": [],
            }
        ],
    }
    scanner.get_session_transcript = lambda session_id: None
    scanner.scan_sessions = lambda: []

    otel = types.ModuleType("services.claude_otel_receiver")
    otel.load_session_otel = lambda session_id: {}

    services = types.ModuleType("services")
    monkeypatch.setitem(sys.modules, "services", services)
    monkeypatch.setitem(sys.modules, "services.session_scanner", scanner)
    monkeypatch.setitem(sys.modules, "services.claude_otel_receiver", otel)

    from agent_quality_eval.evaluation.api import eval_observed_turn

    report = eval_observed_turn("recover-turn-hook-evidence", 4, db_path=str(tmp_path / "eval.db"))

    assert report["judge"]["source_event"] == "manual-hook-report-recovery"
    assert report["judge"]["status"] != "running"


def test_manual_eval_degrades_to_fallback_when_hook_report_errored(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_COT_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_QUALITY_EVAL_HOME", str(tmp_path / "eval-home"))

    from agent_quality_eval.evaluation.critic import write_critic_report

    write_critic_report(
        {
            "schema_version": CRITIC_REPORT_SCHEMA_VERSION,
            "eval_method": "agent_critic_v1",
            "session_id": "hook-error-fallback",
            "turn_index": 0,
            "status": "error",
            "source_event": "cursor-stream:Stop",
            "created_at": "2026-01-01T00:00:00Z",
            "reason": "runner failed",
        }
    )

    scanner = types.ModuleType("services.session_scanner")
    scanner.get_session_cot = lambda session_id: {
        "session_id": session_id,
        "agent_type": "cursor",
        "turns": [{"turn_index": 0, "user_query": "hello", "final_response": "hi", "steps": []}],
    }
    scanner.get_session_transcript = lambda session_id: None
    scanner.scan_sessions = lambda: []

    otel = types.ModuleType("services.claude_otel_receiver")
    otel.load_session_otel = lambda session_id: {}

    services = types.ModuleType("services")
    monkeypatch.setitem(sys.modules, "services", services)
    monkeypatch.setitem(sys.modules, "services.session_scanner", scanner)
    monkeypatch.setitem(sys.modules, "services.claude_otel_receiver", otel)

    from agent_quality_eval.evaluation.api import eval_observed_turn

    report = eval_observed_turn("hook-error-fallback", 0, db_path=str(tmp_path / "eval.db"))

    assert report["judge"]["source_event"] == "manual-agent-critic-fallback"
    assert report["judge"]["status"] != "running"


def test_real_hook_report_can_replace_manual_hook_recovery(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_COT_DATA_ROOT", str(tmp_path / "data"))

    from agent_quality_eval.evaluation.critic import run_critic_for_cot, write_critic_report

    write_critic_report(
        {
            "schema_version": CRITIC_REPORT_SCHEMA_VERSION,
            "eval_method": "agent_critic_v1",
            "session_id": "replace-recovery",
            "turn_index": 0,
            "status": "completed",
            "source_event": "manual-hook-report-recovery",
            "created_at": "2026-01-01T00:00:00Z",
            "structured": {"summary_conclusion": "manual recovery"},
        }
    )

    report = run_critic_for_cot(
        {
            "session_id": "replace-recovery",
            "agent_type": "codebuddy",
            "turns": [{"turn_index": 0, "user_query": "hello", "final_response": "hi", "steps": []}],
        },
        session_id="replace-recovery",
        turn_index=0,
        agent_type="codebuddy",
        source_event="codebuddy-stream:Stop",
        persist_eval=False,
    )

    assert report["source_event"] == "codebuddy-stream:Stop"
    assert report["status"] != "running"


def test_runner_persists_under_canonical_cot_session_id(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_COT_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_QUALITY_EVAL_HOME", str(tmp_path / "eval-home"))

    cot_dir = tmp_path / "data" / "cot"
    cot_dir.mkdir(parents=True)
    (cot_dir / "codex-short-id_cot.json").write_text(
        json.dumps(
            {
                "session_id": "codex-canonical-id",
                "agent_type": "codex",
                "turns": [
                    {
                        "turn_index": 0,
                        "user_query": "smoke",
                        "final_response": "done",
                        "steps": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    from agent_quality_eval.evaluation.critic import critic_report_path, run_critic_for_session

    report = run_critic_for_session(
        session_id="codex-short-id",
        agent_type="codex",
        source_event="codex-stream:Stop",
        wait_seconds=0,
        persist_eval=False,
    )

    assert report["session_id"] == "codex-canonical-id"
    assert critic_report_path("codex-canonical-id", 0).is_file()
    assert not critic_report_path("codex-short-id", 0).exists()
