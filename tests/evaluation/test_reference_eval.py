from __future__ import annotations

import json

from agent_quality_eval.evaluation.reference_eval import (
    delete_turn_reference_answer,
    evaluate_turn_against_reference,
    get_reference_dataset,
    load_turn_reference_answer,
    parse_reference_dataset,
    save_turn_reference_answer,
    upload_reference_dataset,
)


def test_parse_reference_dataset_json_supports_cases() -> None:
    parsed = parse_reference_dataset(
        "gold.json",
        json.dumps(
            {
                "name": "Gold Answers",
                "version": "v1",
                "cases": [
                    {
                        "id": "case-a",
                        "question": "What is regression detection?",
                        "expected_answer": "Regression detection checks whether a candidate broke previously working behavior.",
                        "keywords": ["candidate", "previously working behavior"],
                        "assertions": [{"type": "contains", "value": "Regression"}],
                    }
                ],
            }
        ),
    )

    assert parsed["dataset_id"].startswith("gold-answers-v1-")
    assert parsed["cases"][0]["id"] == "case-a"
    assert parsed["cases"][0]["expected_answer"].startswith("Regression detection")
    assert parsed["cases"][0]["keywords"] == ["candidate", "previously working behavior"]


def test_parse_reference_dataset_json_supports_single_answer_object() -> None:
    parsed = parse_reference_dataset(
        "single-answer.json",
        json.dumps(
            {
                "question": "What is the expected output?",
                "standard_answer": "The expected output is a concise answer.",
                "keywords": ["concise answer"],
            }
        ),
    )

    assert len(parsed["cases"]) == 1
    assert parsed["cases"][0]["question"] == "What is the expected output?"
    assert parsed["cases"][0]["expected_answer"] == "The expected output is a concise answer."
    assert parsed["cases"][0]["normalization"]["mode"] == "schema_only"


def test_parse_reference_dataset_normalizes_process_requirements() -> None:
    parsed = parse_reference_dataset(
        "gold.json",
        json.dumps(
            {
                "question": "Use the required tool before answering",
                "expected_answer": "done",
                "trace_expectations": {
                    "required_tools": ["search"],
                    "must_include": ["validated evidence"],
                },
                "steps": [{"description": "inspect the tool output"}],
            }
        ),
    )

    reqs = parsed["cases"][0]["process_requirements"]
    assert reqs["required_tools"] == ["search"]
    assert "validated evidence" in reqs["must_include"]
    assert reqs["steps"] == ["inspect the tool output"]
    assert parsed["cases"][0]["normalization"]["mode"] == "schema_only"


def test_upload_reference_dataset_persists_normalized_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_QUALITY_EVAL_HOME", str(tmp_path))

    summary = upload_reference_dataset(
        "answers.yaml",
        """
name: Answer Keys
version: v2
tests:
  - id: exact
    question: Say hello
    expected_answer: hello world
""",
    )

    stored = get_reference_dataset(summary["dataset_id"])
    assert summary["case_count"] == 1
    assert stored["cases"][0]["question"] == "Say hello"
    assert (tmp_path / "reference-evals" / "datasets" / summary["dataset_id"] / "normalized.json").exists()


def test_save_turn_reference_answer_binds_gold_to_trace(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_QUALITY_EVAL_HOME", str(tmp_path))

    saved = save_turn_reference_answer(
        "session-a",
        3,
        "gold.json",
        json.dumps({"question": "Q", "standard_answer": "A", "keywords": ["A"]}),
    )
    loaded = load_turn_reference_answer("session-a", 3)

    assert saved["reference_answer"]["expected_answer"] == "A"
    assert loaded is not None
    assert loaded["reference_answer"]["expected_answer"] == "A"
    assert loaded["session_id"] == "session-a"
    assert loaded["turn_index"] == 3


def test_delete_turn_reference_answer_removes_binding(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_QUALITY_EVAL_HOME", str(tmp_path))

    save_turn_reference_answer(
        "session-a",
        3,
        "gold.json",
        json.dumps({"question": "Q", "standard_answer": "A"}),
    )

    assert delete_turn_reference_answer("session-a", 3) is True
    assert load_turn_reference_answer("session-a", 3) is None
    assert delete_turn_reference_answer("session-a", 3) is False


def test_reference_eval_scores_current_turn_against_expected_answer(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_QUALITY_EVAL_HOME", str(tmp_path))
    summary = upload_reference_dataset(
        "answers.json",
        json.dumps(
            {
                "name": "Trace Gold",
                "cases": [
                    {
                        "id": "regression",
                        "question": "Explain regression detection",
                        "expected_answer": "Regression detection checks that a candidate change has not broken baseline behavior.",
                        "keywords": ["candidate", "baseline"],
                    }
                ],
            }
        ),
    )
    turn_eval = {
        "session_id": "s1",
        "turn_index": 0,
        "metrics": {
            "user_query": "Explain regression detection",
            "total_tokens": 120,
            "input_tokens": 80,
            "output_tokens": 40,
            "tool_count": 1,
        },
    }
    turn_context = {
        "turn": {
            "user_query": "Explain regression detection",
            "final_response": "Regression detection checks whether a candidate breaks existing baseline behavior.",
            "steps": [{"step_type": "final_response", "content": "done"}],
        }
    }

    result = evaluate_turn_against_reference(
        session_id="s1",
        turn_index=0,
        dataset_id=summary["dataset_id"],
        case_id="regression",
        turn_eval=turn_eval,
        turn_context=turn_context,
    )

    assert result["verdict"] in {"pass", "partial"}
    assert result["final_score"] >= 0.5
    assert result["deterministic"]["keyword_coverage"] == 1.0
    assert result["trace_metrics"]["total_tokens"] == 120
    assert (tmp_path / "reference-evals" / "runs" / result["run_id"] / "reference-eval.json").exists()


def test_reference_eval_scores_process_requirements(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_QUALITY_EVAL_HOME", str(tmp_path))
    summary = upload_reference_dataset(
        "answers.json",
        json.dumps(
            {
                "name": "Trace Process Gold",
                "cases": [
                    {
                        "id": "process",
                        "question": "Find evidence",
                        "expected_answer": "Evidence was validated.",
                        "process_requirements": {
                            "required_tools": ["search"],
                            "must_include": ["validated evidence"],
                        },
                    }
                ],
            }
        ),
    )
    turn_eval = {
        "session_id": "s-process",
        "turn_index": 0,
        "metrics": {
            "user_query": "Find evidence",
            "final_response": "Evidence was validated.",
            "total_tokens": 120,
            "tool_count": 1,
        },
    }
    turn_context = {
        "turn": {
            "user_query": "Find evidence",
            "final_response": "Evidence was validated.",
            "steps": [
                {"step_type": "tool_execution", "content": "validated evidence", "metadata": {"tool_name": "search"}},
            ],
        }
    }

    result = evaluate_turn_against_reference(
        session_id="s-process",
        turn_index=0,
        dataset_id=summary["dataset_id"],
        case_id="process",
        turn_eval=turn_eval,
        turn_context=turn_context,
    )

    process = result["deterministic"]["process_requirements"]
    assert process["applicable"] is True
    assert process["score"] == 1.0
    assert process["total"] == 2
