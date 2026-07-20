from __future__ import annotations

import json
from types import SimpleNamespace

from agent_quality_eval.evaluation.reference_eval import (
    confirm_turn_reference_answer,
    delete_turn_reference_answer,
    evaluate_turn_against_reference,
    get_reference_dataset,
    has_gold_evidence,
    load_turn_reference_answer,
    parse_reference_dataset,
    preview_reference_upload,
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


def test_parse_natural_markdown_acceptance_note_without_internal_labels() -> None:
    parsed = parse_reference_dataset(
        "gold.md",
        """# 我对这次任务的预期

用户想解决的问题

如何验证这次修改？

我认为合理的结果

运行测试并确认没有回归。

验收时我主要看

测试通过，旧行为保持不变。
""",
    )

    case = parsed["cases"][0]
    assert case["question"] == "如何验证这次修改？"
    assert case["expected_answer"] == "运行测试并确认没有回归。"
    assert case["rubric"] == "测试通过，旧行为保持不变。"


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


def test_preview_nested_gold_reports_mapping_without_persisting(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_QUALITY_EVAL_HOME", str(tmp_path))
    raw = json.dumps(
        {
            "task": "Explain the change",
            "ground_truth": {
                "response": "The candidate must preserve baseline behavior.",
            },
            "criteria": "Mention both candidate and baseline.",
        }
    )

    preview = preview_reference_upload(
        "nested.json",
        raw,
        session_id="session-a",
        turn_index=2,
    )

    assert preview["canonical"]["reference_answer"]["expected_answer"].startswith("The candidate")
    assert preview["eval_mode"] == "gold"
    assert preview["confirm_token"]
    assert not (tmp_path / "reference-evals" / "turn-answers").exists()
    assert any(
        item["canonical_field"] == "expected_answer"
        for item in preview["normalization_report"]["mapping"]
    )


def test_confirm_preview_persists_raw_canonical_and_binding_hash(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_QUALITY_EVAL_HOME", str(tmp_path))
    preview = preview_reference_upload(
        "gold.yaml",
        "question: Q\nrubric: Must be concise\n",
        session_id="session-a",
        turn_index=3,
    )

    saved = confirm_turn_reference_answer(
        "session-a",
        3,
        preview["confirm_token"],
    )

    assert saved["schema_version"] == "turn-reference-answer-v2"
    assert saved["eval_mode"] == "gold"
    assert saved["binding_hash"]
    assert saved["raw"]["content"].startswith("question:")
    assert saved["normalization_report"]["canonical_hash"] == saved["binding_hash"]


def test_confirm_preview_rejects_different_turn(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_QUALITY_EVAL_HOME", str(tmp_path))
    preview = preview_reference_upload(
        "gold.txt",
        "Expected output",
        session_id="session-a",
        turn_index=1,
    )

    try:
        confirm_turn_reference_answer("session-a", 2, preview["confirm_token"])
    except ValueError as exc:
        assert "different turn" in str(exc)
    else:
        raise AssertionError("cross-turn preview confirmation must fail")


def test_parse_jsonl_and_csv_gold_formats() -> None:
    parsed_jsonl = parse_reference_dataset(
        "gold.jsonl",
        '{"prompt":"Q","answer":"A"}\n{"prompt":"Q2","answer":"A2"}\n',
    )
    parsed_csv = parse_reference_dataset(
        "gold.csv",
        "instruction,standard_answer,keywords\nQ,A,\"one,two\"\n",
    )

    assert [case["expected_answer"] for case in parsed_jsonl["cases"]] == ["A", "A2"]
    assert parsed_csv["cases"][0]["question"] == "Q"
    assert parsed_csv["cases"][0]["expected_answer"] == "A"
    assert parsed_csv["cases"][0]["keywords"] == ["one", "two"]


def test_gold_evidence_allows_process_only_and_generic_fallback() -> None:
    assert has_gold_evidence({"process_requirements": {"required_tools": ["search"]}})
    assert not has_gold_evidence({"question": "Inspect this trace"})


def test_structured_gold_does_not_default_missing_assertions_to_full_credit(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AGENT_QUALITY_EVAL_HOME", str(tmp_path))
    summary = upload_reference_dataset(
        "semantic.json",
        json.dumps(
            {
                "question": "How should regression work?",
                "expected_answer": "A candidate should preserve baseline capabilities.",
                "keywords": ["candidate", "baseline"],
            }
        ),
    )
    result = evaluate_turn_against_reference(
        session_id="semantic",
        turn_index=1,
        dataset_id=summary["dataset_id"],
        case_id=None,
        turn_eval={"metrics": {"user_query": "How should regression work?"}},
        turn_context={
            "turn": {
                "user_query": "How should regression work?",
                "final_response": "Compare the candidate with its baseline and reject capability loss.",
                "steps": [],
            }
        },
    )

    assert result["deterministic"]["assertion_score"] is None
    assert result["deterministic"]["keyword_coverage"] == 1.0
    assert result["verdict"] in {"pass", "partial"}


def test_arbitrary_upload_uses_configured_judge_without_overwriting_source_answer(
    monkeypatch,
) -> None:
    import agent_quality_eval.evaluation.providers as providers
    import agent_quality_eval.evaluation.settings as settings

    class FakeSettings:
        enabled = True
        provider = "test"
        model = "normalizer"

        def to_provider_config(self):
            return {"type": "test"}

    class FakeProvider:
        def call(self, prompt):
            assert "无需" not in prompt
            return SimpleNamespace(
                error=None,
                output=json.dumps(
                    {
                        "question": "",
                        "expected_answer": "模型不应覆盖原始答案",
                        "rubric": "应覆盖安装、重启和验证。",
                        "keywords": ["安装", "重启", "验证"],
                        "assertions": [],
                        "process_requirements": {},
                        "source_paths": {
                            "rubric": "derived_from_expected_answer",
                            "keywords": "derived_from_expected_answer",
                        },
                        "confidence": "high",
                        "notes": [],
                    },
                    ensure_ascii=False,
                ),
                performance=SimpleNamespace(token_usage={"total_tokens": 32}),
            )

    monkeypatch.setattr(settings, "load_critic_settings", lambda: FakeSettings())
    monkeypatch.setattr(providers, "load_provider", lambda config: FakeProvider())

    original_answer = "安装 hook 后需要重启 IDE，并开启新会话验证 trace。"
    preview = preview_reference_upload(
        "my-final-answer.txt",
        original_answer,
        issue_token=False,
    )

    normalized = preview["canonical"]["reference_answer"]
    assert normalized["expected_answer"] == original_answer
    assert normalized["keywords"] == ["安装", "重启", "验证"]
    assert preview["normalization_method"] == "llm-assisted-schema-normalization"
    assert preview["normalization_report"]["llm_normalization"]["status"] == "completed"


def test_unknown_private_schema_can_be_semantically_normalized(monkeypatch) -> None:
    import agent_quality_eval.evaluation.providers as providers
    import agent_quality_eval.evaluation.settings as settings

    class FakeSettings:
        enabled = True
        provider = "test"
        model = "normalizer"

        def to_provider_config(self):
            return {"type": "test"}

    response = SimpleNamespace(
        error=None,
        output=json.dumps(
            {
                "question": "是否需要重启？",
                "expected_answer": "需要重启 IDE。",
                "rubric": "明确说明需要重启。",
                "keywords": ["重启", "IDE"],
                "assertions": [],
                "process_requirements": {},
                "source_paths": {"expected_answer": "payload.result"},
                "confidence": "medium",
                "notes": [],
            },
            ensure_ascii=False,
        ),
        performance=SimpleNamespace(token_usage={}),
    )
    monkeypatch.setattr(settings, "load_critic_settings", lambda: FakeSettings())
    monkeypatch.setattr(
        providers,
        "load_provider",
        lambda config: SimpleNamespace(call=lambda prompt: response),
    )

    preview = preview_reference_upload(
        "private.json",
        json.dumps({"payload": {"result": "需要重启 IDE。"}}, ensure_ascii=False),
        issue_token=False,
    )

    normalized = preview["canonical"]["reference_answer"]
    assert normalized["expected_answer"] == "需要重启 IDE。"
    assert preview["eval_mode"] == "gold"


def test_llm_can_extract_verbatim_answer_span_from_novice_note(monkeypatch) -> None:
    import agent_quality_eval.evaluation.providers as providers
    import agent_quality_eval.evaluation.settings as settings

    class FakeSettings:
        enabled = True
        provider = "test"
        model = "normalizer"

        def to_provider_config(self):
            return {"type": "test"}

    answer = "需要重启 IDE，并开启新会话验证。"
    response = SimpleNamespace(
        error=None,
        output=json.dumps(
            {
                "question": "安装 hook 后要做什么？",
                "expected_answer": answer,
                "rubric": "说明重启和验证。",
                "keywords": ["重启", "验证"],
                "assertions": [],
                "process_requirements": {},
                "source_paths": {"expected_answer": "plain_text"},
                "confidence": "high",
                "notes": [],
            },
            ensure_ascii=False,
        ),
        performance=SimpleNamespace(token_usage={}),
    )
    monkeypatch.setattr(settings, "load_critic_settings", lambda: FakeSettings())
    monkeypatch.setattr(
        providers,
        "load_provider",
        lambda config: SimpleNamespace(call=lambda prompt: response),
    )
    raw = f"这是我保存的一份资料。\n\n用户大概在问安装 hook 后怎么处理。\n\n{answer}"

    preview = preview_reference_upload("note.txt", raw, issue_token=False)

    normalized = preview["canonical"]["reference_answer"]
    assert normalized["expected_answer"] == answer
    assert normalized["original_expected_answer"] == raw


def test_unknown_schema_without_judge_preserves_complete_upload(monkeypatch) -> None:
    import agent_quality_eval.evaluation.settings as settings

    class UnconfiguredSettings:
        enabled = True

        def to_provider_config(self):
            return None

    monkeypatch.setattr(settings, "load_critic_settings", lambda: UnconfiguredSettings())
    raw = json.dumps({"private_wrapper": {"result_blob": "my final answer"}})

    preview = preview_reference_upload("private.json", raw, issue_token=False)

    normalized = preview["canonical"]["reference_answer"]
    assert normalized["expected_answer"] == raw
    assert preview["normalization_method"] == "whole-content-fallback"
    assert any(
        item["source_path"] == "whole_file"
        for item in preview["normalization_report"]["mapping"]
    )


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
