"""Configuration loading for declarative eval pipelines."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models import TestCase


@dataclass
class EvalConfig:
    name: str
    providers: list[dict[str, Any]]
    tests: list[TestCase]
    default_assertions: list[dict[str, Any]] = field(default_factory=list)
    judge: dict[str, Any] | None = None
    num_trials: int = 1
    pass_threshold: float = 0.6
    dataset_name: str = "default"
    dataset_version: str = "v1"
    output_dir: str = "./results"
    output_file: str | None = None
    report_file: str | None = None
    store_path: str | None = None
    baseline: str | None = None
    regression_policy: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    base_dir: Path = field(default_factory=Path.cwd)


def _read_structured_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _load_tests(raw_tests: Any, base_dir: Path) -> list[TestCase]:
    if raw_tests is None:
        return []
    if isinstance(raw_tests, (str, os.PathLike)):
        tests_path = Path(raw_tests)
        if not tests_path.is_absolute():
            tests_path = base_dir / tests_path
        data = _read_structured_file(tests_path)
        if isinstance(data, dict):
            raw_tests = data.get("tests", data.get("questions", []))
        else:
            raw_tests = data
    if isinstance(raw_tests, dict):
        raw_tests = raw_tests.get("tests", raw_tests.get("questions", []))
    tests: list[TestCase] = []
    for idx, item in enumerate(raw_tests or [], 1):
        if isinstance(item, str):
            item = {"id": str(idx), "question": item}
        tests.append(TestCase.from_dict(item, index=idx))
    return tests


def load_eval_config(config_path: str | os.PathLike[str]) -> EvalConfig:
    path = Path(config_path).resolve()
    data = _read_structured_file(path)
    base_dir = path.parent
    settings = data.get("settings", {}) or {}
    dataset = data.get("dataset", data.get("datasets", {})) or {}
    if isinstance(dataset, list):
        dataset = dataset[0] if dataset else {}
    output = data.get("output", settings.get("output", {})) or {}
    regression_policy = data.get("regression_policy", data.get("regression", {})) or {}

    default_assertions = (
        data.get("default_assertions")
        or data.get("defaultAssertions")
        or data.get("global_assertions")
        or []
    )

    tests = _load_tests(data.get("tests", data.get("questions")), base_dir)
    if not tests:
        raise ValueError(f"No tests found in {path}")

    providers = data.get("providers") or data.get("provider")
    if isinstance(providers, dict):
        providers = [providers]
    if not providers:
        raise ValueError(f"No providers configured in {path}")

    return EvalConfig(
        name=str(data.get("name") or data.get("pipeline_name") or path.stem),
        providers=list(providers),
        tests=tests,
        default_assertions=list(default_assertions),
        judge=data.get("judge") or data.get("llm_judge") or data.get("evaluation_mode"),
        num_trials=int(data.get("num_trials") or settings.get("num_trials") or 1),
        pass_threshold=float(
            data.get("pass_threshold")
            or settings.get("pass_threshold")
            or data.get("pass_criteria", {}).get("min_overall_pass_rate")
            or 0.6
        ),
        dataset_name=str(dataset.get("name") or data.get("dataset_name") or path.stem),
        dataset_version=str(dataset.get("version") or data.get("dataset_version") or "v1"),
        output_dir=str(output.get("dir") or settings.get("output_dir") or "./results"),
        output_file=output.get("json") or settings.get("output_file"),
        report_file=output.get("html") or settings.get("report_file"),
        store_path=data.get("store_path") or settings.get("store_path"),
        baseline=data.get("baseline") or settings.get("baseline"),
        regression_policy=regression_policy,
        metadata=dict(data.get("metadata") or {}),
        base_dir=base_dir,
    )


def write_default_config(path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sample = {
        "name": "sample-agent-eval",
        "providers": [
            {
                "type": "mock",
                "name": "candidate",
                "responses": {
                    "你好": "你好，我可以帮助你完成问题分析、工具调用和结果总结。",
                    "返回 JSON": '{"ok": true, "items": [1, 2, 3]}',
                },
                "default_response": "这是一个 mock agent 回答，包含 ID 10086，且没有错误。",
            }
        ],
        "settings": {
            "num_trials": 1,
            "pass_threshold": 0.6,
            "output_dir": "./results",
        },
        "dataset": {"name": "sample", "version": "v1"},
        "defaultAssertions": [
            {"type": "non-empty"},
            {"type": "no-error"},
            {"type": "response-time", "value": 30},
        ],
        "tests": [
            {
                "id": "hello",
                "question": "你好",
                "assert": [{"type": "min-length", "value": 10}],
            },
            {
                "id": "json",
                "question": "返回 JSON",
                "assert": [{"type": "contains-json"}],
            },
        ],
    }
    target.write_text(yaml.safe_dump(sample, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target
