"""Experiment runner for automated agent evals."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from .assertions import JudgeFunc, run_assertions
from .config import EvalConfig, load_eval_config
from .models import CaseResult, ExperimentResult, TrialResult
from .providers import BaseProvider, load_provider, load_providers
from .report import ReportService
from .store import DatasetStore


class ExperimentRunner:
    def __init__(
        self,
        config: EvalConfig,
        *,
        store: DatasetStore | None = None,
        providers: list[BaseProvider] | None = None,
    ):
        self.config = config
        self.store = store or DatasetStore(config.store_path)
        self.providers = providers or load_providers(config.providers, base_dir=config.base_dir)
        self.judge = self._create_judge(config.judge)

    def _create_judge(self, judge_config: dict[str, Any] | None) -> JudgeFunc | None:
        if not judge_config:
            return None
        if not judge_config.get("type") and judge_config.get("api_url"):
            # Knot AGUI judge is common internally; OpenAI-compatible users can set type explicitly.
            judge_config = {**judge_config, "type": judge_config.get("provider_type", "knot")}
        provider = load_provider(judge_config, base_dir=self.config.base_dir)

        def judge(question: str, answer: str, criteria: str, expected_answer: str | None = None) -> tuple[float, str]:
            prompt = _build_judge_prompt(question, answer, criteria, expected_answer)
            response = provider.call(prompt)
            if response.error:
                return 0.0, response.error
            return _parse_judge_response(response.output)

        return judge

    def run(self) -> ExperimentResult:
        self.store.upsert_dataset(
            self.config.dataset_name,
            self.config.dataset_version,
            self.config.tests,
            metadata=self.config.metadata,
        )
        experiment_id = f"exp-{uuid.uuid4().hex[:12]}"
        result = ExperimentResult(
            experiment_id=experiment_id,
            name=self.config.name,
            dataset_name=self.config.dataset_name,
            dataset_version=self.config.dataset_version,
            providers=[p.name for p in self.providers],
            metadata=self.config.metadata,
        )

        for provider in self.providers:
            for test_case in self.config.tests:
                case_result = CaseResult(test_case=test_case, provider_name=provider.name)
                num_trials = int(test_case.num_trials or self.config.num_trials or 1)
                assertions = list(test_case.assertions) + list(self.config.default_assertions)
                if not assertions:
                    assertions = [{"type": "non-empty"}]
                for trial_number in range(1, num_trials + 1):
                    response = provider.call(test_case.question)
                    trace = response.trace or test_case.trace
                    scores = run_assertions(
                        response.output,
                        assertions,
                        question=test_case.question,
                        expected_answer=test_case.expected_answer,
                        response_time=response.response_time,
                        performance=response.performance,
                        trace=trace,
                        llm_judge_func=self.judge,
                    )
                    if response.error:
                        scores.append(
                            _error_score(
                                name="provider-error",
                                reason=response.error,
                            )
                        )
                    trial = TrialResult(
                        trial_id=f"{experiment_id}-{provider.name}-{test_case.id}-{trial_number}".replace(" ", "_"),
                        trial_number=trial_number,
                        test_case_id=test_case.id,
                        provider_name=provider.name,
                        answer=response.output,
                        response_time=response.response_time,
                        conversation_id=response.conversation_id,
                        scores=scores,
                        performance=response.performance,
                        error=response.error,
                        trace=trace,
                    )
                    trial.compute(self.config.pass_threshold)
                    case_result.trials.append(trial)
                case_result.compute()
                result.case_results.append(case_result)

        result.compute()
        self.store.save_experiment(result)
        ReportService().write_outputs(result, self.config)
        return result


def _error_score(name: str, reason: str):
    from .models import ScoreResult

    return ScoreResult(
        name=name,
        type=name,
        score=0.0,
        passed=False,
        reason=reason,
        threshold=0.5,
    )


def _build_judge_prompt(
    question: str,
    answer: str,
    criteria: str,
    expected_answer: str | None = None,
) -> str:
    expected = f"\n## 参考答案\n{expected_answer}\n" if expected_answer else ""
    return f"""你是一个严格但公平的 Agent 质量评估器。请根据评估标准给出 0 到 1 的分数。

## 用户问题
{question}

## Agent 回答
{answer}
{expected}
## 评估标准
{criteria}

只输出 JSON，不要输出额外文本：
{{"score": 0.0, "reason": "评分理由"}}
"""


def _parse_judge_response(text: str) -> tuple[float, str]:
    text = (text or "").strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return 0.5, f"Judge 返回非 JSON: {text[:200]}"
    try:
        data = json.loads(match.group(0))
        return float(data.get("score", 0.0)), str(data.get("reason", ""))
    except Exception as exc:
        return 0.5, f"Judge JSON 解析失败: {exc}; raw={text[:200]}"


def run_eval(config_path: str | Path, *, db_path: str | None = None) -> ExperimentResult:
    config = load_eval_config(config_path)
    if db_path:
        config.store_path = db_path
    return ExperimentRunner(config).run()
