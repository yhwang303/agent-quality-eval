"""Provider registry and built-in agent callers."""

from __future__ import annotations

import importlib.util
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import requests

from .models import PerformanceMetrics, ProviderResponse


class BaseProvider:
    def __init__(self, config: dict[str, Any] | None = None, name: str | None = None):
        self.config = config or {}
        self.name = name or self.config.get("name") or self.__class__.__name__

    def call(self, input_text: str, **kwargs: Any) -> ProviderResponse:
        raise NotImplementedError


class MockProvider(BaseProvider):
    def call(self, input_text: str, **kwargs: Any) -> ProviderResponse:
        delay = float(self.config.get("delay", 0.01))
        time.sleep(delay)
        responses = self.config.get("responses", {}) or {}
        output = responses.get(input_text, self.config.get("default_response", "Mock response"))
        perf = PerformanceMetrics(
            total_duration=delay,
            time_to_first_token=delay,
            token_usage={"total_tokens": len(input_text.split()) + len(str(output).split())},
            tool_calls=list(self.config.get("tool_calls", [])),
        )
        trace = self.config.get("trace")
        return ProviderResponse(
            output=str(output),
            conversation_id=f"mock-{uuid.uuid4().hex[:8]}",
            performance=perf,
            trace=trace,
        )


class FunctionProvider(BaseProvider):
    def __init__(self, func: Callable[..., Any], config: dict[str, Any] | None = None, name: str | None = None):
        super().__init__(config, name or getattr(func, "__name__", "function"))
        self.func = func

    def call(self, input_text: str, **kwargs: Any) -> ProviderResponse:
        start = time.time()
        try:
            result = self.func(input_text, **kwargs)
            perf = PerformanceMetrics(total_duration=time.time() - start)
            if isinstance(result, ProviderResponse):
                if result.performance.total_duration == 0:
                    result.performance.total_duration = perf.total_duration
                return result
            if isinstance(result, tuple):
                output = result[0] if len(result) >= 1 else ""
                conv_id = result[1] if len(result) >= 2 else ""
                maybe_perf = result[2] if len(result) >= 3 else perf
                if not isinstance(maybe_perf, PerformanceMetrics):
                    maybe_perf = perf
                return ProviderResponse(str(output), str(conv_id), maybe_perf)
            if isinstance(result, dict):
                perf = PerformanceMetrics.from_dict(result.get("performance"))
                if perf.total_duration == 0:
                    perf.total_duration = float(result.get("response_time") or time.time() - start)
                return ProviderResponse(
                    output=str(result.get("output", result.get("answer", ""))),
                    conversation_id=str(result.get("conversation_id", "")),
                    performance=perf,
                    raw_response=result,
                    trace=result.get("trace"),
                    error=result.get("error"),
                )
            return ProviderResponse(output=str(result), performance=perf)
        except Exception as exc:
            return ProviderResponse(
                output="",
                performance=PerformanceMetrics(total_duration=time.time() - start),
                error=f"{type(exc).__name__}: {exc}",
            )


class OpenAICompatibleProvider(BaseProvider):
    def call(self, input_text: str, **kwargs: Any) -> ProviderResponse:
        api_url = self.config.get("api_url") or self.config.get("base_url")
        if api_url and not str(api_url).endswith("/chat/completions"):
            api_url = str(api_url).rstrip("/") + "/chat/completions"
        if not api_url:
            api_url = "https://api.openai.com/v1/chat/completions"
        token = self.config.get("api_token") or self.config.get("api_key")
        token = token or os.environ.get(self.config.get("api_key_env", "OPENAI_API_KEY"), "")
        if not token:
            return ProviderResponse(output="", error="missing OpenAI-compatible api token")

        ptype = str(self.config.get("type") or "").lower()
        is_timiai = ptype == "timiai" or "timiai.woa.com" in str(api_url)
        headers = {"Authorization": token if is_timiai else f"Bearer {token}", "Content-Type": "application/json"}
        system_prompt = self.config.get("system_prompt", "You are a helpful assistant.")
        body = {
            "model": self.config.get("model") or self.config.get("llm_model") or "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
            ],
            "max_tokens": int(self.config.get("max_tokens", 2048)),
        }
        if self.config.get("response_format"):
            body["response_format"] = self.config["response_format"]
        model_name = str(body["model"]).lower()
        if not model_name.startswith("gpt-5"):
            body["temperature"] = float(self.config.get("temperature", 0.0))
        start = time.time()
        try:
            response = requests.post(api_url, headers=headers, json=body, timeout=float(self.config.get("timeout", 120)))
            response.raise_for_status()
            data = response.json()
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            usage = data.get("usage") or {}
            return ProviderResponse(
                output=content,
                conversation_id=str(data.get("id", "")),
                performance=PerformanceMetrics(
                    total_duration=time.time() - start,
                    token_usage={
                        "total_tokens": usage.get("total_tokens", 0),
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                    },
                ),
                raw_response=data,
                trace={"finish_reason": choice.get("finish_reason")},
            )
        except Exception as exc:
            return ProviderResponse(
                output="",
                performance=PerformanceMetrics(total_duration=time.time() - start),
                error=f"{type(exc).__name__}: {exc}",
            )


class KnotAgentProvider(BaseProvider):
    def call(self, input_text: str, **kwargs: Any) -> ProviderResponse:
        api_url = self.config.get("api_url", "")
        api_token = self.config.get("api_token", "")
        if not api_url or not api_token:
            return ProviderResponse(output="", error="missing Knot api_url or api_token")
        stream = bool(self.config.get("stream", True))
        headers = {"x-knot-api-token": api_token}
        body = {
            "input": {
                "message": input_text,
                "conversation_id": kwargs.get("conversation_id", ""),
                "model": self.config.get("model", ""),
                "stream": stream,
                "chat_extra": {},
            }
        }
        if self.config.get("agent_client_uuid"):
            body["input"]["chat_extra"]["agent_client_uuid"] = self.config["agent_client_uuid"]

        start = time.time()
        perf = PerformanceMetrics()
        answer = ""
        conversation_id = ""
        try:
            session = requests.Session()
            response = session.post(
                api_url,
                json=body,
                headers=headers,
                stream=stream,
                timeout=float(self.config.get("timeout", 300)),
            )
            response.raise_for_status()
            first_token_seen = False
            for chunk in response.iter_lines():
                if not chunk:
                    continue
                line = chunk.decode("utf-8", errors="replace").strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line in {"", "[DONE]"}:
                    if line == "[DONE]":
                        break
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw = event.get("rawEvent", {})
                if raw.get("conversation_id") and not conversation_id:
                    conversation_id = raw["conversation_id"]
                etype = event.get("type")
                if etype == "TEXT_MESSAGE_CONTENT":
                    if not first_token_seen:
                        perf.time_to_first_token = time.time() - start
                        first_token_seen = True
                    content = raw.get("content") or event.get("content") or ""
                    if not content and isinstance(event.get("delta"), dict):
                        content = event["delta"].get("content", "")
                    answer += content
                    if self.config.get("verbose"):
                        print(content, end="", flush=True)
                elif etype == "STEP_STARTED":
                    perf.steps[raw.get("step_name", "unknown")] = {"start": time.time()}
                elif etype == "STEP_FINISHED":
                    step_name = raw.get("step_name", "unknown")
                    step = perf.steps.setdefault(step_name, {})
                    step["end"] = time.time()
                    if "start" in step:
                        step["duration"] = step["end"] - step["start"]
                    if raw.get("token_usage"):
                        perf.token_usage = raw["token_usage"]
                elif etype == "TOOL_CALL_START":
                    perf.tool_calls.append({"name": raw.get("name", "unknown"), "start": time.time(), "input": raw})
                elif etype == "TOOL_CALL_END" and perf.tool_calls:
                    perf.tool_calls[-1]["end"] = time.time()
                    perf.tool_calls[-1]["duration"] = perf.tool_calls[-1]["end"] - perf.tool_calls[-1].get("start", perf.tool_calls[-1]["end"])
            perf.total_duration = time.time() - start
            response.close()
            session.close()
            if self.config.get("verbose"):
                print()
            return ProviderResponse(output=answer, conversation_id=conversation_id, performance=perf)
        except Exception as exc:
            perf.total_duration = time.time() - start
            return ProviderResponse(output=answer, conversation_id=conversation_id, performance=perf, error=f"{type(exc).__name__}: {exc}")


def _load_python_provider(spec: str, base_dir: Path | None = None) -> FunctionProvider:
    _, rest = spec.split("python:", 1)
    parts = rest.split(":")
    path = Path(parts[0])
    func_name = parts[1] if len(parts) > 1 else "call_api"
    if not path.is_absolute() and base_dir:
        path = base_dir / path
    module_name = f"agent_quality_eval_user_provider_{path.stem}_{uuid.uuid4().hex[:8]}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if not module_spec or not module_spec.loader:
        raise ValueError(f"Cannot load python provider: {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    func = getattr(module, func_name)
    return FunctionProvider(func, name=path.stem)


def load_provider(config: dict[str, Any] | str | Callable[..., Any] | BaseProvider, base_dir: Path | None = None) -> BaseProvider:
    if isinstance(config, BaseProvider):
        return config
    if callable(config):
        return FunctionProvider(config)
    if isinstance(config, str):
        if config.startswith("python:"):
            return _load_python_provider(config, base_dir)
        raise ValueError(f"Unsupported provider string: {config}")
    ptype = str(config.get("type", "mock")).lower().replace("_", "-")
    name = config.get("name")
    if ptype in {"mock", "fixture"}:
        return MockProvider(config, name)
    if ptype in {"openai", "openai-compatible", "hunyuan", "knot-llm", "timiai", "deepseek"}:
        return OpenAICompatibleProvider(config, name)
    if ptype in {"knot", "knot-agent", "knot_agent"}:
        return KnotAgentProvider(config, name)
    if ptype.startswith("python:"):
        provider = _load_python_provider(ptype, base_dir)
        provider.config.update(config)
        provider.name = name or provider.name
        return provider
    raise ValueError(f"Unknown provider type: {ptype}")


def load_providers(configs: list[dict[str, Any]], base_dir: Path | None = None) -> list[BaseProvider]:
    return [load_provider(item, base_dir=base_dir) for item in configs]
