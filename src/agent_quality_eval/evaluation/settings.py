"""Persist local eval settings for the agent critic sidecar."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from ._secret_store import decrypt_secret, encrypt_secret, is_encrypted
from .store import default_home


SETTINGS_PATH = default_home() / "config" / "settings.json"


PROVIDER_MODELS: dict[str, list[str]] = {
    "timiai": ["gpt-4o-mini", "gpt-5.4", "gpt-4o", "glm-5.1", "glm-5.2"],
    "deepseek": ["deepseek-v4-flash", "deepseek-v4-pro"],
}


@dataclass
class CriticSettings:
    enabled: bool = True
    provider: str = "timiai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    timeout: int = 120

    def masked(self) -> dict[str, Any]:
        data = asdict(self)
        data["api_key"] = ""
        data["has_api_key"] = bool(self.api_key)
        data["models"] = PROVIDER_MODELS
        return data

    def to_provider_config(self) -> dict[str, Any] | None:
        if not self.enabled or not self.api_key.strip():
            return None
        provider = self.provider.lower()
        if provider == "deepseek":
            return {
                "type": "deepseek",
                "api_url": "https://api.deepseek.com/chat/completions",
                "api_key": self.api_key,
                "model": self.model or "deepseek-v4-flash",
                "threshold": 0.7,
                "timeout": self.timeout,
                "max_tokens": 8192,
                "response_format": {"type": "json_object"},
            }
        return {
            "type": "timiai",
            "api_url": "http://api.timiai.woa.com/ai_api_manage/llmproxy/chat/completions",
            "api_key": self.api_key,
            "model": self.model or "gpt-4o-mini",
            "threshold": 0.7,
            "timeout": self.timeout,
            "max_tokens": 8192,
            "response_format": {"type": "json_object"},
        }

    def to_judge_config(self) -> dict[str, Any] | None:
        """Backward-compatible alias for older code paths."""
        return self.to_provider_config()


LlmJudgeSettings = CriticSettings


def _read_settings_blob() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def load_critic_settings() -> CriticSettings:
    raw = _read_settings_blob()
    critic = raw.get("critic") if isinstance(raw, dict) else {}
    if not isinstance(critic, dict):
        critic = raw.get("llm_judge") if isinstance(raw.get("llm_judge"), dict) else {}
    provider = str(critic.get("provider") or "timiai").lower()
    if provider not in PROVIDER_MODELS:
        provider = "timiai"
    model = str(critic.get("model") or (PROVIDER_MODELS[provider][0]))
    if model not in PROVIDER_MODELS[provider]:
        model = PROVIDER_MODELS[provider][0]
    try:
        timeout = int(critic.get("timeout") or 120)
    except (TypeError, ValueError):
        timeout = 120
    # Prefer the encrypted field. Fall back to legacy plaintext ``api_key``
    # for one migration cycle: after the first ``save_critic_settings``, that
    # legacy key is stripped from disk. We deliberately do not migrate here
    # to keep read paths side-effect free.
    stored_key = str(critic.get("api_key_enc") or "")
    if stored_key:
        api_key = decrypt_secret(stored_key)
    else:
        api_key = str(critic.get("api_key") or "")
    return CriticSettings(
        enabled=bool(critic.get("enabled", True)),
        provider=provider,
        model=model,
        api_key=api_key,
        timeout=max(15, min(timeout, 300)),
    )


def save_critic_settings(payload: dict[str, Any]) -> CriticSettings:
    current = load_critic_settings()
    provider = str(payload.get("provider") or current.provider or "timiai").lower()
    if provider not in PROVIDER_MODELS:
        provider = "timiai"
    api_key = payload.get("api_key")
    if api_key in {None, "", "********"}:
        api_key = current.api_key
    try:
        timeout = int(payload.get("timeout") or current.timeout or 120)
    except (TypeError, ValueError):
        timeout = current.timeout
    model = str(payload.get("model") or current.model or PROVIDER_MODELS[provider][0])
    if model not in PROVIDER_MODELS[provider]:
        model = PROVIDER_MODELS[provider][0]
    settings = CriticSettings(
        enabled=bool(payload.get("enabled", current.enabled)),
        provider=provider,
        model=model,
        api_key=str(api_key or ""),
        timeout=max(15, min(timeout, 300)),
    )
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = _read_settings_blob()

    def _persisted_view(s: CriticSettings) -> dict[str, Any]:
        # Stored form: never includes plain-text ``api_key``. Instead we hold
        # ``api_key_enc`` (opaque, decryptable only by the current OS user).
        # ``has_api_key`` is a hint for external readers of the file.
        blob = asdict(s)
        blob.pop("api_key", None)
        blob["api_key_enc"] = encrypt_secret(s.api_key) if s.api_key else ""
        blob["has_api_key"] = bool(s.api_key)
        return blob

    persisted = _persisted_view(settings)
    raw["critic"] = persisted
    # Keep the legacy key in sync so older frontend bundles and configs keep
    # working, while the runtime treats it as Agent Critic settings.
    raw["llm_judge"] = persisted
    SETTINGS_PATH.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return settings


def load_llm_judge_settings() -> CriticSettings:
    return load_critic_settings()


def save_llm_judge_settings(payload: dict[str, Any]) -> CriticSettings:
    return save_critic_settings(payload)
