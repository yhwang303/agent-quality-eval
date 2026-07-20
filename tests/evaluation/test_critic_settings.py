from __future__ import annotations

import json
from pathlib import Path


def test_critic_settings_models_include_new_versions() -> None:
    from agent_quality_eval.evaluation.settings import PROVIDER_MODELS

    assert "glm-5.1" in PROVIDER_MODELS["timiai"]
    assert "glm-5.2" in PROVIDER_MODELS["timiai"]
    assert PROVIDER_MODELS["deepseek"] == ["deepseek-v4-flash", "deepseek-v4-pro"]


def test_deprecated_deepseek_model_falls_back_to_v4(tmp_path: Path, monkeypatch) -> None:
    import agent_quality_eval.evaluation.settings as settings

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "critic": {
                    "enabled": True,
                    "provider": "deepseek",
                    "model": "deepseek-chat",
                    "api_key": "test-key",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "SETTINGS_PATH", settings_path)

    loaded = settings.load_critic_settings()

    assert loaded.provider == "deepseek"
    assert loaded.model == "deepseek-v4-flash"
    assert loaded.to_provider_config()["model"] == "deepseek-v4-flash"
    assert loaded.to_provider_config()["max_tokens"] == 8192
