from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from ai.config import AISettings


def test_settings_read_gigachat_configuration_without_exposing_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "gigachat")
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "secret-value")
    monkeypatch.setenv("UNRELATED_VARIABLE", "ignored")

    settings = AISettings(_env_file=None)

    assert settings.llm_provider == "gigachat"
    assert isinstance(settings.gigachat_credentials, SecretStr)
    assert settings.gigachat_credentials.get_secret_value() == "secret-value"
    assert "secret-value" not in repr(settings)


def test_settings_reject_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    with pytest.raises(ValidationError):
        AISettings(_env_file=None)
