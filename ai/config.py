"""Typed configuration for the assistant's LLM providers."""

from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AISettings(BaseSettings):
    """Load AI settings from environment variables and a local ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: Literal["gigachat", "ollama"] = "gigachat"

    gigachat_credentials: SecretStr | None = None
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_model: str = "GigaChat-2-Pro"
    gigachat_verify_ssl_certs: bool = True

    ollama_model: str = "qwen2.5-coder:7b"
    ollama_base_url: str = "http://localhost:11434"
