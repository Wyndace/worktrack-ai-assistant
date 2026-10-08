"""Single construction point for every supported LLM provider."""

from crewai import LLM
from crewai.llms.base_llm import BaseLLM

from ai.config import AISettings
from ai.gigachat_llm import GigaChatCrewLLM


def get_llm(settings: AISettings | None = None) -> BaseLLM:
    """Build the provider selected in ``.env`` without changing application code."""

    settings = settings or AISettings()
    if settings.llm_provider == "gigachat":
        if settings.gigachat_credentials is None:
            raise ValueError(
                "GIGACHAT_CREDENTIALS не задан. Скопируйте .env.example в .env "
                "и добавьте персональный ключ GigaChat API."
            )
        return GigaChatCrewLLM(
            credentials=settings.gigachat_credentials.get_secret_value(),
            model=settings.gigachat_model,
            scope=settings.gigachat_scope,
            verify_ssl_certs=settings.gigachat_verify_ssl_certs,
        )

    return LLM(
        model=f"ollama/{settings.ollama_model}",
        base_url=settings.ollama_base_url,
        temperature=0.1,
    )
