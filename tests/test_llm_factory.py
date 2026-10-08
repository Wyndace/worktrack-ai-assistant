import pytest

from ai.config import AISettings
from ai.llm_factory import get_llm


def test_gigachat_requires_credentials() -> None:
    settings = AISettings(llm_provider="gigachat", gigachat_credentials=None)

    with pytest.raises(ValueError, match="GIGACHAT_CREDENTIALS"):
        get_llm(settings)
