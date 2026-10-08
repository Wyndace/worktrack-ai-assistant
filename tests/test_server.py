from pathlib import Path

from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    HTTP_502_BAD_GATEWAY,
)
from litestar.testing import TestClient

from ai.config import AISettings
from ai.server import create_app


def _settings() -> AISettings:
    return AISettings(
        _env_file=None,
        app_host="127.0.0.1",
        app_port=8000,
        gigachat_credentials="test-only",
    )


def test_litestar_health_and_assessment_endpoints() -> None:
    calls: list[str] = []

    def fake_runner(prompt: str) -> tuple[str, Path]:
        calls.append(prompt)
        return "ready, score 100", Path("ai/traces/test.jsonl")

    app = create_app(settings=_settings(), runner=fake_runner)

    with TestClient(app=app) as client:
        health = client.get("/health")
        assessment = client.post(
            "/assess",
            json={"request": "  Проверь WT-42  "},
        )

    assert health.status_code == HTTP_200_OK
    assert health.json() == {
        "status": "ok",
        "service": "worktrack-ai-assistant",
        "provider": "gigachat",
        "model": "GigaChat-2-Pro",
    }
    assert assessment.status_code == HTTP_200_OK
    assert assessment.json() == {
        "status": "ok",
        "response": "ready, score 100",
        "trace": "ai/traces/test.jsonl",
    }
    assert calls == ["Проверь WT-42"]


def test_litestar_rejects_invalid_assessment_request() -> None:
    app = create_app(settings=_settings())

    with TestClient(app=app) as client:
        response = client.post("/assess", json={"request": "   "})

    assert response.status_code == HTTP_400_BAD_REQUEST
    assert response.json() == {
        "status": "error",
        "code": "invalid_request",
        "message": "Поле request должно быть непустой строкой длиной до 64 КиБ.",
    }


def test_litestar_rejects_oversized_request_body() -> None:
    app = create_app(settings=_settings())

    with TestClient(app=app) as client:
        response = client.post("/assess", json={"request": "x" * (65 * 1024)})

    assert response.status_code == HTTP_413_REQUEST_ENTITY_TOO_LARGE


def test_litestar_hides_provider_failure_details() -> None:
    def failing_runner(_: str) -> tuple[str, Path]:
        raise RuntimeError("secret provider details")

    app = create_app(settings=_settings(), runner=failing_runner)

    with TestClient(app=app, raise_server_exceptions=False) as client:
        response = client.post("/assess", json={"request": "Проверь задачу"})

    assert response.status_code == HTTP_502_BAD_GATEWAY
    assert response.json() == {
        "status": "error",
        "code": "assessment_failed",
        "message": "GigaChat не выполнил оценку задачи.",
    }
    assert "secret provider details" not in response.text
