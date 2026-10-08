import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ai.config import AISettings
from ai.server import create_server


def _request_json(url: str, request: Request | None = None) -> tuple[int, dict]:
    try:
        with urlopen(request or url, timeout=2) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def test_http_health_and_assessment_endpoints() -> None:
    calls: list[str] = []

    def fake_runner(prompt: str) -> tuple[str, Path]:
        calls.append(prompt)
        return "ready, score 100", Path("ai/traces/test.jsonl")

    settings = AISettings(
        _env_file=None,
        app_host="127.0.0.1",
        app_port=0,
        gigachat_credentials="test-only",
    )
    server = create_server(settings, fake_runner)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        status, health = _request_json(f"{base_url}/health")
        body = json.dumps({"request": "Проверь WT-42"}).encode()
        assessment_request = Request(
            f"{base_url}/assess",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        assessment_status, assessment = _request_json(
            f"{base_url}/assess", assessment_request
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert status == 200
    assert health == {
        "status": "ok",
        "service": "worktrack-ai-assistant",
        "provider": "gigachat",
        "model": "GigaChat-2-Pro",
    }
    assert assessment_status == 200
    assert assessment["response"] == "ready, score 100"
    assert assessment["trace"] == "ai/traces/test.jsonl"
    assert calls == ["Проверь WT-42"]


def test_http_rejects_invalid_assessment_request() -> None:
    settings = AISettings(_env_file=None, app_host="127.0.0.1", app_port=0)
    server = create_server(settings)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request = Request(
        f"http://127.0.0.1:{server.server_port}/assess",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        status, response = _request_json(request.full_url, request)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert status == 400
    assert response["code"] == "invalid_request"
