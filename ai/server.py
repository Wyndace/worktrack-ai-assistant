"""Minimal HTTP API for the WorkTrack AI assistant."""

import json
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from ai.config import AISettings
from ai.run_agent import run

Runner = Callable[[str], tuple[str, Path]]
MAX_BODY_SIZE = 64 * 1024


class RequestError(ValueError):
    """A safe validation error that can be returned to an API client."""


def _health_payload(settings: AISettings) -> dict[str, Any]:
    model = (
        settings.gigachat_model
        if settings.llm_provider == "gigachat"
        else settings.ollama_model
    )
    return {
        "status": "ok",
        "service": "worktrack-ai-assistant",
        "provider": settings.llm_provider,
        "model": model,
    }


def _parse_assessment_request(body: bytes) -> str:
    if len(body) > MAX_BODY_SIZE:
        raise RequestError("Тело запроса превышает 64 КиБ.")

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise RequestError("Тело запроса должно быть корректным JSON.") from None

    if not isinstance(payload, dict):
        raise RequestError("Корневое значение должно быть JSON-объектом.")

    request = payload.get("request")
    if not isinstance(request, str) or not request.strip():
        raise RequestError("Поле request должно быть непустой строкой.")
    return request.strip()


def create_server(
    settings: AISettings | None = None,
    runner: Runner = run,
) -> HTTPServer:
    """Create a configured HTTP server without starting its event loop."""

    resolved_settings = settings or AISettings()

    class Handler(BaseHTTPRequestHandler):
        server_version = "WorkTrackAI/0.1"

        def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            response = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._write_json(HTTPStatus.OK, _health_payload(resolved_settings))
                return
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"status": "error", "code": "not_found"},
            )

        def do_POST(self) -> None:
            if self.path != "/assess":
                self._write_json(
                    HTTPStatus.NOT_FOUND,
                    {"status": "error", "code": "not_found"},
                )
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length < 0 or content_length > MAX_BODY_SIZE:
                    raise RequestError("Тело запроса превышает 64 КиБ.")
                request = _parse_assessment_request(self.rfile.read(content_length))
            except (RequestError, ValueError) as error:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "status": "error",
                        "code": "invalid_request",
                        "message": str(error),
                    },
                )
                return

            try:
                response, trace_path = runner(request)
            except Exception:
                self.log_error("assessment failed")
                self._write_json(
                    HTTPStatus.BAD_GATEWAY,
                    {
                        "status": "error",
                        "code": "assessment_failed",
                        "message": "GigaChat не выполнил оценку задачи.",
                    },
                )
                return

            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "response": response,
                    "trace": str(trace_path),
                },
            )

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}", flush=True)

    return HTTPServer(
        (resolved_settings.app_host, resolved_settings.app_port),
        Handler,
    )


def main() -> None:
    settings = AISettings()
    server = create_server(settings)
    print(
        f"WorkTrack AI слушает http://{settings.app_host}:{server.server_port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
