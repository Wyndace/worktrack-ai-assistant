"""Litestar HTTP API for the WorkTrack AI assistant."""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from granian import Granian
from granian.constants import Interfaces
from litestar import Litestar, Request, Response, get, post
from litestar.datastructures import State
from litestar.exceptions import ValidationException
from litestar.openapi import OpenAPIConfig
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_502_BAD_GATEWAY,
)
from pydantic import BaseModel, Field, field_validator

from ai.config import AISettings
from ai.run_agent import run

Runner = Callable[[str], tuple[str, Path]]
MAX_BODY_SIZE = 64 * 1024
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Non-sensitive service readiness information."""

    status: str
    service: str
    provider: str
    model: str


class AssessmentRequest(BaseModel):
    """Natural-language WorkTrack task submitted for assessment."""

    request: str = Field(min_length=1, max_length=MAX_BODY_SIZE)

    @field_validator("request")
    @classmethod
    def strip_request(cls, value: str) -> str:
        """Reject whitespace-only requests and normalize surrounding spaces."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("Поле request должно быть непустой строкой.")
        return normalized


class AssessmentResponse(BaseModel):
    """Successful assessment response."""

    status: str
    response: str
    trace: str


class ErrorResponse(BaseModel):
    """Stable public error response without provider internals."""

    status: str = "error"
    code: str
    message: str


def _model_name(settings: AISettings) -> str:
    return (
        settings.gigachat_model
        if settings.llm_provider == "gigachat"
        else settings.ollama_model
    )


@get(path="/health", status_code=HTTP_200_OK)
async def health(request: Request[Any, Any, Any]) -> HealthResponse:
    """Return process readiness and the configured LLM provider."""

    settings: AISettings = request.app.state.settings
    return HealthResponse(
        status="ok",
        service="worktrack-ai-assistant",
        provider=settings.llm_provider,
        model=_model_name(settings),
    )


@post(path="/assess", status_code=HTTP_200_OK)
async def assess(
    request: Request[Any, Any, Any], data: AssessmentRequest
) -> AssessmentResponse | Response[ErrorResponse]:
    """Assess one natural-language WorkTrack task through GigaChat and the Tool."""

    runner: Runner = request.app.state.runner
    assessment_lock: asyncio.Lock = request.app.state.assessment_lock
    try:
        # run() sets a process-wide trace target, so serialize assessments until
        # tracing is migrated to request-local context.
        async with assessment_lock:
            response, trace_path = await asyncio.to_thread(runner, data.request)
    except Exception as error:
        logger.error("GigaChat assessment failed (%s)", type(error).__name__)
        return Response(
            content=ErrorResponse(
                code="assessment_failed",
                message="GigaChat не выполнил оценку задачи.",
            ),
            status_code=HTTP_502_BAD_GATEWAY,
        )

    return AssessmentResponse(
        status="ok",
        response=response,
        trace=str(trace_path),
    )


def validation_exception_handler(
    _: Request[Any, Any, Any], __: ValidationException
) -> Response[ErrorResponse]:
    """Return the API's stable error contract for invalid request bodies."""

    return Response(
        content=ErrorResponse(
            code="invalid_request",
            message="Поле request должно быть непустой строкой длиной до 64 КиБ.",
        ),
        status_code=HTTP_400_BAD_REQUEST,
    )


def create_app(
    *,
    settings: AISettings | None = None,
    runner: Runner = run,
) -> Litestar:
    """Create a Litestar application with replaceable runtime dependencies."""

    resolved_settings = settings or AISettings()
    return Litestar(
        route_handlers=[health, assess],
        state=State(
            {
                "settings": resolved_settings,
                "runner": runner,
                "assessment_lock": asyncio.Lock(),
            }
        ),
        exception_handlers={ValidationException: validation_exception_handler},
        openapi_config=OpenAPIConfig(
            title="WorkTrack AI Assistant",
            version="0.1.0",
        ),
        request_max_body_size=MAX_BODY_SIZE,
    )


app = create_app()


def main() -> None:
    """Run the Litestar ASGI application with Granian."""

    settings = AISettings()
    Granian(
        "ai.server:app",
        address=settings.app_host,
        port=settings.app_port,
        interface=Interfaces.ASGI,
    ).serve()


if __name__ == "__main__":
    main()
