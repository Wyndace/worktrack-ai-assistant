"""Deterministic readiness assessment for WorkTrack work items."""

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

from crewai.tools import tool

ALLOWED_PRIORITIES = {"low", "medium", "high", "critical"}
AMBIGUOUS_PHRASES = (
    "нормально",
    "удобно",
    "быстро",
    "как-нибудь",
    "возможно",
    "по возможности",
    "и так далее",
    "и т.д.",
)


def _json_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _write_trace_event(work_item_json: str, result: str) -> None:
    trace_file = os.getenv("WORKTRACK_TRACE_FILE")
    if not trace_file:
        return

    path = Path(trace_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "event": "tool_called",
        "tool": "assess_work_item",
        "input": work_item_json,
        "output": json.loads(result),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _parse_payload(work_item_json: str) -> Any:
    try:
        return json.loads(work_item_json)
    except json.JSONDecodeError as error:
        if error.msg != "Extra data":
            raise

        trailing = work_item_json[error.pos :].strip()
        if trailing not in {"}", "]"}:
            raise
        return json.loads(work_item_json[: error.pos])


def _add_finding(
    findings: list[dict[str, str]],
    *,
    code: str,
    severity: str,
    message: str,
) -> None:
    findings.append({"code": code, "severity": severity, "message": message})


def assess_work_item_payload(work_item_json: str) -> str:
    """Assess one serialized WorkTrack work item and always return JSON."""

    try:
        payload = _parse_payload(work_item_json)
    except (json.JSONDecodeError, TypeError):
        return _json_result(
            {
                "status": "error",
                "code": "invalid_json",
                "message": "Передайте задачу как корректный JSON-объект.",
            }
        )

    if not isinstance(payload, dict):
        return _json_result(
            {
                "status": "error",
                "code": "invalid_payload",
                "message": "Корневое значение должно быть JSON-объектом задачи.",
            }
        )

    findings: list[dict[str, str]] = []
    recommendations: list[str] = []
    score = 100

    title = payload.get("title")
    if not _non_empty_string(title):
        score -= 20
        _add_finding(
            findings,
            code="missing_title",
            severity="blocker",
            message="Не задано краткое название задачи.",
        )
        recommendations.append("Добавьте название с ожидаемым изменением системы.")

    description = payload.get("description")
    if not _non_empty_string(description):
        score -= 25
        _add_finding(
            findings,
            code="missing_description",
            severity="blocker",
            message="Не описан ожидаемый результат задачи.",
        )
        recommendations.append(
            "Опишите текущее состояние, требуемое изменение "
            "и результат для пользователя."
        )
    elif len(description.strip()) < 40:
        score -= 10
        _add_finding(
            findings,
            code="description_too_short",
            severity="warning",
            message="Описание короче 40 символов и, вероятно, не задаёт контекст.",
        )
        recommendations.append(
            "Расширьте описание задачи минимум до одного полного сценария."
        )

    criteria = payload.get("acceptance_criteria")
    valid_criteria = (
        isinstance(criteria, list)
        and bool(criteria)
        and all(_non_empty_string(item) for item in criteria)
    )
    if not valid_criteria:
        score -= 30
        _add_finding(
            findings,
            code="missing_acceptance_criteria",
            severity="blocker",
            message="Нет проверяемых критериев приёмки.",
        )
        recommendations.append(
            "Добавьте критерии в формате: действие, наблюдаемый результат, ошибка."
        )
    elif len(criteria) < 2:
        score -= 10
        _add_finding(
            findings,
            code="insufficient_acceptance_criteria",
            severity="warning",
            message=(
                "Одного критерия недостаточно для основного и ошибочного сценариев."
            ),
        )
        recommendations.append("Добавьте отдельный критерий для ошибочного сценария.")

    priority = payload.get("priority")
    if priority not in ALLOWED_PRIORITIES:
        score -= 10
        _add_finding(
            findings,
            code="invalid_priority",
            severity="blocker",
            message="Приоритет должен быть low, medium, high или critical.",
        )
        recommendations.append("Укажите один из поддерживаемых приоритетов WorkTrack.")

    if not _non_empty_string(payload.get("assignee")):
        score -= 10
        _add_finding(
            findings,
            code="missing_assignee",
            severity="warning",
            message="Не назначен ответственный исполнитель.",
        )
        recommendations.append("Назначьте исполнителя или ответственную роль.")

    due_date = payload.get("due_date")
    if not _non_empty_string(due_date):
        score -= 5
        _add_finding(
            findings,
            code="missing_due_date",
            severity="warning",
            message="Не указан срок выполнения.",
        )
        recommendations.append("Укажите срок в формате YYYY-MM-DD.")
    else:
        try:
            date.fromisoformat(due_date)
        except ValueError:
            score -= 5
            _add_finding(
                findings,
                code="invalid_due_date",
                severity="warning",
                message="Срок должен быть календарной датой в формате YYYY-MM-DD.",
            )
            recommendations.append("Исправьте due_date на корректную ISO-дату.")

    dependencies = payload.get("dependencies", [])
    if not isinstance(dependencies, list) or not all(
        _non_empty_string(item) for item in dependencies
    ):
        score -= 5
        _add_finding(
            findings,
            code="invalid_dependencies",
            severity="warning",
            message="Зависимости должны быть массивом непустых идентификаторов.",
        )
        recommendations.append(
            "Передайте dependencies как массив идентификаторов задач."
        )

    combined_text = " ".join(
        value
        for value in (
            title if isinstance(title, str) else "",
            description if isinstance(description, str) else "",
            " ".join(criteria) if valid_criteria else "",
        )
        if value
    ).lower()
    ambiguous = sorted(
        phrase for phrase in AMBIGUOUS_PHRASES if phrase in combined_text
    )
    if ambiguous:
        score -= min(20, 5 * len(ambiguous))
        _add_finding(
            findings,
            code="ambiguous_language",
            severity="warning",
            message="Найдены неоднозначные формулировки: " + ", ".join(ambiguous),
        )
        recommendations.append(
            "Замените субъективные слова измеримыми значениями "
            "или наблюдаемым поведением."
        )

    score = max(0, score)
    has_blockers = any(item["severity"] == "blocker" for item in findings)
    return _json_result(
        {
            "status": "ok",
            "ready": score >= 75 and not has_blockers,
            "score": score,
            "findings": findings,
            "recommendations": recommendations,
        }
    )


@tool("assess_work_item")
def assess_work_item(work_item_json: str) -> str:
    """Проверяет готовность задачи WorkTrack к передаче исполнителю.

    Передай один JSON-объект строкой. Поля: title, description,
    acceptance_criteria (массив строк), priority (low, medium, high или critical),
    assignee, due_date (YYYY-MM-DD), dependencies (массив строк). Инструмент
    возвращает JSON со score, ready, findings и recommendations. Используй его
    для каждой проверки постановки; не оценивай полноту задачи самостоятельно.
    """

    result = assess_work_item_payload(work_item_json)
    _write_trace_event(work_item_json, result)
    return result
