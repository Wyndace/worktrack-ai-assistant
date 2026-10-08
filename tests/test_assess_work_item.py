import json

from ai.tools.assess_work_item import assess_work_item_payload


def test_complete_work_item_is_ready() -> None:
    result = json.loads(
        assess_work_item_payload(
            json.dumps(
                {
                    "title": "Добавить комментарии к рабочему треку",
                    "description": (
                        "Пользователь добавляет текстовый комментарий к существующему "
                        "треку и видит его в хронологической истории."
                    ),
                    "acceptance_criteria": [
                        (
                            "POST /tracks/{id}/comments возвращает 201 "
                            "для валидного текста"
                        ),
                        (
                            "GET /tracks/{id}/comments возвращает комментарии "
                            "по created_at"
                        ),
                        (
                            "Пустой комментарий возвращает ошибку 400 "
                            "с кодом validation_error"
                        ),
                    ],
                    "priority": "high",
                    "assignee": "backend",
                    "due_date": "2026-10-15",
                    "dependencies": ["WT-01"],
                },
                ensure_ascii=False,
            )
        )
    )

    assert result["status"] == "ok"
    assert result["ready"] is True
    assert result["score"] >= 75
    assert not [item for item in result["findings"] if item["severity"] == "blocker"]


def test_vague_work_item_returns_actionable_findings() -> None:
    result = json.loads(
        assess_work_item_payload(
            json.dumps(
                {
                    "title": "Комментарии",
                    "description": (
                        "Сделать нормально и удобно, возможно как-нибудь быстро"
                    ),
                    "acceptance_criteria": [],
                    "priority": "urgent",
                },
                ensure_ascii=False,
            )
        )
    )

    codes = {item["code"] for item in result["findings"]}
    assert result["status"] == "ok"
    assert result["ready"] is False
    assert "missing_acceptance_criteria" in codes
    assert "invalid_priority" in codes
    assert "ambiguous_language" in codes
    assert result["recommendations"]


def test_invalid_json_is_returned_as_safe_error() -> None:
    result = json.loads(assess_work_item_payload("{not-json"))

    assert result == {
        "status": "error",
        "code": "invalid_json",
        "message": "Передайте задачу как корректный JSON-объект.",
    }


def test_non_object_payload_is_returned_as_safe_error() -> None:
    result = json.loads(assess_work_item_payload("[]"))

    assert result["status"] == "error"
    assert result["code"] == "invalid_payload"
