"""Run the WorkTrack quality-gate agent and persist an auditable trace."""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from crewai import Agent, Crew, Process, Task

from ai.config import AISettings
from ai.llm_factory import get_llm
from ai.tools import assess_work_item

DEFAULT_REQUEST = """
Проверь постановку задачи WorkTrack и обязательно вызови assess_work_item:

Название: Добавить комментарии к рабочему треку.
Описание: Пользователь добавляет текстовый комментарий к существующему треку и
видит его в хронологической истории.
Критерии приёмки:
1. POST /tracks/{id}/comments возвращает 201 для валидного текста.
2. GET /tracks/{id}/comments сортирует комментарии по created_at.
3. Пустой комментарий возвращает 400 с кодом validation_error.
Приоритет: high. Исполнитель: backend. Срок: 2026-10-15. Зависимости: WT-01.
""".strip()


def _trace_path(provider: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("ai/traces") / f"run-{provider}-{stamp}.jsonl"


def run(request: str) -> tuple[str, Path]:
    """Execute one assessment and return the final response and trace path."""

    settings = AISettings()
    trace_path = _trace_path(settings.llm_provider)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["WORKTRACK_TRACE_FILE"] = str(trace_path.resolve())
    with trace_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "event": "run_started",
                    "provider": settings.llm_provider,
                    "model": (
                        settings.gigachat_model
                        if settings.llm_provider == "gigachat"
                        else settings.ollama_model
                    ),
                    "request": request,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )

    agent = Agent(
        role="Аналитик качества постановок WorkTrack",
        goal=(
            "До передачи задачи исполнителю выявлять неполноту, неоднозначность "
            "и непроверяемые критерии"
        ),
        backstory=(
            "Ты работаешь в команде WorkTrack. Для любой оценки постановки "
            "обязательно вызываешь assess_work_item и основываешь вывод только "
            "на его JSON-результате."
        ),
        llm=get_llm(settings),
        tools=[assess_work_item],
        allow_delegation=False,
        verbose=True,
        max_iter=4,
    )
    task = Task(
        description=request,
        expected_output=(
            "Краткий вердикт ready/not ready, score, список блокеров и конкретные "
            "правки постановки. До ответа обязательно вызвать assess_work_item."
        ),
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    result = str(crew.kickoff())

    with trace_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"event": "run_completed", "response": result},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
    return result, trace_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", nargs="?", default=DEFAULT_REQUEST)
    args = parser.parse_args()
    result, trace_path = run(args.request)
    print(result)
    print(f"Trace: {trace_path}")


if __name__ == "__main__":
    main()
