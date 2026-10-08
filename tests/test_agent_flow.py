import json
from pathlib import Path
from typing import Any

import pytest
from crewai import Agent, Crew, Process, Task
from langchain_core.messages import AIMessage

from ai.gigachat_llm import GigaChatCrewLLM
from ai.tools import assess_work_item


class ScriptedGigaChat:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = responses
        self.invocations: list[list[Any]] = []

    def bind_tools(
        self, tools: list[dict[str, Any]], *, tool_choice: str
    ) -> "ScriptedGigaChat":
        assert tools[0]["function"]["name"] == "assess_work_item"
        assert tool_choice == "auto"
        return self

    def invoke(self, messages: list[Any], *, stop: list[str] | None) -> AIMessage:
        assert stop in (None, ["\nObservation:"])
        self.invocations.append(messages)
        return self.responses.pop(0)


def test_crewai_executes_gigachat_tool_call_and_records_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    work_item = json.dumps(
        {
            "title": "Комментарии",
            "description": "Добавить комментарии к рабочему треку WorkTrack.",
            "acceptance_criteria": [],
            "priority": "high",
            "assignee": "backend",
            "due_date": "2026-10-15",
            "dependencies": [],
        },
        ensure_ascii=False,
    )
    client = ScriptedGigaChat(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "assess_work_item",
                        "args": {"work_item_json": work_item},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="not ready: отсутствуют критерии приёмки"),
        ]
    )
    llm = GigaChatCrewLLM(
        credentials="not-used-by-fake",
        model="GigaChat-2-Pro",
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=True,
        client=client,
    )
    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("WORKTRACK_TRACE_FILE", str(trace_path))

    agent = Agent(
        role="Аналитик WorkTrack",
        goal="Проверить постановку задачи",
        backstory="Всегда использует assess_work_item.",
        llm=llm,
        tools=[assess_work_item],
        allow_delegation=False,
        max_iter=3,
    )
    task = Task(
        description="Проверь постановку и вызови assess_work_item.",
        expected_output="Вердикт готовности.",
        agent=agent,
    )

    result = Crew(agents=[agent], tasks=[task], process=Process.sequential).kickoff()

    assert "not ready" in str(result)
    assert len(client.invocations) == 2
    trace_events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert trace_events[0]["event"] == "tool_called"
    assert trace_events[0]["output"]["ready"] is False
