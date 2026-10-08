import json
from typing import Any

from langchain_core.messages import AIMessage

from ai.gigachat_llm import GigaChatCrewLLM


class FakeGigaChat:
    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.bound_tools: list[dict[str, Any]] | None = None
        self.tool_choice: str | None = None

    def bind_tools(
        self, tools: list[dict[str, Any]], *, tool_choice: str
    ) -> "FakeGigaChat":
        self.bound_tools = tools
        self.tool_choice = tool_choice
        return self

    def invoke(self, messages: list[Any], *, stop: list[str] | None) -> AIMessage:
        assert messages
        assert stop is None
        return self.response


def _adapter(client: FakeGigaChat) -> GigaChatCrewLLM:
    return GigaChatCrewLLM(
        credentials="not-used-by-fake",
        model="GigaChat-2-Pro",
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=True,
        client=client,
    )


def test_adapter_returns_text_content() -> None:
    adapter = _adapter(FakeGigaChat(AIMessage(content="Готово")))

    assert adapter.call("Проверь задачу") == "Готово"


def test_adapter_normalizes_native_tool_call_for_crewai() -> None:
    fake = FakeGigaChat(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "assess_work_item",
                    "args": {"work_item_json": "{}"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
    )
    adapter = _adapter(fake)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "assess_work_item",
                "description": "Проверяет задачу",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    result = adapter.call("Проверь задачу", tools=tools)

    assert fake.bound_tools == tools
    assert fake.tool_choice == "auto"
    assert isinstance(result, list)
    assert result[0]["function"]["name"] == "assess_work_item"
    assert json.loads(result[0]["function"]["arguments"]) == {"work_item_json": "{}"}
