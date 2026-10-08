"""CrewAI adapter backed by the official GigaChat LangChain integration."""

import json
import uuid
from typing import Any

from crewai.llms.base_llm import BaseLLM
from langchain_core.messages import AIMessage, convert_to_messages
from langchain_gigachat import GigaChat
from pydantic import PrivateAttr


class GigaChatCrewLLM(BaseLLM):
    """Expose GigaChat chat and function calling through CrewAI's ``BaseLLM``."""

    llm_type: str = "gigachat"
    provider: str = "gigachat"
    _client: Any = PrivateAttr()

    def __init__(
        self,
        *,
        credentials: str,
        model: str,
        scope: str,
        verify_ssl_certs: bool,
        temperature: float = 0.1,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            model=model,
            provider="gigachat",
            temperature=temperature,
        )
        self._client = client or GigaChat(
            credentials=credentials,
            model=model,
            scope=scope,
            verify_ssl_certs=verify_ssl_certs,
            temperature=temperature,
        )

    def call(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: Any | None = None,
    ) -> str | list[dict[str, Any]]:
        """Call GigaChat and normalize native tool calls for CrewAI."""

        del callbacks, available_functions, from_task, from_agent, response_model
        raw_messages = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else messages
        )
        langchain_messages = convert_to_messages(raw_messages)
        client = self._client
        if tools:
            client = client.bind_tools(tools, tool_choice="auto")

        response: AIMessage = client.invoke(
            langchain_messages,
            stop=self.stop_sequences or None,
        )
        if response.tool_calls:
            return [
                {
                    "id": call.get("id") or uuid.uuid4().hex,
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(
                            call.get("args", {}), ensure_ascii=False, sort_keys=True
                        ),
                    },
                }
                for call in response.tool_calls
            ]

        content = response.content
        if isinstance(content, str):
            return self._apply_stop_words(content)
        return self._apply_stop_words(str(content))

    def supports_function_calling(self) -> bool:
        """GigaChat supports one native function call per assistant message."""

        return True

    def supports_stop_words(self) -> bool:
        """Stop sequences are passed to the underlying LangChain client."""

        return True

    def get_context_window_size(self) -> int:
        """Return a conservative context limit suitable for agent prompts."""

        return 32_768
