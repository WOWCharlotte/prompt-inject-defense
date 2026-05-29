from __future__ import annotations

from openai import OpenAI

from intent_shield.llm.base import BaseLLM


class OpenAICompatLLM(BaseLLM):
    """LLM client compatible with any OpenAI-format API (OpenAI, Qwen, Ollama, vLLM)."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def chat_sync(self, messages: list[dict], temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""
