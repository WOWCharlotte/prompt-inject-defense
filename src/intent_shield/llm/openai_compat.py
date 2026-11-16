from __future__ import annotations

from openai import OpenAI

from intent_shield.llm.base import BaseLLM, ResponseT


class OpenAICompatLLM(BaseLLM):
    """OpenAI-compatible client; structured output requires the Responses API."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model

    def chat_sync(self, messages: list[dict], temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    def parse_sync(
        self,
        messages: list[dict],
        text_format: type[ResponseT],
        temperature: float = 0.0,
    ) -> tuple[ResponseT, str]:
        response = self.client.responses.parse(
            model=self.model,
            input=messages,
            text_format=text_format,
            temperature=temperature,
        )
        if response.status != "completed":
            raise ValueError(f"Structured LLM response was not completed: {response.status}")
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("LLM returned no structured output (refusal or empty response)")
        return parsed, response.output_text
