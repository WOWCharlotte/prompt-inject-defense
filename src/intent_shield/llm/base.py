from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat_sync(self, messages: list[dict], temperature: float = 0.0) -> str:
        """Send messages and return the assistant's text response."""
        ...

    @abstractmethod
    def parse_sync(
        self,
        messages: list[dict],
        text_format: type[ResponseT],
        temperature: float = 0.0,
    ) -> tuple[ResponseT, str]:
        """Return validated structured output and the original response text.

        Raise an exception if the response is incomplete or cannot be parsed.
        """
        ...
