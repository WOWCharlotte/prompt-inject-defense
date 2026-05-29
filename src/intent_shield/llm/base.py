from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat_sync(self, messages: list[dict], temperature: float = 0.0) -> str:
        """Send messages and return the assistant's text response."""
        ...
