from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for text embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a list of texts."""
        ...


class BaseReranker(ABC):
    """Abstract base class for reranking providers."""

    @abstractmethod
    def rerank(self, query: str, documents: list[str], top_n: int = 3) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query.

        Returns list of (original_index, relevance_score) sorted by score descending.
        """
        ...
