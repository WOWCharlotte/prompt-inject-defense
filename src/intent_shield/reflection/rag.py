from __future__ import annotations

import json

from intent_shield.embed.base import BaseReranker
from intent_shield.reflection.store import ThreatStore
from intent_shield.types import RetrievedCase


class ThreatRAG:
    """RAG pipeline: embed query -> ChromaDB search -> rerank -> Top-K cases."""

    def __init__(
        self,
        store: ThreatStore,
        reranker: BaseReranker,
        top_k: int = 3,
        relevance_threshold: float = 0.5,
    ):
        self.store = store
        self.reranker = reranker
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

    def retrieve(
        self, user_instruction: str, tool_name: str, tool_params: dict
    ) -> list[RetrievedCase]:
        """Retrieve one tool's Top-K documents above the strict rerank threshold."""
        if self.top_k <= 0:
            return []

        query_block = (
            f"{user_instruction} | {tool_name} | "
            f"{json.dumps(tool_params, ensure_ascii=False)}"
        )

        # Stage 1: dense retrieval (fetch extra candidates for reranking)
        candidates = self.store.search(query_block, tool_name=tool_name, n_results=20)
        if not candidates:
            return []

        # Stage 2: rerank with cross-encoder
        documents = [case.text for case in candidates]

        reranked = self.reranker.rerank(
            query=query_block,
            documents=documents,
            top_n=min(self.top_k, len(candidates)),
        )

        # Keep the stored content and both scores, rather than rebuilding metadata.
        results = []
        for idx, score in sorted(reranked, key=lambda item: item[1], reverse=True):
            if score > self.relevance_threshold:
                results.append(
                    candidates[idx].model_copy(update={"rerank_score": float(score)})
                )
        return results[:self.top_k]
