from __future__ import annotations

import json

from intent_shield.embed.base import BaseReranker
from intent_shield.reflection.store import ThreatStore
from intent_shield.types import ReflectionCase


class ThreatRAG:
    """RAG pipeline: embed query -> ChromaDB search -> rerank -> Top-K cases."""

    def __init__(self, store: ThreatStore, reranker: BaseReranker, top_k: int = 3):
        self.store = store
        self.reranker = reranker
        self.top_k = top_k

    def retrieve(
        self, user_instruction: str, tool_name: str, tool_params: dict
    ) -> list[ReflectionCase]:
        """Retrieve top-K relevant historical threat cases."""
        if self.store.count() == 0:
            return []

        query_block = (
            f"{user_instruction} | {tool_name} | "
            f"{json.dumps(tool_params, ensure_ascii=False)}"
        )

        # Stage 1: dense retrieval (fetch extra candidates for reranking)
        candidates = self.store.search(query_block, n_results=min(20, self.store.count()))
        if not candidates:
            return []

        # Stage 2: rerank with cross-encoder
        documents = [doc for doc, _ in candidates]
        metadatas = [meta for _, meta in candidates]

        reranked = self.reranker.rerank(
            query=query_block,
            documents=documents,
            top_n=self.top_k,
        )

        # Build ReflectionCase objects from top results
        results = []
        for idx, _score in reranked:
            meta = metadatas[idx]
            results.append(
                ReflectionCase(
                    attack_feature=meta.get("attack_feature", ""),
                    failed_reason=meta.get("failed_reason", ""),
                    new_guideline=meta.get("new_guideline", ""),
                    source_instruction=meta.get("source_instruction", ""),
                    source_tool=meta.get("source_tool", ""),
                    source_params=meta.get("source_params", ""),
                )
            )
        return results
