from __future__ import annotations

import hashlib
import json

import chromadb

from intent_shield.embed.base import BaseEmbedder
from intent_shield.types import ReflectionCase, RetrievedCase


class ThreatStore:
    """ChromaDB-backed vector store for historical threat cases."""

    def __init__(self, persist_dir: str, collection_name: str, embedder: BaseEmbedder):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = embedder

    def add_case(self, case: ReflectionCase) -> str:
        """Store a case once per text hash, without embedding duplicate text."""
        text_block = self._case_to_text(case)
        doc_id = f"case_{hashlib.sha256(text_block.encode('utf-8')).hexdigest()}"
        if self.collection.get(ids=[doc_id], include=[])["ids"]:
            return doc_id
        embedding = self.embedder.embed([text_block])[0]
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text_block],
            metadatas=[{"source_tool": case.source_tool}],
        )
        return doc_id

    def search(
        self, query_text: str, tool_name: str, n_results: int = 10
    ) -> list[RetrievedCase]:
        """Search within one tool's cases; return documents and similarity scores."""
        count = self.collection.count()
        if count == 0 or n_results <= 0:
            return []
        where = {"source_tool": tool_name}
        if not self.collection.get(where=where, limit=1, include=[])["ids"]:
            return []
        embedding = self.embedder.embed([query_text])[0]
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, count),
            where=where,
            include=["documents", "distances"],
        )
        if not results["documents"] or not results["documents"][0]:
            return []
        return [
            RetrievedCase(
                document_id=doc_id,
                text=document,
                source_tool=tool_name,
                similarity_score=1.0 - float(distance),
            )
            for doc_id, document, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0],
                strict=True,
            )
        ]

    def count(self) -> int:
        return self.collection.count()

    @staticmethod
    def _case_to_text(case: ReflectionCase) -> str:
        params = case.source_params
        try:
            params = json.dumps(
                json.loads(params), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        except json.JSONDecodeError:
            pass
        return (
            f"攻击特征: {case.attack_feature}\n"
            f"失效原因: {case.failed_reason}\n"
            f"防御指南: {case.new_guideline}\n"
            f"来源指令: {case.source_instruction}\n"
            f"来源工具: {case.source_tool}\n"
            f"工具参数: {params}"
        )
