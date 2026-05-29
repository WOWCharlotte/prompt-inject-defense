from __future__ import annotations

import uuid

import chromadb

from intent_shield.embed.base import BaseEmbedder
from intent_shield.types import ReflectionCase


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
        """Embed and store a reflection case. Returns the document ID."""
        text_block = self._case_to_text(case)
        embedding = self.embedder.embed([text_block])[0]
        doc_id = f"case_{uuid.uuid4().hex[:12]}"
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text_block],
            metadatas=[case.model_dump()],
        )
        return doc_id

    def search(self, query_text: str, n_results: int = 10) -> list[tuple[str, dict]]:
        """Embed query and search ChromaDB. Returns (document, metadata) pairs."""
        if self.collection.count() == 0:
            return []
        embedding = self.embedder.embed([query_text])[0]
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, self.collection.count()),
        )
        if not results["documents"] or not results["documents"][0]:
            return []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        return list(zip(docs, metas))

    def count(self) -> int:
        return self.collection.count()

    @staticmethod
    def _case_to_text(case: ReflectionCase) -> str:
        return (
            f"攻击特征: {case.attack_feature}\n"
            f"失效原因: {case.failed_reason}\n"
            f"防御指南: {case.new_guideline}\n"
            f"来源指令: {case.source_instruction}\n"
            f"来源工具: {case.source_tool}"
        )
