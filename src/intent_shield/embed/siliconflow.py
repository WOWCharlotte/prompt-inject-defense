from __future__ import annotations

import httpx

from intent_shield.embed.base import BaseEmbedder, BaseReranker


class SiliconFlowEmbedder(BaseEmbedder):
    """Text embedder using SiliconFlow API (BAAI/bge-m3)."""

    def __init__(self, base_url: str, api_key: str, model: str = "BAAI/bge-m3"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts, "encoding_format": "float"},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()["data"]
        data.sort(key=lambda x: x["index"])
        return [item["embedding"] for item in data]


class SiliconFlowReranker(BaseReranker):
    """Document reranker using SiliconFlow API (BAAI/bge-reranker-v2-m3)."""

    def __init__(self, base_url: str, api_key: str, model: str = "BAAI/bge-reranker-v2-m3"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def rerank(self, query: str, documents: list[str], top_n: int = 3) -> list[tuple[int, float]]:
        response = httpx.post(
            f"{self.base_url}/rerank",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": False,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        results = response.json()["results"]
        return [(r["index"], r["relevance_score"]) for r in results]
