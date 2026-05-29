from __future__ import annotations

import os

from pydantic import BaseModel, Field


class ShieldConfig(BaseModel):
    """Configuration for IntentShield defense framework."""

    # LLM Judge
    llm_base_url: str = Field(
        default_factory=lambda: os.getenv("INTENT_SHIELD_LLM_BASE_URL", "https://api.openai.com/v1")
    )
    llm_api_key: str = Field(default_factory=lambda: os.getenv("INTENT_SHIELD_LLM_KEY", ""))
    llm_model: str = Field(
        default_factory=lambda: os.getenv("INTENT_SHIELD_LLM_MODEL", "gpt-4o-mini")
    )
    temperature: float = 0.0

    # Embedding & Reranking (SiliconFlow)
    embed_base_url: str = Field(
        default_factory=lambda: os.getenv("INTENT_SHIELD_EMBED_BASE_URL", "https://api.siliconflow.cn/v1")
    )
    embed_api_key: str = Field(default_factory=lambda: os.getenv("INTENT_SHIELD_EMBED_KEY", ""))
    embed_model: str = Field(
        default_factory=lambda: os.getenv("INTENT_SHIELD_EMBED_MODEL", "BAAI/bge-m3")
    )
    reranker_model: str = Field(
        default_factory=lambda: os.getenv("INTENT_SHIELD_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    )

    # Risk thresholds
    alpha: float = 0.3  # < alpha -> Safe
    beta: float = 0.7  # > beta -> Dangerous

    # RAG
    top_k: int = 3
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "threat_cases"

    # Safety norms (OWASP-inspired)
    norms: list[str] = [
        "财产安全: 禁止未授权的资金转移、支付、购买操作",
        "数据隐私: 禁止将用户个人数据发送至非授权第三方",
        "法规合规: 禁止违反数据保护法规(如GDPR)的操作",
        "身心健康: 禁止可能危害用户身心健康的操作",
        "禁止冒犯: 禁止生成或传播冒犯性、歧视性内容",
    ]

    # Behavior on suspicious decision
    on_suspicious: str = "block"  # "block" | "ask_user" | "log_only"
