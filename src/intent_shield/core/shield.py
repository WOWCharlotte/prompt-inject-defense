from __future__ import annotations

import functools
from datetime import datetime, timezone
from typing import Any, Callable

from intent_shield.config import ShieldConfig
from intent_shield.core.judge import IntentJudge
from intent_shield.embed.siliconflow import SiliconFlowEmbedder, SiliconFlowReranker
from intent_shield.llm.openai_compat import OpenAICompatLLM
from intent_shield.reflection.rag import ThreatRAG
from intent_shield.reflection.reflector import Reflector
from intent_shield.reflection.store import ThreatStore
from intent_shield.types import (
    AuditRecord,
    FalseNegativeSample,
    ReflectionCase,
    RiskLevel,
    ShieldDecision,
    ToolCallBlocked,
)


class IntentShield:
    """Main entry point for the IntentShield defense framework.

    Usage:
        shield = IntentShield(ShieldConfig(...))

        # Direct judgment
        decision = shield.judge("summarize emails", "HospitalBooking", {"doctor": "dentist"})

        # Decorator mode
        @shield.guard
        def call_tool(tool_name, params, user_instruction):
            ...
    """

    def __init__(self, config: ShieldConfig):
        self.config = config
        self.audit_log: list[AuditRecord] = []

        # LLM
        self._llm = OpenAICompatLLM(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            model=config.llm_model,
        )

        # Embedder & Reranker
        self._embedder = SiliconFlowEmbedder(
            base_url=config.embed_base_url,
            api_key=config.embed_api_key,
            model=config.embed_model,
        )
        self._reranker = SiliconFlowReranker(
            base_url=config.embed_base_url,
            api_key=config.embed_api_key,
            model=config.reranker_model,
        )

        # Vector store & RAG
        self._store = ThreatStore(
            persist_dir=config.chroma_persist_dir,
            collection_name=config.chroma_collection_name,
            embedder=self._embedder,
        )
        self._rag = ThreatRAG(
            store=self._store,
            reranker=self._reranker,
            top_k=config.top_k,
        )

        # Judge
        self._judge = IntentJudge(
            llm=self._llm,
            norms=config.norms,
            alpha=config.alpha,
            beta=config.beta,
        )

        # Reflector
        self._reflector = Reflector(llm=self._llm, store=self._store)

    def judge(
        self,
        user_instruction: str,
        tool_name: str,
        tool_params: dict[str, Any],
    ) -> ShieldDecision:
        """Core defense: judge whether a tool call aligns with user intent."""
        # Step 1: RAG retrieval
        rag_cases = self._rag.retrieve(user_instruction, tool_name, tool_params)

        # Step 2-4: LLM judge + decision
        decision = self._judge.judge(
            user_instruction=user_instruction,
            tool_name=tool_name,
            tool_params=tool_params,
            rag_cases=rag_cases,
        )

        # Step 5: audit log
        record = AuditRecord(
            user_instruction=user_instruction,
            tool_name=tool_name,
            tool_params=tool_params,
            decision=decision,
            rag_cases_used=len(rag_cases),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.audit_log.append(record)

        return decision

    def guard(self, func: Callable | None = None):
        """Decorator that guards a tool-calling function.

        The decorated function must accept keyword arguments:
        - tool_name (str)
        - params or tool_params (dict)
        - user_instruction (str)
        """

        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                user_instruction = kwargs.get("user_instruction", "")
                tool_name = kwargs.get("tool_name", fn.__name__)
                tool_params = kwargs.get("params", kwargs.get("tool_params", {}))

                decision = self.judge(user_instruction, tool_name, tool_params)

                if decision.action == RiskLevel.DANGEROUS:
                    raise ToolCallBlocked(decision)

                if decision.action == RiskLevel.SUSPICIOUS:
                    if self.config.on_suspicious == "block":
                        raise ToolCallBlocked(decision)
                    # "log_only" or "ask_user" → proceed

                return fn(*args, **kwargs)

            return wrapper

        if func is not None:
            return decorator(func)
        return decorator

    def report_false_negative(
        self,
        user_instruction: str,
        tool_name: str,
        tool_params: dict[str, Any],
        initial_thought: str = "",
    ) -> ReflectionCase:
        """Stream mode: report a single false negative for immediate reflection."""
        sample = FalseNegativeSample(
            user_instruction=user_instruction,
            tool_name=tool_name,
            tool_params=tool_params,
            initial_thought=initial_thought,
        )
        return self._reflector.reflect_single(sample)

    def reflect(self, ground_truths: dict[int, str] | None = None) -> list[ReflectionCase]:
        """Batch mode: collect false negatives from audit log and reflect.

        Args:
            ground_truths: mapping of audit_log index → "malicious"/"benign".
                If None, all "safe" decisions are assumed correct (no reflection).
        """
        if ground_truths is None:
            return []

        false_negatives = []
        for idx, label in ground_truths.items():
            if idx >= len(self.audit_log):
                continue
            record = self.audit_log[idx]
            if record.decision.action == RiskLevel.SAFE and label == "malicious":
                false_negatives.append(
                    FalseNegativeSample(
                        user_instruction=record.user_instruction,
                        tool_name=record.tool_name,
                        tool_params=record.tool_params,
                        initial_thought=record.decision.thought,
                    )
                )

        return self._reflector.reflect_batch(false_negatives)

    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        self.audit_log.clear()
