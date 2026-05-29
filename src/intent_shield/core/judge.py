from __future__ import annotations

from intent_shield.core.decision import make_decision, parse_judge_response
from intent_shield.core.prompt import build_judge_prompt
from intent_shield.llm.base import BaseLLM
from intent_shield.types import ReflectionCase, ShieldDecision


class IntentJudge:
    """LLM-based intent alignment judge."""

    def __init__(self, llm: BaseLLM, norms: list[str], alpha: float, beta: float):
        self.llm = llm
        self.norms = norms
        self.alpha = alpha
        self.beta = beta

    def judge(
        self,
        user_instruction: str,
        tool_name: str,
        tool_params: dict,
        rag_cases: list[ReflectionCase] | None = None,
    ) -> ShieldDecision:
        """Evaluate intent alignment between user instruction and tool call."""
        rag_cases = rag_cases or []

        messages = build_judge_prompt(
            user_instruction=user_instruction,
            tool_name=tool_name,
            tool_params=tool_params,
            norms=self.norms,
            rag_cases=rag_cases,
        )

        raw_response = self.llm.chat_sync(messages, temperature=0.0)
        judge_result = parse_judge_response(raw_response)

        return make_decision(
            judge_result=judge_result,
            alpha=self.alpha,
            beta=self.beta,
            user_instruction=user_instruction,
            tool_name=tool_name,
            tool_params=tool_params,
            rag_cases=rag_cases,
        )
