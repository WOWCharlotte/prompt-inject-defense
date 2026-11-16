from __future__ import annotations

import json
import re
from typing import Any

from intent_shield.types import JudgeResult, RetrievedCase, RiskLevel, ShieldDecision


def parse_judge_response(raw_response: str) -> JudgeResult:
    """Parse the LLM judge's JSON response with multiple fallback strategies."""
    data = _extract_json(raw_response)

    if data is None:
        return JudgeResult(
            thought="Failed to parse LLM response: " + raw_response[:200],
            risk_score=0.8,
        )

    risk_score = float(data.get("risk_score", 0.8))
    risk_score = max(0.0, min(1.0, risk_score))

    return JudgeResult(
        thought=data.get("thought", ""),
        risk_score=risk_score,
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    """Try multiple strategies to extract JSON from LLM output."""
    # Strategy 1: direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "user_goal" in data:
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: extract from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and "user_goal" in data:
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 3: find first { ... } block (non-greedy, no nested braces)
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and "user_goal" in data:
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 4: greedy match for nested braces (captures full structure)
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            # Only take the first balanced JSON object (everything up to the first unmatched brace)
            data = json.loads(match.group(1))
            if isinstance(data, dict) and "user_goal" in data:
                return data
        except json.JSONDecodeError:
            pass

    return None


def make_decision(
    judge_result: JudgeResult,
    alpha: float,
    beta: float,
    user_instruction: str,
    tool_name: str,
    tool_params: dict[str, Any],
    rag_cases: list[RetrievedCase],
) -> ShieldDecision:
    """Apply threshold-based segmented decision logic."""
    score = judge_result.risk_score

    if score < alpha:
        action = RiskLevel.SAFE
    elif score > beta:
        action = RiskLevel.DANGEROUS
    else:
        action = RiskLevel.SUSPICIOUS

    return ShieldDecision(
        action=action,
        risk_score=score,
        thought=judge_result.thought,
        tool_name=tool_name,
        tool_params=tool_params,
        user_instruction=user_instruction,
        rag_cases=rag_cases,
    )
