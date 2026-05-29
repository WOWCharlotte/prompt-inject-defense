from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    DANGEROUS = "dangerous"


class JudgeResult(BaseModel):
    """Raw output from the LLM judge."""

    thought: str
    risk_score: float = Field(ge=0.0, le=1.0)
    decision: RiskLevel


class ReflectionCase(BaseModel):
    """A stored historical threat case or reflection report."""

    attack_feature: str
    failed_reason: str
    new_guideline: str
    source_instruction: str = ""
    source_tool: str = ""
    source_params: str = ""


class ShieldDecision(BaseModel):
    """Final decision returned to the caller."""

    action: RiskLevel
    risk_score: float
    thought: str
    tool_name: str
    tool_params: dict[str, Any]
    user_instruction: str
    rag_cases: list[ReflectionCase] = []


class AuditRecord(BaseModel):
    """Audit log entry for each judgment."""

    user_instruction: str
    tool_name: str
    tool_params: dict[str, Any]
    decision: ShieldDecision
    rag_cases_used: int = 0
    timestamp: str = ""


class FalseNegativeSample(BaseModel):
    """A sample where the shield incorrectly judged as safe."""

    user_instruction: str
    tool_name: str
    tool_params: dict[str, Any]
    initial_thought: str = ""
    ground_truth: str = "malicious"


class ToolCallBlocked(Exception):
    """Raised when a tool call is blocked by IntentShield."""

    def __init__(self, decision: ShieldDecision):
        self.decision = decision
        super().__init__(
            f"Tool call '{decision.tool_name}' blocked: "
            f"risk_score={decision.risk_score:.2f}, action={decision.action.value}"
        )
