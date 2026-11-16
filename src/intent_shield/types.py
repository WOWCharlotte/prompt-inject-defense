from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(str, Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    DANGEROUS = "dangerous"


class JudgeResult(BaseModel):
    """Raw output from the LLM judge."""

    thought: str
    risk_score: float = Field(ge=0.0, le=1.0)


class RiskAnalysis(BaseModel):
    """Risk signals required in the judge's structured response."""

    model_config = ConfigDict(extra="forbid")

    parameter_injection: bool
    workflow_hijacking: bool
    data_exfiltration: bool
    privilege_escalation: bool


class JudgeResponse(BaseModel):
    """Complete model output matching the judge prompt's JSON example."""

    model_config = ConfigDict(extra="forbid")

    user_goal: str
    tool_behavior: str
    thought: str
    intent_alignment: Literal["direct", "indirect", "drift"]
    risk_analysis: RiskAnalysis
    risk_score: float = Field(ge=0.0, le=1.0)


class ReflectionCase(BaseModel):
    """A stored historical threat case or reflection report."""

    attack_feature: str
    failed_reason: str
    new_guideline: str
    source_instruction: str = ""
    source_tool: str = ""
    source_params: str = ""


class RetrievedCase(BaseModel):
    """Stored document with dense retrieval and reranking scores."""

    document_id: str
    text: str
    source_tool: str
    similarity_score: float
    rerank_score: float | None = None


class ShieldDecision(BaseModel):
    """Final decision returned to the caller."""

    action: RiskLevel
    risk_score: float
    thought: str
    tool_name: str
    tool_params: dict[str, Any]
    user_instruction: str
    rag_cases: list[RetrievedCase] = Field(default_factory=list)


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
    actual_attack_behavior: str = ""
    ground_truth: str = "malicious"


class ToolCallBlocked(Exception):
    """Raised when a tool call is blocked by IntentShield."""

    def __init__(self, decision: ShieldDecision):
        self.decision = decision
        super().__init__(
            f"Tool call '{decision.tool_name}' blocked: "
            f"risk_score={decision.risk_score:.2f}, action={decision.action.value}"
        )
