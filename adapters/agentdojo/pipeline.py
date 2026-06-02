"""Post-execution intent alignment detection for AgentDojo pipelines."""
from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from intent_shield.types import RiskLevel, ShieldDecision

# Import agentdojo components - will work at runtime if agentdojo is installed
try:
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.agent_pipeline.errors import AbortAgentError  # noqa: F401
    from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
    from agentdojo.types import ChatMessage, ChatToolResultMessage, text_content_block_from_string
except ImportError:
    # Fallback for type checking only - not used at runtime
    from typing import Protocol as BasePipelineElement
    from typing import Any as EmptyEnv, Any as Env, Any as FunctionsRuntime
    from typing import Any as ChatMessage, Any as ChatToolResultMessage
    from typing import Any as text_content_block_from_string
    from typing import Any as AbortAgentError


class DetectionLog(BaseModel):
    """One post-execution detection record.

    Attributes:
        timestamp: ISO 8601 UTC timestamp of the decision.
        tool_name: Name of the tool whose output was judged, or "unknown".
        tool_args: Args of the assistant tool_call that produced this output
            (best-effort, from the most recent assistant message).
        user_instruction_preview: First user message, truncated to 200 chars.
        output_preview: Tool output content, truncated to 200 chars.
        risk_score: Final risk score in [0, 1].
        thought: Free-form reasoning from the judge.
        action: Final decision (`safe` / `suspicious` / `dangerous`).
        masked: True if the tool message content was replaced entirely.
        error: Exception class name if `shield.judge()` raised, else None.
    """

    timestamp: str
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    user_instruction_preview: str
    output_preview: str
    risk_score: float = Field(ge=0.0, le=1.0)
    thought: str
    action: RiskLevel
    masked: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_decision(tool_name: str, user_instruction: str) -> ShieldDecision:
    return ShieldDecision(
        action=RiskLevel.SAFE,
        risk_score=0.0,
        thought="Permissive fallback",
        tool_name=tool_name,
        tool_params={},
        user_instruction=user_instruction,
    )
