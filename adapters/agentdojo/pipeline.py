"""Post-execution intent alignment detection for AgentDojo pipelines."""
from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from intent_shield.types import RiskLevel, ShieldDecision

# Import agentdojo - will work at runtime if installed
try:
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.agent_pipeline.errors import AbortAgentError  # noqa: F401
    from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
    from agentdojo.types import ChatMessage, ChatToolResultMessage, text_content_block_from_string
except ImportError:
    from typing import Protocol as BasePipelineElement, Any
    Env = FunctionsRuntime = ChatMessage = ChatToolResultMessage = AbortAgentError = Any

    class EmptyEnv:  # type: ignore[no-redef]
        pass

    def text_content_block_from_string(text: str) -> dict:  # type: ignore[misc]
        return {"type": "text", "content": text}


class DetectionLog(BaseModel):
    """One post-execution detection record."""

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


# ---------------------------------------------------------------------------
# IntentShieldElement
# ---------------------------------------------------------------------------


class IntentShieldElement(BasePipelineElement):
    """Post-execution intent alignment check on tool outputs.

    Sits in `ToolsExecutionLoop` after `ToolsExecutor`. Judges whether
    following the tool output would deviate from the user's intent, then
    masks (or sanitizes) the message based on the decision.

    Args:
        shield: IntentShield instance for judgment.
        mask_output: If True, suspicious outputs are fully replaced with a
            marker; if False, only injection patterns are removed in place.
            Dangerous outputs are always fully masked.
    """

    _SANITIZE_PATTERNS = [
        r"ignore\s+previous\s+instructions",
        r"disregard\s+all\s+prior\s+commands",
        r"forget\s+everything\s+you\s+were\s+told",
        r"do\s+not\s+follow\s+original\s+instructions",
    ]

    def __init__(self, shield: Any, mask_output: bool = True) -> None:
        self.shield = shield
        self.mask_output = mask_output
        self._detection_log: list[DetectionLog] = []

    # ---- public API ----

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict[str, Any] = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict[str, Any]]:
        if not messages:
            return query, runtime, env, messages, extra_args

        last = messages[-1]
        if last.get("role") != "tool":
            return query, runtime, env, messages, extra_args

        user_instruction = self._extract_user_instruction(messages)
        if not user_instruction:
            return query, runtime, env, messages, extra_args

        tool_call = last.get("tool_call")
        tool_name = tool_call.function if tool_call else "unknown"
        tool_content = self._get_text_content(last.get("content", []))
        tool_call_id = last.get("tool_call_id")
        tool_args = self._extract_tool_args(messages, tool_call_id)

        decision, judge_error = self._judge_output_alignment(
            user_instruction, tool_name, tool_args, tool_content
        )

        masked = False
        if decision.action == RiskLevel.DANGEROUS:
            masked = self._apply_mask(last, decision.risk_score, "dangerous")
        elif decision.action == RiskLevel.SUSPICIOUS and self.mask_output:
            masked = self._apply_mask(last, decision.risk_score, "suspicious")
        elif decision.action == RiskLevel.SUSPICIOUS and not self.mask_output:
            self._apply_sanitize(last)

        self._detection_log.append(
            DetectionLog(
                timestamp=_now_iso(),
                tool_name=tool_name,
                tool_args=tool_args,
                user_instruction_preview=user_instruction[:200],
                output_preview=tool_content[:200],
                risk_score=decision.risk_score,
                thought=decision.thought,
                action=decision.action,
                masked=masked,
                error=judge_error,
            )
        )
        return query, runtime, env, messages, extra_args

    def get_detection_log(self) -> list[DetectionLog]:
        return list(self._detection_log)

    def reset_detection_log(self) -> None:
        self._detection_log = []

    # ---- internals ----

    def _judge_output_alignment(
        self,
        user_instruction: str,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_content: str,
    ) -> tuple[ShieldDecision, str | None]:
        """Call shield.judge(); return (decision, error_str|None)."""
        if not tool_content:
            return _safe_decision(tool_name, user_instruction), None
        try:
            decision = self.shield.judge(
                user_instruction=user_instruction,
                tool_name="follow_external_instruction",
                tool_params={
                    "external_content": tool_content[:1000],
                    "original_tool": tool_name,
                    "original_tool_args": tool_args,
                },
            )
            return decision, None
        except Exception as e:
            error_str = f"{type(e).__name__}: {e}"
            decision = ShieldDecision(
                action=RiskLevel.DANGEROUS,
                risk_score=1.0,
                thought=f"Judge error: {error_str}",
                tool_name=tool_name,
                tool_params={},
                user_instruction=user_instruction,
            )
            return decision, error_str

    def _apply_mask(
        self, tool_message: ChatMessage, risk_score: float, reason: str
    ) -> bool:
        tool_message["content"] = [
            text_content_block_from_string(
                f"[IntentShield: Output masked due to {reason} content (risk={risk_score:.2f})]"
            )
        ]
        tool_message["error"] = f"IntentShield: {reason} content detected"
        return True

    def _apply_sanitize(self, tool_message: ChatMessage) -> None:
        original = self._get_text_content(tool_message.get("content", []))
        result = original
        for pattern in self._SANITIZE_PATTERNS:
            result = re.sub(pattern, "[filtered]", result, flags=re.IGNORECASE)
        if result != original:
            tool_message["content"] = [text_content_block_from_string(result)]

    def _get_text_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("content", ""))
            return "\n".join(parts)
        return str(content)

    def _extract_user_instruction(self, messages: Sequence[ChatMessage]) -> str:
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):
                    blocks = [
                        c.get("content", "")
                        for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    ]
                    return "\n".join(blocks)
                if isinstance(content, str):
                    return content
        return ""

    def _extract_tool_args(
        self, messages: Sequence[ChatMessage], tool_call_id: str | None
    ) -> dict[str, Any]:
        """Bottom-up search for the most recent assistant message and match args.

        - Stops at the first assistant message (going from bottom up). If that
          assistant has a `tool_call` with matching `id`, returns its `args`.
        - Stops at the first non-assistant, non-tool message (treat as boundary).
        - Returns `{}` on any miss.
        """
        if not tool_call_id:
            return {}
        for msg in reversed(messages):
            role = msg.get("role")
            if role == "assistant":
                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    tc_id = getattr(tc, "id", None) or (
                        tc.get("id") if isinstance(tc, dict) else None
                    )
                    if tc_id == tool_call_id:
                        args = getattr(tc, "args", None) or (
                            tc.get("args") if isinstance(tc, dict) else None
                        )
                        return dict(args) if args else {}
                return {}
            if role != "tool":
                return {}
        return {}
