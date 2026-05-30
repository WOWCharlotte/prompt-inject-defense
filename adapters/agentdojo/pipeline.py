from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from intent_shield import IntentShield, RiskLevel

# Import agentdojo components - will work at runtime if agentdojo is installed
try:
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
    from agentdojo.types import ChatMessage
except ImportError:
    # Fallback for type checking only - not used at runtime
    from typing import Protocol as BasePipelineElement
    from typing import Any as EmptyEnv, Any as Env, Any as FunctionsRuntime
    from typing import Any as ChatMessage


class IntentShieldElement(BasePipelineElement):
    """Intercepts tool calls before execution and validates intent alignment.

    Insert this element into ToolsExecutionLoop before ToolsExecutor to enable
    real-time intent alignment checking against prompt injection attacks.

    Args:
        shield: The IntentShield instance used for judgment.
        on_suspicious: How to handle suspicious tool calls. "block" (default) or "log".
    """

    def __init__(
        self,
        shield: IntentShield,
        on_suspicious: Literal["block", "log"] = "block",
    ):
        self.shield = shield
        self.on_suspicious = on_suspicious
        self._blocked_log: list[dict[str, Any]] = []
        self._judge_log: list[dict[str, Any]] = []  # All judge calls for tracing

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict[str, Any] = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict[str, Any]]:
        # Only process assistant messages with tool_calls
        if not messages:
            return query, runtime, env, messages, extra_args

        last_message = messages[-1]
        if last_message["role"] != "assistant":
            return query, runtime, env, messages, extra_args

        if not last_message["tool_calls"]:
            return query, runtime, env, messages, extra_args

        # Extract user instruction from first user message
        user_instruction = self._extract_user_instruction(messages)
        if not user_instruction:
            # Cannot determine user intent, allow all tool calls
            return query, runtime, env, messages, extra_args

        # Evaluate each tool call
        new_tool_calls = []
        for tool_call in last_message["tool_calls"]:
            decision = self._judge_tool_call(
                user_instruction=user_instruction,
                tool_call=tool_call,
            )

            # Log all judge calls for tracing
            self._judge_log.append({
                "tool": tool_call.function,
                "args": tool_call.args,
                "risk_score": decision.risk_score,
                "thought": decision.thought,
                "action": decision.action.value,
                "user_instruction": user_instruction[:200] + "..." if len(user_instruction) > 200 else user_instruction,
            })

            if decision.action == RiskLevel.DANGEROUS:
                self._blocked_log.append({
                    "tool": tool_call.function,
                    "args": tool_call.args,
                    "risk_score": decision.risk_score,
                    "thought": decision.thought,
                })
                continue  # Remove: do not execute

            if decision.action == RiskLevel.SUSPICIOUS:
                if self.on_suspicious == "block":
                    self._blocked_log.append({
                        "tool": tool_call.function,
                        "args": tool_call.args,
                        "risk_score": decision.risk_score,
                        "thought": decision.thought,
                    })
                    continue  # Remove: suspicious + block mode

            # Allow: safe or suspicious + log mode
            new_tool_calls.append(tool_call)

        # Update messages in-place (preserving immutability pattern)
        messages[-1]["tool_calls"] = new_tool_calls if new_tool_calls else None

        return query, runtime, env, messages, extra_args

    def _judge_tool_call(
        self,
        user_instruction: str,
        tool_call: Any,
    ) -> Any:
        """Judge a single tool call. Returns ShieldDecision.

        On LLM errors, returns a conservative decision (dangerous) to implement
        Default Deny security policy.
        """
        try:
            return self.shield.judge(
                user_instruction=user_instruction,
                tool_name=tool_call.function,
                tool_params=tool_call.args,
            )
        except Exception:
            # Conservative: block on error (Default Deny)
            from intent_shield.types import ShieldDecision
            return ShieldDecision(
                action=RiskLevel.DANGEROUS,
                risk_score=1.0,
                thought=f"Judge error: {type(Exception).__name__}",
                tool_name=tool_call.function,
                tool_params=tool_call.args,
                user_instruction=user_instruction,
            )

    def _extract_user_instruction(self, messages: Sequence[ChatMessage]) -> str:
        """Extract user instruction from the first user message in the conversation.

        Returns empty string if no user message is found.
        """
        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, list):
                    text_blocks = [
                        c.get("text", "") for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    ]
                    return "\n".join(text_blocks)
                if isinstance(content, str):
                    return content
        return ""

    def get_blocked_log(self) -> list[dict[str, Any]]:
        """Return log of blocked tool calls for post-analysis."""
        return self._blocked_log

    def get_judge_log(self) -> list[dict[str, Any]]:
        """Return log of all judge LLM calls for tracing."""
        return self._judge_log

    def reset_blocked_log(self) -> None:
        """Clear the blocked log (e.g., between benchmark runs)."""
        self._blocked_log = []