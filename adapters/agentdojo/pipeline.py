"""Post-execution intent alignment detection for AgentDojo pipelines."""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
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
    run_id: str | None = None
    suite_name: str | None = None
    task_id: str | None = None
    user_task_id: str | None = None
    injection_task_id: str | None = None
    model: str | None = None
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    user_instruction_preview: str
    output_preview: str
    risk_score: float = Field(ge=0.0, le=1.0)
    thought: str
    action: RiskLevel
    outcome: str = "none"  # "masked" | "sanitized" | "none"
    error: str | None = None
    judge_latency_ms: float | None = None


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


def _get_call_attr(tool_call: Any, attr: str) -> Any:
    """Safely read an attribute from a tool_call object or dict."""
    if tool_call is None:
        return None
    val = getattr(tool_call, attr, None)
    if val is None and isinstance(tool_call, dict):
        val = tool_call.get(attr)
    return val


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

    def __init__(
        self,
        shield: Any,
        mask_output: bool = True,
        run_id: str | None = None,
        suite_name: str | None = None,
        model: str | None = None,
        on_log: Callable[[DetectionLog], None] | None = None,
    ) -> None:
        self.shield = shield
        self.mask_output = mask_output
        self.run_id = run_id
        self.suite_name = suite_name
        self.model = model
        self.on_log = on_log
        self._detection_log: list[DetectionLog] = []
        # Per-task context (set/cleared by the benchmark harness via set_task_context)
        self._task_context: dict[str, str | None] = {
            "task_id": None,
            "user_task_id": None,
            "injection_task_id": None,
        }

    def set_task_context(
        self,
        task_id: str | None = None,
        user_task_id: str | None = None,
        injection_task_id: str | None = None,
    ) -> None:
        """Update task context for subsequent detection logs."""
        if task_id is not None:
            self._task_context["task_id"] = task_id
        if user_task_id is not None:
            self._task_context["user_task_id"] = user_task_id
        if injection_task_id is not None:
            self._task_context["injection_task_id"] = injection_task_id

    def reset_task_context(self) -> None:
        self._task_context = {
            "task_id": None,
            "user_task_id": None,
            "injection_task_id": None,
        }

    # ---- public API ----

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict[str, Any] = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict[str, Any]]:
        """Inspect the latest tool message and judge against the user instruction.

        `query` is the original user task string propagated through the pipeline —
        it is used directly as the user instruction (no message scanning needed).

        Tool name and args are read from the tool message's `tool_call` field,
        which agentdojo populates from the preceding assistant tool_call response.
        """
        if not messages:
            return query, runtime, env, messages, extra_args

        last = messages[-1]
        if last.get("role") != "tool":
            return query, runtime, env, messages, extra_args

        # query IS the user instruction — no need to scan messages
        user_instruction = query
        if not user_instruction:
            return query, runtime, env, messages, extra_args

        # tool_call carries function name + args directly on the tool message
        tool_call = last.get("tool_call")
        tool_name = _get_call_attr(tool_call, "function") or "unknown"
        raw_args = _get_call_attr(tool_call, "args")
        tool_args: dict[str, Any] = dict(raw_args) if raw_args else {}
        tool_content = self._get_text_content(last.get("content", ""))
        # Re-pack the message content as a single text block so downstream
        # agentdojo code (which iterates `for el in message["content"]`) keeps
        # working. Keep local `tool_content` as a plain string for the judge
        # and DetectionLog.
        if isinstance(last.get("content"), list):
            last["content"] = [text_content_block_from_string(tool_content)]

        decision, judge_error, judge_latency_ms = self._judge_output_alignment(
            user_instruction, tool_name, tool_args, tool_content
        )

        outcome = "none"
        if decision.action == RiskLevel.DANGEROUS:
            self._apply_mask(last, decision.risk_score, "dangerous")
            outcome = "masked"
        elif decision.action == RiskLevel.SUSPICIOUS and self.mask_output:
            self._apply_mask(last, decision.risk_score, "suspicious")
            outcome = "masked"
        elif decision.action == RiskLevel.SUSPICIOUS and not self.mask_output:
            self._apply_sanitize(last)
            outcome = "sanitized"

        log = DetectionLog(
            timestamp=_now_iso(),
            run_id=self.run_id,
            suite_name=self.suite_name,
            task_id=self._task_context.get("task_id"),
            user_task_id=self._task_context.get("user_task_id"),
            injection_task_id=self._task_context.get("injection_task_id"),
            model=self.model,
            tool_name=tool_name,
            tool_args=tool_args,
            user_instruction_preview=user_instruction[:200],
            output_preview=tool_content[:200],
            risk_score=decision.risk_score,
            thought=decision.thought,
            action=decision.action,
            outcome=outcome,
            error=judge_error,
            judge_latency_ms=judge_latency_ms,
        )
        self._detection_log.append(log)
        if self.on_log is not None:
            try:
                self.on_log(log)
            except Exception as cb_err:  # don't let logging break the pipeline
                print(f"[IntentShield] on_log callback error: {cb_err}", file=__import__("sys").stderr)
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
    ) -> tuple[ShieldDecision, str | None, float | None]:
        """Call shield.judge(); return (decision, error_str|None, latency_ms)."""
        import time

        if not tool_content:
            return _safe_decision(tool_name, user_instruction), None, 0.0
        start = time.perf_counter()
        try:
            decision = self.shield.judge(
                user_instruction=user_instruction,
                tool_name=tool_name,
                tool_params=tool_args,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            return decision, None, latency_ms
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000.0
            error_str = f"{type(e).__name__}: {e}"
            decision = ShieldDecision(
                action=RiskLevel.DANGEROUS,
                risk_score=1.0,
                thought=f"Judge error: {error_str}",
                tool_name=tool_name,
                tool_params=tool_args,
                user_instruction=user_instruction,
            )
            return decision, error_str, latency_ms

    def _apply_mask(
        self, tool_message: ChatMessage, risk_score: float, reason: str
    ) -> None:
        tool_message["content"] = [
            text_content_block_from_string(
                f"[IntentShield: Output masked due to {reason} content (risk={risk_score:.2f})]"
            )
        ]
        tool_message["error"] = f"IntentShield: {reason} content detected"

    def _apply_sanitize(self, tool_message: ChatMessage) -> None:
        original = self._get_text_content(tool_message.get("content", ""))
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
