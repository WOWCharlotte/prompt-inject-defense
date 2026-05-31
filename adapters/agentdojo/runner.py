from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentdojo.agent_pipeline import AgentPipeline, BasePipelineElement
from agentdojo.agent_pipeline.agent_pipeline import (
    AgentPipeline,
    InitQuery,
    SystemMessage,
    ToolsExecutionLoop,
)
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM, OpenAILLMToolFilter
from agentdojo.agent_pipeline.tool_execution import ToolsExecutor, tool_result_to_str
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import (
    benchmark_suite_with_injections,
    benchmark_suite_without_injections,
)
from agentdojo.logging import OutputLogger
from agentdojo.models import MODEL_PROVIDERS, ModelsEnum
from agentdojo.task_suite.load_suites import get_suite
from rich.live import Live

from intent_shield import IntentShield

from .pipeline import IntentShieldElement


# Tool filter prompt from agentdojo
TOOL_FILTER_PROMPT = (
    "Your task is to filter the list of tools to only include those that are relevant to the user's task."
    "Output the comma-separated list containing the names of the necessary tools."
    "Do not write tool names that do not exist."
)


def get_llm_provider(model: str | ModelsEnum) -> tuple[str, str]:
    """Get provider and model name from model identifier.

    Handles both short names (e.g., "gpt-4o-mini") and full names
    (e.g., "gpt-4o-mini-2024-07-18") by matching against MODEL_PROVIDERS.

    Also handles OpenAI-compatible models (qwen, deepseek, etc.) by detecting
    patterns and returning "openai-compatible" as provider with base_url info.
    """
    if isinstance(model, ModelsEnum):
        model_str = model.value
    else:
        model_str = model

    # Try to match as ModelsEnum value (exact match first)
    for enum_val in ModelsEnum:
        if enum_val.value == model_str:
            return MODEL_PROVIDERS[enum_val], enum_val.value

    # Try substring match for short names
    for enum_val in ModelsEnum:
        if model_str in enum_val.value or enum_val.value.replace("-", "").startswith(model_str.replace("-", "")):
            return MODEL_PROVIDERS[enum_val], enum_val.value

    # Detect OpenAI-compatible models
    model_lower = model_str.lower()
    if "qwen" in model_lower:
        return "openai-compatible", model_str
    if "deepseek" in model_lower:
        return "openai-compatible", model_str
    if "glm" in model_lower or "zhipu" in model_lower:
        return "openai-compatible", model_str
    if "mistral" in model_lower:
        return "openai-compatible", model_str
    if "llama" in model_lower and "anthropic" not in model_lower:
        return "openai-compatible", model_str

    # Fallback: assume openai for model names containing common patterns
    if "gpt" in model_lower:
        return "openai", model_str
    if "claude" in model_lower:
        return "anthropic", model_str

    return "openai", model_str


def build_intent_shield_pipeline(
    shield: IntentShield,
    model: str | ModelsEnum,
    on_suspicious: str = "block",
) -> AgentPipeline:
    """Build AgentPipeline with IntentShieldElement inserted.

    Pipeline structure:
        [SystemMessage] → [InitQuery] → [LLM] → [ToolsExecutionLoop]
                                                       ↑
                                            [IntentShieldElement, ToolsExecutor, LLM]
    """
    provider, model_str = get_llm_provider(model)

    # Create LLM element based on provider
    import anthropic
    import cohere
    import google.genai
    import openai

    if provider == "openai":
        client = openai.OpenAI()
        llm = OpenAILLM(client, model_str)
    elif provider == "anthropic":
        client = anthropic.Anthropic()
        llm = AnthropicLLM(client, model_str)
    elif provider == "cohere":
        client = cohere.Client()
        from agentdojo.agent_pipeline.llms.cohere_llm import CohereLLM
        llm = CohereLLM(client, model_str)
    elif provider == "google":
        client = google.genai.Client(
            vertexai=True,
            project=None,  # Will use env var
            location=None,
        )
        from agentdojo.agent_pipeline.llms.google_llm import GoogleLLM
        llm = GoogleLLM(model_str, client)
    elif provider == "openai-compatible":
        # Support for qwen, deepseek, glm, mistral, etc. via OpenAI-compatible API
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        client = openai.OpenAI(base_url=base_url, api_key=api_key)
        llm = OpenAILLM(client, model_str)
    else:
        # Default to openai
        client = openai.OpenAI()
        llm = OpenAILLM(client, model_str)

    # Load default system message
    import importlib.resources
    import yaml
    package_files = importlib.resources.files("agentdojo")
    path = package_files / "data" / "system_messages.yaml"
    with importlib.resources.as_file(path) as p, p.open() as f:
        system_messages = yaml.safe_load(f)
    system_message = system_messages.get("default", "")

    # Create IntentShield element
    intent_element = IntentShieldElement(
        shield=shield,
        on_suspicious=on_suspicious,
    )

    # Build tools execution loop with IntentShield
    tools_loop = ToolsExecutionLoop([
        intent_element,
        ToolsExecutor(tool_output_formatter=tool_result_to_str),
        llm,
    ])

    # Build full pipeline
    pipeline = AgentPipeline([
        SystemMessage(system_message),
        InitQuery(),
        llm,
        tools_loop,
    ])
    pipeline.name = model_str  # Must be a valid model name for attack registry

    return pipeline


class AnthropicLLM:
    """Wrapper for Anthropic LLM to match agentdojo's interface."""

    def __init__(self, client: anthropic.Anthropic, model: str):
        self.client = client
        self.model = model

    def query(self, query, runtime, env, messages, extra_args):
        # Use agentdojo's AnthropicLLM directly
        from agentdojo.agent_pipeline.llms.anthropic_llm import AnthropicLLM as BaseAnthropicLLM
        return BaseAnthropicLLM(self.client, self.model).query(query, runtime, env, messages, extra_args)


class AgentDojoRunner:
    """Runs AgentDojo benchmark with IntentShield integrated.

    Collects utility and security metrics, plus blocked tool call logs
    for post-analysis and reflection.

    Args:
        shield: The IntentShield instance for intent alignment checking.
        log_dir: Directory to save trace logs and metrics.
        on_suspicious: How to handle suspicious tool calls ("block" or "log").
    """

    def __init__(
        self,
        shield: IntentShield,
        log_dir: str | Path = "./results",
        on_suspicious: str = "block",
    ):
        self.shield = shield
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.on_suspicious = on_suspicious

    def run(
        self,
        suite_name: str,
        model: str | ModelsEnum,
        attack: str | None = None,
        user_tasks: tuple[str, ...] = (),
        injection_tasks: tuple[str, ...] = (),
        benchmark_version: str = "v1.2.2",
        force_rerun: bool = False,
        max_user_tasks: int | None = None,
        max_injection_tasks: int | None = None,
    ) -> dict[str, Any]:
        """Run AgentDojo benchmark with IntentShield.

        Args:
            suite_name: Name of the task suite (e.g., "workspace", "banking").
            model: Model identifier or ModelsEnum value.
            attack: Attack type name (e.g., "important_instructions", "tool_knowledge").
            user_tasks: Tuple of user task IDs to run.
            injection_tasks: Tuple of injection task IDs to run.
            benchmark_version: Benchmark version string.
            force_rerun: Whether to re-run tasks with existing results.
            max_user_tasks: Limit number of user tasks (random sample if needed).
            max_injection_tasks: Limit number of injection tasks (random sample if needed).

        Returns:
            Dict containing utility_avg, security_avg, blocked_tool_calls, run_id.
        """
        import random

        # Build pipeline with IntentShield
        pipeline = build_intent_shield_pipeline(
            shield=self.shield,
            model=model if isinstance(model, str) else model.value,
            on_suspicious=self.on_suspicious,
        )

        # Load suite
        suite = get_suite(benchmark_version, suite_name)

        # Apply max limits if specified (random sampling)
        if max_user_tasks and not user_tasks:
            all_user_task_ids = list(suite.user_tasks.keys())
            if len(all_user_task_ids) > max_user_tasks:
                user_tasks = tuple(random.sample(all_user_task_ids, max_user_tasks))
                print(f"  Limited to {max_user_tasks} user tasks (random sample)")

        if max_injection_tasks and not injection_tasks and attack:
            all_injection_ids = list(suite.injection_tasks.keys())
            if len(all_injection_ids) > max_injection_tasks:
                injection_tasks = tuple(random.sample(all_injection_ids, max_injection_tasks))
                print(f"  Limited to {max_injection_tasks} injection tasks (random sample)")

        # Create log directory for traces
        trace_dir = self.log_dir / "traces" / pipeline.name / suite_name
        trace_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [DEBUG] Trace directory: {trace_dir}", flush=True)

        # Wrap benchmark with OutputLogger for trace logging
        logger = OutputLogger(str(trace_dir))
        with logger:
            print("  [DEBUG] Starting benchmark...", flush=True)
            # Run benchmark
            if attack:
                attacker = load_attack(attack, suite, pipeline)
                print(f"  [DEBUG] Attack loaded: {attack}", flush=True)
                print(f"  [DEBUG] Running benchmark_suite_with_injections...", flush=True)
                results = benchmark_suite_with_injections(
                    pipeline,
                    suite,
                    attacker,
                    user_tasks=user_tasks if user_tasks else None,
                    injection_tasks=injection_tasks if injection_tasks else None,
                    logdir=trace_dir,
                    force_rerun=force_rerun,
                    benchmark_version=benchmark_version,
                )
                print(f"  [DEBUG] benchmark_suite_with_injections completed", flush=True)
            else:
                print(f"  [DEBUG] Running benchmark_suite_without_injections...", flush=True)
                results = benchmark_suite_without_injections(
                    pipeline,
                    suite,
                    user_tasks=user_tasks if user_tasks else None,
                    logdir=trace_dir,
                    force_rerun=force_rerun,
                    benchmark_version=benchmark_version,
                )
                print(f"  [DEBUG] benchmark_suite_without_injections completed", flush=True)

        # Collect blocked counts and judge logs from IntentShieldElement
        blocked_count, judge_logs = self._collect_shield_data(pipeline)

        # Compute metrics
        utility_avg = self._compute_avg(results["utility_results"])
        security_avg = self._compute_avg(results["security_results"])

        # Save judge logs to file for analysis
        if judge_logs:
            judge_logs_path = self.log_dir / f"judge_logs_{self._run_id}.json"
            with open(judge_logs_path, "w", encoding="utf-8") as f:
                json.dump(judge_logs, f, indent=2, ensure_ascii=False)
            print(f"  Judge logs saved to {judge_logs_path}")

        # Save metrics

        # Save metrics
        metrics = {
            "run_id": self._run_id,
            "suite_name": suite_name,
            "model": str(model),
            "attack": attack,
            "benchmark_version": benchmark_version,
            "utility_avg": utility_avg,
            "security_avg": security_avg,
            "blocked_tool_calls": blocked_count,
            "total_judge_calls": len(judge_logs),
            "on_suspicious": self.on_suspicious,
            # Task details
            "user_tasks_run": list(results["utility_results"].keys()) if results else [],
            "injection_tasks_run": list(results.get("injection_tasks_utility_results", {}).keys()) if results else [],
            "num_user_tasks": len(results["utility_results"]) if results else 0,
            "num_injection_tasks": len(results.get("injection_tasks_utility_results", {})) if results else 0,
        }

        metrics_path = self.log_dir / f"metrics_{suite_name}_{self._run_id}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        return metrics

    def _collect_shield_data(self, pipeline: AgentPipeline) -> tuple[int, list[dict[str, Any]]]:
        """Traverse pipeline to collect blocked counts and all judge logs."""
        count = 0
        all_judge_logs = []
        for element in pipeline.elements:
            if isinstance(element, ToolsExecutionLoop):
                for inner in element.elements:
                    if isinstance(inner, IntentShieldElement):
                        count += len(inner.get_blocked_log())
                        all_judge_logs.extend(inner.get_judge_log())
        return count, all_judge_logs

    def _compute_avg(self, results: dict) -> float:
        """Compute average from dict values (utility/security results)."""
        if not results:
            return 0.0
        values = list(results.values())
        # Results are bools: True=1, False=0
        return sum(1 for v in values if v) / len(values) if values else 0.0