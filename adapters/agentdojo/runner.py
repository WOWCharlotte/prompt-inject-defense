"""AgentDojo benchmark runner with IntentShield integration."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import (
    benchmark_suite_with_injections,
    benchmark_suite_without_injections,
)
from agentdojo.models import MODEL_PROVIDERS, ModelsEnum
from agentdojo.task_suite.load_suites import get_suite

from adapters.agentdojo.pipeline import IntentShieldElement


# OpenAI-compatible model name patterns → forced to openai-compatible provider.
_OPENAI_COMPAT_PATTERNS = ("qwen", "deepseek", "glm", "zhipu", "mistral")


def get_llm_provider(model: str | ModelsEnum) -> tuple[str, str]:
    """Identify the provider and canonical model name for a model identifier.

    Resolution order:
        1. Exact `ModelsEnum` value match → (provider, full_name)
        2. Substring / dash-stripped match against `ModelsEnum`
        3. Heuristic: contains "qwen"/"deepseek"/"glm"/"zhipu"/"mistral"
           → ("openai-compatible", original string)
        4. Heuristic: contains "gpt" → ("openai", original string)
        5. Heuristic: contains "claude" → ("anthropic", original string)
        6. Fallback: ("openai", original string)
    """
    model_str = model.value if isinstance(model, ModelsEnum) else model

    # 1. exact enum match
    for enum_val in ModelsEnum:
        if enum_val.value == model_str:
            return MODEL_PROVIDERS[enum_val], enum_val.value

    # 2. fuzzy match (substring or dash-stripped prefix)
    needle = model_str.replace("-", "")
    for enum_val in ModelsEnum:
        full = enum_val.value
        if model_str in full or full.replace("-", "").startswith(needle):
            return MODEL_PROVIDERS[enum_val], full

    # 3-5. heuristic
    lower = model_str.lower()
    for pattern in _OPENAI_COMPAT_PATTERNS:
        if pattern in lower:
            return "openai-compatible", model_str
    if "gpt" in lower:
        return "openai", model_str
    if "claude" in lower:
        return "anthropic", model_str

    # 6. fallback
    return "openai", model_str


def build_intent_shield_pipeline(
    shield: Any,
    model: str | ModelsEnum,
    mask_output: bool = True,
) -> Any:
    """Build an AgentPipeline with IntentShieldElement inserted.

    All models are routed through `OpenAILLM` + `openai.OpenAI()`. OpenAI-
    compatible providers (qwen, deepseek, glm, zhipu, mistral) are reached
    via the `OPENAI_COMPATIBLE_BASE_URL` and `OPENAI_COMPATIBLE_API_KEY`
    environment variables.

    Pipeline structure:
        [SystemMessage] → [InitQuery] → [LLM] → [ToolsExecutionLoop(
            [ToolsExecutor, IntentShieldElement, LLM]
        )]

    The element is placed AFTER `ToolsExecutor` so that the tool message is
    already in `messages` when the element runs.
    """
    import importlib.resources

    import openai
    import yaml
    from agentdojo.agent_pipeline import AgentPipeline
    from agentdojo.agent_pipeline.agent_pipeline import InitQuery, SystemMessage
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
    from agentdojo.agent_pipeline.tool_execution import (
        ToolsExecutionLoop,
        ToolsExecutor,
        tool_result_to_str,
    )

    _provider, model_str = get_llm_provider(model)

    base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    llm = OpenAILLM(client, model_str)

    # Load default system message from agentdojo package data
    package_files = importlib.resources.files("agentdojo")
    path = package_files / "data" / "system_messages.yaml"
    with importlib.resources.as_file(path) as p, open(p) as f:
        system_messages = yaml.safe_load(f)
    system_message = system_messages.get("default", "")

    intent_element = IntentShieldElement(shield=shield, mask_output=mask_output)
    tools_loop = ToolsExecutionLoop([
        ToolsExecutor(tool_output_formatter=tool_result_to_str),
        intent_element,
        llm,
    ])

    pipeline = AgentPipeline([
        SystemMessage(system_message),
        InitQuery(),
        llm,
        tools_loop,
    ])
    pipeline.name = model_str
    return pipeline


def _is_intent_shield_element(obj: Any) -> bool:
    """Duck-typed check for IntentShieldElement.

    agentdojo's BasePipelineElement may be a Protocol (no @runtime_checkable),
    making isinstance() raise TypeError. Fall back to duck-typing in that case.
    """
    try:
        return isinstance(obj, IntentShieldElement)
    except TypeError:
        return hasattr(obj, "get_detection_log") and hasattr(obj, "reset_detection_log")


class AgentDojoRunner:
    """Runs AgentDojo benchmark with IntentShield integrated.

    Args:
        shield: The IntentShield instance for intent alignment checking.
        log_dir: Directory to save NDJSON judge logs and metrics JSON.
        mask_output: Forwarded to IntentShieldElement. When True, suspicious
            tool outputs are fully masked; when False, they are sanitized
            in place. Dangerous outputs are always fully masked.
    """

    def __init__(
        self,
        shield: Any,
        log_dir: str | Path = "./results",
        mask_output: bool = True,
    ) -> None:
        self.shield = shield
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.mask_output = mask_output
        self._run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def _collect_shield_data(
        self, pipeline: Any
    ) -> tuple[int, list[Any]]:
        """Aggregate DetectionLogs from every IntentShieldElement in the pipeline.

        Returns:
            (blocked_count, logs) where `blocked_count` is the number of
            logs whose `masked` field is True.
        """
        from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop

        blocked = 0
        all_logs: list[Any] = []
        for element in pipeline.elements:
            if isinstance(element, ToolsExecutionLoop):
                for inner in element.elements:
                    if _is_intent_shield_element(inner):
                        logs = inner.get_detection_log()
                        all_logs.extend(logs)
                        blocked += sum(1 for log in logs if log.masked)
        return blocked, all_logs

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
        """Run the AgentDojo benchmark with IntentShield integrated.

        Returns a dict with utility_avg, security_avg, blocked_tool_calls,
        total_judge_calls, run_id, plus task / attack / config metadata.
        """
        import random

        pipeline = build_intent_shield_pipeline(
            shield=self.shield,
            model=model if isinstance(model, str) else model.value,
            mask_output=self.mask_output,
        )
        suite = get_suite(benchmark_version, suite_name)

        # Optional random sampling of user / injection tasks
        if max_user_tasks and not user_tasks:
            all_user_ids = list(suite.user_tasks.keys())
            if len(all_user_ids) > max_user_tasks:
                user_tasks = tuple(random.sample(all_user_ids, max_user_tasks))
        if max_injection_tasks and not injection_tasks and attack:
            all_inj_ids = list(suite.injection_tasks.keys())
            if len(all_inj_ids) > max_injection_tasks:
                injection_tasks = tuple(random.sample(all_inj_ids, max_injection_tasks))

        if attack:
            attacker = load_attack(attack, suite, pipeline)
            results = benchmark_suite_with_injections(
                pipeline,
                suite,
                attacker,
                user_tasks=user_tasks if user_tasks else None,
                injection_tasks=injection_tasks if injection_tasks else None,
                logdir=self.log_dir / "traces",
                force_rerun=force_rerun,
                benchmark_version=benchmark_version,
            )
        else:
            results = benchmark_suite_without_injections(
                pipeline,
                suite,
                user_tasks=user_tasks if user_tasks else None,
                logdir=self.log_dir / "traces",
                force_rerun=force_rerun,
                benchmark_version=benchmark_version,
            )

        blocked_count, judge_logs = self._collect_shield_data(pipeline)
        utility_avg = self._compute_avg(results.get("utility_results", {}))
        security_avg = self._compute_avg(results.get("security_results", {}))

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
            "mask_output": self.mask_output,
            "user_tasks_run": list(results.get("utility_results", {}).keys()),
            "injection_tasks_run": list(
                results.get("injection_tasks_utility_results", {}).keys()
            ),
            "num_user_tasks": len(results.get("utility_results", {})),
            "num_injection_tasks": len(
                results.get("injection_tasks_utility_results", {})
            ),
        }

        self._save_outputs(metrics, judge_logs, suite_name)
        return metrics

    def _save_outputs(
        self,
        metrics: dict[str, Any],
        judge_logs: list[Any],
        suite_name: str,
    ) -> None:
        """Write NDJSON judge logs and metrics JSON. Failures are warnings, not errors."""
        if judge_logs:
            judge_logs_path = self.log_dir / f"judge_logs_{self._run_id}.ndjson"
            try:
                with open(judge_logs_path, "w", encoding="utf-8") as f:
                    for log in judge_logs:
                        f.write(log.model_dump_json() + "\n")
            except OSError as e:
                print(f"Warning: failed to save judge logs to {judge_logs_path}: {e}")

        metrics_path = self.log_dir / f"metrics_{suite_name}_{self._run_id}.json"
        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
        except OSError as e:
            print(f"Warning: failed to save metrics to {metrics_path}: {e}")

    @staticmethod
    def _compute_avg(results: dict[str, bool]) -> float:
        if not results:
            return 0.0
        truthy = sum(1 for v in results.values() if v)
        return truthy / len(results)