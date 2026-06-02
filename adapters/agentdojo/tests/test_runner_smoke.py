"""Smoke tests for AgentDojoRunner + get_llm_provider + build_intent_shield_pipeline."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from adapters.agentdojo.pipeline import DetectionLog, IntentShieldElement
from adapters.agentdojo.runner import AgentDojoRunner, build_intent_shield_pipeline, get_llm_provider
from agentdojo.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
from agentdojo.models import ModelsEnum
from intent_shield.types import RiskLevel


# ---------------------------------------------------------------------------
# get_llm_provider
# ---------------------------------------------------------------------------

def test_get_llm_provider_exact_enum_match():
    # Use a real ModelsEnum value
    enum_value = next(iter(ModelsEnum))
    provider, model_str = get_llm_provider(enum_value)
    assert provider in {"openai", "anthropic", "cohere", "google"}
    assert model_str == enum_value.value


def test_get_llm_provider_qwen_openai_compat():
    provider, model_str = get_llm_provider("qwen2.5-7b-instruct")
    assert provider == "openai-compatible"
    assert model_str == "qwen2.5-7b-instruct"


def test_get_llm_provider_gpt_substring_fallback():
    provider, model_str = get_llm_provider("gpt-4o-mini-custom")
    # Substring matching against ModelsEnum may or may not match; either way
    # the heuristic must classify as "openai" (because it contains "gpt").
    assert provider in {"openai", "openai-compatible"}
    assert "gpt-4o-mini-custom" in model_str or "gpt-4o" in model_str


def test_get_llm_provider_unknown_falls_back_to_openai():
    provider, model_str = get_llm_provider("some-totally-unknown-model-xyz")
    assert provider == "openai"
    assert model_str == "some-totally-unknown-model-xyz"


# ---------------------------------------------------------------------------
# build_intent_shield_pipeline
# ---------------------------------------------------------------------------

def _shield_mock() -> MagicMock:
    return MagicMock()


def test_build_pipeline_returns_agent_pipeline():
    pipeline = build_intent_shield_pipeline(_shield_mock(), "gpt-4o-mini")
    assert isinstance(pipeline, AgentPipeline)


def test_build_pipeline_element_position_is_after_tools_executor():
    pipeline = build_intent_shield_pipeline(_shield_mock(), "gpt-4o-mini")
    # The last element in the top-level pipeline is the ToolsExecutionLoop
    loop = pipeline.elements[-1]
    assert isinstance(loop, ToolsExecutionLoop)
    element_types = [type(e).__name__ for e in loop.elements]
    # Required order: [ToolsExecutor, IntentShieldElement, LLM]
    assert element_types == ["ToolsExecutor", "IntentShieldElement", type(loop.elements[-1]).__name__]
    # The IntentShieldElement must come AFTER the ToolsExecutor
    tools_exec_idx = element_types.index("ToolsExecutor")
    shield_idx = element_types.index("IntentShieldElement")
    assert shield_idx > tools_exec_idx


def test_build_pipeline_uses_intent_shield_element_with_shield():
    shield = _shield_mock()
    pipeline = build_intent_shield_pipeline(shield, "gpt-4o-mini", mask_output=False)
    loop = pipeline.elements[-1]
    # Use type name check: isinstance fails because BasePipelineElement is a Protocol
    shield_elems = [e for e in loop.elements if type(e).__name__ == "IntentShieldElement"]
    assert len(shield_elems) == 1
    assert shield_elems[0].shield is shield
    assert shield_elems[0].mask_output is False



def test_build_pipeline_name_is_full_model_string():
    pipeline = build_intent_shield_pipeline(_shield_mock(), "gpt-4o-mini")
    assert pipeline.name is not None
    assert "gpt-4o" in pipeline.name


# ---------------------------------------------------------------------------
# AgentDojoRunner._collect_shield_data
# ---------------------------------------------------------------------------

def _make_intent_element_with_logs(n: int) -> IntentShieldElement:
    el = IntentShieldElement(shield=MagicMock(), mask_output=True)
    for i in range(n):
        el._detection_log.append(DetectionLog(
            timestamp="2026-06-02T00:00:00+00:00",
            tool_name=f"tool_{i}",
            tool_args={},
            user_instruction_preview="u",
            output_preview="o",
            risk_score=0.5,
            thought="t",
            action=RiskLevel.SUSPICIOUS,
            masked=(i % 2 == 0),
            error=None,
        ))
    return el


def _make_pipeline_with_shield(elements_in_loop: list) -> AgentPipeline:
    loop = ToolsExecutionLoop(elements_in_loop)
    return AgentPipeline([loop])


def test_collect_shield_data_counts_masked_logs(tmp_path):
    runner = AgentDojoRunner(shield=MagicMock(), log_dir=str(tmp_path), mask_output=True)
    el = _make_intent_element_with_logs(4)  # masked indices: 0, 2
    pipeline = _make_pipeline_with_shield([el, MagicMock()])
    blocked_count, logs = runner._collect_shield_data(pipeline)
    assert blocked_count == 2
    assert len(logs) == 4


def test_collect_shield_data_returns_empty_when_no_shield_in_pipeline(tmp_path):
    runner = AgentDojoRunner(shield=MagicMock(), log_dir=str(tmp_path), mask_output=True)
    pipeline = _make_pipeline_with_shield([MagicMock(), MagicMock()])
    blocked_count, logs = runner._collect_shield_data(pipeline)
    assert blocked_count == 0
    assert logs == []


def test_collect_shield_data_collects_from_multiple_elements(tmp_path):
    runner = AgentDojoRunner(shield=MagicMock(), log_dir=str(tmp_path), mask_output=True)
    el1 = _make_intent_element_with_logs(2)
    el2 = _make_intent_element_with_logs(3)
    pipeline = _make_pipeline_with_shield([el1, MagicMock(), el2])
    blocked_count, logs = runner._collect_shield_data(pipeline)
    # el1: masked at index 0 (i=0 even), i=1 odd → 1 masked
    # el2: masked at i=0, i=2 → 2 masked
    assert blocked_count == 3
    assert len(logs) == 5


# ---------------------------------------------------------------------------
# AgentDojoRunner.run() — full smoke (mocked benchmark)
# ---------------------------------------------------------------------------

def test_run_without_attack_writes_metrics_and_logs(tmp_path):
    shield = MagicMock()
    runner = AgentDojoRunner(shield=shield, log_dir=str(tmp_path), mask_output=True)

    # Build a pipeline with one IntentShieldElement seeded with logs
    intent_el = _make_intent_element_with_logs(3)  # all even-idx masked → 0,2 masked = 2 blocked
    loop = ToolsExecutionLoop([ToolsExecutor(), intent_el, MagicMock()])
    pipeline = AgentPipeline([loop])

    fake_results = {
        "utility_results": {"task_0": True, "task_1": False},
        "security_results": {"inj_0": True},
    }

    with patch("adapters.agentdojo.runner.build_intent_shield_pipeline", return_value=pipeline), \
         patch("adapters.agentdojo.runner.get_suite", return_value=MagicMock(user_tasks={}, injection_tasks={})), \
         patch("adapters.agentdojo.runner.benchmark_suite_without_injections", return_value=fake_results):
        metrics = runner.run("workspace", "gpt-4o-mini")

    assert metrics["suite_name"] == "workspace"
    assert metrics["model"] == "gpt-4o-mini"
    assert metrics["attack"] is None
    assert metrics["utility_avg"] == 0.5  # 1 of 2 utility_results True
    assert metrics["security_avg"] == 1.0  # 1 of 1 security_results True
    assert metrics["total_judge_calls"] == 3
    assert metrics["mask_output"] is True
    assert metrics["num_user_tasks"] == 2
    assert metrics["num_injection_tasks"] == 0

    # Verify files written
    judge_log_files = list(tmp_path.glob("judge_logs_*.ndjson"))
    metrics_files = list(tmp_path.glob("metrics_workspace_*.json"))
    assert len(judge_log_files) == 1
    assert len(metrics_files) == 1
    # NDJSON should have 3 lines
    lines = judge_log_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_run_with_attack_loads_attacker(tmp_path):
    shield = MagicMock()
    runner = AgentDojoRunner(shield=shield, log_dir=str(tmp_path), mask_output=False)

    intent_el = _make_intent_element_with_logs(0)
    loop = ToolsExecutionLoop([ToolsExecutor(), intent_el, MagicMock()])
    pipeline = AgentPipeline([loop])

    fake_results = {
        "utility_results": {},
        "security_results": {},
        "injection_tasks_utility_results": {"inj_0": False},
    }
    fake_attacker = MagicMock()
    fake_suite = MagicMock(user_tasks={}, injection_tasks={})

    with patch("adapters.agentdojo.runner.build_intent_shield_pipeline", return_value=pipeline), \
         patch("adapters.agentdojo.runner.get_suite", return_value=fake_suite), \
         patch("adapters.agentdojo.runner.load_attack", return_value=fake_attacker) as mock_load, \
         patch("adapters.agentdojo.runner.benchmark_suite_with_injections", return_value=fake_results):
        metrics = runner.run("workspace", "gpt-4o-mini", attack="tool_knowledge")

    mock_load.assert_called_once_with("tool_knowledge", fake_suite, pipeline)
    assert metrics["attack"] == "tool_knowledge"
    assert metrics["num_injection_tasks"] == 1
    assert metrics["mask_output"] is False
