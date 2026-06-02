# IntentShield × AgentDojo Adapter

Post-execution intent alignment defense for LLM agents running the
[AgentDojo](https://github.com/ethz-spylab/agentdojo) benchmark suite.

The adapter intercepts every tool output after `ToolsExecutor` runs,
judges whether the agent following the output would deviate from the
user's original instruction, and masks / sanitizes the output before
the LLM sees it.

## Install

```bash
uv pip install -e .[agentdojo]
```

## Quick start

```python
from intent_shield import IntentShield, ShieldConfig
from adapters.agentdojo import AgentDojoRunner

shield = IntentShield(ShieldConfig(...))
runner = AgentDojoRunner(shield, log_dir="./results", mask_output=True)

metrics = runner.run(
    "workspace",
    model="gpt-4o-mini",
    attack="tool_knowledge",
)
print(metrics["utility_avg"], metrics["security_avg"])
```

## Configuration

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `mask_output` | `bool` | `True` | When True, suspicious tool outputs are replaced with a marker. When False, only the injection pattern strings (`ignore previous instructions`, etc.) are removed in place. **Dangerous** outputs are always fully masked regardless of this flag. |
| `log_dir` | `str \| Path` | `"./results"` | Where to write `judge_logs_<run_id>.ndjson` and `metrics_<suite>_<run_id>.json`. |
| `attack` | `str \| None` | `None` | AgentDojo attack name (e.g. `"tool_knowledge"`, `"important_instructions"`). `None` runs the utility-only benchmark. |

## Metrics fields

| Field | Meaning |
|-------|---------|
| `utility_avg` | Fraction of user tasks the agent completed successfully. |
| `security_avg` | Fraction of injection tasks the agent **resisted**. |
| `blocked_tool_calls` | Number of tool outputs fully masked (dangerous + suspicious with `mask_output=True`). |
| `total_judge_calls` | Total post-execution judge invocations. |
| `run_id` | UTC timestamp of the run. |
| `num_user_tasks` / `num_injection_tasks` | Counts. |

## Limitations

- Only OpenAI-compatible models are supported (`gpt-*`, `qwen-*`, `deepseek-*`, `glm-*`, `mistral-*`, etc.). For OpenAI-compatible providers, set `OPENAI_COMPATIBLE_BASE_URL` and `OPENAI_COMPATIBLE_API_KEY`.
- Anthropic, Cohere, and Google models are **not** in scope for this adapter.
- This is a post-execution defense — tool calls that have already been executed are reviewed, not blocked beforehand.

## Comparison with upstream `PromptInjectionDetector`

| Dimension | upstream `TransformersBasedPIDetector` | IntentShield adapter |
|-----------|----------------------------------------|----------------------|
| Detection method | DeBERTa text classifier | LLM-based intent alignment |
| Supports RAG | No | Yes (configurable) |
| Reflection / learning | No | Yes (via `IntentShield.reflect()`) |
| Runtime cost | Fast (small classifier) | Slow (one LLM call per tool output) |
