"""IntentShield evaluation CLI.

Usage:
    python scripts/evaluate.py --dataset injecagent --setting base --max-cases 10
    python scripts/evaluate.py --dataset injecagent --setting enhanced --no-rag
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "agentdojo" / "src"))  # For agentdojo submodule
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file if present
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    import os

    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from intent_shield import IntentShield, ShieldConfig


def main():
    parser = argparse.ArgumentParser(description="IntentShield Evaluation CLI")
    parser.add_argument(
        "--dataset",
        choices=["injecagent", "agentdojo"],
        required=True,
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--setting",
        choices=["base", "enhanced"],
        default="base",
        help="Attack setting (InjecAgent only)",
    )
    parser.add_argument(
        "--attack-type",
        choices=["dh", "ds", "all"],
        default="all",
        help="Attack type to evaluate (InjecAgent only)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit number of test cases (for quick testing)",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM model for the judge (default: from .env or gpt-4o-mini)",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="LLM API base URL (default: from .env or https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG (no vector DB, no embedding/reranking)",
    )
    parser.add_argument(
        "--suite",
        default="workspace",
        help="AgentDojo task suite (default: workspace)",
    )
    parser.add_argument(
        "--attack",
        default=None,
        help="AgentDojo attack type (e.g., important_instructions, tool_knowledge)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model to use for benchmark (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--model-base-url",
        default=None,
        help="Base URL for OpenAI-compatible model (e.g., https://api.deepseek.com/v1)",
    )
    parser.add_argument(
        "--model-api-key",
        default=None,
        help="API key for OpenAI-compatible model",
    )
    parser.add_argument(
        "--defense",
        choices=["intent_shield", "none"],
        default="intent_shield",
        help="Defense to use for AgentDojo (default: intent_shield)",
    )
    parser.add_argument(
        "--on-suspicious",
        choices=["block", "log"],
        default="block",
        help="How to handle suspicious tool calls (default: block)",
    )
    parser.add_argument(
        "--reflect",
        action="store_true",
        help="Trigger reflection after benchmark to collect false negatives",
    )
    parser.add_argument(
        "--user-task",
        "-ut",
        action="append",
        default=None,
        dest="user_tasks",
        help="Specific user task ID to run (can be repeated)",
    )
    parser.add_argument(
        "--injection-task",
        "-it",
        action="append",
        default=None,
        dest="injection_tasks",
        help="Specific injection task ID to run (can be repeated)",
    )
    parser.add_argument(
        "--max-user-tasks",
        type=int,
        default=None,
        help="Limit number of user tasks to run (random sample if not enough)",
    )
    parser.add_argument(
        "--max-injection-tasks",
        type=int,
        default=None,
        help="Limit number of injection tasks to run (random sample if not enough)",
    )
    args = parser.parse_args()

    # Build config: CLI args override .env, which overrides defaults
    config_kwargs = {"rag_enabled": not args.no_rag}
    if args.llm_base_url:
        config_kwargs["llm_base_url"] = args.llm_base_url
    if args.llm_model:
        config_kwargs["llm_model"] = args.llm_model

    config = ShieldConfig(**config_kwargs)
    shield = IntentShield(config)

    if args.dataset == "injecagent":
        run_injecagent(shield, args)
    elif args.dataset == "agentdojo":
        run_agentdojo(shield, args)


def run_injecagent(shield: IntentShield, args):
    from adapters.injecagent.runner import InjecAgentRunner

    data_dir = PROJECT_ROOT / "InjecAgent" / "data"
    runner = InjecAgentRunner(shield)

    all_metrics = {}

    if args.attack_type in ("dh", "all"):
        dh_path = data_dir / f"test_cases_dh_{args.setting}.json"
        if dh_path.exists():
            print(f"Running Direct Harm ({args.setting})...")
            metrics = runner.run(dh_path, max_cases=args.max_cases)
            all_metrics["DH"] = metrics.summary()
            print(f"  DH ASR: {metrics.asr_dh:.2f}% ({metrics.dh_succ} succ / {metrics.valid_cases} valid)")
        else:
            print(f"  File not found: {dh_path}")

    if args.attack_type in ("ds", "all"):
        ds_path = data_dir / f"test_cases_ds_{args.setting}.json"
        if ds_path.exists():
            print(f"Running Data Stealing ({args.setting})...")
            shield.clear_audit_log()
            metrics = runner.run(ds_path, max_cases=args.max_cases)
            all_metrics["DS"] = metrics.summary()
            print(f"  DS ASR: {metrics.asr_ds:.2f}% ({metrics.ds_s1_succ} succ / {metrics.valid_cases} valid)")
        else:
            print(f"  File not found: {ds_path}")

    print("\n=== Results ===")
    print(json.dumps(all_metrics, indent=2, ensure_ascii=False))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


def run_agentdojo(shield: IntentShield, args):
    from agentdojo.models import ModelsEnum
    from adapters.agentdojo.runner import AgentDojoRunner

    # Convert model string to ModelsEnum if applicable
    model = args.model
    for enum_val in ModelsEnum:
        if enum_val.value == args.model:
            model = enum_val
            break

    # Set environment variables for OpenAI-compatible models (qwen, deepseek, etc.)
    if args.model_base_url:
        os.environ["OPENAI_COMPATIBLE_BASE_URL"] = args.model_base_url
    if args.model_api_key:
        os.environ["OPENAI_COMPATIBLE_API_KEY"] = args.model_api_key

    # Convert user_tasks/injection_tasks to tuples if provided via action.append
    user_tasks = tuple(args.user_tasks) if args.user_tasks else ()
    injection_tasks = tuple(args.injection_tasks) if args.injection_tasks else ()

    runner = AgentDojoRunner(
        shield=shield,
        log_dir=PROJECT_ROOT / "results",
        on_suspicious=args.on_suspicious,
    )

    print(f"Running AgentDojo benchmark:")
    print(f"  Suite: {args.suite}")
    print(f"  Model: {args.model}")
    print(f"  Attack: {args.attack or 'none'}")
    print(f"  Defense: {args.defense}")
    print(f"  RAG: {'enabled' if not args.no_rag else 'disabled'}")
    print()

    metrics = runner.run(
        suite_name=args.suite,
        model=model,
        attack=args.attack,
        user_tasks=user_tasks,
        injection_tasks=injection_tasks,
        max_user_tasks=args.max_user_tasks,
        max_injection_tasks=args.max_injection_tasks,
    )

    print("\n=== Results ===")
    print(f"Utility (task success): {metrics['utility_avg']:.1%}")
    print(f"Security (attack blocked): {metrics['security_avg']:.1%}")
    print(f"Blocked tool calls: {metrics['blocked_tool_calls']}")

    # Trigger reflection if requested
    if args.reflect and args.attack:
        print("\nRunning reflection on false negatives...")
        reflect_results(shield, args, metrics)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


def reflect_results(shield: IntentShield, args, metrics: dict):
    """Collect false negatives from trace logs and trigger reflection.

    Loads trace logs from results/traces/ directory, identifies cases where
    security=False (attack succeeded despite IntentShield), and triggers
    batch reflection to improve future detection.
    """
    import json
    from pathlib import Path

    trace_dir = PROJECT_ROOT / "results" / "traces"
    if not trace_dir.exists():
        print(f"No trace directory found at {trace_dir}")
        return

    # Find pipeline trace directories
    pipeline_name = f"intent_shield-{args.model}"
    pipeline_dir = trace_dir / pipeline_name / args.suite

    if not pipeline_dir.exists():
        print(f"No trace logs found for {pipeline_name}/{args.suite}")
        return

    false_negatives = []
    user_instruction = ""

    # Walk through all trace files
    for user_task_dir in pipeline_dir.iterdir():
        if not user_task_dir.is_dir():
            continue
        for attack_dir in user_task_dir.iterdir():
            if not attack_dir.is_dir():
                continue
            for trace_file in attack_dir.iterdir():
                if trace_file.suffix != ".json":
                    continue
                try:
                    with open(trace_file, "r", encoding="utf-8") as f:
                        trace = json.load(f)

                    # Check if this is a false negative (security=False means attack succeeded)
                    if trace.get("security") is False:
                        # Extract messages to find user instruction and tool calls
                        messages = trace.get("messages", [])
                        user_instruction = extract_user_instruction_from_messages(messages)
                        tool_calls = extract_tool_calls_from_messages(messages)

                        if user_instruction and tool_calls:
                            # Get the last tool call that was executed (attack succeeded)
                            for tc in reversed(tool_calls):
                                false_negatives.append({
                                    "user_instruction": user_instruction,
                                    "tool_name": tc["function"],
                                    "tool_params": tc["args"],
                                    "initial_thought": f"Attack succeeded: {tc['function']}",
                                })
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Failed to parse {trace_file}: {e}")
                    continue

    if not false_negatives:
        print("No false negatives found in trace logs.")
        return

    print(f"Found {len(false_negatives)} false negatives. Running reflection...")

    # Convert to FalseNegativeSample and trigger reflection
    for fn in false_negatives:
        from intent_shield.types import FalseNegativeSample
        sample = FalseNegativeSample(**fn)
        try:
            shield.report_false_negative(
                user_instruction=sample.user_instruction,
                tool_name=sample.tool_name,
                tool_params=sample.tool_params,
                initial_thought=sample.initial_thought,
            )
            print(f"  Reflected: {sample.tool_name}")
        except Exception as e:
            print(f"  Warning: Failed to reflect {sample.tool_name}: {e}")

    print(f"Reflection complete. {len(false_negatives)} cases added to RAG store.")


def extract_user_instruction_from_messages(messages: list) -> str:
    """Extract user instruction from the first user message in the conversation."""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                text_blocks = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
                return "\n".join(text_blocks)
            return content if isinstance(content, str) else ""
    return ""


def extract_tool_calls_from_messages(messages: list) -> list:
    """Extract all tool calls from assistant messages."""
    tool_calls = []
    for msg in messages:
        if msg.get("role") == "assistant":
            tc_list = msg.get("tool_calls") or []
            tool_calls.extend(tc_list)
    return tool_calls


if __name__ == "__main__":
    main()
