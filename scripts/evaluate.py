"""IntentShield evaluation CLI.

Usage:
    python scripts/evaluate.py --dataset injecagent --setting base --max-cases 10
    python scripts/evaluate.py --dataset injecagent --setting enhanced --no-rag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
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
        print("AgentDojo adapter not yet implemented.")
        sys.exit(1)


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


if __name__ == "__main__":
    main()
