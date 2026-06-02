"""Test IntentShield on normal (non-injection) user cases from InjecAgent.

Usage:
    python scripts/test_user_cases.py [--max-cases N] [--no-rag]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file if present
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            import os
            os.environ.setdefault(key.strip(), value.strip())

from intent_shield import IntentShield, ShieldConfig


def main():
    parser = argparse.ArgumentParser(description="Test IntentShield on user cases")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of cases")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG")
    parser.add_argument("--output", default=None, help="Output JSON file for results")

    default_output = PROJECT_ROOT / "results" / f"user_cases_{datetime.now().strftime('%Y:%m:%d_%H%M%S')}.json"
    parser.set_defaults(output=str(default_output))
    args = parser.parse_args()

    # Load test cases
    data_path = PROJECT_ROOT / "InjecAgent" / "data" / "user_cases.jsonl"
    lines = data_path.read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines]

    if args.max_cases:
        cases = cases[: args.max_cases]

    print(f"Testing {len(cases)} user cases (RAG={'enabled' if not args.no_rag else 'disabled'})")
    print("-" * 60)

    # Initialize shield
    config = ShieldConfig(rag_enabled=not args.no_rag)
    shield = IntentShield(config)

    results = {
        "total": len(cases),
        "safe": 0,
        "suspicious": 0,
        "dangerous": 0,
        "errors": 0,
        "details": [],
    }

    for i, case in enumerate(cases):
        tool_name = case["User Tool"]
        user_instruction = case["User Instruction"]
        tool_params = eval(case["Tool Parameters"])  # noqa: PGH001

        try:
            trace = shield.judge_with_trace(
                user_instruction=user_instruction,
                tool_name=tool_name,
                tool_params=tool_params,
            )
            decision = trace.decision
            action = decision.action.value
            risk_score = decision.risk_score

            if action == "safe":
                results["safe"] += 1
            elif action == "suspicious":
                results["suspicious"] += 1
            else:
                results["dangerous"] += 1

            results["details"].append({
                "case_id": i,
                "tool": tool_name,
                "user_instruction": user_instruction,
                "action": action,
                "risk_score": risk_score,
                "thought": decision.thought,
                "llm_input": trace.prompt_messages,
                "llm_output": trace.raw_response,
            })

            print(f"[{i + 1}/{len(cases)}] {action:12s} (score={risk_score:.2f}) - {user_instruction[:60]}...")

        except Exception as e:
            results["errors"] += 1
            results["details"].append({
                "case_id": i,
                "tool": tool_name,
                "error": str(e),
                "llm_input": None,
                "llm_output": None,
            })
            print(f"[{i + 1}/{len(cases)}] ERROR: {e}")

    print("-" * 60)
    print(f"Results: {results['safe']} safe, {results['suspicious']} suspicious, {results['dangerous']} dangerous, {results['errors']} errors")
    print(f"Success rate (safe): {results['safe'] / results['total'] * 100:.1f}%")

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()