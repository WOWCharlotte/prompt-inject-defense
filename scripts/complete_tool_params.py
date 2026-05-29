#!/usr/bin/env python3
"""
LLM-based tool parameter completion for InjecAgent datasets.

Reads attacker_cases_dh.jsonl and attacker_cases_ds.jsonl, uses LLM to generate
valid tool parameters based on tools.json schema, and outputs enriched JSONL files.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path and load .env
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from intent_shield.config import ShieldConfig


def load_tools_schema() -> dict[str, dict]:
    """Load tools.json and build tool_name -> parameter_schema mapping."""
    with open("InjecAgent/data/tools.json", encoding="utf-8") as f:
        toolkits = json.load(f)

    schema = {}
    for toolkit in toolkits:
        prefix = toolkit["toolkit"]
        for tool in toolkit["tools"]:
            full_name = f"{prefix}{tool['name']}"
            params = {
                p["name"]: {"type": p["type"], "required": p.get("required", False), "description": p.get("description", "")}
                for p in tool.get("parameters", [])
            }
            schema[full_name] = params
    return schema


def build_completion_prompt(
    tool_name: str,
    attacker_instruction: str,
    param_schema: dict[str, Any],
) -> str:
    """Build prompt for LLM to extract/fill tool parameters."""
    required_params = [n for n, p in param_schema.items() if p.get("required")]
    optional_params = [n for n, p in param_schema.items() if not p.get("required")]

    required_text = "\n".join(
        f"- {n} ({p['type']}): {p.get('description', '')}" for n, p in param_schema.items() if p.get("required")
    )
    optional_text = "\n".join(
        f"- {n} ({p['type']}): {p.get('description', '')}" for n, p in param_schema.items() if not p.get("required")
    )

    return f"""你是一个工具参数提取专家。

攻击者指令: "{attacker_instruction}"

工具完整名称: {tool_name}

工具参数定义:
必需参数:
{required_text or "(无)"}

可选参数:
{optional_text or "(无)"}

请根据攻击者指令，提取或推断出所有必需参数的值。只返回JSON格式的工具参数，不要包含其他文字。

示例输出格式:
{{"param_name1": "value1", "param_name2": "value2"}}

只输出JSON，不要其他解释文字。"""


def complete_case_params(
    client: OpenAI,
    model: str,
    tool_name: str,
    attacker_instruction: str,
    param_schema: dict[str, Any],
) -> dict[str, Any]:
    """Call LLM to complete tool parameters for a single case."""
    prompt = build_completion_prompt(tool_name, attacker_instruction, param_schema)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个精确的工具参数提取助手。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content

    # Parse JSON response
    try:
        params = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        import re

        match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        if match:
            params = json.loads(match.group(1))
        else:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                params = json.loads(match.group(0))
            else:
                raise ValueError(f"Cannot parse JSON from: {raw}")

    return params


def process_dataset(
    client: OpenAI,
    model: str,
    input_path: str,
    output_path: str,
    schema: dict[str, dict],
    dataset_name: str,
) -> tuple[int, int]:
    """Process one dataset (DH or DS), return (success_count, failure_count)."""
    success = 0
    failures = []

    with open(input_path, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f]

    print(f"\n[{dataset_name}] Processing {len(cases)} cases...")

    results = []
    for i, case in enumerate(cases):
        case_num = i + 1
        tools = case["Attacker Tools"]
        instruction = case["Attacker Instruction"]

        print(f"\n[{dataset_name}] Case {case_num}/{len(cases)}")
        print(f"  Tools: {tools}")
        print(f"  Instruction: {instruction[:80]}...")

        case_results = []
        for tool_name in tools:
            if tool_name not in schema:
                print(f"  [WARN] Tool not in schema: {tool_name}")
                case_results.append({"tool": tool_name, "params": {}, "error": "tool_not_in_schema"})
                continue

            try:
                params = complete_case_params(client, model, tool_name, instruction, schema[tool_name])
                print(f"  [OK] {tool_name}: {list(params.keys())}")
                case_results.append({"tool": tool_name, "params": params})
                success += 1
            except Exception as e:
                print(f"  [FAIL] {tool_name}: {e}")
                case_results.append({"tool": tool_name, "params": {}, "error": str(e)})
                failures.append((case_num, tool_name, str(e)))

        # Build enriched case
        enriched = {
            **case,
            "Attacker Tools": tools,  # keep original
            "LLM Generated Params": {r["tool"]: r["params"] for r in case_results},
            "Enrichment Errors": {r["tool"]: r["error"] for r in case_results if "error" in r},
        }
        results.append(enriched)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[{dataset_name}] Complete! Success: {success}, Failures: {len(failures)}")
    if failures:
        print("  Failures:")
        for case_num, tool_name, err in failures:
            print(f"    Case {case_num}, {tool_name}: {err}")

    return success, len(failures)


def main():
    # Load config from environment
    config = ShieldConfig()

    api_key = config.llm_api_key
    base_url = config.llm_base_url or "https://api.openai.com/v1"
    model = config.llm_model or "gpt-4o-mini"

    if not api_key:
        print("ERROR: INTENT_SHIELD_LLM_KEY not set in environment")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    schema = load_tools_schema()
    print(f"Loaded schema for {len(schema)} tools")

    # Process DH dataset
    dh_success, dh_fail = process_dataset(
        client,
        model,
        "InjecAgent/data/attacker_cases_dh.jsonl",
        "InjecAgent/data/attacker_cases_dh_enriched.jsonl",
        schema,
        "DH",
    )

    # Process DS dataset
    ds_success, ds_fail = process_dataset(
        client,
        model,
        "InjecAgent/data/attacker_cases_ds.jsonl",
        "InjecAgent/data/attacker_cases_ds_enriched.jsonl",
        schema,
        "DS",
    )

    print(f"\n{'='*50}")
    print(f"Total: DH {dh_success}/{dh_success + dh_fail}, DS {ds_success}/{ds_success + ds_fail}")
    print(f"Output files:")
    print(f"  InjecAgent/data/attacker_cases_dh_enriched.jsonl")
    print(f"  InjecAgent/data/attacker_cases_ds_enriched.jsonl")


if __name__ == "__main__":
    main()