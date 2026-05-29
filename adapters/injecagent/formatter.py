from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path

# Cache for enriched cases loaded once
_ENRICHED_DH: list[dict] | None = None
_ENRICHED_DS: list[dict] | None = None


def _load_enriched_cases() -> tuple[list[dict], list[dict]]:
    """Load enriched cases from both datasets (lazy, cached)."""
    global _ENRICHED_DH, _ENRICHED_DS

    if _ENRICHED_DH is None:
        data_dir = Path(__file__).parent.parent.parent / "InjecAgent" / "data"
        dh_path = data_dir / "attacker_cases_dh_enriched.jsonl"
        ds_path = data_dir / "attacker_cases_ds_enriched.jsonl"

        _ENRICHED_DH = []
        if dh_path.exists():
            with open(dh_path, encoding="utf-8") as f:
                for line in f:
                    _ENRICHED_DH.append(json.loads(line))

        _ENRICHED_DS = []
        if ds_path.exists():
            with open(ds_path, encoding="utf-8") as f:
                for line in f:
                    _ENRICHED_DS.append(json.loads(line))

    return _ENRICHED_DH, _ENRICHED_DS


def extract_attack_info(case: dict) -> tuple[str, str, dict]:
    """Extract (user_instruction, attacker_tool, attacker_params) from a test case.

    For IntentShield evaluation, we simulate the scenario where the agent
    has been compromised and is about to call the attacker's tool.
    We ask the shield to judge whether this tool call aligns with user intent.
    """
    user_instruction = case["User Instruction"]
    attacker_tool = case["Attacker Tools"][0]

    # Extract params from the attacker instruction or tool response
    attacker_instruction = case.get("Attacker Instruction", "")
    tool_params = _build_attacker_params(attacker_tool, attacker_instruction)

    return user_instruction, attacker_tool, tool_params


def _find_enriched_params(
    attacker_instruction: str, attacker_tool: str
) -> dict | None:
    """Look up LLM-generated params from enriched files.

    Matches by Attacker Instruction exact string.
    Returns None if no match found.
    """
    dh_cases, ds_cases = _load_enriched_cases()

    for case in dh_cases + ds_cases:
        if case.get("Attacker Instruction") == attacker_instruction:
            params_dict = case.get("LLM Generated Params", {})
            if attacker_tool in params_dict:
                return params_dict[attacker_tool]
            # Tool not in this case's params — treat as no match
    return None


def _build_attacker_params(tool_name: str, instruction: str) -> dict:
    """Build tool params from enriched files, fallback to instruction+tool name.

    First checks attacker_cases_dh_enriched.jsonl / attacker_cases_ds_enriched.jsonl
    for LLM-generated params. Falls back to a minimal representation if not found.
    """
    params = _find_enriched_params(instruction, tool_name)
    if params is not None:
        return params

    # Fallback — log warning and return minimal representation
    warnings.warn(
        f"[IntentShield] Missing params for tool '{tool_name}', "
        f"instruction: '{instruction[:60]}...'. "
        f"Falling back to instruction-as-param."
    )
    return {"instruction": instruction, "tool": tool_name}


def classify_attack_type(case: dict) -> str:
    """Classify a test case into DH (Direct Harm), DL (Data Leak), or EH (Email Hijack).

    InjecAgent's data stealing involves two steps:
    - Step 1: data extraction (DL)
    - Step 2: email exfiltration (EH)

    Direct harm cases are single-step (DH).
    """
    attacker_tools = case.get("Attacker Tools", [])

    # Data stealing cases have multiple attacker tools (extract + exfiltrate)
    if len(attacker_tools) >= 2:
        # Check if second tool is email-related
        second_tool = attacker_tools[1].lower() if len(attacker_tools) > 1 else ""
        if "email" in second_tool or "gmail" in second_tool:
            return "DS"  # Data Stealing (covers both DL and EH)
    return "DH"  # Direct Harm
