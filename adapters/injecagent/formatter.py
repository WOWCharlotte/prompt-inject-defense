from __future__ import annotations

import ast
import json


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


def _build_attacker_params(tool_name: str, instruction: str) -> dict:
    """Build a simplified params dict from the attacker instruction.

    Since InjecAgent doesn't provide structured attacker params directly,
    we construct a minimal representation from the instruction text.
    """
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
