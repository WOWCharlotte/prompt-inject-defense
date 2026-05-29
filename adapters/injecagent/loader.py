from __future__ import annotations

import json
from pathlib import Path


def load_test_cases(path: str | Path) -> list[dict]:
    """Load InjecAgent test cases from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tools(path: str | Path) -> dict[str, dict]:
    """Load tool definitions from tools.json. Returns {toolkit_tool_name: tool_spec}."""
    with open(path, "r", encoding="utf-8") as f:
        toolkits = json.load(f)

    tools = {}
    for toolkit in toolkits:
        toolkit_name = toolkit["name_for_model"]
        for tool in toolkit["tools"]:
            full_name = f"{toolkit_name}{tool['name']}"
            tools[full_name] = tool
    return tools
