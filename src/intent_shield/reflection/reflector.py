from __future__ import annotations

import json
import re

from intent_shield.llm.base import BaseLLM
from intent_shield.reflection.store import ThreatStore
from intent_shield.types import FalseNegativeSample, ReflectionCase

REFLECTION_SYSTEM_PROMPT = """你是一个 AI 安全反思引擎（Dynamic Reflection Engine）。

你的任务不是简单总结一次漏报，而是从一次防御失败中，提炼未来可复用的安全经验，帮助系统持续进化。

你必须将当前案例视为一个"新的攻击模式（Attack Pattern）"，目标是：
- 可泛化的攻击特征
- 可复用的检测经验
- 可长期存储的安全先验
- 可用于未来 RAG 检索的防御知识"""

REFLECTION_USER_PROMPT_TEMPLATE = """## 漏报案例

- **用户原始指令**: {user_instruction}
- **被调用工具**: {tool_name}
- **工具参数**: {tool_params}
- **系统初始推理**: {initial_thought}
- **最终实际风险行为**: {actual_attack_behavior}

---

## 核心分析目标

1. 为什么这个攻击"表面合理"？
2. 攻击者利用了哪些语义伪装？
3. 用户原始目标是如何逐步偏移的？
4. 为什么初始审核误判为安全？
5. 当前防御规则遗漏了什么？
6. 未来如何识别类似攻击？

---

## 强制分析维度

### 1. Semantic Camouflage（语义伪装）
分析攻击为什么看起来像正常 workflow：
- 合法工具伪装、合理参数伪装
- 合法前置步骤伪装、间接任务依赖伪装
- "帮助用户"伪装

### 2. Intent Drift Path（意图漂移路径）
分析用户原始目标如何一步步偏移至攻击者目标。
重点关注：workflow hijacking、参数污染、隐式任务替换、工具链偏移、外部内容诱导。

### 3. Root Cause of Failure（失效根因）
为什么初始 Judge 未能识别风险：
- 过度相信参数、错误允许"合理推断"
- 未检测 workflow 跳转、忽略隐式数据流
- 缺少历史攻击经验、过度依赖表面语义

### 4. Attack Pattern Abstraction（攻击模式抽象）
必须抽象为通用攻击模式：
- Parameter Injection、Workflow Hijacking、Semantic Drift
- Cross-Tool Escalation、Hidden Data Exfiltration、Context Poisoning

### 5. Future Defense Strategy（未来防御策略）
必须具体、可执行、可泛化，禁止模糊建议。

---

## 输出格式（必须严格遵守）

返回 JSON：
__JSON_OUTPUT__"""

REFLECTION_OUTPUT_SCHEMA = {
    "attack_summary": "string",
    "root_cause_analysis": {
        "primary_failure_reason": "string",
        "missing_detection_signal": "string",
        "incorrect_assumption": "string",
        "security_boundary_gap": "string"
    },
    "attack_pattern": {
        "pattern_name": "string",
        "pattern_category": "string",
        "generalized_attack_feature": "string",
        "trigger_signal": ["string"]
    },
    "defense_patch": {
        "new_detection_rule": "string",
        "risk_escalation_condition": "string",
        "future_handling_strategy": "string",
        "recommended_risk_level": "safe | suspicious | dangerous"
    }
}


class Reflector:
    """Generates structured reflection reports from false negative samples."""

    def __init__(self, llm: BaseLLM, store: ThreatStore):
        self.llm = llm
        self.store = store

    def reflect_single(self, sample: FalseNegativeSample) -> ReflectionCase:
        """Generate reflection for a single false negative and store it."""
        actual_behavior = getattr(sample, "actual_attack_behavior", None) or sample.initial_thought or "未知"

        user_content = REFLECTION_USER_PROMPT_TEMPLATE.format(
            user_instruction=sample.user_instruction,
            tool_name=sample.tool_name,
            tool_params=json.dumps(sample.tool_params, ensure_ascii=False, indent=2),
            initial_thought=sample.initial_thought or "无初始判断记录",
            actual_attack_behavior=actual_behavior,
        ).replace("__JSON_OUTPUT__", json.dumps(REFLECTION_OUTPUT_SCHEMA, ensure_ascii=False, indent=2))

        messages = [
            {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw = self.llm.chat_sync(messages, temperature=0.0)
        data = self._parse_reflection(raw)

        case = ReflectionCase(
            attack_feature=data.get("attack_summary", ""),
            failed_reason=self._flatten_root_cause(data),
            new_guideline=self._build_guideline_from_patch(data),
            source_instruction=sample.user_instruction,
            source_tool=sample.tool_name,
            source_params=json.dumps(sample.tool_params, ensure_ascii=False),
        )

        self.store.add_case(case)
        return case

    @staticmethod
    def _flatten_root_cause(data: dict) -> str:
        rca = data.get("root_cause_analysis", {})
        if not rca:
            return data.get("failed_reason", "")
        parts = [
            f"主因: {rca.get('primary_failure_reason', '')}",
            f"遗漏信号: {rca.get('missing_detection_signal', '')}",
            f"错误假设: {rca.get('incorrect_assumption', '')}",
            f"边界缺口: {rca.get('security_boundary_gap', '')}",
        ]
        return " | ".join(p for p in parts if rca.get(p.split(": ")[0], ""))

    @staticmethod
    def _build_guideline_from_patch(data: dict) -> str:
        patch = data.get("defense_patch", {})
        guideline = data.get("new_guideline", "")
        if patch and not guideline:
            guideline = (
                f"[检测规则] {patch.get('new_detection_rule', '')} | "
                f"[升级条件] {patch.get('risk_escalation_condition', '')} | "
                f"[策略] {patch.get('future_handling_strategy', '')}"
            )
        return guideline

    def reflect_batch(self, samples: list[FalseNegativeSample]) -> list[ReflectionCase]:
        """Batch reflection after an evaluation round."""
        return [self.reflect_single(s) for s in samples]

    @staticmethod
    def _parse_reflection(raw: str) -> dict:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try markdown code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Greedy brace matching for nested JSON
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {
            "attack_summary": "解析失败",
            "root_cause_analysis": {},
            "defense_patch": {},
        }
