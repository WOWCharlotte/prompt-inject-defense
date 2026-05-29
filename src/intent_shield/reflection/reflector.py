from __future__ import annotations

import json
import re

from intent_shield.llm.base import BaseLLM
from intent_shield.reflection.store import ThreatStore
from intent_shield.types import FalseNegativeSample, ReflectionCase

REFLECTION_SYSTEM_PROMPT = """\
你是一个AI安全防御系统的反思引擎。你的任务是分析防御系统的漏报案例\
（即系统错误地判定为安全，但实际包含恶意意图的工具调用），\
并生成结构化的反思报告，用于增强未来的防御能力。"""

REFLECTION_USER_PROMPT_TEMPLATE = """\
## 漏报案例分析

以下是一个防御系统未能拦截的恶意工具调用:

- **用户原始指令**: {user_instruction}
- **被调用的工具**: {tool_name}
- **工具参数**: {tool_params}
- **系统初始判断思路**: {initial_thought}

## 分析要求

请深入分析此漏报案例，生成结构化反思报告。你需要:
1. 识别攻击者利用的隐蔽特征（为什么这个工具调用看起来合理但实际有害）
2. 分析防御系统失效的根本原因（为什么初始审核未能识别）
3. 提出针对性的防御补丁（未来遇到类似模式时应如何判断）

请严格按照以下JSON格式输出:
{{"attack_feature": "攻击者利用的隐蔽特征描述", "failed_reason": "防御系统失效的根本原因分析", "new_guideline": "针对此类攻击模式的具体防御指南"}}"""


class Reflector:
    """Generates structured reflection reports from false negative samples."""

    def __init__(self, llm: BaseLLM, store: ThreatStore):
        self.llm = llm
        self.store = store

    def reflect_single(self, sample: FalseNegativeSample) -> ReflectionCase:
        """Generate reflection for a single false negative and store it."""
        messages = [
            {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": REFLECTION_USER_PROMPT_TEMPLATE.format(
                    user_instruction=sample.user_instruction,
                    tool_name=sample.tool_name,
                    tool_params=json.dumps(sample.tool_params, ensure_ascii=False),
                    initial_thought=sample.initial_thought or "无初始判断记录",
                ),
            },
        ]

        raw = self.llm.chat_sync(messages, temperature=0.0)
        data = self._parse_reflection(raw)

        case = ReflectionCase(
            attack_feature=data.get("attack_feature", ""),
            failed_reason=data.get("failed_reason", ""),
            new_guideline=data.get("new_guideline", ""),
            source_instruction=sample.user_instruction,
            source_tool=sample.tool_name,
            source_params=json.dumps(sample.tool_params, ensure_ascii=False),
        )

        self.store.add_case(case)
        return case

    def reflect_batch(self, samples: list[FalseNegativeSample]) -> list[ReflectionCase]:
        """Batch reflection after an evaluation round."""
        return [self.reflect_single(s) for s in samples]

    @staticmethod
    def _parse_reflection(raw: str) -> dict:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {
            "attack_feature": "解析失败",
            "failed_reason": "解析失败",
            "new_guideline": "解析失败",
        }
