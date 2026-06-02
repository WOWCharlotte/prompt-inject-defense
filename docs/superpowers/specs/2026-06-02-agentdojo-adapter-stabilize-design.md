# AgentDojo 适配器稳定化设计

> 日期: 2026-06-02
> 状态: 设计已通过，待实现
> 目标: 把 post-execution 模型的半成品稳定为可用形态
> 范围: 仅 `adapters/agentdojo/` 模块

---

## 1. 背景与目标

`adapters/agentdojo/pipeline.py` 已被部分重构为"post-execution 检测"（judge tool 输出后掩码/净化），但 `runner.py` 仍引用旧的 pre-execution API、构造器签名不匹配、累积了大量死代码（自定义 `AnthropicLLM`、未挂入主路径的 provider 类、冗余 import）。原版设计文档 `docs/plans/2026-05-29-agentdojo-adapter-design.md` 已从工作区删除。

本次重构目标：

- 保留 post-execution 模型，不回到 pre-execution
- 修复 runner 与元素间的契约不一致（方法名、构造器参数、日志字段）
- 清理 `runner.py` 中的死代码与未挂入主路径的 provider 分支
- 把错误策略从"permissive on error"改回"Default Deny"
- 恢复并更新设计文档
- 增加最小测试覆盖

**不在范围内**：恢复 pre-execution 模型、对接上游 `PromptInjectionDetector` 基类、扩展对 Anthropic / Google / Cohere 模型的支持、新增 loader.py / metrics.py / config.py 等独立模块。

---

## 2. 架构

### 2.1 文件布局

```
adapters/agentdojo/
├── __init__.py                  # 重导出 IntentShieldElement, DetectionLog, AgentDojoRunner, build_intent_shield_pipeline
├── pipeline.py                  # IntentShieldElement + DetectionLog
├── runner.py                    # AgentDojoRunner + build_intent_shield_pipeline + get_llm_provider
├── README.md                    # 适配器使用说明（新增）
└── tests/
    ├── __init__.py
    ├── test_pipeline_element.py # 元素单测（mock shield）
    └── test_runner_smoke.py     # Runner 冒烟（mock pipeline + shield）
```

`runner.py` 总行数控制在 300 行以内。

### 2.2 ToolsExecutionLoop 内部顺序

```
[ToolsExecutor] → [IntentShieldElement] → [LLM]
   (执行)             (审查输出)          (生成下一轮)
```

元素从 loop 第一个位置移到 ToolsExecutor 之后。功能等价，但代码读起来更直观（执行 → 验证 → 生成下一轮）。

### 2.3 整体结构图

```
AgentPipeline:
[SystemMessage] → [InitQuery] → [LLM] → [ToolsExecutionLoop
                                              ├ ToolsExecutor
                                              ├ IntentShieldElement   ← post-execution 检测
                                              └ LLM]
```

---

## 3. 组件 API

### 3.1 IntentShieldElement

```python
from pydantic import BaseModel
from intent_shield.types import RiskLevel

class DetectionLog(BaseModel):
    """一条 post-execution 检测记录。"""
    timestamp: str
    tool_name: str
    tool_args: dict[str, Any]        # 来自上一轮 assistant tool_calls.args
    user_instruction_preview: str   # 截断 200 字
    output_preview: str              # 截断 200 字
    risk_score: float
    thought: str
    action: RiskLevel
    masked: bool                     # 实际是否被整段掩码
    error: str | None = None         # judge 异常信息


class IntentShieldElement(BasePipelineElement):
    """Post-execution intent alignment check on tool outputs.

    元素位于 ToolsExecutionLoop 内部、ToolsExecutor 之后。
    当 messages[-1] 为 tool 消息时，调用 IntentShield.judge 评估
    "agent 若 follow 该输出内容是否会偏离用户意图"；按决策掩码/净化输出。
    """

    def __init__(
        self,
        shield: IntentShield,
        mask_output: bool = True,  # 影响 SUSPICIOUS 处置；DANGEROUS 总是掩码
    ): ...

    def query(self, query, runtime, env, messages, extra_args) -> tuple: ...

    # 公开接口
    def get_detection_log(self) -> list[DetectionLog]: ...
    def reset_detection_log(self) -> None: ...

    # 内部
    def _judge_output_alignment(user_instruction, tool_name, tool_args, tool_content) -> ShieldDecision: ...
    def _mask_output(tool_message, risk_score, reason) -> None: ...
    def _sanitize_content(content: str) -> str: ...
    def _extract_user_instruction(messages) -> str: ...
    def _extract_tool_args(messages, tool_call_id: str | None) -> dict[str, Any]: ...
    def _get_text_content(content) -> str: ...
    def _safe_dangerous_decision(tool_name, user_instruction, error) -> ShieldDecision: ...
```

### 3.2 风险处置策略（私有常量 `_ACTION_POLICY`）

| 等级 | 处置 |
|------|------|
| SAFE | 透传，不动 messages |
| SUSPICIOUS + `mask_output=True` | 整段掩码 |
| SUSPICIOUS + `mask_output=False` | 原位净化（移除 `ignore previous instructions` 等模式） |
| DANGEROUS | **总是** 整段掩码（与 `mask_output` 无关） |
| shield.judge 异常 | Default Deny → dangerous → 整段掩码 |

### 3.3 AgentDojoRunner

```python
def get_llm_provider(model: str | ModelsEnum) -> tuple[str, str]:
    """识别 provider 与完整 model 名称。

    优先级：
    1. ModelsEnum 精确匹配 → (provider, full_name)
    2. ModelsEnum 模糊匹配（前缀 / 去 dash 包含） → (provider, full_name)
    3. 启发式：含 "qwen"/"deepseek"/"glm"/"zhipu"/"mistral" → ("openai-compatible", 原串)
    4. 启发式：含 "gpt" → ("openai", 原串)；含 "claude" → ("anthropic", 原串)
    5. 兜底：("openai", 原串)
    """


def build_intent_shield_pipeline(
    shield: IntentShield,
    model: str | ModelsEnum,
    mask_output: bool = True,
) -> AgentPipeline:
    """构建带 IntentShieldElement 的 AgentPipeline。

    所有模型统一走 OpenAILLM + openai.OpenAI()。
    openai-compatible 模型通过环境变量注入：
      OPENAI_COMPATIBLE_BASE_URL, OPENAI_COMPATIBLE_API_KEY
    """


class AgentDojoRunner:
    def __init__(
        self,
        shield: IntentShield,
        log_dir: str | Path = "./results",
        mask_output: bool = True,  # 重命名自 on_suspicious
    ): ...

    def run(
        self,
        suite_name: str,
        model: str | ModelsEnum,
        attack: str | None = None,
        user_tasks: tuple[str, ...] = (),
        injection_tasks: tuple[str, ...] = (),
        benchmark_version: str = "v1.2.2",
        force_rerun: bool = False,
        max_user_tasks: int | None = None,
        max_injection_tasks: int | None = None,
    ) -> dict[str, Any]: ...
```

返回的 metrics 字典字段：

```python
{
    "run_id": str,
    "suite_name": str,
    "model": str,
    "attack": str | None,
    "benchmark_version": str,
    "utility_avg": float,
    "security_avg": float,
    "blocked_tool_calls": int,    # 整段掩码次数
    "total_judge_calls": int,
    "mask_output": bool,
    "user_tasks_run": list[str],
    "injection_tasks_run": list[str],
    "num_user_tasks": int,
    "num_injection_tasks": int,
}
```

### 3.4 __init__.py

```python
from .pipeline import IntentShieldElement, DetectionLog
from .runner import AgentDojoRunner, build_intent_shield_pipeline

__all__ = [
    "IntentShieldElement",
    "DetectionLog",
    "AgentDojoRunner",
    "build_intent_shield_pipeline",
]
```

---

## 4. 数据流

### 4.1 单轮 tool 调用（带检测）

```
[LLM 上一轮生成 tool_calls]
        │
        ▼
[ToolsExecutor.query]
   │  runtime.run_function()  →  FormattedOutput
   │  append ChatToolResultMessage(role="tool", content=[...])
        │
        ▼
[IntentShieldElement.query]
   │  1. messages[-1].role == "tool" 才继续
   │  2. user_instruction = self._extract_user_instruction(messages)
   │  3. tool_name = last_message.tool_call.function (or "unknown")
   │     tool_content = self._get_text_content(last_message.content)
   │  3b. tool_args = self._extract_tool_args(messages, last_message.tool_call_id)
   │      # 自下而上搜索最近一条 role=="assistant" 的消息，
   │      # 在其 tool_calls 列表中按 tool_call_id 匹配 args。
   │      # 找到 assistant 但未匹配 → 停止搜索（最近的 assistant 就是它）。
   │      # 遇到非 assistant / 非 tool 消息 → 停止。
   │      # tool_call_id 为 None / 找不到 / 上游无 args → tool_args = {}。
   │  4. shield.judge(
   │       user_instruction=user_instruction,
   │       tool_name="follow_external_instruction",
   │       tool_params={
   │         "external_content": tool_content[:1000],
   │         "original_tool": tool_name,
   │         "original_tool_args": tool_args,
   │       },
   │     )
   │  5. 按 _ACTION_POLICY 处置 tool_message
   │  6. self._detection_log.append(DetectionLog(
   │       ..., tool_name=tool_name, tool_args=tool_args, ...))
        │
        ▼
[LLM] 生成下一轮回复
```

### 4.2 Runner.run 一次执行

```
AgentDojoRunner.run(suite_name, model, attack, ...)
  │  pipeline = build_intent_shield_pipeline(self.shield, model, self.mask_output)
  │  suite = get_suite(benchmark_version, suite_name)
  │  (可选 max_user_tasks / max_injection_tasks 随机采样)
  │  if attack:
  │    attacker = load_attack(attack, suite, pipeline)
  │    results = benchmark_suite_with_injections(pipeline, suite, attacker, ...)
  │  else:
  │    results = benchmark_suite_without_injections(pipeline, suite, ...)
  │
  │  shield_logs, blocked_count = self._collect_shield_data(pipeline)
  │    # 遍历 pipeline.elements[*].elements[*]，聚合 DetectionLog
  │    # blocked_count = sum(1 for log in logs if log.masked)
  │
  │  metrics = { ... }   # 见 3.3
  │
  │  try:
  │    with open(judge_logs_path, "w", encoding="utf-8") as f:
  │      for log in shield_logs:
  │        f.write(log.model_dump_json() + "\n")   # NDJSON
  │  except OSError as e:
  │    print(f"Warning: failed to save judge logs: {e}")
  │
  │  try:
  │    with open(metrics_path, "w", encoding="utf-8") as f:
  │      json.dump(metrics, f, indent=2, ensure_ascii=False)
  │  except OSError as e:
  │    print(f"Warning: failed to save metrics: {e}")
  │
  │  return metrics
```

### 4.3 build_intent_shield_pipeline 内部

```
provider, model_str = get_llm_provider(model)
base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "https://api.openai.com/v1")
api_key  = os.getenv("OPENAI_COMPATIBLE_API_KEY", os.getenv("OPENAI_API_KEY", ""))
client = openai.OpenAI(base_url=base_url, api_key=api_key)
llm = OpenAILLM(client, model_str)
system_message = load from agentdojo.data.system_messages.yaml["default"]

intent_element = IntentShieldElement(shield=shield, mask_output=mask_output)
tools_loop = ToolsExecutionLoop([
    ToolsExecutor(tool_output_formatter=tool_result_to_str),
    intent_element,
    llm,
])
pipeline = AgentPipeline([
    SystemMessage(system_message),
    InitQuery(),
    llm,
    tools_loop,
])
pipeline.name = model_str
return pipeline
```

**关键简化**：删除分 provider 构造 client 的 if-elif 链（OpenAI / Anthropic / Cohere / Google），统一走 OpenAILLM + openai.OpenAI()，通过环境变量支持 openai-compatible。

---

## 5. 错误处理

| 异常场景 | 检测位置 | 处置 |
|----------|----------|------|
| agentdojo 未安装 | `pipeline.py` import | 保留 `try/except ImportError` 退化为 `Protocol/Any` 占位 |
| `messages` 为空 | `IntentShieldElement.query` | 直接 `return`，不记录日志 |
| `messages[-1].role != "tool"` | 同上 | 直接 `return`，不记录日志 |
| 提取不到 user_instruction | `_extract_user_instruction` | 返回空串 → 元素 `return` |
| `tool_call` 缺失 | 元素 `query` | `tool_name = "unknown"`，仍写 DetectionLog（`tool_args={}`）；若决策为 dangerous/suspicious(mask=True) 仍可对 content 字段执行掩码 |
| `shield.judge()` 异常 | `_judge_output_alignment` | `_safe_dangerous_decision()` → dangerous → 整段掩码（Default Deny）；DetectionLog.error 记录 |
| `_sanitize_content` 内部异常 | `_mask_output` sanitize 分支 | 退化为整段掩码，不抛 |
| `get_suite` 失败 | `AgentDojoRunner.run` | 向上抛（用户配置错误） |
| `benchmark_suite_*` 异常 | 同上 | 向上抛（评估失败，需用户感知） |
| LLM API 异常 | OpenAILLM | agentdojo 自身处理；元素不重复 |
| 写 `judge_logs` / `metrics` 失败 | `AgentDojoRunner.run` | `try/except OSError → print 警告 + 不阻塞 return` |

`_safe_dangerous_decision` 实现：

```python
def _safe_dangerous_decision(
    tool_name: str, user_instruction: str, error: Exception
) -> ShieldDecision:
    return ShieldDecision(
        action=RiskLevel.DANGEROUS,
        risk_score=1.0,
        thought=f"Judge error: {type(error).__name__}: {error}",
        tool_name=tool_name,
        tool_params={},
        user_instruction=user_instruction,
    )
```

---

## 6. 测试

### 6.1 `tests/test_pipeline_element.py`（单元，~120 行）

| 用例 | 验证 |
|------|------|
| `test_safe_passes_through` | SAFE 决策 → messages 不变，log.masked=False |
| `test_dangerous_always_masks` | DANGEROUS + mask_output=False → 仍然整段掩码 |
| `test_suspicious_masks_when_flag_true` | SUSPICIOUS + mask_output=True → 整段掩码 |
| `test_suspicious_sanitizes_when_flag_false` | SUSPICIOUS + mask_output=False → 注入模式被替换为 `[filtered]` |
| `test_judge_error_triggers_default_deny` | shield.judge 抛异常 → dangerous → 整段掩码，log.error 记录 |
| `test_no_user_instruction_skips` | 提取不到 user_instruction → 不调用 judge、不动 messages |
| `test_non_tool_message_skips` | messages[-1].role == "user" → 跳过 |
| `test_detection_log_resettable` | reset 后 get_detection_log() 为空 |
| `test_tool_args_extracted_from_previous_assistant` | messages 含 (user, assistant(tool_calls), tool) 三条 → DetectionLog.tool_args 等于 assistant tool_calls 的 args |
| `test_tool_args_with_multiple_tool_results_in_batch` | assistant 一次发起 2 个 tool_call，产生 2 条 tool 消息；IntentShieldElement 在第 2 条上运行时，args 来自 assistant.tool_calls 中对应的那个 |
| `test_tool_args_with_prior_batch_residue` | 上一轮残留 tool 消息 + 当前轮 assistant(tool_calls) + 新 tool 消息；args 来自当前轮的 assistant，不是历史残留 |

**测试手段**：`unittest.mock.MagicMock` 替换 `IntentShield`；构造最小 `messages` 列表（user + assistant(tool_calls) + tool 消息）。

### 6.2 `tests/test_runner_smoke.py`（冒烟，~60 行）

| 用例 | 验证 |
|------|------|
| `test_build_pipeline_returns_valid_pipeline` | 返回对象是 `AgentPipeline`；`IntentShieldElement` 出现在 `ToolsExecutionLoop.elements` 内 |
| `test_get_llm_provider_known_model` | `"gpt-4o-mini"` → `("openai", "gpt-4o-mini-...")` 之一 |
| `test_get_llm_provider_qwen_openai_compat` | `"qwen2.5-7b-instruct"` → `("openai-compatible", 原串)` |
| `test_collect_shield_data_extracts_logs` | mock pipeline → DetectionLog 列表正确聚合 |

**测试手段**：mock 整个 `benchmark_suite_*` 调用；不真正跑 LLM。

### 6.3 运行

```bash
cd prompt-inject-defense
uv run pytest adapters/agentdojo/tests -v
```

---

## 7. README.md 大纲

1. 适配器定位（一段话：IntentShield 集成进 AgentDojo benchmark）
2. 安装
   ```bash
   uv pip install -e .[agentdojo]
   ```
3. 快速开始
   ```python
   from intent_shield import IntentShield, ShieldConfig
   from adapters.agentdojo import AgentDojoRunner

   shield = IntentShield(ShieldConfig(...))
   runner = AgentDojoRunner(shield, log_dir="./results", mask_output=True)
   metrics = runner.run("workspace", model="gpt-4o-mini", attack="tool_knowledge")
   ```
4. 关键参数表
   - `mask_output`（True = 整段掩码；False = SUSPICIOUS 时原位净化）
5. 指标字段说明
   - `utility_avg` / `security_avg` / `blocked_tool_calls` / `total_judge_calls`
6. 局限
   - 仅支持 OpenAI 兼容模型（Anthropic / Google / Cohere 不在范围）
7. 与 upstream `PromptInjectionDetector` 的差异（1.5 行对比表）

---

## 8. 范围与不做的事

**不做**：
- 不恢复 pre-execution 模型
- 不对接上游 `agentdojo.agent_pipeline.pi_detector.PromptInjectionDetector` 基类
- 不扩展对 Anthropic / Google / Cohere 模型的支持
- 不新增 `loader.py` / `metrics.py` / `config.py` 等独立模块
- 不在 `IntentShield` core 内新增 API（如 `detect_following_alignment`）
- 不改 `src/intent_shield/` 任何代码

**做**：
- 修复 pipeline/runner 之间的契约不一致
- 清理 runner.py 中的死代码、未挂入主路径的 provider 分支、冗余 import
- 把错误策略改回 Default Deny
- 元素在 loop 中的位置从第一个挪到 ToolsExecutor 之后
- 恢复并更新设计文档（本文档）
- 新增单元 + 冒烟测试
- 新增 README

---

## 9. 验收标准

1. `uv run pytest adapters/agentdojo/tests -v` 全部通过
2. `python -c "from adapters.agentdojo import IntentShieldElement, AgentDojoRunner, build_intent_shield_pipeline"` 成功
3. `pipeline.name` 设置为完整 model 字符串，attack 注册表可识别
4. 元素在 `ToolsExecutionLoop.elements` 内的顺序为 `[ToolsExecutor, IntentShieldElement, LLM]`
5. `AgentDojoRunner.run()` 返回的 metrics 包含 §3.3 列出的全部字段
6. `runner.py` 总行数 ≤ 300
7. `pipeline.py` 总行数 ≤ 200
8. `IntentShieldElement.__init__` 不接受 `on_suspicious` 参数（破坏性变更）
9. `AgentDojoRunner.__init__` 不接受 `on_suspicious` 参数（破坏性变更）
10. `docs/plans/2026-05-29-agentdojo-adapter-design.md` 可选保留（标记 deprecated）或删除 —— 由用户决定
