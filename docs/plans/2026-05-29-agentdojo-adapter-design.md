# AgentDojo 适配器设计文档

> 日期: 2026-05-29
> 状态: 设计完成，待实现

---

## 1. 整体架构

**核心思路**：将 `IntentShieldElement` 作为 `BasePipelineElement` 插入到 `ToolsExecutionLoop` 中，位于 `ToolsExecutor` 之前。它拦截每个 tool_call，调用 `shield.judge()` 进行意图对齐审查。

```
AgentPipeline 完整结构:
[SystemMessage] → [InitQuery] → [LLM] → [ToolsExecutionLoop]
                                              ↑
                                    [IntentShieldElement, ToolsExecutor]
                                              ↑
                                    对每个 tool_call 调用 judge()

Tool call 被判断为 Dangerous → 从 tool_calls 列表中移除
Tool call 被判断为 Suspicious → 根据配置阻断或记录
```

**与 AgentDojo 内置防御的关键差异**：

| 维度 | TransformersBasedPIDetector | IntentShieldElement |
|------|------------------------------|---------------------|
| 检测时机 | tool 输出后（事后检测） | tool 执行前（主动拦截） |
| 检测方法 | 文本分类模型（deberta） | LLM 意图对齐判断 |
| 防御效果 | 数据已泄露，仅标记 | 在攻击发生前阻断 |
| RAG 支持 | 无 | 有（动态反思学习） |

---

## 2. 组件设计

### 2.1 IntentShieldElement

```python
# adapters/agentdojo/pipeline.py

class IntentShieldElement(BasePipelineElement):
    """在 ToolsExecutor 执行前拦截并审查 tool_calls。"""

    def __init__(
        self,
        shield: IntentShield,
        on_suspicious: Literal["block", "log"] = "block",
    ):
        self.shield = shield
        self.on_suspicious = on_suspicious
        self._blocked_log: list[dict] = []

    def query(self, query, runtime, env, messages, extra_args):
        # 仅处理 assistant 消息带 tool_calls 的情况
        if not messages or messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        if not messages[-1]["tool_calls"]:
            return query, runtime, env, messages, extra_args

        # 动态获取 user_instruction：从第一条 user message
        user_instruction = self._extract_user_instruction(messages)
        if not user_instruction:
            return query, runtime, env, messages, extra_args

        # 审查每个 tool_call
        new_tool_calls = []
        for tool_call in messages[-1]["tool_calls"]:
            decision = self.shield.judge(
                user_instruction=user_instruction,
                tool_name=tool_call.function,
                tool_params=tool_call.args,
            )
            if decision.action == RiskLevel.DANGEROUS:
                self._blocked_log.append({
                    "tool": tool_call.function,
                    "args": tool_call.args,
                    "risk_score": decision.risk_score,
                })
                continue  # 移除，不执行
            elif decision.action == RiskLevel.SUSPICIOUS:
                if self.on_suspicious == "block":
                    self._blocked_log.append({...})
                    continue
            new_tool_calls.append(tool_call)

        # 更新 messages 中的 tool_calls
        messages[-1]["tool_calls"] = new_tool_calls or None
        return query, runtime, env, messages, extra_args

    def _extract_user_instruction(self, messages) -> str:
        """从 messages 中提取第一条 user message 内容。"""
        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, list):
                    text_blocks = [c.get("text", "") for c in content if c.get("type") == "text"]
                    return "\n".join(text_blocks)
                return content if isinstance(content, str) else ""
        return ""

    def get_blocked_log(self) -> list[dict]:
        """返回被阻断的 tool_call 记录，用于事后分析。"""
        return self._blocked_log
```

**错误处理策略**：

| 异常场景 | 处理方式 | 理由 |
|----------|----------|------|
| 无法提取 user_instruction | 放行所有 tool_calls | 不影响正常流程 |
| LLM 调用失败 | 保守阻断（移除所有 tool_calls） | Default Deny |
| JSON 解析失败 | 回退到 safe 判定 | 避免误阻断 |

### 2.2 AgentDojoRunner

```python
# adapters/agentdojo/runner.py

class AgentDojoRunner:
    """将 IntentShield 集成到 AgentDojo benchmark 并收集双维指标。"""

    def __init__(
        self,
        shield: IntentShield,
        log_dir: str | Path = "./results",
        on_suspicious: str = "block",
    ):
        self.shield = shield
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run(
        self,
        suite_name: str,
        model: str,
        attack: str | None = None,
        user_tasks: tuple[str, ...] = (),
        injection_tasks: tuple[str, ...] = (),
        benchmark_version: str = "v1.2.2",
    ) -> dict:
        """运行 AgentDojo benchmark 并返回 Utility + Security 双维结果。"""
        # 1. 创建带 IntentShield 的 pipeline
        pipeline = self._build_pipeline(model)

        # 2. 加载 suite
        suite = get_suite(benchmark_version, suite_name)

        # 3. 运行 benchmark
        if attack:
            attacker = load_attack(attack, suite, pipeline)
            results = benchmark_suite_with_injections(
                pipeline, suite, attacker,
                user_tasks=user_tasks, injection_tasks=injection_tasks,
                logdir=self.log_dir / "traces",
                force_rerun=False, benchmark_version=benchmark_version,
            )
        else:
            results = benchmark_suite_without_injections(
                pipeline, suite, user_tasks=user_tasks,
                logdir=self.log_dir / "traces",
                force_rerun=False, benchmark_version=benchmark_version,
            )

        # 4. 收集阻断日志
        blocked_count = self._collect_blocked_count(pipeline)

        return {
            "utility_avg": self._compute_avg(results["utility_results"]),
            "security_avg": self._compute_avg(results["security_results"]),
            "blocked_tool_calls": blocked_count,
            "run_id": self._run_id,
        }

    def _build_pipeline(self, model: str) -> AgentPipeline:
        """构建带 IntentShieldElement 的完整 pipeline。"""
        llm = get_llm_from_model(model)
        system_msg = load_system_message(None)

        intent_shield_element = IntentShieldElement(
            shield=self.shield,
            on_suspicious=self.on_suspicious,
        )

        tools_loop = ToolsExecutionLoop([
            intent_shield_element,
            ToolsExecutor(),
            llm,
        ])

        pipeline = AgentPipeline([
            SystemMessage(system_msg),
            InitQuery(),
            llm,
            tools_loop,
        ])
        pipeline.name = f"intent_shield-{model}"
        return pipeline

    def _collect_blocked_count(self, pipeline: AgentPipeline) -> int:
        """遍历 pipeline 收集 IntentShieldElement 的阻断计数。"""
        count = 0
        for element in pipeline.elements:
            if isinstance(element, ToolsExecutionLoop):
                for inner in element.elements:
                    if isinstance(inner, IntentShieldElement):
                        count += len(inner._blocked_log)
        return count
```

---

## 3. CLI 集成

### 3.1 scripts/evaluate.py 扩展

```bash
# 运行 workspace suite，使用 intent_shield 防御
python scripts/evaluate.py \
    --dataset agentdojo \
    --suite workspace \
    --attack important_instructions \
    --model gpt-4o-mini \
    --defense intent_shield \
    --logdir ./results

# 运行 banking suite，启用 RAG 反思
python scripts/evaluate.py \
    --dataset agentdojo \
    --suite banking \
    --attack tool_knowledge \
    --model gpt-4o-mini \
    --defense intent_shield \
    --rag-enabled

# 触发反思进化（benchmark 后自动收集假阴性并反思）
python scripts/evaluate.py \
    --dataset agentdojo \
    --suite workspace \
    --attack important_instructions \
    --model gpt-4o-mini \
    --defense intent_shield \
    --reflect
```

### 3.2 评估指标

| 指标 | 含义 |
|------|------|
| `utility_avg` | 任务完成率（越高越好） |
| `security_avg` | 攻击阻断率（越高越好） |
| `blocked_tool_calls` | 被 IntentShieldElement 阻断的工具调用总数 |

---

## 4. 反思功能集成

### 4.1 假阴性收集流程

```
benchmark 运行完成
    ↓
收集漏报：security=False 的 case
    ↓
构造 FalseNegativeSample
    ↓
调用 shield.report_false_negative() 或 shield.reflect()
    ↓
ReflectionCase 存入 ChromaDB
    ↓
下次 benchmark 时 RAG 检索生效
```

### 4.2 反射触发时机

- **"no reflect"** — 仅评估，不触发反思
- **"after_benchmark"** — benchmark 结束后批量反思所有漏报

---

## 5. 文件结构

```
adapters/
├── injecagent/
│   ├── loader.py
│   ├── formatter.py
│   └── runner.py
└── agentdojo/
    ├── __init__.py
    ├── loader.py          # 加载 AgentDojo task suites
    ├── pipeline.py        # IntentShieldElement 实现
    └── runner.py         # AgentDojoRunner 实现

src/intent_shield/
└── ...
```

---

## 6. 依赖

AgentDojo 适配器依赖 AgentDojo 包本身：

```toml
[project.optional-dependencies]
agentdojo = ["agentdojo"]
```

---

## 7. 验收测试

| 测试 | 预期结果 |
|------|----------|
| 无防御 baseline | utility 100%, security 取决于攻击 |
| 有 IntentShield (block) | utility 略低（误阻断）, security 显著提升 |
| Dangerous tool_call | 被移除，不执行 |
| LLM 调用失败 | 保守阻断所有 tool_calls |
| 无法获取 user_instruction | 放行所有 tool_calls |
| 反思功能 | benchmark 后假阴性触发反思，RAG 案例增加 |

---

## 8. 关键设计决策总结

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 插入位置 | ToolsExecutionLoop 内，ToolsExecutor 之前 | 与其他防御元素风格一致，实时拦截 |
| user_instruction 获取 | 从第一条 user message 动态提取 | 无需修改 benchmark 脚本 |
| Dangerous 处理 | 直接移除 tool_call | 最小侵入性，默认拒绝 |
| Suspicious 默认行为 | block | 安全第一，宁误报不漏报 |
| 错误时策略 | 保守阻断 | Default Deny |
| 反思支持 | 需要 | 持续进化能力 |