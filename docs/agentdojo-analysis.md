# AgentDojo 分析报告

## 1. 项目概述

**AgentDojo** 是由 ETH Zurich 和 Invariant Labs 开发的动态基准测试环境，用于评估针对 LLM Agent 的 Prompt Injection 攻击与防御。

- 论文: https://arxiv.org/abs/2406.13352
- 核心目标: 提供标准化框架，测试 LLM Agent 在面对 prompt injection 攻击时的安全性和功能性

## 2. 架构设计

```
src/agentdojo/
├── agent_pipeline/          # 核心 Agent 执行管道
│   ├── llms/               # LLM 提供商实现 (OpenAI, Anthropic, Cohere, Google)
│   ├── agent_pipeline.py   # 管道组合与配置
│   ├── basic_elements.py   # 系统消息、查询初始化
│   ├── tool_execution.py   # 工具调用执行
│   ├── pi_detector.py      # Prompt Injection 检测器
│   └── planner.py          # Agent 规划逻辑
├── attacks/                 # 攻击实现
│   ├── base_attacks.py     # 基础攻击类
│   ├── baseline_attacks.py # 简单攻击 (direct, ignore_previous 等)
│   ├── important_instructions_attacks.py  # 复杂攻击
│   ├── dos_attacks.py      # 拒绝服务攻击
│   └── attack_registry.py  # 攻击注册系统
├── default_suites/         # 预构建任务套件
│   ├── v1/ ~ v1_2_2/      # 多版本任务套件
│   └── tools/              # 共享工具实现
├── task_suite/             # 任务套件框架
├── functions_runtime.py    # 函数执行运行时
├── base_tasks.py           # 基础任务类
├── benchmark.py            # 基准测试逻辑
└── scripts/benchmark.py    # CLI 入口
```

## 3. 核心机制

### 3.1 执行流程

1. **管道构建**: 组合 SystemMessage → InitQuery → LLM → ToolsExecutionLoop
2. **任务执行**: 同时运行用户任务（合法操作）和注入任务（攻击目标）
3. **攻击注入**: 在工具输出中的注入向量位置插入恶意指令
4. **双维评估**: 分别评估 Utility（功能完成度）和 Security（安全性）

### 3.2 攻击体系

| 攻击类型 | 描述 | 复杂度 |
|---------|------|--------|
| DirectAttack | 直接注入目标指令 | 低 |
| IgnorePreviousAttack | "忽略之前的指令" | 低 |
| SystemMessageAttack | 伪装系统消息 | 中 |
| InjecAgentAttack | 模拟 InjecAgent 风格 | 中 |
| ImportantInstructionsAttack | 精心构造的复杂攻击 | 高 |
| ToolKnowledgeAttack | 包含工具规格信息 | 高 |
| DoSAttack (多种变体) | 拒绝服务攻击 | 中 |

攻击通过 canary 值检测可见注入点，仅在 Agent 实际可见的位置注入。

### 3.3 防御机制

| 防御方法 | 原理 |
|---------|------|
| tool_filter | 使用 LLM 过滤可用工具 |
| transformers_pi_detector | DeBERTa 模型检测工具输出中的注入 |
| spotlighting_with_delimiting | 用分隔符标记工具输出，指示模型忽略其中内容 |
| repeat_user_prompt | 在工具执行循环中重复注入原始用户提示 |

## 4. 任务套件

四个模拟真实场景的任务域:

| 套件 | 场景 | 工具示例 |
|------|------|---------|
| Workspace | 邮件、日历、云盘 | send_email, create_calendar_event |
| Banking | 金融交易 | transfer_money, check_balance |
| Travel | 旅行预订 | search_flights, book_hotel |
| Slack | 消息平台 | send_message, create_channel |

每个套件包含: 用户任务 + 注入任务 + 环境状态(YAML) + 注入向量

## 5. 支持的模型

| 提供商 | 模型 |
|--------|------|
| OpenAI | GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo |
| Anthropic | Claude 3/3.5/3.7 系列 |
| Cohere | Command R, Command R+ |
| Google | Gemini 1.5/2.0/2.5 系列 |
| Together | Mixtral, Llama 3 |
| Local | 通过 vLLM 的自定义模型 |

## 6. 评估指标

- **Utility Score**: 用户任务完成率 = `成功任务数 / 总任务数`
- **Security Score**: 注入攻击防御率 = `成功防御数 / 总攻击数`
- 支持基于消息和基于函数调用 trace 的双重评估方式

## 7. 设计亮点

1. **管道组合模式**: 灵活组合 LLM、防御、工具执行组件
2. **函数注册系统**: 从 Python 函数自动提取 schema（docstring → 描述，type hints → 参数）
3. **攻击/防御分离**: 任意攻击可与任意防御组合，形成完整评估矩阵
4. **版本化系统**: 支持多个基准版本，追踪改进和回归
5. **注入向量系统**: YAML 模板 + 占位符，运行时动态注入
6. **惰性求值与缓存**: ground truth 管道仅运行一次，canary 检测使用 LRU 缓存
