# IntentShield

**Pluggable defense framework against indirect prompt injection attacks on LLM agents.**

IntentShield 通过意图对齐交叉验证和 RAG 驱动的动态反思机制，检测 LLM Agent 中的间接提示注入攻击。

## 核心特性

- **意图对齐验证**: 通过 LLM 裁判交叉验证用户指令与工具调用的意图一致性
- **RAG 动态反思**: ChromaDB 向量数据库存储历史威胁案例，持续自我进化
- **可插拔架构**: 装饰器/Hook 模式，零侵入集成任意 Agent 系统
- **多模型支持**: OpenAI API 兼容，支持 GPT/Qwen/DeepSeek/GLM 等
- **双数据集适配器**: InjecAgent + AgentDojo 评估支持

## 架构概览

```
intent_shield/
├── src/intent_shield/
│   ├── core/           # 核心防御逻辑
│   │   ├── shield.py   # IntentShield 主类 + guard 装饰器
│   │   ├── judge.py    # LLM 裁判
│   │   ├── prompt.py   # 安全提示词构建
│   │   └── decision.py # 风险评分解析
│   ├── reflection/     # 动态反思机制
│   │   ├── rag.py      # ChromaDB 检索
│   │   ├── reflector.py # 反思报告生成
│   │   └── store.py    # 向量存储抽象
│   ├── llm/            # LLM 抽象层
│   ├── embed/          # 嵌入与重排序
│   ├── config.py       # 配置管理
│   └── types.py        # 数据类型
├── adapters/
│   ├── injecagent/     # InjecAgent 适配器
│   └── agentdojo/      # AgentDojo 适配器
└── scripts/
    └── evaluate.py    # 统一评估 CLI
```

## 安装

```bash
# 基础依赖
pip install -e .

# 评估依赖 (InjecAgent/AgentDojo)
pip install -e ".[eval]"

# AgentDojo 额外依赖
pip install -e ".[agentdojo]"
```

## 快速开始

### Python API

```python
from intent_shield import IntentShield, ShieldConfig

# 初始化
shield = IntentShield(ShieldConfig(
    llm_base_url="https://api.openai.com/v1",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    embed_base_url="https://api.siliconflow.cn/v1",
    embed_api_key="sf-...",
))

# 方式1: 装饰器拦截
@shield.guard
def call_tool(tool_name: str, params: dict, user_instruction: str):
    return execute_tool(tool_name, params)

# 方式2: 直接判定
decision = shield.judge(
    user_instruction="总结近期未读邮件",
    tool_name="HospitalBooking",
    tool_params={"doctor": "dentist", "time": "tomorrow"}
)
# decision.action: "safe" | "suspicious" | "dangerous"
# decision.risk_score: 0.0 ~ 1.0
```

### CLI 评估

**InjecAgent 数据集:**

```bash
# 直接危害攻击 (Direct Harm)
python scripts/evaluate.py --dataset injecagent --attack-type dh --setting base

# 数据窃取攻击 (Data Stealing)
python scripts/evaluate.py --dataset injecagent --attack-type ds --setting enhanced

# 禁用 RAG 快速测试
python scripts/evaluate.py --dataset injecagent --no-rag --attack-type all --max-cases 10
```

**AgentDojo 数据集:**

```bash
# Workspace 套件
python scripts/evaluate.py --dataset agentdojo --suite workspace --model gpt-4o-mini

# 使用 DeepSeek/Qwen 模型
python scripts/evaluate.py --dataset agentdojo --suite workspace \
    --model deepseek-chat \
    --model-base-url https://api.deepseek.com/v1 \
    --model-api-key YOUR_KEY

# 限制任务数量
python scripts/evaluate.py --dataset agentdojo --suite workspace \
    --max-user-tasks 5 --max-injection-tasks 5

# 触发反思进化
python scripts/evaluate.py --dataset agentdojo --suite workspace \
    --attack important_instructions --reflect
```

### CLI 参数详解

**通用参数:**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集类型：`injecagent` 或 `agentdojo` | 必需 |
| `--output` | 结果输出 JSON 文件路径 | - |
| `--no-rag` | 禁用 RAG，向量检索和反思 | 启用 |

**InjecAgent 参数:**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--setting` | 攻击设置：`base` 或 `enhanced` | `base` |
| `--attack-type` | 攻击类型：`dh`（直接危害）、`ds`（数据窃取）、`all` | `all` |
| `--max-cases` | 限制测试用例数量 | 全部 |
| `--llm-model` | Judge LLM 模型 | `gpt-4o-mini` |
| `--llm-base-url` | Judge LLM API 地址 | 环境变量或 OpenAI |

**AgentDojo 参数:**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--suite` | 任务套件：`workspace`、`banking`、`travel`、`slack` | `workspace` |
| `--model` | Agent 模型 | `gpt-4o-mini` |
| `--model-base-url` | Agent API 基础地址（用于 OpenAI 兼容模型） | - |
| `--model-api-key` | Agent API 密钥 | - |
| `--attack` | 攻击类型（如 `important_instructions`、`tool_knowledge`） | 无 |
| `--defense` | 防御策略：`intent_shield` 或 `none` | `intent_shield` |
| `--on-suspicious` | 可疑行为处理：`block`（拦截）或 `log`（记录） | `block` |
| `--max-user-tasks` | 限制用户任务数量 | 全部 |
| `--max-injection-tasks` | 限制注入任务数量 | 全部 |
| `--user-task` / `-ut` | 指定运行的特定用户任务 ID（可重复指定） | - |
| `--injection-task` / `-it` | 指定运行的特定注入任务 ID（可重复指定） | - |
| `--reflect` | 评估后触发反思，收集假阴性样本 | 关闭 |

## 配置

可通过 `ShieldConfig` 或环境变量配置:

| 参数 | 环境变量 | 默认值 |
|------|---------|--------|
| `llm_api_key` | `INTENT_SHIELD_LLM_KEY` | - |
| `embed_api_key` | `INTENT_SHIELD_EMBED_KEY` | - |
| `llm_base_url` | - | `https://api.openai.com/v1` |
| `embed_base_url` | - | `https://api.siliconflow.cn/v1` |
| `llm_model` | - | `gpt-4o-mini` |
| `alpha` (安全阈值) | - | `0.3` |
| `beta` (危险阈值) | - | `0.7` |

或使用 `.env` 文件:

```bash
INTENT_SHIELD_LLM_KEY=sk-...
INTENT_SHIELD_EMBED_KEY=sf-...
```

## 防御流程

```
用户指令 → RAG检索历史案例 → 构建安全提示词 → LLM裁判判定 → 分段决策
                                                     ├─ score < 0.3 → Safe
                                                     ├─ 0.3 ≤ score ≤ 0.7 → Suspicious
                                                     └─ score > 0.7 → Dangerous
```

## 反思机制

当检测到假阴性（攻击成功但未被拦截）时:

1. **批量模式**: `shield.reflect(audit_log)` 扫描审核日志，批量生成反思报告
2. **流式模式**: `shield.report_false_negative(...)` 单条即时入库

反思结果存储至 ChromaDB，下次 RAG 检索时可获取最新威胁模式。

## 项目结构

```
.
├── src/intent_shield/     # 核心库
├── adapters/             # 数据集适配器
│   ├── injecagent/       # InjecAgent 评估
│   └── agentdojo/        # AgentDojo 评估
├── scripts/              # CLI 脚本
├── tests/                # 单元测试
├── data/                 # ChromaDB 持久化目录
└── results/              # 评估结果
```

## License

MIT