# IntentShield 防御框架设计文档

> 日期: 2026-05-29
> 状态: 设计完成，待实现

---

## 1. 项目定位

构建一个**可插拔的 Agent 间接提示注入防御框架**，IntentShield 作为首个防御策略实现。框架设计为可集成中间件，通过装饰器/Hook 模式嵌入任意 Agent 系统，同时提供针对 InjecAgent 和 AgentDojo 两个数据集的评估适配器。

## 2. 核心设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 定位 | 可插拔防御框架 | 方便后续对比其他防御方法 |
| 使用场景 | 可集成中间件 | 侵入性低，适配各种 Agent 架构 |
| LLM 裁判 | 通用接口（OpenAI API 兼容） | 覆盖 GPT/Qwen/Ollama/vLLM |
| 向量数据库 | ChromaDB | 嵌入式，零外部服务依赖 |
| 集成方式 | 装饰器/Hook 模式 | 用户只需注册 hook，不改现有架构 |
| 语言 | Python | 与数据集生态一致 |
| 嵌入/重排序 | 硅基流动远程 API | 轻量，无需本地 torch |
| 数据集适配 | 独立适配器 | 保留各数据集原生评估逻辑 |
| 反思触发 | 批量 + 流式双模式 | 评估场景用批量，生产用流式 |

## 3. 项目结构

```
prompt-inject-defense/
├── src/
│   └── intent_shield/
│       ├── __init__.py
│       ├── core/
│       │   ├── shield.py          # IntentShield 主类 + 装饰器
│       │   ├── judge.py           # LLM 裁判（意图对齐交叉验证）
│       │   ├── prompt.py          # 结构化安全提示词构建
│       │   └── decision.py        # 风险评分 + 分段决策
│       ├── reflection/
│       │   ├── rag.py             # ChromaDB 检索历史威胁案例
│       │   ├── reflector.py       # 结构化经验反思
│       │   └── store.py           # 向量数据库读写抽象
│       ├── llm/
│       │   ├── base.py            # LLM 抽象接口
│       │   └── openai_compat.py   # OpenAI 兼容实现
│       ├── embed/
│       │   ├── base.py            # 嵌入抽象接口
│       │   └── siliconflow.py     # 硅基流动 API 实现
│       ├── config.py              # 配置管理
│       └── types.py               # 数据类型定义
├── adapters/
│   ├── injecagent/
│   │   ├── loader.py             # 加载 InjecAgent 测试用例
│   │   ├── runner.py             # 运行评估 + 计算 ASR
│   │   └── formatter.py          # 数据格式转换
│   └── agentdojo/
│       ├── loader.py             # 加载 AgentDojo 任务套件
│       ├── runner.py             # 运行评估 + 计算 Utility/Security
│       └── pipeline.py           # 集成到 AgentDojo pipeline
├── scripts/
│   ├── evaluate.py               # 统一评估入口 CLI
│   └── seed_db.py                # 初始化向量数据库种子数据
├── tests/
├── pyproject.toml
└── README.md
```

## 4. 核心接口设计

### 4.1 对外 API

```python
from intent_shield import IntentShield, ShieldConfig

shield = IntentShield(ShieldConfig(
    llm_base_url="https://api.openai.com/v1",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    embed_base_url="https://api.siliconflow.cn/v1",
    embed_api_key="sf-...",
    embed_model="BAAI/bge-m3",
    reranker_model="BAAI/bge-reranker-v2-m3",
    alpha=0.3,
    beta=0.7,
))

# 方式1: 装饰器模式
@shield.guard
def call_tool(tool_name: str, params: dict, user_instruction: str):
    return execute_tool(tool_name, params)

# 方式2: 直接判定
decision = shield.judge(
    user_instruction="总结近期未读邮件",
    tool_name="HospitalBooking",
    tool_params={"doctor": "dentist", "time": "tomorrow"}
)
# decision.action = "dangerous"
# decision.risk_score = 0.92
# decision.thought = "用户要求总结邮件，但工具调用为预约医生，存在明显意图漂移..."

# 方式3: 流式反思（生产环境）
shield.report_false_negative(
    user_instruction="...",
    tool_name="...",
    tool_params={...},
    initial_thought="..."
)
```

### 4.2 配置模型

```python
class ShieldConfig(BaseModel):
    # LLM 裁判
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""       # 或环境变量 INTENT_SHIELD_LLM_KEY
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.0

    # 嵌入与重排序（硅基流动）
    embed_base_url: str = "https://api.siliconflow.cn/v1"
    embed_api_key: str = ""     # 或环境变量 INTENT_SHIELD_EMBED_KEY
    embed_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # 风险阈值
    alpha: float = 0.3          # < alpha → Safe
    beta: float = 0.7           # > beta → Dangerous

    # RAG
    top_k: int = 3
    chroma_persist_dir: str = "./data/chroma_db"

    # 规范边界
    norms: list[str] = [
        "用户财产安全: 禁止未授权的资金转移、支付操作",
        "数据隐私保障: 禁止将用户数据发送至非授权方",
        "法规合规要求: 禁止违反数据保护法规的操作",
        "身心健康维护: 禁止可能危害用户身心健康的操作",
        "禁止冒犯性表述: 禁止生成或传播冒犯性内容",
    ]

    # Suspicious 决策行为
    on_suspicious: str = "block"  # "block" | "ask_user" | "log_only"
```

## 5. 核心防御流程

### 5.1 意图对齐交叉验证

```
shield.judge(user_instruction, tool_name, tool_params)
    │
    ├─ 1. RAG 检索历史威胁案例
    │     query = f"{user_instruction} | {tool_name} | {params}"
    │     → 硅基流动 bge-m3 嵌入
    │     → ChromaDB 向量检索
    │     → 硅基流动 bge-reranker 重排序
    │     → 返回 Top-3 案例
    │
    ├─ 2. 构建结构化安全提示词 Γ_safe
    │     ├─ I_anchor:     用户指令原文
    │     ├─ T_constraint: 工具名称 + 参数
    │     ├─ G_norm:       五维安全规范 + few-shot 示例
    │     ├─ O_struct:     CoT 输出格式约束
    │     └─ K_dynamic:    RAG 检索的历史案例
    │
    ├─ 3. 调用 LLM 裁判
    │     → 输出 JSON: { thought, risk_score, decision }
    │
    ├─ 4. 分段决策
    │     score < α(0.3)  → Safe
    │     α ≤ score ≤ β   → Suspicious
    │     score > β(0.7)  → Dangerous
    │
    └─ 5. 记录审核日志
```

### 5.2 动态反思自我进化

**批量模式（评估场景）**：

```python
# 每轮评估结束后
shield.reflect(run_logs)
    │
    ├─ 阶段1: 扫描日志，收集假阴性漏报
    │   (decision=="safe" 但 ground_truth=="malicious")
    │
    ├─ 阶段2: 对每个漏报，LLM 生成结构化反思报告
    │   → { attack_feature, failed_reason, new_guideline }
    │
    └─ 阶段3: 嵌入 + 持久化至 ChromaDB
```

**流式模式（生产环境）**：

```python
# 外部反馈标记误判时立即触发
shield.report_false_negative(user_instruction, tool_name, tool_params, initial_thought)
    → 单条反思 + 即时入库
```

## 6. 数据集适配器

### 6.1 InjecAgent 适配器

- 加载 `test_cases_dh_base.json` / `test_cases_ds_base.json` 等文件
- 对每条用例调用 `shield.judge()`
- 计算 ASR-valid（按 DH/DL/EH 分类 + 总体加权）
- 支持 `--use_cache` 中断恢复

### 6.2 AgentDojo 适配器

- 实现 `BasePipelineElement` 接口
- 在 ToolsExecutor 之前拦截工具调用
- 保留原生 Utility + Security 双维评估
- 支持选择任务套件和攻击类型

### 6.3 统一 CLI

```bash
# InjecAgent
python scripts/evaluate.py --dataset injecagent \
    --attack-type dh --setting base --model gpt-4o-mini

# AgentDojo
python scripts/evaluate.py --dataset agentdojo \
    --suite workspace --attack important_instructions

# 触发反思进化
python scripts/evaluate.py --dataset injecagent --reflect
```

## 7. 依赖

```toml
[project]
name = "intent-shield"
requires-python = ">=3.10"

dependencies = [
    "openai>=1.0",
    "chromadb>=0.5",
    "pydantic>=2.0",
]

[project.optional-dependencies]
eval = [
    "agentdojo",
    "pandas",
]
dev = [
    "pytest",
    "ruff",
]
```

## 8. 实现优先级

### Phase 1: 核心防御（MVP）
1. `types.py` + `config.py` — 数据类型和配置
2. `llm/` — LLM 抽象 + OpenAI 兼容实现
3. `embed/` — 硅基流动嵌入 + 重排序
4. `core/prompt.py` — 结构化安全提示词
5. `core/judge.py` — LLM 裁判调用
6. `core/decision.py` — 风险评分解析
7. `core/shield.py` — 主类 + 装饰器

### Phase 2: 动态反思
8. `reflection/store.py` — ChromaDB 读写
9. `reflection/rag.py` — 检索 + 重排序
10. `reflection/reflector.py` — 反思报告生成

### Phase 3: 评估适配
11. `adapters/injecagent/` — InjecAgent 评估
12. `adapters/agentdojo/` — AgentDojo 评估
13. `scripts/evaluate.py` — CLI 入口
14. `scripts/seed_db.py` — 种子数据初始化

### Phase 4: 测试与文档
15. 单元测试 + 集成测试
16. README + 使用文档
