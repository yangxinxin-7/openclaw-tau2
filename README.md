# openclaw-tau2

使用 [OpenClaw](https://github.com/openclaw/openclaw) 作为被测 Agent，在 [TAU-2 (τ²-bench)](https://github.com/sierra-research/tau2-bench) 基准上进行评测，并通过 LanceDB 长期记忆实现 train→test 知识迁移。

## 架构

```
TAU-2 编排器
    │
    ├─ 用户模拟器（LLM）         ← 模拟客户
    └─ OpenClawAgent（本项目）   ← 被评测方
           │
           └─ POST http://127.0.0.1:18789/v1/responses  (Gateway 模式)
                  │
                  └─ 火山引擎 ARK 模型
```

TAU-2 业务工具（`get_user_details`、`cancel_reservation` 等）以原生 function tools 形式传给模型，由 TAU-2 环境实际执行；OpenClaw 原生工具（`memory_store`、`memory_recall` 等）由模型自由调用，不经过 TAU-2。

## 快速开始

```bash
# 1. 安装依赖
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e tau2-bench/

# 2. 启动 Gateway（另开终端，保持运行）
openclaw gateway

# 3. 快速验证（mock 域，1 个任务）
.venv/bin/python run_eval.py --domain mock --num-tasks 1 --use-gateway --force
```

## 环境要求

- Python 3.11
- Node.js ≥ 22
- OpenClaw 已安装，并在 `~/.openclaw/openclaw.json` 中完成以下配置：
  - 火山引擎 ARK 模型（`models.providers.ark`）
  - 三个域对应的 Agent（`agents.list`）：`tau2-airline`、`tau2-retail`、`tau2-telecom`
  - memory-lancedb 插件启用（`plugins.entries.memory-lancedb`）

`~/.openclaw/openclaw.json` 中 agents 配置示例：

```json
"agents": {
  "list": [
    { "id": "tau2-airline", "name": "TAU-2 Airline", "model": "ark/<your-endpoint-id>" },
    { "id": "tau2-retail",  "name": "TAU-2 Retail",  "model": "ark/<your-endpoint-id>" },
    { "id": "tau2-telecom", "name": "TAU-2 Telecom", "model": "ark/<your-endpoint-id>" }
  ]
}
```

## 使用流程

推荐先在 train split 上积累记忆，再跑 test 评测。

### 第一步：训练（积累 Memory）

```bash
# 三个域并行训练（各开一个终端）
.venv/bin/python run_train.py --domain airline --use-gateway
.venv/bin/python run_train.py --domain retail  --use-gateway
.venv/bin/python run_train.py --domain telecom --use-gateway

# 调试：只跑前 5 个任务
.venv/bin/python run_train.py --domain airline --use-gateway --num-tasks 5

# 中断后续跑（自动跳过已完成任务）
.venv/bin/python run_train.py --domain airline --use-gateway

# 从头重跑（清除 checkpoint 和 session）
.venv/bin/python run_train.py --domain airline --use-gateway --force
```

训练结束后记忆自动保存在各 Agent 的 workspace，评测时自动加载。

### 第二步：评测

```bash
.venv/bin/python run_eval.py --domain airline --use-gateway --force
.venv/bin/python run_eval.py --domain retail  --use-gateway --force
.venv/bin/python run_eval.py --domain telecom --use-gateway --force
```

### 查看结果

结果保存在 `tau2-bench/data/tau2/simulations/openclaw_<domain>_test.json`。

```bash
# TAU-2 自带 Web UI
tau2 view

# 命令行快速查看
.venv/bin/python - <<'EOF'
import json
with open("tau2-bench/data/tau2/simulations/openclaw_airline_test.json") as f:
    d = json.load(f)
for s in d["simulations"]:
    print(f"{s['task_id']}: reward={s['reward_info']['reward']:.2f}  ({s['termination_reason']})")
EOF
```

## Memory（LanceDB）

OpenClaw 通过 `memory-lancedb` 插件为每个 Agent 提供独立的长期记忆，三个域互相隔离。

### 工作流程

1. **训练阶段**：每个任务完成后，Agent 收到评测反馈（得分、失败原因、正确操作序列），主动调用 `memory_store` 将经验写入 LanceDB
2. **评测阶段**：每次对话前，`before_agent_start` hook 自动向量检索相关记忆（top-3，相似度阈值 0.3），以 `<relevant-memories>` 注入上下文

### 存储路径

| 域 | Agent ID | 记忆路径 |
|----|----------|---------|
| airline | `tau2-airline` | `~/.openclaw/workspace/tau2-airline/memory/lancedb` |
| retail  | `tau2-retail`  | `~/.openclaw/workspace/tau2-retail/memory/lancedb`  |
| telecom | `tau2-telecom` | `~/.openclaw/workspace/tau2-telecom/memory/lancedb` |

### 可用工具

| 工具 | 说明 |
|------|------|
| `memory_store` | 保存新记忆（text、importance、category） |
| `memory_recall` | 向量检索相关记忆 |
| `memory_forget` | 按 ID 或语义搜索删除记忆 |

### 插件配置（`~/.openclaw/openclaw.json`）

```json
"memory-lancedb": {
  "enabled": true,
  "config": {
    "dbPath": "memory/lancedb",
    "embedding": {
      "provider": "doubao",
      "model": "<your-ark-embedding-endpoint-id>",
      "url": "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal",
      "apiKey": "${VOLCANO_ENGINE_API_KEY}"
    },
    "autoCapture": false,
    "autoRecall": true
  }
}
```

> `model` 填写火山引擎 ARK 上部署的 embedding endpoint ID。**必须使用向量范数接近 1.0 的模型**（如豆包多模态 embedding），否则 L2 距离计算失效，召回结果为空。

### 查看记忆

```bash
# 汇总各域记忆数量
.venv/bin/python show_memory.py

# 查看指定域的记忆详情
.venv/bin/python show_memory.py --domain airline
.venv/bin/python show_memory.py --domain retail --search "refund"
.venv/bin/python show_memory.py --domain telecom --limit 20
```

## 统计 Token 消耗

```bash
# 汇总 train 各域 token 消耗
.venv/bin/python count_train_tokens.py

# 按 session 粒度统计
.venv/bin/python count_tokens.py
```

## 参考

### 参数说明

| 参数 | 默认值 | 适用脚本 | 说明 |
|------|--------|---------|------|
| `--domain` | `mock` | 两者 | 评测域：`mock` / `airline` / `retail` / `telecom` |
| `--use-gateway` | 否 | 两者 | 走 Gateway HTTP API（需先启动 `openclaw gateway`） |
| `--force` | 否 | 两者 | 忽略 checkpoint，从头跑 |
| `--num-tasks` | 全部 | 两者 | 限制任务数量，调试用 |
| `--max-steps` | `100`/`30` | 两者 | 每个任务最大对话轮数 |
| `--task-split` | `test` | eval | 数据集划分：`train` / `test` / `base` |
| `--num-trials` | `1` | eval | 每个任务跑几次（论文标准为 4） |
| `--task-ids` | — | eval | 逗号分隔的指定任务 ID |

### 数据集规模

| 域 | train | test | base（全量）| 说明 |
|----|-------|------|------------|------|
| `mock` | — | — | 9 | 任务管理，用于快速验证 |
| `airline` | 30 | 20 | 50 | 航班预订、改签、取消 |
| `retail` | 74 | 40 | 114 | 电商订单、退货、换货 |
| `telecom` | 74 | 40 | 114 | 电信客服，含双控场景 |

### 任务终止条件

| 原因 | 说明 |
|------|------|
| `USER_STOP` | 用户模拟器判断任务完成或无法继续 |
| `MAX_STEPS` | 超过 `--max-steps` |
| `TOO_MANY_ERRORS` | 连续工具调用错误超过 10 次 |

### 行业基准（SOTA 参考）

tau2-bench 自带结果，4 trials，Pass^1 指标：

| 模型 | Airline | Retail | Telecom |
|------|---------|--------|---------|
| GPT-4.1 | 0.78 | 0.90 | 0.50 |
| Claude 3.7 Sonnet | 0.68 | 0.93 | 0.70 |
| GPT-4.1-mini | 0.68 | 0.85 | 0.68 |
| o4-mini | 0.76 | 0.91 | 0.60 |

### 项目结构

```
openclaw-tau2/
├── openclaw_agent.py        # OpenClaw → TAU-2 适配器（核心）
├── run_eval.py              # 评测脚本
├── run_train.py             # 训练脚本（Memory 积累）
├── show_memory.py           # 查看各域 LanceDB 记忆内容
├── count_tokens.py          # 按 session 统计 Token 消耗
├── count_train_tokens.py    # 汇总 train 各域 Token 消耗
├── tau2-bench/              # TAU-2 框架（sierra-research/tau2-bench）
│   ├── src/tau2/            # 框架源码
│   └── data/tau2/           # 数据（任务集、数据库、策略文档）
└── .venv/                   # Python 3.11 虚拟环境
```

## 已应用的补丁

以下问题在本项目中已修复，上游 PR 合并后可移除对应补丁。

### 1. clientTools 丢失（OpenClaw 核心）

**现象**：TAU-2 业务工具无法被调用，`/v1/responses` 传入的 `clientTools` 在内部调用链中丢失。

**原因**：`runEmbeddedPiAgent` 调用 `runEmbeddedAttempt` 时漏传了 `clientTools` 参数（[PR #49695](https://github.com/openclaw/openclaw/pull/49695)）。

**修复**：在 `/opt/homebrew/lib/node_modules/openclaw/dist/auth-profiles-*.js` 中找到 `runEmbeddedAttempt` 的调用处，补上：

```js
clientTools: params.clientTools,
```

### 2. memory-lancedb 插件支持豆包 Embedding

**来源**：[jiulingyun/openclaw-cn PR #448](https://github.com/jiulingyun/openclaw-cn/pull/448)（memory-lancedb: add Doubao embedding provider）

原插件仅支持 OpenAI embedding，本项目应用了此 PR，为 `memory-lancedb` 插件增加了 `doubao` provider 支持（`DoubaoEmbeddings` 类、豆包多模态 embedding endpoint）。

### 3. autoRecall 无效（memory-lancedb 插件）

**现象**：记忆存储正常，但召回始终返回空结果。

**原因**：部分火山引擎 embedding 模型输出向量范数约为 128（非归一化），L2 距离极大，`score = 1/(1+distance)` 接近 0，低于召回阈值。

**修复**：换用向量范数接近 1.0 的豆包多模态 embedding 模型。

### 4. Embedding 序列长度超限（memory-lancedb 插件）

**现象**：`max_sequence_length exceeded (5652 > 4096)`，在注入了领域策略的长 prompt 上触发。

**修复**：`before_agent_start` hook 中将 prompt 截取前 2000 字符再向量化。

### 5. Table already exists 并发竞争（memory-lancedb 插件）

**现象**：三个域并发训练时报 `Table 'memories' already exists`。

**原因**：多进程同时初始化 LanceDB，`tableNames` 检查与 `createTable` 之间存在竞态。

**修复**：捕获 `already exists` 错误，fallback 到 `openTable`。

### 6. 工具结果触发记忆注入（memory-lancedb 插件）

**现象**：工具调用返回的 JSON 结果也被注入 `<relevant-memories>`，导致上下文污染。

**原因**：`before_agent_start` hook 在每次 `/v1/responses` 调用时触发，包括工具结果回传。

**修复**：prompt 以 `{` 或 `[{` 开头时跳过召回。

### 7. 各 Agent 记忆未隔离（memory-lancedb 插件）

**现象**：三个域的 Agent 共享同一个 `~/memory/lancedb`，记忆相互污染。

**原因**：`api.resolvePath("memory/lancedb")` 以 gateway 启动目录（`~`）为基准，与 agent workspace 无关。

**修复**：工具和 hook 改为 factory 模式，通过 `ctx.workspaceDir` 构建各 agent 独立的 DB 路径。

## 常见问题

**`VOLCANO_ENGINE_API_KEY` 未找到**：脚本从 `~/.openclaw/openclaw.json` 自动读取，也可手动设置：
```bash
export VOLCANO_ENGINE_API_KEY=your-key
```

**超时错误（TimeoutError）**：检查 Gateway 是否已启动（`openclaw gateway`）。默认超时 600 秒，如仍超时可修改 `openclaw_agent.py` 中 `_call_responses_api` 的 `timeout` 参数。

**`MAX_STEPS` 提前终止**：复杂任务需要更多轮次，加大 `--max-steps 60`。

**task set has changed 报错**：本次与上次任务集不同（如改了 `--num-tasks`），加 `--force` 重新跑。
