# openclaw-tau2

使用 [OpenClaw](https://github.com/openclaw/openclaw) 作为被测 Agent，在 [TAU-2 (τ²-bench)](https://github.com/sierra-research/tau2-bench) 基准上进行评测。

## 项目简介

TAU-2 是一个开源的 AI Agent 评测框架，模拟客服场景（航空、零售、电信），通过三方对话评估 Agent 能力：**被测 Agent**、**用户模拟器**（LLM）、**有状态的领域环境**。

本项目将 OpenClaw 接入 TAU-2 的 `LocalAgent` 接口，通过 `/v1/responses` HTTP API 使用原生 function tools 调用业务系统工具，同时保留 OpenClaw 自身的 memory 等原生能力。

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

## 环境准备

### 依赖

- Python 3.11
- Node.js ≥ 22
- OpenClaw 已安装并配置（`~/.openclaw/openclaw.json`）

### 安装

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e tau2-bench/
```

## 使用方法

### 评测

```bash
# 启动 Gateway（另开终端保持运行）
openclaw gateway

# 快速测试：mock 域，1 个任务
python run_eval.py --domain mock --num-tasks 1 --use-gateway --force

# 标准评测：test split（默认）
python run_eval.py --domain airline --use-gateway
python run_eval.py --domain retail  --use-gateway
python run_eval.py --domain telecom --use-gateway

# 覆盖已有结果重新跑
python run_eval.py --domain airline --use-gateway --force

# 续跑（不加 --force，自动跳过已完成任务）
python run_eval.py --domain airline --use-gateway
```

### 训练（Memory 积累）

```bash
# 在 train split 上跑，让 OpenClaw 积累 memory
python run_train.py --domain airline --use-gateway
python run_train.py --domain airline --use-gateway --num-tasks 5  # 调试用

# 训练完后跑 test 评测
python run_eval.py --domain airline --use-gateway --force
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--domain` | `mock` | 评测域：`mock` / `airline` / `retail` / `telecom` |
| `--task-split` | `test` | 数据集划分：`train` / `test` / `base` |
| `--num-tasks` | 全部 | 限制任务数量，调试用 |
| `--num-trials` | `1` | 每个任务跑几次（论文标准为 4） |
| `--task-ids` | — | 逗号分隔的指定任务 ID |
| `--max-steps` | `30` | 每个任务最大对话轮数 |
| `--use-gateway` | 否 | 走 Gateway HTTP API（需先启动 `openclaw gateway`） |
| `--force` | 否 | 删除已有结果文件，从头跑 |

## 数据集

| 域 | train | test | base（全量）| 说明 |
|----|-------|------|------------|------|
| `mock` | — | — | 9 | 任务管理，用于快速验证 |
| `airline` | 30 | 20 | 50 | 航班预订、改签、取消 |
| `retail` | 74 | 40 | 114 | 电商订单、退货、换货 |
| `telecom` | 74 | 40 | 114 | 电信客服，含双控场景 |

## 查看结果

结果保存在 `tau2-bench/data/tau2/simulations/<save-to>.json`（默认 `openclaw_<domain>_test.json`）。

```bash
# TAU-2 自带 Web UI
tau2 view

# 命令行快速查看
python - <<'EOF'
import json
with open("tau2-bench/data/tau2/simulations/openclaw_airline_test.json") as f:
    d = json.load(f)
for s in d["simulations"]:
    print(f"{s['task_id']}: reward={s['reward_info']['reward']:.2f}  ({s['termination_reason']})")
EOF
```

### 统计 Token 消耗

```bash
# 统计所有 session
python count_tokens.py

# 统计指定 session
python count_tokens.py tau2-abc123 tau2-def456
```

## 行业基准（SOTA 参考）

以下为 tau2-bench 自带结果，4 trials，Pass^1 指标：

| 模型 | Airline | Retail | Telecom |
|------|---------|--------|---------|
| GPT-4.1 | 0.78 | 0.90 | 0.50 |
| Claude 3.7 Sonnet | 0.68 | 0.93 | 0.70 |
| GPT-4.1-mini | 0.68 | 0.85 | 0.68 |
| o4-mini | 0.76 | 0.91 | 0.60 |

## 工作原理

### 调用流程

```
OpenClawAgent.get_init_state()
  └─ 创建 session（tau2-<uuid>），发送 [SYSTEM CONTEXT] 注入领域策略

每轮对话：
  TAU-2 Orchestrator
    → OpenClawAgent.generate_next_message(UserMessage / ToolMessage)
        → 调用 POST /v1/responses，传入业务工具定义（native function tools）
        → 解析响应：
            function_call  →  AssistantMessage(tool_calls=[...])  → TAU-2 执行工具
            message        →  AssistantMessage(content="...")     → 发给用户模拟器
```

### 工具调用机制

TAU-2 业务工具通过 `/v1/responses` 的 `tools` 字段以原生 function tools 形式传给模型，模型直接输出结构化的 `function_call`，不需要文本格式解析。

**两类工具**：
- **OpenClaw 原生工具**（memory、文件读写等）：模型自由调用，不经过 TAU-2
- **TAU-2 业务工具**（`get_user_details`、`cancel_reservation` 等）：通过 client-side function tools 机制，由 TAU-2 环境实际执行

### 结束条件

| 原因 | 说明 |
|------|------|
| `USER_STOP` | 用户模拟器判断任务完成或无法继续 |
| `MAX_STEPS` | 超过 `--max-steps`（默认 30） |
| `TOO_MANY_ERRORS` | 连续工具调用错误超过 10 次 |

## 项目结构

```
openclaw-tau2/
├── openclaw_agent.py   # OpenClaw → TAU-2 适配器（核心）
├── run_eval.py         # 评测脚本
├── run_train.py        # 训练脚本（Memory 积累）
├── count_tokens.py     # Token 消耗统计工具
├── tau2-bench/         # TAU-2 框架（sierra-research/tau2-bench）
│   ├── src/tau2/       # 框架源码
│   └── data/tau2/      # 数据（任务集、数据库、策略文档）
└── .venv/              # Python 3.11 虚拟环境
```

## 常见问题

**`VOLCANO_ENGINE_API_KEY` 未找到**：从 `~/.openclaw/openclaw.json` 自动读取，也可手动设置：
```bash
export VOLCANO_ENGINE_API_KEY=your-key
```

**超时错误（TimeoutError）**：Gateway 未启动，先运行 `openclaw gateway`。

**`MAX_STEPS` 提前终止**：增大步数 `--max-steps 60`，复杂任务（多个工具调用）需要更多步数。

**File already exists，询问 resume**：加 `--force` 重新跑，或不加 `--force` 续跑已中断的任务。

**task set has changed 报错**：上次和这次的任务集不同（比如改了 `--num-tasks` 或 `--task-split`），加 `--force` 重新跑。
