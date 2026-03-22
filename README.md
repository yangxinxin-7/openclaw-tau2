# TAU-2 × OpenClaw 评测

使用 [OpenClaw](https://github.com/openclaw/openclaw) 作为被测 Agent，在 [TAU-2 (τ²-bench)](https://github.com/sierra-research/tau2-bench) 基准上进行评测。

## 项目简介

TAU-2 是一个开源的 AI Agent 评测框架，模拟客服场景（航空、零售、电信），通过三方对话评估 Agent 能力：**被测 Agent**、**用户模拟器**（LLM）、**有状态的领域环境**。

本项目将 OpenClaw 接入 TAU-2 的 `LocalAgent` 接口。每个任务分配一个独立的 OpenClaw session，TAU-2 领域工具调用从 OpenClaw 的文本输出中解析后交由 TAU-2 环境执行，OpenClaw 自身的原生工具（memory、文件读写等）则不受限制。

```
TAU-2 编排器
    │
    ├─ 用户模拟器（LLM）         ← 模拟客户
    └─ OpenClawAgent（本项目）   ← 被评测方
           │
           ├─ [本地模式] openclaw agent --local --json
           └─ [Gateway模式] POST http://127.0.0.1:18789/v1/responses
                  │
                  └─ 火山引擎 ARK 模型（ep-20260224172301-9hs95）
```

## 环境准备

### 依赖

- Python 3.11
- Node.js ≥ 22
- OpenClaw 已安装并配置（`~/.openclaw/openclaw.json`）

### 安装

```bash
# 创建虚拟环境
python3.11 -m venv .venv
source .venv/bin/activate

# 安装 tau2-bench
pip install -e tau2-bench/

# 验证 openclaw
openclaw --version
```

## 使用方法

```bash
source .venv/bin/activate

# 快速测试：mock 域，1 个任务
python run_eval.py --domain mock --num-tasks 1

# 开发调试：airline 训练集，1 trial
python run_eval.py --domain airline --task-split train --num-trials 1

# 正式评测（对齐论文标准）：base 全量，4 trials
python run_eval.py --domain airline --task-split base --num-trials 4
python run_eval.py --domain retail  --task-split base --num-trials 4
python run_eval.py --domain telecom --task-split base --num-trials 4

# 指定任务 ID
python run_eval.py --domain mock --task-ids create_task_1,update_task_1

# 覆盖已有结果（不询问 resume）
python run_eval.py --domain mock --num-tasks 1 --force

# 通过 Gateway 运行（session 在 dashboard 可见）
openclaw gateway   # 另开一个终端保持运行
python run_eval.py --domain mock --num-tasks 1 --use-gateway
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--domain` | `mock` | 评测域：`mock` / `airline` / `retail` / `telecom` |
| `--task-split` | 默认全量 | 数据集划分：`train` / `test` / `base` |
| `--num-tasks` | 全部 | 限制任务数量，调试用 |
| `--num-trials` | `1` | 每个任务跑几次（论文标准为 4） |
| `--task-ids` | — | 逗号分隔的指定任务 ID |
| `--user-llm` | 同 OpenClaw | 用户模拟器的 LiteLLM 模型字符串 |
| `--save-to` | `openclaw_<domain>[_<split>]` | 结果文件名 |
| `--max-steps` | `30` | 每个任务最大对话轮数 |
| `--use-gateway` | 否 | 走 Gateway HTTP API（需先启动 `openclaw gateway`） |
| `--force` | 否 | 覆盖已有结果文件，不询问 resume |

### PyCharm 配置

1. **解释器**：`Settings → Python Interpreter → Existing Environment` → 选择 `.venv/bin/python3.11`
2. **运行配置**：`Edit Configurations → Python`，填写：
   - Script：`/Users/bytedance/viking/tau-2/run_eval.py`
   - Parameters：`--domain mock --num-tasks 1 --force`
   - Working directory：`/Users/bytedance/viking/tau-2`
   - Environment variables：`VOLCANO_ENGINE_API_KEY=<key>;OPENAI_API_KEY=<key>;OPENAI_API_BASE=https://ark.cn-beijing.volces.com/api/v3`

## 查看结果

结果保存在 `tau2-bench/data/tau2/simulations/<save-to>.json`。

```bash
# TAU-2 自带 Web UI
source .venv/bin/activate
tau2 view

# 命令行快速查看
python - <<'EOF'
import json
with open("tau2-bench/data/tau2/simulations/openclaw_airline.json") as f:
    d = json.load(f)
for s in d["simulations"]:
    print(f"{s['task_id']}: reward={s['reward_info']['reward']:.2f}  ({s['termination_reason']})")
EOF
```

### 查看 OpenClaw 对话历史

```bash
# 列出所有 tau2 session 文件
ls ~/.openclaw/agents/main/sessions/tau2-*.jsonl

# 读取某个 session 的对话内容
python3 -c "
import json
with open('/Users/bytedance/.openclaw/agents/main/sessions/tau2-<id>.jsonl') as f:
    for line in f:
        msg = json.loads(line)
        t = msg.get('type', '')
        if t in ('user', 'assistant'):
            content = msg.get('content', '')
            if isinstance(content, list):
                content = ' '.join(c.get('text','') for c in content if isinstance(c,dict))
            print(f'[{t.upper()}] {content[:300]}')
            print()
"
```

**注意**：本地模式（`--local`）的 session 不在 dashboard 显示。使用 `--use-gateway` 时，session 以 `openresponses-user:tau2-xxx` 形式注册，在 dashboard 可见。

### 评测指标

- **Reward**：0.0–1.0，综合 DB 状态匹配 + 沟通得分
- **Pass^k**：k 次独立试验中至少 1 次成功的概率，论文标准指标

## 数据集说明

| 域 | train | test | base（全量）| 说明 |
|----|-------|------|------------|------|
| `mock` | — | — | 9 | 任务管理（创建/更新任务），用于快速验证 |
| `airline` | 30 | 20 | 50 | 航班预订、改签、取消 |
| `retail` | 74 | 40 | 114 | 电商订单、退货、换货 |
| `telecom` | 74 | 40 | 114 | 电信客服，含双控场景（agent + 用户各自操作） |

**推荐流程**：
1. 用 `train` split 开发调试
2. 用 `test` split 验证（避免过拟合）
3. 最终用 `base` + 4 trials 对齐论文标准

**重复运行说明**：每次运行都从磁盘重新加载干净的数据库，工具调用只修改内存，不写回文件，重复跑不会污染数据。

## 行业基准（SOTA 参考）

以下为仓库内自带结果，4 trials，Pass^1 指标：

| 模型 | Airline | Retail | Telecom |
|------|---------|--------|---------|
| GPT-4.1 | 0.78 | 0.90 | 0.50 |
| Claude 3.7 Sonnet | 0.68 | 0.93 | 0.70 |
| GPT-4.1-mini | 0.68 | 0.85 | 0.68 |
| o4-mini | 0.76 | 0.91 | 0.60 |

> Telecom 是最难的域（双控模式，agent 和用户都需要执行操作），顶级模型标准模式下也只有 0.50–0.70。

## 工作原理

### 调用流程

```
OpenClawAgent.get_init_state()
  └─ 创建 session（tau2-<uuid>），发送 [SYSTEM CONTEXT] 注入领域策略和工具描述

每轮对话：
  TAU-2 Orchestrator
    → OpenClawAgent.generate_next_message(UserMessage / ToolMessage)
        → 将消息转为文本，调用 OpenClaw
        → 解析响应：
            TOOL_CALL: {...}  →  AssistantMessage(tool_calls=[...])  → TAU-2 执行工具
            普通文本          →  AssistantMessage(content="...")      → 发给用户模拟器
```

### 两种调用模式

| | 本地模式（默认） | Gateway 模式（`--use-gateway`） |
|--|----------------|-------------------------------|
| 命令 | `openclaw agent --local --json` | `POST /v1/responses` HTTP API |
| 是否需要 gateway | 否 | 是（需先运行 `openclaw gateway`） |
| dashboard 可见 | 否 | 是（`openresponses-user:tau2-xxx`） |
| session 文件 | `~/.openclaw/agents/main/sessions/tau2-*.jsonl` | 同左 |

### 响应格式解析

```
本地模式：  {"payloads": [{"text": "..."}], "meta": {...}}
Gateway CLI：{"result": {"payloads": [...]}, "status": "ok"}
HTTP API：  {"output": [{"content": [{"type": "output_text", "text": "..."}]}]}
```
三种格式均已兼容处理。

### 工具调用机制

OpenClaw 有**两类工具**：

**1. OpenClaw 原生工具**（memory、文件读写、浏览器等）
- 自由使用，不经过我们的代码
- 当 agent 认为有重要信息时，会自动调用 memory 保存

**2. TAU-2 领域工具**（与业务系统交互）
- OpenClaw 输出 `TOOL_CALL:` 格式，由我们拦截并交给 TAU-2 执行
- 格式：`TOOL_CALL: {"name": "create_task", "arguments": {"user_id": "user_1", "title": "Meeting"}}`
- 关键约束：只传用户明确提到的参数，不推断可选字段（否则 DB hash 比对会失败）

### 结束条件

| 原因 | 触发方 | 说明 |
|------|--------|------|
| `USER_STOP` | 用户模拟器 | 回复中含 `###STOP###`，任务完成或无法继续 |
| `AGENT_STOP` | agent | 回复中含 `###STOP###`（当前未启用） |
| `MAX_STEPS` | 框架 | 超过 `--max-steps`（默认 30） |
| `TOO_MANY_ERRORS` | 框架 | 连续工具调用错误超过 10 次 |

### 评分逻辑

评测结束后，框架重放对话中所有工具调用，与黄金答案对比：
- **DB Check**：将 agent 实际调用的工具在新环境里重放，与黄金答案执行后的数据库状态做 hash 比对
- **Action Check**：检查必要工具是否被调用、参数是否匹配

## 项目结构

```
tau-2/
├── openclaw_agent.py   # OpenClaw → TAU-2 适配器（核心）
├── run_eval.py         # 评测运行脚本
├── tau2-bench/         # TAU-2 框架（来自 sierra-research/tau2-bench）
│   ├── src/tau2/       # 框架源码（LocalAgent 接口、各域实现、评估器）
│   └── data/tau2/      # 数据（任务集、数据库、策略文档、SOTA 结果）
└── .venv/              # Python 3.11 虚拟环境
```

## 常见问题

**`VOLCANO_ENGINE_API_KEY` 未找到**：密钥从 `~/.openclaw/openclaw.json` 自动读取，也可手动设置：
```bash
export VOLCANO_ENGINE_API_KEY=your-key
```

**`This model isn't mapped yet`**：LiteLLM 不认识豆包模型的定价，纯警告日志，不影响评测结果，忽略即可（结果中 Cost 显示 $nan 属正常）。

**File already exists，询问 resume**：加 `--force` 参数自动覆盖，或手动删除 `tau2-bench/data/tau2/simulations/<name>.json`。

**得分偏低**：工具调用基于文本格式解析，如模型未遵循 `TOOL_CALL:` 格式则回退为普通文本。可在 `openclaw_agent.py` 的 `build_system_context` 中增加示例来改善。

**Gateway 模式 session 在 dashboard 找不到**：dashboard 只显示通过消息渠道（Telegram 等）发起的对话。Gateway 模式下的 session 以 `openresponses-user:tau2-xxx` 注册，目前需要直接读 JSONL 文件查看，或等 OpenClaw 后续版本支持在 dashboard 显示此类 session。
