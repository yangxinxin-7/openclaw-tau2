"""Train OpenClaw memory on TAU-2 train split.

For each train task:
  1. Run blind  (agent doesn't know the answer)
  2. Send evaluation feedback into the same session (result + score breakdown + ground truth)
  3. Re-run the task in the same session — agent has full context + feedback, no re-seeding
     (tau2 creates a fresh DB environment for Run 2; only Run 2 messages are evaluated)

After training, evaluate with:
    python run_eval.py --domain <domain> --task-split test

Memory backends
---------------
lancedb (default):
    OpenClaw's built-in LanceDB plugin. Agent manually stores/recalls memories
    via memory_store/memory_recall tools in each session.

openviking:
    OpenViking context database. Each task is run --num-runs times (default 5).
    After every run the full conversation + evaluation result is committed to
    OpenViking, which automatically extracts structured memories (cases, patterns,
    events, …) via LLM. Requires OpenViking to be installed and configured.

Usage:
    python run_train.py --domain retail
    python run_train.py --domain airline --num-tasks 5
    python run_train.py --domain airline --memory-backend openviking --num-runs 5
"""

import argparse
import glob
import json
import os
import pathlib
import sys
import uuid
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tau2-bench", "src"))

from run_eval import get_ark_env, setup_env

OPENCLAW_AGENTS_DIR = pathlib.Path.home() / ".openclaw" / "agents"
OPENCLAW_WORKSPACE_DIR = pathlib.Path.home() / ".openclaw" / "workspace"


def _agent_dirs(agent_id: str = None):
    """Return (sessions_dir, sessions_json, memory_dir) for the given agent."""
    aid = agent_id or "main"
    sessions_dir = OPENCLAW_AGENTS_DIR / aid / "sessions"
    sessions_json = sessions_dir / "sessions.json"
    if agent_id:
        memory_dir = OPENCLAW_WORKSPACE_DIR / agent_id / "memory"
    else:
        memory_dir = OPENCLAW_WORKSPACE_DIR / "memory"
    return sessions_dir, sessions_json, memory_dir


def clear_sessions(agent_id: str = None):
    sessions_dir, sessions_json, memory_dir = _agent_dirs(agent_id)

    files = list(sessions_dir.glob("*.jsonl")) if sessions_dir.exists() else []
    for f in files:
        f.unlink()
    print(f"[Train] Cleared {len(files)} session files from {sessions_dir}")

    if sessions_json.exists():
        sessions_json.unlink()
        print(f"[Train] Cleared sessions.json from {sessions_json}")

    mem_files = list(memory_dir.glob("*")) if memory_dir.exists() else []
    for f in mem_files:
        if f.is_file():
            f.unlink()
    print(f"[Train] Cleared {len(mem_files)} memory files from {memory_dir}")


def count_session_tokens(agent_id: str = None) -> dict:
    sessions_dir, _, _ = _agent_dirs(agent_id)
    total_input = 0
    total_output = 0
    total_tokens = 0
    for path in sessions_dir.glob("*.jsonl") if sessions_dir.exists() else []:
        with open(path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = entry.get("message", {})
                if msg.get("role") != "assistant":
                    continue
                usage = msg.get("usage", {})
                if not usage:
                    continue
                total_input += usage.get("input", 0)
                total_output += usage.get("output", 0)
                total_tokens += usage.get("totalTokens", 0)
    return {"input": total_input, "output": total_output, "totalTokens": total_tokens}


def format_ground_truth(task) -> str:
    ec = task.evaluation_criteria
    if not ec or not ec.actions:
        return ""
    lines = ["Correct action sequence:"]
    for i, action in enumerate(ec.actions, 1):
        lines.append(f"  {i}. {action.get_func_format()}")
    return "\n".join(lines)


def format_reward_feedback(reward_info) -> str:
    if reward_info is None:
        return ""
    lines = []

    if reward_info.reward_breakdown:
        breakdown = ", ".join(
            f"{k.value}: {v:.2f}" for k, v in reward_info.reward_breakdown.items()
        )
        lines.append(f"Score breakdown: {breakdown}")

    if reward_info.db_check is not None and not reward_info.db_check.db_match:
        lines.append("DB state: INCORRECT (your tool calls did not produce the expected database changes)")

    if reward_info.communicate_checks:
        failed = [c for c in reward_info.communicate_checks if not c.met]
        if failed:
            lines.append("Failed communication checks:")
            for c in failed:
                lines.append(f"  - Required to communicate: {c.info}")
                if c.justification:
                    lines.append(f"    Reason: {c.justification}")

    return "\n".join(lines)


def make_agent_class(session_id: str, use_gateway: bool = True, agent_id: str = None):
    """OpenClawAgent pinned to a specific session_id (seeds on first call)."""
    from openclaw_agent import OpenClawAgent, OpenClawAgentState

    class PinnedSessionAgent(OpenClawAgent):
        def __init__(self, tools, domain_policy):
            super().__init__(tools=tools, domain_policy=domain_policy, use_gateway=use_gateway, agent_id=agent_id)

        def get_init_state(self, message_history=None):
            print(f"  [OpenClaw] session: {session_id}")
            return OpenClawAgentState(session_id=session_id)

    return PinnedSessionAgent


def make_reuse_agent_class(session_id: str, use_gateway: bool = True):
    """OpenClawAgent that reuses an existing session without re-seeding."""
    from openclaw_agent import OpenClawAgent, OpenClawAgentState

    class ReuseSessionAgent(OpenClawAgent):
        def __init__(self, tools, domain_policy):
            super().__init__(tools=tools, domain_policy=domain_policy, use_gateway=use_gateway)

        def get_init_state(self, message_history=None):
            print(f"  [OpenClaw Train] reusing session: {session_id}")
            return OpenClawAgentState(session_id=session_id)

    return ReuseSessionAgent


def run_one_task(domain, task, agent_name, user_llm, model_id, max_steps):
    from tau2.run import run_task
    from tau2.evaluator.evaluator import EvaluationType

    sim = run_task(
        domain=domain,
        task=task,
        agent=agent_name,
        user="user_simulator",
        llm_agent=f"openclaw/{model_id}",
        llm_args_agent={},
        llm_user=user_llm,
        llm_args_user={},
        max_steps=max_steps,
        evaluation_type=EvaluationType.ALL,
    )
    reward = sim.reward_info.reward if sim.reward_info else 0.0
    return sim, reward


def commit_to_openviking(domain: str, task, run_idx: int, sim, reward: float, reward_info) -> None:
    """Submit a TAU-2 task run to OpenViking via `ov add-memory` (one-shot: create+add+commit)."""
    import subprocess

    messages = []
    task_id = getattr(task, "id", None) or "unknown"

    try:
        from tau2.run import get_environment_info
        env_info = get_environment_info(domain_name=domain, include_tool_info=False)
        domain_policy = env_info.policy
        from openclaw_agent import _build_system_prompt
        sys_prompt = _build_system_prompt(domain_policy)
        messages.append({"role": "user", "content": f"system:\n{sys_prompt}"})
    except Exception:
        pass

    tool_calls_by_id = {}
    for msg in sim.messages:
        role = getattr(msg, "role", None)
        if str(role) in ("user", "assistant"):
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content.strip():
                messages.append({"role": str(role), "content": f"{role}:\n{content}"})
            tcs = getattr(msg, "tool_calls", None)
            if tcs:
                for tc in tcs:
                    tc_id = getattr(tc, "id", "") or ""
                    tc_role = getattr(tc, "requestor", str(role)) or str(role)
                    tc_name = getattr(tc, "name", "") or ""
                    tc_args = getattr(tc, "arguments", {}) or {}
                    tool_calls_by_id[tc_id] = tc
                    messages.append({
                        "role": str(tc_role),
                        "content": "tool-call:\n"
                        + (f"call_id: {tc_id}\n" if tc_id else "")
                        + f"name: {tc_name}\n"
                        + "arguments: "
                        + json.dumps(tc_args, ensure_ascii=False),
                    })
        elif str(role) == "tool":
            tool_call_id = getattr(msg, "id", "") or ""
            requestor = getattr(msg, "requestor", "assistant") or "assistant"
            tc = tool_calls_by_id.get(tool_call_id)
            tool_name = getattr(tc, "name", "") if tc is not None else ""
            output = getattr(msg, "content", None)
            output_str = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
            error = getattr(msg, "error", False)
            messages.append({
                "role": str(requestor),
                "content": "tool-response:\n"
                + (f"call_id: {tool_call_id}\n" if tool_call_id else "")
                + (f"name: {tool_name}\n" if tool_name else "")
                + (f"error: {bool(error)}\n" if error else "")
                + f"output: {output_str}",
            })

    # Append evaluation result so OpenViking has outcome context for memory extraction
    success = reward >= 1.0
    result_lines = [f"[EVALUATION] success={success} failed={not success} reward={reward:.2f}"]
    feedback = format_reward_feedback(reward_info)
    if feedback:
        result_lines.append(feedback)
    if not success:
        gt_str = format_ground_truth(task)
        if gt_str:
            result_lines.append(gt_str)
    messages.append({"role": "user", "content": "\n".join(result_lines)})

    content_arg = json.dumps(messages, ensure_ascii=False)
    result = subprocess.run(
        ["ov", "add-memory", content_arg],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ov add-memory failed: {result.stderr.strip()}")


def main():
    parser = argparse.ArgumentParser(description="Train OpenClaw memory on TAU-2 train split")
    parser.add_argument("--domain", default="mock", choices=["mock", "airline", "retail", "telecom"])
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--no-gateway", action="store_false", dest="use_gateway", default=True)
    parser.add_argument("--save-to", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Ignore checkpoint and restart from scratch")
    parser.add_argument(
        "--memory-backend", choices=["lancedb", "openviking"], default="openviking",
        help="Memory backend: lancedb (default) or openviking",
    )
    parser.add_argument(
        "--num-runs", type=int, default=None,
        help="Runs per task for openviking backend (default: 5)",
    )
    args = parser.parse_args()

    from loguru import logger
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="WARNING")

    default_model = setup_env()
    _, _, model_id = get_ark_env()
    user_llm = default_model
    use_gateway = args.use_gateway

    from openclaw_agent import _call_openclaw, DOMAIN_AGENT_MAP
    from tau2.registry import registry
    from tau2.run import get_tasks

    use_openviking = args.memory_backend == "openviking"
    num_runs = args.num_runs or (1 if use_openviking else 1)
    agent_id = DOMAIN_AGENT_MAP.get(args.domain)
    split = "train" if args.domain != "mock" else "base"

    if use_openviking:
        print(f"[Train] OpenViking backend  runs_per_task={num_runs}")

    # --- Checkpoint setup ---
    save_to = args.save_to or f"train_{args.domain}_{split}"
    out_dir = pathlib.Path(__file__).parent / "tau2-bench" / "data" / "tau2" / "simulations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{save_to}.json"

    results = []
    completed_ids = set()
    if not args.force and out_path.exists():
        with open(out_path) as f:
            checkpoint = json.load(f)
        results = checkpoint.get("tasks", [])
        completed_ids = {r["task_id"] for r in results}
        print(f"[Train] Resuming from checkpoint: {len(completed_ids)} tasks already done")
    else:
        clear_sessions(agent_id)

    tasks = get_tasks(args.domain, task_split_name=split, num_tasks=args.num_tasks)
    pending = [t for t in tasks if t.id not in completed_ids]
    print(f"[Train] domain={args.domain}  split={split}  total={len(tasks)}  pending={len(pending)}")
    print()

    for i, task in enumerate(pending):
        print(f"[{i+1}/{len(pending)}] Task: {task.id}")

        try:
            if use_openviking:
                # --- OpenViking mode: repeat each task num_runs times ---
                run_rewards = []
                for run_idx in range(1, num_runs + 1):
                    session_id = f"tau2-{uuid.uuid4().hex[:12]}"
                    registry._agents.pop("openclaw_run", None)
                    registry.register_agent(make_agent_class(session_id, use_gateway, agent_id), "openclaw_run")
                    sim, reward = run_one_task(args.domain, task, "openclaw_run", user_llm, model_id, args.max_steps)
                    run_rewards.append(reward)
                    print(f"  Run {run_idx}/{num_runs}: {'SUCCESS' if reward >= 1.0 else 'FAIL'}  (reward={reward:.2f})")

                    commit_to_openviking(args.domain, task, run_idx, sim, reward, sim.reward_info)
                    print(f"    → OpenViking commit done")

                avg_reward = sum(run_rewards) / len(run_rewards)
                results.append({
                    "task_id": task.id,
                    "run_rewards": run_rewards,
                    "avg_reward": avg_reward,
                    "pass_rate": sum(r >= 1.0 for r in run_rewards) / len(run_rewards),
                })
                print(f"  avg_reward={avg_reward:.2f}  pass_rate={results[-1]['pass_rate']:.2f}")
            else:
                # --- LanceDB mode (default): single run + feedback to OpenClaw session ---
                session_id = f"tau2-{uuid.uuid4().hex[:12]}"
                tokens_before = count_session_tokens(agent_id)
                registry._agents.pop("openclaw_run1", None)
                registry.register_agent(make_agent_class(session_id, use_gateway, agent_id), "openclaw_run1")
                sim1, reward1 = run_one_task(args.domain, task, "openclaw_run1", user_llm, model_id, args.max_steps)
                success1 = reward1 >= 1.0
                print(f"  Run 1: {'SUCCESS' if success1 else 'FAIL'}  (reward={reward1:.2f})")

                # Send feedback into the same session
                result_str = f"The task has ended. Your attempt {'succeeded' if success1 else 'failed'} (reward={reward1:.2f})."
                parts = [result_str]
                feedback_str = format_reward_feedback(sim1.reward_info)
                if feedback_str:
                    parts.append(feedback_str)
                gt_str = format_ground_truth(task)
                if gt_str:
                    parts.append(gt_str)
                feedback_message = "\n".join(parts)

                now = datetime.now()
                remember_str = (
                    f"[REMEMBER] The time is {now.strftime('%Y-%m-%d %H:%M:%S')}. "
                    f"Use your memory tools to save what you think is important: "
                    f"memory_store to save a new memory, "
                    f"memory_recall to search existing memories, "
                    f"memory_forget to remove outdated ones."
                )
                full_message = f"[EVALUATION RESULT]\n{feedback_message}\n\n\n{remember_str}"
                print(f"  [feedback → session {session_id}]\n    " + feedback_message.replace("\n", "\n    "))
                _call_openclaw(session_id, full_message, use_gateway=use_gateway, agent_id=agent_id)
                print()

                tokens_after = count_session_tokens(agent_id)
                task_tokens = {
                    "input": tokens_after["input"] - tokens_before["input"],
                    "output": tokens_after["output"] - tokens_before["output"],
                    "totalTokens": tokens_after["totalTokens"] - tokens_before["totalTokens"],
                }
                results.append({
                    "task_id": task.id,
                    "run1_reward": reward1,
                    "run1_success": success1,
                    "tokens": task_tokens,
                })
        except Exception as e:
            print(f"  [ERROR] Task {task.id} failed, skipping: {e}")

        # Save checkpoint after every task
        with open(out_path, "w") as f:
            json.dump({"domain": args.domain, "split": split, "tasks": results}, f, indent=2, ensure_ascii=False)

    # --- Summary ---
    n = len(results)
    if use_openviking:
        avg_rewards = [r.get("avg_reward", 0) for r in results]
        pass_rates = [r.get("pass_rate", 0) for r in results]
        print("=" * 50)
        print(f"[Train Summary]  {n} tasks  domain={args.domain}  split={split}  backend=openviking  runs_per_task={num_runs}")
        print(f"  avg_reward={sum(avg_rewards)/n:.3f}  pass_rate={sum(pass_rates)/n:.3f}")
    else:
        r1_avg = sum(r["run1_reward"] for r in results) / n
        r1_pass = sum(r["run1_success"] for r in results) / n
        print("=" * 50)
        print(f"[Train Summary]  {n} tasks  domain={args.domain}  split={split}  backend=lancedb")
        print(f"  avg_reward={r1_avg:.3f}  pass_rate={r1_pass:.3f}")
        print()
        token_stats = count_session_tokens(agent_id)
        print(f"  Token usage (accumulated):  input={token_stats['input']:,}  output={token_stats['output']:,}  total={token_stats['totalTokens']:,}")
    print()
    print("Memory accumulated. Now run:")
    print(f"  python run_eval.py --domain {args.domain} --task-split test")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
