"""Train OpenClaw memory on TAU-2 train split.

For each train task:
  1. Run blind  (agent doesn't know the answer)
  2. Send evaluation feedback into the same session (result + score breakdown + ground truth)
  3. Re-run the task in the same session — agent has full context + feedback, no re-seeding
     (tau2 creates a fresh DB environment for Run 2; only Run 2 messages are evaluated)

After training, evaluate with:
    python run_eval.py --domain <domain> --task-split test

Usage:
    python run_train.py --domain retail
    python run_train.py --domain airline --num-tasks 5
"""

import argparse
import glob
import json
import os
import pathlib
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tau2-bench", "src"))

from run_eval import get_ark_env, setup_env

SESSIONS_DIR = pathlib.Path.home() / ".openclaw" / "agents" / "main" / "sessions"
MEMORY_DIR = pathlib.Path.home() / ".openclaw" / "workspace" / "memory"


def clear_sessions():
    files = list(SESSIONS_DIR.glob("*.jsonl"))
    for f in files:
        f.unlink()
    print(f"[Train] Cleared {len(files)} session files from {SESSIONS_DIR}")

    mem_files = list(MEMORY_DIR.glob("*")) if MEMORY_DIR.exists() else []
    for f in mem_files:
        if f.is_file():
            f.unlink()
    print(f"[Train] Cleared {len(mem_files)} memory files from {MEMORY_DIR}")


def count_session_tokens() -> dict:
    total_input = 0
    total_output = 0
    total_tokens = 0
    for path in SESSIONS_DIR.glob("*.jsonl"):
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


def make_agent_class(session_id: str, use_gateway: bool = False):
    """OpenClawAgent pinned to a specific session_id (seeds on first call)."""
    from openclaw_agent import OpenClawAgent, OpenClawAgentState

    class PinnedSessionAgent(OpenClawAgent):
        def __init__(self, tools, domain_policy):
            super().__init__(tools=tools, domain_policy=domain_policy, use_gateway=use_gateway)

        def get_init_state(self, message_history=None):
            from openclaw_agent import _call_openclaw
            try:
                seed_msg = f"[SYSTEM CONTEXT]\n{self._system_prompt}"
                _call_openclaw(session_id, seed_msg, use_gateway=self._use_gateway)
                print(f"  [OpenClaw] session: {session_id}")
            except Exception as e:
                print(f"  [OpenClaw] warn: seed failed: {e}")
            return OpenClawAgentState(session_id=session_id)

    return PinnedSessionAgent


def make_reuse_agent_class(session_id: str, use_gateway: bool = False):
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


def main():
    parser = argparse.ArgumentParser(description="Train OpenClaw memory on TAU-2 train split")
    parser.add_argument("--domain", default="mock", choices=["mock", "airline", "retail", "telecom"])
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--use-gateway", action="store_true")
    parser.add_argument("--save-to", type=str, default=None)
    args = parser.parse_args()

    from loguru import logger
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="WARNING")

    default_model = setup_env()
    _, _, model_id = get_ark_env()
    user_llm = default_model
    use_gateway = args.use_gateway

    from openclaw_agent import _call_openclaw
    from tau2.registry import registry
    from tau2.run import get_tasks

    clear_sessions()

    split = "train" if args.domain != "mock" else "base"
    tasks = get_tasks(args.domain, task_split_name=split, num_tasks=args.num_tasks)
    print(f"[Train] domain={args.domain}  split={split}  tasks={len(tasks)}")
    print()

    results = []
    for i, task in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] Task: {task.id}")

        session_id = f"tau2-{uuid.uuid4().hex[:12]}"

        # --- Run 1: blind ---
        print("  Run 1 (blind)...")
        registry._agents.pop("openclaw_run1", None)
        registry.register_agent(make_agent_class(session_id, use_gateway), "openclaw_run1")
        sim1, reward1 = run_one_task(args.domain, task, "openclaw_run1", user_llm, model_id, args.max_steps)
        success1 = reward1 >= 1.0
        print(f"  Run 1: {'SUCCESS' if success1 else 'FAIL'}  (reward={reward1:.2f})")

        # Skip Run 2 if already succeeded, but still send success feedback
        if success1:
            parts = [f"The task has ended. Your attempt succeeded (reward={reward1:.2f})."]
            feedback_str = format_reward_feedback(sim1.reward_info)
            if feedback_str:
                parts.append(feedback_str)
            gt_str = format_ground_truth(task)
            if gt_str:
                parts.append(gt_str)
            feedback_message = "\n".join(parts)
            print(f"  [feedback → session {session_id}]\n    " + feedback_message.replace("\n", "\n    "))
            _call_openclaw(session_id, f"[EVALUATION RESULT]\n{feedback_message}", use_gateway=use_gateway)
            _call_openclaw(session_id, "Remember what's said, keep existing memory.", use_gateway=use_gateway)
            print("  Run 2: skipped (succeeded)")
            print()
            results.append({
                "task_id": task.id,
                "run1_reward": reward1,
                "run1_success": success1,
                "run2_reward": None,
                "run2_success": None,
            })
            continue

        # --- Send feedback into the same session ---
        result_str = f"The task has ended. Your attempt failed (reward={reward1:.2f})."
        parts = [result_str]
        feedback_str = format_reward_feedback(sim1.reward_info)
        if feedback_str:
            parts.append(feedback_str)
        gt_str = format_ground_truth(task)
        if gt_str:
            parts.append(gt_str)
        parts.append("A new customer session will start shortly.")
        feedback_message = "\n".join(parts)

        print(f"  [feedback → session {session_id}]\n    " + feedback_message.replace("\n", "\n    "))
        _call_openclaw(session_id, f"[EVALUATION RESULT]\n{feedback_message}", use_gateway=use_gateway)

        # Reset agent to wait mode before Run 2 starts
        _call_openclaw(session_id, "[SYSTEM] A new customer session is starting. Wait for the customer to speak first.", use_gateway=use_gateway)

        # --- Run 2: reuse same session ---
        print("  Run 2 (same session, with feedback)...")
        registry._agents.pop("openclaw_run2", None)
        registry.register_agent(make_reuse_agent_class(session_id, use_gateway), "openclaw_run2")
        sim2, reward2 = run_one_task(args.domain, task, "openclaw_run2", user_llm, model_id, args.max_steps)
        success2 = reward2 >= 1.0
        print(f"  Run 2: {'SUCCESS' if success2 else 'FAIL'}  (reward={reward2:.2f})")

        # Send Run 2 result feedback
        run2_result = f"The task has ended. Your attempt {'succeeded' if success2 else 'failed'} (reward={reward2:.2f})."
        run2_parts = [run2_result]
        run2_feedback = format_reward_feedback(sim2.reward_info)
        if run2_feedback:
            run2_parts.append(run2_feedback)
        run2_gt = format_ground_truth(task)
        if run2_gt:
            run2_parts.append(run2_gt)
        _call_openclaw(session_id, f"[EVALUATION RESULT]\n" + "\n".join(run2_parts), use_gateway=use_gateway)
        _call_openclaw(session_id, "Remember what's said, keep existing memory.", use_gateway=use_gateway)
        print()

        results.append({
            "task_id": task.id,
            "run1_reward": reward1,
            "run1_success": success1,
            "run2_reward": reward2,
            "run2_success": success2,
        })

    # --- Summary ---
    n = len(results)
    r2_results = [r for r in results if r["run2_reward"] is not None]
    r1_avg = sum(r["run1_reward"] for r in results) / n
    r1_pass = sum(r["run1_success"] for r in results) / n

    print("=" * 50)
    print(f"[Train Summary]  {n} tasks  domain={args.domain}  split={split}")
    print(f"  Run 1 (blind):         avg_reward={r1_avg:.3f}  pass_rate={r1_pass:.3f}")
    if r2_results:
        r2_avg = sum(r["run2_reward"] for r in r2_results) / len(r2_results)
        r2_pass = sum(r["run2_success"] for r in r2_results) / len(r2_results)
        print(f"  Run 2 (w/ feedback):   avg_reward={r2_avg:.3f}  pass_rate={r2_pass:.3f}  ({len(r2_results)}/{n} tasks)")
    print()
    token_stats = count_session_tokens()
    print(f"  Token usage (accumulated):  input={token_stats['input']:,}  output={token_stats['output']:,}  total={token_stats['totalTokens']:,}")
    print()
    print("Memory accumulated. Now run:")
    print(f"  python run_eval.py --domain {args.domain} --task-split test")

    save_to = args.save_to or f"train_{args.domain}_{split}"
    out_dir = pathlib.Path(__file__).parent / "tau2-bench" / "data" / "tau2" / "simulations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{save_to}.json"
    with open(out_path, "w") as f:
        json.dump(
            {"domain": args.domain, "split": split, "tasks": results},
            f, indent=2, ensure_ascii=False,
        )
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
