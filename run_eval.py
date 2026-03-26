"""
MVP: Run TAU-2 evaluation with OpenClaw agent.

Usage:
    python run_eval.py [--domain DOMAIN] [--num-tasks N] [--task-ids ID1,ID2]

Examples:
    python run_eval.py --domain mock --num-tasks 2
    python run_eval.py --domain airline --num-tasks 3
"""

import argparse
import json
import os
import sys

# Add tau2-bench to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tau2-bench", "src"))


def get_ark_env() -> tuple[str, str, str]:
    """Read Volcano Engine ARK config from openclaw.json."""
    cfg_path = os.path.expanduser("~/.openclaw/openclaw.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    api_key = cfg.get("env", {}).get("VOLCANO_ENGINE_API_KEY", "")
    base_url = (
        cfg.get("models", {})
        .get("providers", {})
        .get("ark", {})
        .get("baseUrl", "https://ark.cn-beijing.volces.com/api/v3")
    )
    primary = (
        cfg.get("agents", {}).get("defaults", {}).get("model", {}).get("primary", "")
    )
    model_id = primary.split("/", 1)[-1] if "/" in primary else primary
    return api_key, base_url, model_id


def setup_env():
    """Inject ARK credentials into environment for LiteLLM user simulator."""
    api_key, base_url, model_id = get_ark_env()
    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    # LiteLLM uses OpenAI-compatible format; ARK's OpenAI endpoint is /api/v3
    openai_base = base_url.replace("/api/compatible", "/api/v3")
    os.environ.setdefault("OPENAI_API_BASE", openai_base)
    return f"openai/{model_id}"


def parse_args():
    parser = argparse.ArgumentParser(description="Run TAU-2 eval with OpenClaw")
    parser.add_argument(
        "--domain",
        default="mock",
        choices=["mock", "airline", "retail", "telecom"],
        help="TAU-2 domain (default: mock)",
    )
    parser.add_argument("--task-split", type=str, default="test",
                        help="Task split: train / test / base (default: test)")
    parser.add_argument("--num-tasks", type=int, default=None, help="Limit number of tasks (default: all in split)")
    parser.add_argument("--num-trials", type=int, default=1, help="Trials per task (default: 1)")
    parser.add_argument("--task-ids", type=str, default=None, help="Comma-separated task IDs")
    parser.add_argument(
        "--user-llm",
        type=str,
        default=None,
        help="LLM for user simulator (LiteLLM model string). Defaults to same model as OpenClaw.",
    )
    parser.add_argument("--save-to", type=str, default=None, help="Results file name (default: openclaw_<domain>)")
    parser.add_argument("--max-steps", type=int, default=30, help="Max steps per task (default: 30)")
    parser.add_argument("--no-gateway", action="store_false", dest="use_gateway", default=True,
                        help="Disable gateway routing and run agent locally.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing results file instead of resuming.")
    parser.add_argument(
        "--memory-backend",
        choices=["lancedb", "openviking"],
        default="lancedb",
        help="Memory backend for eval. openviking prepends retrieved ov memories to the first user message.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    default_model = setup_env()
    user_llm = args.user_llm or default_model
    task_ids = [t.strip() for t in args.task_ids.split(",")] if args.task_ids else None
    split = args.task_split
    split_tag = f"_{split}" if split else ""
    save_to = args.save_to or f"openclaw_{args.domain}{split_tag}"

    # Delete existing file if --force
    if args.force:
        from tau2.utils.utils import DATA_DIR
        results_file = DATA_DIR / "simulations" / f"{save_to}.json"
        if results_file.exists():
            results_file.unlink()
            print(f"  Removed: {results_file}")

    print("[TAU-2 x OpenClaw MVP]")
    print(f"  Domain     : {args.domain}")
    print(f"  Split      : {split or 'default'}")
    print(f"  Num tasks  : {args.num_tasks or 'all'}")
    print(f"  Num trials : {args.num_trials}")
    print(f"  User LLM   : {user_llm}")
    print(f"  Memory     : {args.memory_backend}")
    if task_ids:
        print(f"  Task IDs   : {task_ids}")
    print()

    # Import after env setup
    from openclaw_agent import OpenClawAgent, DOMAIN_AGENT_MAP
    from tau2.registry import registry
    from tau2.run import run_domain
    from tau2.data_model.simulation import RunConfig

    # Register OpenClaw agent with per-domain agent_id for memory isolation
    use_gateway = args.use_gateway
    agent_id = DOMAIN_AGENT_MAP.get(args.domain)
    session_ids = []
    prepend_openviking_memory = args.memory_backend == "openviking"
    registry.register_agent(
        type("OpenClawAgentConfigured", (OpenClawAgent,), {
            "__init__": lambda self, tools, domain_policy: OpenClawAgent.__init__(
                self,
                tools=tools,
                domain_policy=domain_policy,
                use_gateway=use_gateway,
                session_collector=session_ids,
                agent_id=agent_id,
                prepend_openviking_memory=prepend_openviking_memory,
            )
        }),
        "openclaw",
    )

    _, _, model_id = get_ark_env()

    config = RunConfig(
        domain=args.domain,
        agent="openclaw",
        llm_agent=f"openclaw/{model_id}",
        user="user_simulator",
        llm_user=user_llm,
        num_trials=args.num_trials,
        num_tasks=args.num_tasks,
        task_ids=task_ids,
        task_split_name=split,
        save_to=save_to,
        max_concurrency=1,  # sequential — openclaw sessions are per-process
        max_steps=args.max_steps,
    )

    try:
        results = run_domain(config)
        print(f"\nDone. Results saved to tau2-bench/data/tau2/simulations/{save_to}.json")
    except Exception as e:
        print(f"\n[Error] {e}")
        raise
    finally:
        if session_ids:
            print(f"\nSessions ({len(session_ids)}):")
            for sid in session_ids:
                print(f"  {sid}")

            from count_tokens import count_tokens
            print(f"\nToken usage:")
            stats = count_tokens(session_ids, domain=args.domain if agent_id else None)
            print(f"  input={stats['input']:,}  output={stats['output']:,}  total={stats['totalTokens']:,}")


if __name__ == "__main__":
    main()
