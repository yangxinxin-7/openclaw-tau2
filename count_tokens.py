"""Count token usage from OpenClaw sessions by tau2 session IDs."""

import json
import pathlib
from typing import Optional


def get_sessions_dir(domain: Optional[str] = None) -> pathlib.Path:
    agent_name = f"tau2-{domain}" if domain else "main"
    return pathlib.Path.home() / ".openclaw" / "agents" / agent_name / "sessions"


def count_tokens(tau2_session_ids: list[str], domain: Optional[str] = None) -> dict:
    sessions_dir = get_sessions_dir(domain)
    sessions_json = sessions_dir / "sessions.json"
    agent_name = f"tau2-{domain}" if domain else "main"

    with open(sessions_json) as f:
        sessions_map = json.load(f)

    total_input = 0
    total_output = 0
    total_tokens = 0

    for tau2_id in tau2_session_ids:
        key = f"agent:{agent_name}:openresponses-user:{tau2_id}"
        entry = sessions_map.get(key)
        if not entry:
            print(f"  [warn] {tau2_id}: not found in {sessions_json}")
            continue

        session_uuid = entry["sessionId"]
        jsonl_path = sessions_dir / f"{session_uuid}.jsonl"
        if not jsonl_path.exists():
            print(f"  [warn] {tau2_id}: jsonl not found ({session_uuid})")
            continue

        s_input = s_output = s_total = 0
        for line in jsonl_path.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = e.get("message", {})
            if msg.get("role") != "assistant":
                continue
            usage = msg.get("usage", {})
            if not usage:
                continue
            s_input += usage.get("input", 0)
            s_output += usage.get("output", 0)
            s_total += usage.get("totalTokens", 0)

        print(f"  {tau2_id} ({session_uuid})  input={s_input:>10,}  output={s_output:>8,}  total={s_total:>10,}")
        total_input += s_input
        total_output += s_output
        total_tokens += s_total

    return {"input": total_input, "output": total_output, "totalTokens": total_tokens}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["airline", "retail", "telecom"], default=None,
                        help="TAU-2 domain agent to read from (default: main)")
    parser.add_argument("sessions", nargs="*", help="tau2 session IDs (e.g. tau2-abc123)")
    args = parser.parse_args()

    sessions_dir = get_sessions_dir(args.domain)
    sessions_json = sessions_dir / "sessions.json"
    agent_name = f"tau2-{args.domain}" if args.domain else "main"

    if args.sessions:
        ids = args.sessions
    else:
        with open(sessions_json) as f:
            sessions_map = json.load(f)
        ids = [
            k.split("openresponses-user:")[-1]
            for k in sessions_map
            if k.startswith(f"agent:{agent_name}:openresponses-user:tau2-")
        ]

    print(f"Agent   : {agent_name}")
    print(f"Sessions: {len(ids)}")
    stats = count_tokens(ids, domain=args.domain)
    print()
    print("=" * 60)
    print(f"Input  : {stats['input']:,}")
    print(f"Output : {stats['output']:,}")
    print(f"Total  : {stats['totalTokens']:,}")
