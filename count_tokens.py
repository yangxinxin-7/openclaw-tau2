"""Count token usage from OpenClaw sessions by tau2 session IDs."""

import json
import pathlib

SESSIONS_DIR = pathlib.Path.home() / ".openclaw" / "agents" / "main" / "sessions"
SESSIONS_JSON = SESSIONS_DIR / "sessions.json"


def count_tokens(tau2_session_ids: list[str]) -> dict:
    with open(SESSIONS_JSON) as f:
        sessions_map = json.load(f)

    total_input = 0
    total_output = 0
    total_tokens = 0

    for tau2_id in tau2_session_ids:
        key = f"agent:main:openresponses-user:{tau2_id}"
        entry = sessions_map.get(key)
        if not entry:
            print(f"  [warn] {tau2_id}: not found in sessions.json")
            continue

        session_uuid = entry["sessionId"]
        jsonl_path = SESSIONS_DIR / f"{session_uuid}.jsonl"
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
    parser.add_argument("sessions", nargs="*", help="tau2 session IDs (e.g. tau2-abc123)")
    args = parser.parse_args()

    if args.sessions:
        ids = args.sessions
    else:
        # default: all tau2 sessions in sessions.json
        with open(SESSIONS_JSON) as f:
            sessions_map = json.load(f)
        ids = [k.split("openresponses-user:")[-1] for k in sessions_map if "openresponses-user:tau2-" in k]

    print(f"Sessions: {len(ids)}")
    stats = count_tokens(ids)
    print()
    print("=" * 60)
    print(f"Input  : {stats['input']:,}")
    print(f"Output : {stats['output']:,}")
    print(f"Total  : {stats['totalTokens']:,}")
