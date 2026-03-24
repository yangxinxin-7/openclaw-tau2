"""Show memories stored by the memory-lancedb plugin.

Usage:
    python show_memory.py                    # show all agents summary
    python show_memory.py --domain airline   # show airline agent memories
    python show_memory.py --domain retail --search "refund"
    python show_memory.py --domain telecom --limit 20
"""

import argparse
import os
import sys

DOMAIN_AGENT_MAP = {
    "airline": "tau2-airline",
    "retail": "tau2-retail",
    "telecom": "tau2-telecom",
}
WORKSPACE_DIR = os.path.expanduser("~/.openclaw/workspace")
DEFAULT_DB_PATH = os.path.expanduser("~/memory/lancedb")


def get_db_path(domain=None):
    if domain:
        agent_id = DOMAIN_AGENT_MAP.get(domain)
        if not agent_id:
            print(f"Unknown domain: {domain}. Choose from: {list(DOMAIN_AGENT_MAP)}")
            sys.exit(1)
        return os.path.join(WORKSPACE_DIR, agent_id, "memory", "lancedb")
    return DEFAULT_DB_PATH


def open_table(db_path):
    try:
        import lancedb
    except ImportError:
        print("lancedb not installed. Run: pip install lancedb")
        sys.exit(1)

    if not os.path.exists(db_path):
        return None, None

    db = lancedb.connect(db_path)
    tables = db.table_names()
    if "memories" not in tables:
        return db, None
    return db, db.open_table("memories")


def show_summary():
    print("=== Memory Summary ===")
    # Default shared path
    _, table = open_table(DEFAULT_DB_PATH)
    default_count = table.count_rows() if table else 0
    print(f"  default ({DEFAULT_DB_PATH}): {default_count} memories")

    for domain, agent_id in DOMAIN_AGENT_MAP.items():
        db_path = os.path.join(WORKSPACE_DIR, agent_id, "memory", "lancedb")
        _, table = open_table(db_path)
        count = table.count_rows() if table else 0
        print(f"  {domain} ({agent_id}): {count} memories")


def show_domain(domain, search=None, limit=50):
    db_path = get_db_path(domain)
    print(f"DB path: {db_path}")

    _, table = open_table(db_path)
    if table is None:
        print("No memories found (table does not exist).")
        return

    import pandas as pd
    df = table.to_pandas()

    if df.empty:
        print("No memories stored.")
        return

    if search:
        mask = df["text"].str.contains(search, case=False, na=False)
        df = df[mask]
        print(f"Filtered by '{search}': {len(df)} matches")

    df = df.sort_values("createdAt", ascending=False).head(limit)

    print(f"\nTotal: {len(df)} memories\n")
    for _, row in df.iterrows():
        from datetime import datetime
        ts = datetime.fromtimestamp(row["createdAt"] / 1000).strftime("%Y-%m-%d %H:%M")
        print(f"[{ts}] [{row['category']}] (importance={row['importance']:.2f})")
        print(f"  {row['text']}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Show OpenClaw LanceDB memories")
    parser.add_argument("--domain", choices=list(DOMAIN_AGENT_MAP.keys()), default=None)
    parser.add_argument("--search", type=str, default=None, help="Filter by keyword")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    if args.domain is None:
        show_summary()
    else:
        show_domain(args.domain, search=args.search, limit=args.limit)


if __name__ == "__main__":
    main()
