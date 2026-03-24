"""Summarize token usage from train simulation JSON files."""

import json
import pathlib

SIM_DIR = pathlib.Path(__file__).parent / "tau2-bench" / "data" / "tau2" / "simulations"

FILES = {
    "airline": SIM_DIR / "train_airline_train.json",
    "retail":  SIM_DIR / "train_retail_train.json",
    "telecom": SIM_DIR / "train_telecom_train.json",
}

header = f"{'Domain':<10} {'Tasks':>6} {'Input':>14} {'Output':>12} {'Total':>14}"
print(header)
print("-" * len(header))

grand = {"tasks": 0, "input": 0, "output": 0, "total": 0}

for domain, path in FILES.items():
    if not path.exists():
        print(f"{domain:<10}  (file not found: {path})")
        continue

    with open(path) as f:
        data = json.load(f)

    tasks = data.get("tasks", [])
    n = len(tasks)
    inp = sum(t.get("tokens", {}).get("input", 0) for t in tasks)
    out = sum(t.get("tokens", {}).get("output", 0) for t in tasks)
    total = sum(t.get("tokens", {}).get("totalTokens", 0) for t in tasks)

    print(f"{domain:<10} {n:>6} {inp:>14,} {out:>12,} {total:>14,}")

    grand["tasks"] += n
    grand["input"] += inp
    grand["output"] += out
    grand["total"] += total

print("-" * len(header))
print(f"{'TOTAL':<10} {grand['tasks']:>6} {grand['input']:>14,} {grand['output']:>12,} {grand['total']:>14,}")
