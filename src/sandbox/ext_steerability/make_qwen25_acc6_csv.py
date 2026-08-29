#!/usr/bin/env python
"""Adapter: Qwen2.5-7B-Instruct plain-arm 6-shot artifacts (chat-template-transfer study,
same locked T=1 sampled exact-match protocol as the GPT-J sweep) -> an nshot_accuracy.csv-schema
file that make_ext_split.py can consume for the qwen25 FV pipeline's competence screen.

Reads artifacts/chat_template_transfer/ext117_6shot/plain/<task>.json (117 tasks, 50 prompts
each) plus origin/lane metadata from the GPT-J sweep CSV; writes
results/qwen25_fv/screen/qwen25_acc6.csv with columns
task,origin,lane,n_shots,n_prompts,accuracy,ci_lo,ci_hi (Wilson 95% CI like the original sweep).
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, GENERAL_DIR, QWEN25_FV_DIR  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_root", type=Path,
                    default=ARTIFACTS_ROOT / "chat_template_transfer" / "ext117_6shot" / "plain")
    ap.add_argument("--out", type=Path, default=QWEN25_FV_DIR / "screen" / "qwen25_acc6.csv")
    args = ap.parse_args()

    meta = {}
    for r in csv.DictReader(open(GENERAL_DIR / "extended_tasks_nshot_sweep" / "nshot_accuracy.csv")):
        meta[r["task"]] = {"origin": r["origin"], "lane": r["lane"]}
    pool = json.load(open(REPO_ROOT / "dataset_files" / "extended_tasks" / "manifest.json"))["tasks"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "origin", "lane", "n_shots", "n_prompts", "accuracy", "ci_lo", "ci_hi"])
        for t in sorted(pool):
            recs = json.load(open(args.in_root / f"{t}.json"))
            n = len(recs)
            k = sum(r["match"] for r in recs)
            lo, hi = wilson(k, n)
            w.writerow([t, meta[t]["origin"], meta[t]["lane"], 6, n,
                        round(k / n, 4), round(lo, 4), round(hi, 4)])
    n_pass = sum(1 for t in pool
                 if sum(r["match"] for r in json.load(open(args.in_root / f"{t}.json")))
                 / 50 >= 0.30)
    print(f"wrote {args.out} ({len(pool)} tasks; {n_pass} pass acc6 >= 0.30)")


if __name__ == "__main__":
    main()
