#!/usr/bin/env python
"""What does the model say when its task-unique read direction u_hat_A is ablated?

Reads the stored per-prompt predictions of the mean-residual ablation run
(artifacts/69_task_run/bottom_up_ablation/bankA_meanresid_top1/n{1,6}shot/<task>.json:
mean_ablation_pc5 = own u_hat_A, cf_mean_ablation_pc5 = counterfactual task's direction),
rebuilds each prompt's query / demos from dataset_files/isolation_prompts_ext/<task>/
train_prompts.json (same order, first n demos), and buckets every prediction:

  correct           exact match to gold
  copy_query        repeats the query input
  copy_demo_target  repeats one of the prompt's demonstration targets
  copy_demo_input   repeats one of the prompt's demonstration inputs
  in_pool_wrong     a valid output of THIS task (seen as a target elsewhere in the task's
                    prompt bank) but not the gold  -> the task was read, the mapping failed
  input_variant     shares a >=4-char prefix with the query input (morphological variant /
                    partial copy)
  other             anything else (free continuation, different task, junk)

Writes results/69_task_run/bottom_up_read_features/ablation/task_unique_meanresid/
failure_modes/{bucket_counts.csv, examples.md}.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT, TASK69_RUN_DIR  # noqa: E402

AR = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "bankA_meanresid_top1"
PR = REPO_ROOT / "dataset_files" / "isolation_prompts_ext"
OUT = TASK69_RUN_DIR / "bottom_up_read_features" / "ablation" / "task_unique_meanresid" / "failure_modes"
BUCKETS = ("correct", "copy_query", "copy_demo_target", "copy_demo_input", "in_pool_wrong", "input_variant", "other")


def norm(s):
    return str(s).strip().lower()


def bucket(pred, gold, q_in, demo_in, demo_out, pool):
    p = norm(pred)
    if p == norm(gold):
        return "correct"
    if p == norm(q_in):
        return "copy_query"
    if p in demo_out:
        return "copy_demo_target"
    if p in demo_in:
        return "copy_demo_input"
    if p in pool:
        return "in_pool_wrong"
    qi = norm(q_in)
    if len(p) >= 4 and len(qi) >= 4 and (p[:4] == qi[:4] or p in qi or qi in p):
        return "input_variant"
    return "other"


def analyse(task, n, n_examples=10):
    recs = json.load(open(PR / task / "train_prompts.json"))
    d = json.load(open(AR / f"n{n}shot" / f"{task}.json"))
    assert len(recs) == d["n_prompts"] == 150
    pool = set()
    for r in recs:
        for dm in r["demos"]:
            pool.add(norm(dm["output"]))
        qo = r["query"]["output"]
        pool.add(norm(qo[0] if isinstance(qo, list) else qo))
    own, cf = d["conditions"]["mean_ablation_pc5"]["preds"], d["conditions"]["cf_mean_ablation_pc5"]["preds"]
    cnt_own, cnt_cf, examples = Counter(), Counter(), []
    for r, g, po, pc in zip(recs, d["golds"], own, cf):
        demos = r["demos"][:n]
        din = {norm(x["input"]) for x in demos}
        dout = {norm(x["output"]) for x in demos}
        qi = str(r["query"]["input"])
        bo, bc = bucket(po, g, qi, din, dout, pool), bucket(pc, g, qi, din, dout, pool)
        cnt_own[bo] += 1
        cnt_cf[bc] += 1
        if bo != "correct" and len(examples) < n_examples:
            examples.append((qi, g, po, bo, pc))
    return cnt_own, cnt_cf, examples, d["cf_task"]


def main():
    tasks = sys.argv[1:] or ["present-past", "singular-plural", "country-capital", "next_number_digits",
                             "english-french", "adjective_to_adverb", "day_after_textual_date",
                             "language_identification", "sentiment", "uppercase_word", "antonym", "capitalize"]
    OUT.mkdir(parents=True, exist_ok=True)
    rows, md = [], ["# Failure modes under own-direction ($\\hat u_A$) mean-ablation\n",
                    "Buckets: see `ablation_failure_modes.py`. Each example: query input → gold | own-ablated prediction [bucket] | counterfactual-ablated prediction.\n"]
    for n in (6, 1):
        md.append(f"\n## {n}-shot\n")
        for t in tasks:
            co, cc, ex, cft = analyse(t, n)
            for arm, c in (("own", co), ("cf", cc)):
                rows.append({"task": t, "n_shots": n, "arm": arm, **{b: c.get(b, 0) for b in BUCKETS}})
            md.append(f"\n### {t}  (counterfactual direction: {cft})\n")
            md.append("| bucket | own $\\hat u_A$ ablated | counterfactual ablated |\n|---|---:|---:|")
            for b in BUCKETS:
                md.append(f"| {b} | {co.get(b, 0)} | {cc.get(b, 0)} |")
            md.append("\nExamples (own-ablation errors):\n")
            for qi, g, po, bo, pc in ex:
                md.append(f"- `{qi}` → **{g}** | own: `{po}` [{bo}] | cf: `{pc}`")
            print(f"n={n} {t:26s} own: " + "  ".join(f"{b}={co.get(b, 0):3d}" for b in BUCKETS))
    with open(OUT / "bucket_counts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    open(OUT / "examples.md", "w").write("\n".join(md) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
