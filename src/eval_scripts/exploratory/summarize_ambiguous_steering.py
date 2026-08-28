"""Collate ambiguous-task steering results: task-specific FV vs train-pooled-head FV at top-10/20/40.
Headline metric = best-layer zero-shot FV-steering top-1 accuracy (and 10-shot-shuffled)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.paths import AMBIGUOUS_DIR

TASKS = ["magnitude", "identity", "count_vowels", "count_consonants"]
NS = [10, 20, 40]


def load(n, task):
    p = AMBIGUOUS_DIR / f"heldout_ambiguous_eval_top{n}" / task / "comparison_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


rows = []
for task in TASKS:
    # task-specific is identical across n; read from whichever n is present
    ts = None
    mt = {}
    nfilt = None
    for n in NS:
        d = load(n, task)
        if d is None:
            continue
        nfilt = d.get("n_filtered_test_examples")
        if ts is None:
            ts = d["task_specific_heads"]
        mt[n] = d["multitask_heads"]
    rows.append((task, nfilt, ts, mt))

print(f"{'task':17s} {'n_test':>6s} | {'task-specific':>22s} | "
      f"{'train top10':>16s} | {'train top20':>16s} | {'train top40':>16s}")
print(f"{'':17s} {'':>6s} | {'zs@L (fs)':>22s} | {'zs@L (fs)':>16s} | {'zs@L (fs)':>16s} | {'zs@L (fs)':>16s}")
print("-" * 110)


def fmt(block):
    if block is None:
        return f"{'--':>16s}"
    zs = block["best_zs_intervention_top1"]
    zl = block["best_zs_layer"]
    fs = block["best_fs_shuffled_intervention_top1"]
    return f"{zs:.2f}@L{zl:<2d}({fs:.2f})"


for task, nfilt, ts, mt in rows:
    tss = "--"
    if ts is not None:
        tss = f"{ts['best_zs_intervention_top1']:.2f}@L{ts['best_zs_layer']:<2d}" \
              f"({ts['best_fs_shuffled_intervention_top1']:.2f})"
    print(f"{task:17s} {str(nfilt):>6s} | {tss:>22s} | "
          f"{fmt(mt.get(10)):>16s} | {fmt(mt.get(20)):>16s} | {fmt(mt.get(40)):>16s}")

print("\nCell = best zero-shot FV-steering top-1 @ best layer (10-shot-shuffled top-1 in parens).")
