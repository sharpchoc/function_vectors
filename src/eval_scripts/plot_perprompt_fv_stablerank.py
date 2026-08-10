#!/usr/bin/env python
"""Per-task stable rank of stacked per-prompt FVs, under two head definitions (CPU).

For each task, stack the 170 per-prompt function vectors v^j_A (fixed10 capture) into a
(170, 4096) matrix and compute the stable rank (sum s^2 / s1^2, fp64 SVD), for BOTH
H = canonical pooled top-40 and H = SANDBOX vanilla_sparse_opt23 (23 heads, unweighted).

Two panels (definitional choice surfaced, not hidden):
  - UNCENTERED stacks (repo SR precedent, parts 13/13b): dominated by the task-mean FV
    direction, so values near 1 measure how large prompt-to-prompt variation is relative
    to the shared task direction.
  - MEAN-CENTERED stacks: stable rank of the fluctuations themselves -- how many effective
    directions the per-prompt FV varies along within a task.

Output: fvstack_stablerank_pertask.{png,csv} in the sandbox perprompt_fv_norms folder
(one of the two definitions is SANDBOX).
"""
import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from utils.paths import ARTIFACTS_ROOT, RESULTS_ROOT

HD = 256


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture_dir", type=Path,
                   default=ARTIFACTS_ROOT / "perprompt_head_activations" / "gptj_27tasks_170prompts" / "fixed10")
    p.add_argument("--top40_heads_path", type=Path,
                   default=ARTIFACTS_ROOT / "multitask_aie_heads_varicl" / "multitask_top_aie_heads.pt")
    p.add_argument("--sparse_heads_path", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox" / "sparse_head_selection" / "vanilla_sparse_opt23_heads.pt")
    p.add_argument("--out_dir", type=Path,
                   default=RESULTS_ROOT / "sandbox" / "perprompt_fv_norms_vanilla_sparse_opt23")
    return p.parse_args()


def stable_rank(X):
    s = np.linalg.svd(X, compute_uv=False)
    return float((s ** 2).sum() / s[0] ** 2)


def main():
    args = parse_args()
    head_sets = {}
    for name, path in (("top40", args.top40_heads_path), ("sparse23", args.sparse_heads_path)):
        heads = [(int(l), int(h)) for l, h, _ in torch.load(path, weights_only=False)["top_heads"]]
        assert heads and len(set(heads)) == len(heads)
        head_sets[name] = heads
    all_heads = sorted(set(head_sets["top40"]) | set(head_sets["sparse23"]))

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    bins = glob.glob(str(hf_home / "hub" / "models--EleutherAI--gpt-j-6b" /
                         "snapshots" / "*" / "pytorch_model.bin"))
    assert bins, "GPT-J pytorch_model.bin not found in HF cache"
    sd = torch.load(bins[0], map_location="cpu", weights_only=True, mmap=True)
    w_o = {(l, h): sd[f"transformer.h.{l}.attn.out_proj.weight"][:, h * HD:(h + 1) * HD].double()
           for l, h in all_heads}
    del sd

    tasks = sorted(p.stem for p in args.capture_dir.glob("*.pt"))
    assert len(tasks) == 27
    sr = {name: {"uncentered": {}, "centered": {}} for name in head_sets}
    for task in tasks:
        acts = torch.load(args.capture_dir / f"{task}.pt", weights_only=False)["activations"].double()
        for name, heads in head_sets.items():
            fvs = torch.zeros(acts.shape[0], 4096, dtype=torch.float64)
            for (l, h) in heads:
                fvs += acts[:, l, h] @ w_o[(l, h)].T
            X = fvs.numpy()
            sr[name]["uncentered"][task] = stable_rank(X)
            sr[name]["centered"][task] = stable_rank(X - X.mean(axis=0, keepdims=True))
        print(f"{task:28s} " + "  ".join(
            f"{n}: unc {sr[n]['uncentered'][task]:5.2f} cen {sr[n]['centered'][task]:6.2f}"
            for n in head_sets))

    order = sorted(tasks, key=lambda t: sr["top40"]["uncentered"][t])
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), dpi=150)
    for ax, kind, title in (
        (axes[0], "uncentered", "UNCENTERED stacks (dominated by the shared task-mean FV "
                                "direction; SR-1 ~ relative prompt-to-prompt variation)"),
        (axes[1], "centered", "MEAN-CENTERED stacks (effective dimensionality of "
                              "within-task per-prompt FV variation)"),
    ):
        ax.bar(x - 0.2, [sr["top40"][kind][t] for t in order], width=0.4,
               color="tab:blue", label="canonical pooled top-40")
        ax.bar(x + 0.2, [sr["sparse23"][kind][t] for t in order], width=0.4,
               color="tab:orange", label="SANDBOX vanilla_sparse_opt23")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel("stable rank  (Σσ² / σ₁²)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=8.5)
    fig.suptitle("Stable rank of each task's stacked per-prompt FVs (170 × 4096, fixed10) — "
                 "tasks sorted by top-40 uncentered SR; fp64 SVD", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.out_dir / "fvstack_stablerank_pertask.png"
    fig.savefig(out_png)
    print(f"wrote {out_png}")

    out_csv = args.out_dir / "fvstack_stablerank_pertask.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "top40_uncentered", "top40_centered",
                    "sparse23_uncentered", "sparse23_centered"])
        for t in tasks:
            w.writerow([t] + [f"{sr[n][k][t]:.4f}" for n in ("top40", "sparse23")
                              for k in ("uncentered", "centered")])
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
