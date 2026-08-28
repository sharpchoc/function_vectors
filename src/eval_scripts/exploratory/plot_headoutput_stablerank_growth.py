#!/usr/bin/env python
"""Cumulative stable rank of stacked FV head-OUTPUT directions across the 20 train tasks (CPU).

Variation of plot_dpayload_stablerank_growth.py: rows are the out_proj-projected task-mean
head activations o = W_O[:, head] @ z_bar — i.e. exactly the 40 per-head FV summands (part 3's
"output" stack) — instead of the W_V^T pullbacks. Stack one random-ordered train task's 40
rows, compute stable rank (sum s^2/s1^2, fp64), append the next task's 40 rows, repeat.
Two panels: UNIT rows (directions only, comparable to the d_payload study) and RAW rows
(FV-weighted, norms span ~1-17 so a few strong heads dominate).

Sanity anchor (part 3, WORKLOG 2026-07-29): present-past single-task SR = 4.63 raw /
12.16 unit — reproduced as an advisory check.

Output: headoutput_stablerank_growth.{png,csv} in attention_head_analysis/top40_head_geometry/.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task_split_path", type=Path,
                   default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--aie_root", type=Path, default=ARTIFACTS_ROOT / "multitask_aie_heads_varicl")
    p.add_argument("--pooled_heads_path", type=Path,
                   default=ARTIFACTS_ROOT / "multitask_aie_heads_varicl" / "multitask_top_aie_heads.pt")
    p.add_argument("--seed", type=int, default=42, help="Headline task-order shuffle.")
    p.add_argument("--n_shuffles", type=int, default=20, help="Random orders for the envelope.")
    p.add_argument("--out_dir", type=Path,
                   default=FV_FORMATION_DIR / "attention_head_analysis" / "top40_head_geometry")
    return p.parse_args()


def stable_rank(X):
    s = np.linalg.svd(X, compute_uv=False)
    return float((s ** 2).sum() / s[0] ** 2)


def main():
    args = parse_args()
    train_tasks = list(json.loads(args.task_split_path.read_text())["train_tasks"])
    heads = [(int(l), int(h)) for l, h, _ in
             torch.load(args.pooled_heads_path, weights_only=False)["top_heads"]]

    # mmap the checkpoint for the 40 out_proj column slices (full fp32 load OOMs at 16 GB cap)
    import glob
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    bins = glob.glob(str(hf_home / "hub" / "models--EleutherAI--gpt-j-6b" /
                         "snapshots" / "*" / "pytorch_model.bin"))
    assert bins, "GPT-J pytorch_model.bin not found in HF cache"
    print(f"mmap-loading out_proj slices from {bins[0]}")
    sd = torch.load(bins[0], map_location="cpu", weights_only=True, mmap=True)
    HD = 256
    w_o = {(l, h): sd[f"transformer.h.{l}.attn.out_proj.weight"][:, h * HD:(h + 1) * HD].double()
           for l, h in heads}
    del sd

    stacks = {"raw": {}, "unit": {}}
    single_sr = {"raw": {}, "unit": {}}
    for task in train_tasks:
        ma = torch.load(args.aie_root / task / f"{task}_mean_head_activations_varicl.pt",
                        weights_only=False)
        raw = []
        for (l, h) in heads:
            o = w_o[(l, h)] @ ma[l, h].to(torch.float64)
            raw.append(o.numpy())
        raw = np.stack(raw)
        stacks["raw"][task] = raw
        stacks["unit"][task] = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        for kind in ("raw", "unit"):
            single_sr[kind][task] = stable_rank(stacks[kind][task])
        print(f"  {task}: single-task SR raw {single_sr['raw'][task]:.2f} / "
              f"unit {single_sr['unit'][task]:.2f}"
              + ("   <-- part-3 anchor: 4.63 / 12.16" if task == "present-past" else ""))

    pp_raw, pp_unit = single_sr["raw"].get("present-past"), single_sr["unit"].get("present-past")
    if pp_raw is not None:
        dev = max(abs(pp_raw - 4.63), abs(pp_unit - 12.16))
        print(f"advisory anchor check (present-past vs part 3): dev {dev:.3f}"
              + ("  OK" if dev < 0.02 else "  MISMATCH — investigate before trusting"))

    def growth(kind, order):
        return np.array([stable_rank(np.concatenate([stacks[kind][t] for t in order[:m]], axis=0))
                         for m in range(1, len(order) + 1)])

    rng = np.random.default_rng(args.seed)
    headline_order = list(train_tasks)
    rng.shuffle(headline_order)
    print(f"\nheadline order (seed {args.seed}): {headline_order}")

    ms = np.arange(1, len(train_tasks) + 1)
    results = {}
    for kind in ("unit", "raw"):
        headline = growth(kind, headline_order)
        env = np.stack([growth(kind, list(np.random.default_rng(1000 + i).permutation(train_tasks)))
                        for i in range(args.n_shuffles)])
        assert np.allclose(env[:, -1], env[0, -1], atol=1e-8), "m=20 SR must be order-independent"
        results[kind] = (headline, env)

    # ---- figure: two panels ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    for ax, kind, label in ((axes[0], "unit", "UNIT rows (directions only)"),
                            (axes[1], "raw", "RAW rows (FV-weighted, norms ~1–17)")):
        headline, env = results[kind]
        ax.fill_between(ms, env.min(0), env.max(0), color="tab:blue", alpha=0.15,
                        label=f"min–max over {args.n_shuffles} random orders")
        ax.plot(ms, env.mean(0), color="tab:blue", lw=1.2, ls="--", label="mean over orders")
        ax.plot(ms, headline, "-o", color="tab:blue", lw=1.8, ms=4, label=f"seed-{args.seed} order")
        msr = np.mean(list(single_sr[kind].values()))
        ax.axhline(msr, color="grey", ls=":", lw=1.3, label=f"mean single-task SR ({msr:.1f})")
        ax.set_xticks(ms)
        ax.set_xlabel("number of train tasks stacked (40 rows each)")
        ax.set_ylabel("stable rank  (Σσ² / σ₁²)")
        ax.set_title(label, fontsize=10.5)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Cumulative stable rank of FV head-OUTPUT stacks (o = W_O·z̄, the per-head FV "
                 "summands), 20 train tasks — fp64 uncentered SVD", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.out_dir / "headoutput_stablerank_growth.png"
    fig.savefig(out_png)
    print(f"wrote {out_png}")

    # ---- csv ----
    out_csv = args.out_dir / "headoutput_stablerank_growth.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "task_added_seed_order",
                    "unit_sr_seed", "unit_sr_mean", "unit_sr_min", "unit_sr_max",
                    "raw_sr_seed", "raw_sr_mean", "raw_sr_min", "raw_sr_max"])
        for i, m in enumerate(ms):
            hu, eu = results["unit"]; hr, er = results["raw"]
            w.writerow([m, headline_order[i],
                        f"{hu[i]:.4f}", f"{eu.mean(0)[i]:.4f}", f"{eu.min(0)[i]:.4f}", f"{eu.max(0)[i]:.4f}",
                        f"{hr[i]:.4f}", f"{er.mean(0)[i]:.4f}", f"{er.min(0)[i]:.4f}", f"{er.max(0)[i]:.4f}"])
        w.writerow([])
        w.writerow(["task", "single_task_sr_unit", "single_task_sr_raw"])
        for t in train_tasks:
            w.writerow([t, f"{single_sr['unit'][t]:.4f}", f"{single_sr['raw'][t]:.4f}"])
    print(f"wrote {out_csv}")

    for kind in ("unit", "raw"):
        headline, env = results[kind]
        print(f"\n[{kind}] m  seed-order SR (task added)      mean [min, max] over orders")
        for i, m in enumerate(ms):
            print(f"{m:2d}  {headline[i]:6.2f} (+{headline_order[i]:<28s}) "
                  f"{env.mean(0)[i]:6.2f} [{env.min(0)[i]:5.2f}, {env.max(0)[i]:5.2f}]")


if __name__ == "__main__":
    main()
