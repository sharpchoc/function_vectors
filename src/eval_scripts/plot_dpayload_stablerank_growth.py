#!/usr/bin/env python
"""Cumulative stable rank of stacked d_payload directions across the 20 train tasks (CPU).

Take the train tasks in a random order; stack the first task's 40 unit d_payload vectors
(pooled top-40 heads, exactly the ablation-SVD rows as in plot_dpayload_pca_alltasks.py),
compute the stable rank (sum s^2 / s1^2, fp64); append the next task's 40 rows and recompute;
repeat through all 20 tasks. The headline trace uses --seed; a --n_shuffles envelope
(min-max + mean over random orders) shows order dependence. If tasks contributed independent
payload geometry the curve would keep climbing; if d_payload geometry is dominated by fixed
per-head directions (part 11) it saturates fast.

Output: dpayload_stablerank_growth.{png,csv} in attention_head_analysis/top40_head_geometry/.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
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
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
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

    # mmap the checkpoint instead of instantiating the model: we only need 40 v_proj slices
    # (full fp32 GPT-J load is ~24 GB and gets OOM-killed under a 16 GB cgroup cap).
    import glob
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    bins = glob.glob(str(hf_home / "hub" / "models--EleutherAI--gpt-j-6b" /
                         "snapshots" / "*" / "pytorch_model.bin"))
    assert bins, "GPT-J pytorch_model.bin not found in HF cache"
    print(f"mmap-loading v_proj slices from {bins[0]}")
    sd = torch.load(bins[0], map_location="cpu", weights_only=True, mmap=True)
    HD = 256
    w_v = {(l, h): sd[f"transformer.h.{l}.attn.v_proj.weight"][h * HD:(h + 1) * HD].double()
           for l, h in heads}
    del sd

    # HARD GATE: rebuilding a stored subspace artifact's d_payloads via this route must match.
    gate_path = ARTIFACTS_ROOT / "payload_subspaces" / "synonym_pooled40heads_k4.pt"
    if gate_path.exists():
        ref = torch.load(gate_path, weights_only=False)
        ma = torch.load(args.aie_root / "synonym" / "synonym_mean_head_activations_varicl.pt",
                        weights_only=False)
        assert [tuple(h) for h in ref["heads"]] == heads, "gate: head list mismatch"
        rebuilt = []
        for (l, h) in heads:
            z = ma[l, h].to(torch.float64)
            d = w_v[(l, h)].T @ (z / z.norm())
            rebuilt.append(d / d.norm())
        dev = (torch.stack(rebuilt) - ref["d_payloads"]).abs().max().item()
        assert dev < 1e-5, f"gate: rebuilt d_payloads deviate from stored artifact ({dev:.2e})"
        print(f"gate passed: synonym d_payloads match stored artifact (max dev {dev:.2e})")

    stacks = {}   # task -> (40, 4096) unit d_payload rows, fp64
    single_sr = {}
    for task in train_tasks:
        ma = torch.load(args.aie_root / task / f"{task}_mean_head_activations_varicl.pt",
                        weights_only=False)
        rows = []
        for (l, h) in heads:
            z = ma[l, h].to(torch.float64)
            d = w_v[(l, h)].T @ (z / z.norm())
            rows.append((d / d.norm()).numpy())
        stacks[task] = np.stack(rows)
        single_sr[task] = stable_rank(stacks[task])
        print(f"  {task}: single-task SR {single_sr[task]:.2f}")

    def growth(order):
        out = []
        for m in range(1, len(order) + 1):
            X = np.concatenate([stacks[t] for t in order[:m]], axis=0)
            out.append(stable_rank(X))
        return np.array(out)

    rng = np.random.default_rng(args.seed)
    headline_order = list(train_tasks)
    rng.shuffle(headline_order)
    headline = growth(headline_order)
    print(f"\nheadline order (seed {args.seed}): {headline_order}")

    env = np.stack([growth(list(np.random.default_rng(1000 + i).permutation(train_tasks)))
                    for i in range(args.n_shuffles)])   # (n_shuffles, 20)
    assert np.allclose(env[:, -1], env[0, -1], atol=1e-8), "m=20 SR must be order-independent"
    ms = np.arange(1, len(train_tasks) + 1)

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    ax.fill_between(ms, env.min(0), env.max(0), color="tab:blue", alpha=0.15,
                    label=f"min–max over {args.n_shuffles} random orders")
    ax.plot(ms, env.mean(0), color="tab:blue", lw=1.2, ls="--", label="mean over orders")
    ax.plot(ms, headline, "-o", color="tab:blue", lw=1.8, ms=4,
            label=f"seed-{args.seed} order")
    msr = np.mean(list(single_sr.values()))
    ax.axhline(msr, color="grey", ls=":", lw=1.3,
               label=f"mean single-task SR ({msr:.1f})")
    ax.plot(ms, np.minimum(40 * ms, 4096) * 0 + env[0, -1], visible=False)  # keep ylim sane
    ax.set_xticks(ms)
    ax.set_xlabel("number of train tasks stacked (40 unit d_payload rows each)")
    ax.set_ylabel("stable rank  (Σσ² / σ₁²)")
    ax.set_title("Cumulative stable rank of pooled-top-40-head d_payload stacks, 20 train tasks\n"
                 "d_payload = unit(W_Vᵀ · unit(task-mean head activation)); fp64 uncentered SVD",
                 fontsize=10.5, loc="left")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out_png = args.out_dir / "dpayload_stablerank_growth.png"
    fig.savefig(out_png)
    print(f"wrote {out_png}")

    # ---- csv ----
    out_csv = args.out_dir / "dpayload_stablerank_growth.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "sr_seed_order", "task_added_seed_order", "sr_mean", "sr_min", "sr_max"])
        for i, m in enumerate(ms):
            w.writerow([m, f"{headline[i]:.4f}", headline_order[i],
                        f"{env.mean(0)[i]:.4f}", f"{env.min(0)[i]:.4f}", f"{env.max(0)[i]:.4f}"])
        w.writerow([])
        w.writerow(["task", "single_task_sr"])
        for t in train_tasks:
            w.writerow([t, f"{single_sr[t]:.4f}"])
    print(f"wrote {out_csv}")

    print("\nm  seed-order SR (task added)      mean [min, max] over orders")
    for i, m in enumerate(ms):
        print(f"{m:2d}  {headline[i]:6.2f} (+{headline_order[i]:<28s}) "
              f"{env.mean(0)[i]:6.2f} [{env.min(0)[i]:5.2f}, {env.max(0)[i]:5.2f}]")


if __name__ == "__main__":
    main()
