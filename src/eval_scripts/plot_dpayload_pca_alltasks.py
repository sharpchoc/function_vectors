#!/usr/bin/env python
"""2D PCA of the d_payload vectors for all 27 tasks (20 train + 7 Stream W test), CPU.

For every task: the 40 unit d_payload = unit(W_V^T @ unit(z_bar)) vectors over the pooled
top-40 train-selected heads (z_bar = the task's cached varicl mean head activation at the
cue token) — i.e. exactly the row vectors whose uncentered SVD defined the ablation
subspaces. All 27 x 40 = 1080 vectors are stacked, mean-centered, and projected onto their
top-2 PCs; scatter colored by task (train = circles, test = triangles).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR

TEST7 = ["landmark-country", "word_length", "capitalize_first_letter", "synonym",
         "lowercase_first_letter", "capitalize", "antonym"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task_split_path", type=Path,
                   default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--aie_root", type=Path, default=ARTIFACTS_ROOT / "multitask_aie_heads_varicl")
    p.add_argument("--pooled_heads_path", type=Path,
                   default=ARTIFACTS_ROOT / "multitask_aie_heads_varicl" / "multitask_top_aie_heads.pt")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--out_dir", type=Path,
                   default=FV_FORMATION_DIR / "attention_head_analysis" / "top40_head_geometry")
    return p.parse_args()


def main():
    args = parse_args()
    split = json.loads(args.task_split_path.read_text())
    train_tasks = list(split["train_tasks"])
    tasks = train_tasks + TEST7                      # 27
    heads = [(int(l), int(h)) for l, h, _ in
             torch.load(args.pooled_heads_path, weights_only=False)["top_heads"]]

    from transformers import AutoModelForCausalLM
    print("Loading model (CPU fp32)...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32,
                                                 low_cpu_mem_usage=True)
    model.eval()
    HD = 256
    w_v_slices = {}
    with torch.no_grad():
        for l, h in heads:
            w_v_slices[(l, h)] = model.transformer.h[l].attn.v_proj.weight[
                h * HD:(h + 1) * HD].double()        # (256, 4096)

    rows, row_task = [], []
    for task in tasks:
        ma = torch.load(args.aie_root / task / f"{task}_mean_head_activations_varicl.pt",
                        weights_only=False)
        for (l, h) in heads:
            z = ma[l, h].to(torch.float64)
            d = w_v_slices[(l, h)].T @ (z / z.norm())
            rows.append((d / d.norm()).numpy())
            row_task.append(task)
    X = np.stack(rows)                               # (1080, 4096)
    print(f"stacked {X.shape}; {len(tasks)} tasks x {len(heads)} heads")

    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    P = U[:, :2] * S[:2]
    print(f"PC1 {var[0]:.1%}, PC2 {var[1]:.1%} of centered variance")

    # 27 distinguishable colors
    cmap20 = plt.get_cmap("tab20")
    cmap20b = plt.get_cmap("tab20b")
    colors = [cmap20(i) for i in range(20)] + [cmap20b(i) for i in (0, 4, 8, 12, 16, 2, 6)]
    task_color = {t: colors[i] for i, t in enumerate(tasks)}

    fig, ax = plt.subplots(figsize=(13.5, 9), dpi=200)
    for t in tasks:
        m = np.array([rt == t for rt in row_task])
        marker = "o" if t in train_tasks else "^"
        ax.scatter(P[m, 0], P[m, 1], s=26, c=[task_color[t]], marker=marker,
                   edgecolors="white", linewidths=0.4,
                   label=f"{t}" + ("" if t in train_tasks else " (test)"), zorder=3)
    ax.axhline(0, color="#d9d8d3", lw=0.8, zorder=1)
    ax.axvline(0, color="#d9d8d3", lw=0.8, zorder=1)
    ax.set_xlabel(f"PC1 ({var[0]:.0%} var)", fontsize=10, color="#454540")
    ax.set_ylabel(f"PC2 ({var[1]:.0%} var)", fontsize=10, color="#454540")
    for spine in ax.spines.values():
        spine.set_color("#d9d8d3")
    ax.tick_params(labelsize=8, color="#d9d8d3")
    ax.set_title("d_payload vectors of the pooled top-40 heads, all 27 tasks — 2D PCA "
                 "(centered)\nd_payload = unit(W_V^T z̄_task); 40 points per task; "
                 "circles = train tasks, triangles = held-out test tasks",
                 fontsize=11, loc="left", color="#29291f")
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "pca2d_dpayload_27tasks.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    np.savez(args.out_dir / "pca2d_dpayload_27tasks.npz",
             pc=P, variance_ratio=var[:10], tasks=np.array(row_task),
             heads=np.array(heads), task_order=np.array(tasks),
             definition="unit(W_V^T @ unit(task-mean head activation)), pooled top-40 heads")


if __name__ == "__main__":
    main()
