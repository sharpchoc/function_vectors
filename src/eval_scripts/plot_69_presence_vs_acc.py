#!/usr/bin/env python
"""Aggregate + plot FV-presence vs n-shot accuracy (capture_69_presence_vs_acc.py).

Per task and n in 0..6: x = mean over the 150 paired prompts of cos(z_l, v_hat_A) at the
query cue; y = temperature-1 sampled exact-match accuracy on the same prompts. One point
per task per n (69 points per panel). 14 figures: one per layer 9..20 plus max-over-layers
and mean-over-layers (per prompt, then prompt-averaged). Each figure has 7 panels (n=0..6)
with Spearman rho / Pearson r annotated; train tasks are circles, held-out tasks triangles.

Outputs (RESULTS/69_task_run/FV_location/presence_vs_accuracy/):
  scatter_L{9..20}.png, scatter_maxL.png, scatter_meanL.png
  presence_vs_acc.npz        per-task matrices (cos means, accs, groups, tasks)
  correlation_summary.csv    Spearman + Pearson per (variant, n)
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR  # noqa: E402

LAYERS = list(range(9, 21))
N_SHOTS = list(range(0, 7))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "presence_vs_acc")
    p.add_argument("--out_dir", type=Path,
                   default=TASK69_RUN_DIR / "FV_location" / "presence_vs_accuracy")
    return p.parse_args()


def scatter_fig(x_tn, acc_tn, groups, label, out_path):
    """x_tn, acc_tn: (n_tasks, 7). One panel per n."""
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharey=True)
    rows = []
    tr = np.array([g == "train" for g in groups])
    for ni, n in enumerate(N_SHOTS):
        ax = axes.flat[ni]
        x, y = x_tn[:, ni], acc_tn[:, ni]
        ax.scatter(x[tr], y[tr], s=18, c="tab:blue", label="train (55)")
        ax.scatter(x[~tr], y[~tr], s=26, c="tab:red", marker="^", label="held-out (14)")
        rho, rho_p = spearmanr(x, y)
        r, r_p = pearsonr(x, y)
        rows.append((label, n, rho, rho_p, r, r_p))
        ax.set_title(f"n={n}   Spearman ρ={rho:.2f} (p={rho_p:.1e})   r={r:.2f}", fontsize=9)
        ax.set_xlabel(f"FV presence  cos @ {label}")
        if ni % 4 == 0:
            ax.set_ylabel("sampled exact-match accuracy")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8, loc="upper left")
    axes.flat[7].axis("off")
    fig.suptitle(f"FV presence at the query cue ({label}) vs n-shot accuracy — 69 tasks, "
                 "150 paired prompts each", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return rows


def main():
    args = parse_args()
    files = sorted(args.in_root.glob("*.npz"))
    assert len(files) == 69, f"expected 69 task files, found {len(files)}"
    tasks, groups, cos_means, accs = [], [], [], []
    cos_max_means, cos_avg_means = [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        assert list(z["layers"]) == LAYERS and list(z["n_shots"]) == N_SHOTS
        tasks.append(f.stem)
        groups.append(str(z["group"]))
        cos = z["cos"]                                    # (7, 150, 12)
        cos_means.append(cos.mean(axis=1))                # (7, 12)
        cos_max_means.append(cos.max(axis=2).mean(axis=1))    # per-prompt max -> (7,)
        cos_avg_means.append(cos.mean(axis=2).mean(axis=1))   # per-prompt mean -> (7,)
        accs.append(z["match"].mean(axis=1))              # (7,)
    cos_means = np.stack(cos_means)                       # (T, 7, 12)
    cos_max_means, cos_avg_means = np.stack(cos_max_means), np.stack(cos_avg_means)
    accs = np.stack(accs)                                 # (T, 7)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "presence_vs_acc.npz",
             cos_by_layer=cos_means, cos_maxL=cos_max_means, cos_meanL=cos_avg_means,
             acc=accs, tasks=np.array(tasks), groups=np.array(groups),
             layers=np.array(LAYERS), n_shots=np.array(N_SHOTS))

    all_rows = []
    for li, l in enumerate(LAYERS):
        all_rows += scatter_fig(cos_means[:, :, li], accs, groups, f"L{l}",
                                args.out_dir / f"scatter_L{l}.png")
    all_rows += scatter_fig(cos_max_means, accs, groups, "maxL9-20",
                            args.out_dir / "scatter_maxL.png")
    all_rows += scatter_fig(cos_avg_means, accs, groups, "meanL9-20",
                            args.out_dir / "scatter_meanL.png")

    with open(args.out_dir / "correlation_summary.csv", "w") as f:
        f.write("variant,n_shots,spearman_rho,spearman_p,pearson_r,pearson_p\n")
        for row in all_rows:
            f.write(f"{row[0]},{row[1]},{row[2]:.4f},{row[3]:.3e},{row[4]:.4f},{row[5]:.3e}\n")
    print(f"wrote {args.out_dir} (14 figures, npz, csv)")


if __name__ == "__main__":
    main()
