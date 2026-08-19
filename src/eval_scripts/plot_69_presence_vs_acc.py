#!/usr/bin/env python
"""Aggregate + plot FV-presence vs n-shot accuracy (capture_69_presence_vs_acc.py).

Per task and n in 0..6: x = mean over the 150 paired prompts of cos(z_l, v_hat_A) at the
query cue; y = temperature-1 sampled exact-match accuracy on the same prompts. One point
per task per n. 14 figures: one per layer 9..20 plus max-over-layers and mean-over-layers
(per prompt, then prompt-averaged). Each figure is a SINGLE panel holding all 69x7 = 483
points, coloured by shot count, with the pooled Spearman/Pearson in the title and the per-n
rho in the legend; train tasks are circles, held-out tasks triangles.

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
                   default=TASK69_RUN_DIR / "write_feature_and_model_accuracy")
    return p.parse_args()


def scatter_fig(x_tn, acc_tn, groups, label, out_path):
    """x_tn, acc_tn: (n_tasks, 7). ONE panel holding every (task, n) point, coloured by n."""
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    rows = []
    tr = np.array([g == "train" for g in groups])
    cmap = plt.get_cmap("viridis")
    for ni, n in enumerate(N_SHOTS):
        x, y = x_tn[:, ni], acc_tn[:, ni]
        c = [cmap(ni / (len(N_SHOTS) - 1))]
        rho, rho_p = spearmanr(x, y)
        r, r_p = pearsonr(x, y)
        rows.append((label, n, rho, rho_p, r, r_p))
        ax.scatter(x[tr], y[tr], s=20, color=c, label=f"n={n}  (ρ={rho:+.2f})")
        ax.scatter(x[~tr], y[~tr], s=32, color=c, marker="^")
    x_all, y_all = x_tn.ravel(), acc_tn.ravel()
    rho_all, p_all = spearmanr(x_all, y_all)
    r_all, rp_all = pearsonr(x_all, y_all)
    rows.append((label, "pooled", rho_all, p_all, r_all, rp_all))
    ax.set_xlabel(f"FV presence   cos(z, v_A) at the query cue @ {label}")
    ax.set_ylabel("sampled exact-match accuracy (temperature 1.0)")
    ax.set_title(f"FV presence vs accuracy @ {label} — 69 tasks × n=0..6 "
                 f"({x_all.size} points)\npooled Spearman ρ={rho_all:+.2f} "
                 f"(p={p_all:.1e}), Pearson r={r_all:+.2f}   "
                 "[circles = train, triangles = held-out]", fontsize=11)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, title="shot count (per-n ρ)", loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return rows


def binned_fig(x_tn, acc_tn, label, out_path, width=0.10, anchor=0.05):
    """Bucket every (task, n) point by presence (width-0.10 bins anchored on 0.05, i.e.
    0.05-0.15, 0.15-0.25, ...) and plot the mean accuracy per bucket."""
    x, y = x_tn.ravel(), acc_tn.ravel()
    lo = anchor - width * np.ceil(max(0.0, anchor - x.min()) / width)
    edges = np.arange(lo, x.max() + width, width)
    idx = np.digitize(x, edges) - 1
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        sem = y[m].std(ddof=1) / np.sqrt(m.sum()) if m.sum() > 1 else 0.0
        rows.append((edges[b], edges[b + 1], int(m.sum()), y[m].mean(), sem))
    ctr = np.array([(a + b) / 2 for a, b, *_ in rows])
    cnt = np.array([c for *_, c, _, _ in rows])
    mean = np.array([m for *_, m, _ in rows])
    sem = np.array([s for *_, s in rows])

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(x, y, s=8, c="0.8", zorder=1, label=f"individual (task, n) points ({x.size})")
    ax.errorbar(ctr, mean, yerr=sem, fmt="o-", color="tab:blue", lw=2, ms=9, capsize=4,
                zorder=3, label="bucket mean ± SEM")
    for c, m, s, k in zip(ctr, mean, sem, cnt):
        ax.annotate(f"n={k}", (c, m + s), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color="tab:blue")
    for e in edges:
        ax.axvline(e, color="0.85", lw=0.8, zorder=0)
    rho, p = spearmanr(x, y)
    ax.set_xlabel(f"FV presence   cos(z, v_A) at the query cue @ {label}   "
                  f"(buckets of {width:g})")
    ax.set_ylabel("sampled exact-match accuracy (temperature 1.0)")
    ax.set_title(f"Accuracy vs FV presence, bucketed @ {label}\n"
                 f"all 69 tasks × n=0..6 pooled; point-level Spearman ρ={rho:+.2f} "
                 f"(p={p:.1e})", fontsize=11)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return [(label, f"[{a:.2f},{b:.2f})", c, m, s) for a, b, c, m, s in rows]


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

    all_rows, bin_rows = [], []
    variants = [(f"L{l}", cos_means[:, :, li], f"L{l}") for li, l in enumerate(LAYERS)]
    variants += [("maxL9-20", cos_max_means, "maxL"), ("meanL9-20", cos_avg_means, "meanL")]
    for label, x_tn, stem in variants:
        all_rows += scatter_fig(x_tn, accs, groups, label, args.out_dir / f"scatter_{stem}.png")
        bin_rows += binned_fig(x_tn, accs, label, args.out_dir / f"binned_{stem}.png")

    with open(args.out_dir / "correlation_summary.csv", "w") as f:
        f.write("variant,n_shots,spearman_rho,spearman_p,pearson_r,pearson_p\n")
        for row in all_rows:
            f.write(f"{row[0]},{row[1]},{row[2]:.4f},{row[3]:.3e},{row[4]:.4f},{row[5]:.3e}\n")
    with open(args.out_dir / "binned_summary.csv", "w") as f:
        f.write("variant,bucket,n_points,mean_acc,sem_acc\n")
        for v, b, c, m, s in bin_rows:
            f.write(f"{v},{b},{c},{m:.4f},{s:.4f}\n")
    print(f"wrote {args.out_dir} ({2 * len(variants)} figures, npz, 2 csv)")


if __name__ == "__main__":
    main()
