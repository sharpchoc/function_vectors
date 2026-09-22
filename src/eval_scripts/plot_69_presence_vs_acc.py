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
import json
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
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR, REPO_ROOT  # noqa: E402
from utils.paper_style import apply_paper_style, C, label_bars  # noqa: E402,F401

apply_paper_style()

LAYERS = list(range(9, 21))      # GPT-J capture band; overwritten from the npz files at run time
N_SHOTS = list(range(0, 7))
N_TASKS_LABEL = "69"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "presence_vs_acc")
    p.add_argument("--out_dir", type=Path,
                   default=TASK69_RUN_DIR / "write_feature_and_model_accuracy")
    # Qwen2.5 port (2026-09-22): captures may hold all 28 layers; the GPT-J band (9-20)
    # variants are still reported, per-layer scatters are restricted with --plot_layers,
    # and the headline layer (peak of the 6-shot presence profile) is written to
    # headline_layer.txt. Defaults reproduce the GPT-J outputs.
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--band", type=str, default="9-20",
                   help="layer band for the maxL/meanL variants (GPT-J 9-20)")
    p.add_argument("--plot_layers", type=str, default="all",
                   help="'all' = scatter/binned figure per captured layer (GPT-J), "
                        "'headline' = only the headline layer, or a comma list")
    p.add_argument("--model_label", default="GPT-J-6B")
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
    ax.set_title(f"FV presence vs accuracy @ {label} — {N_TASKS_LABEL} tasks × n=0..6 "
                 f"({x_all.size} points)\npooled Spearman ρ={rho_all:+.2f} "
                 f"(p={p_all:.1e}), Pearson r={r_all:+.2f}   "
                 "[circles = train, triangles = held-out]")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, axis="both")
    ax.legend(title="shot count (per-n ρ)", loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
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
    ax.scatter(x, y, s=8, color=C.lightgrey, zorder=1,
               label=f"individual (task, n) points ({x.size})")
    ax.errorbar(ctr, mean, yerr=sem, fmt="o-", color=C.write, lw=1.8, ms=7, capsize=3,
                zorder=3, label="bucket mean ± SEM")
    for c, m, s, k in zip(ctr, mean, sem, cnt):
        ax.annotate(f"n={k}", (c, m + s), textcoords="offset points", xytext=(0, 7),
                    ha="center", color=C.write, fontweight="semibold")
    for e in edges:
        ax.axvline(e, color="#E4E4E4", lw=0.5, zorder=0)
    rho, p = spearmanr(x, y)
    ax.set_xlabel(f"FV presence   cos(z, v_A) at the query cue @ {label}   "
                  f"(buckets of {width:g})")
    ax.set_ylabel("sampled exact-match accuracy (temperature 1.0)")
    ax.set_title(f"Accuracy vs FV presence, bucketed @ {label}\n"
                 f"all {N_TASKS_LABEL} tasks × n=0..6 pooled; point-level Spearman ρ={rho:+.2f} "
                 f"(p={p:.1e})")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return [(label, f"[{a:.2f},{b:.2f})", c, m, s) for a, b, c, m, s in rows]


def main():
    global LAYERS, N_TASKS_LABEL
    args = parse_args()
    split = json.load(open(args.split_path))
    split_tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    files = [args.in_root / f"{t}.npz" for t in split_tasks]
    missing = [f.stem for f in files if not f.exists()]
    assert not missing, f"missing {len(missing)} task files: {missing[:5]}"
    N_TASKS_LABEL = str(len(files))
    LAYERS = [int(l) for l in np.load(files[0], allow_pickle=False)["layers"]]
    lo, hi = (int(x) for x in args.band.split("-"))
    band_idx = [LAYERS.index(l) for l in range(lo, hi + 1) if l in LAYERS]
    band_lab = f"L{LAYERS[band_idx[0]]}-{LAYERS[band_idx[-1]]}"
    tasks, groups, cos_means, accs = [], [], [], []
    cos_max_means, cos_avg_means = [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        assert list(z["layers"]) == LAYERS and list(z["n_shots"]) == N_SHOTS
        tasks.append(f.stem)
        groups.append(str(z["group"]))
        cos = z["cos"]                                    # (7, N, n_layers)
        cos_means.append(cos.mean(axis=1))                # (7, n_layers)
        cb = cos[:, :, band_idx]
        cos_max_means.append(cb.max(axis=2).mean(axis=1))    # per-prompt max over band -> (7,)
        cos_avg_means.append(cb.mean(axis=2).mean(axis=1))   # per-prompt mean over band -> (7,)
        accs.append(z["match"].mean(axis=1))              # (7,)
    cos_means = np.stack(cos_means)                       # (T, 7, n_layers)
    cos_max_means, cos_avg_means = np.stack(cos_max_means), np.stack(cos_avg_means)
    accs = np.stack(accs)                                 # (T, 7)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "presence_vs_acc.npz",
             cos_by_layer=cos_means, cos_maxL=cos_max_means, cos_meanL=cos_avg_means,
             acc=accs, tasks=np.array(tasks), groups=np.array(groups),
             layers=np.array(LAYERS), n_shots=np.array(N_SHOTS), band=band_lab)

    # presence profile by layer (mean over tasks, per n) + headline layer = argmax at n=6
    prof = cos_means.mean(axis=0)                         # (7, n_layers)
    headline = LAYERS[int(np.argmax(prof[N_SHOTS.index(6)]))]
    (args.out_dir / "headline_layer.txt").write_text(f"{headline}\n")
    with open(args.out_dir / "presence_by_layer.csv", "w") as f:
        f.write("layer," + ",".join(f"mean_cos_n{n}" for n in N_SHOTS) + "\n")
        for li, l in enumerate(LAYERS):
            f.write(f"{l}," + ",".join(f"{prof[ni, li]:.4f}" for ni in range(len(N_SHOTS))) + "\n")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    cmap = plt.get_cmap("viridis")
    for ni, n in enumerate(N_SHOTS):
        ax.plot(LAYERS, prof[ni], "o-", ms=3.5, lw=1.5, color=cmap(ni / (len(N_SHOTS) - 1)), label=f"n={n}")
    ax.axvline(headline, color="0.4", ls=":", lw=1.2)
    y0, y1 = ax.get_ylim()
    ax.text(headline - 0.3, y0 + 0.03 * (y1 - y0), f"L{headline}\n(6-shot peak)", fontsize=9, va="bottom",
            ha="right", color="0.3")
    ax.set_xlabel("layer (block output) of the cue-token readout")
    ax.set_ylabel("mean cos(z, v̂_A) at the query cue")
    ax.set_title(f"FV presence profile by layer, {args.model_label} ({len(tasks)} tasks; mean over tasks and prompts)")
    ax.legend(title="shot count", fontsize=8, ncol=2)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(args.out_dir / "presence_by_layer.png")
    plt.close(fig)
    print(f"headline layer (argmax of 6-shot mean presence): L{headline}  "
          f"profile n=6: " + " ".join(f"L{l}:{prof[6, li]:.3f}" for li, l in enumerate(LAYERS)))

    if args.plot_layers == "all":
        plot_layers = list(LAYERS)
    elif args.plot_layers == "headline":
        plot_layers = [headline]
    else:
        plot_layers = [int(x) for x in args.plot_layers.split(",")]
    all_rows, bin_rows = [], []
    variants = [(f"L{l}", cos_means[:, :, li], f"L{l}", l in plot_layers) for li, l in enumerate(LAYERS)]
    variants += [(f"max{band_lab}", cos_max_means, "maxL", True), (f"mean{band_lab}", cos_avg_means, "meanL", True)]
    for label, x_tn, stem, do_plot in variants:
        if do_plot:
            all_rows += scatter_fig(x_tn, accs, groups, label, args.out_dir / f"scatter_{stem}.png")
            bin_rows += binned_fig(x_tn, accs, label, args.out_dir / f"binned_{stem}.png")
        else:   # correlations only (no figure) for the non-plotted layers
            for ni, n in enumerate(N_SHOTS):
                rho, rho_p = spearmanr(x_tn[:, ni], accs[:, ni]); r, r_p = pearsonr(x_tn[:, ni], accs[:, ni])
                all_rows.append((label, n, rho, rho_p, r, r_p))
            rho, rho_p = spearmanr(x_tn.ravel(), accs.ravel()); r, r_p = pearsonr(x_tn.ravel(), accs.ravel())
            all_rows.append((label, "pooled", rho, rho_p, r, r_p))

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
