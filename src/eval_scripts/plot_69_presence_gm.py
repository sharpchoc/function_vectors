#!/usr/bin/env python
"""Plot FV presence ABOVE the generic-FV baseline vs n-shot accuracy (capture_69_presence_gm.py).

Per task A and n in 0..6 (69 tasks x 7 = 483 points, NO binning): x = mean over the 150 paired
prompts of  delta_cos = cos(z, v_hat_A) - cos(z, v_hat_gm)  at the query cue (v_hat_gm = unit
equal-task-weighted mean of the 69 task FVs); y = temperature-1 sampled exact-match accuracy on
the same prompts (`match` from the original presence_vs_acc capture). Conventions follow
plot_69_presence_vs_acc.py: viridis by n, train circles / held-out triangles, per-n Spearman in
the legend, pooled Spearman + Pearson in the title, no fitted lines.

Outputs (TASK69_RUN_DIR/write_feature_and_model_accuracy/baseline_subtracted/):
  scatter_L13_minus_gm.png        x = delta_cos @ L13
  scatter_L13_cos_gm.png          x = cos(z, v_hat_gm) @ L13 (reference)
  scatter_meanL9-20_minus_gm.png  x = delta_cos averaged over layers 9..20 per prompt first
  presence_gm_L13.csv             per (task, group, n): mean cos_own, cos_gm, delta_cos, acc
  correlation_summary.csv         Spearman/Pearson per n + pooled, per variant
"""
import argparse
import csv
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
L13 = LAYERS.index(13)
YLABEL = "n-shot accuracy (T=1 sampled exact match)"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gm_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "presence_vs_acc_gm")
    p.add_argument("--acc_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "presence_vs_acc",
                   help="original capture; supplies `match` (accuracy) for the same prompts")
    p.add_argument("--out_dir", type=Path,
                   default=TASK69_RUN_DIR / "write_feature_and_model_accuracy" / "baseline_subtracted")
    return p.parse_args()


def scatter_fig(x_tn, acc_tn, groups, variant, xlabel, title, out_path):
    """x_tn, acc_tn: (n_tasks, 7). ONE panel with every (task, n) point, coloured by n."""
    fig, ax = plt.subplots(figsize=(9.5, 7.5), facecolor="white")
    ax.set_facecolor("white")
    rows = []
    tr = np.array([g == "train" for g in groups])
    cmap = plt.get_cmap("viridis")
    for ni, n in enumerate(N_SHOTS):
        x, y = x_tn[:, ni], acc_tn[:, ni]
        c = [cmap(ni / (len(N_SHOTS) - 1))]
        rho, rho_p = spearmanr(x, y)
        r, r_p = pearsonr(x, y)
        rows.append((variant, n, rho, rho_p, r, r_p))
        ax.scatter(x[tr], y[tr], s=20, color=c, label=f"n={n}  (ρ={rho:+.2f})")
        ax.scatter(x[~tr], y[~tr], s=32, color=c, marker="^")
    x_all, y_all = x_tn.ravel(), acc_tn.ravel()
    rho_all, p_all = spearmanr(x_all, y_all)
    r_all, rp_all = pearsonr(x_all, y_all)
    rows.append((variant, "pooled", rho_all, p_all, r_all, rp_all))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(YLABEL)
    ax.set_title(f"{title}\npooled Spearman ρ={rho_all:+.2f} (p={p_all:.1e}), "
                 f"Pearson r={r_all:+.2f}   [circles = train, triangles = held-out]", fontsize=11)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, title="shot count (per-n ρ)", loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)
    return rows


def main():
    args = parse_args()
    files = sorted(args.gm_root.glob("*.npz"))
    assert len(files) == 69, f"expected 69 task files in {args.gm_root}, found {len(files)}"
    tasks, groups = [], []
    own_l13, gm_l13, d_l13, d_band, accs = [], [], [], [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        assert list(z["layers"]) == LAYERS and list(z["n_shots"]) == N_SHOTS
        a = np.load(args.acc_root / f.name, allow_pickle=False)
        assert list(a["n_shots"]) == N_SHOTS and str(a["group"]) == str(z["group"])
        tasks.append(f.stem)
        groups.append(str(z["group"]))
        co, cg = z["cos_own"], z["cos_gm"]                 # (7, 150, 12)
        d = co - cg
        own_l13.append(co[:, :, L13].mean(axis=1))        # (7,)
        gm_l13.append(cg[:, :, L13].mean(axis=1))
        d_l13.append(d[:, :, L13].mean(axis=1))
        d_band.append(d.mean(axis=2).mean(axis=1))        # per-prompt mean over L9..20 -> (7,)
        accs.append(a["match"].mean(axis=1))              # (7,)
    own_l13, gm_l13, d_l13, d_band, accs = map(np.stack, (own_l13, gm_l13, d_l13, d_band, accs))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    xl_delta = "cos(z, v̂_A) − cos(z, v̂_gm) at the query cue, L13 (mean over 150 prompts)"
    variants = [
        ("L13_minus_gm", d_l13, xl_delta,
         "FV presence above generic-FV baseline vs accuracy @ L13 (69 tasks × n=0..6)"),
        ("L13_cos_gm", gm_l13,
         "cos(z, v̂_gm) at the query cue, L13 (mean over 150 prompts)",
         "Generic-FV presence vs accuracy @ L13 (69 tasks × n=0..6)"),
        ("meanL9-20_minus_gm", d_band,
         "cos(z, v̂_A) − cos(z, v̂_gm) at the query cue, mean over L9–20 (mean over 150 prompts)",
         "FV presence above generic-FV baseline vs accuracy, mean over L9–20 (69 tasks × n=0..6)"),
    ]
    rows = []
    for stem, x_tn, xlabel, title in variants:
        rows += scatter_fig(x_tn, accs, groups, stem, xlabel, title,
                            args.out_dir / f"scatter_{stem}.png")

    with open(args.out_dir / "correlation_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n_shots", "spearman_rho", "spearman_p", "pearson_r", "pearson_p"])
        for v, n, rho, rp, r, pp in rows:
            w.writerow([v, n, f"{rho:.4f}", f"{rp:.3e}", f"{r:.4f}", f"{pp:.3e}"])
    with open(args.out_dir / "presence_gm_L13.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "group", "n_shots", "mean_cos_own_L13", "mean_cos_gm_L13",
                    "mean_delta_cos_L13", "acc"])
        for ti, t in enumerate(tasks):
            for ni, n in enumerate(N_SHOTS):
                w.writerow([t, groups[ti], n, f"{own_l13[ti, ni]:.4f}", f"{gm_l13[ti, ni]:.4f}",
                            f"{d_l13[ti, ni]:.4f}", f"{accs[ti, ni]:.4f}"])

    # numeric summary
    def line(stem):
        per_n = " ".join(f"n={n}:{rho:+.2f}" for v, n, rho, *_ in rows if v == stem and n != "pooled")
        pooled = next(r for r in rows if r[0] == stem and r[1] == "pooled")
        return f"{stem:20s} pooled ρ={pooled[2]:+.3f} r={pooled[4]:+.3f} | per-n ρ: {per_n}"
    print(f"wrote {args.out_dir} ({len(variants)} figures, 2 csv)")
    for stem, *_ in variants:
        print(line(stem))
    print(f"cos_gm @L13 (task means) range {gm_l13.min():+.3f} .. {gm_l13.max():+.3f}; "
          f"delta_cos @L13 range {d_l13.min():+.3f} .. {d_l13.max():+.3f}")


if __name__ == "__main__":
    main()
