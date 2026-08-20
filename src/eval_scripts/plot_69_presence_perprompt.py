#!/usr/bin/env python
"""FV presence vs correctness, one point PER PROMPT (no averaging over the 150 prompts).

Reads the per-task capture artifacts of capture_69_presence_vs_acc.py
(ARTIFACTS/69_task_run/presence_vs_acc/<task>.npz: cos (7,150,12) at the query cue,
match (7,150) sampled exact-match correctness). For a layer variant (L13 by default, plus
mean over L9-20) plots every (task, n, prompt) as one point: x = cos(z_l, v_hat_A) at the
query cue, y = correct (0/1, vertically jittered for visibility), coloured by shot count.
69 tasks x 7 n x 150 prompts = 72,450 points per figure.

Outputs (RESULTS/69_task_run/write_feature_and_model_accuracy/per_prompt/):
  scatter_<variant>_perprompt.png            points only
  scatter_<variant>_perprompt_with_rate.png  + sliding-window P(correct | cos) curve
  perprompt_<variant>.csv                    task, group, n, prompt, cos, correct
  correlation_summary.csv                    point-biserial r / Spearman per n and pooled
"""
import argparse
import csv
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
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR  # noqa: E402

LAYERS = list(range(9, 21))
N_SHOTS = list(range(0, 7))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "presence_vs_acc")
    p.add_argument("--out_dir", type=Path,
                   default=TASK69_RUN_DIR / "write_feature_and_model_accuracy" / "per_prompt")
    p.add_argument("--variants", nargs="+", default=["L13", "meanL9-20"])
    p.add_argument("--window", type=float, default=0.05, help="half-width of the sliding cos window")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_all(in_root):
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    group_of = {t: "train" for t in split["train_tasks"]}
    group_of.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(group_of)
    cos = np.zeros((len(tasks), len(N_SHOTS), 150, len(LAYERS)), np.float32)
    match = np.zeros((len(tasks), len(N_SHOTS), 150), bool)
    for ti, t in enumerate(tasks):
        d = np.load(in_root / f"{t}.npz")
        assert list(d["layers"]) == LAYERS and list(d["n_shots"]) == N_SHOTS
        cos[ti], match[ti] = d["cos"], d["match"]
    return tasks, np.array([group_of[t] for t in tasks]), cos, match


def variant_x(cos, variant):
    if variant.startswith("L") and variant[1:].isdigit():
        return cos[..., LAYERS.index(int(variant[1:]))]
    if variant == "meanL9-20":
        return cos.mean(axis=-1)
    raise ValueError(variant)


def sliding_rate(x, y, window, n_grid=200):
    grid = np.linspace(x.min(), x.max(), n_grid)
    rate = np.full(n_grid, np.nan)
    for i, g in enumerate(grid):
        m = np.abs(x - g) <= window
        if m.sum() >= 200:
            rate[i] = y[m].mean()
    return grid, rate


def scatter(x, y, n_of, label, out_path, with_rate, window, rng, rows):
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=150)
    fig.patch.set_facecolor("white")
    jitter = rng.uniform(-0.035, 0.035, size=y.shape)
    for ni, n in enumerate(N_SHOTS):
        m = n_of == n
        r_pb = pearsonr(x[m], y[m])[0]
        ax.scatter(x[m], y[m] + jitter[m], s=3, alpha=0.18, color=cmap(ni / (len(N_SHOTS) - 1)),
                   linewidths=0, rasterized=True, label=f"n={n}  (r={r_pb:+.2f})")
    if with_rate:
        grid, rate = sliding_rate(x, y, window)
        ax.plot(grid, rate, color="black", lw=2.0, zorder=5,
                label=f"P(correct | cos), sliding window ±{window:g}")
    r_all, _ = pearsonr(x, y)
    rho_all, _ = spearmanr(x, y)
    ax.set_xlabel(f"cos(z, v̂_A) at the query cue, {label} (one value per prompt)")
    ax.set_ylabel("prompt correct (T=1 sampled exact match), jittered")
    ax.set_yticks([0, 1], ["0 (wrong)", "1 (correct)"])
    ax.set_ylim(-0.12, 1.12)
    ax.set_title(f"FV presence vs per-prompt correctness @ {label}\n"
                 f"69 tasks × n=0..6 × 150 prompts = {x.size:,} points; "
                 f"pooled point-biserial r={r_all:+.2f}, ρ={rho_all:+.2f}", fontsize=10.5)
    leg = ax.legend(fontsize=8, title="shot count (per-n point-biserial r)", loc="center left",
                    bbox_to_anchor=(0.01, 0.5), markerscale=4, framealpha=0.9)
    for h in leg.legend_handles:
        try:
            h.set_alpha(1.0)
        except AttributeError:
            pass
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    rows.append([label, "pooled", x.size, round(r_all, 4), round(rho_all, 4)])
    for n in N_SHOTS:
        m = n_of == n
        rows.append([label, n, int(m.sum()), round(pearsonr(x[m], y[m])[0], 4),
                     round(spearmanr(x[m], y[m])[0], 4)])
    print(f"{label}: pooled r={r_all:+.3f} rho={rho_all:+.3f} -> {out_path.name}")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    tasks, groups, cos, match = load_all(args.in_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    T, N, P = match.shape
    task_idx = np.broadcast_to(np.arange(T)[:, None, None], (T, N, P)).ravel()
    n_of = np.broadcast_to(np.array(N_SHOTS)[None, :, None], (T, N, P)).ravel()
    p_idx = np.broadcast_to(np.arange(P)[None, None, :], (T, N, P)).ravel()
    y = match.ravel().astype(float)
    rows = [["variant", "n_shots", "n_points", "pointbiserial_r", "spearman_rho"]]
    for variant in args.variants:
        x = variant_x(cos, variant).ravel().astype(float)
        scatter(x, y, n_of, variant, args.out_dir / f"scatter_{variant}_perprompt.png",
                False, args.window, rng, [])
        scatter(x, y, n_of, variant, args.out_dir / f"scatter_{variant}_perprompt_with_rate.png",
                True, args.window, rng, rows)
        with open(args.out_dir / f"perprompt_{variant}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["task", "group", "n_shots", "prompt", f"cos_{variant}", "correct"])
            for ti, n, pi, xv, yv in zip(task_idx, n_of, p_idx, x, y):
                w.writerow([tasks[ti], groups[ti], n, pi, f"{xv:.5f}", int(yv)])
    with open(args.out_dir / "correlation_summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
