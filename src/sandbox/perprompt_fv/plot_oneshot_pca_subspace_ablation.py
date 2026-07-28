#!/usr/bin/env python
"""SANDBOX: figures for the top-k PCA pre-image subspace ablation study (CPU).

Reads the per-(task, arm) npz written by ablate_oneshot_pca_subspace_logprob.py and renders:

  figures/heatmap_all_arms__{zero,mean}.png
      one grid per op: arm rows (matched, matched_cf, icl10, icl10_cf) x k columns
      (0, 2, 3, 4); each panel is the Stream W-style token-row x start-layer heatmap of the
      task-mean delta log p. Color scale is PER OP (zero-op damage is ~15x the mean-op's;
      a shared scale blanks the mean figure) — the scale is printed in each suptitle and the
      ktrend figure carries the cross-op comparison on one axis.
  figures/ktrend_summary.png
      per site token: min-over-start-layers task-mean delta log p vs k, lines per
      (op, base, cf); horizontal reference lines = Stream W single-direction arms
      (preimage_matched / preimage_icl10 / fv, same statistic) when available.

Grid/summary PNGs only (repo figure policy).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.paths import FV_FORMATION_DIR, RESULTS_ROOT  # noqa: E402

KS = [0, 2, 3, 4]
OPS = ["zero", "mean"]
BASES = ["matched", "matched_cf", "icl10", "icl10_cf"]
BASE_TITLES = {
    "matched": "position-matched cells",
    "matched_cf": "position-matched — counterfactual task",
    "icl10": "icl10 cells",
    "icl10_cf": "icl10 — counterfactual task",
}
OP_TITLES = {"zero": "zero op: remove the subspace component",
             "mean": "mean op: clamp to the all-task grand-mean pre-image component"}
K_TITLES = {0: "k=0 (task-mean dir only)", 2: "k=2 (+top-2 PCs)",
            3: "k=3 (+top-3 PCs)", 4: "k=4 (+top-4 PCs)"}
ROW_TITLES = {"cue1": "cue1 (demo 'A:')", "target1": "target1 (demo label)",
              "final_cue": "final cue (query 'A:')"}
STREAMW_ARMS = {"preimage_matched": "Stream W preimage (matched)",
                "preimage_icl10": "Stream W preimage (icl10)",
                "fv": "Stream W FV direction"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path,
                   default=RESULTS_ROOT / "sandbox/perprompt_ridge_pilot/oneshot_pca_subspace_ablation")
    p.add_argument("--streamw_root", type=Path,
                   default=FV_FORMATION_DIR / "oneshot_preimage_ablation/train_varicl_top40")
    return p.parse_args()


def arm_name(base, k, op):
    b, cf = (base[:-3], "_cf") if base.endswith("_cf") else (base, "")
    return f"pcasub_{b}{cf}_k{k}_{op}"


def task_mean_grids(root, arms):
    """arm -> (row_names, [rows, 28] mean-over-tasks grid, n_tasks); tasks = dirs with npz."""
    task_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name != "figures")
    out = {}
    for arm in arms:
        grids, row_names = [], None
        for d in task_dirs:
            f = d / f"{arm}_delta_logp.npz"
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=False)
            with np.errstate(invalid="ignore"):
                grids.append(np.nanmean(z["delta_logp"], axis=2))
            row_names = [str(r) for r in z["row_names"]]
        if grids:
            out[arm] = (row_names, np.nanmean(np.stack(grids), axis=0), len(grids))
    return out


def main():
    args = parse_args()
    fig_dir = args.root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    arms = [arm_name(b, k, op) for b in BASES for k in KS for op in OPS]
    grids = task_mean_grids(args.root, arms)
    if not grids:
        raise SystemExit(f"no npz under {args.root}")
    n_tasks = max(n for _, _, n in grids.values())

    # --- heatmap grids: one figure per op, arm rows x k columns (per-op scale) ---
    for op in OPS:
        vmax = max(np.nanmax(np.abs(g)) for a, (_, g, _) in grids.items()
                   if a.endswith(f"_{op}"))
        print(f"{op}: scale vmax={vmax:.3f}; {n_tasks} tasks")
        fig, axes = plt.subplots(len(BASES), len(KS),
                                 figsize=(4.6 * len(KS), 2.1 * len(BASES) + 1.2),
                                 squeeze=False, constrained_layout=True)
        last = None
        for bi, base in enumerate(BASES):
            for ki, k in enumerate(KS):
                ax = axes[bi][ki]
                got = grids.get(arm_name(base, k, op))
                if got is None:
                    ax.axis("off")
                    continue
                row_names, g, _ = got
                last = ax.imshow(g, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                                 interpolation="nearest")
                ax.set_yticks(range(len(row_names)))
                ax.set_yticklabels([ROW_TITLES.get(r, r) for r in row_names]
                                   if ki == 0 else [], fontsize=8)
                ax.set_xticks(range(0, g.shape[1], 8))
                ax.tick_params(labelsize=7)
                if bi == 0:
                    ax.set_title(K_TITLES[k], fontsize=10)
                if bi == len(BASES) - 1:
                    ax.set_xlabel("start edit layer L (ablate h.b, b ≥ L)", fontsize=8)
                if ki == 0:
                    ax.set_ylabel(BASE_TITLES[base], fontsize=8)
        fig.suptitle("SANDBOX 1-shot ablation of per-prompt pre-image PCA SUBSPACES "
                     f"(span{{task-mean, top-k PCs}})\n{OP_TITLES[op]} — "
                     f"mean Δ log p(correct), {n_tasks} tasks (scale ±{vmax:.2f}, per op)",
                     fontsize=13)
        fig.colorbar(last, ax=axes, label="log p(ablated) − log p(clean)", shrink=0.7)
        out = fig_dir / f"heatmap_all_arms__{op}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"wrote {out}")

    # --- k-trend summary: min over start layers of the task-mean grid, per site row ---
    ref = task_mean_grids(args.streamw_root, list(STREAMW_ARMS)) if args.streamw_root.exists() else {}
    row_names = next(iter(grids.values()))[0]
    colors = {"matched": "#2a78d6", "icl10": "#d1495b"}
    fig, axes = plt.subplots(len(OPS), len(row_names),
                             figsize=(4.4 * len(row_names), 3.4 * len(OPS)),
                             squeeze=False, sharex=True)
    for oi, op in enumerate(OPS):
        for ri, row in enumerate(row_names):
            ax = axes[oi][ri]
            for base in BASES:
                b = base[:-3] if base.endswith("_cf") else base
                is_cf = base.endswith("_cf")
                ys = [grids[arm_name(base, k, op)][1][ri].min()
                      if arm_name(base, k, op) in grids else np.nan for k in KS]
                ax.plot(KS, ys, marker="o", ms=4, lw=1.2 if is_cf else 2,
                        ls="--" if is_cf else "-", color=colors[b],
                        alpha=0.45 if is_cf else 1.0,
                        label=BASE_TITLES[base] if oi == 0 and ri == 0 else None)
            for (ra, rlabel), c in zip(STREAMW_ARMS.items(), ["#2a78d6", "#d1495b", "#6b6a60"]):
                if ra in ref:
                    ax.axhline(ref[ra][1][ri].min(), color=c, lw=1, ls=":", alpha=0.8,
                               label=rlabel if oi == 0 and ri == 0 else None)
            ax.set_title(f"{OP_TITLES[op].split(':')[0]} — {ROW_TITLES.get(row, row)}", fontsize=10)
            ax.set_xticks(KS)
            ax.grid(True, color="#e5e4dc", lw=0.7)
            ax.set_axisbelow(True)
            if oi == len(OPS) - 1:
                ax.set_xlabel("k (top-k centered PCs; subspace dim = k+1)", fontsize=9)
            if ri == 0:
                ax.set_ylabel("min over L of task-mean Δ log p", fontsize=9)
    fig.legend(loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("SANDBOX subspace-ablation damage vs k (most-damaging start layer; "
                 f"{n_tasks}-task mean; dotted = Stream W single-direction reference)", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    out = fig_dir / "ktrend_summary.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
