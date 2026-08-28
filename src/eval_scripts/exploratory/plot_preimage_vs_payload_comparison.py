#!/usr/bin/env python
"""Direct comparison: original Stream W pre-image/FV ablation vs uniform d_payload-subspace
ablation (anchor mode, zero op) at k = 1, 2, 4, 8 (CPU).

All grids are 7-test-task means of the per-prompt delta log p (ablated - clean) of the first
answer token, on IDENTICAL prompts (both studies import the same Stream W build_prompts).
Layout: 2 rows (own direction/subspace | counterfactual) x 7 columns (preimage_matched,
preimage_icl10, fv, payload k1, k2, k4, k8), one shared symmetric RdBu_r scale.
Payload arms use the ZERO op — the like-for-like comparison with Stream W's projection
ablation (the mean-clamp arms exist in the source roots but are not shown here).
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.eval_scripts.exploratory.plot_oneshot_preimage_ablation import load_arm, render
from utils.paths import FV_FORMATION_DIR

AHM = FV_FORMATION_DIR / "ablation/attention_head_mechanisms"
TEST7 = ["landmark-country", "word_length", "capitalize_first_letter", "synonym",
         "lowercase_first_letter", "capitalize", "antonym"]

# columns: (title, root, own arm, cf arm)
COLUMNS = [
    ("pre-image\n(position-matched)",
     FV_FORMATION_DIR / "ablation/preimages/oneshot/main/train_varicl_top40",
     "preimage_matched", "preimage_matched_cf"),
    ("pre-image\n(icl10)",
     FV_FORMATION_DIR / "ablation/preimages/oneshot/main/train_varicl_top40",
     "preimage_icl10", "preimage_icl10_cf"),
    ("FV direction",
     FV_FORMATION_DIR / "ablation/preimages/oneshot/main/train_varicl_top40",
     "fv", "fv_cf"),
    ("payload subspace\nk=1", AHM / "test7_k_sweep/k1", "payload_zero", "payload_cf_zero"),
    ("payload subspace\nk=2", AHM / "test7_k_sweep/k2", "payload_zero", "payload_cf_zero"),
    ("payload subspace\nk=4", AHM / "test7", "payload_zero", "payload_cf_zero"),
    ("payload subspace\nk=8", AHM / "test7_k_sweep/k8", "payload_zero", "payload_cf_zero"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=TEST7)
    p.add_argument("--out_dir", type=Path, default=AHM / "comparison")
    p.add_argument("--propagated", action="store_true",
                   help="Use the PROPAGATED payload runs (token + all downstream tokens, "
                        "blocks b >= L) instead of anchor mode for the k columns.")
    return p.parse_args()


def task_mean_grid(root, arm, tasks):
    per_task, row_names = [], None
    for task in tasks:
        got = load_arm(root / task, arm)
        if got is None:
            print(f"[missing] {root.name}/{task}/{arm}")
            continue
        row_names, g = got
        per_task.append(g)
    if not per_task:
        return None, None
    return row_names, np.nanmean(np.stack(per_task), axis=0)


def main():
    args = parse_args()
    columns = list(COLUMNS)
    suffix = ""
    subtitle_mode = "anchor mode, project-to-0"
    if args.propagated:
        columns = COLUMNS[:3] + [
            (f"payload propagated\nk={k}", AHM / f"test7_propagated/k{k}",
             "payload_zero", "payload_cf_zero") for k in (1, 2, 4, 8)]
        suffix = "_propagated"
        subtitle_mode = ("payload columns PROPAGATED: site token + all downstream tokens, "
                         "project-to-0")
    grids = {}   # (row_kind, col_idx) -> (row_names, grid)
    for ci, (title, root, own_arm, cf_arm) in enumerate(columns):
        for kind, arm in (("own", own_arm), ("cf", cf_arm)):
            rn, g = task_mean_grid(root, arm, args.tasks)
            if g is not None:
                grids[(kind, ci)] = (rn, g)
    if not grids:
        raise SystemExit("no data found")
    vmax = max(np.nanmax(np.abs(g)) for _, g in grids.values())

    ncols = len(columns)
    fig, axes = plt.subplots(2, ncols, figsize=(3.45 * ncols + 1.5, 6.8), squeeze=False,
                             constrained_layout=True)
    fig.get_layout_engine().set(rect=[0, 0, 1, 0.88])   # headroom for group labels + title
    im = None
    for ci, (title, root, own_arm, cf_arm) in enumerate(columns):
        for r, kind in enumerate(("own", "cf")):
            ax = axes[r][ci]
            if (kind, ci) not in grids:
                ax.axis("off")
                continue
            rn, g = grids[(kind, ci)]
            panel_title = title if r == 0 else f"{title.splitlines()[-1]} — cf"
            im = render(ax, g, rn, vmax, panel_title, show_xlabel=(r == 1),
                        show_ylabels=(ci == 0))
            ax.set_title(panel_title, fontsize=9, pad=6)
            ax.tick_params(labelsize=7)
            if r == 1:
                ax.set_xlabel("start edit layer L", fontsize=8)
            ax.text(0.98, 0.04, f"min {np.nanmin(g):.2f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.2))
    fig.colorbar(im, ax=[a for row in axes for a in row], shrink=0.85, pad=0.005,
                 label="log p(ablated) − log p(clean)")
    fig.suptitle("1-shot projection-ablation, blocks b ≥ L — mean over "
                 f"{len(args.tasks)} test tasks ({subtitle_mode}); "
                 "top row = own direction, bottom = shuffled/cf",
                 fontsize=11, y=0.985)

    # shade the 'previous results' (Stream W) columns and label the two groups
    fig.canvas.draw()
    lx0 = axes[0][0].get_position().x0
    lx1 = axes[1][2].get_position().x1
    rx0 = axes[0][3].get_position().x0
    rx1 = axes[1][ncols - 1].get_position().x1
    y0 = axes[1][0].get_position().y0
    y1 = axes[0][0].get_position().y1
    pad = 0.006
    label_y = 0.915
    band = plt.Rectangle((lx0 - 2 * pad, y0 - 0.075), (lx1 - lx0) + 4 * pad,
                         label_y + 0.028 - (y0 - 0.075), transform=fig.transFigure,
                         facecolor="#dedcd5", edgecolor="none", zorder=-1)
    fig.add_artist(band)
    fig.text((lx0 + lx1) / 2, label_y,
             "PREVIOUS — pre-image / FV direction ablations (Stream W)",
             ha="center", fontsize=10.5, fontweight="bold", color="#454540")
    fig.text((rx0 + rx1) / 2, label_y,
             "NEW — attention-head d_payload subspace ablations",
             ha="center", fontsize=10.5, fontweight="bold", color="#29291f")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"heatmap_preimage_vs_payload_ksweep{suffix}.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

    # numeric companion: min-over-L per (column, row-site, own/cf)
    print(f"\n{'column':28s} {'kind':4s} {'cue1':>7s} {'target1':>8s} {'final_cue':>9s}")
    for ci, (title, *_rest) in enumerate(columns):
        for kind in ("own", "cf"):
            if (kind, ci) not in grids:
                continue
            rn, g = grids[(kind, ci)]
            mins = {str(r): float(np.nanmin(g[i])) for i, r in enumerate(rn)}
            t = title.replace("\n", " ")
            print(f"{t:28s} {kind:4s} {mins.get('cue1', float('nan')):7.2f} "
                  f"{mins.get('target1', float('nan')):8.2f} "
                  f"{mins.get('final_cue', float('nan')):9.2f}")


if __name__ == "__main__":
    main()
