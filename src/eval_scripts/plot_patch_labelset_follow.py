"""
Plot the isolated label-token patching study (pure plotting, no GPU). Reads the summaries written by
patch_labelset_follow.py and renders combined_label_follow_bars.png: one panel per task pair, grouped
bars (baseline, demo2_prelabel open/isolated, both_labels open/isolated, target ceiling), with baseline
& ceiling reference lines and recovery% annotations. Shared y-limits across panels for comparability.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import LABEL_GEOMETRY_DIR

TASKS = ["antonym_synonym", "next_number_digits_prev_number_digits"]
TASK_LABEL = {"antonym_synonym": "antonym→synonym",
              "next_number_digits_prev_number_digits": "prev→next (digits)"}
SETS = ["demo2_prelabel", "both_labels"]
MODES = ["open", "isolated"]
CMAP = {"baseline": "#888888", "target_ceiling": "#55a868", "open": "#c7a0a2", "isolated": "#c44e52"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=str(LABEL_GEOMETRY_DIR / "twoshot" / "label_follow_patch"))
    p.add_argument("--regime", type=str, default="L6_and_above",
                   help="layer-regime subfolder under --root (e.g. L6_and_above, all_layers)")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root) / args.regime
    summaries = {t: json.load(open(root / f"{t}_summary.json")) for t in TASKS}

    # bar layout (shared across panels)
    order = ([("baseline", None)]
             + [(s, m) for s in SETS for m in MODES]
             + [("target_ceiling", None)])
    labels = (["baseline"]
              + [f"{s}\n{m}" for s in SETS for m in MODES]
              + ["target\nceiling"])
    colors = ([CMAP["baseline"]]
              + [CMAP[m] for s in SETS for m in MODES]
              + [CMAP["target_ceiling"]])

    # shared y-limits
    allv = []
    for t in TASKS:
        S = summaries[t]
        allv += [S["baseline_logit_diff"], S["ceiling_logit_diff"]]
        for s in SETS:
            for m in MODES:
                allv.append(S["conditions"][s][m]["mean_logit_diff"])
    lo, hi = min(allv), max(allv)
    pad = 0.12 * (hi - lo)
    ylim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(1, len(TASKS), figsize=(8.0 * len(TASKS), 5.4), squeeze=False)
    for c, t in enumerate(TASKS):
        ax = axes[0][c]
        S = summaries[t]
        base, ceil = S["baseline_logit_diff"], S["ceiling_logit_diff"]

        def value(nm, mode):
            if nm == "baseline":
                return base, None
            if nm == "target_ceiling":
                return ceil, None
            cond = S["conditions"][nm][mode]
            return cond["mean_logit_diff"], cond["recovery"]

        vals, recs = [], []
        for nm, mode in order:
            v, r = value(nm, mode)
            vals.append(v); recs.append(r)
        bars = ax.bar(range(len(vals)), vals, color=colors)
        ax.axhline(base, color=CMAP["baseline"], ls="--", lw=1, label="baseline (source prompt)")
        ax.axhline(ceil, color=CMAP["target_ceiling"], ls="--", lw=1, label="ceiling (target prompt)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8)
        if c == 0:
            ax.set_ylabel("mean logit(tgt_gold) − logit(src_gold)  @ query-final")
        ax.set_title(f"{TASK_LABEL[t]}  (n={S['n_pairs']})", fontsize=11)
        for b, r in zip(bars, recs):
            v = b.get_height()
            lab = f"{v:+.2f}" if r is None else f"{v:+.2f}\n({r*100:.0f}%)"
            ax.text(b.get_x() + b.get_width() / 2, v + (0.06 if v >= 0 else -0.06), lab,
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Do ONLY the demo label tokens drive the output?  "
                 "(isolated = all other non-output tokens pinned to base; % = recovery toward target)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = root / "figures" / "combined_label_follow_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
