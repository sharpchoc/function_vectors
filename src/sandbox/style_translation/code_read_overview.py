#!/usr/bin/env python
"""Overview figures for the code-convention READ-feature test (evidence-token steering), per prompt k:
  summary_bars.png        — mean over families: unsteered k-shot, evidence-steered k-shot (best layer / α), flipped reference
                            (k-shot prompt whose context genuinely shows the target), per direction and pooled
  steered_vs_unsteered.png — one row per family and direction: unsteered → steered (95 % CI) with the reference tick
Reads results/style_translation/<model>/<tag>/read_steer[_k<k>]/best_config.csv, writes next to it."""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.code_families import CODE_SPECS

C = {"nat2alt": "#d62728", "alt2nat": "#1f77b4"}
TITLE = {"nat2alt": "nat context, steered toward the ALTERNATIVE convention", "alt2nat": "alt context, steered toward the NATURAL convention"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen25_base"); ap.add_argument("--tag", default="code"); ap.add_argument("--k_ctx", type=int, default=3)
    args = ap.parse_args()
    k = args.k_ctx
    R = model_paths(args.model)["results"] / args.tag / ("read_steer" if k == 3 else f"read_steer_k{k}")
    rows = [r for r in csv.DictReader(open(R / "best_config.csv")) if r["reference"] not in ("", None)]
    labels = {}
    for spec in CODE_SPECS:
        labels[(spec[0], "nat")] = f"{spec[2]} ({spec[1]})"; labels[(spec[0], "alt")] = f"{spec[3]} ({spec[1]})"
    fams = list(dict.fromkeys(r["family"] for r in rows))
    by = {(r["family"], r["direction"]): r for r in rows}

    # ---- bars ----
    groups = [(TITLE["nat2alt"].replace(", ", "\n"), [r for r in rows if r["direction"] == "nat2alt"]),
              (TITLE["alt2nat"].replace(", ", "\n"), [r for r in rows if r["direction"] == "alt2nat"]), ("both directions pooled", rows)]
    conds = [(f"unsteered {k}-shot prompt", "unsteered_accuracy", "#9e9e9e"), (f"evidence tokens steered, {k}-shot prompt", "accuracy", "#8e44ad"),
             (f"reference: {k}-shot prompt whose context genuinely shows the target", "reference", "#1f77b4")]
    fig, ax = plt.subplots(figsize=(10, 5.2)); w = 0.26
    for gi, (gname, sel) in enumerate(groups):
        for ci, (cname, key, col) in enumerate(conds):
            v = np.array([float(r[key]) for r in sel]); m = v.mean(); e = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
            x = gi + (ci - 1) * w
            ax.bar(x, m, w, color=col, edgecolor="black", linewidth=0.6, yerr=e, capsize=3, label=cname if gi == 0 else None)
            ax.text(x, m + e + 0.015, f"{m:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g for g, _ in groups], fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy (mean over families)", fontsize=10); ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.set_title(f"Read-feature test on Qwen2.5-7B base — {len(fams)} coding-convention families, {k}-shot prompts\n"
                 "read vector (alt − nat evidence-token mean) added at the in-context evidence tokens only\n"
                 "accuracy = target convention used AND correct solution; "
                 "steered = best layer and α per family; error bars = 95% CI of the mean across families", fontsize=9.5)
    fig.tight_layout(); fig.savefig(R / "summary_bars.png", dpi=150); plt.close(fig)

    # ---- dumbbell ----
    order = sorted(fams, key=lambda f: float(by[(f, "nat2alt")]["accuracy"]) if (f, "nat2alt") in by else 0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 0.27 * len(fams) + 2.8))
    for ax, d in zip(axes, ("nat2alt", "alt2nat")):
        target = "alt" if d == "nat2alt" else "nat"
        for i, f in enumerate(order):
            r = by.get((f, d))
            if r is None:
                continue
            u, s, lo, hi, ref = (float(r[x]) for x in ("unsteered_accuracy", "accuracy", "ci_lo", "ci_hi", "reference"))
            ax.plot([u, s], [i, i], color="#bbbbbb", lw=2, zorder=1); ax.plot(u, i, "o", color="#9e9e9e", ms=6, zorder=2)
            ax.errorbar(s, i, xerr=[[s - lo], [hi - s]], fmt="o", color=C[d], ms=6, capsize=2, lw=1, zorder=3)
            ax.plot(ref, i, marker="|", color="black", ms=11, mew=1.6, zorder=4)
            ax.text(1.02, i, f"L{r['layer']}, α={float(r['alpha']):g}", va="center", fontsize=6.5, color="#555555")
        ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{f} → {labels.get((f, target), target)}" for f in order], fontsize=7.5)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-1, len(order)); ax.set_xlabel("accuracy = target convention used AND correct solution (judge)", fontsize=9)
        ax.set_title(TITLE[d], fontsize=10, color=C[d]); ax.grid(axis="x", alpha=0.3); ax.axvline(0.5, color="#dddddd", lw=0.8)
    handles = [Line2D([], [], marker="o", color="#9e9e9e", ls="", ms=6, label=f"unsteered {k}-shot prompt (rate of the other convention)"),
               Line2D([], [], marker="o", color="#444444", ls="", ms=6, label="evidence tokens steered with the read vector (best layer, α; 95% CI)"),
               Line2D([], [], marker="|", color="black", ls="", ms=11, mew=1.6, label=f"reference: {k}-shot prompt whose context genuinely shows the target")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"Evidence-token (read-feature) steering of {k}-shot code prompts on Qwen2.5-7B base — {len(fams)} families, sorted by steered accuracy toward the alternative pole\n"
                 "row label = family → the convention steered toward; ≤ 200 tasks per point", fontsize=11, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.955)); fig.savefig(R / "steered_vs_unsteered.png", dpi=150); plt.close(fig)
    print("->", R / "summary_bars.png", R / "steered_vs_unsteered.png")


if __name__ == "__main__":
    main()
