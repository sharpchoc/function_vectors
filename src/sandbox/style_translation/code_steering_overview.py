#!/usr/bin/env python
"""Dumbbell overview of the code-convention steering results: per family and target pole, unsteered vs steered
(best layer / α, 95 % Wilson CI) with the 4-shot in-context accuracy as a reference marker.
Reads results/style_translation/<model>/<tag>/steering/best_config.csv, writes steered_vs_unsteered.png next to it."""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.code_families import CODE_SPECS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen25_base"); ap.add_argument("--tag", default="code")
    args = ap.parse_args()
    R = model_paths(args.model)["results"] / args.tag
    rows = list(csv.DictReader(open(R / "steering" / "best_config.csv")))
    labels = {}
    for spec in CODE_SPECS:                       # (name, language, nat label, alt label, ...)
        labels[(spec[0], "nat")] = f"{spec[2]} ({spec[1]})"; labels[(spec[0], "alt")] = f"{spec[3]} ({spec[1]})"
    fams = list(dict.fromkeys(r["family"] for r in rows))
    by = {(r["family"], r["target"]): r for r in rows}
    order = sorted(fams, key=lambda f: float(by[(f, "alt")]["accuracy"]))          # sort by steered accuracy toward alt
    C = {"nat": "#1f77b4", "alt": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 0.27 * len(fams) + 2.6))
    for ax, target in zip(axes, ("alt", "nat")):
        for i, f in enumerate(order):
            r = by[(f, target)]
            u, s, lo, hi = float(r["base_accuracy"]), float(r["accuracy"]), float(r["ci_lo"]), float(r["ci_hi"])
            k4 = r["step3_k4"]
            ax.plot([u, s], [i, i], color="#bbbbbb", lw=2, zorder=1)
            ax.plot(u, i, "o", color="#9e9e9e", ms=6, zorder=2)
            ax.errorbar(s, i, xerr=[[s - lo], [hi - s]], fmt="o", color=C[target], ms=6, capsize=2, lw=1, zorder=3)
            if k4 not in ("", None):
                ax.plot(float(k4), i, marker="|", color="black", ms=11, mew=1.6, zorder=4)
            ax.text(1.02, i, f"L{r['layer']}, α={float(r['alpha']):g}", va="center", fontsize=6.5, color="#555555")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"{f} → {labels.get((f, target), target)}" for f in order], fontsize=7.5)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-1, len(order))
        ax.set_xlabel("accuracy = target convention used AND correct solution (judge)", fontsize=9)
        ax.set_title({"alt": "steered toward the ALTERNATIVE convention (the pole the model does not default to)",
                      "nat": "steered toward the NATURAL convention"}[target], fontsize=10, color=C[target])
        ax.grid(axis="x", alpha=0.3); ax.axvline(0.5, color="#dddddd", lw=0.8)
    handles = [Line2D([], [], marker="o", color="#9e9e9e", ls="", ms=6, label="unsteered (same 0-shot prompt)"),
               Line2D([], [], marker="o", color="#444444", ls="", ms=6, label="steered at the cue token (best layer, α; 95% CI)"),
               Line2D([], [], marker="|", color="black", ls="", ms=11, mew=1.6, label="4 in-context examples, no steering")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"Zero-shot steering of coding conventions on Qwen2.5-7B base — {len(fams)} families, sorted by steered accuracy toward the alternative pole\n"
                 "row label = family → the convention steered toward; ≤ 200 tasks per point",
                 fontsize=11, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = R / "steering" / "steered_vs_unsteered.png"
    fig.savefig(out, dpi=150); plt.close(fig); print("->", out)


if __name__ == "__main__":
    main()
