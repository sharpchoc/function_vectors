#!/usr/bin/env python
"""Write-up figure: aggregated write-feature steering — mean over families of (a) the 0-shot prompt unsteered, (b) the same prompt with the
cue-token vector at the family's best layer / α, (c) k = 3 accuracy with the target convention in context (no steering).
Two panels (code 55 / text 16), three groups each (→ alternative, → natural, pooled); error bars = 95 % CI of the mean across families.
Output → <results>/writeup/write_steering_summary.png"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths

K_REF = 3


def main():
    R = model_paths("qwen25_base")["results"]; OUT = R / "writeup"
    d = pd.read_csv(OUT / "steerability_write.csv")
    ref = {}
    for path in (R / "code" / "summary.csv", R / "summary.csv"):
        s = pd.read_csv(path); s = s[s.k == K_REF]
        for _, r in s.iterrows():
            ref[(r.family, r.style)] = r.accuracy
    d["k_ref"] = [ref.get((f, t), np.nan) for f, t in zip(d.family, d.target)]
    conds = [("0-shot prompt, unsteered", "base_accuracy", "#9e9e9e"), ("0-shot prompt, write vector at the cue token", "accuracy", "#d62728"),
             (f"{K_REF}-shot prompt, target convention in context, no steering", "k_ref", "#1f77b4")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    for ax, g in zip(axes, ["code (55 families)", "free-form text (16 lexically diverse families)"]):
        sub = d[d.group == g]
        groups = [("→ alternative\nconvention", sub[sub.target == "alt"]), ("→ natural\nconvention", sub[sub.target == "nat"]), ("both poles\npooled", sub)]
        w = 0.26
        for gi, (gname, sel) in enumerate(groups):
            for ci, (cname, key, col) in enumerate(conds):
                v = sel[key].dropna().values; m = v.mean(); e = 1.96 * v.std(ddof=1) / np.sqrt(len(v)); x = gi + (ci - 1) * w
                ax.bar(x, m, w, color=col, edgecolor="black", linewidth=0.6, yerr=e, capsize=3, label=cname if (gi == 0 and ax is axes[0]) else None)
                ax.text(x, m + e + 0.015, f"{m:.2f}", ha="center", fontsize=9)
        ax.set_xticks(range(3)); ax.set_xticklabels([n for n, _ in groups], fontsize=9.5); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
        ax.set_title(g, fontsize=11, fontweight="bold")
    axes[0].set_ylabel("accuracy (mean over families)", fontsize=10)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Write-feature steering on Qwen2.5-7B base: a cue-token mean-difference vector on a 0-shot prompt\n"
                 "accuracy = target convention used AND faithful translation / correct solution; steered = best layer and α per family; error bars = 95% CI across families", fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(OUT / "write_steering_summary.png", dpi=150); plt.close(fig)
    for g, sub in d.groupby("group"):
        for name, sel in (("alt", sub[sub.target == "alt"]), ("nat", sub[sub.target == "nat"]), ("pooled", sub)):
            print(f"{g:48s} {name:6s}: unsteered {sel.base_accuracy.mean():.2f} | steered {sel.accuracy.mean():.2f} | k={K_REF} {sel.k_ref.mean():.2f}")
    print("->", OUT / "write_steering_summary.png")


if __name__ == "__main__":
    main()
