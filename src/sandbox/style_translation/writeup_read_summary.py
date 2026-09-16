#!/usr/bin/env python
"""Write-up figure: aggregated read-feature steering — mean over families of (a) the 3-shot prompt whose context shows the OTHER convention,
unsteered, (b) the same prompt with the read vector at its evidence tokens, (c) k = 4 accuracy with the target convention genuinely in context.
Two panels (code 55 / text 16), three groups each (nat context → alt, alt context → nat, pooled); error bars = 95 % CI of the mean across families.
Output → <results>/writeup/read_steering_summary.png"""
import sys, json
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


def main():
    R = model_paths("qwen25_base")["results"]; OUT = R / "writeup"
    d = pd.read_csv(OUT / "steerability_read.csv")
    k4 = {}
    for tag, path in (("code", R / "code" / "summary.csv"), ("text", R / "summary.csv")):
        s = pd.read_csv(path); s = s[s.k == 4]
        for _, r in s.iterrows():
            k4[(r.family, r.style)] = r.accuracy
    d["k4"] = [k4.get((f, t), np.nan) for f, t in zip(d.family, d.target)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    conds = [("3-shot prompt, other convention in context, unsteered", "unsteered_accuracy", "#9e9e9e"),
             ("same prompt, read vector at the evidence tokens", "accuracy", "#8e44ad"),
             ("4-shot prompt, target convention in context, no steering", "k4", "#1f77b4")]
    for ax, g in zip(axes, [g_ for g_ in d.group.unique() if g_.startswith("code")] + ["free-form text (16 lexically diverse families)"]):
        sub = d[d.group == g]
        groups = [("nat context\n→ alternative", sub[sub.direction == "nat2alt"]), ("alt context\n→ natural", sub[sub.direction == "alt2nat"]), ("both directions\npooled", sub)]
        w = 0.26
        for gi, (gname, sel) in enumerate(groups):
            for ci, (cname, key, col) in enumerate(conds):
                v = sel[key].dropna().values; m = v.mean(); e = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
                x = gi + (ci - 1) * w
                ax.bar(x, m, w, color=col, edgecolor="black", linewidth=0.6, yerr=e, capsize=3, label=cname if (gi == 0 and ax is axes[0]) else None)
                ax.text(x, m + e + 0.015, f"{m:.2f}", ha="center", fontsize=9)
        ax.set_xticks(range(3)); ax.set_xticklabels([g_ for g_, _ in groups], fontsize=9.5); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
        ax.set_title(g, fontsize=11, fontweight="bold")
    axes[0].set_ylabel("accuracy (mean over families)", fontsize=10)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Read-feature steering on Qwen2.5-7B base: overriding three in-context examples of the other convention\n"
                 "accuracy = target convention used AND faithful translation / correct solution; steered = best layer and α per family; error bars = 95% CI across families", fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(OUT / "read_steering_summary.png", dpi=150); plt.close(fig)
    for g, sub in d.groupby("group"):
        for name, sel in (("nat2alt", sub[sub.direction == "nat2alt"]), ("alt2nat", sub[sub.direction == "alt2nat"]), ("pooled", sub)):
            print(f"{g:48s} {name:8s}: unsteered {sel.unsteered_accuracy.mean():.2f} | steered {sel.accuracy.mean():.2f} | k=4 {sel.k4.mean():.2f}")
    print("->", OUT / "read_steering_summary.png")


if __name__ == "__main__":
    main()
