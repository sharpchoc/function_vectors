#!/usr/bin/env python
"""Write-up figure: write-feature (cue-token) steerability of every selected family on Qwen2.5-7B base.
Per family and target pole: unsteered 0-shot accuracy vs steered 0-shot accuracy (best layer / α, 95 % CI), with the 4-shot in-context
accuracy as a reference tick. Sources: <results>/steering/best_config.csv (16 text families) and <results>/code/steering/best_config.csv (55 code).
Output → <results>/writeup/steerability_write.{png,csv}"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths
import json

C = {"alt": "#d62728", "nat": "#1f77b4"}


def main():
    R = model_paths("qwen25_base")["results"]; OUT = R / "writeup"; OUT.mkdir(exist_ok=True)
    text = pd.read_csv(R / "steering" / "best_config.csv"); text["group"] = "free-form text (16 lexically diverse families)"
    text = text[text.family.isin(json.load(open(R / "pool.json"))["pool"])]
    code = pd.read_csv(R / "code" / "steering" / "best_config.csv"); code = code[code.family.isin(json.load(open(R / "code_pool_full.json"))["pool"])]; code["group"] = f"code ({code.family.nunique()} families)"
    d = pd.concat([code, text]); d["lift"] = d.accuracy - d.base_accuracy
    d.to_csv(OUT / "steerability_write.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(24, 11), gridspec_kw={"height_ratios": [1, 1]})
    for ax, g in zip(axes, [code.group.iloc[0], "free-form text (16 lexically diverse families)"]):
        sub = d[d.group == g]; order = sub[sub.target == "alt"].sort_values("accuracy", ascending=False).family.tolist()
        ticks, labels = [], []
        for i, f in enumerate(order):
            x = i * 2.4
            for j, t in enumerate(("alt", "nat")):
                r = sub[(sub.family == f) & (sub.target == t)].iloc[0]; xo = x + j * 1.0
                ax.bar(xo, r.base_accuracy, 0.42, color=C[t], alpha=0.3, edgecolor="none")
                ax.bar(xo + 0.44, r.accuracy, 0.42, color=C[t], edgecolor="black", linewidth=0.4,
                       yerr=[[r.accuracy - r.ci_lo], [r.ci_hi - r.accuracy]], capsize=1.5, error_kw=dict(lw=0.6))
                if not np.isnan(r.step3_k4):
                    ax.plot([xo - 0.25, xo + 0.7], [r.step3_k4, r.step3_k4], color="black", lw=1.2)
            ticks.append(x + 0.7); labels.append(f)
        ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7.5); ax.set_xlim(-0.8, len(order) * 2.4 - 0.5); ax.set_ylim(0, 1.05)
        ax.set_ylabel("accuracy, 0-shot prompt", fontsize=10); ax.grid(axis="y", alpha=0.3)
        m = sub.groupby("target")[["base_accuracy", "accuracy"]].mean()
        ax.set_title(f"{g} — mean over families: → alt {m.loc['alt', 'base_accuracy']:.2f} unsteered → {m.loc['alt', 'accuracy']:.2f} steered; → nat {m.loc['nat', 'base_accuracy']:.2f} → {m.loc['nat', 'accuracy']:.2f}", fontsize=10.5, fontweight="bold")
    handles = [Patch(color=C["alt"], alpha=0.3, label="→ alternative: unsteered"), Patch(color=C["alt"], label="→ alternative: steered (best layer, α; 95% CI)"),
               Patch(color=C["nat"], alpha=0.3, label="→ natural: unsteered"), Patch(color=C["nat"], label="→ natural: steered (best layer, α; 95% CI)"),
               Line2D([], [], color="black", lw=1.2, label="reference: 4 in-context examples, no steering")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=5, fontsize=9.5, frameon=False)
    fig.suptitle("Write-feature steerability on Qwen2.5-7B base — a cue-token mean-difference vector (nat − alt, L24 for most families) added to a 0-shot prompt\n"
                 "accuracy = target convention used AND faithful translation / correct solution; ≤ 200 items per bar; families sorted by steered accuracy toward the alternative pole", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT / "steerability_write.png", dpi=150); plt.close(fig)
    for g, sub in d.groupby("group"):
        for t, s in sub.groupby("target"):
            print(f"{g:48s} → {t}: unsteered {s.base_accuracy.mean():.2f} → steered {s.accuracy.mean():.2f} | 4-shot ref {s.step3_k4.mean():.2f} | steered ≥ .5: {(s.accuracy >= .5).sum()}/{len(s)} | lift ≥ .2: {(s.lift >= .2).sum()}/{len(s)}")
    print("->", OUT / "steerability_write.png")


if __name__ == "__main__":
    main()
