#!/usr/bin/env python
"""Write-up figure: read-feature (evidence-token) steering of every selected family on Qwen2.5-7B base, k = 3 prompts.
Per family and direction: the k = 3 prompt unsteered (rate of the OTHER convention) vs steered at its evidence tokens with the read vector
(best layer / α, 95 % CI), with the accuracy of a real k = 3 prompt whose context genuinely shows the target as the reference tick.
Sources: <results>/read_steer/best_config.csv (16 text families) and <results>/code/read_steer/best_config.csv (55 code families).
Output → <results>/writeup/steerability_read.{png,csv}"""
import sys, json
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

C = {"nat2alt": "#d62728", "alt2nat": "#1f77b4"}


def main():
    R = model_paths("qwen25_base")["results"]; OUT = R / "writeup"; OUT.mkdir(exist_ok=True)
    text = pd.read_csv(R / "read_steer" / "best_config.csv"); text = text[text.family.isin(json.load(open(R / "pool.json"))["pool"])]; text["group"] = "free-form text (16 lexically diverse families)"
    code = pd.read_csv(R / "code" / "read_steer" / "best_config.csv"); code["group"] = "code (55 families)"
    d = pd.concat([code, text]); d.to_csv(OUT / "steerability_read.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(24, 11))
    for ax, g in zip(axes, ["code (55 families)", "free-form text (16 lexically diverse families)"]):
        sub = d[d.group == g]; order = sub[sub.direction == "nat2alt"].sort_values("accuracy", ascending=False).family.tolist()
        ticks, labels = [], []
        for i, f in enumerate(order):
            x = i * 2.4
            for j, dr in enumerate(("nat2alt", "alt2nat")):
                r = sub[(sub.family == f) & (sub.direction == dr)].iloc[0]; xo = x + j * 1.0
                ax.bar(xo, r.unsteered_accuracy, 0.42, color=C[dr], alpha=0.3, edgecolor="none")
                ax.bar(xo + 0.44, r.accuracy, 0.42, color=C[dr], edgecolor="black", linewidth=0.4,
                       yerr=[[r.accuracy - r.ci_lo], [r.ci_hi - r.accuracy]], capsize=1.5, error_kw=dict(lw=0.6))
                if not np.isnan(r.reference):
                    ax.plot([xo - 0.25, xo + 0.7], [r.reference, r.reference], color="black", lw=1.2)
            ticks.append(x + 0.7); labels.append(f)
        ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7.5); ax.set_xlim(-0.8, len(order) * 2.4 - 0.5); ax.set_ylim(0, 1.05)
        ax.set_ylabel("accuracy toward the target, 3-shot prompt", fontsize=10); ax.grid(axis="y", alpha=0.3)
        m = sub.groupby("direction")[["unsteered_accuracy", "accuracy", "reference"]].mean()
        ax.set_title(f"{g} — mean over families: nat context → alt {m.loc['nat2alt', 'unsteered_accuracy']:.2f} unsteered → {m.loc['nat2alt', 'accuracy']:.2f} steered (real alt context {m.loc['nat2alt', 'reference']:.2f}); "
                     f"alt context → nat {m.loc['alt2nat', 'unsteered_accuracy']:.2f} → {m.loc['alt2nat', 'accuracy']:.2f} (real nat context {m.loc['alt2nat', 'reference']:.2f})", fontsize=10, fontweight="bold")
    handles = [Patch(color=C["nat2alt"], alpha=0.3, label="nat context → alt: unsteered 3-shot prompt"), Patch(color=C["nat2alt"], label="nat context → alt: read vector at the evidence tokens (best layer, α; 95% CI)"),
               Patch(color=C["alt2nat"], alpha=0.3, label="alt context → nat: unsteered"), Patch(color=C["alt2nat"], label="alt context → nat: steered"),
               Line2D([], [], color="black", lw=1.2, label="reference: a real 3-shot prompt whose context shows the target, no steering")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, fontsize=9.5, frameon=False)
    fig.suptitle("Read-feature steerability on Qwen2.5-7B base — the evidence-token mean-difference vector added at the in-context evidence tokens of a 3-shot prompt showing the OTHER convention\n"
                 "accuracy = target convention at the next decision AND faithful translation / correct solution; ≤ 200 items per bar; families sorted by steered accuracy toward the alternative pole", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT / "steerability_read.png", dpi=150); plt.close(fig)
    for g, sub in d.groupby("group"):
        for dr, s in sub.groupby("direction"):
            print(f"{g:48s} {dr}: unsteered {s.unsteered_accuracy.mean():.2f} → steered {s.accuracy.mean():.2f} | reference {s.reference.mean():.2f} | reach≥.5: {int(s.pass_50.sum())}/{len(s)} | steered ≥ .5: {(s.accuracy >= .5).sum()}/{len(s)}")
    print("->", OUT / "steerability_read.png")


if __name__ == "__main__":
    main()
