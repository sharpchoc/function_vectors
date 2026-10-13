#!/usr/bin/env python
"""Compare accuracy by k on the code-translation prompts (model key qwen25_code_tr, analyze.py output) with the original prompts (qwen25_code):
results/code_styles/translation_prompts/{comparison.csv, pooled_by_k.csv, accuracy_by_k_compare.png, README.md (hand-written)}."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "results/code_styles/summary.csv"; NEW = ROOT / "results/code_styles/translation_prompts/summary.csv"; OUT = NEW.parent


def main():
    o = pd.read_csv(OLD); n = pd.read_csv(NEW); fams = sorted(set(n.family))
    o = o[o.family.isin(fams)]
    cols = ["accuracy", "style_ok", "unscorable", "judge_ok", "capped", "n"]
    m = n.merge(o[["family", "style", "k"] + cols], on=["family", "style", "k"], suffixes=("_tr", "_orig"))
    m["d_accuracy"] = m.accuracy_tr - m.accuracy_orig; m.round(4).to_csv(OUT / "comparison.csv", index=False)
    P = m.groupby(["style", "k"])[["accuracy_tr", "accuracy_orig", "style_ok_tr", "style_ok_orig", "unscorable_tr", "unscorable_orig", "judge_ok_tr", "judge_ok_orig"]].mean().round(4)
    P.to_csv(OUT / "pooled_by_k.csv"); print(P.to_string())
    k4 = m[m["k"] == 4].groupby("style")[["accuracy_tr", "accuracy_orig"]].mean(); print(k4.round(3))
    fam4 = m[(m["k"] == 4)].pivot(index="family", columns="style", values=["accuracy_tr", "accuracy_orig"]).round(3); fam4.to_csv(OUT / "per_family_k4.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200, sharey=True)
    for ax, style, lab in zip(axes, ("nat", "alt"), ("natural-convention demonstrations", "alternative-convention demonstrations")):
        for src, col, ls, name in (("tr", "#1f6f8b", "-", "with source program"), ("orig", "#c2541b", "--", "original prompts")):
            g = m[m["style"] == style].groupby("k")[f"accuracy_{src}"].mean(); ax.plot(g.index, g.values, marker="o", lw=2.4, color=col, ls=ls, label=name)
        ax.set_title(lab, fontsize=11); ax.set_xlabel("k demonstrations"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.3); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("accuracy (convention followed and judge OK)"); axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle(f"Accuracy by k, {len(fams)} Python convention families, pooled over families", fontsize=12); fig.tight_layout(); fig.savefig(OUT / "accuracy_by_k_compare.png", facecolor="white")
    json.dump(dict(families=len(fams), k4=k4.round(4).to_dict(), pooled=P.reset_index().to_dict(orient="records")), open(OUT / "compare_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
