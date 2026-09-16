#!/usr/bin/env python
"""Write-up figure: every convention family tested on Qwen2.5-7B base, k = 4 accuracy per pole (nat / alt), shaded by outcome:
kept (both poles ≥ .30), dropped (a pole < .30), or lexically identical (passed but excluded from the read/write study by design).
Full-corpus numbers where a family has them (200 texts / tasks), cheap-check numbers (45–60) otherwise.
Sources: <results>/summary.csv (text families with full corpora), lexical_selection.csv (all lexically diverse text families, full or cheap),
code/summary.csv (55 code families, full), code_selection.csv (5 code families dropped at the cheap check).
Output → <results>/writeup/task_accuracy_k4.{png,csv}"""
import csv, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.family_groups import FIXED
from src.sandbox.style_translation.families import FAMILY

CUT = 0.30


def main():
    R = model_paths("qwen25_base")["results"]; OUT = R / "writeup"; OUT.mkdir(exist_ok=True)
    rows = []
    # code
    full = pd.read_csv(R / "code" / "summary.csv"); full = full[full.k == 4].pivot(index="family", columns="style", values="accuracy")
    n_full = pd.read_csv(R / "code" / "summary.csv"); n_full = n_full[n_full.k == 4].groupby("family").n.first()
    sel = pd.read_csv(R / "code_selection.csv").set_index("family")
    for f, r in sel.iterrows():
        if f in full.index:
            rows.append(dict(family=f, group="code", lang=r.lang, nat=full.loc[f, "nat"], alt=full.loc[f, "alt"], n=int(n_full[f]), source="full", nat_label=r.nat_label, alt_label=r.alt_label))
        else:
            rows.append(dict(family=f, group="code", lang=r.lang, nat=r.nat, alt=r.alt, n=int(r.n), source="cheap", nat_label=r.nat_label, alt_label=r.alt_label))
    # free-form text: lexically diverse (English + non-English) from lexical_selection, lexically identical from summary.csv
    lex = pd.read_csv(R / "lexical_selection.csv")
    for _, r in lex.iterrows():
        rows.append(dict(family=r.family, group="text: lexically diverse (English)" if r.group == "English" else "text: lexically diverse (non-English)", lang=r.lang, nat=r.nat, alt=r.alt, n=int(r.n), source=r.source, nat_label=r.nat_label, alt_label=r.alt_label))
    s = pd.read_csv(R / "summary.csv"); s4 = s[s.k == 4].pivot(index="family", columns="style", values="accuracy"); n4 = s[s.k == 4].groupby("family").n.first()
    for f in FIXED:
        if f in s4.index:
            fam = FAMILY[f]
            rows.append(dict(family=f, group="text: lexically identical", lang="English", nat=s4.loc[f, "nat"], alt=s4.loc[f, "alt"], n=int(n4[f]), source="full", nat_label=fam.nat, alt_label=fam.alt))
    d = pd.DataFrame(rows); d["min"] = d[["nat", "alt"]].min(1)
    import json
    pool = set(json.load(open(R / "pool.json"))["pool"]) | set(json.load(open(R / "code_pool_full.json"))["pool"])
    d["status"] = np.where(d.group == "text: lexically identical", "lexically identical (excluded by design)",
                  np.where(d["min"] < CUT, "dropped (a pole below .30)", np.where(d.family.isin(pool), "kept", "passed the cheap check, not carried into the pool")))
    d.to_csv(OUT / "task_accuracy_k4.csv", index=False)
    # ---- figure: two rows (code / text), families sorted within group by min accuracy
    groups_row = [["code"], ["text: lexically diverse (English)", "text: lexically diverse (non-English)", "text: lexically identical"]]
    C = {"nat": "#1f77b4", "alt": "#d62728"}
    fig, axes = plt.subplots(2, 1, figsize=(24, 11))
    for ax, groups in zip(axes, groups_row):
        x = 0; ticks, labels = [], []
        for g in groups:
            sub = d[d.group == g].sort_values("min", ascending=False)
            x0 = x
            for _, r in sub.iterrows():
                kept = r.status == "kept"; ident = r.status.startswith("lexically"); late = r.status.startswith("passed")
                for j, pole in enumerate(("nat", "alt")):
                    ax.bar(x + (j - 0.5) * 0.4, r[pole], 0.4, color=C[pole], alpha=1.0 if kept else (0.45 if ident else (0.6 if late else 0.25)),
                           hatch=None if kept else ("//" if ident else ("oo" if late else "xx")), edgecolor=C[pole] if kept else "#555555", linewidth=0.5)
                ticks.append(x); labels.append(r.family + ("" if r.source == "full" else "*")); x += 1
            ax.axvline(x - 0.5, color="#999999", lw=1); ax.text((x0 + x - 1) / 2, 1.04, f"{g}  ({len(sub)} families)", ha="center", fontsize=10, fontweight="bold")
            x += 1
        ax.axhline(CUT, color="black", ls="dashed", lw=1)
        ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7); ax.set_xlim(-1, x - 1); ax.set_ylim(0, 1.1); ax.set_ylabel("accuracy at k = 4", fontsize=10); ax.grid(axis="y", alpha=0.3)
    handles = [Patch(color=C["nat"], label="natural convention (the model's default), 4 in-context examples"), Patch(color=C["alt"], label="alternative convention, 4 in-context examples"),
               Patch(facecolor="white", edgecolor="#555555", hatch="xx", label="dropped: a pole below .30"), Patch(facecolor="white", edgecolor="#555555", hatch="//", label="lexically identical: passed, excluded from the read/write study by design"),
               Patch(facecolor="white", edgecolor="#555555", hatch="oo", label="passed the cheap check in the late re-test of pruned ideas; not carried into the pool"),
               plt.Line2D([], [], color="black", ls="dashed", label="cutoff .30")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, fontsize=9.5, frameon=False)
    nk = int((d.status == "kept").sum()); nd = int(d.status.str.startswith("dropped").sum()); ni = int(d.status.str.startswith("lexically").sum()); nl = int(d.status.str.startswith("passed").sum())
    fig.suptitle(f"Every convention family tested on Qwen2.5-7B base — accuracy with 4 in-context examples, per pole ({len(d)} families: {nk} kept, {nd} dropped, {nl} passed late, {ni} lexically identical)\n"
                 "accuracy = uses the convention shown in context AND faithful translation / correct solution; * = cheap check on 45–60 items, others 200", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT / "task_accuracy_k4.png", dpi=150); plt.close(fig)
    print(d.groupby(["group", "status"]).size().to_string()); print("->", OUT / "task_accuracy_k4.png")


if __name__ == "__main__":
    main()
