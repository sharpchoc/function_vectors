#!/usr/bin/env python
"""Selection chart (user request 2026-09-14): every LEXICALLY DIVERSE convention family tried on Qwen2.5-7B base, k = 4
accuracy for both context poles, the 30 % both-poles cutoff, and the accept / reject verdict.
Sources: results/style_translation/qwen25_base/summary.csv (English, 200 texts) and
results/style_translation/qwen25_base/multilingual_k4/k4_check.csv (non-English cheap check, 45–60 texts).
Output: results/style_translation/qwen25_base/lexical_selection.{png,csv}."""
import csv, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.family_groups import LEXICAL
from src.sandbox.style_translation.families import FAMILY
from src.sandbox.style_translation.ml_families import ML_FAMILIES

MP = model_paths("qwen25_base"); CUT = 0.30
en = list(csv.DictReader(open(MP["results"] / "summary.csv")))
ml = list(csv.DictReader(open(MP["results"] / "multilingual_k4" / "k4_check.csv")))


def get(rows, fam, style, key="accuracy"):
    r = [x for x in rows if x["family"] == fam and x["style"] == style and int(float(x["k"])) == 4]
    return (float(r[0][key]), int(r[0]["n"])) if r else (np.nan, 0)


items = []
for f in LEXICAL:
    (a, n), (b, _) = get(en, f, "nat"), get(en, f, "alt")
    if np.isnan(a): continue
    items.append(dict(family=f, group="English", lang="English", nat=a, alt=b, n=n, nat_label=FAMILY[f].nat, alt_label=FAMILY[f].alt))
for fm in ML_FAMILIES:
    (a, n), (b, _) = get(ml, fm.name, "nat"), get(ml, fm.name, "alt")
    if np.isnan(a): continue
    items.append(dict(family=fm.name, group="non-English", lang=fm.tgt_lang, nat=a, alt=b, n=n, nat_label=fm.nat, alt_label=fm.alt))
for it in items:
    it["min"] = min(it["nat"], it["alt"]); it["accept"] = it["min"] >= CUT
items.sort(key=lambda it: (it["group"] != "English", -it["min"]))
with open(MP["results"] / "lexical_selection.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(items[0])); w.writeheader(); w.writerows(items)

fig, ax = plt.subplots(figsize=(max(14, 0.62 * len(items)), 7.0))
x = np.arange(len(items)); w = 0.38
for i, it in enumerate(items):
    if not it["accept"]:
        ax.axvspan(i - 0.5, i + 0.5, color="#f3d6d6", alpha=0.6, zorder=0)
n_en = sum(it["group"] == "English" for it in items)
ax.axvline(n_en - 0.5, color="k", lw=1)
ax.bar(x - w / 2, [it["nat"] for it in items], w, color="#1f77b4", label="natural convention in context (k = 4)")
ax.bar(x + w / 2, [it["alt"] for it in items], w, color="#d62728", label="alternative convention in context (k = 4)")
ax.axhline(CUT, color="k", ls="--", lw=1.2, label=f"cutoff: both poles ≥ {CUT:.0%}")
for i, it in enumerate(items):
    ax.text(i, 1.01, "✓" if it["accept"] else "✗", ha="center", va="bottom", fontsize=13, color="#2a7d2a" if it["accept"] else "#b22222", fontweight="bold")
    ax.text(i, -0.045, f"n={it['n']}", ha="center", va="top", fontsize=6.5, color="grey", transform=ax.get_xaxis_transform())
ax.set_xticks(x); ax.set_xticklabels([f"{it['family']}\n({it['lang']})" if it["group"] != "English" else it["family"] for it in items], rotation=45, ha="right", fontsize=8.5)
ax.set_ylim(0, 1.08); ax.set_ylabel("accuracy at k = 4  (uses the context's convention ∧ faithful)"); ax.grid(axis="y", alpha=0.3)
ax.text((n_en - 1) / 2, 1.13, f"English target — {sum(it['accept'] for it in items if it['group']=='English')} of {n_en} accepted", ha="center", fontsize=10, fontweight="bold")
ax.text(n_en + (len(items) - n_en - 1) / 2, 1.13, f"non-English target (cheap check, 45–60 texts) — {sum(it['accept'] for it in items if it['group']!='English')} of {len(items)-n_en} accepted", ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(0, 1.18)
ax.legend(loc="upper center", fontsize=8.5, frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.32))
fig.suptitle("Which lexically diverse conventions does Qwen2.5-7B (base) learn from 4 in-context examples? "
             f"Accepted (✓) if BOTH poles reach {CUT:.0%}; red panels = rejected", fontsize=11.5, y=0.995)
fig.tight_layout(rect=(0, 0.02, 1, 0.97)); fig.savefig(MP["results"] / "lexical_selection.png", dpi=150)
print(f"{'family':14s} {'group':12s} {'nat':>5s} {'alt':>5s} {'n':>4s} verdict")
for it in items:
    print(f"{it['family']:14s} {it['group']:12s} {it['nat']:5.2f} {it['alt']:5.2f} {it['n']:4d} {'ACCEPT' if it['accept'] else 'reject'}")
print(f"accepted: {sum(it['accept'] for it in items)} of {len(items)}")
