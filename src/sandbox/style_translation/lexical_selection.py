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
_r3 = MP["results"] / "multilingual_k4_round3" / "k4_check.csv"
if _r3.exists():
    ml += list(csv.DictReader(open(_r3)))


def get(rows, fam, style, key="accuracy"):
    r = [x for x in rows if x["family"] == fam and x["style"] == style and int(float(x["k"])) == 4]
    return (float(r[0][key]), int(r[0]["n"])) if r else (np.nan, 0)


items = []
ML_NAMES = {fm.name for fm in ML_FAMILIES}
for f in LEXICAL:
    if f in ML_NAMES: continue                    # multilingual families are handled below
    (a, n), (b, _) = get(en, f, "nat"), get(en, f, "alt")
    if np.isnan(a): continue
    items.append(dict(family=f, group="English", lang="English", nat=a, alt=b, n=n, nat_label=FAMILY[f].nat, alt_label=FAMILY[f].alt))
for fm in ML_FAMILIES:
    (a, n), (b, _) = get(en, fm.name, "nat"), get(en, fm.name, "alt")          # full corpus (summary.csv) if step 3 has run
    src = "full"
    if np.isnan(a):
        (a, n), (b, _) = get(ml, fm.name, "nat"), get(ml, fm.name, "alt"); src = "cheap"
    if np.isnan(a): continue
    items.append(dict(family=fm.name, group="English" if fm.tgt_lang == "English" else "non-English", lang=fm.tgt_lang, nat=a, alt=b, n=n, nat_label=fm.nat, alt_label=fm.alt, source=src))
for it in items:
    it.setdefault("source", "full")
for it in items:
    it["min"] = min(it["nat"], it["alt"]); it["accept"] = it["min"] >= CUT
items.sort(key=lambda it: (it["group"] != "English", -it["min"]))
with open(MP["results"] / "lexical_selection.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(items[0])); w.writeheader(); w.writerows(items)

fig, ax = plt.subplots(figsize=(max(14, 0.62 * len(items)), 7.0))
x = np.arange(len(items)); w = 0.38
for i, it in enumerate(items):
    if not it["accept"]:
        ax.axvspan(i - 0.5, i + 0.5, color="#f3d6d6", alpha=0.7, zorder=0)
n_en = sum(it["group"] == "English" for it in items)
ax.axvline(n_en - 0.5, color="k", lw=1)
ax.bar(x - w / 2, [it["nat"] for it in items], w, color="#1f77b4", label="natural convention in context (k = 4)")
ax.bar(x + w / 2, [it["alt"] for it in items], w, color="#d62728", label="alternative convention in context (k = 4)")
ax.axhline(CUT, color="k", ls="--", lw=1.2, label=f"cutoff: both poles ≥ {CUT:.0%}")
from matplotlib.patches import Patch
ax.set_xticks(x); ax.set_xticklabels([f"{it['family']}\n({it['lang']})" if it["group"] != "English" else (it["family"] + ("\n(cheap)" if it["source"] == "cheap" else "")) for it in items], rotation=40, ha="right", fontsize=9)
ax.set_ylim(0, 1.08); ax.set_ylabel("accuracy at k = 4  (uses the context's convention ∧ faithful)"); ax.grid(axis="y", alpha=0.3)
_en = [it for it in items if it["group"] == "English"]; _nc = sum(it["source"] == "cheap" for it in _en)
ax.text((n_en - 1) / 2, 1.04, f"English target ({len(_en) - _nc} full corpus, {_nc} cheap check)" if _nc else "English target (200 texts per family)", ha="center", fontsize=10.5, fontweight="bold")
ax.text((n_en - 1) / 2, 1.04, "", ha="center")
ax.texts[-2].set_text(ax.texts[-2].get_text() + f" — {sum(it['accept'] for it in _en)} of {len(_en)} accepted")
_ne = [it for it in items if it["group"] != "English"]
_nf = sum(it["source"] == "full" for it in _ne)
_lab = "full corpus" if _nf == len(_ne) else ("cheap check, 45–60 texts" if _nf == 0 else f"{_nf} full corpus, {len(_ne) - _nf} cheap check")
ax.text(n_en + (len(items) - n_en - 1) / 2, 1.04, f"non-English target ({_lab}) — {sum(it['accept'] for it in _ne)} of {len(_ne)} accepted", ha="center", fontsize=10.5, fontweight="bold")
ax.set_ylim(0, 1.10)
h, l = ax.get_legend_handles_labels()
h.append(Patch(facecolor="#f3d6d6", edgecolor="none")); l.append("rejected: at least one pole below the cutoff")
ax.legend(h, l, loc="upper center", fontsize=9, frameon=False, ncol=4, bbox_to_anchor=(0.5, -0.30))
fig.suptitle("Which lexically diverse conventions does Qwen2.5-7B (base) learn from 4 in-context examples? "
             f"accepted if BOTH poles reach {CUT:.0%} accuracy", fontsize=12, y=0.995)
fig.tight_layout(rect=(0, 0.02, 1, 0.97)); fig.savefig(MP["results"] / "lexical_selection.png", dpi=150)
print(f"{'family':14s} {'group':12s} {'nat':>5s} {'alt':>5s} {'n':>4s} verdict")
for it in items:
    print(f"{it['family']:14s} {it['group']:12s} {it['nat']:5.2f} {it['alt']:5.2f} {it['n']:4d} {'ACCEPT' if it['accept'] else 'reject'}")
print(f"accepted: {sum(it['accept'] for it in items)} of {len(items)}")
