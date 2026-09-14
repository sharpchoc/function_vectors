#!/usr/bin/env python
"""Selection chart for the 60 code-convention families (cheap k = 4 check on Qwen2.5-7B base): k = 4 accuracy per pole,
30 % both-poles cutoff, grouped by category. Reads <results>/code_k4/k4_check.csv (ml_k4_analyze.py --tag code_k4).
Output: <results>/code_selection.{png,csv}."""
import csv, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.code_families import CODE_FAMILIES, CODE_SPECS
MP = model_paths("qwen25_base"); CUT = 0.30
rows = list(csv.DictReader(open(MP["results"] / "code_k4" / "k4_check.csv")))
CAT = {}
for i, spec in enumerate(CODE_SPECS):
    CAT[spec[0]] = "naming" if i < 10 else "literals" if i < 18 else "syntax / dialect" if i < 41 else "formatting" if i < 48 else "comments / docs" if i < 53 else "other languages"
def get(f, s, key="accuracy"):
    r = [x for x in rows if x["family"] == f and x["style"] == s and int(float(x["k"])) == 4]
    return (float(r[0][key]), int(r[0]["n"])) if r else (np.nan, 0)
items = []
for fm in CODE_FAMILIES:
    (a, n), (b, _) = get(fm.name, "nat"), get(fm.name, "alt")
    if np.isnan(a): continue
    (u, _) = get(fm.name, "nat", "unscorable"); (ua, _) = get(fm.name, "alt", "unscorable")
    (a0, _), (b0, _) = (next(((float(x["accuracy"]), 0) for x in rows if x["family"] == fm.name and x["style"] == "nat" and int(float(x["k"])) == 0), (np.nan, 0)),
                        next(((float(x["accuracy"]), 0) for x in rows if x["family"] == fm.name and x["style"] == "alt" and int(float(x["k"])) == 0), (np.nan, 0)))
    items.append(dict(family=fm.name, lang=fm.tgt_lang, cat=CAT[fm.name], nat=a, alt=b, nat_k0=a0, alt_k0=b0, n=n, unscorable=(u + ua) / 2, accept=min(a, b) >= CUT, nat_label=fm.nat, alt_label=fm.alt))
order = ["naming", "literals", "syntax / dialect", "formatting", "comments / docs", "other languages"]
items.sort(key=lambda it: (order.index(it["cat"]), -min(it["nat"], it["alt"])))
with open(MP["results"] / "code_selection.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(items[0])); w.writeheader(); w.writerows(items)
fig, ax = plt.subplots(figsize=(max(16, 0.5 * len(items)), 6.4)); x = np.arange(len(items)); w = 0.38
for i, it in enumerate(items):
    if not it["accept"]: ax.axvspan(i - 0.5, i + 0.5, color="#f3d6d6", alpha=0.7, zorder=0)
ax.bar(x - w / 2, [it["nat"] for it in items], w, color="#1f77b4", label="natural convention in context (k = 4)")
ax.bar(x + w / 2, [it["alt"] for it in items], w, color="#d62728", label="alternative convention in context (k = 4)")
ax.scatter(x + w / 2, [it["alt_k0"] for it in items], marker="o", s=22, facecolors="white", edgecolors="#d62728", zorder=3, label="alternative, k = 0 baseline")
ax.axhline(CUT, color="k", ls="--", lw=1.2, label=f"cutoff: both poles ≥ {CUT:.0%}")
b = 0
for cat in order:
    n = sum(1 for it in items if it["cat"] == cat)
    if n:
        ax.axvline(b + n - 0.5, color="k", lw=0.8); ax.text(b + (n - 1) / 2, 1.04, f"{cat} — {sum(it['accept'] for it in items if it['cat'] == cat)}/{n}", ha="center", fontsize=9.5, fontweight="bold"); b += n
ax.set_xticks(x); ax.set_xticklabels([f"{it['family']}\n({it['lang']})" for it in items], rotation=90, fontsize=7.5)
ax.set_ylim(0, 1.10); ax.set_ylabel("accuracy at k = 4 (uses the context's convention ∧ acceptable continuation)"); ax.grid(axis="y", alpha=.3)
h, l = ax.get_legend_handles_labels(); h.append(Patch(facecolor="#f3d6d6", edgecolor="none")); l.append("rejected: at least one pole below the cutoff")
ax.legend(h, l, loc="upper center", fontsize=8.5, frameon=False, ncol=5, bbox_to_anchor=(0.5, -0.42))
fig.suptitle(f"Which code conventions does Qwen2.5-7B (base) follow from 4 in-context decisions? cheap check, ~50 tasks per family; accepted if BOTH poles reach {CUT:.0%} — "
             f"{sum(it['accept'] for it in items)} of {len(items)} accepted", fontsize=11.5, y=0.995)
fig.tight_layout(rect=(0, 0.02, 1, 0.96)); fig.savefig(MP["results"] / "code_selection.png", dpi=140)
print(f"{'family':20s} {'cat':16s} {'nat k0→k4':>11s} {'alt k0→k4':>11s} {'unscor':>6s} {'n':>3s} verdict")
for it in items:
    print(f"{it['family']:20s} {it['cat']:16s} {it['nat_k0']:4.2f}→{it['nat']:4.2f}  {it['alt_k0']:4.2f}→{it['alt']:4.2f} {it['unscorable']:6.2f} {it['n']:3d} {'ACCEPT' if it['accept'] else 'reject'}")
print(f"accepted {sum(it['accept'] for it in items)} of {len(items)}")
