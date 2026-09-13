#!/usr/bin/env python
"""Cheap k = 4 check — figure + table for the multilingual families on Qwen2.5-7B base.
Per family: accuracy (style ∧ judge OK) at k = 4 for both context poles, with the k = 0 baseline as a marker, plus
style-only and judge-OK rates. Output: results/style_translation/qwen25_base/multilingual_k4/{k4_check.png, k4_check.csv}."""
import csv, json, math, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.ml_families import ML_FAMILIES

MP = model_paths("qwen25_base"); OUT = MP["results"] / "multilingual_k4"; OUT.mkdir(parents=True, exist_ok=True)


def wilson(p, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


rows = []
for fam in ML_FAMILIES:
    f = MP["rollouts"] / f"{fam.name}.json"
    if not f.exists(): continue
    recs = [r for r in json.load(open(f)) if r.get("judge")]
    for style in ("nat", "alt"):
        for k in (0, 4):
            rs = [r for r in recs if r["style"] == style and r["k"] == k]
            if not rs: continue
            n = len(rs); acc = sum(r["style_ok"] and r["judge"]["ok"] for r in rs) / n
            rows.append(dict(family=fam.name, lang=fam.tgt_lang, style=style, k=k, n=n, accuracy=acc, style_only=sum(r["style_ok"] for r in rs) / n,
                             judge_ok=sum(bool(r["judge"]["ok"]) for r in rs) / n, unscorable=sum(r["decision"] is None for r in rs) / n,
                             ci_lo=wilson(acc, n)[0], ci_hi=wilson(acc, n)[1], label=fam.nat if style == "nat" else fam.alt))
with open(OUT / "k4_check.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
fams = [f for f in ML_FAMILIES if any(r["family"] == f.name for r in rows)]
fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(fams)), 5.2))
x = np.arange(len(fams)); C = {"nat": "#1f77b4", "alt": "#d62728"}
for j, style in enumerate(("nat", "alt")):
    off = -0.2 if style == "nat" else 0.2
    r4 = [next((r for r in rows if r["family"] == f.name and r["style"] == style and r["k"] == 4), None) for f in fams]
    r0 = [next((r for r in rows if r["family"] == f.name and r["style"] == style and r["k"] == 0), None) for f in fams]
    ys = [r["accuracy"] if r else np.nan for r in r4]; lo = [r["accuracy"] - r["ci_lo"] if r else 0 for r in r4]; hi = [r["ci_hi"] - r["accuracy"] if r else 0 for r in r4]
    ax.bar(x + off, ys, 0.38, color=C[style], yerr=[lo, hi], capsize=3, label=f"{'natural' if style == 'nat' else 'alternative'} context, k = 4 (accuracy = convention ∧ faithful)")
    ax.scatter(x + off, [r["style_only"] if r else np.nan for r in r4], marker="_", s=300, color="k", zorder=3, label="style-only rate, k = 4" if j == 0 else None)
    ax.scatter(x + off, [r["accuracy"] if r else np.nan for r in r0], marker="o", s=28, facecolors="white", edgecolors=C[style], zorder=3, label=f"k = 0 baseline ({'natural' if style == 'nat' else 'alternative'} context)")
ax.set_xticks(x); ax.set_xticklabels([f"{f.name}\n({f.tgt_lang})" for f in fams], fontsize=8); ax.set_ylim(0, 1.02); ax.set_ylabel("accuracy toward the context's convention")
ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2, frameon=False)
n_txt = rows[0]["n"] if rows else 0
fig.suptitle(f"Qwen2.5-7B base, English → target-language translation: does 4 in-context examples make the model follow the convention?\n"
             f"cheap check: {n_txt} texts per bar, k = 4 only (k = 0 baseline as hollow circles), 95% CI", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "k4_check.png", dpi=150)
print(f"{'family':14s} {'lang':10s} {'nat k0→k4':>12s} {'alt k0→k4':>12s} {'style-only alt k4':>17s} {'judge OK k4':>11s} {'unscorable k4':>13s}")
for f in fams:
    g = lambda s, k, key: next((r[key] for r in rows if r["family"] == f.name and r["style"] == s and r["k"] == k), np.nan)
    print(f"{f.name:14s} {f.tgt_lang:10s} {g('nat',0,'accuracy'):5.2f} → {g('nat',4,'accuracy'):4.2f} {g('alt',0,'accuracy'):5.2f} → {g('alt',4,'accuracy'):4.2f} "
          f"{g('alt',4,'style_only'):17.2f} {(g('nat',4,'judge_ok')+g('alt',4,'judge_ok'))/2:11.2f} {(g('nat',4,'unscorable')+g('alt',4,'unscorable'))/2:13.2f}")
