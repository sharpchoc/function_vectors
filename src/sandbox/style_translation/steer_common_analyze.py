#!/usr/bin/env python
"""Step 4e — one common injection layer for all families? Compare, per family and target, the full
accuracy at the common layer (best alpha per family, chosen by full accuracy on the 200 texts; the
screen-picked alpha is reported too) with the accuracy at the family's own best (layer, alpha) from
best_config.csv. Verdict: within -0.03 of the own-best accuracy.
Outputs -> results/style_translation/steering/: common_layer.png, common_layer.csv.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, STYLE_TRANSLATION_RESULTS
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.steer_analyze import wilson
from src.sandbox.style_translation.steer_screen import ALPHAS
from src.sandbox.style_translation.family_groups import grouped_grid, grouped_order

COMMON = ARTIFACTS_ROOT / "style_translation" / "steering" / "common_layer"
OUT = STYLE_TRANSLATION_RESULTS / "steering"
TOL = 0.03


def main():
    best = {(r["family"], r["target"]): r for r in csv.DictReader(open(OUT / "best_config.csv"))}
    screen = {}
    for r in csv.DictReader(open(OUT / "screen.csv")):
        screen[(r["family"], r["target"], int(r["layer"]), float(r["alpha"]))] = float(r["target_rate"])
    fams = [f.name for f in FAMILIES if (COMMON / f"{f.name}.json").exists()]
    rows, pick = [], {}
    L = None
    for fam in fams:
        recs = [r for r in json.load(open(COMMON / f"{fam}.json")) if r.get("judge")]
        L = recs[0]["layer"]
        for target in ("nat", "alt"):
            cands = []
            for a in ALPHAS:
                sel = [r for r in recs if r["style"] == target and r["alpha"] == a]
                if not sel:
                    continue
                acc = float(np.mean([r["decision"] == target and r["judge"]["ok"] for r in sel])); lo, hi = wilson(acc, len(sel))
                row = dict(family=fam, target=target, layer=L, alpha=a, n=len(sel), accuracy=round(acc, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                           style_only=round(float(np.mean([r["decision"] == target for r in sel])), 3),
                           unscorable=round(float(np.mean([r["decision"] is None for r in sel])), 3),
                           judge_ok=round(float(np.mean([r["judge"]["ok"] for r in sel])), 3),
                           reused=any(r.get("reused_from") for r in sel), screen_rate=screen.get((fam, target, L, a)))
                rows.append(row); cands.append(row)
            b = best[(fam, target)]
            by_acc = max(cands, key=lambda r: (r["accuracy"], -r["alpha"]))
            by_screen = max(cands, key=lambda r: (r["screen_rate"] or 0, -r["alpha"]))
            pick[(fam, target)] = dict(best_layer=int(b["layer"]), best_alpha=float(b["alpha"]), best_acc=float(b["accuracy"]),
                                        best_lo=float(b["ci_lo"]), best_hi=float(b["ci_hi"]), common=by_acc, common_screen_alpha=by_screen)
    keys = list(rows[0])
    with open(OUT / "common_layer.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys + ["best_layer", "best_alpha", "best_acc", "delta_vs_best", "within_tol", "picked_by_accuracy", "picked_by_screen"]); w.writeheader()
        for r in rows:
            p = pick[(r["family"], r["target"])]
            w.writerow(r | dict(best_layer=p["best_layer"], best_alpha=p["best_alpha"], best_acc=p["best_acc"], delta_vs_best=round(r["accuracy"] - p["best_acc"], 3),
                                within_tol=r["accuracy"] >= p["best_acc"] - TOL, picked_by_accuracy=r is p["common"], picked_by_screen=r is p["common_screen_alpha"]))

    # ---- figure ---------------------------------------------------------------------------------
    C = {"nat": "#1f77b4", "alt": "#d62728"}
    fig, gax = grouped_grid(fams, panel_w=3.9, panel_h=3.0, top=0.855, sharey=True)
    for fam in grouped_order(fams):
        ax = gax[(fam, 0)]; x = 0; ticks, labels = [], []
        for target in ("nat", "alt"):
            p = pick[(fam, target)]; c = p["common"]
            ax.bar(x, p["best_acc"], color=C[target], alpha=0.45, edgecolor="none", yerr=[[p["best_acc"] - p["best_lo"]], [p["best_hi"] - p["best_acc"]]], capsize=2)
            ax.bar(x + 1, c["accuracy"], color=C[target], edgecolor="black", linewidth=0.8, yerr=[[c["accuracy"] - c["ci_lo"]], [c["ci_hi"] - c["accuracy"]]], capsize=2)
            ok = c["accuracy"] >= p["best_acc"] - TOL
            ax.text(x + 0.5, 1.02, f"→ {target}  {'✓' if ok else '✗ ' + format(c['accuracy'] - p['best_acc'], '+.2f')}", ha="center", fontsize=8, color=C[target] if ok else "black")
            ticks += [x, x + 1]; labels += [f"own best L{p['best_layer']}, α={p['best_alpha']:g}", f"L{L}, α={c['alpha']:g}"]
            x += 3
        ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
        ax.set_title(fam, fontsize=10); ax.set_ylim(0, 1.12); ax.grid(axis="y", alpha=0.3)
    for ax in fig.axes_meta["left"]:
        ax.set_ylabel("accuracy", fontsize=9)
    handles = [Patch(color="#9e9e9e", alpha=0.45, label="own best (layer, α) per family"), Patch(facecolor="#9e9e9e", edgecolor="black", label=f"common layer {L}, best α per family"),
               Patch(color=C["nat"], label="toward nat"), Patch(color=C["alt"], label="toward alt")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=4, fontsize=9.5, frameon=False)
    n_ok = sum(p["common"]["accuracy"] >= p["best_acc"] - TOL for p in pick.values())
    fig.suptitle(f"Can every family steer from the same layer? Layer {L} with a per-family α vs each family's own best (layer, α)\n"
                 f"accuracy = target convention used AND faithful, coherent translation; 200 texts per bar, 95% CI; ✓ = within {TOL:g} of own best ({n_ok} of {len(pick)} targets)",
                 fontsize=12, y=0.995)
    fig.savefig(OUT / "common_layer.png", dpi=150); plt.close(fig)

    print(f"{'family':14s} {'tgt':3s} {'own best':>16s} {'L%d best α' % L:>12s} {'acc':>5s} {'Δ':>6s} {'ok':>3s} | screen-picked α acc")
    for (fam, target), p in sorted(pick.items(), key=lambda kv: (fams.index(kv[0][0]), kv[0][1])):
        c = p["common"]; s = p["common_screen_alpha"]
        print(f"{fam:14s} {target:3s} L{p['best_layer']:<2d} α{p['best_alpha']:<4g} {p['best_acc']:5.2f} {'α=%g' % c['alpha']:>12s} {c['accuracy']:5.2f} {c['accuracy'] - p['best_acc']:+6.2f} {'✓' if c['accuracy'] >= p['best_acc'] - TOL else '✗':>3s} | α={s['alpha']:g} {s['accuracy']:.2f}")
    print(f"within {TOL:g}: {n_ok}/{len(pick)} targets; alt only: {sum(p['common']['accuracy'] >= p['best_acc'] - TOL for (f, t), p in pick.items() if t == 'alt')}/17; nat only: {sum(p['common']['accuracy'] >= p['best_acc'] - TOL for (f, t), p in pick.items() if t == 'nat')}/17")
    print("->", OUT)


if __name__ == "__main__":
    main()
