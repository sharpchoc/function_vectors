#!/usr/bin/env python
"""First-cue steering results with coherence filtering (user spec 2026-09-02).

Reads artifacts/style_properties/steering/full_cue_cue1/<prop>.json (judged by
judge_coherence.py). For every condition and item: coherent? (judge), scorable? (property
classifier on the stored 32-token tail), label. Reported per condition:
  adherence  = P(alt | coherent & scorable)
  unscorable = fraction of COHERENT rollouts the classifier could not score
  incoherent = fraction of all rollouts the judge marked incoherent
Final (layer, dose) per property = the shortlisted setting with the highest adherence among
those whose incoherent rate is <= baseline incoherent + 0.10 (coherence guard).

Outputs in results/style_properties/steering/:
  headline.png          unsteered / steered (coherence-guarded pick) / natural reference
  quality_table.png     per property: unscorable % and incoherent %, unsteered vs steered
  steering_summary.csv  all numbers
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT, STYLE_PROPERTIES_DIR
from src.sandbox.ext_styleprops.properties import PROPS

FULL = ARTIFACTS_ROOT / "style_properties" / "steering" / "full_cuecue1"
SWEEP = ARTIFACTS_ROOT / "style_properties" / "steering" / "sweep_cuecue1"
PRE = ARTIFACTS_ROOT / "style_properties" / "prescreen"
OUT = STYLE_PROPERTIES_DIR / "steering"
GUARD = 0.10


def stats(prop, cond, tgt="alt"):
    """strict  = P(target convention | rollout is coherent)  <- PRIMARY metric: an
                 unscorable rollout counts as 'did not adopt the convention'
       adherence = P(target | coherent AND scorable)  <- secondary, conditional
       unscorable = share of coherent rollouts the classifier cannot score
       incoherent = share of all rollouts the judge called gibberish"""
    tails, coh = cond["tails"], cond.get("coherent") or [None] * len(cond["tails"])
    labs = [PROPS[prop].classify(t) for t in tails]
    n = len(tails)
    inc = sum(c is False for c in coh) / n
    coherent_idx = [i for i in range(n) if coh[i] is not False]      # judge-fail counted as coherent
    unsc = sum(labs[i] is None for i in coherent_idx) / max(len(coherent_idx), 1)
    good = [labs[i] for i in coherent_idx if labs[i] is not None]
    adh = float(np.mean([l == tgt for l in good])) if good else np.nan
    strict = float(np.mean([labs[i] == tgt for i in coherent_idx])) if coherent_idx else np.nan
    return dict(strict=strict, adherence=adh, unscorable=unsc, incoherent=inc, n=n,
                n_coherent=len(coherent_idx), n_scored=len(good))


def main():
    pool = set(json.load(open(REPO_ROOT / "task_splits" / "style_properties_pool.json"))["pass"])
    props = sorted(p.stem for p in FULL.glob("*.json") if p.stem in pool)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, trip = [], []
    for name in props:
        d = json.load(open(FULL / f"{name}.json")); c = d["conditions"]
        base = stats(name, c["baseline_nat2alt"])
        cands = {k: stats(name, v) for k, v in c.items() if k.startswith("cuediff_cue_nat2alt")}
        guarded = {k: s for k, s in cands.items() if s["incoherent"] <= base["incoherent"] + GUARD
                   and not np.isnan(s["strict"])}
        pick_pool = guarded or cands
        pick = max(pick_pool, key=lambda k: pick_pool[k]["strict"])
        st = cands[pick]
        b = d["best_from_sweep"]
        if pick == "cuediff_cue_nat2alt":
            L, a = b["L"], b["alpha"]        # sweep winner was already the cue-derived vector
        elif pick == "cuediff_cue_nat2alt_best":
            # sweep winner was the evidence-derived vector; this bar used the CUE-derived
            # vector at ITS own best sweep setting -> read that setting from the sweep
            sw = json.load(open(SWEEP / f"{name}.json"))["conditions"]
            cb = max((k for k in sw if k.startswith("cuediff_cue_L")
                      and not np.isnan(sw[k]["adherence_tgt"])),
                     key=lambda k: sw[k]["adherence_tgt"])
            L = int(cb.split("_L")[1].split("_")[0]); a = float(cb.split("_a")[1])
        else:
            L = int(pick.split("_L")[1].split("_")[0]); a = float(pick.split("_a")[1])
        recs = json.load(open(PRE / f"{name}.json"))["records"]
        ref = [r for r in recs if r["pol"] == "alt" and r["k"] >= 4 and r["label"]]
        ref_v = float(np.mean([r["label"] == "alt" for r in ref])) if ref else np.nan
        cf = stats(name, c["cfprop_cue_nat2alt"]) if "cfprop_cue_nat2alt" in c else None
        trip.append((name, base, st, ref_v, L, a, cf))
        rows.append(dict(property=name, pick=pick, L=L, alpha=a, guarded=pick in guarded,
                         base_strict=round(base["strict"], 3), steer_strict=round(st["strict"], 3),
                         base_adh=round(base["adherence"], 3) if not np.isnan(base["adherence"]) else "",
                         base_unscorable=round(base["unscorable"], 3), base_incoherent=round(base["incoherent"], 3),
                         steer_adh=round(st["adherence"], 3), steer_unscorable=round(st["unscorable"], 3),
                         steer_incoherent=round(st["incoherent"], 3), steer_n_scored=st["n_scored"],
                         steer_n_coherent=st["n_coherent"], n=st["n"],
                         coherence_guard_met=pick in guarded,
                         cf_adh=round(cf["adherence"], 3) if cf and not np.isnan(cf["adherence"]) else "",
                         cf_unscorable=round(cf["unscorable"], 3) if cf else "",
                         cf_incoherent=round(cf["incoherent"], 3) if cf else "",
                         reference=round(ref_v, 3) if not np.isnan(ref_v) else ""))
    trip.sort(key=lambda t: -t[2]["strict"])

    # headline
    fig, ax = plt.subplots(figsize=(12, 5.4))
    x = np.arange(len(trip)); w = 0.27
    ax.bar(x - w, [t[1]["strict"] for t in trip], w, color="#bdbdbd", label="unsteered")
    ax.bar(x, [t[2]["strict"] for t in trip], w, color="#e63946",
           label="steered: cue-derived vector at the FIRST cue token")
    ax.bar(x + w, [t[3] for t in trip], w, color="#457b9d",
           label="reference: model reading genuine ALT-convention context (k ≥ 4)")
    for xi, t in zip(x, trip):
        ax.text(xi, t[2]["strict"] + 0.02, f"L{t[4]} α{t[5]:g}", ha="center", fontsize=6, color="#555")
        ax.text(xi, -0.055, f"{t[2]['unscorable']*100:.0f}", ha="center", fontsize=7, color="#a00")
        ax.text(xi, -0.105, f"{t[2]['incoherent']*100:.0f}", ha="center", fontsize=7,
                color="#a00" if t[2]["incoherent"] > 0.1 else "#777")
    ax.set_xticks(x, [t[0] for t in trip], rotation=30, ha="right", fontsize=9)
    ax.set_ylim(-0.14, 1.08)
    ax.set_ylabel("fraction of coherent rollouts that\nadopt the ALT convention", fontsize=10)
    ax.text(-0.9, -0.055, "unscorable %", fontsize=7, color="#a00", ha="right")
    ax.text(-0.9, -0.105, "incoherent %", fontsize=7, color="#a00", ha="right")
    ax.set_title("A property vector at the first cue token: does GPT-J adopt the other style convention?\n"
                 "first cue of each document (no prior manifestation) · 32-token T=1 rollouts · rollouts judged "
                 "gibberish by an LLM are dropped\nan unscorable rollout (model never produced the feature) counts "
                 "as NOT adopting", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=8.5, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "headline.png", dpi=170, bbox_inches="tight")

    # quality table figure: unscorable / incoherent, unsteered vs steered
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    names = [t[0] for t in trip]
    for ax, key, ttl in ((axes[0], "unscorable", "unscorable share (of coherent rollouts)"),
                         (axes[1], "incoherent", "incoherent share (judge), of all rollouts")):
        ax.barh(np.arange(len(trip)) + 0.2, [t[1][key] for t in trip], 0.4, color="#bdbdbd", label="unsteered")
        ax.barh(np.arange(len(trip)) - 0.2, [t[2][key] for t in trip], 0.4, color="#e63946", label="steered")
        ax.set_yticks(range(len(trip)), names, fontsize=8)
        ax.set_xlim(0, 1); ax.set_title(ttl, fontsize=10); ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Rollout quality under first-cue steering", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "quality_table.png", dpi=160)

    with open(OUT / "steering_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{'property':15s} STRICT base->steer  (cond.adh)  unscor%  incoh%  n_coh  guard  pick")
    for t in trip:
        g = "ok " if any(r["property"] == t[0] and r["coherence_guard_met"] for r in rows) else "MISS"
        print(f"{t[0]:15s}   {t[1]['strict']:.2f} -> {t[2]['strict']:.2f}      {t[2]['adherence']:.2f}     "
              f"{t[2]['unscorable']*100:3.0f}     {t[2]['incoherent']*100:3.0f}    {t[2]['n_coherent']:4d}  {g}   L{t[4]} a{t[5]:g}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
