#!/usr/bin/env python
"""Write-feature steering on the full pool (CPU). Four cells (L24, L26) x (alpha 2, 4), 40 held-out 0-shot prompts per family,
each steered towards natural and towards alternative; success = TARGET convention AND judge OK. No cell is selected (user decision
2026-09-18): all four are reported side by side with the unsteered baseline and the k = 4 in-context accuracy of step 3.
Sources: steering/full_k3/<shard>/<family>.json (46 families) + steering/sweep_k3/s1 (base, L26) and s7 (L24) for the 10 sweep
families, cut to alpha in {2, 4} and to each family's first 40 held-out doc_ids (same rule as write_sweep.py --limit 40).
Outputs -> <results>/write_steering/full56/: full.csv, by_cell.csv, success_grid.png, gain_by_family.png, steer_vs_k4.png."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.write_sweep_analyze import wilson

LAYERS, ALPHAS, N_DOCS = (24, 26), (2.0, 4.0), 40
CELLS = [(L, a) for L in LAYERS for a in ALPHAS]


def load(MP, pool):
    recs = []
    for f in sorted((MP["steering"] / "full_k3").glob("*/*.json")):
        recs += json.load(open(f))
    for s in ("s1", "s7"):                       # the 10 sweep families: base + L26 from s1, L24 from s7
        for f in sorted((MP["steering"] / "sweep_k3" / s).glob("*.json")):
            recs += [r for r in json.load(open(f)) if r["arm"] == "base" or (r["layer"] in LAYERS and r["alpha"] in ALPHAS)]
    recs = [r for r in recs if r["family"] in pool]
    docs = defaultdict(set)
    for r in recs:
        docs[r["family"]].add(r["doc_id"])
    keep = {f: set(sorted(d)[:N_DOCS]) for f, d in docs.items()}    # first 40 sorted held-out docs, the write_sweep.py --limit rule
    recs = [r for r in recs if r["doc_id"] in keep[r["family"]]]
    unj = sum(1 for r in recs if not r.get("judge")); assert unj == 0, f"{unj} unjudged records"
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--tag", default="full56")
    args = ap.parse_args()
    MP = model_paths(args.model); RES = MP["results"]; OUT = RES / "write_steering" / args.tag; OUT.mkdir(parents=True, exist_ok=True)
    pool = json.load(open(RES / "code_pool.json"))["pool"]
    recs = load(MP, pool)
    fams = [f for f in pool if any(r["family"] == f for r in recs)]; assert len(fams) == len(pool), f"missing families: {set(pool) - set(fams)}"
    k4 = {}
    for row in csv.DictReader(open(RES / "summary.csv")):
        if int(row["k"]) == 4:
            k4[(row["family"], row["style"])] = float(row["accuracy"])
    cell = defaultdict(list)
    for r in recs:
        cell[(r["family"], r["target"], r["layer"], r["alpha"])].append(r)
    S, rows, base = {}, [], {}
    for f in fams:
        b = cell[(f, None, 0, 0.0)]; assert len(b) == N_DOCS, (f, "base", len(b))
        base[f] = {t: float(np.mean([r["decision"] == t and r["judge"]["ok"] for r in b])) for t in ("nat", "alt")} | {"jok": float(np.mean([r["judge"]["ok"] for r in b])), "unsc": float(np.mean([r["decision"] is None for r in b]))}
        for t in ("nat", "alt"):
            for L, a in CELLS:
                c = cell[(f, t, L, a)]; n = len(c); assert n == N_DOCS, (f, t, L, a, n)
                suc = float(np.mean([r["decision"] == t and r["judge"]["ok"] for r in c])); lo, hi = wilson(suc, n)
                S[(f, t, L, a)] = dict(n=n, success=suc, lo=lo, hi=hi, target_rate=float(np.mean([r["decision"] == t for r in c])), judge_ok=float(np.mean([r["judge"]["ok"] for r in c])),
                                       unscorable=float(np.mean([r["decision"] is None for r in c])), margin=float(np.mean([r["margin_target"] for r in c])), top1=float(np.mean([r["top1"] == t for r in c])), unsteered=base[f][t], k4=k4[(f, t)])
                rows.append(dict(family=f, target=t, layer=L, alpha=a, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in S[(f, t, L, a)].items()}))
    with open(OUT / "full.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    by = []
    for t in ("nat", "alt", "both"):
        tt = ("nat", "alt") if t == "both" else (t,)
        row = dict(target=t, unsteered=round(float(np.mean([base[f][x] for f in fams for x in tt])), 4), k4=round(float(np.mean([k4[(f, x)] for f in fams for x in tt])), 4))
        for L, a in CELLS:
            row[f"L{L}_a{a:g}"] = round(float(np.mean([S[(f, x, L, a)]["success"] for f in fams for x in tt])), 4)
            row[f"L{L}_a{a:g}_judge_ok"] = round(float(np.mean([S[(f, x, L, a)]["judge_ok"] for f in fams for x in tt])), 4)
            row[f"L{L}_a{a:g}_n_fams_gain_ge_.20"] = int(sum(np.mean([S[(f, x, L, a)]["success"] - base[f][x] for x in tt]) >= .20 for f in fams))
        by.append(row)
    with open(OUT / "by_cell.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(by[0])); w.writeheader(); w.writerows(by)
    json.dump(dict(cells=[dict(layer=L, alpha=a) for L, a in CELLS], n_docs=N_DOCS, families=fams, by_cell=by), open(OUT / "summary.json", "w"), indent=1)

    # --- success_grid.png: families x (unsteered, 4 cells, k = 4), one panel per direction
    cols = ["unsteered"] + [f"L{L} α{a:g}" for L, a in CELLS] + ["k = 4 in context"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.28 * len(fams) + 2.8), sharey=True)
    for ax, t in zip(axes, ("nat", "alt")):
        M = np.array([[base[f][t]] + [S[(f, t, L, a)]["success"] for L, a in CELLS] + [k4[(f, t)]] for f in fams])
        ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for i in range(len(fams)):
            for j in range(len(cols)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if M[i, j] < .55 else "k")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8); ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams, fontsize=7)
        ax.set_title(f"steering towards the {'natural' if t == 'nat' else 'alternative'} convention\nmean over families: " + " · ".join(f"{M[:, j].mean():.2f}" for j in range(len(cols))) + "  (column order)", fontsize=8.5)
        ax.axvline(0.5, color="w", lw=2); ax.axvline(len(cols) - 1.5, color="w", lw=2)
    fig.suptitle("Write-feature steering, 56 pool families, Qwen2.5-7B base: success (target convention AND judge OK) on 40 held-out 0-shot prompts per family\n"
                 "cue-token steering with α·(μ_nat − μ_alt); unsteered and k = 4 in-context accuracy (step 3, all 200 docs) as references", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(OUT / "success_grid.png", dpi=150); plt.close(fig)

    # --- gain_by_family.png: steered − unsteered per family, 2 directions x 4 cells
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=True)
    for i, t in enumerate(("nat", "alt")):
        for j, (L, a) in enumerate(CELLS):
            ax = axes[i, j]; g = np.array([S[(f, t, L, a)]["success"] - base[f][t] for f in fams]); order = np.argsort(-g)
            ax.bar(range(len(fams)), g[order], color=np.where(g[order] >= 0, "#2b7a4b", "#a3271d"))
            ax.set_xticks(range(len(fams))); ax.set_xticklabels([fams[k] for k in order], rotation=90, fontsize=5.5); ax.axhline(0, color="k", lw=.6)
            ax.set_title(f"→ {'natural' if t == 'nat' else 'alternative'} · L{L} α{a:g} · mean gain {g.mean():+.2f} · ≥ +.20 in {int((g >= .2).sum())}/{len(fams)}", fontsize=9)
            if j == 0: ax.set_ylabel("success − unsteered success")
    fig.suptitle("Gain from steering per family (sorted within each panel)", fontsize=11); fig.tight_layout(); fig.savefig(OUT / "gain_by_family.png", dpi=130); plt.close(fig)

    # --- steer_vs_k4.png: steered success vs k = 4 in-context accuracy
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    for i, t in enumerate(("nat", "alt")):
        for j, (L, a) in enumerate(CELLS):
            ax = axes[i, j]; x = [k4[(f, t)] for f in fams]; y = [S[(f, t, L, a)]["success"] for f in fams]; b0 = [base[f][t] for f in fams]
            ax.scatter(x, b0, s=14, color="#9a9a9a", label="unsteered"); ax.scatter(x, y, s=16, color="#1f4f7a", label="steered")
            for xi, bi, yi in zip(x, b0, y): ax.plot([xi, xi], [bi, yi], color="#bbb", lw=.5, zorder=0)
            ax.plot([0, 1], [0, 1], "k--", lw=.6); ax.set_title(f"→ {'natural' if t == 'nat' else 'alternative'} · L{L} α{a:g}", fontsize=9)
            if i == 1: ax.set_xlabel("k = 4 in-context accuracy")
            if j == 0: ax.set_ylabel("0-shot success"); ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Steered 0-shot success against four in-context examples, per family (grey = unsteered, blue = steered)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "steer_vs_k4.png", dpi=130); plt.close(fig)
    for row in by:
        print(row)
    print("->", OUT)


if __name__ == "__main__":
    main()
