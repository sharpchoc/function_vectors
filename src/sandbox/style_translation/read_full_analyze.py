#!/usr/bin/env python
"""Read-feature steering on the full pool at the selected setting (CPU). Records from read_steer/full_k3/<shard>/<family>.json
(read_sweep.py --layers L --alphas a --with_base --limit 40): per family and direction (target style), success = TARGET convention AND
judge OK, for the unsteered wrong-style prompts, the steered ones, and the step-3 k = 3 accuracy of the target style (ceiling); also the
target-convention rate without the judge. Outputs -> <results>/read_steering/full56/: full.csv, by_direction.csv, headline_bars.png,
success_grid.png, gain_by_family.png."""
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

COL = {"unsteered": "#9e9e9e", "steered": "#1f4f7a", "k3": "#2b7a4b"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--tag", default="full56"); ap.add_argument("--layer", type=int, default=8); ap.add_argument("--alpha", type=float, default=4.0)
    args = ap.parse_args()
    MP = model_paths(args.model); R = MP["results"]; SRC = MP["read_features"].parent / "read_steer" / "full_k3"; OUT = R / "read_steering" / args.tag; OUT.mkdir(parents=True, exist_ok=True)
    pool = json.load(open(R / "code_pool.json"))["pool"]
    recs = []
    for f in sorted(SRC.glob("*/*.json")):
        recs += json.load(open(f))
    unj = sum(1 for r in recs if not r.get("judge")); assert unj == 0, f"{unj} unjudged records"
    fams = [f for f in pool if any(r["family"] == f for r in recs)]; assert len(fams) == len(pool), f"missing: {set(pool) - set(fams)}"
    k3 = {}
    for row in csv.DictReader(open(R / "summary.csv")):
        if int(row["k"]) == 3: k3[(row["family"], row["style"])] = float(row["accuracy"])
    cell = defaultdict(list)
    for r in recs:
        cell[(r["family"], r["context_style"], r["target"])].append(r)
    S, rows = {}, []
    for f in fams:
        for ctx, tgt in (("alt", "nat"), ("nat", "alt")):
            b = cell[(f, ctx, None)]; s = cell[(f, ctx, tgt)]
            assert s and all(r["layer"] == args.layer and r["alpha"] == args.alpha for r in s), (f, ctx)
            def rate(rs, fn): return float(np.mean([fn(r) for r in rs])) if rs else float("nan")
            suc = rate(s, lambda r: r["decision"] == tgt and r["judge"]["ok"]); lo, hi = wilson(suc, len(s))
            S[(f, tgt)] = dict(n=len(s), unsteered=rate(b, lambda r: r["decision"] == tgt and r["judge"]["ok"]), steered=suc, lo=lo, hi=hi,
                              unsteered_target_rate=rate(b, lambda r: r["decision"] == tgt), steered_target_rate=rate(s, lambda r: r["decision"] == tgt),
                              judge_ok_unsteered=rate(b, lambda r: r["judge"]["ok"]), judge_ok_steered=rate(s, lambda r: r["judge"]["ok"]),
                              unscorable_steered=rate(s, lambda r: r["decision"] is None), margin_unsteered=rate(b, lambda r: r["margin_target"]), margin_steered=rate(s, lambda r: r["margin_target"]),
                              n_evidence=rate(s, lambda r: r["n_evidence"]), k3_ceiling=k3.get((f, tgt), float("nan")))
            rows.append(dict(family=f, context=ctx, target=tgt, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in S[(f, tgt)].items()}))
    with open(OUT / "full.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    by = []
    for t in ("nat", "alt", "both"):
        tt = ("nat", "alt") if t == "both" else (t,)
        by.append(dict(target=t, **{k: round(float(np.nanmean([S[(f, x)][k] for f in fams for x in tt])), 4) for k in ("unsteered", "steered", "k3_ceiling", "unsteered_target_rate", "steered_target_rate", "judge_ok_unsteered", "judge_ok_steered", "unscorable_steered")},
                       n_fams_gain_ge_20=int(sum(np.mean([S[(f, x)]["steered"] - S[(f, x)]["unsteered"] for x in tt]) >= .2 for f in fams)),
                       n_fams_at_ceiling=int(sum(np.mean([S[(f, x)]["steered"] - S[(f, x)]["k3_ceiling"] for x in tt]) >= -.05 for f in fams))))
    with open(OUT / "by_direction.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(by[0])); w.writeheader(); w.writerows(by)
    json.dump(dict(layer=args.layer, alpha=args.alpha, n_docs=40, families=fams, by_direction=by), open(OUT / "summary.json", "w"), indent=1)
    # headline bars (pooled, dots = families)
    fig, ax = plt.subplots(figsize=(8.5, 5)); x = np.arange(2); w = .26; rng = np.random.default_rng(0)
    for i, (key, lab) in enumerate((("unsteered", "unsteered (wrong-style demos)"), ("steered", f"steered (read feature, L{args.layer} α{args.alpha:g})"), ("k3_ceiling", "real 3-shot, right-style demos"))):
        ys = [np.nanmean([S[(f, d)][key] for f in fams]) for d in ("nat", "alt")]; ax.bar(x + (i - 1) * w, ys, w, color=COL[{"k3_ceiling": "k3"}.get(key, key)], label=lab)
        for xi, y in zip(x + (i - 1) * w, ys): ax.text(xi, y + .015, f"{y:.2f}", ha="center", fontsize=9)
        for j, d in enumerate(("nat", "alt")):
            pts = [S[(f, d)][key] for f in fams]; ax.scatter(np.full(len(pts), x[j] + (i - 1) * w) + rng.uniform(-.06, .06, len(pts)), pts, s=7, color="k", alpha=.35, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(["→ natural (alt-context prompts)", "→ alternative (nat-context prompts)"]); ax.set_ylim(0, 1.08); ax.set_ylabel("success: target convention AND judge OK"); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"Read-feature steering at every evidence token, {len(fams)} families (dots), 40 held-out k = 3 prompts per family and context", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "headline_bars.png", dpi=150); plt.close(fig)
    # grid: families x (unsteered, steered, ceiling) per direction
    cols = ["unsteered", "steered", "k3 right-style"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.27 * len(fams) + 2.6), sharey=True)
    for ax, d in zip(axes, ("nat", "alt")):
        M = np.array([[S[(f, d)]["unsteered"], S[(f, d)]["steered"], S[(f, d)]["k3_ceiling"]] for f in fams]); ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for i in range(len(fams)):
            for j in range(3): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if M[i, j] < .55 else "k")
        ax.set_xticks(range(3)); ax.set_xticklabels(cols, rotation=20, ha="right", fontsize=8); ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams, fontsize=7)
        ax.set_title(f"→ {'natural' if d == 'nat' else 'alternative'}\nmeans {np.nanmean(M[:, 0]):.2f} · {np.nanmean(M[:, 1]):.2f} · {np.nanmean(M[:, 2]):.2f}", fontsize=9)
    fig.suptitle(f"Read steering, L{args.layer} α{args.alpha:g}: success per family", fontsize=10); fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(OUT / "success_grid.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=False)
    for ax, d in zip(axes, ("nat", "alt")):
        g = np.array([S[(f, d)]["steered"] - S[(f, d)]["unsteered"] for f in fams]); gap = np.array([S[(f, d)]["steered"] - S[(f, d)]["k3_ceiling"] for f in fams]); o = np.argsort(-g)
        ax.bar(range(len(fams)), g[o], color=np.where(g[o] >= 0, "#2b7a4b", "#a3271d"), label="steered − unsteered"); ax.plot(range(len(fams)), gap[o], "k.", label="steered − right-style k = 3 ceiling")
        ax.set_xticks(range(len(fams))); ax.set_xticklabels([fams[k] for k in o], rotation=90, fontsize=6.5); ax.axhline(0, color="k", lw=.6); ax.set_title(f"→ {'natural' if d == 'nat' else 'alternative'} · mean gain {g.mean():+.2f} · ≥ +.20 in {int((g >= .2).sum())}/{len(fams)} · at or above ceiling (−.05) in {int((gap >= -.05).sum())}", fontsize=9); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "gain_by_family.png", dpi=130); plt.close(fig)
    for row in by: print(row)
    print("->", OUT)


if __name__ == "__main__":
    main()
