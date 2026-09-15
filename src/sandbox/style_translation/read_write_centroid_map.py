#!/usr/bin/env python
"""Centroid-level read→write map for the code pool (user request 2026-09-15): one point per (family, pole) = the MEAN read activation
(evidence tokens, layer L_r) and the MEAN write activation (cue token, L24) over the family's k = 4 prompts of that pole — raw activations,
not difference vectors. Ridge (train-mean centring, λ by leave-one-family-out CV on the training centroids) fitted on the training families'
2 centroids each, scored on the held-out families' centroids:
  cos_centroid   cos(pred − train mean, true − train mean) per held-out centroid
  r2_centroid    R² of the held-out centroids around the train mean (raw) — and around the test mean
  cos_conv       cos(pred nat − pred alt, true nat − true alt) per held-out family (the convention vector implied by the two predicted centroids)
Baselines: shuffled read–write pairing of the training centroids; mean-vector baseline for the convention (mean of the training families'
true convention vectors). Also runs the per-PROMPT ridge of read_write_ridge on the same split for comparison.
Outputs → results/<model>/<tag>/read_write_map/centroid_map_<split>.csv and .png"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.read_write_ridge import DualRidge, make_ridge, r2, cos, LAMBDA_GRID, code_strata, load, configure as ridge_configure
import src.sandbox.style_translation.read_write_ridge as rwr


def make_split(pool, frac_test, seed, path):
    cat = code_strata(pool); rng = np.random.default_rng(seed); test = []
    n_test = round(len(pool) * frac_test)
    for c in sorted(set(cat.values())):
        fs = sorted(f for f in pool if cat[f] == c); k = max(1, round(len(fs) * n_test / len(pool)))
        test += [str(x) for x in rng.choice(fs, size=min(k, len(fs)), replace=False)]
    test = sorted(test); train = [f for f in pool if f not in test]
    json.dump({"train": train, "test": test, "note": f"{1-frac_test:.0%}-{frac_test:.0%} split stratified by code category, seed {seed}, written before fitting",
               "category": {f: cat[f] for f in pool}}, open(path, "w"), indent=1)
    return train, test


def centroids(pool, lr, lw):
    R, W, lab = [], [], []
    for f in pool:
        d = np.load(rwr.PAIRS / f"{f}.npz"); X = d[f"read_L{lr}"].astype(np.float64); Y = d[f"write_L{lw}"].astype(np.float64); P = d["pole"]
        for s in ("nat", "alt"):
            R.append(X[P == s].mean(0)); W.append(Y[P == s].mean(0)); lab.append((f, s))
    return np.array(R), np.array(W), lab


def loo_family_cv(R, W, fams, grid):
    scores = np.zeros(len(grid))
    for f in sorted(set(fams)):
        m = np.array([x == f for x in fams])
        M = DualRidge(R[~m], W[~m])
        for i, lam in enumerate(grid):
            scores[i] += r2(M.fit(lam).predict(R[m]), W[m], M.ym)
    return scores / len(set(fams))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_base"); ap.add_argument("--tag", default="code")
    ap.add_argument("--pool", default=None, help="JSON with a 'pool' list (default code_pool_full.json)")
    ap.add_argument("--split", default="80_20", help="name; file fixed_split_<name>.json in the bucket (created if absent)")
    ap.add_argument("--frac_test", type=float, default=0.2); ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--read_layers", nargs="*", type=int, default=[5, 6, 7, 8, 9, 10]); ap.add_argument("--read_layer", type=int, default=8)
    ap.add_argument("--write_layer", type=int, default=24)
    args = ap.parse_args()
    ridge_configure(args.model, args.tag); OUT = rwr.OUT; OUT.mkdir(parents=True, exist_ok=True)
    MP = model_paths(args.model)
    pool = json.load(open(args.pool or (MP["results"] / "code_pool_full.json")))["pool"]
    sp = OUT / f"fixed_split_{args.split}.json"
    if sp.exists():
        d = json.load(open(sp)); train, test = d["train"], d["test"]
    else:
        train, test = make_split(pool, args.frac_test, args.seed, sp)
    print(f"split {args.split}: train {len(train)} / test {len(test)}: {test}")
    cat = code_strata(pool); grid = LAMBDA_GRID; rng = np.random.default_rng(0)
    rows, per = [], []
    for lr in args.read_layers:
        R, W, lab = centroids(pool, lr, args.write_layer)
        fams = [f for f, _ in lab]; mtr = np.array([f in train for f in fams]); mte = ~mtr
        cv = loo_family_cv(R[mtr], W[mtr], [f for f in fams if f in train], grid); lam = grid[int(np.argmax(cv))]
        M = DualRidge(R[mtr], W[mtr]).fit(lam); pred = M.predict(R[mte])
        Msh = DualRidge(R[mtr][rng.permutation(mtr.sum())], W[mtr]).fit(lam); psh = Msh.predict(R[mte])
        true = W[mte]; labs = [lab[i] for i in np.where(mte)[0]]
        cc = [cos(p - M.ym, t - M.ym) for p, t in zip(pred, true)]; csh = [cos(p - M.ym, t - M.ym) for p, t in zip(psh, true)]
        # convention vectors implied by the predicted centroids
        dtr = np.array([W[[i for i, l in enumerate(lab) if l == (g, "nat")][0]] - W[[i for i, l in enumerate(lab) if l == (g, "alt")][0]] for g in train]); dbar = dtr.mean(0)
        conv = []
        for g in test:
            i_n = [i for i, l in enumerate(labs) if l == (g, "nat")][0]; i_a = [i for i, l in enumerate(labs) if l == (g, "alt")][0]
            dp, dt = pred[i_n] - pred[i_a], true[i_n] - true[i_a]; dps = psh[i_n] - psh[i_a]
            conv.append(dict(read_layer=lr, family=g, category=cat[g], cos_conv=cos(dp, dt), cos_conv_shuffled=cos(dps, dt), cos_conv_baseline=cos(dbar, dt),
                             norm_ratio=float(np.linalg.norm(dp) / np.linalg.norm(dt)), cos_centroid_nat=cc[i_n], cos_centroid_alt=cc[i_a],
                             cos_centroid_shuffled=float(np.mean([csh[i_n], csh[i_a]]))))
        per += conv
        row = dict(read_layer=lr, write_layer=args.write_layer, lam=lam, cv_r2=float(cv.max()), n_train_centroids=int(mtr.sum()), n_test_centroids=int(mte.sum()),
                   r2_centroid_trainmean=r2(pred, true, M.ym), r2_centroid_testmean=r2(pred, true, true.mean(0)), r2_centroid_shuffled=r2(psh, true, M.ym),
                   cos_centroid_mean=float(np.mean(cc)), cos_centroid_shuffled=float(np.mean(csh)),
                   cos_conv_mean=float(np.mean([c["cos_conv"] for c in conv])), cos_conv_shuffled=float(np.mean([c["cos_conv_shuffled"] for c in conv])),
                   cos_conv_baseline=float(np.mean([c["cos_conv_baseline"] for c in conv])), r2_conv=r2(np.array([pred[[i for i, l in enumerate(labs) if l == (g, "nat")][0]] - pred[[i for i, l in enumerate(labs) if l == (g, "alt")][0]] for g in test]),
                                                                                                         np.array([true[[i for i, l in enumerate(labs) if l == (g, "nat")][0]] - true[[i for i, l in enumerate(labs) if l == (g, "alt")][0]] for g in test]), 0.0))
        # per-prompt ridge on the same split for comparison (convention cos of held-out families)
        X, Y, F, P = load(pool, lr, args.write_layer); ptr, pte = np.isin(F, train), np.isin(F, test)
        cvp = rwr.group_cv(X[ptr], Y[ptr], F[ptr], LAMBDA_GRID[::2]); lamp = LAMBDA_GRID[::2][int(np.argmax(cvp))]
        Rp = make_ridge(X[ptr], Y[ptr]).fit(lamp); o, perp = rwr.evaluate(Rp, X[pte], Y[pte], F[pte], P[pte], "prompt")
        row.update(prompt_lam=lamp, prompt_cos_conv_mean=o["cos_diff_mean"], prompt_r2_trainmean=o["r2_trainmean"], prompt_r2_centroid_trainmean=o["r2_centroid_trainmean"], prompt_cos_centroid_mean=o["cos_centroid_mean"])
        pp = {x["family"]: x["cos_diff"] for x in perp}
        for c in conv:
            c["prompt_cos_conv"] = pp[c["family"]]
        rows.append(row)
        print(f"read L{lr} → write L{args.write_layer}  CENTROID map (n={int(mtr.sum())} train points, λ={lam:g}): held-out centroid cos {row['cos_centroid_mean']:.2f} (shuffled {row['cos_centroid_shuffled']:.2f}) "
              f"R² train-mean {row['r2_centroid_trainmean']:+.3f} test-mean {row['r2_centroid_testmean']:+.3f} (shuffled {row['r2_centroid_shuffled']:+.3f}) | convention cos {row['cos_conv_mean']:.2f} "
              f"(shuffled {row['cos_conv_shuffled']:.2f}, baseline {row['cos_conv_baseline']:.2f}) R² {row['r2_conv']:+.2f} || PROMPT map: convention cos {o['cos_diff_mean']:.2f}, centroid cos {o['cos_centroid_mean']:.2f}, R² {o['r2_trainmean']:+.3f}", flush=True)
        if lr == args.read_layer:
            for c in conv:
                print(f"    {c['family']:18s} {c['category']:16s} centroid cos nat {c['cos_centroid_nat']:.2f} alt {c['cos_centroid_alt']:.2f} (shuffled {c['cos_centroid_shuffled']:.2f}) | convention cos {c['cos_conv']:.2f} (shuffled {c['cos_conv_shuffled']:+.2f}, baseline {c['cos_conv_baseline']:+.2f}, prompt-map {c['prompt_cos_conv']:.2f}) |pred|/|true| {c['norm_ratio']:.2f}")
    with open(OUT / f"centroid_map_{args.split}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(OUT / f"centroid_map_{args.split}_per_family.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per[0])); w.writeheader(); w.writerows(per)
    # figure
    main = [c for c in per if c["read_layer"] == args.read_layer]; main.sort(key=lambda c: (c["category"], -c["cos_conv"]))
    fams = [c["family"] for c in main]; x = np.arange(len(fams))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [2.6, 1]})
    ax[0].bar(x - .3, [c["cos_conv"] for c in main], .2, color="#1f6c80", label=f"centroid map (read L{args.read_layer} → write L{args.write_layer}): implied convention vector")
    ax[0].bar(x - .1, [c["prompt_cos_conv"] for c in main], .2, color="#7fb3c4", label="per-prompt map, same split")
    ax[0].bar(x + .1, [c["cos_conv_baseline"] for c in main], .2, color="#b8860b", label="baseline: mean training convention vector")
    ax[0].bar(x + .3, [c["cos_conv_shuffled"] for c in main], .2, color="lightgrey", label="shuffled pairing")
    prev = None
    for i, c in enumerate(main):
        if c["category"] != prev:
            ax[0].axvline(i - .5, color="#dddddd", lw=1); ax[0].text(i - .4, 1.0, c["category"], fontsize=7.5, va="top", color="#555555"); prev = c["category"]
    ax[0].set_xticks(x); ax[0].set_xticklabels(fams, rotation=60, ha="right", fontsize=8); ax[0].set_ylim(-0.4, 1.05); ax[0].axhline(0, color="grey", lw=.5)
    ax[0].set_ylabel("cos(predicted, true) convention vector, held-out family"); ax[0].set_title(f"Held-out families ({len(test)} test / {len(train)} train, split {args.split})"); ax[0].legend(fontsize=7.5, loc="lower left")
    Ls = [r["read_layer"] for r in rows]
    ax[1].plot(Ls, [r["cos_centroid_mean"] for r in rows], "-o", color="#8e44ad", label="centroid map: cos of held-out centroids (train-mean centred)")
    ax[1].plot(Ls, [r["cos_conv_mean"] for r in rows], "-o", color="#1f6c80", label="centroid map: convention cos")
    ax[1].plot(Ls, [r["prompt_cos_conv_mean"] for r in rows], "-s", color="#7fb3c4", label="per-prompt map: convention cos")
    ax[1].plot(Ls, [r["cos_conv_baseline"] for r in rows], "--", color="#b8860b", label="mean-vector baseline"); ax[1].plot(Ls, [r["cos_conv_shuffled"] for r in rows], "--", color="grey", label="shuffled")
    ax[1].set_xticks(Ls); ax[1].set_xlabel("read layer"); ax[1].set_ylim(-0.4, 1.05); ax[1].axhline(0, color="grey", lw=.5); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7); ax[1].set_title("read-layer sweep")
    fig.suptitle("Read→write map fitted on (family, pole) MEAN activations (110 points) vs per-prompt map — coding-convention pool, Qwen2.5-7B base", fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUT / f"centroid_map_{args.split}.png", dpi=150); plt.close(fig)
    print("->", OUT)


if __name__ == "__main__":
    main()
