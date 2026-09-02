#!/usr/bin/env python
"""Stage B decodability grid: linear polarity probes over (site role, layer).

Per (property, role in {evid, cue, bg}, layer 0..28): logistic regression nat vs alt on
the captured activations, split BY BASE DOCUMENT (twins share content — never split
within a pair; 3 group-shuffle seeds, 30% test). Layer 0 is the token-embedding
baseline: at evidence sites it quantifies the token-identity shortcut (the R^2=0.245
lesson); cue and background tokens are identity-matched, so ANY decodability there
is contextual state.

Extras: cue-role accuracy by k (prior manifestations) at the best cue layer;
background-role accuracy by token distance since the last manifestation (state decay).

Outputs in results/style_properties/decodability/:
  probe_heatmaps.png  probe_curves.png  probe_grid.csv  probe_grid.npz
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT, STYLE_PROPERTIES_DIR

IN_DIR = ARTIFACTS_ROOT / "style_properties" / "site_acts"
OUT_DIR = STYLE_PROPERTIES_DIR / "decodability"
POOL_PATH = REPO_ROOT / "task_splits" / "style_properties_pool.json"
ROLES = ("evid", "cue", "bg")
N_LAYERS = 29
SEEDS = 3
MAX_PER_CLASS = 1200


def probe_acc(X, y, groups, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(gss.split(X, y, groups))
    if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
        return np.nan
    rng = np.random.RandomState(seed)
    keep = []
    for cls in (0, 1):
        idx = tr[y[tr] == cls]
        keep.append(rng.permutation(idx)[:MAX_PER_CLASS])
    tr = np.concatenate(keep)
    Xtr, Xte = X[tr].astype(np.float32), X[te].astype(np.float32)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=500).fit(sc.transform(Xtr), y[tr])
    return float(clf.score(sc.transform(Xte), y[te]))


def load_prop(name):
    parts = {}
    for pol in ("nat", "alt"):
        z = np.load(IN_DIR / f"{name}__{pol}.npz")
        parts[pol] = {k: z[k] for k in ("acts", "role", "doc", "k", "dist")}
    X = np.concatenate([parts["nat"]["acts"], parts["alt"]["acts"]])  # fp16; cast per slice
    y = np.concatenate([np.zeros(len(parts["nat"]["role"])),
                        np.ones(len(parts["alt"]["role"]))]).astype(int)
    role = np.concatenate([parts["nat"]["role"], parts["alt"]["role"]])
    doc = np.concatenate([parts["nat"]["doc"], parts["alt"]["doc"]])
    kk = np.concatenate([parts["nat"]["k"], parts["alt"]["k"]])
    dist = np.concatenate([parts["nat"]["dist"], parts["alt"]["dist"]])
    return X, y, role, doc, kk, dist


CACHE_DIR = ARTIFACTS_ROOT / "style_properties" / "probe_cache"


def main():
    props = sorted(json.load(open(POOL_PATH))["pass"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    grid = np.full((len(props), len(ROLES), N_LAYERS), np.nan)
    rows, kcurves, dcurves = [], {}, {}

    for pi, name in enumerate(props):
        cache = CACHE_DIR / f"{name}.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            grid[pi] = z["grid"]
            rows.extend(json.loads(str(z["rows"])))
            kcurves[name] = (int(z["klayer"]), list(z["kcurve"]))
            dcurves[name] = (int(z["dlayer"]), list(z["dcurve"]))
            print(f"{name}: cached", flush=True)
            continue
        X, y, role, doc, kk, dist = load_prop(name)
        prop_rows = []
        for ri, rname in enumerate(ROLES):
            m = role == ri
            if m.sum() < 80:
                prop_rows.append(dict(property=name, site_role=rname, best_layer=-1,
                                      best_acc="", emb_baseline_L0=""))
                print(f"{name}/{rname}: only {m.sum()} sites, skipped", flush=True)
                continue
            Xr = X[m]
            for l in range(N_LAYERS):
                accs = [probe_acc(Xr[:, l], y[m], doc[m], s) for s in range(SEEDS)]
                grid[pi, ri, l] = np.nanmean(accs)
            prop_rows.append(dict(property=name, site_role=rname,
                                  best_layer=int(np.nanargmax(grid[pi, ri])),
                                  best_acc=round(float(np.nanmax(grid[pi, ri])), 3),
                                  emb_baseline_L0=round(float(grid[pi, ri, 0]), 3)))
            print(prop_rows[-1], flush=True)
        # k / distance curves (cue and background roles) at their best layers
        if np.all(np.isnan(grid[pi, 1])):
            lbest, kc = -1, [np.nan] * 5
        else:
            lbest = int(np.nanargmax(grid[pi, 1]))
            m = role == 1
            kc = []
            for kb in range(5):
                mm = m & (np.minimum(kk, 4) == kb)
                accs = [probe_acc(X[mm][:, lbest], y[mm], doc[mm], s) for s in range(SEEDS)] \
                    if mm.sum() > 60 else [np.nan]
                kc.append(float(np.nanmean(accs)))
        kcurves[name] = (lbest, kc)
        if np.all(np.isnan(grid[pi, 2])):
            lbg, dc = -1, [np.nan] * 4
        else:
            lbg = int(np.nanargmax(grid[pi, 2]))
            m = role == 2
            dc = []
            for lo, hi in ((1, 15), (16, 40), (41, 90), (91, 100000)):
                mm = m & (dist >= lo) & (dist <= hi)
                accs = [probe_acc(X[mm][:, lbg], y[mm], doc[mm], s) for s in range(SEEDS)] \
                    if mm.sum() > 60 else [np.nan]
                dc.append(float(np.nanmean(accs)))
        dcurves[name] = (lbg, dc)
        rows.extend(prop_rows)
        np.savez(cache, grid=grid[pi], rows=json.dumps(prop_rows),
                 klayer=lbest, kcurve=np.array(kc), dlayer=lbg, dcurve=np.array(dc))
        del X

    # heatmaps: one panel per role, properties x layers
    fig, axes = plt.subplots(1, 3, figsize=(21, 0.45 * len(props) + 2))
    for ri, rname in enumerate(ROLES):
        ax = axes[ri]
        im = ax.imshow(grid[:, ri], vmin=0.5, vmax=1.0, aspect="auto", cmap="viridis")
        ax.set_title(f"{rname} sites — polarity probe accuracy")
        ax.set_yticks(range(len(props)), props, fontsize=7)
        ax.set_xlabel("layer (0 = embedding baseline)")
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "probe_heatmaps.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in props:
        axes[0].plot(range(5), kcurves[name][1], "o-", label=name, alpha=0.7)
        axes[1].plot(range(4), dcurves[name][1], "o-", label=name, alpha=0.7)
    axes[0].set_xlabel("k prior manifestations (4=4+)")
    axes[0].set_title("cue-token decodability vs k (best cue layer)")
    axes[1].set_xticks(range(4), ["1-15", "16-40", "41-90", "91+"])
    axes[1].set_xlabel("token distance since last manifestation")
    axes[1].set_title("background-token decodability vs distance (best bg layer)")
    for ax in axes:
        ax.set_ylim(0.4, 1.02)
        ax.axhline(0.5, color="gray", lw=0.5)
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "probe_curves.png", dpi=150)

    np.savez(OUT_DIR / "probe_grid.npz", grid=grid, props=np.array(props),
             kcurves=json.dumps({k: v[1] for k, v in kcurves.items()}),
             dcurves=json.dumps({k: v[1] for k, v in dcurves.items()}))
    with open(OUT_DIR / "probe_grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
