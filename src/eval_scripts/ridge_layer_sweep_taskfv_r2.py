#!/usr/bin/env python
"""Re-score the layer-sweep ridge maps against the TASK FV target (user request 2026-08-18).

Same fitted linear maps as ridge_layer_sweep.py (avg-of-10 label-token X at layer L ->
per-prompt FV; refit on all 55 train tasks at the layer's stored CV-chosen alpha), but the
evaluation target is each task's FV = the mean of its 150 per-prompt FVs, and the R^2
baseline/denominator is the SPLIT-average FV (mean FV across the split's tasks; an all-69
reference is also reported). Two granularities:
  perprompt : every per-prompt prediction scored against its task's FV
  centroid  : predictions averaged within task first (one point per task)
Reported separately for the 55 train tasks (in-sample fit) and 14 held-out tasks.

Output: artifacts/69_task_run/labeltoken_fv_ridge_layer_sweep/taskfv_r2.json and
results/69_task_run/labeltoken_fv_ridge/layer_sweep/taskfv_r2.csv (+ png)
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR, REPO_ROOT  # noqa: E402
from src.eval_scripts.regress_activation_to_fv_fulldim_ridge import (  # noqa: E402
    ridge_eig_prep, ridge_predict)

ACTS = ARTIFACTS_ROOT / "69_task_run" / "label_avg10_L5-15_acts"
FVS = ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs"
SWEEP = ARTIFACTS_ROOT / "69_task_run" / "labeltoken_fv_ridge_layer_sweep"
OUT = TASK69_RUN_DIR / "labeltoken_fv_ridge" / "layer_sweep"
LAYERS = list(range(5, 16))


def r2_vs_ref(targets, preds, ref):
    """Uniform-average per-dim R^2 with an explicit reference vector as the baseline."""
    resid = ((targets - preds) ** 2).sum(dim=0)
    tot = ((targets - ref) ** 2).sum(dim=0)
    ok = tot > 0
    return float((1 - resid[ok] / tot[ok]).mean())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    train_tasks, test_tasks = sorted(split["train_tasks"]), sorted(split["heldout_tasks"])

    fvmean = {}
    for t in train_tasks + test_tasks:
        y = torch.load(FVS / f"{t}.pt", map_location="cpu", weights_only=False)["fv"].double()
        fvmean[t] = y.mean(dim=0)
    ref = {"train": torch.stack([fvmean[t] for t in train_tasks]).mean(0),
           "test": torch.stack([fvmean[t] for t in test_tasks]).mean(0)}
    ref_all = torch.stack([fvmean[t] for t in train_tasks + test_tasks]).mean(0)

    def load_x(t, L):
        a = torch.load(ACTS / f"{t}.pt", map_location="cpu", weights_only=False)
        return a["acts"][:, a["layers"].index(L)].double()

    rows = []
    for L in LAYERS:
        alpha = json.load(open(SWEEP / f"layer_{L}.json"))["best_alpha"]
        Xtr = torch.cat([load_x(t, L) for t in train_tasks]).to(device)
        Ytr = torch.cat([torch.load(FVS / f"{t}.pt", map_location="cpu",
                                    weights_only=False)["fv"].double()
                         for t in train_tasks]).to(device)
        xbar, ybar, ev, evec, c = ridge_eig_prep(Xtr, Ytr)

        res = {"layer": L, "alpha": alpha}
        for grp, tl in (("train", train_tasks), ("test", test_tasks)):
            preds_pp, tgts_pp, cent_p, cent_t = [], [], [], []
            for t in tl:
                p = ridge_predict(load_x(t, L).to(device), xbar, ybar, ev, evec, c, alpha).cpu()
                preds_pp.append(p)
                tgts_pp.append(fvmean[t].unsqueeze(0).expand(len(p), -1))
                cent_p.append(p.mean(dim=0))
                cent_t.append(fvmean[t])
            P = torch.cat(preds_pp); T = torch.cat(tgts_pp)
            Cp = torch.stack(cent_p); Ct = torch.stack(cent_t)
            res[f"{grp}_perprompt"] = round(r2_vs_ref(T, P, ref[grp]), 4)
            res[f"{grp}_centroid"] = round(r2_vs_ref(Ct, Cp, ref[grp]), 4)
            res[f"{grp}_centroid_refall"] = round(r2_vs_ref(Ct, Cp, ref_all), 4)
        rows.append(res)
        print(f"L{L}: train pp={res['train_perprompt']:.3f} cen={res['train_centroid']:.3f} | "
              f"test pp={res['test_perprompt']:.3f} cen={res['test_centroid']:.3f}", flush=True)

    SWEEP.mkdir(parents=True, exist_ok=True)
    with open(SWEEP / "taskfv_r2.json", "w") as f:
        json.dump(rows, f, indent=1)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "taskfv_r2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(rows[0]))
        for r in rows:
            w.writerow(list(r.values()))

    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=150)
    ax.plot(LAYERS, [r["train_perprompt"] for r in rows], "o--", color="tab:blue",
            alpha=0.55, label="train, per-prompt preds vs task FV")
    ax.plot(LAYERS, [r["train_centroid"] for r in rows], "o-", color="tab:blue",
            label="train, centroid preds vs task FV")
    ax.plot(LAYERS, [r["test_perprompt"] for r in rows], "s--", color="tab:red",
            alpha=0.55, label="held-out, per-prompt preds vs task FV")
    ax.plot(LAYERS, [r["test_centroid"] for r in rows], "s-", color="tab:red",
            label="held-out, centroid preds vs task FV")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(LAYERS)
    ax.set_xlabel("layer of the mean label-token activation (X)")
    ax.set_ylabel("R^2 vs TASK FV (baseline = split-average FV)")
    ax.set_title("Layer-sweep ridge maps re-scored against the task FV", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "taskfv_r2.png", bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
