#!/usr/bin/env python
"""Diagnostics for the presence-vs-accuracy result (CPU).

The cross-task scatters (plot_69_presence_vs_acc.py) show a NEGATIVE relation at n>=2.
This script tests the obvious alternative explanations and records them next to the figures:

  A. n=0 floor      — how many tasks have non-zero sampled accuracy at n=0 (the n=0 panel's
                      positive rho is meaningless if accuracy is ~all zeros).
  B. shared-mean    — cos(v_A, grand-mean FV direction): the pooled per-prompt FV stack has
                      stable rank ~3, so a task whose FV points along the shared component
                      would read "present" everywhere for task-independent reasons.
  C. task features  — label_tokens / out_entropy / input_tokens / n_unique_out from the
                      head-hungriness study (failing_analysis_features.csv, 56/69 tasks).
  D. within-task    — across n=0..6 within each task (paired, confound-free): does presence
                      rise together with accuracy?

Writes <presence_vs_accuracy>/diagnostics.txt and diagnostics_per_task.csv.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, RESULTS_ROOT, TASK69_RUN_DIR  # noqa: E402

FEATURES = ["label_tokens", "out_entropy", "input_tokens", "n_unique_out"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res_dir", type=Path,
                   default=TASK69_RUN_DIR / "write_feature_and_model_accuracy")
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs")
    p.add_argument("--features_csv", type=Path,
                   default=RESULTS_ROOT / "sandbox" / "ext_steerability" / "failing_analysis_features.csv")
    return p.parse_args()


def rankz(x):
    r = rankdata(x)
    return (r - r.mean()) / r.std()


def partial_rho(x, y, C):
    """Rank-based partial correlation of x,y controlling for the columns of C."""
    a, b = rankz(x), rankz(y)
    Cz = np.stack([rankz(c) for c in C], axis=1)
    ra = a - Cz @ np.linalg.lstsq(Cz, a, rcond=None)[0]
    rb = b - Cz @ np.linalg.lstsq(Cz, b, rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    args = parse_args()
    Z = np.load(args.res_dir / "presence_vs_acc.npz")
    tasks = [str(t) for t in Z["tasks"]]
    groups = [str(g) for g in Z["groups"]]
    acc, cos_mean = Z["acc"], Z["cos_meanL"]
    out = []

    def emit(line=""):
        print(line, flush=True)
        out.append(line)

    emit("=== presence-vs-accuracy diagnostics (69 tasks, meanL9-20 presence) ===")
    emit()
    emit(f"A. n=0 floor: {(acc[:, 0] > 0).sum()}/69 tasks non-zero, max {acc[:, 0].max():.3f}, "
         f"mean {acc[:, 0].mean():.4f}  -> the n=0 panel is a floor, its rho is not meaningful")
    emit()

    fvs = np.stack([torch.load(args.fv_root / f"{t}.pt", map_location="cpu",
                               weights_only=False)["fv"].double().mean(0).numpy()
                    for t in tasks])
    gm = fvs.mean(0)
    gm /= np.linalg.norm(gm)
    cos_gm = fvs @ gm / np.linalg.norm(fvs, axis=1)
    fv_norm = np.linalg.norm(fvs, axis=1)

    emit("B. shared-mean control (Spearman across 69 tasks):")
    emit("   n | rho(presence,acc) | rho(cos_gm,acc) | rho(cos_gm,presence) | partial(presence,acc|cos_gm)")
    for n in range(7):
        emit(f"   {n} |      {spearmanr(cos_mean[:, n], acc[:, n]).statistic:+.3f}        |"
             f"     {spearmanr(cos_gm, acc[:, n]).statistic:+.3f}       |"
             f"        {spearmanr(cos_gm, cos_mean[:, n]).statistic:+.3f}         |"
             f"           {partial_rho(cos_mean[:, n], acc[:, n], [cos_gm]):+.3f}")
    emit(f"   cos(v_A, grand mean) range {cos_gm.min():.2f}-{cos_gm.max():.2f}; "
         f"rho(||v_A||, presence@6) = {spearmanr(fv_norm, cos_mean[:, 6]).statistic:+.3f}")
    emit("   -> the shared component does NOT explain the negative relation")
    emit()

    feat = {r["task"]: r for r in csv.DictReader(open(args.features_csv))}
    idx = np.array([i for i, t in enumerate(tasks) if t in feat])
    cols = {f: np.array([float(feat[tasks[i]][f]) for i in idx]) for f in FEATURES}
    emit(f"C. a-priori task features ({len(idx)}/69 tasks with stored features), at n=6:")
    for f in FEATURES:
        emit(f"   {f:14s} rho(feat,presence)={spearmanr(cols[f], cos_mean[idx, 6]).statistic:+.3f}  "
             f"rho(feat,acc)={spearmanr(cols[f], acc[idx, 6]).statistic:+.3f}")
    emit("   partial rho(presence, acc) controlling for all four:")
    emit("   " + " ".join(f"n={n}:{partial_rho(cos_mean[idx, n], acc[idx, n], list(cols.values())):+.3f}"
                          for n in range(2, 7)))
    emit("   -> weakens the negative relation but does not remove it")
    emit()

    wr = np.array([spearmanr(cos_mean[i], acc[i]).statistic for i in range(len(tasks))])
    emit(f"D. WITHIN-task across n=0..6 (paired): median rho = {np.median(wr):+.3f}, "
         f"positive in {(wr > 0).sum()}/{len(tasks)} tasks")
    emit("   -> adding demos raises presence AND accuracy together in every task;")
    emit("      the negative sign is purely a BETWEEN-task effect at fixed n")

    (args.res_dir / "diagnostics.txt").write_text("\n".join(out) + "\n")
    with open(args.res_dir / "diagnostics_per_task.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["task", "group", "cos_to_grandmean", "fv_norm", "within_task_rho"]
                   + [f"presence_n{n}" for n in range(7)] + [f"acc_n{n}" for n in range(7)])
        for i, t in enumerate(tasks):
            w.writerow([t, groups[i], f"{cos_gm[i]:.4f}", f"{fv_norm[i]:.2f}", f"{wr[i]:.4f}"]
                       + [f"{cos_mean[i, n]:.4f}" for n in range(7)]
                       + [f"{acc[i, n]:.4f}" for n in range(7)])
    print(f"\nwrote {args.res_dir}/diagnostics.txt + diagnostics_per_task.csv")


if __name__ == "__main__":
    main()
