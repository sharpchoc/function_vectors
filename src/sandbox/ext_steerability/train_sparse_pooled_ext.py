#!/usr/bin/env python
"""SANDBOX: pooled sparse-optimization head selection on the extended_steerable_90 TRAIN tasks.

Train metric = zero-shot: learn c in [0,1]^448 with loss -log p(full label) + lambda*||c||_1,
injection once at the cue token of "Q: x\nA:" prompts at block --inject_layer output;
100 zero-shot datapoints per train task (from train_zeroshot.json). Head means from the
ext captures (fixed 10-shot train prompts). lambda by 5-fold CV over TASKS (seed-42 fold
split, strict-best mean held-out-task accuracy, fold eval with the WEIGHTED c vector);
final retrain on all train tasks; selection = heads with c > --c_high (top-10-by-c fallback).

Modes: cv (one or more (lambda, fold) cells via --lambdas/--folds - pod sharding),
final (requires all fold artifacts; picks lambda, retrains, writes selection.json),
smoke (single cell, reduced epochs, separate out_root recommended).
"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.sandbox.isolation_upper_bound.run_task import (
    build_contributions_single,
    eval_points_fixed_v,
    load_records,
    record_to_point,
)
from src.sandbox.sparse_head_selection.train_sparse_heads import split_earlystop, train_c
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.paths import ARTIFACTS_ROOT

DEFAULT_OUT = ARTIFACTS_ROOT / "sandbox" / "ext_steerability"
LAMBDAS = [0.005, 0.01, 0.05, 0.2]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["cv", "final", "smoke"], default="cv")
    p.add_argument("--split_path", type=Path, default=REPO_ROOT / "task_splits" / "extended_steerable_90.json")
    p.add_argument("--prompts_root", type=Path, default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inject_layer", type=int, default=9)
    p.add_argument("--points_per_task", type=int, default=100)
    p.add_argument("--kfolds", type=int, default=5)
    p.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS,
                   help="Subset for pod sharding (cv mode); final mode uses the FULL grid.")
    p.add_argument("--folds", type=int, nargs="+", default=None,
                   help="Fold indices subset for pod sharding (cv mode).")
    p.add_argument("--c_high", type=float, default=0.8)
    # pooled-run hyperparameters (~57 batches/epoch at 7200 points - ample steps)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--micro_batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--earlystop_frac", type=float, default=0.1)
    p.add_argument("--init_c", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.2)
    return p.parse_args()


def tc_args(args):
    return types.SimpleNamespace(
        init_c=args.init_c, lr=args.lr, max_epochs=args.max_epochs,
        micro_batch_size=args.micro_batch_size, batch_size=args.batch_size,
        inject_layer=args.inject_layer, patience=args.patience,
        threshold=args.threshold, earlystop_frac=args.earlystop_frac)


def task_folds(train_tasks, kfolds, seed):
    order = np.random.RandomState(seed).permutation(len(train_tasks))
    return [sorted(train_tasks[i] for i in fold) for fold in np.array_split(order, kfolds)]


def weighted_fold_acc(model, model_config, tokenizer, c, C, task_index, fold_tasks,
                      points_by_task, args):
    accs = []
    for t in fold_tasks:
        v = (c.unsqueeze(1) * C[task_index[t]]).sum(dim=0)
        accs.append(eval_points_fixed_v(model, model_config, tokenizer, points_by_task[t],
                                        v, args.inject_layer))
    return float(np.mean(accs))


def main():
    args = parse_args()
    set_seed(args.seed)
    split = json.load(open(args.split_path))
    train_tasks = split["train_tasks"]
    task_index = {t: i for i, t in enumerate(train_tasks)}
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "pooled_sparse").mkdir(exist_ok=True)

    if args.mode == "smoke":
        args.max_epochs = 4
        args.lambdas = args.lambdas[:1]
        args.folds = [0]

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()
    torch.set_grad_enabled(False)

    # contributions from as-loaded weights (before bf16 cast), stacked (T,448,4096)
    C = torch.stack([
        build_contributions_single(
            torch.load(args.out_root / t / "means.pt", map_location="cpu",
                       weights_only=False)["head_means"], model, model_config)
        for t in train_tasks])
    model = model.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)

    points_by_task = {}
    for t in train_tasks:
        recs = load_records(args, t, "train_zeroshot")[:args.points_per_task]
        points_by_task[t] = [record_to_point(r, tokenizer, model_config) for r in recs]
    print(f"train tasks: {len(train_tasks)}, points: "
          f"{sum(len(v) for v in points_by_task.values())}", flush=True)

    folds = task_folds(train_tasks, args.kfolds, args.seed)

    if args.mode in ("cv", "smoke"):
        fold_ids = args.folds if args.folds is not None else list(range(args.kfolds))
        for lam in args.lambdas:
            for fi in fold_ids:
                out = args.out_root / "pooled_sparse" / f"lambda{lam:g}_fold{fi}.pt"
                if out.exists():
                    print(f"skip existing {out.name}", flush=True)
                    continue
                fold_set = set(folds[fi])
                pool = [p for t in train_tasks if t not in fold_set for p in points_by_task[t]]
                run_seed = args.seed + 100000 + int(round(lam * 1e4)) * 100 + fi
                tr, es = split_earlystop(pool, args.earlystop_frac, run_seed)
                torch.set_grad_enabled(True)
                c, history, best_epoch = train_c(model, model_config, tokenizer, tr, es,
                                                 C, task_index, lam, tc_args(args), run_seed,
                                                 desc=f"lam={lam:g} fold{fi}")
                torch.set_grad_enabled(False)
                acc = weighted_fold_acc(model, model_config, tokenizer, c, C, task_index,
                                        sorted(fold_set), points_by_task, args)
                torch.save({"lambda": lam, "fold": fi, "c": c.cpu(), "fold_acc": acc,
                            "fold_tasks": sorted(fold_set), "best_epoch": best_epoch,
                            "fold_eval": "weighted_c@L9"}, out)
                print(f"[lam={lam:g} fold{fi}] heldout-task acc={acc:.3f} "
                      f"best_epoch={best_epoch}", flush=True)
        return

    # final: strict-best lambda from all fold artifacts (grid = --lambdas, default LAMBDAS),
    # retrain on all train tasks
    per_lambda = {}
    for lam in args.lambdas:
        accs = []
        for fi in range(args.kfolds):
            fp = args.out_root / "pooled_sparse" / f"lambda{lam:g}_fold{fi}.pt"
            assert fp.exists(), f"final: missing {fp}"
            accs.append(torch.load(fp, map_location="cpu", weights_only=False)["fold_acc"])
        per_lambda[lam] = float(np.mean(accs))
    best = max(per_lambda.values())
    chosen = min(l for l in args.lambdas if per_lambda[l] == best)
    print(f"per-lambda CV: {per_lambda} -> chosen {chosen:g}", flush=True)

    all_points = [p for t in train_tasks for p in points_by_task[t]]
    run_seed = args.seed + 999
    tr, es = split_earlystop(all_points, args.earlystop_frac, run_seed)
    torch.set_grad_enabled(True)
    c_final, history, best_epoch = train_c(model, model_config, tokenizer, tr, es, C,
                                           task_index, chosen, tc_args(args), run_seed,
                                           desc=f"FINAL lam={chosen:g}")
    torch.set_grad_enabled(False)
    sel = torch.nonzero(c_final > args.c_high).flatten().tolist()
    fallback = False
    if not sel:
        sel = torch.argsort(c_final, descending=True)[:10].tolist()
        fallback = True
    heads = [(i // model_config["n_heads"], i % model_config["n_heads"],
              round(float(c_final[i]), 4)) for i in
             sorted(sel, key=lambda i: -float(c_final[i]))]
    selection = {"sandbox": True, "train_metric": "zeroshot", "inject_layer": args.inject_layer,
                 "chosen_lambda": chosen, "per_lambda": per_lambda, "c_high": args.c_high,
                 "n_selected": len(sel), "fallback_top10": fallback,
                 "selected_heads": heads, "selected_flat": sel,
                 "final_best_epoch": best_epoch,
                 "split_path": str(args.split_path.relative_to(REPO_ROOT)),
                 "n_train_tasks": len(train_tasks),
                 "points_per_task": args.points_per_task}
    torch.save({"c": c_final.cpu(), "history": history}, args.out_root / "pooled_sparse" / "coeffs_final.pt")
    with open(args.out_root / "pooled_sparse" / "selection.json", "w") as f:
        json.dump(selection, f, indent=1)
    print(f"FINAL: lam={chosen:g} n_selected={len(sel)}{' (FALLBACK)' if fallback else ''} "
          f"best_epoch={best_epoch}", flush=True)


if __name__ == "__main__":
    main()
