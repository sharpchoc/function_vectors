#!/usr/bin/env python
"""Sparse-optimisation head selection for a LABEL-SLOT steering vector (read vector).

Mirror of train_sparse_pooled_ext.py (which selected heads for the cue-token FV against a
zero-shot objective at L9). Differences, per the user's 2026-08-17 design:
  * head means come from the LAST DEMO LABEL TOKEN of the clean 10-shot prompts
    (capture_label_head_means.py) rather than the final cue token;
  * the fitting prompts are the 1-shot dummy-output scaffold
    "Q: {in-distribution input}\nA: _\n\nQ: {query}\nA:" and the injection site is the ' _'
    token at block --inject_layer (default 7) output;
  * everything else is the established convention: candidate v_A(c) = sum_h c_h W_O^h m_A[h],
    loss -log p(full label) + lambda*||c||_1, Adam with c clamped to [0,1], 5-fold CV over
    TASKS with the weighted-c fold eval, strict-best lambda, final retrain on all train
    tasks, selection = heads with c > --c_high (top-10-by-c fallback).

Fit on the 55 train tasks only; held-out tasks are never seen here.

Modes: cv (one or more (lambda, fold) cells via --lambdas/--folds for pod sharding),
rescore (recompute fold scores from saved c vectors under a graded criterion),
final (requires all fold artifacts; picks lambda, retrains, writes selection.json),
smoke (single cell, reduced epochs).

FOLD CRITERION (changed 2026-08-17 after diagnosis): exact-match accuracy is degenerate
here — diagnose_label_cv_metric.py showed every lambda/fold at 0.000, including the
all-448-head vector, because injecting at the label slot improves -log p(gold) by only
~1-1.5 nats out of 12-16, never enough to make the gold the argmax. lambda is therefore
selected by held-out-task mean -log p(gold) (lower is better), the graded version of the
training objective; ties break toward the LARGER lambda (sparser). Accuracy is still
recorded for reference.
"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

# local bootstrap for in-repo runs; a PYTHONPATH-supplied repo also works (staged copies)
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.sandbox.isolation_upper_bound.run_task import (
    build_contributions_single,
    eval_points_fixed_v,
)
from src.sandbox.sparse_head_selection.train_sparse_heads import (
    batch_label_logprobs, split_earlystop, train_c)
from src.utils.eval_utils import get_answer_id
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT

DEFAULT_OUT = ARTIFACTS_ROOT / "69_task_run" / "read_vector_head_selection"
MEANS_ROOT = ARTIFACTS_ROOT / "69_task_run" / "label_head_means"
LAMBDAS = [0.005, 0.01, 0.05, 0.2]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["cv", "rescore", "final", "smoke"], default="cv")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--means_root", type=Path, default=MEANS_ROOT)
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inject_layer", type=int, default=7)
    p.add_argument("--points_per_task", type=int, default=100)
    p.add_argument("--kfolds", type=int, default=5)
    p.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    p.add_argument("--folds", type=int, nargs="+", default=None)
    p.add_argument("--c_high", type=float, default=0.8)
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


def scaffold_point(rec, task, tokenizer):
    """1-shot dummy-output scaffold; cue_idx = the ' _' token (the injection site)."""
    q = str(rec["query"]["input"])
    gold = rec["query"]["output"]
    gold = str(gold[0] if isinstance(gold, list) else gold).strip()
    demo_inp = str(rec["demos"][0]["input"])   # in-distribution, never the query
    assert demo_inp != q
    pre = f"Q: {demo_inp}\nA:"
    sentence = f"{pre} _\n\nQ: {q}\nA:"
    prompt_ids = tokenizer(sentence).input_ids
    under = tokenizer(" _").input_ids
    assert len(under) == 1
    cue_idx = prompt_ids.index(under[0])
    assert cue_idx == len(tokenizer(pre).input_ids), "' _' not directly after the demo cue"
    label_ids = get_answer_id(sentence, gold, tokenizer)
    if isinstance(label_ids, int):
        label_ids = [label_ids]
    return {"task": task, "prompt_ids": prompt_ids, "label_ids": list(label_ids),
            "cue_idx": cue_idx, "target": gold, "prompt_index": rec["prompt_index"]}


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


def weighted_fold_logp(model, model_config, tokenizer, c, C, task_index, fold_tasks,
                       points_by_task, args):
    """Mean -log p(full gold label) on the fold's held-out tasks with the weighted-c vector
    (lower is better) - the graded criterion that replaces exact-match accuracy."""
    per_task = {}
    for t in fold_tasks:
        v = (c.unsqueeze(1) * C[task_index[t]]).sum(dim=0)
        pts = points_by_task[t]
        vals = []
        with torch.no_grad():
            for i in range(0, len(pts), args.micro_batch_size):
                b = pts[i:i + args.micro_batch_size]
                vb = v.unsqueeze(0).expand(len(b), -1)
                lp, _ = batch_label_logprobs(model, model_config, tokenizer, b, v=vb,
                                             inject_layer=args.inject_layer)
                vals += [float(x) for x in lp]
        per_task[t] = float(np.mean(vals))
    return float(np.mean(list(per_task.values()))), per_task


def unsteered_fold_logp(model, model_config, tokenizer, fold_tasks, points_by_task, args):
    per_task = {}
    for t in fold_tasks:
        pts = points_by_task[t]
        vals = []
        with torch.no_grad():
            for i in range(0, len(pts), args.micro_batch_size):
                b = pts[i:i + args.micro_batch_size]
                lp, _ = batch_label_logprobs(model, model_config, tokenizer, b, v=None,
                                             inject_layer=args.inject_layer)
                vals += [float(x) for x in lp]
        per_task[t] = float(np.mean(vals))
    return float(np.mean(list(per_task.values()))), per_task


def main():
    args = parse_args()
    set_seed(args.seed)
    split = json.load(open(args.split_path))
    train_tasks = split["train_tasks"]
    task_index = {t: i for i, t in enumerate(train_tasks)}
    (args.out_root / "pooled_sparse").mkdir(parents=True, exist_ok=True)

    if args.mode == "smoke":
        args.max_epochs = 4
        args.lambdas = args.lambdas[:1]
        args.folds = [0]

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()
    torch.set_grad_enabled(False)

    # contributions from the LABEL-token head means, stacked (T, 448, 4096)
    C = torch.stack([
        build_contributions_single(
            torch.load(args.means_root / f"{t}.pt", map_location="cpu",
                       weights_only=False)["head_means"], model, model_config)
        for t in train_tasks])
    model = model.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)

    points_by_task = {}
    for t in train_tasks:
        recs = json.load(open(args.prompts_root / t / "train_prompts.json"))[:args.points_per_task]
        points_by_task[t] = [scaffold_point(r, t, tokenizer) for r in recs]
    print(f"train tasks: {len(train_tasks)}, points: "
          f"{sum(len(v) for v in points_by_task.values())}, inject_layer={args.inject_layer}",
          flush=True)

    folds = task_folds(train_tasks, args.kfolds, args.seed)

    if args.mode == "rescore":
        # graded re-scoring of already-trained cells; no retraining
        for lam in args.lambdas:
            for fi in (args.folds if args.folds is not None else range(args.kfolds)):
                fp = args.out_root / "pooled_sparse" / f"lambda{lam:g}_fold{fi}.pt"
                assert fp.exists(), f"rescore: missing {fp}"
                d = torch.load(fp, map_location="cpu", weights_only=False)
                if "fold_logp" in d:
                    print(f"skip rescored {fp.name}", flush=True)
                    continue
                c = d["c"].to(C.device)
                mean_lp, per_task = weighted_fold_logp(model, model_config, tokenizer, c, C,
                                                       task_index, d["fold_tasks"],
                                                       points_by_task, args)
                base_lp, base_per = unsteered_fold_logp(model, model_config, tokenizer,
                                                        d["fold_tasks"], points_by_task, args)
                d.update({"fold_logp": mean_lp, "fold_logp_per_task": per_task,
                          "fold_logp_unsteered": base_lp,
                          "fold_logp_unsteered_per_task": base_per,
                          "fold_metric": "mean -log p(full gold label), lower better"})
                torch.save(d, fp)
                print(f"[lam={lam:g} fold{fi}] -logp steered={mean_lp:.3f} "
                      f"unsteered={base_lp:.3f} improvement={base_lp - mean_lp:.3f} "
                      f"(acc={d['fold_acc']:.3f}, n>{args.c_high}={d.get('n_above_chigh')})",
                      flush=True)
        return

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
                # gate: the optimisation must actually move c (the 2026-07 collapse bug)
                assert float(c.max()) > args.threshold, \
                    f"lam={lam:g} fold{fi}: c never left init (max={float(c.max()):.3f}) - HARD STOP"
                acc = weighted_fold_acc(model, model_config, tokenizer, c, C, task_index,
                                        sorted(fold_set), points_by_task, args)
                torch.save({"lambda": lam, "fold": fi, "c": c.cpu(), "fold_acc": acc,
                            "fold_tasks": sorted(fold_set), "best_epoch": best_epoch,
                            "n_above_chigh": int((c > args.c_high).sum()),
                            "fold_eval": f"weighted_c@L{args.inject_layer}"}, out)
                print(f"[lam={lam:g} fold{fi}] heldout-task acc={acc:.3f} "
                      f"best_epoch={best_epoch} c_max={float(c.max()):.3f} "
                      f"n>{args.c_high}={int((c > args.c_high).sum())}", flush=True)
        return

    # final: best lambda by the GRADED fold criterion (mean -log p, lower better);
    # ties break toward the larger (sparser) lambda. Accuracy kept for the record.
    per_lambda, per_lambda_acc, per_lambda_uplift = {}, {}, {}
    for lam in LAMBDAS:
        lps, accs, ups = [], [], []
        for fi in range(args.kfolds):
            fp = args.out_root / "pooled_sparse" / f"lambda{lam:g}_fold{fi}.pt"
            assert fp.exists(), f"final: missing {fp}"
            d = torch.load(fp, map_location="cpu", weights_only=False)
            assert "fold_logp" in d, f"final: {fp.name} not rescored (run --mode rescore)"
            lps.append(d["fold_logp"])
            accs.append(d["fold_acc"])
            ups.append(d["fold_logp_unsteered"] - d["fold_logp"])
        per_lambda[lam] = float(np.mean(lps))
        per_lambda_acc[lam] = float(np.mean(accs))
        per_lambda_uplift[lam] = float(np.mean(ups))
    best = min(per_lambda.values())
    chosen = max(l for l in LAMBDAS if per_lambda[l] == best)
    print(f"per-lambda CV -logp: {per_lambda}\n  uplift vs unsteered: {per_lambda_uplift}"
          f"\n  acc (degenerate): {per_lambda_acc}\n  -> chosen {chosen:g}", flush=True)

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
    selection = {"experiment": "read_vector_head_selection",
                 "train_metric": "1shot_dummy_label_slot",
                 "capture_site": "last token of the 10th demo label (clean 10-shot)",
                 "inject_site": "' _' token of the 1-shot dummy-output scaffold",
                 "inject_layer": args.inject_layer,
                 "chosen_lambda": chosen, "per_lambda_logp": per_lambda,
                 "per_lambda_logp_uplift": per_lambda_uplift,
                 "per_lambda_acc_degenerate": per_lambda_acc,
                 "fold_criterion": "mean -log p(full gold label) on held-out tasks",
                 "c_high": args.c_high,
                 "n_selected": len(sel), "fallback_top10": fallback,
                 "selected_heads": heads, "selected_flat": sel,
                 "final_best_epoch": best_epoch,
                 "split_path": str(args.split_path),
                 "n_train_tasks": len(train_tasks),
                 "points_per_task": args.points_per_task}
    torch.save({"c": c_final.cpu(), "history": history},
               args.out_root / "pooled_sparse" / "coeffs_final.pt")
    with open(args.out_root / "pooled_sparse" / "selection.json", "w") as f:
        json.dump(selection, f, indent=1)
    print(f"FINAL: lam={chosen:g} n_selected={len(sel)}{' (FALLBACK)' if fallback else ''} "
          f"best_epoch={best_epoch}", flush=True)


if __name__ == "__main__":
    main()
