#!/usr/bin/env python
"""Sparse selection over the top-40 PC directions of the L6 label-token read feature.

Question: how few of the 40 leading PCs of the 69 per-task L6 label-token means (the basis
plotted in raw_mean_steering/dimensionality) are enough to keep the steering effect?

Gate c in [0,1]^40, SHARED across tasks. Task A's steering vector is the gated projection of
its own read feature onto the basis:

    v_A(c) = sum_j c_j (m_A . v_j) v_j        (c == 1 everywhere -> the 40-PC truncation)

Fit on the 1-shot dummy scaffold "Q: {in-dist input}\nA: _\n\nQ: {query}\nA:", injecting
FIT_ALPHA * v_A(c) at the ' _' token at block-6 output; loss -log p(gold full label) +
lambda*||c||_1 (minimising NLL, i.e. steering must WORK — opposite sign to the ablation
study). 5-fold CV over the 55 TRAIN tasks; lambda chosen by the graded held-out-task
criterion (mean -log p), because exact-match accuracy is degenerate at this site; final
retrain on all train tasks; selection = dims with c > --c_high.

Implementation note: this reuses train_c() unchanged by folding the projection coefficients
into the "contribution" tensor — C[t, j] = FIT_ALPHA * (m_t . v_j) * v_j, so
sum_j c_j C[t, j] is exactly FIT_ALPHA * v_t(c).

Modes: cv (shardable via --lambdas/--folds), rescore (graded scores from saved c), final.
Outputs: artifacts/69_task_run/raw_mean_steering/sparse_pc40/
"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.isolation_upper_bound.run_task import eval_points_fixed_v
from src.sandbox.sparse_head_selection.train_sparse_heads import (
    batch_label_logprobs, split_earlystop, train_c)
from src.utils.eval_utils import get_answer_id
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT

DEFAULT_OUT = ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" / "sparse_pc40"
LAMBDAS = [0.01, 0.05, 0.2, 0.5]
N_PC = 40
FIT_ALPHA = 2.0          # the known optimum for the full read feature at L6
INJECT_LAYER = 6


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["cv", "rescore", "final"], default="cv")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--basis_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" / "pc41_basis.pt")
    p.add_argument("--resid_means_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_resid_means")
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inject_layer", type=int, default=INJECT_LAYER)
    p.add_argument("--points_per_task", type=int, default=100)
    p.add_argument("--kfolds", type=int, default=5)
    p.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    p.add_argument("--folds", type=int, nargs="+", default=None)
    p.add_argument("--c_high", type=float, default=0.8)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--micro_batch_size", type=int, default=16)
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
    """1-shot dummy-label point; cue_idx = the ' _' token (the injection site)."""
    q = str(rec["query"]["input"])
    gold = rec["query"]["output"]
    gold = str(gold[0] if isinstance(gold, list) else gold).strip()
    demo_inp = str(rec["demos"][0]["input"])
    assert demo_inp != q
    pre = f"Q: {demo_inp}\nA:"
    sentence = f"{pre} _\n\nQ: {q}\nA:"
    prompt_ids = tokenizer(sentence).input_ids
    under = tokenizer(" _").input_ids
    assert len(under) == 1
    cue_idx = len(tokenizer(pre).input_ids)          # structural anchor, not a token search
    assert prompt_ids[cue_idx] == under[0]
    label_ids = get_answer_id(sentence, gold, tokenizer)
    if isinstance(label_ids, int):
        label_ids = [label_ids]
    return {"task": task, "prompt_ids": prompt_ids, "label_ids": list(label_ids),
            "cue_idx": cue_idx, "target": gold, "prompt_index": rec["prompt_index"]}


def task_folds(train_tasks, kfolds, seed):
    order = np.random.RandomState(seed).permutation(len(train_tasks))
    return [sorted(train_tasks[i] for i in fold) for fold in np.array_split(order, kfolds)]


def fold_logp(model, model_config, tokenizer, c, C, task_index, fold_tasks, points, args):
    per = {}
    for t in fold_tasks:
        v = (c.unsqueeze(1) * C[task_index[t]]).sum(dim=0)
        vals = []
        with torch.no_grad():
            for i in range(0, len(points[t]), args.micro_batch_size):
                b = points[t][i:i + args.micro_batch_size]
                lp, _ = batch_label_logprobs(model, model_config, tokenizer, b,
                                             v=v.unsqueeze(0).expand(len(b), -1),
                                             inject_layer=args.inject_layer)
                vals += [float(x) for x in lp]
        per[t] = float(np.mean(vals))
    return float(np.mean(list(per.values()))), per


def main():
    args = parse_args()
    set_seed(args.seed)
    split = json.load(open(args.split_path))
    train_tasks = split["train_tasks"]
    task_index = {t: i for i, t in enumerate(train_tasks)}
    (args.out_root).mkdir(parents=True, exist_ok=True)

    basis = torch.load(args.basis_path, map_location="cpu", weights_only=False)
    V = basis["V"][:N_PC].float()                      # (40, 4096), variance-ordered
    assert basis["layer"] == args.inject_layer
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()
    torch.set_grad_enabled(False)

    # C[t, j] = FIT_ALPHA * (m_t . v_j) * v_j  ->  sum_j c_j C[t,j] = FIT_ALPHA * v_t(c)
    Cs = []
    for t in train_tasks:
        m = torch.load(args.resid_means_root / f"{t}.pt", map_location="cpu",
                       weights_only=False)["resid_means"][args.inject_layer].float()
        coef = V @ m                                    # (40,)
        Cs.append((FIT_ALPHA * coef).unsqueeze(1) * V)  # (40, 4096)
    C = torch.stack(Cs).to(model.device)
    print(f"C {tuple(C.shape)} | mean |coef| "
          f"{torch.stack([ (V @ torch.load(args.resid_means_root / f'{t}.pt', map_location='cpu', weights_only=False)['resid_means'][args.inject_layer].float()).abs().mean() for t in train_tasks[:5]]).mean():.3f}",
          flush=True)
    model = model.to(torch.bfloat16)
    for p_ in model.parameters():
        p_.requires_grad_(False)

    points = {}
    for t in train_tasks:
        recs = json.load(open(args.prompts_root / t / "train_prompts.json"))[:args.points_per_task]
        points[t] = [scaffold_point(r, t, tokenizer) for r in recs]
    print(f"{len(train_tasks)} train tasks, {sum(len(v) for v in points.values())} points, "
          f"inject L{args.inject_layer} at the ' _' slot, fit alpha={FIT_ALPHA}", flush=True)
    folds = task_folds(train_tasks, args.kfolds, args.seed)

    if args.mode == "cv":
        for lam in args.lambdas:
            for fi in (args.folds if args.folds is not None else range(args.kfolds)):
                out = args.out_root / f"lambda{lam:g}_fold{fi}.pt"
                if out.exists():
                    print(f"skip {out.name}", flush=True)
                    continue
                fold_set = set(folds[fi])
                pool = [p for t in train_tasks if t not in fold_set for p in points[t]]
                run_seed = args.seed + 100000 + int(round(lam * 1e4)) * 100 + fi
                tr, es = split_earlystop(pool, args.earlystop_frac, run_seed)
                torch.set_grad_enabled(True)
                c, hist, best_ep = train_c(model, model_config, tokenizer, tr, es, C,
                                           task_index, lam, tc_args(args), run_seed,
                                           desc=f"lam={lam:g} fold{fi}")
                torch.set_grad_enabled(False)
                lp, per = fold_logp(model, model_config, tokenizer, c, C, task_index,
                                    sorted(fold_set), points, args)
                acc = float(np.mean([eval_points_fixed_v(
                    model, model_config, tokenizer, points[t],
                    (c.unsqueeze(1) * C[task_index[t]]).sum(dim=0), args.inject_layer)
                    for t in sorted(fold_set)]))
                torch.save({"lambda": lam, "fold": fi, "c": c.cpu(), "fold_logp": lp,
                            "fold_logp_per_task": per, "fold_acc": acc,
                            "fold_tasks": sorted(fold_set), "best_epoch": best_ep,
                            "n_above_chigh": int((c > args.c_high).sum())}, out)
                print(f"[lam={lam:g} fold{fi}] -logp={lp:.3f} acc={acc:.3f} "
                      f"n>{args.c_high}={int((c > args.c_high).sum())}/{N_PC} "
                      f"c_max={float(c.max()):.3f} best_epoch={best_ep}", flush=True)
        return

    # final: pick lambda by the graded criterion, retrain on all train tasks
    per_lambda, per_lambda_acc = {}, {}
    for lam in LAMBDAS:
        lps, accs = [], []
        for fi in range(args.kfolds):
            fp = args.out_root / f"lambda{lam:g}_fold{fi}.pt"
            assert fp.exists(), f"final: missing {fp}"
            d = torch.load(fp, map_location="cpu", weights_only=False)
            lps.append(d["fold_logp"]); accs.append(d["fold_acc"])
        per_lambda[lam] = float(np.mean(lps))
        per_lambda_acc[lam] = float(np.mean(accs))
    best = min(per_lambda.values())
    chosen = max(l for l in LAMBDAS if per_lambda[l] == best)   # ties -> sparser
    print(f"per-lambda -logp: {per_lambda}\n  acc: {per_lambda_acc}\n  chosen {chosen:g}",
          flush=True)

    all_points = [p for t in train_tasks for p in points[t]]
    run_seed = args.seed + 999
    tr, es = split_earlystop(all_points, args.earlystop_frac, run_seed)
    torch.set_grad_enabled(True)
    c_final, hist, best_ep = train_c(model, model_config, tokenizer, tr, es, C, task_index,
                                     chosen, tc_args(args), run_seed,
                                     desc=f"FINAL lam={chosen:g}")
    torch.set_grad_enabled(False)
    sel = torch.nonzero(c_final > args.c_high).flatten().tolist()
    selection = {"experiment": "sparse_pc40_of_L6_label_read_feature",
                 "basis": "top-40 centered PCs of the 69 per-task L6 label-token means",
                 "inject_layer": args.inject_layer, "fit_alpha": FIT_ALPHA,
                 "scaffold": "1-shot dummy '_' label", "chosen_lambda": chosen,
                 "per_lambda_logp": per_lambda, "per_lambda_acc": per_lambda_acc,
                 "c_high": args.c_high, "n_selected": len(sel), "selected_pcs": sel,
                 "c": [round(float(x), 4) for x in c_final],
                 "final_best_epoch": best_ep, "n_train_tasks": len(train_tasks),
                 "points_per_task": args.points_per_task}
    torch.save({"c": c_final.cpu(), "history": hist}, args.out_root / "coeffs_final.pt")
    with open(args.out_root / "selection.json", "w") as f:
        json.dump(selection, f, indent=1)
    print(f"FINAL: lam={chosen:g} selected {len(sel)}/{N_PC} PCs: {sel}", flush=True)


if __name__ == "__main__":
    main()
