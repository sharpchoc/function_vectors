#!/usr/bin/env python
"""Pooled sparse PC-DIRECTION selection on the 69-task-run train tasks (GPU).

Same protocol as sandbox/ext_steerability/train_sparse_pooled_ext.py, but the unit
dictionary is the top-512 UNCENTERED PCs of the train per-prompt FV stack instead of the
448 heads: contributions D[t, i, :] = (v_t . PC_i) PC_i where v_t is task t's mean FV over
the 37 pooled-selected heads (built from means.pt exactly as eval_ext.py does). Learn
c in [0,1]^512 with -log p(full label) + lambda*||c||_1, injection at the cue token @L9;
lambda by 5-fold task CV (weighted-c fold eval, strict best); selection c > 0.8.
Modes: cv (shardable via --lambdas/--folds), final, smoke.
"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.sandbox.isolation_upper_bound.run_task import (  # noqa: E402
    build_contributions_single, eval_points_fixed_v, load_records, record_to_point)
from src.sandbox.sparse_head_selection.train_sparse_heads import split_earlystop, train_c  # noqa: E402
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed  # noqa: E402
from src.utils.paths import ARTIFACTS_ROOT  # noqa: E402

RUN_ROOT = ARTIFACTS_ROOT / "69_task_run"
LAMBDAS = [0.005, 0.01, 0.05, 0.2]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["cv", "final", "smoke"], default="cv")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--head_selection_path", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox" / "ext_steerability" / "prunedfail_seed43"
                   / "pooled_sparse" / "selection.json")
    p.add_argument("--means_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox" / "ext_steerability")
    p.add_argument("--pc_basis_path", type=Path, default=RUN_ROOT / "pc_sparse" / "pc_basis_uncentered.pt")
    p.add_argument("--out_root", type=Path, default=RUN_ROOT / "pc_sparse")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inject_layer", type=int, default=9)
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


def task_folds(train_tasks, kfolds, seed):
    order = np.random.RandomState(seed).permutation(len(train_tasks))
    return [sorted(train_tasks[i] for i in fold) for fold in np.array_split(order, kfolds)]


def weighted_fold_acc(model, model_config, tokenizer, c, D, task_index, fold_tasks,
                      points_by_task, args):
    accs = []
    for t in fold_tasks:
        v = (c.unsqueeze(1) * D[task_index[t]]).sum(dim=0)
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

    head_sel = json.load(open(args.head_selection_path))
    head_flat = torch.tensor(head_sel["selected_flat"])
    basis = torch.load(args.pc_basis_path, map_location="cpu", weights_only=False)

    if args.mode == "smoke":
        args.max_epochs = 4
        args.lambdas = args.lambdas[:1]
        args.folds = [0]

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()
    torch.set_grad_enabled(False)

    pcs = basis["pcs"].float().to(model.device)  # (512, 4096)
    # task FVs from as-loaded weights (identical to eval_ext), then PC-projection dictionary
    D_rows = []
    for t in train_tasks:
        means = torch.load(args.means_root / t / "means.pt", map_location="cpu", weights_only=False)
        C_t = build_contributions_single(means["head_means"], model, model_config)
        v_t = C_t[head_flat.to(C_t.device)].sum(dim=0).float()  # (4096,)
        # consistency gate vs per-prompt fv mean
        pp = torch.load(RUN_ROOT / "perprompt_fvs" / f"{t}.pt", map_location="cpu", weights_only=False)
        v_pp = pp["fv"].float().mean(dim=0).to(v_t.device)
        cos = torch.nn.functional.cosine_similarity(v_t, v_pp, dim=0).item()
        assert cos > 0.999, f"CONSISTENCY GATE FAILED for {t}: cos={cos:.6f} - HARD STOP, report to user"
        coef = pcs @ v_t                       # (512,)
        D_rows.append(coef.unsqueeze(1) * pcs)  # (512, 4096)
    D = torch.stack(D_rows)  # (T, 512, 4096) on device
    print(f"D built: {tuple(D.shape)}; consistency gate passed on {len(train_tasks)} tasks", flush=True)
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
                out = args.out_root / f"lambda{lam:g}_fold{fi}.pt"
                if out.exists():
                    print(f"skip existing {out.name}", flush=True)
                    continue
                fold_set = set(folds[fi])
                pool = [p for t in train_tasks if t not in fold_set for p in points_by_task[t]]
                run_seed = args.seed + 200000 + int(round(lam * 1e4)) * 100 + fi
                tr, es = split_earlystop(pool, args.earlystop_frac, run_seed)
                torch.set_grad_enabled(True)
                c, history, best_epoch = train_c(model, model_config, tokenizer, tr, es,
                                                 D, task_index, lam, tc_args(args), run_seed,
                                                 desc=f"pc lam={lam:g} fold{fi}")
                torch.set_grad_enabled(False)
                acc = weighted_fold_acc(model, model_config, tokenizer, c, D, task_index,
                                        sorted(fold_set), points_by_task, args)
                torch.save({"lambda": lam, "fold": fi, "c": c.cpu(), "fold_acc": acc,
                            "fold_tasks": sorted(fold_set), "best_epoch": best_epoch,
                            "fold_eval": f"weighted_c@L{args.inject_layer}",
                            "inject_layer": args.inject_layer, "unit": "pc512_uncentered"}, out)
                print(f"[pc lam={lam:g} fold{fi}] heldout-task acc={acc:.3f} "
                      f"best_epoch={best_epoch}", flush=True)
        return

    per_lambda = {}
    for lam in LAMBDAS:
        accs = []
        for fi in range(args.kfolds):
            fp = args.out_root / f"lambda{lam:g}_fold{fi}.pt"
            assert fp.exists(), f"final: missing {fp}"
            accs.append(torch.load(fp, map_location="cpu", weights_only=False)["fold_acc"])
        per_lambda[lam] = float(np.mean(accs))
    best = max(per_lambda.values())
    chosen = min(l for l in LAMBDAS if per_lambda[l] == best)
    print(f"per-lambda CV: {per_lambda} -> chosen {chosen:g}", flush=True)

    all_points = [p for t in train_tasks for p in points_by_task[t]]
    run_seed = args.seed + 1999
    tr, es = split_earlystop(all_points, args.earlystop_frac, run_seed)
    torch.set_grad_enabled(True)
    c_final, history, best_epoch = train_c(model, model_config, tokenizer, tr, es, D,
                                           task_index, chosen, tc_args(args), run_seed,
                                           desc=f"pc FINAL lam={chosen:g}")
    torch.set_grad_enabled(False)
    sel = torch.nonzero(c_final > args.c_high).flatten().tolist()
    fallback = False
    if not sel:
        sel = torch.argsort(c_final, descending=True)[:10].tolist()
        fallback = True
    selection = {"unit": "pc512_uncentered", "train_metric": "zeroshot",
                 "inject_layer": args.inject_layer, "chosen_lambda": chosen,
                 "per_lambda": per_lambda, "c_high": args.c_high,
                 "n_selected": len(sel), "fallback_top10": fallback,
                 "selected_pcs": sorted(sel),
                 "selected_c": {int(i): round(float(c_final[i]), 4) for i in sel},
                 "final_best_epoch": best_epoch, "c_max": round(float(c_final.max()), 4),
                 "pc_basis_path": str(args.pc_basis_path),
                 "head_selection_path": str(args.head_selection_path),
                 "split_path": str(args.split_path), "n_train_tasks": len(train_tasks),
                 "points_per_task": args.points_per_task}
    torch.save({"c": c_final.cpu(), "history": history}, args.out_root / "coeffs_final.pt")
    with open(args.out_root / "selection.json", "w") as f:
        json.dump(selection, f, indent=1)
    print(f"FINAL: lam={chosen:g} n_selected={len(sel)}{' (FALLBACK)' if fallback else ''} "
          f"best_epoch={best_epoch} c_max={selection['c_max']}", flush=True)


if __name__ == "__main__":
    main()
