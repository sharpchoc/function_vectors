#!/usr/bin/env python
"""SANDBOX **TRIAL**: staged sparse head selection by competence-ordered task groups.

TRIAL of a user hypothesis (2026-08-15) - NOT the standard pooled recipe. The 72 train
tasks of extended_steerable_90 are sorted by 6-shot accuracy (desc) and split into 5
evenly sized groups (15,15,14,14,14). Stage k optimizes c in [0,1]^{free heads} on group
k's zero-shot prompts with the heads selected in stages 1..k-1 FROZEN IN (their unweighted
sum added as a constant per-task vector during training and eval); L1 applies only to the
new coefficients; lambda per stage by LEAVE-ONE-TASK-OUT CV over the group's tasks
(weighted-c fold eval incl. frozen contribution, strict-best, tie -> smaller lambda).
Selection per stage: free heads with c > --c_high; an EMPTY increment is allowed and
recorded (it means the frozen set already covers the group). Health gate per stage: c
must separate (c_max >= 0.9) - else the stage aborts loudly rather than silently
harvesting a truncated run (lesson of 2026-08-13).

Modes: cv (shardable via --lambdas/--folds), final (LOTO reduce + stage retrain +
selection_stage<k>.json + cumulative selection.json).
"""
import argparse
import json
import math
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
    load_records,
    record_to_point,
)
from src.sandbox.sparse_head_selection.train_sparse_heads import (
    batch_label_logprobs,
    make_batches,
    split_earlystop,
)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.paths import ARTIFACTS_ROOT

DEFAULT_OUT = ARTIFACTS_ROOT / "sandbox" / "ext_steerability" / "staged_sparse_trial"
MEANS_ROOT = ARTIFACTS_ROOT / "sandbox" / "ext_steerability"
LAMBDAS = [0.005, 0.01, 0.05, 0.2]
N_GROUPS = 5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["cv", "final"], default="cv")
    p.add_argument("--stage", type=int, required=True, help="1-based group index.")
    p.add_argument("--split_path", type=Path, default=REPO_ROOT / "task_splits" / "extended_steerable_90.json")
    p.add_argument("--prompts_root", type=Path, default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--means_root", type=Path, default=MEANS_ROOT)
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inject_layer", type=int, default=9)
    p.add_argument("--points_per_task", type=int, default=100)
    p.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    p.add_argument("--folds", type=int, nargs="+", default=None,
                   help="LOTO fold indices (task indices within the group) for pod sharding.")
    p.add_argument("--c_high", type=float, default=0.8)
    p.add_argument("--c_sep_gate", type=float, default=0.9,
                   help="Stage health gate: max learned c must reach this, else hard stop.")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--micro_batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--earlystop_frac", type=float, default=0.1)
    p.add_argument("--init_c", type=float, default=0.5)
    return p.parse_args()


def make_groups(split):
    """5 evenly sized groups of train tasks, ordered by 6-shot acc DESC (group 1 = best)."""
    tasks = sorted(split["train_tasks"], key=lambda t: (-split["acc6"][t], t))
    sizes = [len(tasks) // N_GROUPS + (1 if i < len(tasks) % N_GROUPS else 0)
             for i in range(N_GROUPS)]
    groups, pos = [], 0
    for s in sizes:
        groups.append(tasks[pos:pos + s])
        pos += s
    return groups


def frozen_heads_before(out_root, stage):
    frozen = []
    for k in range(1, stage):
        sel = json.load(open(out_root / f"selection_stage{k}.json"))
        frozen += sel["selected_flat_new"]
    assert len(frozen) == len(set(frozen))
    return frozen


def train_c_staged(model, model_config, tokenizer, train_points, es_points, C_free, const_v,
                   task_index, lam, args, run_seed, desc=""):
    """train_c adapted for the staged trial: v = const_v[task] + sum_h c_h * C_free[task,h].
    C_free: (T, n_free, resid); const_v: (T, resid) frozen-head contribution (no grad)."""
    device = C_free.device
    n_units = C_free.shape[1]
    c = torch.full((n_units,), args.init_c, device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.AdamW([c], lr=args.lr)
    rng = np.random.RandomState(run_seed)
    best = {"es_loss": math.inf, "c": c.detach().clone(), "epoch": -1}
    since_best = 0
    for epoch in range(args.max_epochs):
        for batch in make_batches(train_points, args.batch_size, rng=rng):
            opt.zero_grad(set_to_none=True)
            for mb in range(0, len(batch), args.micro_batch_size):
                micro = batch[mb:mb + args.micro_batch_size]
                t_idx = torch.tensor([task_index[b["task"]] for b in micro], device=device)
                v = const_v[t_idx] + torch.einsum("h,bhd->bd", c, C_free[t_idx])
                nll, _ = batch_label_logprobs(model, model_config, tokenizer, micro, v=v,
                                              inject_layer=args.inject_layer)
                (nll.sum() / len(batch)).backward()
            (lam * c.abs().sum()).backward()
            assert c.grad is not None and torch.isfinite(c.grad).all()
            opt.step()
            with torch.no_grad():
                c.clamp_(0.0, 1.0)
        es_loss = eval_nll(model, model_config, tokenizer, es_points, c.detach(), C_free,
                           const_v, task_index, args)
        print(f"  [{desc}] epoch {epoch}: es_nll={es_loss:.4f} "
              f"active={(c.detach() > 0.2).sum().item()}", flush=True)
        if es_loss < best["es_loss"] - 1e-4:
            best = {"es_loss": es_loss, "c": c.detach().clone(), "epoch": epoch}
            since_best = 0
        else:
            since_best += 1
            if since_best >= args.patience:
                break
    return best["c"], best["epoch"]


def eval_nll(model, model_config, tokenizer, points, c, C_free, const_v, task_index, args):
    tot = 0.0
    with torch.no_grad():
        for batch in make_batches(points, args.batch_size):
            t_idx = torch.tensor([task_index[b["task"]] for b in batch], device=C_free.device)
            v = const_v[t_idx] + torch.einsum("h,bhd->bd", c, C_free[t_idx])
            nll, _ = batch_label_logprobs(model, model_config, tokenizer, batch, v=v,
                                          inject_layer=args.inject_layer)
            tot += nll.sum().item()
    return tot / len(points)


def eval_acc_task(model, model_config, tokenizer, points, v, inject_layer, batch_size=32):
    n = 0
    with torch.no_grad():
        for s in range(0, len(points), batch_size):
            batch = points[s:s + batch_size]
            vb = v.unsqueeze(0).expand(len(batch), -1)
            _, accs = batch_label_logprobs(model, model_config, tokenizer, batch, v=vb,
                                           inject_layer=inject_layer)
            n += sum(accs)
    return n / len(points)


def main():
    args = parse_args()
    set_seed(args.seed)
    split = json.load(open(args.split_path))
    groups = make_groups(split)
    group = groups[args.stage - 1]
    args.out_root.mkdir(parents=True, exist_ok=True)
    with open(args.out_root / "groups.json", "w") as f:
        json.dump({"n_groups": N_GROUPS, "order": "acc6 desc",
                   "groups": groups,
                   "acc6": {t: split["acc6"][t] for g in groups for t in g}}, f, indent=1)

    frozen = frozen_heads_before(args.out_root, args.stage)
    free_idx = [i for i in range(448) if i not in set(frozen)]
    print(f"stage {args.stage}: group={len(group)} tasks, frozen={len(frozen)}, "
          f"free={len(free_idx)}", flush=True)

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()
    torch.set_grad_enabled(False)
    task_index = {t: i for i, t in enumerate(group)}
    C = torch.stack([
        build_contributions_single(
            torch.load(args.means_root / t / "means.pt", map_location="cpu",
                       weights_only=False)["head_means"], model, model_config)
        for t in group])                          # (G, 448, resid) on device
    model = model.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)

    free_t = torch.tensor(free_idx, device=C.device)
    C_free = C[:, free_t, :]
    const_v = (C[:, torch.tensor(frozen, device=C.device), :].sum(dim=1)
               if frozen else torch.zeros(C.shape[0], C.shape[2], device=C.device))

    points_by_task = {}
    for t in group:
        recs = load_records(args, t, "train_zeroshot")[:args.points_per_task]
        points_by_task[t] = [record_to_point(r, tokenizer, model_config) for r in recs]

    if args.mode == "cv":
        fold_ids = args.folds if args.folds is not None else list(range(len(group)))
        for lam in args.lambdas:
            for fi in fold_ids:
                out = args.out_root / f"stage{args.stage}_lambda{lam:g}_fold{fi}.pt"
                if out.exists():
                    print(f"skip {out.name}", flush=True)
                    continue
                fold_task = group[fi]
                pool = [p for t in group if t != fold_task for p in points_by_task[t]]
                run_seed = args.seed + 100000 * args.stage + int(round(lam * 1e4)) * 100 + fi
                tr, es = split_earlystop(pool, args.earlystop_frac, run_seed)
                torch.set_grad_enabled(True)
                c, best_epoch = train_c_staged(model, model_config, tokenizer, tr, es, C_free,
                                               const_v, task_index, lam, args, run_seed,
                                               desc=f"s{args.stage} lam={lam:g} f{fi}")
                torch.set_grad_enabled(False)
                ti = task_index[fold_task]
                v = const_v[ti] + torch.einsum("h,hd->d", c, C_free[ti])
                acc = eval_acc_task(model, model_config, tokenizer, points_by_task[fold_task],
                                    v, args.inject_layer)
                torch.save({"stage": args.stage, "lambda": lam, "fold": fi,
                            "fold_task": fold_task, "c": c.cpu(), "fold_acc": acc,
                            "best_epoch": best_epoch, "c_max": float(c.max())}, out)
                print(f"[s{args.stage} lam={lam:g} f{fi}={fold_task}] acc={acc:.3f} "
                      f"c_max={c.max():.3f} best_epoch={best_epoch}", flush=True)
        return

    # final: LOTO reduce over this stage's folds, retrain on whole group
    per_lambda = {}
    for lam in args.lambdas:
        accs, cmaxes = [], []
        for fi in range(len(group)):
            fp = args.out_root / f"stage{args.stage}_lambda{lam:g}_fold{fi}.pt"
            assert fp.exists(), f"missing {fp}"
            d = torch.load(fp, map_location="cpu", weights_only=False)
            accs.append(d["fold_acc"]); cmaxes.append(d["c_max"])
        per_lambda[lam] = float(np.mean(accs))
    best_acc = max(per_lambda.values())
    chosen = min(l for l in args.lambdas if per_lambda[l] == best_acc)
    print(f"stage {args.stage} per-lambda: {per_lambda} -> {chosen:g}", flush=True)

    all_points = [p for t in group for p in points_by_task[t]]
    run_seed = args.seed + 100000 * args.stage + 999
    tr, es = split_earlystop(all_points, args.earlystop_frac, run_seed)
    torch.set_grad_enabled(True)
    c_final, best_epoch = train_c_staged(model, model_config, tokenizer, tr, es, C_free,
                                         const_v, task_index, chosen, args, run_seed,
                                         desc=f"s{args.stage} FINAL lam={chosen:g}")
    torch.set_grad_enabled(False)
    cmax = float(c_final.max())
    assert cmax >= args.c_sep_gate, \
        (f"STAGE {args.stage} HEALTH GATE FAILED: c_max={cmax:.3f} < {args.c_sep_gate} - "
         f"training did not separate; do NOT trust the threshold (2026-08-13 lesson).")
    sel_local = torch.nonzero(c_final > args.c_high).flatten().tolist()
    sel_new = sorted(int(free_idx[i]) for i in sel_local)
    heads_new = [(i // 16, i % 16,
                  round(float(c_final[free_idx.index(i)]), 4)) for i in sel_new]
    out = {"TRIAL": "staged_sparse_by_competence_groups",
           "stage": args.stage, "group_tasks": group,
           "chosen_lambda": chosen, "per_lambda": per_lambda,
           "c_high": args.c_high, "c_max": cmax, "final_best_epoch": best_epoch,
           "n_frozen_in": len(frozen), "n_new": len(sel_new),
           "selected_flat_new": sel_new, "selected_heads_new": heads_new}
    with open(args.out_root / f"selection_stage{args.stage}.json", "w") as f:
        json.dump(out, f, indent=1)
    torch.save({"c": c_final.cpu(), "free_idx": free_idx},
               args.out_root / f"coeffs_stage{args.stage}.pt")

    cumulative = frozen + sel_new
    heads_cum = [(i // 16, i % 16) for i in cumulative]
    with open(args.out_root / "selection.json", "w") as f:
        json.dump({"TRIAL": "staged_sparse_by_competence_groups",
                   "stages_done": args.stage, "n_selected": len(cumulative),
                   "selected_flat": cumulative,
                   "selected_heads": [[l, h, None] for l, h in heads_cum],
                   "chosen_lambda": {f"stage{k}": json.load(open(args.out_root / f"selection_stage{k}.json"))["chosen_lambda"]
                                     for k in range(1, args.stage + 1)},
                   "stage_of_head": {str(i): k for k in range(1, args.stage + 1)
                                     for i in json.load(open(args.out_root / f"selection_stage{k}.json"))["selected_flat_new"]}},
                  f, indent=1)
    print(f"STAGE {args.stage} DONE: +{len(sel_new)} new heads (cumulative {len(cumulative)}), "
          f"c_max={cmax:.3f}", flush=True)


if __name__ == "__main__":
    main()
