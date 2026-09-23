#!/usr/bin/env python
"""Refit the pooled-sparse FV head selection on a (pruned) split at a FIXED lambda (no CV).

Replicates the FINAL block of train_sparse_pooled_ext.py exactly (same train_c, run seed 42+999),
skipping only the CV lambda selection. Writes <out_root>/pooled_sparse/{selection.json, coeffs_final.pt,
refit.done} in the identical schema. Moved into the repo from /workspace/msj_icl/icl_69/refit_selection_96.py
(2026-09-23); defaults = the Qwen2.5-7B-Instruct 96-task refit (140 heads, lambda 0.001).

--link_means_from <parent_root>: create <out_root>/<task>/means.pt symlinks to <parent_root>/<task>/means.pt
for every task of the split before fitting (the head means do not depend on the split).
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path

import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.isolation_upper_bound.run_task import build_contributions_single, load_records, record_to_point  # noqa: E402
from src.sandbox.sparse_head_selection.train_sparse_heads import split_earlystop, train_c  # noqa: E402
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed  # noqa: E402
from src.utils.paths import REPO_ROOT, ARTIFACTS_ROOT, QWEN25_MODEL  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_path", type=Path, default=REPO_ROOT / "task_splits" / "qwen25_ext_steerable_96_prunedfail.json")
    ap.add_argument("--prompts_root", type=Path, default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext_qwen25")
    ap.add_argument("--out_root", type=Path, default=ARTIFACTS_ROOT / "sandbox" / "ext_steerability_qwen25_96")
    ap.add_argument("--link_means_from", type=Path, default=None,
                    help="parent capture root whose <task>/means.pt get symlinked into --out_root")
    ap.add_argument("--model_name", default=QWEN25_MODEL)
    ap.add_argument("--lam", type=float, default=0.001)
    ap.add_argument("--lam_note", default="fixed (no CV; reused round-1 CV choice)")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--inject_layer", type=int, default=9)
    ap.add_argument("--points_per_task", type=int, default=100); ap.add_argument("--c_high", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--micro_batch_size", type=int, default=32); ap.add_argument("--max_epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=3); ap.add_argument("--earlystop_frac", type=float, default=0.1)
    ap.add_argument("--init_c", type=float, default=0.5); ap.add_argument("--threshold", type=float, default=0.2)
    return ap.parse_args()


def main():
    a = parse_args()
    set_seed(a.seed)
    split = json.load(open(a.split_path))
    train_tasks = split["train_tasks"]
    if a.link_means_from is not None:
        for t in split["train_tasks"] + split["heldout_tasks"]:
            (a.out_root / t).mkdir(parents=True, exist_ok=True)
            dst = a.out_root / t / "means.pt"
            src = a.link_means_from / t / "means.pt"
            assert src.exists(), src
            if not dst.exists():
                os.symlink(os.path.relpath(src, dst.parent), dst)
    task_index = {t: i for i, t in enumerate(train_tasks)}
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(a.model_name)
    model.eval(); torch.set_grad_enabled(False)
    C = torch.stack([build_contributions_single(
            torch.load(a.out_root / t / "means.pt", map_location="cpu", weights_only=False)["head_means"],
            model, model_config) for t in train_tasks])
    model = model.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    points_by_task = {}
    for t in train_tasks:
        recs = load_records(a, t, "train_zeroshot")[:a.points_per_task]
        points_by_task[t] = [record_to_point(r, tokenizer, model_config) for r in recs]
    print(f"train tasks {len(train_tasks)}, points {sum(len(v) for v in points_by_task.values())}", flush=True)

    tcargs = types.SimpleNamespace(init_c=a.init_c, lr=a.lr, max_epochs=a.max_epochs,
        micro_batch_size=a.micro_batch_size, batch_size=a.batch_size, inject_layer=a.inject_layer,
        patience=a.patience, threshold=a.threshold, earlystop_frac=a.earlystop_frac)
    all_points = [p for t in train_tasks for p in points_by_task[t]]
    run_seed = a.seed + 999
    tr, es = split_earlystop(all_points, a.earlystop_frac, run_seed)
    torch.set_grad_enabled(True)
    c_final, history, best_epoch = train_c(model, model_config, tokenizer, tr, es, C, task_index,
                                           a.lam, tcargs, run_seed, desc=f"FINAL lam={a.lam:g}")
    torch.set_grad_enabled(False)
    sel = torch.nonzero(c_final > a.c_high).flatten().tolist(); fallback = False
    if not sel:
        sel = torch.argsort(c_final, descending=True)[:10].tolist(); fallback = True
    heads = [(i // model_config["n_heads"], i % model_config["n_heads"], round(float(c_final[i]), 4))
             for i in sorted(sel, key=lambda i: -float(c_final[i]))]
    sp = a.split_path
    selection = {"sandbox": True, "train_metric": "zeroshot", "inject_layer": a.inject_layer,
        "chosen_lambda": a.lam, "per_lambda": {str(a.lam): a.lam_note},
        "c_high": a.c_high, "n_selected": len(sel), "fallback_top10": fallback, "selected_heads": heads,
        "selected_flat": sel, "final_best_epoch": best_epoch,
        "split_path": str(sp.relative_to(REPO_ROOT)) if sp.is_relative_to(REPO_ROOT) else str(sp),
        "model_name": a.model_name,
        "n_train_tasks": len(train_tasks), "points_per_task": a.points_per_task}
    (a.out_root / "pooled_sparse").mkdir(parents=True, exist_ok=True)
    torch.save({"c": c_final.cpu(), "history": history}, a.out_root / "pooled_sparse" / "coeffs_final.pt")
    json.dump(selection, open(a.out_root / "pooled_sparse" / "selection.json", "w"), indent=1)
    open(a.out_root / "pooled_sparse" / "refit.done", "w").write("ok\n")
    print(f"FINAL lam={a.lam:g} n_selected={len(sel)}{' FALLBACK' if fallback else ''} best_epoch={best_epoch}", flush=True)


if __name__ == "__main__":
    main()
