#!/usr/bin/env python
"""SANDBOX diagnostic: task-specific sparse selection for the zs<0.4 failing train tasks.

For each task: train c in [0,1]^448 on ITS OWN 100 zero-shot points (properly scaled
hyperparams: lr 0.03, 60 epochs, patience 8 - the corrected task-specific recipe), for
lambda in {0.005, 0.05}; product = c>0.8 heads (top-10 fallback); evaluate BOTH products
zero-shot across layers 0..27 on the 50 test queries and keep the better. Answers: can ANY
head-sum vector steer this task zero-shot, and if so how much does its head set overlap
the pooled 39? Writes artifacts/sandbox/ext_steerability/<task>/diag_taskspecific.json.
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

OUT_ROOT = ARTIFACTS_ROOT / "sandbox" / "ext_steerability"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--prompts_root", type=Path, default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inject_layer", type=int, default=9)
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.005, 0.05])
    p.add_argument("--c_high", type=float, default=0.8)
    return p.parse_args()


def tc(args):
    return types.SimpleNamespace(init_c=0.5, lr=0.03, max_epochs=60, micro_batch_size=32,
                                 batch_size=128, inject_layer=args.inject_layer, patience=8,
                                 threshold=0.2, earlystop_frac=0.1)


def main():
    args = parse_args()
    set_seed(args.seed)
    pooled = json.load(open(args.out_root / "pooled_sparse" / "selection.json"))
    pooled_set = set(pooled["selected_flat"])

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()
    torch.set_grad_enabled(False)
    C_by = {}
    for t in args.tasks:
        means = torch.load(args.out_root / t / "means.pt", map_location="cpu", weights_only=False)
        C_by[t] = build_contributions_single(means["head_means"], model, model_config)
    model = model.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)

    for t in args.tasks:
        out = args.out_root / t / "diag_taskspecific.json"
        if out.exists():
            print(f"[{t}] exists, skip", flush=True)
            continue
        C = C_by[t]
        C3 = C.unsqueeze(0)
        tri = {t: 0}
        train_pts = [record_to_point(r, tokenizer, model_config)
                     for r in load_records(args, t, "train_zeroshot")[:100]]
        test_pts = [record_to_point(r, tokenizer, model_config)
                    for r in load_records(args, t, "test_zeroshot")]
        best = None
        for lam in args.lambdas:
            run_seed = args.seed + int(round(lam * 1e4))
            tr, es = split_earlystop(train_pts, 0.1, run_seed)
            torch.set_grad_enabled(True)
            c, hist, be = train_c(model, model_config, tokenizer, tr, es, C3, tri, lam,
                                  tc(args), run_seed, desc=f"{t} lam={lam:g}")
            torch.set_grad_enabled(False)
            sel = torch.nonzero(c > args.c_high).flatten().tolist()
            fb = False
            if not sel:
                sel = torch.argsort(c, descending=True)[:10].tolist()
                fb = True
            v = C[torch.tensor(sel, device=C.device)].sum(dim=0)
            accs = [eval_points_fixed_v(model, model_config, tokenizer, test_pts, v, L)
                    for L in range(model_config["n_layers"])]
            cand = {"lambda": lam, "n_selected": len(sel), "fallback": fb,
                    "best_epoch": be, "zs_best": max(accs),
                    "zs_bestL": int(np.argmax(accs)), "zs_L9": accs[9],
                    "selected_flat": sel,
                    "overlap_with_pooled39": len(set(sel) & pooled_set)}
            print(f"[{t}] lam={lam:g}: n_sel={len(sel)} zs_best={max(accs):.2f}"
                  f"@L{int(np.argmax(accs))} overlap39={cand['overlap_with_pooled39']}", flush=True)
            if best is None or cand["zs_best"] > best["zs_best"]:
                best = cand
        with open(out, "w") as f:
            json.dump({"task": t, "best": best, "note": "task-specific zs sparse diagnostic"},
                      f, indent=1)
    print("DIAG DONE", flush=True)


if __name__ == "__main__":
    main()
