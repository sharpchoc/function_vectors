#!/usr/bin/env python
"""SANDBOX diagnostic: per-task head-hungriness on the ext train tasks.

Per task: task-specific sparse selection on its own 100 zero-shot points (corrected recipe,
lambda in {0.005, 0.05}, keep the better by zs best-layer acc of the c>0.8 product), then a
top-k-by-c sweep AT THE TASK'S BEST LAYER (unweighted head sums, k in K_GRID) to find
k90 = minimal k reaching >= 0.9 x the task's own best accuracy. Writes
artifacts/sandbox/ext_steerability/<task>/diag_headhunger.json (includes the full c vector).
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
K_GRID = [1, 2, 5, 10, 20, 40, 80, 160, 320]


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
        out = args.out_root / t / "diag_headhunger.json"
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
            if not sel:
                sel = torch.argsort(c, descending=True)[:10].tolist()
            v = C[torch.tensor(sel, device=C.device)].sum(dim=0)
            accs = [eval_points_fixed_v(model, model_config, tokenizer, test_pts, v, L)
                    for L in range(model_config["n_layers"])]
            cand = {"lambda": lam, "c": c.cpu(), "n_selected": len(sel),
                    "zs_best": max(accs), "zs_bestL": int(np.argmax(accs))}
            if best is None or cand["zs_best"] > best["zs_best"]:
                best = cand
        # top-k-by-c sweep at the best layer
        order = torch.argsort(best["c"], descending=True)
        L = best["zs_bestL"]
        k_accs = {}
        for k in K_GRID:
            v = C[order[:k].to(C.device)].sum(dim=0)
            k_accs[k] = eval_points_fixed_v(model, model_config, tokenizer, test_pts, v, L)
        ref = best["zs_best"]
        k90 = next((k for k in K_GRID if ref > 0 and k_accs[k] >= 0.9 * ref), None)
        with open(out, "w") as f:
            json.dump({"task": t, "lambda": best["lambda"], "n_selected": best["n_selected"],
                       "zs_best": ref, "zs_bestL": L,
                       "k_accs": {str(k): v for k, v in k_accs.items()}, "k90": k90}, f, indent=1)
        torch.save({"c": best["c"]}, args.out_root / t / "diag_headhunger_c.pt")
        print(f"[{t}] lam={best['lambda']:g} zs_best={ref:.2f}@L{L} n_sel={best['n_selected']} "
              f"k90={k90} k_accs={[f'{k}:{v:.2f}' for k,v in k_accs.items()]}", flush=True)
    print("DIAG DONE", flush=True)


if __name__ == "__main__":
    main()
