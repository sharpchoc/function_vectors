#!/usr/bin/env python
"""SANDBOX: steering eval for extended_steerable_90 tasks with the pooled-selected head set.

Per task: FV = sum over selected heads of that task's W_O-projected 10-shot head means
(unweighted, c > 0.8 set from pooled_sparse/selection.json); inject at the cue token,
alpha=1, layers 0..27; full-label teacher-forced accuracy on the paired test prompts of
{test_zeroshot, test_sametask_shuffled10, test_mixedtask10} + unsteered baselines.
Fan out with --tasks (works for train AND heldout tasks - phase 2 reuses this unchanged).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.sandbox.isolation_upper_bound.run_task import (
    TEST_SETTINGS,
    build_contributions_single,
    eval_points_fixed_v,
    load_records,
    record_to_point,
)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.paths import ARTIFACTS_ROOT

DEFAULT_OUT = ARTIFACTS_ROOT / "sandbox" / "ext_steerability"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--selection_path", type=Path, default=DEFAULT_OUT / "pooled_sparse" / "selection.json")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_name", type=str, default="eval_headset.json",
                   help="Output filename per task (use a distinct name for trials).")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    sel = json.load(open(args.selection_path))
    flat = torch.tensor(sel["selected_flat"])
    print(f"head set: n={sel['n_selected']} lam={sel['chosen_lambda']}", flush=True)

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()
    torch.set_grad_enabled(False)
    n_layers = model_config["n_layers"]

    # per-task FV from as-loaded weights, then bf16 for eval forwards
    fvs = {}
    for t in args.tasks:
        means = torch.load(args.out_root / t / "means.pt", map_location="cpu", weights_only=False)
        C_t = build_contributions_single(means["head_means"], model, model_config)
        fvs[t] = C_t[flat.to(C_t.device)].sum(dim=0).float()
    model = model.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)

    for t in args.tasks:
        out = args.out_root / t / args.out_name
        if out.exists():
            print(f"[{t}] eval exists, skip", flush=True)
            continue
        results = {}
        for setting in TEST_SETTINGS:
            recs = load_records(args, t, setting)
            points = [record_to_point(r, tokenizer, model_config) for r in recs]
            baseline = eval_points_fixed_v(model, model_config, tokenizer, points, None, 9)
            accs = [eval_points_fixed_v(model, model_config, tokenizer, points, fvs[t], L)
                    for L in range(n_layers)]
            results[setting] = {"baseline": baseline, "acc_by_layer": accs,
                                "n_prompts": len(points)}
            print(f"[{t}] {setting}: baseline={baseline:.3f} best={max(accs):.3f} "
                  f"@L{accs.index(max(accs))}", flush=True)
        with open(out, "w") as f:
            json.dump({"task": t, "head_set": sel["selected_heads"],
                       "n_heads": sel["n_selected"], "alpha": 1.0,
                       "readout": "full-label teacher-forced", "settings": results}, f, indent=1)
    print("EVAL DONE")


if __name__ == "__main__":
    main()
