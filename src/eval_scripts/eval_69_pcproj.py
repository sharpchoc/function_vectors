#!/usr/bin/env python
"""Steering eval with PC-PROJECTED task FVs on the 69-task-run tasks (GPU).

Per task: v = sum over SELECTED PCs of (v_task . PC_i) PC_i, where v_task is the task-mean
FV over the 37 pooled-selected heads (built from means.pt as in eval_ext.py) and the PC
selection comes from train_sparse_pcs_69.py final mode (c > 0.8, unweighted). Inject at the
cue token, alpha=1, layers 0-27; full-label teacher-forced accuracy on the paired test
prompts of {test_zeroshot, test_sametask_shuffled10, test_mixedtask10} + unsteered
baselines. Writes <out_root>/evals/<task>.json. Fan out with --tasks.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.sandbox.isolation_upper_bound.run_task import (  # noqa: E402
    TEST_SETTINGS, build_contributions_single, eval_points_fixed_v, load_records,
    record_to_point)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed  # noqa: E402
from src.utils.paths import ARTIFACTS_ROOT  # noqa: E402

RUN_ROOT = ARTIFACTS_ROOT / "69_task_run"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--means_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox" / "ext_steerability")
    p.add_argument("--out_root", type=Path, default=RUN_ROOT / "pc_sparse")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Scale on the projected steering vector; evals go to evals_alpha<a>/ when != 1.")
    p.add_argument("--settings", nargs="+", default=None,
                   help="Subset of test settings (default: all three).")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    sel = json.load(open(args.out_root / "selection.json"))
    basis = torch.load(Path(sel["pc_basis_path"]), map_location="cpu", weights_only=False)
    head_flat = torch.tensor(json.load(open(sel["head_selection_path"]))["selected_flat"])
    pc_idx = torch.tensor(sel["selected_pcs"])
    print(f"PC selection: n={sel['n_selected']} lam={sel['chosen_lambda']}", flush=True)

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()
    torch.set_grad_enabled(False)
    n_layers = model_config["n_layers"]
    pcs_sel = basis["pcs"].float()[pc_idx].to(model.device)  # (k, 4096)

    fvs = {}
    for t in args.tasks:
        means = torch.load(args.means_root / t / "means.pt", map_location="cpu", weights_only=False)
        C_t = build_contributions_single(means["head_means"], model, model_config)
        v_full = C_t[head_flat.to(C_t.device)].sum(dim=0).float()
        coef = pcs_sel @ v_full
        v_proj = coef @ pcs_sel  # (4096,)
        fvs[t] = args.alpha * v_proj
        print(f"[{t}] |v_full|={v_full.norm():.1f} |v_proj|={v_proj.norm():.1f} "
              f"cos={torch.nn.functional.cosine_similarity(v_full, v_proj, dim=0):.3f}", flush=True)
    model = model.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)

    evals_dir = "evals" if args.alpha == 1.0 else f"evals_alpha{args.alpha:g}"
    settings = args.settings or TEST_SETTINGS
    (args.out_root / evals_dir).mkdir(exist_ok=True)
    for t in args.tasks:
        out = args.out_root / evals_dir / f"{t}.json"
        if out.exists():
            print(f"[{t}] eval exists, skip", flush=True)
            continue
        results = {}
        for setting in settings:
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
            json.dump({"task": t, "unit": "pc_projection", "n_pcs": sel["n_selected"],
                       "selected_pcs": sel["selected_pcs"], "alpha": args.alpha,
                       "readout": "full-label teacher-forced", "settings": results}, f, indent=1)
    print("PC-PROJ EVAL DONE")


if __name__ == "__main__":
    main()
