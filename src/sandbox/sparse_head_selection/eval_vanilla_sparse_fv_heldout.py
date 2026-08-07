#!/usr/bin/env python
"""SANDBOX: evaluate the vanilla sparse-optimisation FV (23 heads, c>0.8) on the 9 held-out
test tasks, per-layer, so it can be overlaid on the existing
`heldout_varicl_nheads_sweep` comparison plots.

"Vanilla" = UNWEIGHTED sum of the selected heads' varicl mean outputs (same construction as
the canonical train_varicl top-N FVs; the learned coefficients are used only to pick the
heads, not to weight them). Mirrors sweep_n_heads_varicl_steering.py exactly: same filter set
(clean_rank_list==0 from artifacts/gptj_fv/<task>/fs_results_layer_sweep.json), same seed,
same zero-shot + 10-shot-shuffled per-layer protocol, so curves are directly overlayable.

Writes per task (resumable; skips tasks whose output exists unless --overwrite):
  <sweep_root>/<task>/vanilla_sparse_opt23_by_layer.json   (summarize_results schema)
plus an aggregate <sweep_root>/vanilla_sparse_opt23_summary.json.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_DIR = REPO_ROOT / "src" / "eval_scripts"
for p in (REPO_ROOT, SRC_ROOT, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluate_heldout_multitask_head_fvs import (
    evaluate_fv,
    get_filter_set,
    load_function_vector,
    summarize_results,
    write_json,
)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.prompt_utils import load_dataset
from src.utils.paths import ARTIFACTS_ROOT, STEERING_COMPARISON_DIR

SERIES_NAME = "vanilla_sparse_opt23"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task_split_path", type=Path, default=REPO_ROOT / "task_splits/abstractive_train_test_tasks_29.json")
    p.add_argument("--task_split_key", type=str, default="test_tasks")
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--root_data_dir", type=str, default=str(REPO_ROOT / "dataset_files"))
    p.add_argument("--fv_root", type=Path,
                   default=ARTIFACTS_ROOT / "function_vectors" / "gpt-j" / "sandbox" / "vanilla_sparse_opt23")
    p.add_argument("--filter_fv_root", type=Path, default=ARTIFACTS_ROOT / "gptj_fv")
    p.add_argument("--output_root", type=Path, default=STEERING_COMPARISON_DIR / "heldout_varicl_nheads_sweep")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_split", type=float, default=0.3)
    p.add_argument("--n_shots", type=int, default=10)
    p.add_argument("--edit_layer", type=int, default=-1)
    p.add_argument("--batch_size_baseline", type=int, default=1)
    p.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    p.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    p.add_argument("--generate_str", action="store_true")
    p.add_argument("--metric", type=str, default="f1_score")
    p.set_defaults(filter_to_correct_icl=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tasks = args.tasks or json.loads(args.task_split_path.read_text())[args.task_split_key]

    missing = [str(args.fv_root / t / f"{t}_function_vector.pt") for t in tasks
               if not (args.fv_root / t / f"{t}_function_vector.pt").exists()]
    if missing:
        raise FileNotFoundError("Missing FV files (build them first):\n  " + "\n  ".join(missing))

    torch.set_grad_enabled(False)
    print("Loading model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision)
    model.eval()

    aggregate = {"series": SERIES_NAME, "fv_root": str(args.fv_root), "tasks": tasks,
                 "sandbox": True, "per_task": {}}
    for task in tasks:
        out_dir = args.output_root / task
        out_dir.mkdir(parents=True, exist_ok=True)
        by_layer_path = out_dir / f"{SERIES_NAME}_by_layer.json"
        if by_layer_path.exists() and not args.overwrite:
            print(f"skip {task}: {by_layer_path} exists")
            aggregate["per_task"][task] = json.loads(by_layer_path.read_text())["best"]
            continue

        print(f"\n=== {task} ===", flush=True)
        set_seed(args.seed)
        dataset = load_dataset(task, root_data_dir=args.root_data_dir, test_size=args.test_split, seed=args.seed)

        filter_args = SimpleNamespace(
            filter_to_correct_icl=args.filter_to_correct_icl, generate_str=args.generate_str,
            fv_root=args.filter_fv_root, seed=args.seed, n_shots=args.n_shots, metric=args.metric,
            prefixes=args.prefixes, separators=args.separators, batch_size_baseline=args.batch_size_baseline,
        )
        filter_set, filter_source = get_filter_set(filter_args, task, dataset, model, model_config, tokenizer, out_dir)
        print(f"  filter set: {None if filter_set is None else len(filter_set)} examples ({filter_source})")

        eval_args = SimpleNamespace(
            edit_layer=args.edit_layer, seed=args.seed, n_shots=args.n_shots,
            prefixes=args.prefixes, separators=args.separators, generate_str=args.generate_str,
            metric=args.metric, filter_set=filter_set,
        )
        fv, _ = load_function_vector(args.fv_root / task / f"{task}_function_vector.pt")
        zs, fs = evaluate_fv(eval_args, dataset, fv, model, model_config, tokenizer)
        summ = summarize_results(zs, fs)
        best = {k: summ[k] for k in ("best_zs_layer", "best_zs_intervention_top1",
                                     "best_fs_shuffled_layer", "best_fs_shuffled_intervention_top1")}
        write_json(by_layer_path, {**summ, "best": best,
                                   "n_filtered_test_examples": None if filter_set is None else int(len(filter_set)),
                                   "filter_source": filter_source, "n_heads": 23, "sandbox": True})
        aggregate["per_task"][task] = best
        print(f"  {task}: best zs {best['best_zs_intervention_top1']:.3f} @L{best['best_zs_layer']}, "
              f"best fs-shuffled {best['best_fs_shuffled_intervention_top1']:.3f} @L{best['best_fs_shuffled_layer']}",
              flush=True)

    write_json(args.output_root / f"{SERIES_NAME}_summary.json", aggregate)
    print(f"\nwrote {args.output_root / (SERIES_NAME + '_summary.json')}")


if __name__ == "__main__":
    main()
