#!/usr/bin/env python
"""Sweep top-N (10/20/30/40) train_varicl FVs and measure held-out per-layer steering.

For each of the 9 test tasks and each N, load the prebuilt train_varicl FV (N=10 ->
`train_varicl`, else `train_varicl_top{N}`), evaluate zero-shot + 10-shot-shuffled
intervention top-1 across all 28 layers using the SAME filter set / seed as the existing
held-out eval (so curves are comparable across N and to the earlier baselines). Writes
per-task by-layer JSON + an aggregate best-layer summary; the fixed-ICL-multitask and
task-specific baselines (from each task's comparison_summary.json) are recorded for context.
Plotting is done separately by plot_nheads_sweep_with_baselines.py.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SRC_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_heldout_multitask_head_fvs import (
    evaluate_fv,
    get_filter_set,
    load_function_vector,
    summarize_results,
    write_json,
)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.prompt_utils import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Sweep top-N train_varicl FVs for held-out per-layer steering.")
    p.add_argument("--task_split_path", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--task_split_key", type=str, default="test_tasks")
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--n_values", nargs="+", type=int, default=[10, 20, 30, 40])
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--eval_root", type=Path, default=Path("results/heldout_multitask_head_eval"),
                   help="Holds each task's comparison_summary.json (for baseline context + filter source check).")
    p.add_argument("--filter_fv_root", type=Path, default=Path("results/gptj_fv"))
    p.add_argument("--fv_root_base", type=Path, default=Path("results/function_vectors/gpt-j"),
                   help="N=10 -> <base>/train_varicl ; N>10 -> <base>/train_varicl_top{N}.")
    p.add_argument("--output_root", type=Path, default=Path("results/heldout_varicl_nheads_sweep"))
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
    p.set_defaults(filter_to_correct_icl=True, generate_str=False)
    p.add_argument("--no_global_summary", action="store_true",
                   help="Skip the shared summary write (use for parallel shards; merge separately).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def fv_dir_for_n(base, n):
    return base / ("train_varicl" if n == 10 else f"train_varicl_top{n}")


def load_test_tasks(args):
    if args.tasks is not None:
        return args.tasks
    return json.loads(args.task_split_path.read_text())[args.task_split_key]


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks = load_test_tasks(args)

    # Pre-check all FV files exist before loading the model.
    missing = []
    for task in tasks:
        for n in args.n_values:
            fp = fv_dir_for_n(args.fv_root_base, n) / task / f"{task}_function_vector.pt"
            if not fp.exists():
                missing.append(str(fp))
    if missing:
        raise FileNotFoundError("Missing FV files (build them first):\n  " + "\n  ".join(missing))

    torch.set_grad_enabled(False)
    print("Loading model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision
    )
    model.eval()

    aggregate = {"tasks": tasks, "n_values": args.n_values, "per_task": {}}

    for task in tasks:
        print(f"\n=== {task} ===")
        set_seed(args.seed)
        dataset = load_dataset(task, root_data_dir=args.root_data_dir, test_size=args.test_split, seed=args.seed)

        filter_args = SimpleNamespace(
            filter_to_correct_icl=args.filter_to_correct_icl, generate_str=args.generate_str,
            fv_root=args.filter_fv_root, seed=args.seed, n_shots=args.n_shots, metric=args.metric,
            prefixes=args.prefixes, separators=args.separators, batch_size_baseline=args.batch_size_baseline,
        )
        out_dir = args.output_root / task
        out_dir.mkdir(parents=True, exist_ok=True)
        filter_set, filter_source = get_filter_set(filter_args, task, dataset, model, model_config, tokenizer, out_dir)
        n_filt = None if filter_set is None else int(len(filter_set))
        print(f"  filter set: {n_filt} examples")

        eval_args = SimpleNamespace(
            edit_layer=args.edit_layer, seed=args.seed, n_shots=args.n_shots,
            prefixes=args.prefixes, separators=args.separators, generate_str=args.generate_str,
            metric=args.metric, filter_set=filter_set,
        )

        by_n = {}
        best_by_n = {}
        for n in args.n_values:
            fv_path = fv_dir_for_n(args.fv_root_base, n) / task / f"{task}_function_vector.pt"
            fv, _ = load_function_vector(fv_path)
            print(f"  top-{n}: evaluating ({fv_path})")
            zs, fs = evaluate_fv(eval_args, dataset, fv, model, model_config, tokenizer)
            summ = summarize_results(zs, fs)
            by_n[n] = summ
            best_by_n[n] = {
                "best_zs_layer": summ["best_zs_layer"],
                "best_zs_intervention_top1": summ["best_zs_intervention_top1"],
                "best_fs_shuffled_layer": summ["best_fs_shuffled_layer"],
                "best_fs_shuffled_intervention_top1": summ["best_fs_shuffled_intervention_top1"],
            }

        write_json(out_dir / "nheads_sweep_by_layer.json", {str(n): by_n[n] for n in args.n_values})
        # Plotting is handled separately by plot_nheads_sweep_with_baselines.py (single
        # canonical figure that also overlays the task-specific + train-selected baselines).

        # Pull baselines for context (if the prior eval ran).
        baselines = {}
        comp = args.eval_root / task / "comparison_summary.json"
        if comp.exists():
            cj = json.loads(comp.read_text())
            baselines = {
                "multitask_fixed_icl_best_zs": cj["multitask_heads"]["best_zs_intervention_top1"],
                "multitask_fixed_icl_best_fs": cj["multitask_heads"]["best_fs_shuffled_intervention_top1"],
                "task_specific_best_zs": cj["task_specific_heads"]["best_zs_intervention_top1"],
                "task_specific_best_fs": cj["task_specific_heads"]["best_fs_shuffled_intervention_top1"],
            }
        task_entry = {
            "n_filtered_test_examples": n_filt,
            "filter_source": filter_source,
            "best_by_n": best_by_n,
            "baselines": baselines,
        }
        # Per-task entry file so parallel shards never clobber a shared summary.
        write_json(out_dir / "sweep_task_entry.json", task_entry)
        aggregate["per_task"][task] = task_entry
        print(f"  done {task}")

    if not args.no_global_summary:
        agg_path = args.output_root / "nheads_sweep_summary.json"
        write_json(agg_path, aggregate)
        print(f"\n{agg_path}")


if __name__ == "__main__":
    main()
