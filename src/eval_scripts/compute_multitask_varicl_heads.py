#!/usr/bin/env python
"""Stage-1: variable-ICL, train-pooled top-AIE head selection (GPT-J, 3 shards / 1 GPU).

Structurally mirrors compute_multitask_top_aie_heads.py (same --reduce/shard flow, global
RNG seeding, and per-task <task>_cie_result.pt aggregation) but under a variable-length
prompt regime. Per task:
  1. variable-ICL correctness filter -> correct query indices (rank 0)
  2. cap to --max_successful_prompts (default 170)
  3. single-position mean head activations at the query token over those indices
     -> <task>/<task>_mean_head_activations_varicl.pt  (shape (n_layers, n_heads, head_dim))
  4. CIE over the SAME indices (shuffle_labels=True, seed_base = seed + cie_seed_offset)
     -> <task>/<task>_cie_result.pt
The reduce step pools CIE over the TRAIN tasks only (default --task_split_key train_tasks).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.eval_scripts.compute_multitask_top_aie_heads import (
    aggregate_task_results,
    assert_abstractive,
    infer_dims_from_results,
    load_tasks,
    select_shard,
    task_result_path,
    top_heads_from_scores,
    torch_load_trusted,
    write_global_artifact,
)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.prompt_utils import load_dataset
from src.utils.varicl_utils import (
    batch_varicl_last_token_intervention,
    build_varicl_prompt_data,
    get_last_token_mean_head_activations,
    varicl_correctness_filter,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute globally top attention heads by averaging causal indirect effect over "
            "variable-ICL query prompts (random 1-10 shots) for the train tasks in a task split."
        )
    )
    parser.add_argument(
        "--task_split_path",
        type=Path,
        default=Path("task_splits/abstractive_train_test_tasks_29.json"),
        help="JSON file containing train_tasks/test_tasks.",
    )
    parser.add_argument(
        "--task_split_key",
        type=str,
        default="train_tasks",
        help="Key in --task_split_path that contains the task names to aggregate.",
    )
    parser.add_argument(
        "--task_split_keys",
        nargs="+",
        default=None,
        help=(
            "Optional list of keys in --task_split_path whose task lists are concatenated "
            "(e.g. --task_split_keys train_tasks test_tasks). Overrides --task_split_key when set."
        ),
    )
    parser.add_argument(
        "--all_split_tasks",
        action="store_true",
        help=(
            "Use every task in both train_tasks and test_tasks from --task_split_path. "
            "Shorthand for --task_split_keys train_tasks test_tasks."
        ),
    )
    parser.add_argument("--tasks", nargs="+", default=None, help="Optional explicit task subset/override.")
    parser.add_argument("--root_data_dir", type=str, default="dataset_files")
    parser.add_argument("--save_path_root", type=Path, default=Path("results/multitask_aie_heads_varicl"))
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_split", type=float, default=0.3)
    parser.add_argument("--min_shots", type=int, default=1)
    parser.add_argument("--max_shots", type=int, default=10)
    parser.add_argument("--max_successful_prompts", type=int, default=170)
    parser.add_argument("--cie_seed_offset", type=int, default=500000)
    parser.add_argument("--n_top_heads", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--query_split",
        choices=["train", "valid", "test"],
        default="valid",
        help="Dataset split whose examples become query prompts.",
    )
    parser.add_argument(
        "--demo_split",
        choices=["train", "valid", "test"],
        default="train",
        help="Dataset split used for in-context demonstrations.",
    )
    parser.add_argument(
        "--max_prompts_per_task",
        type=int,
        default=None,
        help="Optional smoke/partial-run cap applied AFTER the --max_successful_prompts cap.",
    )
    parser.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    parser.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    parser.add_argument("--filter_to_correct_icl", dest="filter_to_correct_icl", action="store_true")
    parser.add_argument("--no_filter_to_correct_icl", dest="filter_to_correct_icl", action="store_false")
    parser.set_defaults(filter_to_correct_icl=True)
    parser.add_argument("--batch_size_filter_eval", type=int, default=1)
    parser.add_argument("--save_per_prompt_effects", dest="save_per_prompt_effects", action="store_true")
    parser.add_argument("--no_save_per_prompt_effects", dest="save_per_prompt_effects", action="store_false")
    parser.set_defaults(save_per_prompt_effects=False)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--abstractive_only",
        action="store_true",
        help="Assert every selected task lives in dataset_files/abstractive before running.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of parallel worker processes splitting the task list (data parallel across GPU instances).",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="This worker's shard id in [0, num_shards). Each worker processes tasks[shard_index::num_shards].",
    )
    parser.add_argument(
        "--reduce",
        action="store_true",
        help=(
            "Skip CIE computation and instead aggregate the per-task <task>_cie_result.pt files already "
            "present under --save_path_root into the global top-head artifact + metadata."
        ),
    )
    return parser.parse_args()


def mean_activations_path(root, task):
    return root / task / f"{task}_mean_head_activations_varicl.pt"


def compute_varicl_filter_set(args, task, dataset, model, model_config, tokenizer, task_index, task_output_dir):
    """Variable-ICL correctness filter -> correct query indices (rank 0), capped at the limit."""
    save_path = task_output_dir / f"fs_results_varicl_{args.query_split}.json"

    if save_path.exists() and not args.overwrite:
        print(f"Loading variable-ICL query filter for {task}: {save_path}")
        with open(save_path, "r") as f:
            fs_results = json.load(f)
        source_path = save_path
    else:
        print(f"Computing variable-ICL query filter for {task} on {args.query_split}")
        set_seed(args.seed)
        clean_rank_list = varicl_correctness_filter(
            dataset=dataset,
            args=args,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            task_index=task_index,
            seed_base=args.seed,
        )
        fs_results = {"clean_rank_list": clean_rank_list}
        with open(save_path, "w") as f:
            json.dump(fs_results, f, indent=2)
        source_path = save_path

    if "clean_rank_list" not in fs_results:
        raise KeyError(f"{source_path} does not contain clean_rank_list")

    if args.filter_to_correct_icl:
        correct = np.where(np.array(fs_results["clean_rank_list"]) == 0)[0]
    else:
        correct = np.arange(len(fs_results["clean_rank_list"]))
    if len(correct) == 0:
        raise RuntimeError(f"No ICL-correct {args.query_split} examples found for {task}")

    query_indices = correct[: args.max_successful_prompts]
    return query_indices, str(source_path)


def compute_task_effects(args, task, task_index, dataset, query_indices, mean_activations, model, model_config, tokenizer):
    query_indices = list(query_indices)
    if args.max_prompts_per_task is not None:
        query_indices = query_indices[: args.max_prompts_per_task]
    query_count = len(query_indices)
    if query_count == 0:
        raise RuntimeError(f"No query prompts selected for {task}")

    task_sum = torch.zeros(model_config["n_layers"], model_config["n_heads"], dtype=torch.float64)
    per_prompt_effects = [] if args.save_per_prompt_effects else None
    batch_size = max(1, int(args.batch_size))

    progress = tqdm(range(0, query_count, batch_size), desc=task, total=(query_count + batch_size - 1) // batch_size)
    for batch_start in progress:
        batch_end = min(query_count, batch_start + batch_size)
        batch_query_indices = query_indices[batch_start:batch_end]
        prompt_batch = [
            build_varicl_prompt_data(
                dataset, args, model_config, task_index=task_index, query_idx=int(query_idx),
                shuffle_labels=True, seed_base=args.seed + args.cie_seed_offset,
            )
            for query_idx in batch_query_indices
        ]
        batch_effects = batch_varicl_last_token_intervention(
            prompt_data_batch=prompt_batch,
            avg_activations=mean_activations,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
        ).double()
        task_sum += batch_effects.sum(dim=0)
        if per_prompt_effects is not None:
            per_prompt_effects.append(batch_effects.float().cpu())

    task_mean = task_sum / max(1, query_count)
    if per_prompt_effects is not None:
        per_prompt_effects = torch.cat(per_prompt_effects, dim=0)
    return task_sum, task_mean, query_count, per_prompt_effects


def main():
    args = parse_args()
    # Compatibility attributes consumed by the shared write_global_artifact() metadata dump.
    # The variable-ICL regime has no single n_shots, no canonical mean-activations reuse root,
    # and always shuffles CIE labels; record those facts for the metadata.
    args.n_shots = f"variable[{args.min_shots},{args.max_shots}]"
    args.shuffle_labels = True
    args.mean_activations_root = args.save_path_root
    args.save_path_root.mkdir(parents=True, exist_ok=True)
    output_path = args.save_path_root / "multitask_top_aie_heads.pt"
    metadata_path = args.save_path_root / "multitask_top_aie_heads_metadata.json"

    tasks, split_metadata, task_split_keys = load_tasks(args)
    if args.abstractive_only:
        assert_abstractive(tasks, args.root_data_dir)
    if task_split_keys is None:
        source_desc = "--tasks override"
    else:
        source_desc = ", ".join(task_split_keys)
    print(f"Selected {len(tasks)} tasks from {source_desc}: {tasks}")

    # Only the full single-process run (--num_shards 1) or the explicit --reduce step
    # produces the combined cross-task artifact. Worker shards write per-task files only.
    writes_global = args.reduce or args.num_shards <= 1
    if writes_global and output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists. Pass --overwrite to recompute.")

    # ---- REDUCE: aggregate already-computed per-task results, no model needed ----
    if args.reduce:
        n_layers, n_heads = infer_dims_from_results(args, tasks)
        global_sum, total_prompts, per_task_summary = aggregate_task_results(args, tasks, n_layers, n_heads)
        print(f"Reducing {len(per_task_summary)} per-task results, {total_prompts} prompts total.")
        write_global_artifact(
            args, tasks, task_split_keys, split_metadata, global_sum, total_prompts,
            per_task_summary, n_layers, n_heads, model_config=None,
            output_path=output_path, metadata_path=metadata_path,
        )
        return

    # ---- COMPUTE: full run or one worker shard ----
    shard_tasks = select_shard(tasks, args.shard_index, args.num_shards)
    if args.num_shards > 1:
        print(f"Shard {args.shard_index}/{args.num_shards} handling {len(shard_tasks)} tasks: {shard_tasks}")

    torch.set_grad_enabled(False)
    set_seed(args.seed)
    print("Loading model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision
    )
    model.eval()

    global_sum = torch.zeros(model_config["n_layers"], model_config["n_heads"], dtype=torch.float64)
    total_prompts = 0
    per_task_summary = []

    # task_index is the position within the FULL task list so prompt RNG seeds are
    # identical regardless of how tasks are sharded across workers.
    for task in shard_tasks:
        task_index = tasks.index(task)
        print(f"\n=== {task} (global index {task_index}) ===")
        task_output_dir = args.save_path_root / task
        task_output_dir.mkdir(parents=True, exist_ok=True)
        result_path = task_result_path(args.save_path_root, task)

        if result_path.exists() and not args.overwrite:
            print(f"Reusing existing per-task result: {result_path}")
            continue

        dataset = load_dataset(task, root_data_dir=args.root_data_dir, test_size=args.test_split, seed=args.seed)
        query_indices, filter_path = compute_varicl_filter_set(
            args, task, dataset, model, model_config, tokenizer, task_index, task_output_dir
        )

        # Single-position mean head activations at the query token over the SAME indices.
        mean_path = mean_activations_path(args.save_path_root, task)
        if mean_path.exists() and not args.overwrite:
            print(f"Loading mean activations for {task}: {mean_path}")
            mean_activations = torch_load_trusted(mean_path, map_location="cpu")
        else:
            print(f"Computing variable-ICL last-token mean activations for {task}")
            set_seed(args.seed)
            mean_activations = get_last_token_mean_head_activations(
                dataset, args, model, model_config, tokenizer,
                task_index=task_index, query_indices=query_indices, seed_base=args.seed,
            )
            torch.save(mean_activations.detach().cpu(), mean_path)

        task_sum, task_mean, n_prompts, per_prompt_effects = compute_task_effects(
            args, task, task_index, dataset, query_indices, mean_activations, model, model_config, tokenizer
        )

        task_mean_path = task_output_dir / f"{task}_mean_indirect_effect_over_{args.query_split}.pt"
        torch.save(task_mean.float().cpu(), task_mean_path)
        per_prompt_path = None
        if per_prompt_effects is not None:
            per_prompt_path = task_output_dir / f"{task}_per_prompt_indirect_effect_{args.query_split}.pt"
            torch.save(per_prompt_effects, per_prompt_path)

        task_top_heads = top_heads_from_scores(task_mean.float(), args.n_top_heads)
        task_result = {
            "task": task,
            "indirect_effect_sum": task_sum.cpu(),
            "mean_indirect_effect": task_mean.float().cpu(),
            "n_prompts": int(n_prompts),
            "n_query_candidates": int(len(query_indices)),
            "query_split": args.query_split,
            "demo_split": args.demo_split,
            "n_top_heads": int(args.n_top_heads),
            "top_heads": task_top_heads,
            "filter_to_correct_icl": args.filter_to_correct_icl,
            "filter_path": filter_path,
            "mean_activations_path": str(mean_path),
            "mean_indirect_effect_path": str(task_mean_path),
            "per_prompt_indirect_effect_path": None if per_prompt_path is None else str(per_prompt_path),
            "n_layers": int(model_config["n_layers"]),
            "n_heads": int(model_config["n_heads"]),
            "min_shots": int(args.min_shots),
            "max_shots": int(args.max_shots),
            "max_successful_prompts": int(args.max_successful_prompts),
            "cie_seed_offset": int(args.cie_seed_offset),
        }
        torch.save(task_result, result_path)
        print(f"Wrote per-task CIE result: {result_path}")

        global_sum += task_sum
        total_prompts += n_prompts
        per_task_summary.append(
            {
                "task": task,
                "n_prompts": int(n_prompts),
                "n_query_candidates": int(len(query_indices)),
                "filter_to_correct_icl": args.filter_to_correct_icl,
                "filter_path": filter_path,
                "mean_activations_path": str(mean_path),
                "mean_indirect_effect_path": str(task_mean_path),
                "per_prompt_indirect_effect_path": None if per_prompt_path is None else str(per_prompt_path),
                "cie_result_path": str(result_path),
                "top_heads": task_top_heads,
            }
        )

    if not writes_global:
        print(
            f"\nShard {args.shard_index}/{args.num_shards} finished. "
            f"Run with --reduce (same --save_path_root and task selection) to build the combined artifact."
        )
        return

    # Single-process full run: aggregate from the per-task files we just wrote so the
    # result is identical to what --reduce would produce (handles --overwrite skips too).
    n_layers, n_heads = model_config["n_layers"], model_config["n_heads"]
    global_sum, total_prompts, per_task_summary = aggregate_task_results(args, tasks, n_layers, n_heads)
    write_global_artifact(
        args, tasks, task_split_keys, split_metadata, global_sum, total_prompts,
        per_task_summary, n_layers, n_heads, model_config=model_config,
        output_path=output_path, metadata_path=metadata_path,
    )


if __name__ == "__main__":
    main()
