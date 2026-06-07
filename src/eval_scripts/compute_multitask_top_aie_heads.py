#!/usr/bin/env python
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

from src.compute_indirect_effect import batch_activation_replacement_last_token_intervention
from src.utils.eval_utils import n_shot_eval_no_intervention
from src.utils.extract_utils import get_mean_head_activations
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.prompt_utils import get_dummy_token_labels, load_dataset, word_pairs_to_prompt_data


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute globally top attention heads by averaging causal indirect effect "
            "over every query prompt for the train tasks in a task split."
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
    parser.add_argument("--tasks", nargs="+", default=None, help="Optional explicit task subset/override.")
    parser.add_argument("--root_data_dir", type=str, default="dataset_files")
    parser.add_argument("--save_path_root", type=Path, default=Path("results/multitask_aie_heads"))
    parser.add_argument(
        "--mean_activations_root",
        type=Path,
        default=Path("results/gptj_fv"),
        help="Root containing <task>/<task>_mean_head_activations.pt files to reuse.",
    )
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_split", type=float, default=0.3)
    parser.add_argument("--n_shots", type=int, default=10)
    parser.add_argument("--n_top_heads", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--query_split",
        choices=["train", "valid", "test"],
        default="train",
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
        help="Optional cap for smoke tests or partial runs. Default uses every query prompt.",
    )
    parser.add_argument("--n_mean_activations_trials", type=int, default=100)
    parser.add_argument("--batch_size_mean_activations", type=int, default=1)
    parser.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    parser.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    parser.add_argument("--shuffle_labels", dest="shuffle_labels", action="store_true")
    parser.add_argument("--no_shuffle_labels", dest="shuffle_labels", action="store_false")
    parser.set_defaults(shuffle_labels=True)
    parser.add_argument("--filter_to_correct_icl", dest="filter_to_correct_icl", action="store_true")
    parser.add_argument("--no_filter_to_correct_icl", dest="filter_to_correct_icl", action="store_false")
    parser.set_defaults(filter_to_correct_icl=True)
    parser.add_argument("--batch_size_filter_eval", type=int, default=1)
    parser.add_argument("--recompute_mean_activations", action="store_true")
    parser.add_argument("--save_per_prompt_effects", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def torch_load_trusted(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_tasks(args):
    if args.tasks is not None:
        return args.tasks, {}

    with open(args.task_split_path, "r") as f:
        split = json.load(f)
    if args.task_split_key not in split:
        raise KeyError(f"{args.task_split_path} does not contain key '{args.task_split_key}'")
    return split[args.task_split_key], split


def mean_activations_path(root, task):
    return root / task / f"{task}_mean_head_activations.pt"


def load_or_compute_mean_activations(args, task, dataset, model, model_config, tokenizer, task_output_dir, filter_set=None):
    save_path = mean_activations_path(args.save_path_root, task)
    reuse_path = mean_activations_path(args.mean_activations_root, task)

    if reuse_path.exists() and not args.recompute_mean_activations:
        print(f"Loading mean activations for {task}: {reuse_path}")
        mean_activations = torch_load_trusted(reuse_path, map_location="cpu")
        return mean_activations, str(reuse_path)

    if save_path.exists() and not args.recompute_mean_activations:
        print(f"Loading mean activations for {task}: {save_path}")
        mean_activations = torch_load_trusted(save_path, map_location="cpu")
        return mean_activations, str(save_path)

    print(f"Computing mean activations for {task}")
    set_seed(args.seed)
    mean_activations = get_mean_head_activations(
        dataset,
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        n_icl_examples=args.n_shots,
        N_TRIALS=args.n_mean_activations_trials,
        prefixes=args.prefixes,
        separators=args.separators,
        batch_size=args.batch_size_mean_activations,
        filter_set=filter_set,
    )
    task_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(mean_activations.detach().cpu(), save_path)
    return mean_activations.detach().cpu(), str(save_path)


def load_or_compute_filter_set(args, task, dataset, model, model_config, tokenizer, task_output_dir):
    if not args.filter_to_correct_icl:
        return np.arange(len(dataset[args.query_split])), None

    if args.query_split == "valid":
        reuse_path = args.mean_activations_root / task / "fs_results_validation.json"
        save_path = task_output_dir / "fs_results_validation.json"
    else:
        reuse_path = args.mean_activations_root / task / f"fs_results_{args.query_split}.json"
        save_path = task_output_dir / f"fs_results_{args.query_split}.json"

    if reuse_path.exists():
        print(f"Loading ICL-correct query filter for {task}: {reuse_path}")
        with open(reuse_path, "r") as f:
            fs_results = json.load(f)
        source_path = reuse_path
    elif save_path.exists() and not args.overwrite:
        print(f"Loading ICL-correct query filter for {task}: {save_path}")
        with open(save_path, "r") as f:
            fs_results = json.load(f)
        source_path = save_path
    else:
        print(f"Computing ICL-correct query filter for {task} on {args.query_split}")
        set_seed(args.seed + 42)
        fs_results = n_shot_eval_no_intervention(
            dataset=dataset,
            n_shots=args.n_shots,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            compute_ppl=True,
            test_split=args.query_split,
            prefixes=args.prefixes,
            separators=args.separators,
            batch_size=args.batch_size_filter_eval,
        )
        with open(save_path, "w") as f:
            json.dump(fs_results, f, indent=2)
        source_path = save_path

    if "clean_rank_list" not in fs_results:
        raise KeyError(f"{source_path} does not contain clean_rank_list")
    filter_set = np.where(np.array(fs_results["clean_rank_list"]) == 0)[0]
    if len(filter_set) == 0:
        raise RuntimeError(f"No ICL-correct {args.query_split} examples found for {task}")
    return filter_set, str(source_path)


def sample_demo_indices(query_idx, query_split, demo_split, n_query, n_demo, n_shots, rng):
    candidates = np.arange(n_demo)
    if query_split == demo_split and n_query == n_demo and n_demo > 1:
        candidates = candidates[candidates != query_idx]

    replace = len(candidates) < n_shots
    if len(candidates) == 0:
        raise ValueError("No demonstration candidates are available.")
    return rng.choice(candidates, size=n_shots, replace=replace)


def build_prompt_data(dataset, args, model_config, task_index, query_idx):
    query_data = dataset[args.query_split][query_idx]
    n_query = len(dataset[args.query_split])
    n_demo = len(dataset[args.demo_split])
    rng = np.random.default_rng(args.seed + 100_000 * task_index + query_idx)
    demo_indices = sample_demo_indices(
        query_idx=query_idx,
        query_split=args.query_split,
        demo_split=args.demo_split,
        n_query=n_query,
        n_demo=n_demo,
        n_shots=args.n_shots,
        rng=rng,
    )
    demo_pairs = dataset[args.demo_split][demo_indices]

    # word_pairs_to_prompt_data handles label shuffling through NumPy's global RNG.
    np.random.seed(args.seed + 100_000 * task_index + query_idx)
    prepend_bos = False if model_config["prepend_bos"] else True
    return word_pairs_to_prompt_data(
        demo_pairs,
        query_target_pair=query_data,
        prepend_bos_token=prepend_bos,
        shuffle_labels=args.shuffle_labels,
        prefixes=args.prefixes,
        separators=args.separators,
    )


def top_heads_from_scores(scores, n_top_heads):
    k = min(int(n_top_heads), scores.numel())
    topk_vals, topk_inds = torch.topk(scores.reshape(-1), k=k, largest=True)
    layers, heads = np.unravel_index(topk_inds.cpu().numpy(), tuple(scores.shape))
    return [
        (int(layer), int(head), round(float(score), 6))
        for layer, head, score in zip(layers, heads, topk_vals.cpu())
    ]


def compute_task_effects(args, task, task_index, dataset, query_indices, mean_activations, model, model_config, tokenizer):
    query_indices = list(query_indices)
    if args.max_prompts_per_task is not None:
        query_indices = query_indices[: args.max_prompts_per_task]
    query_count = len(query_indices)
    if query_count == 0:
        raise RuntimeError(f"No query prompts selected for {task}")

    dummy_labels = get_dummy_token_labels(
        args.n_shots,
        tokenizer=tokenizer,
        prefixes=args.prefixes,
        separators=args.separators,
        model_config=model_config,
    )

    task_sum = torch.zeros(model_config["n_layers"], model_config["n_heads"], dtype=torch.float64)
    per_prompt_effects = [] if args.save_per_prompt_effects else None
    batch_size = max(1, int(args.batch_size))

    progress = tqdm(range(0, query_count, batch_size), desc=task, total=(query_count + batch_size - 1) // batch_size)
    for batch_start in progress:
        batch_end = min(query_count, batch_start + batch_size)
        batch_query_indices = query_indices[batch_start:batch_end]
        prompt_batch = [
            build_prompt_data(dataset, args, model_config, task_index=task_index, query_idx=int(query_idx))
            for query_idx in batch_query_indices
        ]
        batch_effects = batch_activation_replacement_last_token_intervention(
            prompt_data_batch=prompt_batch,
            avg_activations=mean_activations,
            dummy_labels=dummy_labels,
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
    args.save_path_root.mkdir(parents=True, exist_ok=True)
    output_path = args.save_path_root / "multitask_top_aie_heads.pt"
    metadata_path = args.save_path_root / "multitask_top_aie_heads_metadata.json"

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists. Pass --overwrite to recompute.")

    tasks, split_metadata = load_tasks(args)
    print(f"Aggregating {len(tasks)} tasks from {args.task_split_key}: {tasks}")

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

    for task_index, task in enumerate(tasks):
        print(f"\n=== {task} ===")
        task_output_dir = args.save_path_root / task
        task_output_dir.mkdir(parents=True, exist_ok=True)

        dataset = load_dataset(task, root_data_dir=args.root_data_dir, test_size=args.test_split, seed=args.seed)
        query_indices, filter_path = load_or_compute_filter_set(
            args, task, dataset, model, model_config, tokenizer, task_output_dir
        )
        mean_filter_set = query_indices if args.query_split == "valid" else None
        mean_activations, mean_path = load_or_compute_mean_activations(
            args, task, dataset, model, model_config, tokenizer, task_output_dir, filter_set=mean_filter_set
        )

        task_sum, task_mean, n_prompts, per_prompt_effects = compute_task_effects(
            args, task, task_index, dataset, query_indices, mean_activations, model, model_config, tokenizer
        )
        global_sum += task_sum
        total_prompts += n_prompts

        task_mean_path = task_output_dir / f"{task}_mean_indirect_effect_over_{args.query_split}.pt"
        torch.save(task_mean.float().cpu(), task_mean_path)
        per_prompt_path = None
        if per_prompt_effects is not None:
            per_prompt_path = task_output_dir / f"{task}_per_prompt_indirect_effect_{args.query_split}.pt"
            torch.save(per_prompt_effects, per_prompt_path)

        per_task_summary.append(
            {
                "task": task,
                "n_prompts": int(n_prompts),
                "n_query_candidates": int(len(query_indices)),
                "filter_to_correct_icl": args.filter_to_correct_icl,
                "filter_path": filter_path,
                "mean_activations_path": mean_path,
                "mean_indirect_effect_path": str(task_mean_path),
                "per_prompt_indirect_effect_path": None if per_prompt_path is None else str(per_prompt_path),
                "top_heads": top_heads_from_scores(task_mean.float(), args.n_top_heads),
            }
        )

    global_mean = global_sum / max(1, total_prompts)
    top_heads = top_heads_from_scores(global_mean.float(), args.n_top_heads)

    result = {
        "mean_indirect_effect": global_mean.float().cpu(),
        "top_heads": top_heads,
        "n_top_heads": int(args.n_top_heads),
        "tasks": tasks,
        "total_prompts": int(total_prompts),
        "per_task": per_task_summary,
    }
    torch.save(result, output_path)

    metadata = {
        "model_name": args.model_name,
        "model_config": model_config,
        "task_split_path": str(args.task_split_path),
        "task_split_key": args.task_split_key,
        "split_metadata": split_metadata,
        "tasks": tasks,
        "query_split": args.query_split,
        "demo_split": args.demo_split,
        "n_shots": args.n_shots,
        "n_top_heads": args.n_top_heads,
        "batch_size": args.batch_size,
        "max_prompts_per_task": args.max_prompts_per_task,
        "shuffle_labels": args.shuffle_labels,
        "filter_to_correct_icl": args.filter_to_correct_icl,
        "batch_size_filter_eval": args.batch_size_filter_eval,
        "seed": args.seed,
        "test_split": args.test_split,
        "prefixes": args.prefixes,
        "separators": args.separators,
        "mean_activations_root": str(args.mean_activations_root),
        "save_path_root": str(args.save_path_root),
        "total_prompts": int(total_prompts),
        "top_heads": top_heads,
        "per_task": per_task_summary,
        "output_path": str(output_path),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nTop heads:")
    for layer, head, score in top_heads:
        print(f"L{layer} H{head}: {score}")
    print(output_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
