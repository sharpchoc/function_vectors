#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SRC_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compute_task_fv_from_multitask_heads import compute_function_vector_from_heads, load_top_heads
from src.utils.eval_utils import n_shot_eval, n_shot_eval_no_intervention
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.prompt_utils import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "For held-out tasks, compare steering with FVs built from a shared multitask "
            "top-head set against saved task-specific FVs."
        )
    )
    parser.add_argument(
        "--task_split_path",
        type=Path,
        default=Path("task_splits/abstractive_train_test_tasks_29.json"),
    )
    parser.add_argument("--task_split_key", type=str, default="test_tasks")
    parser.add_argument("--tasks", nargs="+", default=None, help="Optional explicit task override.")
    parser.add_argument("--root_data_dir", type=str, default="dataset_files")
    parser.add_argument("--fv_root", type=Path, default=Path("results/gptj_fv"))
    parser.add_argument(
        "--heads_path",
        type=Path,
        default=Path("results/multitask_aie_heads/multitask_top_aie_heads.pt"),
    )
    parser.add_argument("--output_root", type=Path, default=Path("results/heldout_multitask_head_eval"))
    parser.add_argument("--n_top_heads", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_split", type=float, default=0.3)
    parser.add_argument("--n_shots", type=int, default=10)
    parser.add_argument(
        "--edit_layer",
        type=int,
        default=-1,
        help="Layer for steering. Use -1 to sweep every layer.",
    )
    parser.add_argument("--batch_size_baseline", type=int, default=1)
    parser.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    parser.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    parser.add_argument("--generate_str", action="store_true")
    parser.add_argument("--metric", type=str, default="f1_score")
    parser.add_argument("--filter_to_correct_icl", dest="filter_to_correct_icl", action="store_true")
    parser.add_argument("--no_filter_to_correct_icl", dest="filter_to_correct_icl", action="store_false")
    parser.set_defaults(filter_to_correct_icl=True)
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


def load_function_vector(path):
    data = torch_load_trusted(path, map_location="cpu")
    fv = data["function_vector"] if isinstance(data, dict) else data
    top_heads = data.get("top_heads") if isinstance(data, dict) else None
    return fv.detach().float().cpu().reshape(-1), top_heads


def get_filter_set(args, task, dataset, model, model_config, tokenizer, output_dir):
    if not args.filter_to_correct_icl:
        return None, None

    key = "score" if args.generate_str else "clean_rank_list"
    target_val = 1 if args.generate_str else 0
    existing_path = args.fv_root / task / "fs_results_layer_sweep.json"
    output_path = output_dir / "fs_results_filter_source.json"

    if existing_path.exists():
        with open(existing_path, "r") as f:
            fs_results = json.load(f)
        source_path = existing_path
    else:
        set_seed(args.seed)
        fs_results = n_shot_eval_no_intervention(
            dataset=dataset,
            n_shots=args.n_shots,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            compute_ppl=not args.generate_str,
            generate_str=args.generate_str,
            metric=args.metric,
            prefixes=args.prefixes,
            separators=args.separators,
            batch_size=args.batch_size_baseline,
        )
        with open(output_path, "w") as f:
            json.dump(fs_results, f, indent=2)
        source_path = output_path

    if key not in fs_results:
        raise KeyError(f"{source_path} does not contain key '{key}'")
    filter_set = np.where(np.array(fs_results[key]) == target_val)[0]
    return filter_set, str(source_path)


def evaluate_fv(args, dataset, fv, model, model_config, tokenizer):
    if args.edit_layer == -1:
        layers = range(model_config["n_layers"])
    else:
        layers = [args.edit_layer]

    zs_results = {}
    fs_shuffled_results = {}
    for layer in layers:
        set_seed(args.seed)
        zs_results[int(layer)] = n_shot_eval(
            dataset=dataset,
            fv_vector=fv,
            edit_layer=int(layer),
            n_shots=0,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            filter_set=args.filter_set,
            prefixes=args.prefixes,
            separators=args.separators,
            generate_str=args.generate_str,
            metric=args.metric,
        )
        set_seed(args.seed)
        fs_shuffled_results[int(layer)] = n_shot_eval(
            dataset=dataset,
            fv_vector=fv,
            edit_layer=int(layer),
            n_shots=args.n_shots,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            filter_set=args.filter_set,
            shuffle_labels=True,
            prefixes=args.prefixes,
            separators=args.separators,
            generate_str=args.generate_str,
            metric=args.metric,
        )
    return zs_results, fs_shuffled_results


def top1(result):
    if "intervention_topk" in result:
        return float(result["intervention_topk"][0][1])
    if "intervention_score" in result:
        return float(np.mean(result["intervention_score"]))
    return None


def summarize_results(zs_results, fs_shuffled_results):
    zs_by_layer = {str(layer): top1(result) for layer, result in zs_results.items()}
    fs_by_layer = {str(layer): top1(result) for layer, result in fs_shuffled_results.items()}
    best_zs_layer = max(zs_by_layer, key=zs_by_layer.get) if zs_by_layer else None
    best_fs_layer = max(fs_by_layer, key=fs_by_layer.get) if fs_by_layer else None
    return {
        "zs_intervention_top1_by_layer": zs_by_layer,
        "fs_shuffled_intervention_top1_by_layer": fs_by_layer,
        "best_zs_layer": None if best_zs_layer is None else int(best_zs_layer),
        "best_zs_intervention_top1": None if best_zs_layer is None else zs_by_layer[best_zs_layer],
        "best_fs_shuffled_layer": None if best_fs_layer is None else int(best_fs_layer),
        "best_fs_shuffled_intervention_top1": None if best_fs_layer is None else fs_by_layer[best_fs_layer],
    }


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(json_safe(data), f, indent=2)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks, split_metadata = load_tasks(args)
    top_heads, heads_data = load_top_heads(args.heads_path, args.n_top_heads)

    torch.set_grad_enabled(False)
    print("Loading model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision
    )
    model.eval()

    aggregate = {
        "tasks": tasks,
        "task_split_path": str(args.task_split_path),
        "task_split_key": args.task_split_key,
        "split_metadata": split_metadata,
        "heads_path": str(args.heads_path),
        "heads_source_tasks": heads_data.get("tasks"),
        "n_top_heads": len(top_heads),
        "top_heads": top_heads,
        "comparisons": {},
    }

    for task in tasks:
        print(f"\n=== {task} ===")
        task_output_dir = args.output_root / task
        task_output_dir.mkdir(parents=True, exist_ok=True)
        task_summary_path = task_output_dir / "comparison_summary.json"
        if task_summary_path.exists() and not args.overwrite:
            raise FileExistsError(f"{task_summary_path} exists. Pass --overwrite to replace it.")

        set_seed(args.seed)
        dataset = load_dataset(task, root_data_dir=args.root_data_dir, test_size=args.test_split, seed=args.seed)

        mean_path = args.fv_root / task / f"{task}_mean_head_activations.pt"
        task_specific_fv_path = args.fv_root / task / f"{task}_function_vector.pt"
        if not mean_path.exists():
            raise FileNotFoundError(mean_path)
        if not task_specific_fv_path.exists():
            raise FileNotFoundError(task_specific_fv_path)

        mean_activations = torch_load_trusted(mean_path, map_location="cpu")
        multitask_fv = compute_function_vector_from_heads(mean_activations, top_heads, model, model_config)
        multitask_fv = multitask_fv.detach().float().cpu().reshape(-1)
        task_specific_fv, task_specific_heads = load_function_vector(task_specific_fv_path)

        multitask_fv_path = task_output_dir / f"{task}_function_vector_multitask_top{len(top_heads)}.pt"
        torch.save(
            {
                "function_vector": multitask_fv,
                "top_heads": top_heads,
                "n_top_heads": len(top_heads),
                "task": task,
                "heads_path": str(args.heads_path),
                "mean_activations_path": str(mean_path),
            },
            multitask_fv_path,
        )

        filter_set, filter_source = get_filter_set(args, task, dataset, model, model_config, tokenizer, task_output_dir)
        args.filter_set = filter_set

        print("Evaluating multitask-head FV")
        multitask_zs, multitask_fs = evaluate_fv(args, dataset, multitask_fv, model, model_config, tokenizer)
        print("Evaluating task-specific FV")
        task_specific_zs, task_specific_fs = evaluate_fv(args, dataset, task_specific_fv, model, model_config, tokenizer)

        write_json(task_output_dir / "multitask_heads_zs_results.json", multitask_zs)
        write_json(task_output_dir / "multitask_heads_fs_shuffled_results.json", multitask_fs)
        write_json(task_output_dir / "task_specific_zs_results.json", task_specific_zs)
        write_json(task_output_dir / "task_specific_fs_shuffled_results.json", task_specific_fs)

        task_summary = {
            "task": task,
            "multitask_fv_path": str(multitask_fv_path),
            "task_specific_fv_path": str(task_specific_fv_path),
            "mean_activations_path": str(mean_path),
            "filter_to_correct_icl": args.filter_to_correct_icl,
            "filter_source": filter_source,
            "n_filtered_test_examples": None if filter_set is None else int(len(filter_set)),
            "multitask_heads": {
                "top_heads": top_heads,
                **summarize_results(multitask_zs, multitask_fs),
            },
            "task_specific_heads": {
                "top_heads": task_specific_heads,
                **summarize_results(task_specific_zs, task_specific_fs),
            },
        }
        write_json(task_summary_path, task_summary)
        aggregate["comparisons"][task] = task_summary

    aggregate_path = args.output_root / "heldout_multitask_vs_task_specific_summary.json"
    write_json(aggregate_path, aggregate)
    print(aggregate_path)


if __name__ == "__main__":
    main()
