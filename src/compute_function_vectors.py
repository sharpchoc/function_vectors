import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from compute_indirect_effect import compute_indirect_effect
from utils.eval_utils import n_shot_eval_no_intervention
from utils.extract_utils import compute_function_vector, compute_universal_function_vector, get_mean_head_activations
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and store task function vectors without post-FV intervention/baseline evals."
    )
    parser.add_argument("--dataset_names", nargs="+", required=True, help="Task names without .json suffix.")
    parser.add_argument("--n_top_heads", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    parser.add_argument("--root_data_dir", type=str, default="dataset_files")
    parser.add_argument("--save_path_root", type=str, default="results/gptj_fv")
    parser.add_argument("--ie_path_root", type=str, default=None, help="Optional root to reuse mean/IE tensors from another result tree.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test_split", type=float, default=0.3)
    parser.add_argument("--n_shots", type=int, default=10)
    parser.add_argument("--n_mean_activations_trials", type=int, default=100)
    parser.add_argument("--n_indirect_effect_trials", type=int, default=25)
    parser.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    parser.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    parser.add_argument("--generate_str", action="store_true", help="Use generated-string scoring for the pre-FV ICL filter.")
    parser.add_argument("--metric", type=str, default="f1_score")
    parser.add_argument("--universal_set", action="store_true", help="Use hard-coded universal heads instead of task indirect effects.")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if {task}_function_vector.pt already exists.")
    parser.add_argument("--filter_to_correct_icl", dest="filter_to_correct_icl", action="store_true")
    parser.add_argument("--no_filter_to_correct_icl", dest="filter_to_correct_icl", action="store_false")
    parser.set_defaults(filter_to_correct_icl=True)
    parser.add_argument("--continue_on_error", action="store_true", help="Log task failures and continue with later tasks.")
    return parser.parse_args()


def torch_load_trusted(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def task_dir(root, dataset_name):
    return Path(root) / dataset_name


def json_safe_top_heads(top_heads):
    return [(int(layer), int(head), float(score)) for layer, head, score in top_heads]


def load_or_compute_filter_set(args, dataset_name, dataset, model, model_config, tokenizer, save_dir):
    if not args.filter_to_correct_icl:
        return None

    fs_results_file_name = save_dir / "fs_results_layer_sweep.json"
    fs_validation_file_name = save_dir / "fs_results_validation.json"
    key = "score" if args.generate_str else "clean_rank_list"
    target_val = 1 if args.generate_str else 0

    filter_set_validation = None
    if fs_results_file_name.exists():
        with open(fs_results_file_name, "r") as f:
            fs_results = json.load(f)
        if key not in fs_results:
            raise KeyError(f"{fs_results_file_name} does not contain key '{key}'")
        print(f"Loaded existing ICL filter results: {fs_results_file_name}")
    else:
        print(f"Computing pre-FV {args.n_shots}-shot ICL filter for {dataset_name}")
        set_seed(args.seed + 42)
        fs_results_validation = n_shot_eval_no_intervention(
            dataset=dataset,
            n_shots=args.n_shots,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            compute_ppl=not args.generate_str,
            generate_str=args.generate_str,
            metric=args.metric,
            test_split="valid",
            prefixes=args.prefixes,
            separators=args.separators,
        )
        with open(fs_validation_file_name, "w") as f:
            json.dump(fs_results_validation, f, indent=2)
        filter_set_validation = np.where(np.array(fs_results_validation[key]) == target_val)[0]

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
        )
        with open(fs_results_file_name, "w") as f:
            json.dump(fs_results, f, indent=2)

    if filter_set_validation is None:
        if fs_validation_file_name.exists():
            with open(fs_validation_file_name, "r") as f:
                fs_results_validation = json.load(f)
            if key in fs_results_validation:
                filter_set_validation = np.where(np.array(fs_results_validation[key]) == target_val)[0]
        if filter_set_validation is None:
            print("Validation filter unavailable; mean/IE computation will use all validation examples.")

    filter_set = np.where(np.array(fs_results[key]) == target_val)[0]
    with open(save_dir / "icl_filter_metadata.json", "w") as f:
        json.dump(
            {
                "filter_to_correct_icl": args.filter_to_correct_icl,
                "metric_key": key,
                "target_val": target_val,
                "n_test_correct": int(len(filter_set)),
                "n_validation_correct": None if filter_set_validation is None else int(len(filter_set_validation)),
            },
            f,
            indent=2,
        )
    return filter_set_validation


def load_or_compute_mean_activations(args, dataset_name, dataset, model, model_config, tokenizer, save_dir, load_dir, filter_set_validation):
    save_path = save_dir / f"{dataset_name}_mean_head_activations.pt"
    load_path = load_dir / f"{dataset_name}_mean_head_activations.pt"
    if load_path.exists() and not args.overwrite:
        print(f"Loading mean activations: {load_path}")
        mean_activations = torch_load_trusted(load_path, map_location="cpu")
        if save_path != load_path and not save_path.exists():
            torch.save(mean_activations, save_path)
        return mean_activations, str(load_path)

    print(f"Computing mean head activations for {dataset_name}")
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
        filter_set=filter_set_validation,
    )
    torch.save(mean_activations, save_path)
    return mean_activations, str(save_path)


def load_or_compute_indirect_effect(args, dataset_name, dataset, mean_activations, model, model_config, tokenizer, save_dir, load_dir, filter_set_validation):
    if args.universal_set:
        return None, None

    save_path = save_dir / f"{dataset_name}_indirect_effect.pt"
    load_path = load_dir / f"{dataset_name}_indirect_effect.pt"
    if load_path.exists() and not args.overwrite:
        print(f"Loading indirect effect: {load_path}")
        indirect_effect = torch_load_trusted(load_path, map_location="cpu")
        if save_path != load_path and not save_path.exists():
            torch.save(indirect_effect, save_path)
        return indirect_effect, str(load_path)

    print(f"Computing indirect effects for {dataset_name}")
    set_seed(args.seed)
    indirect_effect = compute_indirect_effect(
        dataset,
        mean_activations,
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        n_shots=args.n_shots,
        n_trials=args.n_indirect_effect_trials,
        last_token_only=True,
        prefixes=args.prefixes,
        separators=args.separators,
        filter_set=filter_set_validation,
    )
    torch.save(indirect_effect, save_path)
    return indirect_effect, str(save_path)


def compute_and_save_task_fv(args, dataset_name, model, model_config, tokenizer):
    save_dir = task_dir(args.save_path_root, dataset_name)
    load_root = args.ie_path_root if args.ie_path_root is not None else args.save_path_root
    load_dir = task_dir(load_root, dataset_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    fv_path = save_dir / f"{dataset_name}_function_vector.pt"
    if fv_path.exists() and not args.overwrite:
        print(f"Skipping {dataset_name}; function vector already exists: {fv_path}")
        return {"dataset_name": dataset_name, "status": "skipped", "function_vector_path": str(fv_path)}

    print(f"\n=== {dataset_name} ===")
    set_seed(args.seed)
    dataset = load_dataset(dataset_name, root_data_dir=args.root_data_dir, test_size=args.test_split, seed=args.seed)

    filter_set_validation = load_or_compute_filter_set(
        args, dataset_name, dataset, model, model_config, tokenizer, save_dir
    )

    mean_activations, mean_path = load_or_compute_mean_activations(
        args, dataset_name, dataset, model, model_config, tokenizer, save_dir, load_dir, filter_set_validation
    )
    indirect_effect, indirect_path = load_or_compute_indirect_effect(
        args, dataset_name, dataset, mean_activations, model, model_config, tokenizer, save_dir, load_dir, filter_set_validation
    )

    print(f"Computing function vector for {dataset_name}")
    if args.universal_set:
        fv, top_heads = compute_universal_function_vector(
            mean_activations, model, model_config=model_config, n_top_heads=args.n_top_heads
        )
    else:
        fv, top_heads = compute_function_vector(
            mean_activations, indirect_effect, model, model_config=model_config, n_top_heads=args.n_top_heads
        )

    fv = fv.detach().float().cpu().reshape(-1)
    top_heads_json = json_safe_top_heads(top_heads)
    torch.save(
        {
            "function_vector": fv,
            "top_heads": top_heads_json,
            "n_top_heads": args.n_top_heads,
            "dataset_name": dataset_name,
            "model_name": args.model_name,
            "seed": args.seed,
        },
        fv_path,
    )

    metadata = {
        "dataset_name": dataset_name,
        "model_name": args.model_name,
        "model_config": model_config,
        "seed": args.seed,
        "test_split": args.test_split,
        "n_shots": args.n_shots,
        "n_mean_activations_trials": args.n_mean_activations_trials,
        "n_indirect_effect_trials": args.n_indirect_effect_trials,
        "n_top_heads": args.n_top_heads,
        "universal_set": args.universal_set,
        "filter_to_correct_icl": args.filter_to_correct_icl,
        "prefixes": args.prefixes,
        "separators": args.separators,
        "mean_activations_path": mean_path,
        "indirect_effect_path": indirect_path,
        "function_vector_path": str(fv_path),
        "top_heads": top_heads_json,
    }
    metadata_path = save_dir / f"{dataset_name}_function_vector_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved function vector: {fv_path}")
    return {"dataset_name": dataset_name, "status": "ok", "function_vector_path": str(fv_path)}


def main():
    args = parse_args()
    Path(args.save_path_root).mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision
    )
    model.eval()

    batch_results = []
    for dataset_name in args.dataset_names:
        try:
            result = compute_and_save_task_fv(args, dataset_name, model, model_config, tokenizer)
            batch_results.append(result)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"ERROR for {dataset_name}: {exc}")
            batch_results.append({"dataset_name": dataset_name, "status": "error", "error": repr(exc)})

    batch_log_path = Path(args.save_path_root) / "function_vector_batch_log.json"
    with open(batch_log_path, "w") as f:
        json.dump({"args": vars(args), "results": batch_results}, f, indent=2)
    print(f"Wrote batch log: {batch_log_path}")


if __name__ == "__main__":
    main()
