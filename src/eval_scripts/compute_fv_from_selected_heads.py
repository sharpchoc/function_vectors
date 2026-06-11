#!/usr/bin/env python
"""Build a function vector for a task from a precomputed set of selected heads.

Goes quickly from "which heads do we want" -> "function vector for task X" by combining:
  * a head set, from a head-selection artifact (multitask_top_aie_heads.pt or a subset
    selection produced by select_heads_from_cie_subset.py), or given explicitly via --heads
  * that task's stored mean head activations (reused from results/gptj_fv/<task>/ or any
    --mean_activations_root; computed and saved once if not present anywhere)

The FV is the sum over selected heads (L,H) of out_proj applied to the head's mean
last-token activation -- identical to compute_universal_function_vector, but with our heads.

Examples:
  # FV for one task using the all-tasks multitask head selection
  python src/eval_scripts/compute_fv_from_selected_heads.py \
    --heads_artifact results/multitask_aie_heads_all_tasks/multitask_top_aie_heads.pt \
    --tasks antonym --n_top_heads 40

  # FVs for several tasks from a subset selection
  python src/eval_scripts/compute_fv_from_selected_heads.py \
    --heads_artifact results/multitask_aie_heads_all_tasks/subsets/lexical_top_heads.pt \
    --tasks antonym synonym country-capital
"""
import argparse
import json
from pathlib import Path

import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.utils.extract_utils import get_mean_head_activations
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.prompt_utils import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", required=True, help="Task(s) to build a function vector for.")
    parser.add_argument(
        "--heads_artifact",
        type=Path,
        default=None,
        help="A .pt selection with a 'top_heads' list (multitask_top_aie_heads.pt or a subset selection).",
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        default=None,
        help="Explicit heads as 'L,H' pairs (overrides --heads_artifact). e.g. --heads 15,5 9,14",
    )
    parser.add_argument(
        "--n_top_heads",
        type=int,
        default=None,
        help="Use only the first N heads from the artifact/list (defaults to all available).",
    )
    parser.add_argument(
        "--mean_activations_root",
        type=Path,
        nargs="+",
        default=[Path("results/gptj_fv")],
        help="Root(s) searched (in order) for <task>/<task>_mean_head_activations.pt.",
    )
    parser.add_argument(
        "--save_path_root",
        type=Path,
        default=Path("results/multitask_aie_heads_all_tasks"),
        help="Where to write FVs and any newly-computed mean activations.",
    )
    parser.add_argument("--fv_tag", type=str, default="selected_heads", help="Filename tag for the saved FV.")
    parser.add_argument("--root_data_dir", type=str, default="dataset_files")
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_split", type=float, default=0.3)
    parser.add_argument("--n_shots", type=int, default=10)
    parser.add_argument("--n_mean_activations_trials", type=int, default=100)
    parser.add_argument("--batch_size_mean_activations", type=int, default=1)
    parser.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    parser.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    return parser.parse_args()


def torch_load_trusted(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def resolve_top_heads(args):
    if args.heads is not None:
        heads = []
        for pair in args.heads:
            layer, head = pair.replace("(", "").replace(")", "").split(",")[:2]
            heads.append((int(layer), int(head), None))
    elif args.heads_artifact is not None:
        artifact = torch_load_trusted(args.heads_artifact, map_location="cpu")
        if "top_heads" not in artifact:
            raise KeyError(f"{args.heads_artifact} has no 'top_heads' entry.")
        heads = [(int(t[0]), int(t[1]), (float(t[2]) if len(t) > 2 else None)) for t in artifact["top_heads"]]
    else:
        raise ValueError("Provide --heads_artifact or --heads.")
    if args.n_top_heads is not None:
        heads = heads[: args.n_top_heads]
    return heads


def mean_activations_path(root, task):
    return root / task / f"{task}_mean_head_activations.pt"


def load_or_compute_mean_activations(args, task, model, model_config, tokenizer):
    for root in args.mean_activations_root:
        path = mean_activations_path(root, task)
        if path.exists():
            print(f"  reusing mean activations: {path}")
            return torch_load_trusted(path, map_location="cpu")

    # Not stored anywhere -> compute once and persist under save_path_root.
    print(f"  computing mean activations for {task} (not found in {args.mean_activations_root})")
    dataset = load_dataset(task, root_data_dir=args.root_data_dir, test_size=args.test_split, seed=args.seed)
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
    ).detach().cpu()
    out_path = mean_activations_path(args.save_path_root, task)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mean_activations, out_path)
    print(f"  saved mean activations: {out_path}")
    return mean_activations


def build_function_vector(mean_activations, top_heads, model, model_config):
    resid_dim = model_config["resid_dim"]
    n_heads = model_config["n_heads"]
    head_dim = resid_dim // n_heads
    device = model.device
    name = model_config["name_or_path"]
    T = -1  # last token

    function_vector = torch.zeros((1, 1, resid_dim)).to(device)
    for L, H, _ in top_heads:
        if "gpt2-xl" in name:
            out_proj = model.transformer.h[L].attn.c_proj
        elif "gpt-j" in name:
            out_proj = model.transformer.h[L].attn.out_proj
        elif "llama" in name or "qwen" in name.lower():
            out_proj = model.model.layers[L].self_attn.o_proj
        elif "gpt-neox" in name:
            out_proj = model.gpt_neox.layers[L].attention.dense
        else:
            raise NotImplementedError(f"Unknown out_proj layout for {name}")

        x = torch.zeros(resid_dim)
        x[H * head_dim:(H + 1) * head_dim] = mean_activations[L, H, T]
        function_vector += out_proj(x.reshape(1, 1, resid_dim).to(device).to(model.dtype))
        function_vector = function_vector.to(model.dtype)
    return function_vector.reshape(1, resid_dim)


def main():
    args = parse_args()
    top_heads = resolve_top_heads(args)
    print(f"Using {len(top_heads)} heads: {[(L, H) for L, H, _ in top_heads]}")

    torch.set_grad_enabled(False)
    set_seed(args.seed)
    print("Loading model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision
    )
    model.eval()

    args.save_path_root.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        print(f"\n=== {task} ===")
        mean_activations = load_or_compute_mean_activations(args, task, model, model_config, tokenizer)
        function_vector = build_function_vector(mean_activations, top_heads, model, model_config)

        task_dir = args.save_path_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        fv_path = task_dir / f"{task}_fv_{args.fv_tag}.pt"
        torch.save(function_vector.detach().cpu(), fv_path)
        meta_path = task_dir / f"{task}_fv_{args.fv_tag}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "task": task,
                    "model_name": args.model_name,
                    "n_heads_used": len(top_heads),
                    "top_heads": [[L, H, s] for L, H, s in top_heads],
                    "heads_artifact": None if args.heads_artifact is None else str(args.heads_artifact),
                    "fv_shape": list(function_vector.shape),
                    "fv_path": str(fv_path),
                },
                f,
                indent=2,
            )
        print(f"  saved FV {tuple(function_vector.shape)}: {fv_path}")


if __name__ == "__main__":
    main()
