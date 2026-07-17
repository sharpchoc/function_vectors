#!/usr/bin/env python
"""SANDBOX (not repo standard): per-prompt attention-head captures for FV-target regressions.

For every prompt of the fixed 10-shot residual-activation capture set (29 manifest tasks x
170 prompts, seed 42), this script rebuilds the exact same prompt (same hash-based RNGs as
`extract_targeted_residual_stream_activations.py`, asserted against the stored index.json
query_indices) and records, at the final prompt token:

  * all 448 head activations: the input to each layer's attn out_proj, viewed as
    (n_layers=28, n_heads=16, head_dim=256), fp16
  * the per-prompt "head-sum target": sum over the FV's top-40 heads of
    out_proj(zero-padded head slot) -- the same construction as
    `compute_function_vector_from_heads`, applied to THIS prompt's activations instead of
    task means. By linearity, the per-task mean of these targets equals the FV built from
    this prompt set's mean head activations (verified per task; hard stop on failure).

Output mirrors the residual-capture shard format under
  artifacts/sandbox/perprompt_head_acts/<name>/<task>/<split>/{shard_*.pt, index.json}
plus a one-off top40_outproj_slices.pt (fp16 W_O slices) at the root for CPU-side reuse.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from baukit import TraceDict
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from extract_targeted_residual_stream_activations import (  # noqa: E402
    make_prompt,
    sample_demo_indices,
    sample_query_indices,
    selected_token_records,
)
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed  # noqa: E402
from utils.paths import ARTIFACTS_ROOT  # noqa: E402
from utils.prompt_utils import get_token_meta_labels, load_dataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="SANDBOX: capture per-prompt head activations + top-40 head-sum targets.")
    p.add_argument("--task_manifest", type=Path, default=REPO_ROOT / "task_splits/abstractive_train_test_tasks_29.json")
    p.add_argument("--tasks", nargs="+", default=None, help="Override: subset of tasks (default: all manifest tasks).")
    p.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "valid", "test"])
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--root_data_dir", type=str, default=str(REPO_ROOT / "dataset_files"))
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_varicl_top40",
                   help="FV set whose heads.pt defines the top-40 heads and whose FVs are compared informationally.")
    p.add_argument("--reference_activations_root", type=Path,
                   default=ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_4tokens",
                   help="Existing residual capture whose index.json query_indices must be reproduced exactly.")
    p.add_argument("--output_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox/perprompt_head_acts/gptj_train_varicl_top40")
    # Prompt-sampling knobs -- MUST match the reference capture (asserted via query_indices).
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_split", type=float, default=0.3)
    p.add_argument("--n_shots", type=int, default=10)
    p.add_argument("--max_train_prompts", type=int, default=130)
    p.add_argument("--max_test_prompts", type=int, default=40)
    p.add_argument("--max_valid_prompts", type=int, default=None)
    p.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    p.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    p.add_argument("--shard_size", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--mean_check_min_cos", type=float, default=0.999,
                   help="Linearity gate: cosine between mean(per-prompt targets) and FV rebuilt from mean head acts.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def torch_load_trusted(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def get_out_projs(model, model_config):
    """Per-layer out_proj modules, mirroring compute_function_vector's model dispatch (GPT-J here)."""
    name = model_config["name_or_path"]
    if "gpt-j" not in name:
        raise NotImplementedError(f"SANDBOX capture only wired for GPT-J, got {name}")
    return [model.transformer.h[layer].attn.out_proj for layer in range(model_config["n_layers"])]


def headsum_target(head_acts, top_heads, out_projs, resid_dim, head_dim):
    """Sum over top heads of out_proj(zero-padded head slot). head_acts: [n_layers, n_heads, head_dim].

    Exactly replicates compute_function_vector_from_heads' per-head zero-pad + module call,
    batched per layer (GPT-J out_proj is bias-free, so per-head vs batched is identical anyway).
    """
    device = head_acts.device
    target = torch.zeros(resid_dim, device=device, dtype=torch.float32)
    by_layer = {}
    for layer, head, *_ in top_heads:
        by_layer.setdefault(int(layer), []).append(int(head))
    for layer, heads in by_layer.items():
        x = torch.zeros(len(heads), resid_dim, device=device, dtype=head_acts.dtype)
        for i, head in enumerate(heads):
            x[i, head * head_dim:(head + 1) * head_dim] = head_acts[layer, head]
        target += out_projs[layer](x).float().sum(dim=0)
    return target


def build_fv_from_mean(mean_head_acts, top_heads, out_projs, resid_dim, head_dim, model_dtype):
    """FV from mean head activations, same construction (used for the linearity gate)."""
    return headsum_target(mean_head_acts.to(model_dtype), top_heads, out_projs, resid_dim, head_dim)


def flush_shard(head_acts, targets, metadata, output_dir, shard_index, config):
    if not head_acts:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"shard_{shard_index:05d}.pt"
    torch.save(
        {
            "head_activations": torch.stack(head_acts, dim=0),   # [n, n_layers, n_heads, head_dim] fp16
            "targets": torch.stack(targets, dim=0),              # [n, resid_dim] fp32
            "metadata": metadata,
            "config": config,
        },
        output_path,
    )
    return output_path


def capture_split(task, split, dataset, model, tokenizer, model_config, out_projs, top_heads, args):
    n_layers = model_config["n_layers"]
    n_heads = model_config["n_heads"]
    resid_dim = model_config["resid_dim"]
    head_dim = resid_dim // n_heads
    attn_hooks = model_config["attn_hook_names"]

    query_indices = sample_query_indices(task, split, len(dataset[split]), args)

    # ---- HARD ALIGNMENT GATE: regenerated prompts must be the stored capture's prompts ----
    ref_index_path = args.reference_activations_root / task / split / "index.json"
    ref_config = load_json(ref_index_path)["config"]
    ref_queries = [int(x) for x in ref_config["query_indices"]]
    if [int(x) for x in query_indices] != ref_queries:
        raise RuntimeError(
            f"ALIGNMENT GATE FAILED for {task}/{split}: regenerated query_indices do not match "
            f"{ref_index_path}. STOP -- user adjudicates. First 5 regenerated={query_indices[:5]}, "
            f"stored={ref_queries[:5]}"
        )
    for key, ours in (("seed", args.seed), ("n_shots", args.n_shots), ("prefixes", args.prefixes),
                      ("separators", args.separators), ("test_split", args.test_split)):
        if ref_config.get(key) != ours:
            raise RuntimeError(f"ALIGNMENT GATE FAILED for {task}/{split}: config {key!r} mismatch: "
                               f"ours={ours!r} stored={ref_config.get(key)!r}")

    output_dir = args.output_root / task / split
    config = {
        "sandbox": True,
        "task": task,
        "split": split,
        "model_name": args.model_name,
        "seed": args.seed,
        "test_split": args.test_split,
        "n_shots": args.n_shots,
        "sampled_prompt_count": len(query_indices),
        "available_prompt_count": len(dataset[split]),
        "query_indices": [int(x) for x in query_indices],
        "prefixes": args.prefixes,
        "separators": args.separators,
        "fv_root": str(args.fv_root),
        "top_heads": [[int(l), int(h)] for l, h, *_ in top_heads],
        "reference_activations_root": str(args.reference_activations_root),
        "head_activations_dtype": "torch.float16",
        "targets_dtype": "torch.float32",
        "target_construction": "sum over top-40 heads of out_proj(zero-padded per-prompt head activation) at final prompt token",
    }

    shard_heads, shard_targets, shard_metadata, shard_paths = [], [], [], []
    shard_index = 0
    sum_targets = torch.zeros(resid_dim, dtype=torch.float64)
    sum_head_acts = torch.zeros(n_layers, n_heads, head_dim, dtype=torch.float64)

    for prompt_index, query_idx in enumerate(tqdm(query_indices, desc=f"{task}/{split}")):
        demo_indices = sample_demo_indices(task, split, int(query_idx), dataset, args)
        prompt_data = make_prompt(dataset, split, int(query_idx), demo_indices, model_config,
                                  args.prefixes, args.separators)
        query = prompt_data["query_target"]["input"]
        token_labels, prompt_string = get_token_meta_labels(
            prompt_data, tokenizer, query=query, prepend_bos=model_config["prepend_bos"]
        )
        (final_record,) = selected_token_records(token_labels, args.n_shots, ["last_prompt_token"])
        token_position = final_record["token_position"]

        inputs = tokenizer([prompt_string], return_tensors="pt").to(model.device)
        if token_position >= inputs["input_ids"].shape[1]:
            raise IndexError(f"{task}/{split} prompt {prompt_index}: token position {token_position} "
                             f"exceeds sequence length {inputs['input_ids'].shape[1]}")
        with TraceDict(model, layers=attn_hooks, retain_input=True, retain_output=False) as td:
            model(**inputs)

        head_acts = torch.stack(
            [td[hook].input[0, token_position, :].view(n_heads, head_dim) for hook in attn_hooks],
            dim=0,
        )  # [n_layers, n_heads, head_dim], model dtype
        target = headsum_target(head_acts, top_heads, out_projs, resid_dim, head_dim)

        sum_targets += target.double().cpu()
        sum_head_acts += head_acts.double().cpu()
        shard_heads.append(head_acts.cpu().to(torch.float16))
        shard_targets.append(target.cpu())
        shard_metadata.append({
            "task": task,
            "split": split,
            "prompt_index": int(prompt_index),
            "query_source_index": int(query_idx),
            "query_input": prompt_data["query_target"]["input"].strip(),
            "query_output": prompt_data["query_target"]["output"].strip(),
            "token_role": "last_prompt_token",
            "icl_example_index": None,
            "token_position": int(token_position),
            "token_text": final_record["token_text"],
            "token_label": final_record["token_label"],
        })

        if len(shard_heads) >= args.shard_size:
            shard_paths.append(str(flush_shard(shard_heads, shard_targets, shard_metadata,
                                               output_dir, shard_index, config)))
            shard_heads, shard_targets, shard_metadata = [], [], []
            shard_index += 1

    shard_path = flush_shard(shard_heads, shard_targets, shard_metadata, output_dir, shard_index, config)
    if shard_path is not None:
        shard_paths.append(str(shard_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "index.json", "w") as f:
        json.dump({"config": config, "shards": [Path(s).name for s in shard_paths]}, f, indent=2)

    n = len(query_indices)
    return sum_targets / n, sum_head_acts / n, n


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_grad_enabled(False)

    manifest = load_json(args.task_manifest)
    tasks = list(args.tasks) if args.tasks is not None else list(manifest["train_tasks"]) + list(manifest["test_tasks"])

    heads_data = torch_load_trusted(args.fv_root / "heads.pt", map_location="cpu")
    top_heads = heads_data["top_heads"]
    print(f"[sandbox-capture] {len(tasks)} tasks | {len(top_heads)} top heads from {args.fv_root / 'heads.pt'}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "capture_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"{summary_path} exists; pass --overwrite to replace.")

    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device,
                                                                  revision=args.revision)
    model.eval()
    model_dtype = next(model.parameters()).dtype
    out_projs = get_out_projs(model, model_config)
    n_heads = model_config["n_heads"]
    resid_dim = model_config["resid_dim"]
    head_dim = resid_dim // n_heads

    # One-off: save the top-40 W_O slices for CPU-side target reconstruction.
    slices = {
        f"L{int(l)}H{int(h)}": out_projs[int(l)].weight[:, int(h) * head_dim:(int(h) + 1) * head_dim].cpu().clone()
        for l, h, *_ in top_heads
    }
    torch.save({"sandbox": True, "top_heads": [[int(l), int(h)] for l, h, *_ in top_heads],
                "fv_root": str(args.fv_root), "out_proj_bias": False,
                "slices": slices}, args.output_root / "top40_outproj_slices.pt")

    summary = []
    for task in tasks:
        print(f"Loading Dataset: {task}")
        dataset = load_dataset(task, root_data_dir=args.root_data_dir, test_size=args.test_split, seed=args.seed)
        task_sum_targets = torch.zeros(resid_dim, dtype=torch.float64)
        task_sum_heads = torch.zeros(model_config["n_layers"], n_heads, head_dim, dtype=torch.float64)
        task_n = 0
        for split in args.splits:
            mean_targets, mean_heads, n = capture_split(task, split, dataset, model, tokenizer, model_config,
                                                        out_projs, top_heads, args)
            task_sum_targets += mean_targets * n
            task_sum_heads += mean_heads * n
            task_n += n

        # ---- LINEARITY GATE: mean(per-prompt targets) == FV(mean head activations) ----
        mean_targets = (task_sum_targets / task_n).float()
        mean_heads = (task_sum_heads / task_n).float()
        fv_from_mean = build_fv_from_mean(mean_heads.to(model.device), top_heads, out_projs,
                                          resid_dim, head_dim, model_dtype).cpu()
        cos_lin = torch.nn.functional.cosine_similarity(mean_targets, fv_from_mean, dim=0).item()
        rel_l2 = (mean_targets - fv_from_mean).norm().item() / max(fv_from_mean.norm().item(), 1e-12)
        if cos_lin < args.mean_check_min_cos:
            raise RuntimeError(
                f"LINEARITY GATE FAILED for {task}: cos(mean targets, FV-from-mean-heads)={cos_lin:.6f} "
                f"rel_l2={rel_l2:.6f}. STOP -- user adjudicates."
            )

        # Informational: distance to the stored varicl FV (different prompt set; expected close, not equal).
        stored = torch_load_trusted(args.fv_root / task / f"{task}_function_vector.pt", map_location="cpu")
        stored_fv = (stored["function_vector"] if isinstance(stored, dict) else stored).float().reshape(-1)
        cos_stored = torch.nn.functional.cosine_similarity(mean_targets, stored_fv, dim=0).item()
        mse_stored = torch.mean((mean_targets - stored_fv) ** 2).item()
        row = {
            "task": task, "n_prompts": task_n,
            "linearity_cos": cos_lin, "linearity_rel_l2": rel_l2,
            "cos_vs_stored_varicl_fv": cos_stored, "mse_vs_stored_varicl_fv": mse_stored,
            "mean_target_norm": mean_targets.norm().item(), "stored_fv_norm": stored_fv.norm().item(),
        }
        summary.append(row)
        print(f"[sandbox-capture] {task}: linearity cos={cos_lin:.6f} rel_l2={rel_l2:.2e} | "
              f"vs stored FV cos={cos_stored:.4f} mse={mse_stored:.5f} "
              f"(||mean tgt||={row['mean_target_norm']:.3f} ||stored||={row['stored_fv_norm']:.3f})")

    with open(summary_path, "w") as f:
        json.dump({"sandbox": True, "tasks": summary,
                   "fv_root": str(args.fv_root),
                   "output_root": str(args.output_root)}, f, indent=2)
    print(f"[sandbox-capture] DONE: {len(summary)} tasks -> {summary_path}")


if __name__ == "__main__":
    main()
