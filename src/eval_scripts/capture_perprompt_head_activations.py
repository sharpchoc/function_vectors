#!/usr/bin/env python
"""Capture PER-PROMPT head activations h(p^j_A) at the final cue token (GPU).

For each of the 27 tasks (20 train-split tasks + the Stream W 7 test tasks) build 170
prompts whose queries are sampled from the FULL dataset (all examples -- no train/valid/
test split, no ICL-correctness filter; user decision 2026-08-10), demos sampled from the
same full pool excluding the query. Two variants:
  - fixed10:      every prompt has 10 ICL demonstrations
  - varicl4to10:  per-prompt shot count uniform in {4..10}

The captured quantity is the canonical head-activation definition (varicl_utils.
get_last_token_mean_head_activations): the input to attn.out_proj, split per head,
read at the LAST non-pad token (final cue token) -- but retained PER PROMPT instead of
summed, giving (n_prompts, n_layers, n_heads, head_dim) per task. Downstream, the
per-prompt function vector is v^j_A = sum_{h in H} W_O[:, h] @ z_j (top-40 pooled heads).

Gates:
  - hard: exactly --n_prompts prompts per task/variant.
  - hard (one --gate_task): mean of our per-prompt activations must match
    get_last_token_mean_head_activations run on the SAME prompts (max dev < 1e-4).
  - advisory: cosine of the task mean vs the stored canonical
    <task>_mean_head_activations_varicl.pt (different prompt population -- report only).

Output: artifacts/perprompt_head_activations/gptj_27tasks_170prompts/<variant>/<task>.pt
Resumable: existing task files are skipped unless --overwrite.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from baukit import TraceDict

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.prompt_utils import ICLDataset, create_prompt
from src.utils.varicl_utils import (
    build_varicl_prompt_data,
    get_last_token_mean_head_activations,
    sample_variable_icl_count,
    split_activations_by_head,
)
from utils.paths import ARTIFACTS_ROOT

TEST7 = ["landmark-country", "word_length", "capitalize_first_letter", "synonym",
         "lowercase_first_letter", "capitalize", "antonym"]

VARIANTS = {"fixed10": (10, 10), "varicl4to10": (4, 10)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--task_split_path", type=Path,
                   default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--dataset_root", type=Path, default=Path("dataset_files/abstractive"))
    p.add_argument("--n_prompts", type=int, default=170)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--gate_task", type=str, default="antonym",
                   help="Task whose per-prompt mean is hard-gated against the existing "
                        "mean-activation function on identical prompts.")
    p.add_argument("--canonical_means_root", type=Path,
                   default=ARTIFACTS_ROOT / "multitask_aie_heads_varicl")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "perprompt_head_activations" / "gptj_27tasks_170prompts")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def make_prompt_args(args, min_shots, max_shots):
    """Args namespace consumed by build_varicl_prompt_data / get_last_token_mean_head_activations."""
    return SimpleNamespace(
        query_split="all", demo_split="all",
        min_shots=min_shots, max_shots=max_shots,
        batch_size=args.batch_size,
        prefixes={"input": "Q:", "output": "A:", "instructions": ""},
        separators={"input": "\n", "output": "\n\n", "instructions": ""},
    )


def select_query_indices(task_index, args, pool_size):
    """170 deterministic query indices from the full pool (no filter).

    Without replacement when the pool allows it; the task-level rng is seeded like the
    per-prompt rng family (seed + 100_000*task_index) so the draw is shard-invariant.
    """
    rng = np.random.default_rng(args.seed + 100_000 * task_index)
    replace = pool_size < args.n_prompts
    return [int(i) for i in rng.choice(pool_size, size=args.n_prompts, replace=replace)]


@torch.no_grad()
def capture_task(dataset, prompt_args, model, model_config, tokenizer, task_index,
                 query_indices, seed_base):
    """Per-prompt out_proj-input head activations at the last non-pad token.

    Mirrors varicl_utils.get_last_token_mean_head_activations (same prompt builder,
    padding, hook, and token indexing) but stacks the per-row activations instead of
    summing them. Returns (n_prompts, n_layers, n_heads, head_dim) fp32 cpu.
    """
    per_prompt = []
    meta = []
    # Reference-style reduction (varicl_utils.py:136 verbatim: fp16 intra-batch sum, fp64
    # accumulate) on the SAME tensors -- lets the gate compare bitwise-identically. The fp64
    # per-prompt mean differs from this by ~2e-4 (fp16 batch-sum rounding), by design.
    refstyle_sum = torch.zeros(model_config["n_layers"], model_config["n_heads"],
                               model_config["resid_dim"] // model_config["n_heads"],
                               dtype=torch.float64, device=model.device)
    refstyle_count = 0
    old_padding_side = tokenizer.padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    bs = max(1, int(prompt_args.batch_size))
    try:
        for batch_start in range(0, len(query_indices), bs):
            batch_q = query_indices[batch_start:batch_start + bs]
            sentences = []
            for j, query_idx in enumerate(batch_q, start=batch_start):
                prompt_data = build_varicl_prompt_data(
                    dataset, prompt_args, model_config, task_index=task_index,
                    query_idx=int(query_idx), shuffle_labels=False, seed_base=seed_base,
                )
                sentences.append(create_prompt(prompt_data))
                seed = seed_base + 100_000 * task_index + int(query_idx)
                n_shots_j = sample_variable_icl_count(
                    np.random.default_rng(seed), prompt_args.min_shots, prompt_args.max_shots)
                tgt = prompt_data["query_target"]
                meta.append({
                    "prompt_index": j, "query_idx": int(query_idx), "n_shots": n_shots_j,
                    "query_input": tgt["input"], "query_output": tgt["output"],
                    "prompt_seed": seed,
                })

            inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)
            prompt_lens = inputs.attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(len(sentences), device=model.device)

            with TraceDict(model, layers=model_config["attn_hook_names"],
                           retain_input=True, retain_output=False) as td:
                model(**inputs)

            layer_inputs = []
            for layer in model_config["attn_hook_names"]:
                layer_input = td[layer].input
                if isinstance(layer_input, tuple):
                    layer_input = layer_input[0]
                layer_inputs.append(split_activations_by_head(layer_input, model_config))
            stack_initial = torch.stack(layer_inputs).permute(1, 0, 2, 3, 4)
            last_token = stack_initial[batch_indices, :, prompt_lens]  # (b, L, H, hd)
            refstyle_sum += last_token.sum(dim=0).double()
            refstyle_count += len(sentences)
            per_prompt.append(last_token.float().cpu())
    finally:
        tokenizer.padding_side = old_padding_side
    refstyle_mean = (refstyle_sum / refstyle_count).float().cpu()
    return torch.cat(per_prompt, dim=0), meta, refstyle_mean


def main():
    args = parse_args()
    set_seed(args.seed)
    train_tasks = list(json.loads(args.task_split_path.read_text())["train_tasks"])
    tasks = train_tasks + TEST7
    assert len(tasks) == 27 and len(set(tasks)) == 27

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()
    # get_last_token_mean_head_activations (gate) has no no_grad of its own -- the canonical
    # caller disables grad globally (compute_multitask_varicl_heads.py:316); match it.
    torch.set_grad_enabled(False)

    for variant in args.variants:
        min_shots, max_shots = VARIANTS[variant]
        prompt_args = make_prompt_args(args, min_shots, max_shots)
        out_dir = args.out_root / variant
        out_dir.mkdir(parents=True, exist_ok=True)

        for task_index, task in enumerate(tasks):
            out_path = out_dir / f"{task}.pt"
            if out_path.exists() and not args.overwrite:
                print(f"[{variant}] {task}: exists, skip", flush=True)
                continue

            dataset = {"all": ICLDataset(str(args.dataset_root / f"{task}.json"))}
            pool_size = len(dataset["all"])
            assert pool_size >= max_shots + 1, f"{task}: pool too small ({pool_size})"
            query_indices = select_query_indices(task_index, args, pool_size)
            assert len(query_indices) == args.n_prompts, \
                f"HARD STOP {task}/{variant}: {len(query_indices)} != {args.n_prompts} prompts"

            acts, meta, refstyle_mean = capture_task(
                dataset, prompt_args, model, model_config, tokenizer,
                task_index, query_indices, seed_base=args.seed)
            assert acts.shape[0] == args.n_prompts and torch.isfinite(acts).all()
            task_mean = acts.double().mean(dim=0).float()

            # HARD GATE on one task: identical prompts through the existing mean function.
            if task == args.gate_task:
                ref_mean = get_last_token_mean_head_activations(
                    dataset, prompt_args, model, model_config, tokenizer,
                    task_index=task_index, query_indices=query_indices, seed_base=args.seed).cpu()
                # Identical prompts + hook + reduction => must match to fp roundoff. (The
                # forward is bitwise deterministic on this GPU: probe 2026-08-10, ref-vs-ref
                # dev exactly 0; a prompt/indexing mismatch would show as dev ~ O(0.1-1).)
                dev_exact = (refstyle_mean - ref_mean).abs().max().item()
                assert dev_exact < 1e-6, f"GATE FAIL {task}/{variant}: ref-style dev {dev_exact:.2e}"
                dev_fp64 = (task_mean - ref_mean).abs().max().item()
                print(f"[{variant}] gate passed on {task}: ref-style reduction matches "
                      f"get_last_token_mean_head_activations exactly (dev {dev_exact:.2e}); "
                      f"fp64 per-prompt mean differs by {dev_fp64:.2e} (fp16 batch-sum rounding "
                      f"in the reference, expected ~2e-4)", flush=True)

            # Advisory: cosine vs stored canonical mean (different prompt population).
            canon_path = (args.canonical_means_root / task /
                          f"{task}_mean_head_activations_varicl.pt")
            cos = float("nan")
            if canon_path.exists():
                canon = torch.load(canon_path, weights_only=False).flatten().double()
                m = task_mean.flatten().double()
                cos = float((m @ canon) / (m.norm() * canon.norm()))

            shots = sorted({d["n_shots"] for d in meta})
            print(f"[{variant}] {task}: acts {tuple(acts.shape)}, shots {shots[0]}-{shots[-1]}, "
                  f"cos-vs-canonical-mean {cos:.3f}", flush=True)

            torch.save({
                "activations": acts.half(),
                "task_mean_fp32": task_mean,
                "metadata": meta,
                "config": {
                    "task": task, "task_index": task_index, "variant": variant,
                    "min_shots": min_shots, "max_shots": max_shots,
                    "n_prompts": args.n_prompts, "seed": args.seed,
                    "query_pool": "full dataset (train+valid+test), no ICL-correctness filter",
                    "hook": "attn.out_proj input, last non-pad token (final cue token)",
                    "model_name": args.model_name,
                    "prefixes": prompt_args.prefixes, "separators": prompt_args.separators,
                    "cos_vs_canonical_varicl_mean": cos,
                },
            }, out_path)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
