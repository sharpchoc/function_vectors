#!/usr/bin/env python
"""SANDBOX (not repo standard): per-prompt FVs for the steering-eval query sets.

For each held-out test task, this captures one per-prompt FV per FILTERED steering-eval query
(filter set = clean_rank_list==0 from artifacts/gptj_fv/<task>/fs_results_layer_sweep.json,
indices into dataset['test']): a 10-shot CLEAN-ICL prompt is built with that example as the
final query (demos via the established stable_rng(seed, task, "test", query_idx,
"demo_indices") convention, exactly the sandbox capture's prompts), and the top-40
varicl_top40 head-sum vector at the final prompt token is recorded.

Output: <output_root>/<task>/fv_bank.pt = {"fvs": {test_index: fp32 [4096]}, "config": ...}.

GATE: query indices that overlap the original sandbox capture's test split must reproduce the
stored per-prompt targets (identical prompts by construction) to fp16-forward tolerance;
mismatch is a hard stop (user adjudicates).
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
    selected_token_records,
)
from sandbox.perprompt_fv.capture_perprompt_head_activations import (  # noqa: E402
    get_out_projs,
    headsum_target,
    load_json,
    torch_load_trusted,
)
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed  # noqa: E402
from utils.paths import ARTIFACTS_ROOT  # noqa: E402
from utils.prompt_utils import get_token_meta_labels, load_dataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="SANDBOX: capture per-query FVs for the steering-eval filter sets.")
    p.add_argument("--task_split_path", type=Path, default=REPO_ROOT / "task_splits/abstractive_train_test_tasks_29.json")
    p.add_argument("--task_split_key", type=str, default="test_tasks")
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--root_data_dir", type=str, default=str(REPO_ROOT / "dataset_files"))
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_varicl_top40")
    p.add_argument("--filter_fv_root", type=Path, default=ARTIFACTS_ROOT / "gptj_fv",
                   help="Where each task's fs_results_layer_sweep.json (filter source) lives.")
    p.add_argument("--reference_capture_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox/perprompt_head_acts/gptj_train_varicl_top40",
                   help="Original sandbox capture; overlapping test queries must reproduce its targets.")
    p.add_argument("--output_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox/perprompt_head_acts/gptj_train_varicl_top40_evalqueries")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_split", type=float, default=0.3)
    p.add_argument("--n_shots", type=int, default=10)
    p.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    p.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--gate_min_cos", type=float, default=0.9999)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_filter_set(filter_fv_root, task):
    """The steering eval's filter set: dataset['test'] indices with clean 10-shot rank 0."""
    path = filter_fv_root / task / "fs_results_layer_sweep.json"
    fs_results = load_json(path)
    return np.where(np.array(fs_results["clean_rank_list"]) == 0)[0].tolist(), str(path)


def load_reference_targets(reference_capture_root, task):
    """query_source_index -> stored per-prompt target, from the original capture's test split."""
    split_dir = reference_capture_root / task / "test"
    index = load_json(split_dir / "index.json")
    out = {}
    for shard in index["shards"]:
        data = torch_load_trusted(split_dir / Path(shard).name, map_location="cpu")
        for i, meta in enumerate(data["metadata"]):
            out[int(meta["query_source_index"])] = data["targets"][i].float()
    return out


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_grad_enabled(False)

    if args.tasks is not None:
        tasks = list(args.tasks)
    else:
        tasks = json.loads(args.task_split_path.read_text())[args.task_split_key]

    heads_data = torch_load_trusted(args.fv_root / "heads.pt", map_location="cpu")
    top_heads = heads_data["top_heads"]
    print(f"[evalquery-capture] {len(tasks)} tasks | {len(top_heads)} top heads")

    args.output_root.mkdir(parents=True, exist_ok=True)

    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device,
                                                                  revision=args.revision)
    model.eval()
    out_projs = get_out_projs(model, model_config)
    n_heads = model_config["n_heads"]
    resid_dim = model_config["resid_dim"]
    head_dim = resid_dim // n_heads
    attn_hooks = model_config["attn_hook_names"]

    summary = []
    for task in tasks:
        out_path = args.output_root / task / "fv_bank.pt"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"{out_path} exists; pass --overwrite to replace.")
        dataset = load_dataset(task, root_data_dir=args.root_data_dir, test_size=args.test_split, seed=args.seed)
        filter_set, filter_source = load_filter_set(args.filter_fv_root, task)
        reference = load_reference_targets(args.reference_capture_root, task)
        overlap = [q for q in filter_set if q in reference]
        print(f"[{task}] filter set {len(filter_set)} queries | overlap with reference capture {len(overlap)}")

        fvs = {}
        gate_worst_cos, gate_worst_reldiff = 1.0, 0.0
        for query_idx in tqdm(filter_set, desc=task):
            demo_indices = sample_demo_indices(task, "test", int(query_idx), dataset, args)
            prompt_data = make_prompt(dataset, "test", int(query_idx), demo_indices, model_config,
                                      args.prefixes, args.separators)
            query = prompt_data["query_target"]["input"]
            token_labels, prompt_string = get_token_meta_labels(
                prompt_data, tokenizer, query=query, prepend_bos=model_config["prepend_bos"]
            )
            (final_record,) = selected_token_records(token_labels, args.n_shots, ["last_prompt_token"])
            token_position = final_record["token_position"]

            inputs = tokenizer([prompt_string], return_tensors="pt").to(model.device)
            if token_position >= inputs["input_ids"].shape[1]:
                raise IndexError(f"{task} query {query_idx}: token position {token_position} out of range")
            with TraceDict(model, layers=attn_hooks, retain_input=True, retain_output=False) as td:
                model(**inputs)
            head_acts = torch.stack(
                [td[hook].input[0, token_position, :].view(n_heads, head_dim) for hook in attn_hooks],
                dim=0,
            )
            fv = headsum_target(head_acts, top_heads, out_projs, resid_dim, head_dim).cpu()
            fvs[int(query_idx)] = fv

            if int(query_idx) in reference:
                ref = reference[int(query_idx)]
                cos = torch.nn.functional.cosine_similarity(fv, ref, dim=0).item()
                reldiff = (fv - ref).norm().item() / max(ref.norm().item(), 1e-12)
                gate_worst_cos = min(gate_worst_cos, cos)
                gate_worst_reldiff = max(gate_worst_reldiff, reldiff)

        # ---- GATE: overlapping queries must reproduce the stored capture targets ----
        if overlap and gate_worst_cos < args.gate_min_cos:
            raise RuntimeError(
                f"REPRO GATE FAILED for {task}: worst cos vs reference capture = {gate_worst_cos:.6f} "
                f"(worst rel L2 {gate_worst_reldiff:.2e}) over {len(overlap)} overlap queries. "
                f"STOP -- user adjudicates."
            )
        print(f"[{task}] GATE: {len(overlap)} overlap queries, worst cos={gate_worst_cos:.6f} "
              f"worst rel_l2={gate_worst_reldiff:.2e}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "sandbox": True,
            "fvs": fvs,
            "config": {
                "task": task, "split": "test", "model_name": args.model_name,
                "seed": args.seed, "test_split": args.test_split, "n_shots": args.n_shots,
                "prefixes": args.prefixes, "separators": args.separators,
                "fv_root": str(args.fv_root), "filter_source": filter_source,
                "top_heads": [[int(l), int(h)] for l, h, *_ in top_heads],
                "demo_convention": "stable_rng(seed, task, 'test', query_idx, 'demo_indices')",
                "n_queries": len(fvs),
                "gate": {"n_overlap": len(overlap), "worst_cos": gate_worst_cos,
                         "worst_rel_l2": gate_worst_reldiff},
            },
        }, out_path)
        summary.append({"task": task, "n_queries": len(fvs), "n_overlap": len(overlap),
                        "gate_worst_cos": gate_worst_cos, "gate_worst_rel_l2": gate_worst_reldiff})
        print(f"[{task}] wrote {out_path} ({len(fvs)} FVs)")

    with open(args.output_root / "capture_summary.json", "w") as f:
        json.dump({"sandbox": True, "tasks": summary, "fv_root": str(args.fv_root)}, f, indent=2)
    print(f"[evalquery-capture] DONE: {len(summary)} tasks")


if __name__ == "__main__":
    main()
