#!/usr/bin/env python
"""SANDBOX (not repo standard): steering effectiveness of PER-PROMPT FVs (matched-query arm).

Same protocol as the held-out steering comparison family (evaluate_heldout_multitask_head_fvs /
evaluate_heldout_varicl_fv): GPT-J, seed 42, filter set = clean-ICL-correct test queries, FV
added to the residual output of each edit layer at the last token, zero-shot + 10-shot-shuffled
conditions, intervention top-1 across all 28 layers. ONE change: instead of a single task-level
FV, each eval query j is steered by ITS OWN per-prompt FV (from capture_eval_query_fvs.py: a
10-shot clean-ICL prompt with that example as final query).

`n_shot_eval_perquery` is a line-for-line copy of eval_utils.n_shot_eval's single-token path
with `fv_vector` swapped per query; RNG consumption is identical, so prompts match the cached
baseline arms exactly.

GATE MODE (--gate_constant_varicl): runs the modified loop with a CONSTANT bank (every query ->
the task's train_varicl_top40 FV) and compares the per-layer top-1 curves to the cached
varicl_top40 curves from heldout_varicl_nheads_sweep/<task>/nheads_sweep_by_layer.json["40"].
Tolerance (user decision 2026-07-26 after adjudicating the initial exact-match failure): each
cell may differ by at most ONE flipped query (cross-stack fp tie-breaks vs the mid-June cached
run; determinism on-stack verified), and at most --gate_max_flip_cells cells may differ at all.
Anything larger = hard stop, user adjudicates.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
EVAL_SCRIPTS_DIR = SRC_ROOT / "eval_scripts"
for p in (str(REPO_ROOT), str(SRC_ROOT), str(EVAL_SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluate_heldout_multitask_head_fvs import (  # noqa: E402
    get_filter_set,
    load_function_vector,
    summarize_results,
    torch_load_trusted,
    write_json,
)
from utils.eval_utils import (  # noqa: E402
    compute_individual_token_rank,
    compute_top_k_accuracy,
    function_vector_intervention,
    get_answer_id,
)
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed  # noqa: E402
from utils.paths import ARTIFACTS_ROOT, RESULTS_ROOT, STEERING_COMPARISON_DIR  # noqa: E402
from utils.prompt_utils import create_prompt, load_dataset, word_pairs_to_prompt_data  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="SANDBOX: matched per-prompt FV steering eval.")
    p.add_argument("--task_split_path", type=Path, default=REPO_ROOT / "task_splits/abstractive_train_test_tasks_29.json")
    p.add_argument("--task_split_key", type=str, default="test_tasks")
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--fv_bank_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox/perprompt_head_acts/gptj_train_varicl_top40_evalqueries")
    p.add_argument("--filter_fv_root", type=Path, default=ARTIFACTS_ROOT / "gptj_fv")
    p.add_argument("--output_root", type=Path,
                   default=RESULTS_ROOT / "sandbox/perprompt_ridge_pilot/steering_eval")
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
    p.add_argument("--filter_to_correct_icl", dest="filter_to_correct_icl", action="store_true")
    p.add_argument("--no_filter_to_correct_icl", dest="filter_to_correct_icl", action="store_false")
    p.set_defaults(filter_to_correct_icl=True)
    # Gate mode: constant varicl_top40 bank, compare to cached nheads-sweep curves.
    p.add_argument("--gate_constant_varicl", action="store_true")
    p.add_argument("--gate_fv_root", type=Path, default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_varicl_top40")
    p.add_argument("--gate_reference_root", type=Path,
                   default=STEERING_COMPARISON_DIR / "heldout_varicl_nheads_sweep")
    p.add_argument("--gate_max_flip_cells", type=int, default=6,
                   help="Max cells (of 56) allowed to differ from the cached curves, each by "
                        "at most one flipped query (cross-stack fp tolerance, user-approved).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def n_shot_eval_perquery(dataset, fv_bank, edit_layer, n_shots, model, model_config, tokenizer,
                         shuffle_labels=False, filter_set=None, prefixes=None, separators=None):
    """eval_utils.n_shot_eval (single-token path), with fv_vector = fv_bank[j] per query.

    RNG consumption is identical to the original (same np.random calls in the same order),
    so prompts are bit-identical to the family's other arms.
    """
    clean_rank_list = []
    intervention_rank_list = []
    prepend_bos = False if model_config["prepend_bos"] else True

    if filter_set is None:
        filter_set = np.arange(len(dataset["test"]))

    for j in tqdm(range(len(dataset["test"])), total=len(dataset["test"])):
        if j not in filter_set:
            continue
        if n_shots == 0:
            word_pairs = {"input": [], "output": []}
        else:
            word_pairs = dataset["train"][np.random.choice(len(dataset["train"]), n_shots, replace=False)]
        word_pairs_test = dataset["test"][j]

        if prefixes is not None and separators is not None:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test,
                                                    prepend_bos_token=prepend_bos,
                                                    shuffle_labels=shuffle_labels,
                                                    prefixes=prefixes, separators=separators)
        else:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test,
                                                    prepend_bos_token=prepend_bos,
                                                    shuffle_labels=shuffle_labels)

        query, target = prompt_data["query_target"]["input"], prompt_data["query_target"]["output"]
        query = query[0] if isinstance(query, list) else query
        target = target[0] if isinstance(target, list) else target

        sentence = [create_prompt(prompt_data)]
        target_token_id = get_answer_id(sentence[0], target, tokenizer)

        fv_vector = fv_bank[int(j)]  # <-- the only change vs n_shot_eval

        clean_output, intervention_output = function_vector_intervention(
            sentence, target=[target], edit_layer=edit_layer, function_vector=fv_vector,
            model=model, model_config=model_config, tokenizer=tokenizer, compute_nll=False)

        clean_rank_list.append(compute_individual_token_rank(clean_output, target_token_id))
        intervention_rank_list.append(compute_individual_token_rank(intervention_output, target_token_id))

    return {
        "clean_topk": [(K, compute_top_k_accuracy(clean_rank_list, K)) for K in range(1, 4)],
        "clean_rank_list": clean_rank_list,
        "intervention_topk": [(K, compute_top_k_accuracy(intervention_rank_list, K)) for K in range(1, 4)],
        "intervention_rank_list": intervention_rank_list,
    }


def evaluate_bank(args, dataset, fv_bank, model, model_config, tokenizer, filter_set):
    layers = range(model_config["n_layers"]) if args.edit_layer == -1 else [args.edit_layer]
    zs_results, fs_shuffled_results = {}, {}
    for layer in layers:
        set_seed(args.seed)
        zs_results[int(layer)] = n_shot_eval_perquery(
            dataset=dataset, fv_bank=fv_bank, edit_layer=int(layer), n_shots=0,
            model=model, model_config=model_config, tokenizer=tokenizer,
            filter_set=filter_set, prefixes=args.prefixes, separators=args.separators)
        set_seed(args.seed)
        fs_shuffled_results[int(layer)] = n_shot_eval_perquery(
            dataset=dataset, fv_bank=fv_bank, edit_layer=int(layer), n_shots=args.n_shots,
            model=model, model_config=model_config, tokenizer=tokenizer,
            filter_set=filter_set, shuffle_labels=True, prefixes=args.prefixes, separators=args.separators)
    return zs_results, fs_shuffled_results


def run_gate(args, task, dataset, model, model_config, tokenizer, filter_set, task_dir):
    """Constant-bank run must reproduce the cached varicl_top40 per-layer top-1 curves, up to
    isolated single-query fp flips (tolerance user-approved 2026-07-26)."""
    fv, _ = load_function_vector(args.gate_fv_root / task / f"{task}_function_vector.pt")
    bank = {int(j): fv for j in filter_set}
    zs, fs = evaluate_bank(args, dataset, bank, model, model_config, tokenizer, filter_set)
    got = summarize_results(zs, fs)
    ref = json.loads((args.gate_reference_root / task / "nheads_sweep_by_layer.json").read_text())["40"]
    one_flip = 1.0 / len(filter_set) + 1e-9
    report = {"task": task, "n_queries": int(len(filter_set)), "checked": 0,
              "exact_cells": 0, "flip_cells": [], "max_abs_diff": 0.0}
    for key in ("zs_intervention_top1_by_layer", "fs_shuffled_intervention_top1_by_layer"):
        for layer, ref_val in ref[key].items():
            diff = abs(float(got[key][layer]) - float(ref_val))
            report["checked"] += 1
            report["max_abs_diff"] = max(report["max_abs_diff"], diff)
            if diff <= 1e-12:
                report["exact_cells"] += 1
            elif diff <= one_flip:
                report["flip_cells"].append(f"{key}/L{layer}")
            else:
                raise RuntimeError(
                    f"LOOP GATE FAILED for {task} {key} L{layer}: got {got[key][layer]} vs cached "
                    f"{ref_val} (diff {diff:.4f} > one query = {one_flip:.4f}). "
                    f"STOP -- user adjudicates.")
    if len(report["flip_cells"]) > args.gate_max_flip_cells:
        raise RuntimeError(
            f"LOOP GATE FAILED for {task}: {len(report['flip_cells'])} cells differ "
            f"(> {args.gate_max_flip_cells} allowed): {report['flip_cells']}. STOP -- user adjudicates.")
    write_json(task_dir / "loop_gate_report.json", {**report, "reference": str(args.gate_reference_root),
                                                    "gate_max_flip_cells": args.gate_max_flip_cells})
    print(f"[GATE] {task}: PASSED -- {report['exact_cells']}/{report['checked']} cells exact, "
          f"{len(report['flip_cells'])} single-query flips {report['flip_cells']}")


def main():
    args = parse_args()
    if args.tasks is not None:
        tasks = list(args.tasks)
    else:
        tasks = json.loads(args.task_split_path.read_text())[args.task_split_key]

    torch.set_grad_enabled(False)
    print("Loading model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device,
                                                                  revision=args.revision)
    model.eval()

    for task in tasks:
        print(f"\n=== {task} ===")
        task_dir = args.output_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        out_path = task_dir / "perprompt_summary.json"
        if out_path.exists() and not args.overwrite and not args.gate_constant_varicl:
            raise FileExistsError(f"{out_path} exists. Pass --overwrite to replace it.")

        set_seed(args.seed)
        dataset = load_dataset(task, root_data_dir=args.root_data_dir, test_size=args.test_split, seed=args.seed)
        filter_args = SimpleNamespace(
            filter_to_correct_icl=args.filter_to_correct_icl, generate_str=False,
            fv_root=args.filter_fv_root, seed=args.seed, n_shots=args.n_shots, metric="f1_score",
            prefixes=args.prefixes, separators=args.separators, batch_size_baseline=args.batch_size_baseline,
        )
        filter_set, filter_source = get_filter_set(filter_args, task, dataset, model, model_config,
                                                   tokenizer, task_dir)
        print(f"  filter set: {len(filter_set)} examples ({filter_source})")

        if args.gate_constant_varicl:
            run_gate(args, task, dataset, model, model_config, tokenizer, filter_set, task_dir)
            continue

        bank_data = torch_load_trusted(args.fv_bank_root / task / "fv_bank.pt", map_location="cpu")
        fv_bank = {int(k): v.detach().float() for k, v in bank_data["fvs"].items()}
        missing = [int(j) for j in filter_set if int(j) not in fv_bank]
        if missing:
            raise KeyError(f"{task}: {len(missing)} filter-set queries missing from fv_bank "
                           f"(first: {missing[:5]}). Re-run capture_eval_query_fvs.py.")

        zs, fs = evaluate_bank(args, dataset, fv_bank, model, model_config, tokenizer, filter_set)
        write_json(task_dir / "perprompt_zs_results.json", zs)
        write_json(task_dir / "perprompt_fs_shuffled_results.json", fs)
        summary = summarize_results(zs, fs)
        write_json(out_path, {
            "sandbox": True,
            "task": task,
            "arm": "perprompt_matched_query_fv",
            "fv_bank_root": str(args.fv_bank_root),
            "filter_source": filter_source,
            "n_filtered_test_examples": int(len(filter_set)),
            "perprompt": summary,
        })
        print(f"  wrote {out_path} | best zs L{summary['best_zs_layer']}="
              f"{summary['best_zs_intervention_top1']:.3f} | best fs L{summary['best_fs_shuffled_layer']}="
              f"{summary['best_fs_shuffled_intervention_top1']:.3f}")


if __name__ == "__main__":
    main()
