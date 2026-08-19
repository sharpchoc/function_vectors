#!/usr/bin/env python
"""Stage 1: per-prompt model correctness on the residual-activation capture prompts.

For every task in the 29-task manifest, rebuild the *exact* 10-shot prompts that were used to
capture residual activations (same seed / sampling as
src/extract_targeted_residual_stream_activations.py), generate the model's answer greedily, cut the
answer at the newline (the repo's parse_generation regex), and exact-match it against the target.

"Correct" here is the FULL generated answer matching the target (NOT first-token rank), as requested.

Writes one JSON per task under <output_dir>/correctness/<task>.json plus a correctness_summary.json,
keyed for a robust join by (split, query_source_index). Consumed by cosine_activation_to_task_fv.py.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import create_prompt, load_dataset
from utils.eval_utils import parse_generation, exact_match_score, metric_max_over_ground_truths
# Reuse the capture's deterministic prompt construction so prompts are byte-for-byte identical.
from extract_targeted_residual_stream_activations import (
    sample_query_indices,
    sample_demo_indices,
    make_prompt,
)
from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--capture_root", type=Path,
                   default=ARTIFACTS_ROOT / "residual_activations" / "gptj_56tasks_170prompts_4tokens",
                   help="Capture dir whose index.json query_indices are cross-checked for alignment.")
    p.add_argument("--output_dir", type=Path, default=FV_FORMATION_DIR / "activation_to_fv_decoding/cosine/activation_to_task_fv")
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_split", type=float, default=0.3)
    p.add_argument("--n_shots", type=int, default=10)
    p.add_argument("--max_train_prompts", type=int, default=130)
    p.add_argument("--max_test_prompts", type=int, default=40)
    p.add_argument("--max_valid_prompts", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    p.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    p.add_argument("--tasks", nargs="+", default=None, help="Optional subset (e.g. for a quick sanity run).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sampling_args(args):
    """Namespace with the attributes the imported capture helpers read."""
    return SimpleNamespace(
        seed=args.seed, n_shots=args.n_shots,
        max_train_prompts=args.max_train_prompts, max_test_prompts=args.max_test_prompts,
        max_valid_prompts=args.max_valid_prompts,
        prefixes=args.prefixes, separators=args.separators,
    )


def captured_query_indices(capture_root, task, split):
    """The query_indices recorded by the capture for cross-checking (or None if absent)."""
    index_path = capture_root / task / split / "index.json"
    if not index_path.exists():
        return None
    cfg = json.loads(index_path.read_text())["config"]
    return [int(x) for x in cfg.get("query_indices", [])]


@torch.no_grad()
def generate_answer(prompt_string, model, tokenizer, max_new_tokens):
    inputs = tokenizer(prompt_string, return_tensors="pt").to(model.device)
    out = model.generate(inputs.input_ids, do_sample=False, max_new_tokens=max_new_tokens,
                         pad_token_id=tokenizer.eos_token_id)
    new_tokens = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens)


def score_task(task, model, tokenizer, model_config, args, sargs):
    dataset = load_dataset(task, root_data_dir=args.root_data_dir, test_size=args.test_split, seed=args.seed)
    records = []
    n_correct = 0
    for split in args.splits:
        query_indices = sample_query_indices(task, split, len(dataset[split]), sargs)
        captured = captured_query_indices(args.capture_root, task, split)
        if captured is not None and captured != [int(x) for x in query_indices]:
            raise AssertionError(
                f"{task}/{split}: regenerated query_indices != captured ({len(query_indices)} vs "
                f"{len(captured)}); prompts would not match the activation capture.")
        for prompt_index, query_idx in enumerate(query_indices):
            demo_indices = sample_demo_indices(task, split, int(query_idx), dataset, sargs)
            prompt_data = make_prompt(dataset, split, int(query_idx), demo_indices,
                                      model_config, args.prefixes, args.separators)
            prompt_string = create_prompt(prompt_data)
            target = prompt_data["query_target"]["output"].strip()
            generated = generate_answer(prompt_string, model, tokenizer, args.max_new_tokens)
            parsed, score = parse_generation(generated, [target], exact_match_score)
            correct = bool(score == 1.0)
            n_correct += int(correct)
            records.append({
                "split": split, "prompt_index": prompt_index,
                "query_source_index": int(query_idx),
                "query_input": prompt_data["query_target"]["input"].strip(),
                "target": target, "generated": generated, "parsed": parsed.strip(),
                "correct": correct,
            })
    accuracy = n_correct / len(records) if records else float("nan")
    return records, accuracy


def main():
    args = parse_args()
    manifest = json.loads(args.task_manifest.read_text())
    tasks = list(manifest["train_tasks"]) + list(manifest["test_tasks"])
    if args.tasks is not None:
        tasks = [t for t in tasks if t in set(args.tasks)]

    out_dir = args.output_dir / "correctness"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    set_seed(args.seed)
    print("Loading model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision)
    model.eval()

    sargs = sampling_args(args)
    summary = {"model_name": args.model_name, "seed": args.seed, "n_shots": args.n_shots,
               "metric": "exact_match (newline-cutoff, full answer)", "per_task_accuracy": {}}
    for task in tasks:
        out_path = out_dir / f"{task}.json"
        if out_path.exists() and not args.overwrite:
            print(f"  {task}: exists, skipping (use --overwrite)")
            summary["per_task_accuracy"][task] = json.loads(out_path.read_text()).get("accuracy")
            continue
        records, accuracy = score_task(task, model, tokenizer, model_config, args, sargs)
        out_path.write_text(json.dumps({
            "task": task, "n_prompts": len(records), "n_correct": sum(r["correct"] for r in records),
            "accuracy": accuracy, "records": records,
        }, indent=2))
        summary["per_task_accuracy"][task] = accuracy
        print(f"  {task:26s} acc={accuracy:.3f} ({sum(r['correct'] for r in records)}/{len(records)})")

    (args.output_dir / "correctness_summary.json").write_text(json.dumps(summary, indent=2))
    accs = [a for a in summary["per_task_accuracy"].values() if a is not None]
    print(f"\nWrote correctness for {len(tasks)} tasks. Mean accuracy {sum(accs)/len(accs):.3f}")
    print(args.output_dir / "correctness_summary.json")


if __name__ == "__main__":
    main()
