#!/usr/bin/env python
"""GPT-4-judged top-1 accuracy for the paired TWO-shot capture.

Two-demo sibling of judge_oneshot_paired.py. For each prompt in
artifacts/twoshot_paired_graded/<pair>/grading.json we (1) rebuild the EXACT 2-shot prompt
(two matched-label demos + shared query), (2) greedy-generate GPT-J's top-1 answer, and (3) ask
an OpenAI judge whether the answer is correct for the task. Reuses the judge systems / parsing /
key handling from judge_oneshot_paired.py. For digit number tasks the judge system is the base
number judge (next_number / prev_number); the answer parser keeps digits.

OPENAI_API_KEY read from env, falling back to /proc/1/environ.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.prompt_utils import word_pairs_to_prompt_data, create_prompt
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.paths import ARTIFACTS_ROOT, LABEL_GEOMETRY_DIR

from judge_oneshot_paired import JUDGE_SYSTEMS, get_openai_key, extract_answer, judge


def parse_args():
    p = argparse.ArgumentParser(description="GPT-4-judged top-1 accuracy for paired TWO-shot capture.")
    p.add_argument("--graded_dir", type=Path, default=ARTIFACTS_ROOT / "twoshot_paired_graded" / "antonym_synonym")
    p.add_argument("--function_tasks", nargs="+", default=["antonym", "synonym"])
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--gen_batch_size", type=int, default=16)
    p.add_argument("--judge_model", type=str, default="gpt-4.1")
    p.add_argument("--judge_batch_size", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    p.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    p.add_argument("--output_root", type=Path, default=LABEL_GEOMETRY_DIR / "twoshot" / "judge")
    return p.parse_args()


def judge_task_name(task):
    """next_number_digits -> next_number (base number judge); others unchanged."""
    return task.replace("_digits", "")


def build_prompt(demo_inputs, labels, query_input, args):
    pd = word_pairs_to_prompt_data(
        {"input": list(demo_inputs), "output": list(labels)},
        query_target_pair={"input": query_input, "output": query_input},
        prepend_bos_token=False, shuffle_labels=False,
        prefixes=args.prefixes, separators=args.separators, prepend_space=True,
    )
    return create_prompt(pd)


def generate_answers(model, tokenizer, args, rows, task):
    multiword = judge_task_name(task) in ("next_number", "prev_number")
    prompts = [build_prompt(r["demo_inputs"], [r["label1"], r["label2"]], r["query"], args) for r in rows]
    tokenizer.padding_side = "left"
    out = []
    for i in range(0, len(rows), args.gen_batch_size):
        chunk_p = prompts[i:i + args.gen_batch_size]
        chunk_r = rows[i:i + args.gen_batch_size]
        enc = tokenizer(chunk_p, return_tensors="pt", padding=True).to(args.device)
        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        cont = gen[:, enc["input_ids"].shape[1]:]
        for k, r in enumerate(chunk_r):
            completion = tokenizer.decode(cont[k], skip_special_tokens=True)
            out.append({"function_task": task, "query_input": r["query"],
                        "label1": r["label1"], "label2": r["label2"],
                        "demo_inputs": r["demo_inputs"], "gold_output": r["gold"],
                        "first_tok_rank": r["gold_first_tok_rank"],
                        "generated": extract_answer(completion, multiword=multiword),
                        "raw_completion": completion})
        if (i + len(chunk_r)) % 96 == 0 or i + len(chunk_r) == len(rows):
            print(f"    {task}: generated {i+len(chunk_r)}/{len(rows)}")
    return out


def main():
    args = parse_args()
    grading = json.loads((args.graded_dir / "grading.json").read_text())
    api_key = get_openai_key()

    print("Loading model...")
    torch.set_grad_enabled(False)
    model, tokenizer, _cfg = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    set_seed(args.seed)

    overall = {}
    for task in args.function_tasks:
        rows = [g for g in grading if g["function_task"] == task]
        print(f"\n[{task}] {len(rows)} prompts")
        records = generate_answers(model, tokenizer, args, rows, task)
        verdicts = judge(records, JUDGE_SYSTEMS[judge_task_name(task)], args.judge_model, api_key, args.judge_batch_size)
        for r, v in zip(records, verdicts):
            r["judge_correct"] = bool(v["correct"])
            r["copied_input"] = r["generated"].strip().lower() == r["query_input"].strip().lower()
            r["exact_match_gold"] = r["generated"].strip().lower() == str(r["gold_output"]).strip().lower()

        n = len(records)
        judged = sum(r["judge_correct"] for r in records)
        copied = sum(r["copied_input"] for r in records)
        ft1 = sum(r["first_tok_rank"] < 1 for r in records)
        summary = {"function_task": task, "n": n, "judge_model": args.judge_model, "n_shots": 2,
                   "seed": args.seed,
                   "judge_top1_accuracy": judged / n, "first_token_top1_accuracy": ft1 / n,
                   "gold_exact_match_accuracy": sum(r["exact_match_gold"] for r in records) / n,
                   "copied_input_count": copied}
        out_dir = args.output_root / task
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "judged_results.json").write_text(json.dumps({"summary": summary, "records": records}, indent=2))
        overall[task] = summary
        print(f"[{task}] GPT-4 judge top-1 = {judged}/{n} = {judged/n:.3f}  "
              f"(first-token top-1 {ft1}/{n} = {ft1/n:.3f}; copied {copied})")

    print("\n=== SUMMARY ===")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
