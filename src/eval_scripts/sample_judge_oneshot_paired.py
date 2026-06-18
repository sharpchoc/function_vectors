#!/usr/bin/env python
"""Sample-based correctness for the paired 1-shot capture: n temperature samples per prompt,
GPT-4-judged, store the FRACTION judged correct (k/n).

For each prompt in results/oneshot_paired_graded/<pair>/grading.json, draw --n_samples completions
at --temperature (single model load, via num_return_sequences), judge each with the strict per-task
GPT-4 prompt (reused from judge_oneshot_paired.JUDGE_SYSTEMS), and record k/n. Stamps onto
grading.json AND every matching activation row (both source and target roles) in shard_*.pt:
  - `frac_correct_temp<T>_n<N>` (float in [0,1])
  - `n_correct_temp<T>_n<N>`    (int 0..N)
Match key = (function_task, output_word, query) — unique per prompt. In-place rewrite. Writes the
full per-sample generations + verdicts to results/oneshot_<task>_judge_sample<N>/judged_results.json.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from eval_scripts.judge_oneshot_paired import (
    JUDGE_SYSTEMS, get_openai_key, extract_answer, build_prompt, judge,
)


def parse_args():
    p = argparse.ArgumentParser(description="n-sample GPT-4-judged fraction-correct for paired 1-shot capture.")
    p.add_argument("--graded_dir", type=Path, default=Path("results/oneshot_paired_graded/antonym_synonym"))
    p.add_argument("--function_tasks", nargs="+", default=["antonym", "synonym"])
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--judge_model", type=str, default="gpt-4.1")
    p.add_argument("--judge_batch_size", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    p.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    p.add_argument("--output_root", type=Path, default=Path("results"))
    return p.parse_args()


def sample_answers(model, tokenizer, args, rows, task):
    """Return a flat list of per-sample records (n_samples per prompt)."""
    multiword = task in ("next_number", "prev_number")
    out = []
    for j, r in enumerate(rows):
        prompt = build_prompt(r["demo_input"], r["output_word"], r["query"], args)
        inp = tokenizer(prompt, return_tensors="pt").to(args.device)
        gen = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=True,
                             temperature=args.temperature, num_return_sequences=args.n_samples,
                             pad_token_id=tokenizer.eos_token_id)
        for s in range(args.n_samples):
            completion = tokenizer.decode(gen[s][inp["input_ids"].shape[1]:], skip_special_tokens=True)
            out.append({"function_task": task, "query_input": r["query"], "output_word": r["output_word"],
                        "demo_input": r["demo_input"], "gold_output": r["gold"], "sample_idx": s,
                        "generated": extract_answer(completion, multiword=multiword),
                        "raw_completion": completion})
        if (j + 1) % 100 == 0:
            print(f"    {task}: sampled {j+1}/{len(rows)} prompts x{args.n_samples}")
    return out


def main():
    args = parse_args()
    suffix = f"temp{args.temperature:g}_n{args.n_samples}"
    frac_field, cnt_field = f"frac_correct_{suffix}", f"n_correct_{suffix}"

    grading_path = args.graded_dir / "grading.json"
    grading = json.loads(grading_path.read_text())
    api_key = get_openai_key()

    print("Loading model...")
    torch.set_grad_enabled(False)
    model, tokenizer, _cfg = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    model.eval()
    set_seed(args.seed)

    # key -> (n_correct, n_samples)
    agg = {}
    overall = {}
    for task in args.function_tasks:
        rows = [g for g in grading if g["function_task"] == task]
        print(f"\n[{task}] {len(rows)} prompts x {args.n_samples} samples")
        records = sample_answers(model, tokenizer, args, rows, task)
        verdicts = judge(records, JUDGE_SYSTEMS[task], args.judge_model, api_key, args.judge_batch_size)
        for rec, v in zip(records, verdicts):
            rec["judge_correct"] = bool(v["correct"])
            key = (task, rec["output_word"], rec["query_input"])
            nc, ns = agg.get(key, (0, 0))
            agg[key] = (nc + int(rec["judge_correct"]), ns + 1)

        task_keys = [(task, r["output_word"], r["query"]) for r in rows]
        fracs = [agg[k][0] / agg[k][1] for k in task_keys if k in agg]
        summary = {"function_task": task, "n_prompts": len(rows), "n_samples": args.n_samples,
                   "temperature": args.temperature, "judge_model": args.judge_model, "seed": args.seed,
                   "mean_frac_correct": sum(fracs) / len(fracs) if fracs else 0.0,
                   "n_prompts_any_correct": sum(f > 0 for f in fracs),
                   "n_prompts_all_correct": sum(f == 1.0 for f in fracs)}
        out_dir = args.output_root / f"oneshot_{task}_judge_sample{args.n_samples}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "judged_results.json").write_text(json.dumps({"summary": summary, "records": records}, indent=2))
        overall[task] = summary
        print(f"[{task}] mean fraction-correct = {summary['mean_frac_correct']:.3f}  "
              f"(any-correct {summary['n_prompts_any_correct']}/{len(rows)}, "
              f"all-correct {summary['n_prompts_all_correct']}/{len(rows)})")

    # stamp grading.json
    g_tagged = 0
    for g in grading:
        k = (g["function_task"], g["output_word"], g["query"])
        if k in agg:
            nc, ns = agg[k]
            g[cnt_field] = nc
            g[frac_field] = nc / ns
            g_tagged += 1
    grading_path.write_text(json.dumps(grading, indent=2))
    print(f"\ntagged {g_tagged}/{len(grading)} grading.json rows with {frac_field}+{cnt_field}")

    # stamp shards
    n_rows = n_tagged = 0
    for sp in sorted(glob.glob(str(args.graded_dir / "shard_*.pt"))):
        data = torch.load(sp, map_location="cpu", weights_only=False)
        for m in data["metadata"]:
            n_rows += 1
            k = (m["function_task"], m["output_word"], m["query_word"])
            if k in agg:
                nc, ns = agg[k]
                m[cnt_field] = nc
                m[frac_field] = nc / ns
                n_tagged += 1
        torch.save(data, sp)
    print(f"tagged {n_tagged}/{n_rows} activation rows with {frac_field}+{cnt_field}")

    print("\n=== SUMMARY ===")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
