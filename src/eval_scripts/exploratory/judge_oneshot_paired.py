#!/usr/bin/env python
"""GPT-4-judged top-1 accuracy for the paired 1-shot capture (antonym AND synonym).

First-token exact-match undercounts open-ended tasks (many valid antonyms/synonyms). For each
task we (1) rebuild the EXACT prompts from the corrected capture
(artifacts/oneshot_paired_graded/<pair>/grading.json — shared-input query, shared-output demo
label), (2) greedy-generate GPT-J's top-1 answer word, and (3) ask an OpenAI judge whether the
answer is a valid antonym / synonym of the query. The SAME WORD (or a cap/plural/inflectional
variant) is explicitly NOT a correct response.

Loads the model once and runs every requested task. OPENAI_API_KEY read from env, falling back
to /proc/1/environ. Supersedes judge_synonym_oneshot.py.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import requests
from utils.prompt_utils import word_pairs_to_prompt_data, create_prompt
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.paths import ARTIFACTS_ROOT, LABEL_GEOMETRY_DIR


JUDGE_SYSTEMS = {
    "synonym": (
        "You are a strict synonym judge. For each (input_word, answer_word) pair decide whether "
        "answer_word is a valid synonym of input_word in standard English: the two words must share "
        "essentially the same meaning and be interchangeable in some common context. "
        "IMPORTANT: the SAME WORD does NOT count as a correct response — if answer_word is identical "
        "to input_word, or is merely a capitalization, plural, or inflectional variant of it (e.g. "
        "happy->happy, happy->Happy, cat->cats, run->running), it MUST be judged false. Also FALSE "
        "if answer_word is a non-word, is an antonym (opposite meaning), or is merely topically "
        "related/associated but not synonymous (e.g. doctor->hospital, hot->sun). A genuine close "
        "near-synonym counts as true. Respond ONLY with a JSON array, one object per pair, in order: "
        '{"input": ..., "answer": ..., "correct": true|false}.'
    ),
    "antonym": (
        "You are a strict antonym judge. For each (input_word, answer_word) pair decide whether "
        "answer_word is a valid antonym (opposite meaning) of input_word in standard English. "
        "IMPORTANT: the SAME WORD does NOT count as a correct response — if answer_word is identical "
        "to input_word, or is merely a capitalization, plural, or inflectional variant of it, it MUST "
        "be judged false. Also FALSE if answer_word is a synonym or near-synonym (same meaning), is a "
        "non-word, or is merely topically related/associated but not opposite in meaning. A genuine "
        "near-opposite counts as true. Respond ONLY with a JSON array, one object per pair, in order: "
        '{"input": ..., "answer": ..., "correct": true|false}.'
    ),
    "next_number": (
        "You are a strict next-number judge. The input is a number (written as a word or words). The "
        "correct answer is the number immediately AFTER it (input + 1). For each (input, answer) pair "
        "decide whether answer denotes exactly input+1, in ANY standard notation (word form or digits, "
        "with or without 'and', e.g. 'one hundred', '100', 'a hundred' all = 100). IMPORTANT: the SAME "
        "NUMBER as the input does NOT count. FALSE if answer is any other number, equals the input, or "
        "is not a number. Respond ONLY with a JSON array, one object per pair, in order: "
        '{"input": ..., "answer": ..., "correct": true|false}.'
    ),
    "prev_number": (
        "You are a strict previous-number judge. The input is a number (written as a word or words). "
        "The correct answer is the number immediately BEFORE it (input - 1). For each (input, answer) "
        "pair decide whether answer denotes exactly input-1, in ANY standard notation (word form or "
        "digits, with or without 'and'). IMPORTANT: the SAME NUMBER as the input does NOT count. FALSE "
        "if answer is any other number, equals the input, or is not a number. Respond ONLY with a JSON "
        'array, one object per pair, in order: {"input": ..., "answer": ..., "correct": true|false}.'
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="GPT-4-judged top-1 accuracy for paired 1-shot capture (antonym/synonym).")
    p.add_argument("--graded_dir", type=Path, default=ARTIFACTS_ROOT / "oneshot_paired_graded" / "antonym_synonym")
    p.add_argument("--function_tasks", nargs="+", default=["antonym", "synonym"])
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--do_sample", action="store_true",
                   help="Sample instead of greedy decoding (one sample/prompt at --temperature).")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature (only used with --do_sample).")
    p.add_argument("--output_suffix", type=str, default="",
                   help="Appended to the per-task output dir (e.g. '_temp1' -> oneshot_<task>_judge_temp1).")
    p.add_argument("--judge_model", type=str, default="gpt-4.1")
    p.add_argument("--judge_batch_size", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    p.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    p.add_argument("--output_root", type=Path, default=LABEL_GEOMETRY_DIR)
    return p.parse_args()


def get_openai_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        for line in open("/proc/1/environ", "rb").read().split(b"\0"):
            if line.startswith(b"OPENAI_API_KEY="):
                key = line.decode().split("=", 1)[1]
                break
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in env or /proc/1/environ")
    return key


def extract_answer(generated_text, multiword=False):
    """First answer line after the prompt; trimmed at newline / next 'Q:'.
    multiword=True keeps a full phrase (numbers like 'one hundred one'); else first word run."""
    ans = generated_text.split("\n")[0].strip()
    ans = re.split(r"\bQ:", ans)[0].strip()
    pat = r"[A-Za-z0-9'\- ]+" if multiword else r"[A-Za-z'\-]+"
    m = re.match(pat, ans)
    return m.group(0).strip() if m else ans


def build_prompt(demo_input, demo_output, query_input, args):
    pd = word_pairs_to_prompt_data(
        {"input": [demo_input], "output": [demo_output]},
        query_target_pair={"input": query_input, "output": query_input},
        prepend_bos_token=False, shuffle_labels=False,
        prefixes=args.prefixes, separators=args.separators, prepend_space=True,
    )
    return create_prompt(pd)


def generate_answers(model, tokenizer, args, rows, task):
    multiword = task in ("next_number", "prev_number")
    out = []
    for j, r in enumerate(rows):
        prompt = build_prompt(r["demo_input"], r["output_word"], r["query"], args)
        inp = tokenizer(prompt, return_tensors="pt").to(args.device)
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        if args.do_sample:
            gen_kwargs.update(do_sample=True, temperature=args.temperature)
        else:
            gen_kwargs.update(do_sample=False)
        gen = model.generate(**inp, **gen_kwargs)
        completion = tokenizer.decode(gen[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        out.append({"function_task": task, "query_input": r["query"], "output_word": r["output_word"],
                    "demo_input": r["demo_input"], "gold_output": r["gold"],
                    "first_tok_rank": r["gold_first_tok_rank"],
                    "generated": extract_answer(completion, multiword=multiword),
                    "raw_completion": completion})
        if (j + 1) % 100 == 0:
            print(f"    {task}: generated {j+1}/{len(rows)}")
    return out


def _judge_batch(pairs, system, judge_model, api_key):
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": judge_model, "temperature": 0,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": json.dumps(pairs)}]},
        timeout=120,
    )
    resp.raise_for_status()
    body = resp.json()
    content = body["choices"][0]["message"]["content"].strip()
    content = re.sub(r"^```(json)?|```$", "", content, flags=re.M).strip()
    verdicts = json.loads(content)
    if len(verdicts) != len(pairs):
        raise ValueError(f"Judge returned {len(verdicts)} verdicts for {len(pairs)} pairs")
    return verdicts


def judge(records, system, judge_model, api_key, batch_size=50):
    pairs = [{"input": r["query_input"], "answer": r["generated"]} for r in records]
    verdicts = []
    for i in range(0, len(pairs), batch_size):
        verdicts.extend(_judge_batch(pairs[i:i + batch_size], system, judge_model, api_key))
        print(f"    judged {i + 1}-{min(i + batch_size, len(pairs))} / {len(pairs)}")
    return verdicts


def main():
    args = parse_args()
    grading = json.loads((args.graded_dir / "grading.json").read_text())
    api_key = get_openai_key()

    print("Loading model...")
    torch.set_grad_enabled(False)
    model, tokenizer, _cfg = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    model.eval()
    set_seed(args.seed)

    overall = {}
    for task in args.function_tasks:
        rows = [g for g in grading if g["function_task"] == task]
        print(f"\n[{task}] {len(rows)} prompts")
        records = generate_answers(model, tokenizer, args, rows, task)
        verdicts = judge(records, JUDGE_SYSTEMS[task], args.judge_model, api_key, args.judge_batch_size)
        for r, v in zip(records, verdicts):
            r["judge_correct"] = bool(v["correct"])
            r["copied_input"] = r["generated"].strip().lower() == r["query_input"].strip().lower()
            r["exact_match_gold"] = r["generated"].strip().lower() == r["gold_output"].strip().lower()

        n = len(records)
        judged = sum(r["judge_correct"] for r in records)
        copied = sum(r["copied_input"] for r in records)
        ft1 = sum(r["first_tok_rank"] < 1 for r in records)
        summary = {"function_task": task, "n": n, "judge_model": args.judge_model,
                   "do_sample": args.do_sample, "temperature": args.temperature if args.do_sample else 0.0,
                   "seed": args.seed,
                   "judge_top1_accuracy": judged / n, "first_token_top1_accuracy": ft1 / n,
                   "gold_exact_match_accuracy": sum(r["exact_match_gold"] for r in records) / n,
                   "copied_input_count": copied}
        out_dir = args.output_root / f"oneshot_{task}_judge{args.output_suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "judged_results.json").write_text(json.dumps({"summary": summary, "records": records}, indent=2))
        overall[task] = summary
        print(f"[{task}] GPT-4 judge top-1 = {judged}/{n} = {judged/n:.3f}  "
              f"(first-token top-1 {ft1}/{n} = {ft1/n:.3f}; copied {copied})")

    print("\n=== SUMMARY ===")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
