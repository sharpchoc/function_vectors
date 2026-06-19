#!/usr/bin/env python
"""Judge GPT-J's 10-shot rhyme generations with GPT-4: does the answer ACTUALLY rhyme?

The standard eval scores only the single gold first token (rhyme: 0.024 top-1), which
penalizes valid alternative rhymes. Here we (1) rebuild the same 10-shot prompts the eval
used (same seed/template, demos from train, queries from the test split), (2) greedy-generate
GPT-J's actual answer word, and (3) ask an OpenAI judge model whether that word truly rhymes
with the query input. Reports judge-accuracy alongside gold exact-match.

The OPENAI_API_KEY is read from the environment, falling back to the container env
(/proc/1/environ) since RunPod pod env vars don't reach interactive shells.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import requests
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.prompt_utils import load_dataset, word_pairs_to_prompt_data, create_prompt, ICLDataset
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.paths import GENERAL_DIR


def parse_args():
    p = argparse.ArgumentParser(description="GPT-4-judged rhyme accuracy for GPT-J 10-shot generations.")
    p.add_argument("--task", type=str, default="rhyme")
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--n_shots", type=int, default=10)
    p.add_argument("--test_split", type=str, default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--judge_model", type=str, default="gpt-4.1")
    p.add_argument("--judge_batch_size", type=int, default=50,
                   help="Pairs per judge request; chunked so large n doesn't overflow one call.")
    p.add_argument("--all_queries", action="store_true",
                   help="Use EVERY example in the dataset as a query (max n), drawing the n_shots "
                        "demos from the other examples (leave-one-out). Overrides --test_split.")
    p.add_argument("--data_path", type=str, default=None,
                   help="Explicit dataset json for --all_queries (default: dataset_files/abstractive/<task>.json).")
    p.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    p.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    p.add_argument("--output_dir", type=Path, default=GENERAL_DIR / "rhyme_judge_eval")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--skip_generation", action="store_true",
                   help="Reuse existing generations.json and only (re)run the judge.")
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


def extract_answer(generated_text):
    """First answer line after the prompt: strip at newline / next 'Q:'; first word-ish token run."""
    ans = generated_text.split("\n")[0].strip()
    ans = re.split(r"\bQ:", ans)[0].strip()
    m = re.match(r"[A-Za-z'\-]+", ans)
    return m.group(0) if m else ans


def _query_demo_iter(args, dataset):
    """Yield (query_pair, demo_word_pairs) for each query.

    Default: queries from --test_split, demos sampled from train (Stream C behaviour).
    --all_queries: every example is a query (max n); its demos are sampled leave-one-out
    from all the OTHER examples, so demos and query stay disjoint with no leakage.
    """
    if args.all_queries:
        full = ICLDataset(args.data_path or f"dataset_files/abstractive/{args.task}.json")
        n = len(full)
        print(f"{args.task}: all_queries mode, n={n} (demos drawn leave-one-out, {args.n_shots}-shot)")
        all_idx = np.arange(n)
        for j in range(n):
            pool = all_idx[all_idx != j]
            demo_idx = np.random.choice(pool, args.n_shots, replace=False)
            yield full[j], full[demo_idx], n
    else:
        n = len(dataset[args.test_split])
        for j in range(n):
            demo_idx = np.random.choice(len(dataset["train"]), args.n_shots, replace=False)
            yield dataset[args.test_split][j], dataset["train"][demo_idx], n


def generate_answers(args):
    dataset = load_dataset(args.task, root_data_dir=args.root_data_dir, test_size=0.3, seed=args.seed)
    print(f"{args.task}: split sizes train={len(dataset['train'])} valid={len(dataset['valid'])} test={len(dataset['test'])}")
    torch.set_grad_enabled(False)
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    model.eval()
    prepend_bos = False if model_config["prepend_bos"] else True

    set_seed(args.seed)
    records = []
    for j, (word_pairs_test, word_pairs, n) in enumerate(_query_demo_iter(args, dataset)):
        prompt_data = word_pairs_to_prompt_data(
            word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos,
            shuffle_labels=False, prefixes=args.prefixes, separators=args.separators,
        )
        sentence = create_prompt(prompt_data)
        inputs = tokenizer(sentence, return_tensors="pt").to(args.device)
        out = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        query = prompt_data["query_target"]["input"]
        gold = prompt_data["query_target"]["output"]
        gold = gold[0] if isinstance(gold, list) else gold
        answer = extract_answer(completion)
        records.append({"query_input": query, "gold_output": gold, "generated": answer,
                        "raw_completion": completion})
        print(f"  [{j+1:2}/{n}] {query!r} -> {answer!r}  (gold {gold!r})")
    return records


JUDGE_SYSTEM = (
    "You are a strict rhyme judge. For each (input_word, answer_word) pair decide whether "
    "answer_word is a true rhyme of input_word in standard English pronunciation: the final "
    "stressed vowel sound and everything after it must match. Slant/eye rhymes count as false. "
    "The answer is also false if answer_word is identical to input_word, is a non-word, or is "
    "just a repetition/inflection of the input (e.g. cat->cats). Respond ONLY with a JSON array, "
    'one object per pair, in order: {"input": ..., "answer": ..., "rhymes": true|false}.'
)


def _judge_batch(pairs, judge_model, api_key):
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": judge_model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": json.dumps(pairs)},
            ],
        },
        timeout=120,
    )
    resp.raise_for_status()
    body = resp.json()
    content = body["choices"][0]["message"]["content"].strip()
    content = re.sub(r"^```(json)?|```$", "", content, flags=re.M).strip()
    verdicts = json.loads(content)
    if len(verdicts) != len(pairs):
        raise ValueError(f"Judge returned {len(verdicts)} verdicts for {len(pairs)} pairs")
    usage = body.get("usage", {})
    return verdicts, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def judge(records, judge_model, api_key, batch_size=50):
    pairs = [{"input": r["query_input"], "answer": r["generated"]} for r in records]
    verdicts, ptok, ctok = [], 0, 0
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        v, p, c = _judge_batch(chunk, judge_model, api_key)
        verdicts.extend(v)
        ptok += p
        ctok += c
        print(f"  judged pairs {i + 1}-{i + len(chunk)} / {len(pairs)}")
    if len(verdicts) != len(records):
        raise ValueError(f"Judge returned {len(verdicts)} verdicts for {len(records)} pairs")
    print(f"judge: {judge_model}, {ptok}+{ctok} tokens over {((len(pairs) - 1) // batch_size) + 1} request(s)")
    return verdicts


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gen_path = args.output_dir / "generations.json"

    if args.skip_generation and gen_path.exists():
        records = json.loads(gen_path.read_text())
        print(f"Reusing {len(records)} generations from {gen_path}")
    else:
        records = generate_answers(args)
        gen_path.write_text(json.dumps(records, indent=2))
        print(f"Wrote {gen_path}")

    api_key = get_openai_key()
    verdicts = judge(records, args.judge_model, api_key, batch_size=args.judge_batch_size)
    for r, v in zip(records, verdicts):
        r["judge_rhymes"] = bool(v["rhymes"])
        r["exact_match_gold"] = r["generated"].strip().lower() == r["gold_output"].strip().lower()

    n = len(records)
    judged = sum(r["judge_rhymes"] for r in records)
    exact = sum(r["exact_match_gold"] for r in records)
    copied = sum(r["generated"].strip().lower() == r["query_input"].strip().lower() for r in records)
    summary = {
        "task": args.task, "n_shots": args.n_shots, "test_split": args.test_split, "n": n,
        "judge_model": args.judge_model,
        "judged_rhyme_accuracy": judged / n,
        "gold_exact_match_accuracy": exact / n,
        "copied_input_count": copied,
        "generations_path": str(gen_path),
    }
    (args.output_dir / "judged_results.json").write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nJudge-accepted rhymes: {judged}/{n} = {judged/n:.3f}  (gold exact-match: {exact}/{n} = {exact/n:.3f})")


if __name__ == "__main__":
    main()
