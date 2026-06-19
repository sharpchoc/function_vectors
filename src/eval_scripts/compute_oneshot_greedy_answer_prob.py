#!/usr/bin/env python
"""Greedy answer probability + probability-gated GPT-4-correct tag for the paired 1-shot capture.

For each prompt in artifacts/oneshot_paired_graded/<pair>/grading.json, regenerate the GREEDY answer
while capturing per-step token probabilities, and define the answer probability as the product of
the greedy (argmax) token probabilities over the ANSWER SPAN = generated tokens before the first
newline token. For a single-token answer this equals the first-token probability; for a multi-token
answer (e.g. number words) it is the joint product. Empty answers (model emits "\n" first) get
prob 0.0.

Then stamp into grading.json AND every matching activation row (both source and target roles) in
shard_*.pt:
  - `<prob_field>` (float, default `greedy_answer_prob`), and
  - `<tag_field>`  (bool)  = existing greedy `judge_top1` AND `<prob_field>` >= --threshold
    (default field name `judge_top1_p<thr%>`, e.g. judge_top1_p70 at threshold 0.70).

Reuses the already-computed greedy GPT-4 verdict (grading.json `judge_top1`); makes NO API calls.
Match key = (function_task, output_word, query) — unique per prompt. In-place rewrite.

Build the prompt exactly as judge_oneshot_paired.py does, so the regenerated greedy answer is
identical to the one that was judged.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.prompt_utils import word_pairs_to_prompt_data, create_prompt
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.paths import ARTIFACTS_ROOT


def parse_args():
    p = argparse.ArgumentParser(description="Greedy answer probability + prob-gated GPT-4-correct tag.")
    p.add_argument("--graded_dir", type=Path, default=ARTIFACTS_ROOT / "oneshot_paired_graded" / "antonym_synonym")
    p.add_argument("--function_tasks", nargs="+", default=["antonym", "synonym"])
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--prob_field", type=str, default="greedy_answer_prob")
    p.add_argument("--tag_field", type=str, default=None,
                   help="Bool field name; default judge_top1_p<thr%%> (e.g. judge_top1_p70).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    p.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    return p.parse_args()


def build_prompt(demo_input, demo_output, query_input, args):
    pd = word_pairs_to_prompt_data(
        {"input": [demo_input], "output": [demo_output]},
        query_target_pair={"input": query_input, "output": query_input},
        prepend_bos_token=False, shuffle_labels=False,
        prefixes=args.prefixes, separators=args.separators, prepend_space=True,
    )
    return create_prompt(pd)


def greedy_answer_prob(model, tokenizer, prompt, args):
    """Return (answer_prob, n_answer_tokens): product of greedy per-token probs over the generated
    tokens before the first newline token. Empty answer (newline first) -> (0.0, 0)."""
    inp = tokenizer(prompt, return_tensors="pt").to(args.device)
    out = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id,
                          output_scores=True, return_dict_in_generate=True)
    gen_ids = out.sequences[0, inp["input_ids"].shape[1]:]          # (gen_len,)
    prob = 1.0
    n = 0
    for step, tok in enumerate(gen_ids.tolist()):
        if "\n" in tokenizer.decode([tok]):                         # first newline ends the answer
            break
        p = torch.softmax(out.scores[step][0].float(), dim=-1)[tok].item()
        prob *= p
        n += 1
    return (prob if n > 0 else 0.0), n


def main():
    args = parse_args()
    thr = args.threshold
    tag_field = args.tag_field or f"judge_top1_p{int(round(thr * 100))}"

    grading_path = args.graded_dir / "grading.json"
    grading = json.loads(grading_path.read_text())

    print("Loading model...")
    torch.set_grad_enabled(False)
    model, tokenizer, _cfg = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    model.eval()
    set_seed(args.seed)

    # compute greedy answer prob per prompt; key = (function_task, output_word, query)
    prob_of = {}
    rows = [g for g in grading if g["function_task"] in args.function_tasks]
    for j, g in enumerate(rows):
        prompt = build_prompt(g["demo_input"], g["output_word"], g["query"], args)
        prob, ntok = greedy_answer_prob(model, tokenizer, prompt, args)
        prob_of[(g["function_task"], g["output_word"], g["query"])] = (prob, ntok)
        if (j + 1) % 100 == 0:
            print(f"  greedy prob {j + 1}/{len(rows)}")

    def gated(judge_top1, prob):
        return bool(judge_top1) and (prob >= thr)

    # tag grading.json
    g_tagged = 0
    for g in grading:
        pr = prob_of.get((g["function_task"], g["output_word"], g["query"]))
        if pr is None:
            continue
        prob, _ = pr
        g[args.prob_field] = prob
        g[tag_field] = gated(g.get("judge_top1", False), prob)
        g_tagged += 1
    grading_path.write_text(json.dumps(grading, indent=2))
    print(f"tagged {g_tagged}/{len(grading)} grading.json rows with {args.prob_field}+{tag_field}")

    # tag shards
    n_rows = n_tagged = 0
    for sp in sorted(glob.glob(str(args.graded_dir / "shard_*.pt"))):
        data = torch.load(sp, map_location="cpu", weights_only=False)
        for m in data["metadata"]:
            n_rows += 1
            pr = prob_of.get((m["function_task"], m["output_word"], m["query_word"]))
            if pr is None:
                continue
            prob, _ = pr
            m[args.prob_field] = prob
            m[tag_field] = gated(m.get("judge_top1", False), prob)
            n_tagged += 1
        torch.save(data, sp)
    print(f"tagged {n_tagged}/{n_rows} activation rows with {args.prob_field}+{tag_field}")

    # report
    for task in args.function_tasks:
        rs = [g for g in grading if g["function_task"] == task and tag_field in g]
        if not rs:
            continue
        gp = sum(g[tag_field] for g in rs)
        jt = sum(bool(g.get("judge_top1", False)) for g in rs)
        hi = sum(g[args.prob_field] >= thr for g in rs)
        meanp = sum(g[args.prob_field] for g in rs) / len(rs)
        print(f"  {task}: judge_top1 {jt}/{len(rs)} -> {tag_field} {gp}/{len(rs)} "
              f"(prob>= {thr}: {hi}/{len(rs)}; mean answer prob {meanp:.3f})")


if __name__ == "__main__":
    main()
