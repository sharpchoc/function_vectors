"""
Eval for the `ambiguous` task-disambiguation datasets (3 + 1 + 1 prompts).

For each task file f (one side of an ambiguous pair), the partner side is f' (same
inputs, outputs agree on the OVERLAP region and differ on the DIFFERENTIATOR region).
Each prompt:
    - 3 demos sampled from the OVERLAP (ambiguous: consistent with both f and f'),
    - 1 demo (the 4th) sampled from the DIFFERENTIATOR, using f's output (disambiguates -> f),
    - 1 query (the 5th) sampled from the DIFFERENTIATOR; scored against f's output.
The 3 overlap demos are ambiguous; only the 4th demo reveals that the task is f, so
accuracy measures whether the model infers + applies f from a single disambiguating example.

Metrics per task (n prompts): accuracy (model produces f's answer) and a diagnostic
`matches_partner` rate (model instead produces f''s answer = followed the prior/other rule).
Scoring is token-level exact match on the gold answer's token ids (repo convention).

Cross-prompt batched greedy generation. Run with:
    HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 \
      python src/eval_scripts/eval_ambiguous_disambiguation.py
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import word_pairs_to_prompt_data, create_prompt
from utils.paths import AMBIGUOUS_DIR

AMBIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "dataset_files", "ambiguous")

# pair -> (task_file_1, task_file_2)
PAIRS = [
    ("magnitude", "identity"),
    ("past_tense", "past_participle"),
    ("first_letter", "last_letter"),
    ("capital_city", "largest_city"),
    ("round", "truncate"),
    ("first_digit", "last_digit"),
    ("american", "british"),
    ("reverse", "identity_word"),
    ("count_vowels", "count_consonants"),
]


def load_task(name):
    return json.load(open(os.path.join(AMBIG_DIR, name + ".json")))


def split_overlap_differ(data_a, data_b):
    """Indices where the two task files agree (overlap) vs differ (differentiator)."""
    assert [x["input"] for x in data_a] == [x["input"] for x in data_b]
    overlap, differ = [], []
    for i, (a, b) in enumerate(zip(data_a, data_b)):
        (overlap if a["output"] == b["output"] else differ).append(i)
    return overlap, differ


def build_prompts(data, overlap_idx, differ_idx, n_prompts, rng,
                  n_shared_demos=3, n_diff_demos=1):
    """For one task: list of (prompt_str, gold_output, query_input).

    Each prompt = n_shared_demos overlap demos + n_diff_demos differentiator demos
    + 1 differentiator query. (3+1+1 default; 3+2+1 with n_diff_demos=2.)
    """
    prompts = []
    for _ in range(n_prompts):
        shared = rng.choice(overlap_idx, size=n_shared_demos, replace=False)
        # n_diff_demos differentiator demos + 1 query, all distinct (query = last)
        diff_pick = rng.choice(differ_idx, size=n_diff_demos + 1, replace=False)
        demo_diff = [int(j) for j in diff_pick[:-1]]
        query_idx = int(diff_pick[-1])

        inputs = [data[i]["input"] for i in shared] + [data[j]["input"] for j in demo_diff]
        outputs = [data[i]["output"] for i in shared] + [data[j]["output"] for j in demo_diff]
        query = data[query_idx]

        prompt_data = word_pairs_to_prompt_data(
            {"input": inputs, "output": outputs},
            query_target_pair={"input": query["input"], "output": query["output"]},
            prepend_bos_token=False, shuffle_labels=False, prepend_space=True,
        )
        prompts.append((create_prompt(prompt_data), query["output"], query["input"]))
    return prompts


def batched_topk(model, tok, prompt_strs, batch_size, device, k=5):
    """Single forward pass; return the top-k next-token ids at the answer position per prompt
    (left-padded, so index -1 is the real final token for every sequence)."""
    top_ids = []
    for i in range(0, len(prompt_strs), batch_size):
        chunk = prompt_strs[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]
        top_ids.extend(torch.topk(logits, k, dim=-1).indices.tolist())
    return top_ids


def score_topk(tok, top_ids, golds, alts):
    """First-answer-token top-1 / top-2 accuracy (gold token = first token of ' '+answer).
    partner_top1 = model's argmax is the OTHER function's first token."""
    n_t1 = n_t2 = n_partner = 0
    examples = []
    for tk, gold, alt in zip(top_ids, golds, alts):
        g = tok(" " + gold, add_special_tokens=False).input_ids[0]
        a = tok(" " + alt, add_special_tokens=False).input_ids[0]
        n_t1 += (tk[0] == g)
        n_t2 += (g in tk[:2])
        n_partner += (tk[0] == a)
        if len(examples) < 5:
            examples.append({"gold": gold, "top2": [tok.decode([x]).strip() for x in tk[:2]],
                             "top1_correct": bool(tk[0] == g)})
    n = len(top_ids)
    return n_t1 / n, n_t2 / n, n_partner / n, examples


def batched_beam(model, tok, prompt_strs, batch_size, device, k=5, max_new_tokens=8):
    """Beam search; return the top-k full-answer strings (best-first) per prompt — the
    whole-word analogue of top-k (each candidate is the decoded answer up to the newline)."""
    cands = []
    for i in range(0, len(prompt_strs), batch_size):
        chunk = prompt_strs[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=k,
                                  num_return_sequences=k, do_sample=False, early_stopping=True,
                                  pad_token_id=tok.eos_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]                 # (chunk*k, new_len)
        new = new.reshape(len(chunk), k, -1)
        for row in new:                                          # per prompt: k beams best-first
            cands.append([tok.decode(b, skip_special_tokens=True).split("\n")[0].strip()
                          for b in row])
    return cands


def _norm(s):
    return s.strip().lower().rstrip(".,!?;:")


def score_wordtopk(cands, golds, alts):
    """Full-word top-1 / top-2 and partner@1 (string match on the whole decoded answer)."""
    n_t1 = n_t2 = n_partner = 0
    examples = []
    for cs, gold, alt in zip(cands, golds, alts):
        cn = [_norm(c) for c in cs]
        g, a = _norm(gold), _norm(alt)
        n_t1 += (cn[0] == g)
        n_t2 += (g in cn[:2])
        n_partner += (cn[0] == a)
        if len(examples) < 5:
            examples.append({"gold": gold, "top2": cs[:2], "top1_correct": bool(cn[0] == g)})
    n = len(cands)
    return n_t1 / n, n_t2 / n, n_partner / n, examples


def batched_generate(model, tok, prompt_strs, max_new_tokens, batch_size, device):
    """Greedy-generate, return the new-token id lists for each prompt (cross-prompt batched)."""
    new_tokens = []
    for i in range(0, len(prompt_strs), batch_size):
        chunk = prompt_strs[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=tok.eos_token_id)
        new = gen[:, enc["input_ids"].shape[1]:].tolist()
        new_tokens.extend(new)
    return new_tokens


def score(tok, new_tokens, golds, alts):
    """Token-level exact match of the gold answer (space-prepended) and partner answer."""
    n_correct = n_partner = 0
    examples = []
    for new, gold, alt in zip(new_tokens, golds, alts):
        gold_ids = tok(" " + gold, add_special_tokens=False).input_ids
        alt_ids = tok(" " + alt, add_special_tokens=False).input_ids
        ok = new[:len(gold_ids)] == gold_ids
        partner = new[:len(alt_ids)] == alt_ids
        n_correct += ok
        n_partner += partner
        if len(examples) < 5:
            examples.append({"gold": gold, "alt": alt,
                             "pred": tok.decode(new).split("\n")[0].strip(),
                             "correct": bool(ok)})
    n = len(new_tokens)
    return n_correct / n, n_partner / n, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="EleutherAI/gpt-j-6b")
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--n_shared_demos", type=int, default=3)
    ap.add_argument("--n_diff_demos", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--scoring", choices=["topk", "wordtopk", "exact"], default="topk",
                    help="topk = first-answer-token top-1/top-2 (forward pass); "
                         "wordtopk = FULL-WORD top-1/top-2 via beam search (whole answer line); "
                         "exact = full-answer greedy exact match (generation)")
    ap.add_argument("--beam_k", type=int, default=5, help="num beams for wordtopk")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=str(AMBIGUOUS_DIR / "ambiguous_disambiguation" / "eval_summary.json"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, _ = load_gpt_model_and_tokenizer(args.model_name, device=device)
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    results = {}
    for a, b in PAIRS:
        da, db = load_task(a), load_task(b)
        overlap, differ = split_overlap_differ(da, db)
        print(f"\n=== pair {a} | {b}: overlap={len(overlap)} differ={len(differ)} ===")
        for task, data, partner in [(a, da, db), (b, db, da)]:
            rng = np.random.RandomState(args.seed)  # same demo/query draws across paired tasks
            prompts = build_prompts(data, overlap, differ, args.n_prompts, rng,
                                    n_shared_demos=args.n_shared_demos,
                                    n_diff_demos=args.n_diff_demos)
            prompt_strs = [p[0] for p in prompts]
            golds = [p[1] for p in prompts]
            # partner gold (the OTHER function's answer on the same query input)
            input_to_partner = {x["input"]: x["output"] for x in partner}
            alts = [input_to_partner[p[2]] for p in prompts]

            if args.scoring in ("topk", "wordtopk"):
                if args.scoring == "topk":
                    top_ids = batched_topk(model, tok, prompt_strs, args.batch_size, device)
                    top1, top2, partner_rate, examples = score_topk(tok, top_ids, golds, alts)
                else:
                    cands = batched_beam(model, tok, prompt_strs, args.batch_size, device,
                                         k=args.beam_k, max_new_tokens=args.max_new_tokens)
                    top1, top2, partner_rate, examples = score_wordtopk(cands, golds, alts)
                results[task] = {"pair": f"{a}|{b}", "n_prompts": args.n_prompts,
                                 "scoring": args.scoring,
                                 "top1": round(top1, 4), "top2": round(top2, 4),
                                 "partner_top1": round(partner_rate, 4), "examples": examples}
                print(f"  {task:16s} top1={top1:.3f}  top2={top2:.3f}  "
                      f"partner@1={partner_rate:.3f}")
            else:
                new_tokens = batched_generate(model, tok, prompt_strs, args.max_new_tokens,
                                              args.batch_size, device)
                acc, partner_rate, examples = score(tok, new_tokens, golds, alts)
                results[task] = {"pair": f"{a}|{b}", "n_prompts": args.n_prompts,
                                 "accuracy": round(acc, 4),
                                 "matches_partner": round(partner_rate, 4),
                                 "neither": round(1 - acc - partner_rate, 4),
                                 "examples": examples}
                print(f"  {task:16s} acc={acc:.3f}  partner={partner_rate:.3f}  "
                      f"neither={1-acc-partner_rate:.3f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {args.output}")
    print(f"\n=== SUMMARY ({args.scoring}) ===")
    for task, r in results.items():
        if args.scoring in ("topk", "wordtopk"):
            print(f"  {task:16s} {r['pair']:28s} top1={r['top1']:.3f}  "
                  f"top2={r['top2']:.3f}  partner@1={r['partner_top1']:.3f}")
        else:
            print(f"  {task:16s} {r['pair']:28s} acc={r['accuracy']:.3f}  "
                  f"partner={r['matches_partner']:.3f}  neither={r['neither']:.3f}")


if __name__ == "__main__":
    main()
