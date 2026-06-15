"""Show example 3+1+1 prompts for selected ambiguous pairs and how GPT-J tokenizes them.
Tokenizer only (no model weights). Run with HF_HOME=... HF_HUB_OFFLINE=1."""
import os
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval_ambiguous_disambiguation import load_task, split_overlap_differ, build_prompts, PAIRS  # noqa

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")

SHOW = [("magnitude", "identity"), ("count_vowels", "count_consonants")]


def pieces(text):
    ids = tok(text, add_special_tokens=False).input_ids
    toks = tok.convert_ids_to_tokens(ids)  # GPT-2 BPE: 'Ġ' marks a leading space
    return ids, toks


for a, b in SHOW:
    da, db = load_task(a), load_task(b)
    overlap, differ = split_overlap_differ(da, db)
    partner_of = {a: {x["input"]: x["output"] for x in db},
                  b: {x["input"]: x["output"] for x in da}}
    for task, data in [(a, da), (b, db)]:
        rng = np.random.RandomState(7)
        prompt, gold, qin = build_prompts(data, overlap, differ, 1, rng)[0]
        print("=" * 78)
        print(f"TASK: {task}   (query input={qin!r}  gold={gold!r}  "
              f"partner={partner_of[task][qin]!r})")
        print("-" * 78)
        print("PROMPT (repr, \\n shown):")
        print("   " + repr(prompt))
        print("\nPROMPT rendered:")
        print(prompt)
        print("<<<end of prompt — model generates here>>>")
        ids, toks = pieces(prompt)
        print(f"\nprompt = {len(ids)} tokens; last 12 BPE tokens: {toks[-12:]}")
        for lbl, ans in [("GOLD", gold), ("PARTNER", partner_of[task][qin])]:
            aids, atoks = pieces(" " + ans)  # answer is emitted as ' '+answer after 'A:'
            print(f"  {lbl} ' {ans}' -> ids={aids}  pieces={atoks}  (first-token id={aids[0]})")
        print()
