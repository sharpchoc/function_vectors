#!/usr/bin/env python
"""style_free_text variation — prompts WITHOUT the Spanish/English scaffold.

Same pairs, same cue-token definition, but the prompt is just the English twin (style P) cut right
after cue token k. Cues are recomputed on the header-free tokenisation (the first English token can
tokenise differently without a preceding newline, and all indices shift). k = 0 for all_caps /
sentence_caps has NO preceding token: the prompt is GPT-J's <|endoftext|> start token.

Output: artifacts/style_free_text/prompts/<family>.json — same item schema as
style_translation.build_prompts (prompt_ids, seg_prefix, next_nat/next_alt, ref_sentence,
context_tail; es_text kept only for reference, never shown to the model or the judge).
Does not modify anything under dataset_files/ or results/style_translation/.
"""
import argparse
import json
import sys
from pathlib import Path

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, STYLE_TRANSLATION_DATA
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.cue_tokens import TOKENIZER, decision_points, variant, common_prefix_len, next_tokens
from src.sandbox.style_translation.build_prompts import ref_sentence

PAIRS = STYLE_TRANSLATION_DATA / "pairs"
OUT = ARTIFACTS_ROOT / "style_free_text" / "prompts"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES])
    ap.add_argument("--K", type=int, default=5)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    eos = tok.eos_token_id
    OUT.mkdir(parents=True, exist_ok=True)
    total = empty = 0
    for fam in args.families:
        recs = json.load(open(PAIRS / f"{fam}.json"))
        items = []
        for r in recs:
            dps = decision_points(r)[: args.K]
            for P in ("nat", "alt"):
                text = r[f"text_{P}"]
                enc = tok(text, return_offsets_mapping=True)
                ids, offs = enc.input_ids, enc.offset_mapping
                for k, i in enumerate(dps):
                    flip = {i} | ({i + 1} if fam == "curly_quotes" else set())
                    ids_b = tok(variant(r, P, flip)).input_ids
                    n = common_prefix_len(ids, ids_b)          # tokens shared before the divergence
                    opp_start = r["opps"][i][f"{P}_span"][0]
                    if n == 0:                                 # no context at all (first sentence, k=0)
                        prompt_ids, cue_tok, cue_end = [eos], "<|endoftext|>", 0
                        empty += 1
                    else:
                        prompt_ids, cue_tok, cue_end = ids[:n], tok.decode([ids[n - 1]]), offs[n - 1][1]
                    other = "alt" if P == "nat" else "nat"
                    items.append({
                        "doc_id": r["doc_id"], "family": fam, "style": P, "k": k,
                        "prompt_ids": prompt_ids, "cue_tok": cue_tok,
                        "seg_prefix": text[opp_start:cue_end] if cue_end > opp_start else "",
                        f"next_{P}": tok.decode(next_tokens(ids, n, ids_b)),
                        f"next_{other}": tok.decode(next_tokens(ids_b, n, ids)),
                        "ref_sentence": ref_sentence(text, cue_end, opp_start),
                        "context_tail": text[:cue_end][-500:],
                        "es_text": r["text_es"],
                    })
        json.dump(items, open(OUT / f"{fam}.json", "w"))
        total += len(items)
        print(f"{fam:14s} items={len(items):5d} max_prompt={max(len(it['prompt_ids']) for it in items)} "
              f"min_prompt={min(len(it['prompt_ids']) for it in items)}", flush=True)
    print("total items", total, "| empty-context prompts (eos start):", empty)


if __name__ == "__main__":
    main()
