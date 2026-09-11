#!/usr/bin/env python
"""Step 3a — cue tokens for every pair, in GPT-J's tokenisation.

USER DEFINITION (2026-09-07): the cue token is the token immediately preceding a style choice =
the last token that is the same whichever style the model is about to produce.
  * k = 0: the last token shared by the two twins (may be word-internal: learn|ed vs learn|t).
  * k >= 1: judged on the context as it actually reads (nat or alt twin): take that twin up to and
    including opportunity k, form the version that differs ONLY in how opportunity k is rendered,
    and the cue is the last token the two share.
  * sentence families (double_space / all_caps / sentence_caps): the cue is the period that closes
    the previous sentence; for the first sentence of all_caps / sentence_caps it is the newline
    after "English:" (the last prompt token before the English begins).
  * curly_quotes: ONE decision per quotation, at the opening mark (the closing mark is rendered
    but not a decision point).

Prompt format: "Spanish:\\n{text_es}\\n\\nEnglish:\\n{english twin}".

For each pair record adds
  cues = {"tokenizer": ..., "nat": [...], "alt": [...]}  (list per CONTEXT polarity, one per decision
  point k): {k, cue_idx (token index in the full prompt of that twin), cue_tok, cue_char_end
  (char offset in the twin text where the cue token ends), opp_char_start, word_internal (cue ends
  inside the word that carries the opportunity), next_nat, next_alt (the first token(s) that follow
  the cue under each rendering, up to where the two renderings re-converge or 4 tokens)}.
Sanity: identical cue token id under both renderings by construction; counts per polarity.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import STYLE_TRANSLATION_DATA
from src.sandbox.style_translation.families import FAMILIES

PAIRS = STYLE_TRANSLATION_DATA / "pairs"
HEADER = "Spanish:\n{es}\n\nEnglish:\n"
TOKENIZER = "EleutherAI/gpt-j-6B"


def decision_points(rec):
    """Indices into rec['opps'] that count as decision points (curly_quotes: opening marks only)."""
    if rec["family"] == "curly_quotes":
        return [i for i, o in enumerate(rec["opps"]) if o["alt"] == "“"]
    return list(range(len(rec["opps"])))


def variant(rec, pol, k_opp_indices):
    """Twin `pol` with the opportunities in k_opp_indices rendered in the OTHER polarity."""
    other = "alt" if pol == "nat" else "nat"
    text = rec[f"text_{pol}"]
    spans = [tuple(o[f"{pol}_span"]) for o in rec["opps"]]
    out, pos = [], 0
    for i, o in enumerate(rec["opps"]):
        a, b = spans[i]
        out.append(text[pos:a])
        out.append(o[other] if i in k_opp_indices else o[pol])
        pos = b
    out.append(text[pos:])
    return "".join(out)


def common_prefix_len(a_ids, b_ids):
    n = 0
    for x, y in zip(a_ids, b_ids):
        if x != y:
            break
        n += 1
    return n


def next_tokens(ids, start, other_ids, max_n=4):
    """Tokens from `start` until the two id lists re-align (same token at same offset) or max_n."""
    out = []
    for j in range(start, min(len(ids), start + max_n)):
        out.append(ids[j])
        if j < len(other_ids) and ids[j] == other_ids[j] and j > start:
            break
    return out


def cues_for(rec, tok, pol):
    header = HEADER.format(es=rec["text_es"])
    text = rec[f"text_{pol}"]
    full = header + text
    enc = tok(full, return_offsets_mapping=True)
    ids, offs = enc.input_ids, enc.offset_mapping
    h = len(header)
    dps = decision_points(rec)
    cues = []
    for k, i in enumerate(dps):
        flip = {i}
        if rec["family"] == "curly_quotes":          # flip the closing mark of the same quotation too
            flip.add(i + 1)
        alt_text = variant(rec, pol, flip)
        enc_b = tok(header + alt_text, return_offsets_mapping=True)
        ids_b = enc_b.input_ids
        n = common_prefix_len(ids, ids_b)
        assert n >= 1, "prompts share at least the header"
        cue_idx = n - 1
        cue_end = offs[cue_idx][1]                     # char offset in `full`
        opp_start = h + rec["opps"][i][f"{pol}_span"][0]
        word_internal = cue_end > opp_start or (cue_end == opp_start and full[cue_end - 1].isalnum() and full[opp_start:opp_start + 1].isalnum() and False)
        # word-internal := the cue token ends strictly inside the span carrying the opportunity
        word_internal = cue_end > opp_start
        cues.append({
            "k": k, "opp_index": i, "cue_idx": cue_idx, "cue_tok": tok.decode([ids[cue_idx]]),
            "cue_tok_id": ids[cue_idx], "cue_char_end": cue_end - h, "opp_char_start": opp_start - h,
            "word_internal": bool(word_internal),
            "next_%s" % pol: tok.decode(next_tokens(ids, n, ids_b)),
            "next_%s" % ("alt" if pol == "nat" else "nat"): tok.decode(next_tokens(ids_b, n, ids)),
            "in_header": cue_end <= h,
        })
    return cues, len(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES])
    ap.add_argument("--examples", type=int, default=0, help="print N example cues per family")
    ap.add_argument("--model", default="gptj", help="models.MODELS key; non-default models write cues to artifacts, not into pairs/")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from src.sandbox.style_translation.models import paths as model_paths
    MP = model_paths(args.model); TOKENIZER = MP["tokenizer"]
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    if MP["cues"] is not None:
        MP["cues"].mkdir(parents=True, exist_ok=True)

    print(f"{'family':14s} {'pairs':>5s} {'cues/text':>9s} {'word-internal':>13s} {'in header':>9s} {'max prompt tok':>14s}  top cue tokens (nat ctx)")
    for fam in args.families:
        recs = json.load(open(PAIRS / f"{fam}.json"))
        top = collections.Counter(); wi = ih = ncue = 0; maxlen = 0
        for r in recs:
            r["cues"] = {"tokenizer": TOKENIZER, "prompt_format": HEADER.replace("\n", "\\n") + "{english}"}
            for pol in ("nat", "alt"):
                cues, L = cues_for(r, tok, pol)
                r["cues"][pol] = cues; maxlen = max(maxlen, L)
                if pol == "nat":
                    for c in cues:
                        top[c["cue_tok"]] += 1; ncue += 1; wi += c["word_internal"]; ih += c["in_header"]
            # identity check: k=0 cue token id must be the same in both contexts
            assert r["cues"]["nat"][0]["cue_tok_id"] == r["cues"]["alt"][0]["cue_tok_id"], r["doc_id"]
            assert len(r["cues"]["nat"]) == len(r["cues"]["alt"]) == len(decision_points(r))
        if MP["cues"] is None:
            json.dump(recs, open(PAIRS / f"{fam}.json", "w"), indent=0, ensure_ascii=False)
        else:
            json.dump([{"doc_id": r["doc_id"], "cues": r["cues"]} for r in recs], open(MP["cues"] / f"{fam}.json", "w"))
        print(f"{fam:14s} {len(recs):5d} {ncue/len(recs):9.1f} {wi/ncue:13.2f} {ih/ncue:9.2f} {maxlen:14d}  "
              + ", ".join(f"{t!r}:{c}" for t, c in top.most_common(5)))
        if args.examples:
            for r in recs[:1]:
                text = r["text_nat"]
                for c in r["cues"]["nat"][: args.examples]:
                    a = max(0, c["cue_char_end"] - 45)
                    print(f"   k={c['k']}  …{text[a:c['cue_char_end']]}【{c['cue_tok']!r}】{text[c['cue_char_end']:c['cue_char_end'] + 25]!r}"
                          f"   next nat={c['next_nat']!r} | next alt={c['next_alt']!r}")


if __name__ == "__main__":
    main()
