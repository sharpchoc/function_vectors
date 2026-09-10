#!/usr/bin/env python
"""Step 6a — evidence tokens: which tokens of a k = 4 prompt carry the convention (CPU).

User definition (2026-09-10): the tokenizer's NATURAL segmentation only, never a forced split. For
one rendered instance (decision point) the evidence tokens are the style-bearing tokens: from the
position where the twin and its single-instance flip first differ in token ids (= cue_idx + 1, the
cue token itself is never included) up to the last token that starts inside the rendered span (span
trimmed of trailing whitespace unless it is all whitespace). Whole-word tokens stay whole
(" learned" / " learnt", " colour"); where the tokenizer itself splits, only the differing pieces
count ("'s" vs " is", "st" in "Among|st"). If the rendered span is entirely shared with the other twin
(nat "Among" vs alt "Among|st"), the evidence is the divergence token itself (" these" vs "st"),
as for the absence poles generally. curly_quotes: opening mark + the paired closing mark.

Input: pairs (opps + stored cues) and the step-3 k = 4 prompts. Output per family:
artifacts/style_translation/read_features/evidence/<family>.json — one record per (doc, pole) with
`prompt_len`, and per instance k = 0..3: `idx` (token positions in the prompt), `toks` (decoded).
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
from src.utils.paths import ARTIFACTS_ROOT, STYLE_TRANSLATION_DATA
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.cue_tokens import HEADER, TOKENIZER, decision_points, variant, common_prefix_len

PAIRS = STYLE_TRANSLATION_DATA / "pairs"
PROMPTS = ARTIFACTS_ROOT / "style_translation" / "prompts"
OUT = ARTIFACTS_ROOT / "style_translation" / "read_features" / "evidence"
K = 4


def span_tokens(offs, h, span, text, start, prompt_len):
    """Token indices >= start whose char start lies inside the span (trailing whitespace trimmed)."""
    s, e = span
    piece = text[s:e]
    end_char = h + (s + len(piece.rstrip()) if piece.strip() else e)
    idx = [j for j in range(start, len(offs)) if offs[j][0] < end_char and j < prompt_len]
    return idx


def evidence_for(rec, tok, pol, prompt_ids):
    header = HEADER.format(es=rec["text_es"]); text = rec[f"text_{pol}"]; full = header + text
    enc = tok(full, return_offsets_mapping=True); ids, offs = enc.input_ids, enc.offset_mapping
    assert ids[:len(prompt_ids)] == list(prompt_ids), "stored prompt is not a prefix of the twin tokenisation"
    h = len(header); dps = decision_points(rec); out = []
    stored = {c["k"]: c for c in rec["cues"][pol]}
    for k in range(K):
        i = dps[k]
        flip = {i} | ({i + 1} if rec["family"] == "curly_quotes" else set())
        ids_b = tok(header + variant(rec, pol, flip)).input_ids
        n = common_prefix_len(ids, ids_b)
        assert n - 1 == stored[k]["cue_idx"], f"cue mismatch {rec['doc_id']} {pol} k={k}"
        assert n < len(ids) and (n >= len(ids_b) or ids[n] != ids_b[n]), "no divergence after the cue"
        idx = span_tokens(offs, h, rec["opps"][i][f"{pol}_span"], text, n, len(prompt_ids))
        if rec["family"] == "curly_quotes":                       # paired closing mark of the same quotation
            s2 = rec["opps"][i + 1][f"{pol}_span"]
            first = next((j for j in range(n, len(offs)) if offs[j][1] > h + s2[0]), None)
            if first is not None:
                idx += span_tokens(offs, h, s2, text, first, len(prompt_ids))
        idx = sorted(set(idx))
        if not idx:      # span fully shared with the other twin (e.g. "Among" vs "Among|st"): the divergence token reveals the choice
            idx = [n]
            assert n < len(prompt_ids), f"divergence token outside the prompt {rec['doc_id']} {pol} k={k}"
        assert min(idx) == n and max(idx) < len(prompt_ids), "evidence outside the prompt or not at the divergence"
        assert stored[k]["cue_idx"] not in idx
        out.append({"k": k, "opp_index": i, "idx": idx, "toks": [tok.decode([ids[j]]) for j in idx]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES])
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    OUT.mkdir(parents=True, exist_ok=True)
    for fam in args.families:
        pairs = {p["doc_id"]: p for p in json.load(open(PAIRS / f"{fam}.json"))}
        prompts = [p for p in json.load(open(PROMPTS / f"{fam}.json")) if p["k"] == K]
        recs, per_inst, examples = [], collections.Counter(), collections.defaultdict(list)
        for p in prompts:
            ev = evidence_for(pairs[p["doc_id"]], tok, p["style"], p["prompt_ids"])
            recs.append({"doc_id": p["doc_id"], "pole": p["style"], "prompt_len": len(p["prompt_ids"]), "instances": ev})
            for e in ev:
                per_inst[len(e["idx"])] += 1
                if len(examples[p["style"]]) < 3:
                    examples[p["style"]].append("".join(e["toks"]))
        json.dump(recs, open(OUT / f"{fam}.json", "w"), ensure_ascii=False)
        dist = sorted(per_inst.items())
        print(f"{fam:14s} prompts {len(recs)} | tokens per instance: " + ", ".join(f"{n}:{c}" for n, c in dist[:6]) + (" ..." if len(dist) > 6 else "")
              + f" | nat e.g. {examples['nat']} | alt e.g. {examples['alt']}", flush=True)


if __name__ == "__main__":
    main()
