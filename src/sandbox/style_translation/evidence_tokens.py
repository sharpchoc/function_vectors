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

Code families (user decision 2026-09-15): the twin builder merges nearby edits into one opportunity span
(`['rock', 'paper', 'scissors']` for py_quotes), so the evidence is restricted to the tokens of the span
that actually DIFFER between the nat rendering and the flipped rendering (token-level difflib on the
natural segmentation of both); shared content tokens inside the span are dropped (`n_shared_dropped`).
Empty result -> the whole-span rule, then the divergence token, as for text.

Input: pairs (opps + stored cues) and the step-3 k = 4 prompts. Output per family:
artifacts/style_translation/read_features/evidence/<family>.json — one record per (doc, pole) with
`prompt_len`, and per instance k = 0..3: `idx` (token positions in the prompt), `toks` (decoded).
"""
import argparse
import collections
import difflib
import json
import statistics
import sys
from pathlib import Path

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, STYLE_TRANSLATION_DATA
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.cue_tokens import HEADER, TOKENIZER, decision_points, variant, common_prefix_len, header_for
from src.sandbox.style_translation.models import paths as model_paths, arch
from src.sandbox.style_translation.ml_families import ML_FAMILY

PAIRS = STYLE_TRANSLATION_DATA / "pairs"
PROMPTS = ARTIFACTS_ROOT / "style_translation" / "prompts"
OUT = ARTIFACTS_ROOT / "style_translation" / "read_features" / "evidence"
K = 4


def configure(model="gptj"):
    global PROMPTS, OUT, TOKENIZER
    MP = model_paths(model); PROMPTS, OUT, TOKENIZER = MP["prompts"], MP["evidence"], MP["tokenizer"]


def span_tokens(offs, h, span, text, start, prompt_len):
    """Token indices >= start whose char start lies inside the span (trailing whitespace trimmed)."""
    s, e = span
    piece = text[s:e]
    end_char = h + (s + len(piece.rstrip()) if piece.strip() else e)
    idx = [j for j in range(start, len(offs)) if offs[j][0] < end_char and j < prompt_len]
    return idx


def is_code(fam):
    return getattr(ML_FAMILY.get(fam), "domain", "text") == "code"


def diff_only(tok, ids, idx, ids_b, offs_b, h, s, other_piece, text_b, n):
    """Code rule: keep only the nat-side span tokens that differ from the flipped rendering's span tokens."""
    idx_b = span_tokens(offs_b, h, (s, s + len(other_piece)), text_b, n, len(ids_b))
    a = [tok.decode([ids[j]]) for j in idx]; b = [tok.decode([ids_b[j]]) for j in idx_b]
    keep = []
    for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            keep += idx[i1:i2]
    return keep


def evidence_for(rec, tok, pol, prompt_ids):
    header = header_for(rec); text = rec[f"text_{pol}"]; full = header + text
    enc = tok(full, return_offsets_mapping=True); ids, offs = enc.input_ids, enc.offset_mapping
    assert ids[:len(prompt_ids)] == list(prompt_ids), "stored prompt is not a prefix of the twin tokenisation"
    h = len(header); dps = decision_points(rec); out = []
    stored = {c["k"]: c for c in rec["cues"][pol]}
    code = is_code(rec["family"]); other = "alt" if pol == "nat" else "nat"
    for k in range(K):
        i = dps[k]
        flip = {i} | ({i + 1} if rec["family"] == "curly_quotes" else set())
        text_b = variant(rec, pol, flip)
        enc_b = tok(header + text_b, return_offsets_mapping=True); ids_b, offs_b = enc_b.input_ids, enc_b.offset_mapping
        n = common_prefix_len(ids, ids_b)
        assert n - 1 == stored[k]["cue_idx"], f"cue mismatch {rec['doc_id']} {pol} k={k}"
        assert n < len(ids) and (n >= len(ids_b) or ids[n] != ids_b[n]), "no divergence after the cue"
        idx = span_tokens(offs, h, rec["opps"][i][f"{pol}_span"], text, n, len(prompt_ids))
        n_shared = 0
        if code and idx:
            keep = diff_only(tok, ids, idx, ids_b, offs_b, h, rec["opps"][i][f"{pol}_span"][0], rec["opps"][i][other], text_b, n)
            if keep:
                n_shared = len(idx) - len(keep); idx = keep
        if rec["family"] == "curly_quotes":                       # paired closing mark of the same quotation
            s2 = rec["opps"][i + 1][f"{pol}_span"]
            first = next((j for j in range(n, len(offs)) if offs[j][1] > h + s2[0]), None)
            if first is not None:
                idx += span_tokens(offs, h, s2, text, first, len(prompt_ids))
        idx = sorted(set(idx))
        outside = False
        if not idx:      # span fully shared with the other twin (e.g. "Among" vs "Among|st"): the divergence token reveals the choice
            if n < len(prompt_ids):
                idx = [n]
            else:        # adjacent opportunities merged into one token by the tokeniser (CJK): instance k's evidence lies beyond the k=4 cue
                outside = True
        if idx:
            assert (min(idx) == n or code) and max(idx) < len(prompt_ids), "evidence outside the prompt or not at the divergence"
            assert stored[k]["cue_idx"] not in idx
        out.append({"k": k, "opp_index": i, "idx": idx, "toks": [tok.decode([ids[j]]) for j in idx], "outside_prompt": outside,
                    "diff_only": code, "n_shared_dropped": n_shared})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES])
    ap.add_argument("--model", default="gptj", help="models.MODELS key (weights + artifact/results folders)")
    args = ap.parse_args()
    configure(args.model)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    MP = model_paths(args.model)
    OUT.mkdir(parents=True, exist_ok=True)
    for fam in args.families:
        pairs = {p["doc_id"]: p for p in json.load(open(PAIRS / f"{fam}.json"))}
        if MP["cues"] is not None:                     # tokeniser-specific cues live in artifacts for non-default models
            cues = {c["doc_id"]: c["cues"] for c in json.load(open(MP["cues"] / f"{fam}.json"))}
            for r in pairs.values():
                if r["doc_id"] in cues:
                    r["cues"] = cues[r["doc_id"]]
        prompts = [p for p in json.load(open(PROMPTS / f"{fam}.json")) if p["k"] == K]
        recs, per_inst, examples = [], collections.Counter(), collections.defaultdict(list)
        for p in prompts:
            ev = evidence_for(pairs[p["doc_id"]], tok, p["style"], p["prompt_ids"])
            recs.append({"doc_id": p["doc_id"], "pole": p["style"], "prompt_len": len(p["prompt_ids"]), "instances": ev})
            for e in ev:
                per_inst[len(e["idx"])] += 1   # 0 = evidence outside the prompt (merged adjacent opportunities)
                if len(examples[p["style"]]) < 3:
                    examples[p["style"]].append("".join(e["toks"]))
        json.dump(recs, open(OUT / f"{fam}.json", "w"), ensure_ascii=False)
        dist = sorted(per_inst.items())
        insts = [e for r in recs for e in r["instances"]]
        med = statistics.median([len(e["idx"]) for e in insts]) if insts else 0
        ws = _mean([bool(e["idx"]) and "".join(e["toks"]).strip() == "" for e in insts])
        outside = _mean([e["outside_prompt"] for e in insts]); dropped = sum(e.get("n_shared_dropped", 0) for e in insts)
        print(f"{fam:14s} prompts {len(recs)} | tokens per instance: " + ", ".join(f"{n}:{c}" for n, c in dist[:6]) + (" ..." if len(dist) > 6 else "")
              + f" | median {med:g} | whitespace-only {ws:.2f} | outside {outside:.3f} | shared tokens dropped {dropped}"
              + f" | nat e.g. {examples['nat']} | alt e.g. {examples['alt']}", flush=True)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


if __name__ == "__main__":
    main()
