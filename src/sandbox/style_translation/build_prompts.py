#!/usr/bin/env python
"""Step 3 — materialise the GPT-J prompts: for every pair, style P in {nat, alt}, k in 0..K-1,
the prompt is  "Spanish:\\n{text_es}\\n\\nEnglish:\\n{text_P}"  cut right AFTER cue token k
(token-exact: ids[: cue_idx + 1], asserted against the stored cue token id and char offset).

Per item: doc_id, family, style, k, prompt_ids, cue_tok, seg_prefix (twin[opp_char_start:
cue_char_end], the already-generated start of a word-internal cue), next_nat, next_alt,
ref_sentence (the twin's own continuation from the cue to the end of that sentence), es_text.
Output: artifacts/style_translation/prompts/<family>.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, STYLE_TRANSLATION_DATA
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.cue_tokens import HEADER, TOKENIZER

PAIRS = STYLE_TRANSLATION_DATA / "pairs"
OUT = ARTIFACTS_ROOT / "style_translation" / "prompts"
_SENT_END = re.compile(r'[.!?]+["”]?(?=\s|$)')


def ref_sentence(text, cue_char_end, opp_char_start):
    m = _SENT_END.search(text, max(cue_char_end, opp_char_start))
    end = m.end() if m else len(text)
    return text[cue_char_end:end]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES])
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--model", default="gptj", help="models.MODELS key (tokeniser + artifact folder)")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from src.sandbox.style_translation.models import paths as model_paths
    MP = model_paths(args.model); OUT = MP["prompts"]
    tok = AutoTokenizer.from_pretrained(MP["tokenizer"])
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for fam in args.families:
        recs = json.load(open(PAIRS / f"{fam}.json"))
        if MP["cues"] is not None:                     # tokeniser-specific cues live in artifacts for non-default models
            cues = {c["doc_id"]: c["cues"] for c in json.load(open(MP["cues"] / f"{fam}.json"))}
            for r in recs:
                r["cues"] = cues[r["doc_id"]]
        items = []
        for r in recs:
            header = HEADER.format(es=r["text_es"]); h = len(header)
            for P in ("nat", "alt"):
                text = r[f"text_{P}"]
                enc = tok(header + text, return_offsets_mapping=True)
                ids, offs = enc.input_ids, enc.offset_mapping
                for c in r["cues"][P][: args.K]:
                    ci = c["cue_idx"]
                    assert ids[ci] == c["cue_tok_id"], (r["doc_id"], P, c["k"])
                    assert offs[ci][1] == h + c["cue_char_end"], (r["doc_id"], P, c["k"], offs[ci], c["cue_char_end"])
                    items.append({
                        "doc_id": r["doc_id"], "family": fam, "style": P, "k": c["k"],
                        "prompt_ids": ids[: ci + 1], "cue_tok": c["cue_tok"],
                        "seg_prefix": text[c["opp_char_start"]: c["cue_char_end"]],
                        "next_nat": c["next_nat"], "next_alt": c["next_alt"],
                        "ref_sentence": ref_sentence(text, c["cue_char_end"], c["opp_char_start"]),
                        "context_tail": text[: c["cue_char_end"]][-500:],
                        "es_text": r["text_es"],
                    })
        json.dump(items, open(OUT / f"{fam}.json", "w"))
        total += len(items)
        ks = {}
        for it in items:
            ks[(it["style"], it["k"])] = ks.get((it["style"], it["k"]), 0) + 1
        print(f"{fam:14s} items={len(items):5d} per (style,k)={sorted(set(ks.values()))} "
              f"max_prompt={max(len(it['prompt_ids']) for it in items)}", flush=True)
    print("total items", total)


if __name__ == "__main__":
    main()
