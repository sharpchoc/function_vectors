#!/usr/bin/env python
"""Code-translation prompts (user design 2026-10-13): the existing prompt of a document (header + Python twin cut at the cue) gets a new header that
carries the task description, the document's accepted source program in another language (code_translate.py / code_translate_finish.py output) and
a "Python translation:" label; the Python prefix and the cue are unchanged. Documents without an accepted translation are skipped.
  header = "Task:\\n{task}\\n\\n{Lang} solution:\\n{code}\\n\\nPython translation:\\n"
Writes <model_tr prompts>/<family>.json with the original schema plus `lang`; checks that the new prompt is a token prefix of the full new document.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths
from src.sandbox.style_translation.cue_tokens import header_for
from src.sandbox.style_translation.logprob_margin import PAIRS

HEADER = "Task:\n{task}\n\n{lang} solution:\n{code}\n\nPython translation:\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", required=True); ap.add_argument("--src", type=Path, default=None, help="translation records dir")
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--model_tr", default="qwen25_code_tr")
    args = ap.parse_args(); MP = paths(args.model); MT = paths(args.model_tr); MT["prompts"].mkdir(parents=True, exist_ok=True)
    src = args.src or MP["prompts"].parent / "translation" / "v1"; tok = AutoTokenizer.from_pretrained(MP["tokenizer"])
    lex = MP["prompts"].parent / "scoring_lexicon.json"
    if lex.exists() and not (MT["prompts"].parent / "scoring_lexicon.json").exists():
        (MT["prompts"].parent / "scoring_lexicon.json").write_text(lex.read_text())
    for fam in args.families:
        T = {r["doc_id"]: r for r in json.load(open(src / f"{fam}.json")) if r["passed"]}
        recs = {r["doc_id"]: r for r in json.load(open(PAIRS / f"{fam}.json"))}
        out = []; skipped_doc = set(); mismatch = 0; cache = {}
        for it in json.load(open(MP["prompts"] / f"{fam}.json")):
            d = it["doc_id"]
            if d not in T:
                skipped_doc.add(d); continue
            rec = recs[d]; old_h = header_for(rec); full_old = tok.decode(it["prompt_ids"])
            assert full_old.startswith(old_h), (fam, d)
            prefix = full_old[len(old_h):]
            new_h = HEADER.format(task=rec["text_es"], lang=T[d]["lang"], code=T[d]["code"].rstrip("\n"))
            ids = tok(new_h + prefix).input_ids
            key = (d, it["style"])
            if key not in cache:
                cache[key] = tok(new_h + rec[f"text_{it['style']}"]).input_ids
            if cache[key][: len(ids)] != ids:                      # the cue must stay a clean token boundary in the new tokenisation
                mismatch += 1; continue
            new = dict(it); new["prompt_ids"] = ids; new["lang"] = T[d]["lang"]; new["header"] = new_h; new["context_tail"] = it["context_tail"]
            out.append(new)
        tmp = MT["prompts"] / f"{fam}.json.tmp"; json.dump(out, open(tmp, "w"), ensure_ascii=False); os.replace(tmp, MT["prompts"] / f"{fam}.json")   # atomic: shards read these files
        n_tok = sum(len(x["prompt_ids"]) for x in out) / max(len(out), 1)
        print(f"{fam}: {len(out)} prompts ({len({x['doc_id'] for x in out})} docs; {len(skipped_doc)} docs without accepted translation, {mismatch} prompts with cue-boundary mismatch); mean length {n_tok:.0f} tokens", flush=True)


if __name__ == "__main__":
    main()
