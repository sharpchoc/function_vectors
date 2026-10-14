#!/usr/bin/env python
"""Evidence-token positions for the code-translation prompts (model key qwen25_code_tr): the Python part of every prompt tokenises identically
after the new header, so each evidence index of the original prompt (read_features/evidence/<family>.json) shifts by the header length
difference. Verified per prompt (token sequences after the headers must be identical); prompts that fail are dropped and counted."""
import argparse
import json
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


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--families", nargs="*", required=True)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--model_tr", default="qwen25_code_tr")
    args = ap.parse_args(); MP = paths(args.model); MT = paths(args.model_tr); MT["evidence"].mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MP["tokenizer"])
    for fam in args.families:
        old = {(p["doc_id"], p["style"], p["k"]): p for p in json.load(open(MP["prompts"] / f"{fam}.json"))}
        ev = {(e["doc_id"], e["pole"], e["k"]): e for e in json.load(open(MP["evidence"] / f"{fam}.json"))}
        recs = {r["doc_id"]: r for r in json.load(open(PAIRS / f"{fam}.json"))}
        out = []; bad = 0; missing = 0; hcache = {}
        for p in json.load(open(MT["prompts"] / f"{fam}.json")):
            key = (p["doc_id"], p["style"], p["k"]); o = old[key]; e = ev.get(key)
            if e is None:
                missing += 1; continue
            if p["doc_id"] not in hcache:
                hcache[p["doc_id"]] = (len(tok(header_for(recs[p["doc_id"]])).input_ids), len(tok(p["header"]).input_ids))
            oh, nh = hcache[p["doc_id"]]; shift = nh - oh
            if o["prompt_ids"][oh:] != p["prompt_ids"][nh:] or len(p["prompt_ids"]) - len(o["prompt_ids"]) != shift or e["prompt_len"] != len(o["prompt_ids"]):
                bad += 1; continue
            new = dict(e); new["idx"] = [i + shift for i in e["idx"]]; new["prompt_len"] = len(p["prompt_ids"]); new["shift"] = shift; out.append(new)
        json.dump(out, open(MT["evidence"] / f"{fam}.json", "w"), ensure_ascii=False)
        print(f"{fam}: {len(out)} evidence records ({bad} tokenisation mismatches dropped, {missing} prompts without evidence record); mean evidence tokens {sum(len(x['idx']) for x in out) / max(len(out), 1):.1f}", flush=True)


if __name__ == "__main__":
    main()
