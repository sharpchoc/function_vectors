#!/usr/bin/env python
"""Re-decide the convention of stored rollouts with the context-aware scorer (code_scoring.decide_code) — no sampling, no judging.
Families in code_scoring.CTX_FAMILIES only; the previous decision is kept once as `decision_v1`. If the context-aware rule finds nothing,
the exact next-token fallback of scoring.decide still applies (the completion starts with the twin's own rendering)."""
import argparse
import json
import sys
from pathlib import Path

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import STYLE_TRANSLATION_DATA
from src.sandbox.style_translation.code_scoring import CTX_FAMILIES, decide_code
from src.sandbox.style_translation.cue_tokens import header_for
from src.sandbox.style_translation.models import paths as model_paths


def fallback(tail, next_nat, next_alt):
    for label, nxt in sorted((("nat", next_nat), ("alt", next_alt)), key=lambda kv: -len(kv[1])):
        if nxt and tail.startswith(nxt[: max(1, min(len(nxt), 6))]):
            return label
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--families", nargs="*", default=sorted(CTX_FAMILIES))
    args = ap.parse_args(); MP = model_paths(args.model)
    lex_p = MP["prompts"].parent / "scoring_lexicon.json"; lexicon = json.load(open(lex_p))["lexicon"] if lex_p.exists() else {}
    for fam in args.families:
        pairs = {r["doc_id"]: r for r in json.load(open(STYLE_TRANSLATION_DATA / "pairs" / f"{fam}.json"))}
        cues = {c["doc_id"]: c for c in json.load(open(MP["cues"] / f"{fam}.json"))}
        path = MP["rollouts"] / f"{fam}.json"; recs = json.load(open(path)); changed = 0
        for r in recs:
            p = pairs[r["doc_id"]]; prompt = header_for(p) + p[f"text_{r['style']}"][: cues[r["doc_id"]]["cues"][r["style"]][r["k"]]["cue_char_end"]]
            d = decide_code(fam, prompt, r["seg_prefix"], r["tail"], r["next_nat"], r["next_alt"], lexicon)
            if d is None:
                d = fallback(r["tail"], r["next_nat"], r["next_alt"])
            r.setdefault("decision_v1", r["decision"]); changed += d != r["decision_v1"]
            r["decision"] = d; r["style_ok"] = (d == r["style"]); r["scorer"] = "context_v2"
        json.dump(recs, open(path, "w"), ensure_ascii=False)
        n = len(recs); u1 = sum(r["decision_v1"] is None for r in recs) / n; u2 = sum(r["decision"] is None for r in recs) / n
        print(f"{fam:16s} decisions changed {changed:5d} / {n} | unscorable {u1:.2f} -> {u2:.2f}", flush=True)


if __name__ == "__main__":
    main()
