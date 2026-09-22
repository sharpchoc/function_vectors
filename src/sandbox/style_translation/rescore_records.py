#!/usr/bin/env python
"""Re-score stored rollout and steering records with the CURRENT scorer (2026-09-23, after scorer fixes): recompute `decision` and
`style_ok` from the stored cut `tail`, exactly as rollout.py / write_sweep.py / read_sweep.py do — decide_any(fam, prompt_text, seg_prefix,
tail, next_nat, next_alt, lexicon) with prompt_text = the decoded prompt of the record's (doc_id, style, k) item (write-steering records:
the k = 0 prompt; read-steering records: the k = 3 prompt of the record's context_style). `judge` fields untouched.
    rescore_records.py --families f1 f2 ... [--apply]      (dry run by default: prints how many decisions would change)"""
import argparse, glob, json, sys
from pathlib import Path
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from transformers import AutoTokenizer
from src.sandbox.style_translation.models import paths as model_paths, MODELS
from src.sandbox.style_translation.code_scoring import decide_any


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--families", nargs="+", required=True); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(); MP = model_paths(a.model); A = MP["prompts"].parent
    tok = AutoTokenizer.from_pretrained(MODELS[a.model]["tokenizer"]); lex_p = A / "scoring_lexicon.json"; lexicon = json.load(open(lex_p))["lexicon"] if lex_p.exists() else {}
    for fam in a.families:
        texts = {(it["doc_id"], it["style"], it["k"]): tok.decode(it["prompt_ids"]) for it in json.load(open(MP["prompts"] / f"{fam}.json"))}
        files = [A / "rollouts" / f"{fam}.json"] + [Path(p) for p in glob.glob(str(A / "steering" / "full_k3" / "*" / f"{fam}.json"))] + [Path(p) for p in glob.glob(str(A / "read_steer" / "full_k3" / "*" / f"{fam}.json"))]
        for path in files:
            if not path.exists():
                continue
            recs = json.load(open(path)); changed = unsc_before = unsc_after = ok_before = ok_after = missing = 0
            for r in recs:
                if "context_style" in r:                               # read-steering record: k = 3 prompt in the context style
                    key = (r["doc_id"], r["context_style"], r["k"])
                elif "arm" in r:                                       # write-steering record: k = 0 prompt (identical for both poles)
                    key = (r["doc_id"], "nat", r["k"]) if (r["doc_id"], "nat", r["k"]) in texts else (r["doc_id"], "alt", r["k"])
                else:                                                  # rollout record
                    key = (r["doc_id"], r["style"], r["k"])
                pt = texts.get(key)
                if pt is None:
                    missing += 1; continue
                d = decide_any(fam, pt, r["seg_prefix"], r["tail"], r["next_nat"], r["next_alt"], lexicon)
                tgt = r.get("target") if "arm" in r and "context_style" not in r else r["style"]
                so = (d == tgt) if tgt not in (None, "none") else None
                unsc_before += r.get("decision") is None; unsc_after += d is None; ok_before += bool(r.get("style_ok")); ok_after += bool(so)
                if d != r.get("decision") or so != r.get("style_ok"):
                    changed += 1
                    if a.apply:
                        r["decision"], r["style_ok"] = d, so
            print(f"{fam:20s} {str(path.relative_to(A))}: {len(recs)} records, changed {changed}, unscorable {unsc_before} -> {unsc_after}, style_ok {ok_before} -> {ok_after}, missing prompt {missing}{'  [APPLIED]' if a.apply and changed else ''}", flush=True)
            if a.apply and changed:
                json.dump(recs, open(path, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
