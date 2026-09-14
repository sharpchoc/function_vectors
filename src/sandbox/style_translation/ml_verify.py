#!/usr/bin/env python
"""Verification pass for the multilingual corpora (plan 2026-09-14, Phase 1.2). Gemini 2.5 Flash, one call per pair record:
  fluent      the target text reads like a native speaker wrote it (no grammar/word-choice errors)
  natural_pole  the text is written entirely in the family's NATURAL convention (no alternative forms slipped in)
  anchors_ok  the required items are used correctly (family-specific check, e.g. es_rae2010 adverb/pronoun rule)
  faithful    the English source is a faithful rendering of the target text (no added/omitted content)
pass = all four. Writes back into dataset_files/style_translation/pairs/<family>.json (fields verify, pass) and prints counts.
Records already verified are skipped (resumable). `--drop_failed` rewrites the pairs file with pass=True records only
(the original is kept as pairs/<family>.unfiltered.json)."""
import argparse, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import STYLE_TRANSLATION_DATA
from src.sandbox.style_translation.gen_spanish import load_key, URL
from src.sandbox.style_translation.ml_families import ML_FAMILY

PAIRS = STYLE_TRANSLATION_DATA / "pairs"
PROMPT = """You are checking a bilingual text pair for a linguistics dataset.

TARGET TEXT ({lang}), which must be written entirely in this convention: {variety}.
Required items (natural forms; the parenthesised alternatives must NOT appear): {items}
{extra}
TARGET TEXT:
{tgt}

ENGLISH SOURCE (should be a faithful translation of the target text):
{src}

Answer with JSON only:
{{"fluent": true/false (grammatical, natural {lang} a native speaker would write; minor stylistic quirks are fine),
 "natural_pole": true/false (NO alternative-convention forms appear anywhere in the target text),
 "anchors_ok": true/false (the required items that appear are used correctly and naturally),
 "faithful": true/false (the English conveys the same content as the target text, no added or missing sentences),
 "notes": "<15 words"}}"""


def judge(key, model, fam, rec):
    items = ", ".join(f"{a} (not {b})" for a, b in fam.pairs[:60]) if fam.pairs else fam.variety_nat
    body = {"model": model, "temperature": 0.0, "max_tokens": 160, "messages": [{"role": "user", "content": PROMPT.format(
        lang=fam.tgt_lang, variety=fam.variety_nat, items=items, extra=fam.verify_extra, tgt=rec["text_nat"], src=rec["text_es"])}]}
    for attempt in range(5):
        try:
            r = requests.post(URL, json=body, timeout=90, headers={"Authorization": f"Bearer {key}"}); r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip(); raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
            v, _ = json.JSONDecoder(strict=False).raw_decode(raw[raw.index("{"):])
            v = {k: bool(v.get(k)) for k in ("fluent", "natural_pole", "anchors_ok", "faithful")} | {"notes": str(v.get("notes", ""))[:200], "judge": model}
            return rec["doc_id"], v
        except Exception as e:
            if attempt == 4:
                print(f"{rec['doc_id']} VERIFY FAILED: {e}", flush=True); return rec["doc_id"], None
            time.sleep(2 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=list(ML_FAMILY))
    ap.add_argument("--judge", default="google/gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--drop_failed", action="store_true")
    args = ap.parse_args()
    key = load_key()
    for name in args.families:
        fam = ML_FAMILY[name]; path = PAIRS / f"{name}.json"
        recs = json.load(open(path)); by = {r["doc_id"]: r for r in recs}
        todo = [r for r in recs if not r.get("verify")]
        with ThreadPoolExecutor(args.workers) as ex:
            for fu in as_completed([ex.submit(judge, key, args.judge, fam, r) for r in todo]):
                d, v = fu.result()
                if v is not None:
                    by[d]["verify"] = v; by[d]["pass"] = all(v[k] for k in ("fluent", "natural_pole", "anchors_ok", "faithful"))
        json.dump(recs, open(path, "w"), ensure_ascii=False, indent=0)
        vs = [r["verify"] for r in recs if r.get("verify")]
        n = len(vs) or 1
        print(f"{name:14s} verified {len(vs):3d}/{len(recs):3d} | pass {sum(r.get('pass') is True for r in recs):3d} | fluent {sum(v['fluent'] for v in vs)/n:.2f} "
              f"natural_pole {sum(v['natural_pole'] for v in vs)/n:.2f} anchors {sum(v['anchors_ok'] for v in vs)/n:.2f} faithful {sum(v['faithful'] for v in vs)/n:.2f}", flush=True)
        if args.drop_failed:
            json.dump(recs, open(PAIRS / f"{name}.unfiltered.json", "w"), ensure_ascii=False, indent=0)
            keep = [r for r in recs if r.get("pass") is True]
            json.dump(keep, open(path, "w"), ensure_ascii=False, indent=0)
            print(f"{name:14s} kept {len(keep)} passing pairs", flush=True)


if __name__ == "__main__":
    main()
