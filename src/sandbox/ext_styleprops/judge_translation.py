#!/usr/bin/env python
"""Translation-correctness scoring of sampled continuations (translation-framing variant).

USER DECISION 2026-09-06: Gemini judge PRIMARY, normalised exact match SECONDARY.

For each prescreen record (see prescreen_adherence.py; needs ref_nat/ref_alt/ctx_tail):
  trans_judge  Gemini (OpenRouter) verdict CORRECT/INCORRECT -> True/False, None on judge
               failure. The judge sees a window of the Spanish source around the aligned
               position, the end of the English prefix, the reference continuation (the
               document's own next tokens = the gold translation) and the FRAGMENT, and is
               told to ignore case/spelling/spacing/quote/dash/number-format conventions.
  trans_exact  normalised exact match: lowercase, quotes/dashes straightened, whitespace
               collapsed; True if the fragment agrees with EITHER twin's reference on their
               common prefix (length = min(len), must be >= MIN_COMMON chars).
Empty / whitespace-only fragments are INCORRECT without an API call.

--exact_only  compute trans_exact only (used for the plain English-only baseline, where
              "faithful translation" is undefined; README documents the asymmetry).

Verdicts are written back into the record JSON (idempotent; already-judged records are
skipped), mirroring judge_coherence.py.
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT
from src.sandbox.ext_styleprops.gen_corpus import load_key, URL

PROPS_DIR = REPO_ROOT / "dataset_files" / "style_properties" / "props"
ES_CORPUS = REPO_ROOT / "dataset_files" / "style_properties" / "base_corpus_es.json"
MIN_COMMON = 8
ES_WINDOW = 350   # chars either side of the aligned position

PROMPT = """You are checking a language model that translates a Spanish passage into English. It has
written the English translation up to a point and then produced a short FRAGMENT continuing it.
Decide whether the FRAGMENT is a CORRECT continuation of the translation.

Answer CORRECT if the fragment renders what comes next in the Spanish source (the REFERENCE shows
the gold continuation). A paraphrase with the same meaning is CORRECT. The fragment may be cut off
mid-word or mid-sentence; judge only the part that is there. IGNORE all of these completely: upper
or lower case, American vs British spelling, single vs double spaces, straight vs curly quotes,
hyphen vs dash, digits vs spelled-out numbers, ampersand vs "and", contractions, Oxford comma,
punctuation placement around quotes, "%" vs "percent".

Answer INCORRECT if the fragment is in Spanish, repeats or restarts the passage, introduces
content that is not in the source at this point, starts a heading or a new unrelated text, or
contradicts the source.

Examples:
  REFERENCE: " check your tire pressure at least once a week"  FRAGMENT: " CHECK YOUR TIRE PRESSURE AT LEAST" -> CORRECT
  REFERENCE: " check your tire pressure at least once a week"  FRAGMENT: " inspect the tyre pressure weekly, especially" -> CORRECT
  REFERENCE: " check your tire pressure at least once a week"  FRAGMENT: " revisar la presión de los neumáticos" -> INCORRECT
  REFERENCE: " check your tire pressure at least once a week"  FRAGMENT: " buy a new bicycle every spring" -> INCORRECT
  REFERENCE: " check your tire pressure at least once a week"  FRAGMENT: "\\n\\nSpanish:\\nMantener una bicicleta" -> INCORRECT

SPANISH SOURCE (window around the current position): ...{es_window}...

ENGLISH TRANSLATION SO FAR (end): ...{ctx}

REFERENCE (gold continuation): {ref}

FRAGMENT TO JUDGE: {tail}

Reply with exactly one word: CORRECT or INCORRECT."""

_NORM = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": "-", "–": "-", " ": " "})


def normalise(s):
    s = s.translate(_NORM).lower()
    return re.sub(r"\s+", " ", s).strip()


def exact_match(tail, refs):
    t = normalise(tail)
    for ref in refs:
        r = normalise(ref)
        n = min(len(t), len(r))
        if n >= MIN_COMMON and t[:n] == r[:n]:
            return True
    return False


def es_window(es_text, frac):
    c = int(frac * len(es_text))
    return es_text[max(0, c - ES_WINDOW):c + ES_WINDOW]


def judge_one(key, model, es_win, ctx, ref, tail):
    if not tail.strip():
        return False
    body = {"model": model, "temperature": 0.0, "max_tokens": 8,
            "messages": [{"role": "user", "content": PROMPT.format(
                es_window=es_win, ctx=ctx, ref=json.dumps(ref), tail=json.dumps(tail))}]}
    for attempt in range(4):
        try:
            r = requests.post(URL, json=body, timeout=60,
                              headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            ans = r.json()["choices"][0]["message"]["content"].strip().upper()
            if "INCORRECT" in ans:
                return False
            if "CORRECT" in ans:
                return True
            return None
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def prefix_frac(rec, docs):
    """Fraction of the English document consumed by the prompt (for Spanish alignment)."""
    text = docs[rec["doc_id"]][f"text_{rec['pol']}"]
    tail = rec["ctx_tail"]
    i = text.find(tail)
    if i < 0:
        return 0.5
    return (i + len(tail)) / max(len(text), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path,
                    default=ARTIFACTS_ROOT / "style_properties" / "prescreen_translate")
    ap.add_argument("--props", nargs="*", default=None)
    ap.add_argument("--es_corpus", type=Path, default=ES_CORPUS)
    ap.add_argument("--model", default="google/gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--exact_only", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="judge only the first N unjudged records per property (sanity runs)")
    args = ap.parse_args()

    files = sorted(args.dir.glob("*.json"))
    if args.props:
        files = [f for f in files if f.stem in args.props]
    key = None if args.exact_only else load_key()
    es = None if args.exact_only else {d["doc_id"]: d["text_es"]
                                       for d in json.load(open(args.es_corpus))}
    for f in files:
        d = json.load(open(f))
        recs = d["records"]
        assert all("ref_nat" in r for r in recs), f"{f.name}: records lack refs (run --add_refs)"
        for r in recs:
            r["trans_exact"] = exact_match(r["tail"], (r["ref_nat"], r["ref_alt"]))
        if args.exact_only:
            json.dump(d, open(f, "w"))
            print(f"{d['property']}: exact={sum(r['trans_exact'] for r in recs)/len(recs):.3f}",
                  flush=True)
            continue
        docs = {x["doc_id"]: x for x in json.load(open(PROPS_DIR / f"{d['property']}.json"))["docs"]}
        todo = [i for i, r in enumerate(recs) if r.get("trans_judge", "unset") in ("unset", None)]
        if args.limit is not None:
            todo = todo[:args.limit]
        if not todo:
            print(f"{d['property']}: already judged", flush=True)
            continue
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {}
            for i in todo:
                r = recs[i]
                win = es_window(es[r["doc_id"]], prefix_frac(r, docs))
                futs[ex.submit(judge_one, key, args.model, win, r["ctx_tail"],
                               r["ref_nat"], r["tail"])] = i
            n = 0
            for fu in as_completed(futs):
                recs[futs[fu]]["trans_judge"] = fu.result()
                n += 1
                if n % 2000 == 0:
                    json.dump(d, open(f, "w"))
                    print(f"{d['property']}: {n}/{len(todo)}", flush=True)
        json.dump(d, open(f, "w"))
        judged = [r for r in recs if r.get("trans_judge") is not None]
        print(f"{d['property']}: judged {len(todo)} | correct="
              f"{sum(r['trans_judge'] for r in judged)/max(len(judged),1):.3f} "
              f"exact={sum(r['trans_exact'] for r in recs)/len(recs):.3f} "
              f"judge_fail={sum(r.get('trans_judge', 0) is None for r in recs)}", flush=True)


if __name__ == "__main__":
    main()
