#!/usr/bin/env python
"""Spanish translation of the style-properties base corpus (translation-framing variant).

For the translation-framed prompt  "Spanish:\n<es>\n\nEnglish:\n<English twin up to the cue>"
every base document gets ONE Spanish rendering, shared by both English polarities. USER
DECISION 2026-09-06: the Spanish is *neutral and fixed* — standard Spanish typography
regardless of the English twin's convention (single spaces, straight quotes, digits, % sign,
sentence capitalisation, no em dashes), so the English prefix is the only source of the
convention. A deterministic fix-up enforces spaces/quotes/apostrophes after the LLM call; the
remaining leak counts are written to an audit csv.

Output: dataset_files/style_properties/base_corpus_es.json  [{doc_id, text_es}]
        results/style_properties/translation_framing/es_audit.csv
Resumable by doc_id (reuses gen_corpus.load_key / request pattern; Gemini via OpenRouter).
"""
import argparse
import csv
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
from src.utils.paths import REPO_ROOT, STYLE_PROPERTIES_DIR
from src.sandbox.ext_styleprops.gen_corpus import load_key, URL

BASE_PATH = REPO_ROOT / "dataset_files" / "style_properties" / "base_corpus.json"
OUT_PATH = REPO_ROOT / "dataset_files" / "style_properties" / "base_corpus_es.json"
AUDIT_PATH = STYLE_PROPERTIES_DIR / "translation_framing" / "es_audit.csv"

PROMPT = """Translate the following English passage into Spanish.

Requirements:
- Faithful, sentence-by-sentence translation: keep every sentence, in order, with the same
  content. Do not summarise, add, or drop anything. Keep the paragraph breaks of the source.
- Standard Spanish typography: straight double quotes ("...") for any quotation, a single space
  between sentences, normal sentence capitalisation (never all caps), no em dashes or en dashes
  (rephrase with commas or a hyphen-minus " - " instead), no ellipsis characters.
- Numbers: keep every numeral as digits exactly as in the source ("3" stays "3"; a spelled-out
  number like "three" stays spelled out as "tres"); write ordinals that are digits in the source
  with the Spanish marker ("3rd" -> "3.º") and spelled-out ordinals as words ("third" ->
  "tercero"); keep the percent sign as "%"; render "&" as "y".
- Output ONLY the Spanish text. No title, no preamble, no notes, no markdown.

English passage:
\"\"\"
{text}
\"\"\""""

_CURLY = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                        "«": '"', "»": '"', " ": " ", "…": "..."})


def fixup(txt):
    txt = txt.translate(_CURLY).replace("\r", "")
    txt = txt.strip().strip('"').strip()
    # Spanish parenthetical dashes -> the neutral " - " (numeric ranges keep their dash)
    txt = re.sub(r"(?<!\d)\s*[—–]\s*(?!\d)", " - ", txt)
    txt = re.sub(r"[ \t]+\n", "\n", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = re.sub(r" {2,}", " ", txt)
    return txt


_SENT = re.compile(r"[^.!?\n]+[.!?]?")


def audit(en, es):
    caps = 0
    for m in _SENT.finditer(es):
        letters = [c for c in m.group(0) if c.isalpha()]
        if len(letters) >= 3 and sum(c.isupper() for c in letters) / len(letters) >= 0.8:
            caps += 1
    return {
        "words_en": len(en.split()), "words_es": len(es.split()),
        "ratio": round(len(es.split()) / max(len(en.split()), 1), 3),
        "em_dash": es.count("—") + es.count("–"),
        "double_space": len(re.findall(r" {2,}", es)),
        "curly": sum(es.count(c) for c in "‘’“”«»"),
        "allcaps_sent": caps,
        "paras_en": en.count("\n\n") + 1, "paras_es": es.count("\n\n") + 1,
    }


def translate_one(key, model, doc):
    body = {"model": model, "temperature": 0.0, "max_tokens": 1800,
            "messages": [{"role": "user", "content": PROMPT.format(text=doc["text"])}]}
    for attempt in range(5):
        try:
            r = requests.post(URL, json=body, timeout=180,
                              headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            es = fixup(r.json()["choices"][0]["message"]["content"])
            ratio = len(es.split()) / max(len(doc["text"].split()), 1)
            if not (0.8 <= ratio <= 1.6):
                raise ValueError(f"length ratio {ratio:.2f}")
            return {"doc_id": doc["doc_id"], "text_es": es}
        except Exception as e:
            if attempt == 4:
                print(f"{doc['doc_id']} FAILED: {e}", flush=True)
                return None
            time.sleep(3 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="translate only the first N docs")
    ap.add_argument("--audit_only", action="store_true")
    ap.add_argument("--refix", action="store_true",
                    help="re-apply fixup() to the stored translations, rewrite, audit; no API")
    args = ap.parse_args()

    base = {d["doc_id"]: d for d in json.load(open(BASE_PATH))}
    done = {}
    if OUT_PATH.exists():
        done = {d["doc_id"]: d for d in json.load(open(OUT_PATH))}
        print(f"resume: {len(done)} docs already translated", flush=True)

    if args.refix:
        for d in done.values():
            d["text_es"] = fixup(d["text_es"])
        json.dump(sorted(done.values(), key=lambda d: d["doc_id"]),
                  open(OUT_PATH, "w"), indent=0, ensure_ascii=False)
        print(f"refix: rewrote {len(done)} docs", flush=True)
    elif not args.audit_only:
        todo = [d for k, d in sorted(base.items()) if k not in done]
        if args.limit is not None:
            todo = todo[:args.limit]
        print(f"{len(todo)} docs to translate with {args.model}", flush=True)
        key = load_key()

        def dump():
            json.dump(sorted(done.values(), key=lambda d: d["doc_id"]),
                      open(OUT_PATH, "w"), indent=0, ensure_ascii=False)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(translate_one, key, args.model, d): d["doc_id"] for d in todo}
            n = 0
            for f in as_completed(futs):
                rec = f.result()
                n += 1
                if rec:
                    done[rec["doc_id"]] = rec
                if n % 25 == 0:
                    dump()
                    print(f"{n}/{len(todo)} done ({len(done)} total)", flush=True)
        dump()
        print(f"final: {len(done)} docs -> {OUT_PATH}", flush=True)

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in sorted(done):
        a = audit(base[k]["text"], done[k]["text_es"])
        a["doc_id"] = k
        rows.append(a)
    cols = ["doc_id", "words_en", "words_es", "ratio", "em_dash", "double_space", "curly",
            "allcaps_sent", "paras_en", "paras_es"]
    with open(AUDIT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    if rows:
        import statistics
        print(f"audit: {len(rows)} docs | ratio median {statistics.median(r['ratio'] for r in rows):.2f} "
              f"[{min(r['ratio'] for r in rows):.2f}, {max(r['ratio'] for r in rows):.2f}] | "
              f"docs with em_dash {sum(r['em_dash'] > 0 for r in rows)}, double_space "
              f"{sum(r['double_space'] > 0 for r in rows)}, curly {sum(r['curly'] > 0 for r in rows)}, "
              f"allcaps_sent {sum(r['allcaps_sent'] > 0 for r in rows)}, paragraph mismatch "
              f"{sum(r['paras_en'] != r['paras_es'] for r in rows)}", flush=True)
    print(f"audit -> {AUDIT_PATH}", flush=True)


if __name__ == "__main__":
    main()
