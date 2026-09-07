#!/usr/bin/env python
"""Step 2b — verify each house-style English translation with Claude Haiku 4.5 (OpenRouter),
ONE text per call (same rationale as verify_spanish.py: sub-agents given hundreds of texts
script instead of reading).

Per text the judge sees the Spanish source, the normalised English (text_nat), and the tested
family's two styles, and returns strict JSON:
  coherent (bool), fluent (bool), faithful (bool: same content and sentence order, nothing added
  or dropped, numbers/quotations/lists preserved), style_consistent (bool: the family's feature
  appears ONLY in its nat form), anchors (English spans where the family's choice is made), notes.
Counts and conventions stay deterministic (translate_english.audit_nat / k_en).
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
from src.sandbox.style_translation.families import FAMILIES, FAMILY
from src.sandbox.style_translation.gen_spanish import load_key, URL
from src.sandbox.style_translation.translate_english import load_out, save_out

PROMPT = """You are verifying ONE English translation of a Spanish paragraph for a style study.
Style family: {name}. In English this family has two styles:
  NAT = {nat}
  ALT = {alt}
The translation is supposed to use ONLY the NAT style for this family (and standard American
English otherwise). Decision point = {opportunity}.

SPANISH SOURCE:
{es}

ENGLISH TRANSLATION:
{en}

Judge and answer in JSON only:
1. "coherent": true if the English flows as one coherent paragraph (no truncation, no
   contradictions, no disconnected sentences). Ignore length and typography.
2. "fluent": true if it is natural, grammatical English (minor stiffness is fine; false only for
   real errors, Spanish words left untranslated, or machine-like wording).
3. "faithful": true if it says the same things as the Spanish in the same order — no sentence
   added, dropped, or merged; numbers, percentages, ordinals, quotations, and lists preserved.
4. "style_consistent": true if every occurrence of this family's feature in the English is in the
   NAT form and none is in the ALT form.
5. "anchors": the exact English spans where the {name} choice is made (each occurrence, in order).
6. "notes": one short sentence, or "".

Reply with a single JSON object with keys "coherent","fluent","faithful","style_consistent","anchors","notes". No markdown."""


def judge_one(key, model, fam, rec):
    body = {"model": model, "temperature": 0.0, "max_tokens": 700,
            "messages": [{"role": "user", "content": PROMPT.format(
                name=fam.name, nat=fam.nat, alt=fam.alt, opportunity=fam.opportunity,
                es=rec["text_es"], en=rec["text_nat"])}]}
    for attempt in range(5):
        try:
            r = requests.post(URL, json=body, timeout=90, headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
            v, _ = json.JSONDecoder(strict=False).raw_decode(raw[raw.index("{"):])
            anchors = [str(a) for a in v.get("anchors", [])]
            return rec["doc_id"], {k: bool(v.get(k)) for k in ("coherent", "fluent", "faithful", "style_consistent")} | \
                {"k_found": len(anchors), "anchors": anchors, "notes": str(v.get("notes", ""))[:300], "judge": model}
        except Exception as e:
            if attempt == 4:
                print(f"{rec['doc_id']} JUDGE FAILED: {e}", flush=True)
                return rec["doc_id"], None
            time.sleep(2 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES])
    ap.add_argument("--model", default="anthropic/claude-haiku-4.5")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    key = load_key()
    data = {n: load_out(n) for n in args.families}
    jobs = []
    for n, recs in data.items():
        todo = [r for r in recs if "text_nat" in r and not r.get("verify_en")]
        jobs += [(FAMILY[n], r) for r in todo[: args.limit]]
    print(f"{len(jobs)} translations to judge with {args.model}", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_one, key, args.model, f, r): (f.name, r) for f, r in jobs}
        for fu in as_completed(futs):
            _, v = fu.result(); n, rec = futs[fu]; done += 1
            if v is not None:
                rec["verify_en"] = v
            if done % 500 == 0:
                for m in args.families:
                    save_out(m, data[m])
                print(f"{done}/{len(jobs)}", flush=True)
    for n in args.families:
        save_out(n, data[n])
    print(f"\n{'family':14s} {'judged':>6s} {'coh':>4s} {'flu':>4s} {'faith':>5s} {'style':>5s} {'all4':>5s}")
    for n, recs in data.items():
        ver = [r["verify_en"] for r in recs if r.get("verify_en")]
        if not ver:
            continue
        print(f"{n:14s} {len(ver):6d} {sum(v['coherent'] for v in ver):4d} {sum(v['fluent'] for v in ver):4d} "
              f"{sum(v['faithful'] for v in ver):5d} {sum(v['style_consistent'] for v in ver):5d} "
              f"{sum(all(v[k] for k in ('coherent','fluent','faithful','style_consistent')) for v in ver):5d}")


if __name__ == "__main__":
    main()
