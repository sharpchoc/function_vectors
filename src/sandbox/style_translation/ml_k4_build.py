#!/usr/bin/env python
"""Cheap k = 4 check for the multilingual families (user decision 2026-09-13).
Base = the SAME 60 house-style English texts for every family (text_nat of whilst / oxford_comma / contractions, docs
t000–t019). Gemini 2.5 Flash translates each into the target language in the NATURAL style with the family's anchor
instruction; the text is normalised to the natural pole with the family's registry property, the ALTERNATIVE twin is
rendered deterministically (lexicon / rule / zhconv), and texts with ≥ 5 opportunities become pair records
(schema of dataset_files/style_translation/pairs/*.json, source stored in text_es, plus langs = {src, tgt}).
Then: cue_tokens.py --model qwen25_base, build_prompts.py --model qwen25_base --K 5, and the prompts are filtered to
k ∈ {0, 4} (k = 0 = free baseline). Raw translations cached in dataset_files/style_translation/multilingual/<fam>.json.
"""
import argparse, json, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import STYLE_TRANSLATION_DATA
from src.sandbox.style_translation.gen_spanish import load_key, URL
from src.sandbox.style_translation.ml_families import ML_FAMILIES, ML_FAMILY
from src.sandbox.style_translation.models import paths as model_paths

PAIRS = STYLE_TRANSLATION_DATA / "pairs"
RAW = STYLE_TRANSLATION_DATA / "multilingual"
BASE_FAMS = ("whilst", "oxford_comma", "contractions")
PY = sys.executable

PROMPT = """Rewrite the following English paragraph as ONE paragraph in {variety}. Keep the topic, the point of view and roughly the
length (120-180 words), but you may freely rephrase, reorder, add or drop details so that the text reads like it was originally
written in that language.
{instruction}
This anchor requirement is the point of the exercise: the paragraph MUST contain at least 7 of the required items (count them before answering).
Natural, fluent prose a native speaker would write; no title, no notes, no markdown, no quotation marks around the paragraph. Return only the paragraph.

English paragraph:
{text}"""

BACK = """Translate the following {lang} paragraph into natural, faithful {src} ({srcnote}). Translate every sentence;
do not add or omit content; one paragraph; no notes or markdown. Return only the translation.

{lang} paragraph:
{text}"""
SRCNOTE = {"English": "American spelling, plain style", "Spanish": "neutral standard Spanish, plain style"}


def base_texts(n_per=20):
    out = []
    for f in BASE_FAMS:
        recs = json.load(open(PAIRS / f"{f}.json"))[:n_per]
        out += [(r["doc_id"], r["text_nat"]) for r in recs]
    return out


def clean(t):
    t = t.strip().replace("\r", "")
    t = re.sub(r"</?\w+[^>]*>|\*\*|__", "", t)
    t = re.sub(r"^\s*(?:Translation|Traducción|Tradução|Traduction|Übersetzung|翻译|翻譳|翻訳)\s*[:：]\s*", "", t)
    t = re.sub(r"\s*\n\s*", " ", t)
    return t.strip().strip('"“”«»「」').strip()


def _chat(key, content, temperature, max_tokens=1200):
    body = {"model": "google/gemini-2.5-flash", "temperature": temperature, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}
    r = requests.post(URL, json=body, timeout=120, headers={"Authorization": f"Bearer {key}"}); r.raise_for_status()
    return clean(r.json()["choices"][0]["message"]["content"])


def translate_one(key, fam, doc_id, text):
    """Two steps (as in the original pipeline, languages swapped): anchor-rich target text in the natural style, then a faithful
    English back-translation = the SOURCE. Up to 3 tries to reach >= 5 opportunities."""
    best = None
    for attempt in range(5):
        try:
            tgt = _chat(key, PROMPT.format(variety=fam.variety_nat, instruction=fam.instruction, text=text), 0.9)
            if len(tgt) < 40:
                raise ValueError("too short")
            k = len(fam.prop.find_opps(to_nat(fam, tgt)))
            if best is None or k > best[1]:
                best = (tgt, k)
            if k >= 5 or attempt >= 2:
                src = _chat(key, BACK.format(lang=fam.tgt_lang, src=fam.src_lang, srcnote=SRCNOTE.get(fam.src_lang, "plain style"), text=best[0]), 0.3)
                if len(src) < 40:
                    raise ValueError("back-translation too short")
                return {"doc_id": f"{fam.name}__{doc_id}", "family": fam.name, "base_doc": doc_id, "text_es": src, "text_tgt_raw": best[0], "base_en": text}
        except Exception as e:
            if attempt == 4:
                print(f"{fam.name}/{doc_id} FAILED: {e}", flush=True); return None
            time.sleep(3 * (attempt + 1))


def to_nat(fam, text):
    p = fam.prop
    if hasattr(p, "to_nat"):
        return p.to_nat(text)
    opps = p.find_opps(text)
    out, pos = [], 0
    for o in opps:
        out.append(text[pos:o.start]); out.append(o.nat); pos = o.end
    out.append(text[pos:]); return "".join(out)


def twins(fam, text_nat):
    p = fam.prop
    opps = p.find_opps(text_nat)
    out, pos, spans, shift = [], 0, [], 0
    for o in opps:
        out.append(text_nat[pos:o.start]); a = o.start + shift
        out.append(o.alt); spans.append((a, a + len(o.alt))); shift += len(o.alt) - (o.end - o.start); pos = o.end
    out.append(text_nat[pos:]); text_alt = "".join(out)
    recs = [{"k": i, "nat": o.nat, "alt": o.alt, "nat_span": [o.start, o.end], "alt_span": list(spans[i])} for i, o in enumerate(opps)]
    for r in recs:
        assert text_nat[r["nat_span"][0]:r["nat_span"][1]] == r["nat"] and text_alt[r["alt_span"][0]:r["alt_span"][1]] == r["alt"]
    return text_alt, recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in ML_FAMILIES])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--min_opps", type=int, default=5)
    ap.add_argument("--ks", default="0,4")
    ap.add_argument("--skip_prompts", action="store_true")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    key = load_key(); bases = base_texts()
    for name in args.families:
        fam = ML_FAMILY[name]
        raw_path = RAW / f"{name}.json"
        raw = {r["doc_id"]: r for r in json.load(open(raw_path))} if raw_path.exists() else {}
        todo = [(d, t) for d, t in bases if f"{name}__{d}" not in raw]
        if todo:
            with ThreadPoolExecutor(args.workers) as ex:
                for fu in as_completed([ex.submit(translate_one, key, fam, d, t) for d, t in todo]):
                    r = fu.result()
                    if r: raw[r["doc_id"]] = r
            json.dump(sorted(raw.values(), key=lambda r: r["doc_id"]), open(raw_path, "w"), ensure_ascii=False, indent=0)
        pairs, ks = [], []
        for r in sorted(raw.values(), key=lambda r: r["doc_id"]):
            text_nat = to_nat(fam, r["text_tgt_raw"])
            text_alt, opps = twins(fam, text_nat); ks.append(len(opps))
            if len(opps) < args.min_opps:
                continue
            pairs.append({"doc_id": r["doc_id"], "family": name, "topic": r["base_doc"], "angle": "ml", "text_es": r["text_es"],
                          "langs": {"src": fam.src_lang, "tgt": fam.tgt_lang}, "text_nat": text_nat, "text_alt": text_alt, "opps": opps,
                          "k_en": len(opps), "pass": True, "verify": {"skipped": "cheap k=4 check"}})
        json.dump(pairs, open(PAIRS / f"{name}.json", "w"), ensure_ascii=False, indent=0)
        ks_sorted = sorted(ks)
        print(f"{name:14s} translated={len(raw):3d} opps median={ks_sorted[len(ks_sorted)//2] if ks_sorted else 0:3d} "
              f">= {args.min_opps}: {len(pairs):3d} pairs  | sample: {pairs[0]['text_nat'][:90] if pairs else '-'!r}", flush=True)
    if args.skip_prompts:
        return
    fams = [f for f in args.families if (PAIRS / f"{f}.json").exists() and json.load(open(PAIRS / f"{f}.json"))]
    subprocess.run([PY, "src/sandbox/style_translation/cue_tokens.py", "--model", "qwen25_base", "--families", *fams], check=True, cwd=_BOOT)
    subprocess.run([PY, "src/sandbox/style_translation/build_prompts.py", "--model", "qwen25_base", "--K", "5", "--families", *fams], check=True, cwd=_BOOT)
    keep = {int(k) for k in args.ks.split(",")}
    MP = model_paths("qwen25_base")
    for f in fams:
        p = MP["prompts"] / f"{f}.json"; items = [it for it in json.load(open(p)) if it["k"] in keep]
        json.dump(items, open(p, "w")); print(f"{f}: {len(items)} prompts kept (k in {sorted(keep)})")


if __name__ == "__main__":
    main()
