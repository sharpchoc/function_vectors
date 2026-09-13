#!/usr/bin/env python
"""Judge the multilingual pilot rollouts (Gemini 2.5 Flash via OpenRouter): per output, is it in the target language,
faithful to the English source, and fluent/natural? Writes judged.json and prints the per-language / per-arm table."""
import json, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT
from src.sandbox.style_translation.gen_spanish import load_key, URL
D = ARTIFACTS_ROOT / "style_translation" / "multilingual_pilot"
LANGS = {"de": "German", "pt": "Portuguese", "fr": "French", "es": "Spanish", "nl": "Dutch", "ro": "Romanian"}
PROMPT = """You are grading a machine translation from English into {lang}.

English source:
{src}

Candidate {lang} translation (it may stop early; do not penalise incompleteness, judge what is there):
{out}

Answer with JSON only: {{"in_language": true/false (is the candidate written in {lang}?), "faithful": true/false (does what is there convey the source meaning without inventions or omissions of whole clauses?), "fluent": true/false (would a native speaker find it grammatical and natural?), "coverage": fraction 0-1 of the source content covered, "notes": "<15 words"}}"""

def judge(key, model, r):
    body = {"model": model, "temperature": 0.0, "max_tokens": 150, "messages": [{"role": "user", "content": PROMPT.format(lang=LANGS[r["lang"]], src=r["source"], out=r["output"] or "(empty)")}]}
    for attempt in range(5):
        try:
            resp = requests.post(URL, json=body, timeout=90, headers={"Authorization": f"Bearer {key}"}); resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip(); raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
            v, _ = json.JSONDecoder(strict=False).raw_decode(raw[raw.index("{"):]); return v
        except Exception as e:
            if attempt == 4: print("JUDGE FAILED", r["lang"], r["doc_id"], e, flush=True); return None
            time.sleep(2 * (attempt + 1))

def main():
    recs = json.load(open(D / "rollouts.json")); key = load_key(); model = "google/gemini-2.5-flash"
    with ThreadPoolExecutor(24) as ex:
        futs = {ex.submit(judge, key, model, r): r for r in recs}
        for f in as_completed(futs): futs[f]["judge"] = f.result()
    json.dump(recs, open(D / "judged.json", "w"), ensure_ascii=False, indent=1)
    print(f"{'lang':5s} {'arm':7s} {'n':>3s} {'in-lang':>8s} {'faithful':>9s} {'fluent':>7s} {'faith∧flu':>10s} {'coverage':>9s} {'empty':>6s}")
    for code in LANGS:
        for arm in ("header", "demo"):
            rs = [r for r in recs if r["lang"] == code and r["arm"] == arm and r.get("judge")]
            if not rs: continue
            g = lambda k: sum(bool(r["judge"].get(k)) for r in rs) / len(rs)
            both = sum(bool(r["judge"].get("faithful")) and bool(r["judge"].get("fluent")) for r in rs) / len(rs)
            cov = sum(float(r["judge"].get("coverage") or 0) for r in rs) / len(rs)
            empty = sum(not r["output"].strip() for r in rs) / len(rs)
            print(f"{code:5s} {arm:7s} {len(rs):3d} {g('in_language'):8.2f} {g('faithful'):9.2f} {g('fluent'):7.2f} {both:10.2f} {cov:9.2f} {empty:6.2f}")

if __name__ == "__main__":
    main()
