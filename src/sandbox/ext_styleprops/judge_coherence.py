#!/usr/bin/env python
"""LLM-judge coherence of stored steering rollouts (user spec 2026-09-02).

For every stored (condition, item) in a full-mode steering JSON, the judge sees the last
~60 tokens of the prompt and the sampled continuation and answers COHERENT / INCOHERENT.
Coherent = fluent English that plausibly continues the passage. The judge is told to
IGNORE spelling / punctuation / capitalisation / number-format conventions (those are the
manipulated variable). Verdicts are written back into the JSON as "coherent": [bool|None].

Usage: python judge_coherence.py --dir artifacts/style_properties/steering/full_cue_cue1
       [--conds baseline_nat2alt cuediff_cue_nat2alt ...] (default: all conditions)
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.ext_styleprops.gen_corpus import load_key

URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT = """Decide whether a text fragment is FLUENT ENGLISH or GIBBERISH.

The fragment is a language model's continuation of a passage. It was cut off after a fixed
number of tokens, so it almost always ends mid-sentence or mid-word.

Answer GIBBERISH only if the text itself is broken: word salad, random characters, repeated
tokens, collapsed syntax, strings of unrelated fragments.

Answer FLUENT for everything else. In particular these are ALL FLUENT:
  - cut off mid-word or mid-sentence ("...development at inap")
  - changes topic, or does not follow on from the passage
  - factually wrong, odd, or dull content
  - unusual spelling, ALL CAPS, all lowercase, missing or doubled punctuation, curly quotes,
    numbers as words or digits, stray markup or encoding artefacts
  - lists, headings, or a new paragraph

Examples:
  " Simply fill little dents and scrapes with wood filler, let it dry, then paint the fen" -> FLUENT
  " THE CRAFT OF HIKING AS A M" -> FLUENT
  " i will use a small" -> FLUENT
  " haneen kam ymont i idseller" -> GIBBERISH
  " OURSORT, the SouTHWESTERN BLTHOW IT WORK" -> GIBBERISH

PASSAGE (end of prompt): ...{ctx}

FRAGMENT TO JUDGE: {tail}

Reply with exactly one word: FLUENT or GIBBERISH."""


def judge_one(key, model, ctx, tail):
    body = {"model": model, "temperature": 0, "max_tokens": 8,
            "messages": [{"role": "user", "content": PROMPT.format(ctx=ctx[-300:], tail=tail)}]}
    for attempt in range(4):
        try:
            r = requests.post(URL, json=body, timeout=60, headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            ans = r.json()["choices"][0]["message"]["content"].strip().upper()
            # prefix-tolerant: a truncated reply ("INCOHER") must still parse
            if "GIB" in ans:
                return False
            if "FLU" in ans:
                return True
            return None
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--conds", nargs="*", default=None)
    ap.add_argument("--model", default="google/gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    key = load_key()
    for f in sorted(args.dir.glob("*.json")):
        d = json.load(open(f))
        conds = [c for c in d["conditions"] if d["conditions"][c].get("tails")
                 and (not args.conds or any(c.startswith(pfx) for pfx in args.conds))]
        jobs = []
        for c in conds:
            cond = d["conditions"][c]
            if cond.get("coherent") is not None and len(cond["coherent"]) == len(cond["tails"]):
                continue
            ctxs = d["ctx_alt"] if c.endswith("alt2nat") else d["ctx_nat"]
            cond["coherent"] = [None] * len(cond["tails"])
            for i, t in enumerate(cond["tails"]):
                jobs.append((c, i, ctxs[i], t))
        if not jobs:
            print(f"{d['property']}: already judged", flush=True)
            continue
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(judge_one, key, args.model, ctx, t): (c, i) for c, i, ctx, t in jobs}
            for fu in as_completed(futs):
                c, i = futs[fu]
                d["conditions"][c]["coherent"][i] = fu.result()
        for c in conds:
            v = d["conditions"][c]["coherent"]
            print(f"{d['property']} | {c}: coherent={sum(x is True for x in v)/max(len(v),1):.2f} "
                  f"(judge-fail {sum(x is None for x in v)})", flush=True)
        json.dump(d, open(f, "w"), indent=1)
    print("judging done", flush=True)


if __name__ == "__main__":
    main()
