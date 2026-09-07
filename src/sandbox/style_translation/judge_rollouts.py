#!/usr/bin/env python
"""Step 3 — faithfulness/coherence judge for GPT-J completions (Gemini 2.5 Flash via OpenRouter,
ONE rollout per call; user: "use a Gemini model to check the translation still makes sense").

The judge sees the Spanish source, the English translation so far (last 500 chars), the reference
continuation (our twin's own next sentence), the model's completion cut at the sentence end, and
the `capped` flag (cap hit before a sentence end -> a truncated completion must not be penalised).
It answers OK / NOT OK on: coherent English that faithfully renders what comes next in the
Spanish (paraphrase fine), ignoring case/spelling/spacing/quotes/dashes/number format (style is
scored deterministically elsewhere). Verdict written back into the rollout record (`judge`).
Idempotent; resumable.
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
from src.utils.paths import ARTIFACTS_ROOT
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.gen_spanish import load_key, URL

ROLL = ARTIFACTS_ROOT / "style_translation" / "rollouts"

PROMPT = """A language model is translating a Spanish paragraph into English. It has produced the English
up to a point and then generated a COMPLETION. Decide whether the completion is acceptable as a
continuation of the translation.

Answer OK if the completion is coherent English that faithfully renders what comes next in the
Spanish (the REFERENCE shows one correct rendering; a paraphrase with the same meaning is fine;
the completion may be shorter than the reference). IGNORE completely: upper/lower case, American
vs British spelling, single vs double spaces, straight vs curly quotes, hyphens vs dashes, digits vs
words, "%" vs "percent", ampersands, contractions, comma placement.{capnote}

Answer NOT OK if the completion is in Spanish, restarts or repeats the passage, introduces content
that is not in the source at this point, begins an unrelated text or a new heading, contradicts the
source, or is incoherent / gibberish.

SPANISH SOURCE:
{es}

ENGLISH SO FAR (end): ...{ctx}

REFERENCE (one correct continuation): {ref}

COMPLETION: {tail}

Reply with a single JSON object: {{"ok": true or false, "notes": "<one short sentence>"}}"""

CAPNOTE = (" NOTE: this completion was CUT OFF by a token limit before the sentence ended; it may stop"
           " mid-sentence or mid-word. Do NOT mark it NOT OK for being truncated or incomplete.")


def judge_one(key, model, r):
    if not r["tail"].strip():
        return r["doc_id"], r["style"], r["k"], {"ok": False, "notes": "empty completion", "judge": model}
    body = {"model": model, "temperature": 0.0, "max_tokens": 120,
            "messages": [{"role": "user", "content": PROMPT.format(
                capnote=CAPNOTE if r["capped"] else "", es=r["es_text"], ctx=r["context_tail"],
                ref=json.dumps(r["ref_sentence"]), tail=json.dumps(r["tail"]))}]}
    for attempt in range(5):
        try:
            resp = requests.post(URL, json=body, timeout=90, headers={"Authorization": f"Bearer {key}"})
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
            v, _ = json.JSONDecoder(strict=False).raw_decode(raw[raw.index("{"):])
            return r["doc_id"], r["style"], r["k"], {"ok": bool(v.get("ok")), "notes": str(v.get("notes", ""))[:200], "judge": model}
        except Exception as e:
            if attempt == 4:
                print(f"{r['doc_id']}/{r['style']}/{r['k']} JUDGE FAILED: {e}", flush=True)
                return r["doc_id"], r["style"], r["k"], None
            time.sleep(2 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES])
    ap.add_argument("--model", default="google/gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    key = load_key()
    for fam in args.families:
        f = ROLL / f"{fam}.json"
        if not f.exists():
            continue
        recs = json.load(open(f))
        todo = [r for r in recs if r.get("judge") is None][: args.limit]
        if not todo:
            print(f"{fam}: already judged", flush=True); continue
        idx = {(r["doc_id"], r["style"], r["k"]): r for r in recs}
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(judge_one, key, args.model, r) for r in todo]
            for fu in as_completed(futs):
                d, s, k, v = fu.result(); done += 1
                if v is not None:
                    idx[(d, s, k)]["judge"] = v
                if done % 1000 == 0:
                    json.dump(recs, open(f, "w"), ensure_ascii=False)
        json.dump(recs, open(f, "w"), ensure_ascii=False)
        ok = [r for r in recs if r.get("judge")]
        print(f"{fam}: judged {len(todo)} | ok rate {sum(r['judge']['ok'] for r in ok)/max(len(ok),1):.3f} "
              f"| fails {sum(r.get('judge') is None for r in recs)}", flush=True)


if __name__ == "__main__":
    main()
