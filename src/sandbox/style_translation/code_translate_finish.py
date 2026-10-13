#!/usr/bin/env python
"""Finish the code-translation corpus (user 2026-10-14): for every unresolved document (code_translate.py output),
  (a) one-judge rejections get a TIE-BREAK by a third judge (GPT-5, same questionnaire): accepted on 2 of 3;
  (b) documents still rejected get up to --extra_rounds more repair rounds (previous code + both judges' notes), each followed by the two judges and,
      if exactly one rejects, the tie-break.
Records are updated in place (fields: tiebreak {ok,...}, passed_by 'both' | 'tiebreak', attempts, history)."""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.gen_spanish import load_key
from src.sandbox.style_translation.logprob_margin import PAIRS
from src.sandbox.style_translation.code_translate import STYLE, JUDGE, _chat, translate, judge, defs_match, syntax_check


def third_judge(key, model, rec, lang, code):
    v, err = _chat(key, model, JUDGE.format(lang_head=STYLE[lang].split("\n")[0].rstrip(":"), style=STYLE[lang], code=rec["text_nat"], translation=code), 8000, attempts=4, reasoning="medium")
    if not v:
        return {"ok": False, "notes": f"judge error: {err}"}
    qs = {q: bool(v.get(q)) for q in ("q1_definitions", "q2_logic", "q3_runnable", "q4_style")}
    return {"ok": all(qs.values()), **qs, "notes": str(v.get("notes", ""))[:400] if not all(qs.values()) else "none", "model": model}


def settle(key, args, judges, rec, lang, r, J):
    """Given two-judge verdicts J for r['code']: pass if both ok; if exactly one rejects, tie-break; returns (passed, feedback)."""
    oks = [j["ok"] for j in J.values()]
    if all(oks):
        r["judges"], r["passed"], r["passed_by"] = J, True, "both"; return True, ""
    if sum(oks) == 1 and "syntax" not in J:
        T = third_judge(key, args.third, rec, lang, r["code"]); r["tiebreak"] = T
        r["history"].append({"attempt": r["attempts"], "tiebreak": T})
        if T["ok"]:
            r["judges"], r["passed"], r["passed_by"] = J, True, "tiebreak"; return True, ""
        fb = "; ".join(f"{n}: {j['notes']}" for n, j in list(J.items()) + [("third", T)] if not j["ok"])
        return False, fb
    return False, "; ".join(f"{n}: {j['notes']}" for n, j in J.items() if not j["ok"])


def finish(key, args, judges, rec, lang, r):
    if r["passed"] or not r.get("code"):
        return r
    J = r.get("judges") or {}
    ok, feedback = settle(key, args, judges, rec, lang, r, J) if J else (False, r["history"][-1].get("feedback", "") if r["history"] else "")
    rounds = 0
    while not ok and rounds < args.extra_rounds:
        rounds += 1; r["attempts"] += 1
        code, defs, err = translate(key, args.model, rec, lang, feedback, previous=r["code"])
        if code is None:
            r["history"].append({"attempt": r["attempts"], "error": err}); continue
        if not defs_match(defs, rec["text_nat"]):
            feedback = "The translated definitions do not match the Python source definitions in order; translate every definition in order."
            r["history"].append({"attempt": r["attempts"], "fidelity_ok": False, "defs_model": defs, "feedback": feedback}); r["code"] = code; continue
        r["code"] = code; J = judge(key, judges, rec, lang, code)
        r["history"].append({"attempt": r["attempts"], "fidelity_ok": True, "judges": J})
        ok, feedback = settle(key, args, judges, rec, lang, r, J)
        r["history"][-1]["feedback"] = feedback
        if not ok:
            r["judges"] = J
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", required=True); ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--model", default="openai/gpt-5"); ap.add_argument("--third", default="openai/gpt-5")
    ap.add_argument("--judge_gemini", default="google/gemini-2.5-pro"); ap.add_argument("--judge_claude", default="anthropic/claude-sonnet-4.5")
    ap.add_argument("--extra_rounds", type=int, default=3); ap.add_argument("--workers", type=int, default=50)
    args = ap.parse_args(); key = load_key(); judges = {"gemini": args.judge_gemini, "claude": args.judge_claude}
    for fam in args.families:
        f = args.src / f"{fam}.json"; R = json.load(open(f)); recs = {x["doc_id"]: x for x in json.load(open(PAIRS / f"{fam}.json"))}
        todo = [r for r in R if not r["passed"]]; before = sum(r["passed"] for r in R)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            out = list(ex.map(lambda r: finish(key, args, judges, recs[r["doc_id"]], r["lang"], r), todo))
        by = {r["doc_id"]: r for r in out}; R = [by.get(r["doc_id"], r) for r in R]
        json.dump(R, open(f, "w"), ensure_ascii=False, indent=1)
        print(f"{fam}: passed {before} -> {sum(r['passed'] for r in R)}/{len(R)} (tiebreak {sum(r.get('passed_by') == 'tiebreak' for r in R)})", flush=True)
    print("finish done", flush=True)


if __name__ == "__main__":
    main()
