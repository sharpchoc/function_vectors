#!/usr/bin/env python
"""Merge Haiku-agent verdicts into the Spanish corpus and decide pass/fail.

Agents (one per family) read the candidate list written by `--prepare` and write
verdicts/<family>.json: [{doc_id, coherent: bool, fluent: bool, k_found: int, anchors: [str],
violations: [str], notes: str}].

pass = coherent and fluent and k_found >= 5 and regex_k >= 5 and no violations (agent or audit).
`--prepare` writes cands/<family>.jsonl (doc_id, text) for the unverified texts + rubric.md.
`--collect` merges verdicts, prints per-family tallies, and writes spanish_audit.csv.
`--finalize` keeps 200 passing, de-duplicated texts per family in final/<family>.json.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import STYLE_TRANSLATION_DATA
from src.sandbox.style_translation.families import FAMILIES, FAMILY, CONVENTIONS

SP_DIR = STYLE_TRANSLATION_DATA / "spanish"
FINAL_DIR = STYLE_TRANSLATION_DATA / "final"
TARGET = 200
MARKUP_RX = re.compile(r"\*\*|<b>|</b>|<i>|markdown|html|bold|formatting|asterisk|negrita", re.I)

RUBRIC = """You are verifying Spanish source texts for a style-translation study. Family: {name}.
When this Spanish is translated into English, the translator must choose between two styles:
  NAT = {nat}
  ALT = {alt}
Decision point = {opportunity}.
Anchors to list: {anchors_hint}.

For EVERY text in the candidate file, judge:
1. coherent: is it a coherent paragraph that flows and makes sense on its own (not a list of
   disconnected sentences, no contradictions, no truncation)?
2. fluent: is it natural, grammatical Spanish (minor stiffness is fine; machine-like errors,
   English words, or broken syntax are not)?
3. k_found: count the decision points precisely — list each anchor (the exact Spanish words or
   construction) in `anchors`, then k_found = len(anchors). Count only genuine cases where an
   English translation really faces the NAT/ALT choice. Do not count duplicates of the same
   token twice unless it occurs twice.
4. violations: any breach of the fixed Spanish conventions below (straight or curly double
   quotes, single-glyph ellipsis, spaced hyphen or en dash as aside, numbers written as words,
   ordinals as words, «por ciento», N% without space, double spaces, all-caps sentences).
{conventions}

Output: write a JSON array to the output path given to you, one object per text, in the same
order as the input, exactly with keys {{"doc_id","coherent","fluent","k_found","anchors",
"violations","notes"}}. `notes` is a short string (may be empty). No markdown, JSON only.
Work through the file in chunks if it is long; do not skip any text."""


def prepare(out_root):
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "cands").mkdir(exist_ok=True); (out_root / "verdicts").mkdir(exist_ok=True)
    (out_root / "rubrics").mkdir(exist_ok=True)
    for fam in FAMILIES:
        f = SP_DIR / f"{fam.name}.json"
        if not f.exists():
            continue
        recs = [r for r in json.load(open(f)) if r.get("verify") is None]
        with open(out_root / "cands" / f"{fam.name}.jsonl", "w") as fh:
            for r in recs:
                fh.write(json.dumps({"doc_id": r["doc_id"], "text": r["text_es"]}, ensure_ascii=False) + "\n")
        (out_root / "rubrics" / f"{fam.name}.md").write_text(RUBRIC.format(
            name=fam.name, nat=fam.nat, alt=fam.alt, opportunity=fam.opportunity,
            anchors_hint=fam.anchors_hint, conventions=CONVENTIONS))
        print(f"{fam.name}: {len(recs)} candidates -> {out_root / 'cands' / (fam.name + '.jsonl')}")


MIN_WORDS, MAX_WORDS = 100, 230   # length is judged deterministically (user: ~120-180 words)


def decide(r):
    """pass = Haiku says coherent & fluent & >=5 anchors, regex agrees (>=5), the deterministic
    audit finds no convention violation, and the length is in range. Agent-reported convention
    violations are NOT used (sub-agents hallucinated them; the audit is the authority)."""
    v = r.get("verify")
    return bool(v and v.get("coherent") and v.get("fluent") and int(v.get("k_found", 0)) >= 5
                and r["regex_k"] >= 5 and not r["violations"] and MIN_WORDS <= r["words"] <= MAX_WORDS)


def collect(out_root):
    rows = []
    for fam in FAMILIES:
        f = SP_DIR / f"{fam.name}.json"; vf = (out_root / "verdicts" / f"{fam.name}.json") if out_root else None
        if not f.exists():
            continue
        recs = json.load(open(f))
        if out_root is not None and vf.exists():
            raw = vf.read_text().strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
            verdicts = {v["doc_id"]: v for v in json.loads(raw)}
            for r in recs:
                if r["doc_id"] in verdicts:
                    r["verify"] = verdicts[r["doc_id"]]
        for r in recs:
            v = r.get("verify")
            if v and not decide(r) and not r["violations"] and MARKUP_RX.search(json.dumps(v, ensure_ascii=False)):
                # failure attributed to markup that the post-edit has since stripped -> re-verify
                r["verify"] = None
                r["reverify_reason"] = "markup"
            r["pass"] = decide(r) if r.get("verify") else None
        json.dump(sorted(recs, key=lambda r: r["doc_id"]), open(f, "w"), indent=0, ensure_ascii=False)
        ver = [r for r in recs if r.get("verify")]
        row = dict(family=fam.name, n=len(recs), verified=len(ver),
                   n_pass=sum(bool(r["pass"]) for r in recs),
                   fail_coherent=sum(1 for r in ver if not r["verify"].get("coherent")),
                   fail_fluent=sum(1 for r in ver if not r["verify"].get("fluent")),
                   fail_k_agent=sum(1 for r in ver if int(r["verify"].get("k_found", 0)) < 5),
                   fail_k_regex=sum(1 for r in ver if r["regex_k"] < 5),
                   fail_violation=sum(1 for r in ver if r["violations"]),
                   fail_length=sum(1 for r in ver if not (MIN_WORDS <= r["words"] <= MAX_WORDS)),
                   words_median=sorted(r["words"] for r in recs)[len(recs) // 2])
        rows.append(row)
    with open(STYLE_TRANSLATION_DATA / "spanish_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{'family':14s} {'n':>4s} {'verif':>5s} {'PASS':>5s} {'!coh':>4s} {'!flu':>4s} {'k<5 agent':>9s} {'k<5 regex':>9s} {'viol':>4s}")
    for r in rows:
        print(f"{r['family']:14s} {r['n']:4d} {r['verified']:5d} {r['n_pass']:5d} {r['fail_coherent']:4d} "
              f"{r['fail_fluent']:4d} {r['fail_k_agent']:9d} {r['fail_k_regex']:9d} {r['fail_violation']:4d}")
    return rows


def _norm(t):
    return re.sub(r"[^a-záéíóúñü ]", "", t.lower())


def _grams(t, n=5):
    w = _norm(t).split()
    return {" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))}


def finalize():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    short = {}
    for fam in FAMILIES:
        recs = [r for r in json.load(open(SP_DIR / f"{fam.name}.json")) if r.get("pass")]
        kept, seen_open, seen_grams = [], set(), []
        for r in sorted(recs, key=lambda r: r["doc_id"]):
            op = _norm(r["text_es"])[:60]
            g = _grams(r["text_es"])
            if op in seen_open or any(len(g & h) / max(len(g | h), 1) > 0.5 for h in seen_grams):
                continue
            kept.append(r); seen_open.add(op); seen_grams.append(g)
            if len(kept) == TARGET:
                break
        json.dump(kept, open(FINAL_DIR / f"{fam.name}.json", "w"), indent=0, ensure_ascii=False)
        print(f"{fam.name:14s} pass={len(recs):4d} unique kept={len(kept):4d}")
        if len(kept) < TARGET:
            short[fam.name] = TARGET - len(kept)
    print("SHORT:", short if short else "none — all selected families have 200")
    return short


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", type=Path, default=None, help="scratch dir for cands/ rubrics/ verdicts/ (agent route only)")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--families", nargs="*", default=None, help="restrict collect/finalize to these families (others untouched)")
    a = ap.parse_args()
    if a.families:
        FAMILIES[:] = [f for f in FAMILIES if f.name in a.families]
    if a.prepare: prepare(a.out_root)
    if a.collect and a.out_root is None: pass
    if a.collect: collect(a.out_root)
    if a.finalize: finalize()
