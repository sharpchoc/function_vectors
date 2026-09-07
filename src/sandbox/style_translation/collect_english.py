#!/usr/bin/env python
"""Step 2c — decide pass/fail for each English translation, build the alt twin mechanically,
audit both twins, and finalize the paired corpus.

pass = k_en >= 5 and audit_nat == [] and Haiku: coherent & faithful (fluent/style_consistent recorded only).
Alt twin: text_alt = render(text_nat, PROPS[family].find_opps(text_nat), "alt"); the two twins
are identical outside the opportunity spans by construction; audited anyway.

--collect   recompute pass, write english_audit.csv, print tallies
--finalize  write pairs/<family>.json (Spanish record + English twins) with text_nat, text_alt, opps (k, nat/alt spans), k_en,
            verify_en for the passing records (first 200 by doc_id); report SHORT families
"""
import argparse
import csv
import json
import sys
from pathlib import Path

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import STYLE_TRANSLATION_DATA
from src.sandbox.ext_styleprops.properties import PROPS, render
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.translate_english import load_out, save_out, audit_nat

FINAL_DIR = STYLE_TRANSLATION_DATA / "final"     # Spanish (read-only here)
PAIRS_DIR = STYLE_TRANSLATION_DATA / "pairs"     # output of this step
TARGET = 200
# gating judge keys: coherent + faithful. `fluent` and `style_consistent` are recorded but not
# gating: style is judged deterministically (audit_nat / k_en; the judge hallucinated style
# inconsistencies, e.g. "periods outside quotes" on texts that have them inside), and most
# fluency flags are consequences of the fixed conventions ("At 1st", capitals after ...).
KEYS = ("coherent", "faithful")
INFO_KEYS = ("fluent", "style_consistent")


def decide(r):
    v = r.get("verify_en")
    return bool("text_nat" in r and r["k_en"] >= 5 and not r["audit_nat"] and v and all(v.get(k) for k in KEYS))


def build_twins(r):
    prop = PROPS[r["family"]]
    opps = prop.find_opps(r["text_nat"])
    text_nat, spans_nat = render(r["text_nat"], opps, "nat")
    text_alt, spans_alt = render(r["text_nat"], opps, "alt")
    assert text_nat == r["text_nat"], "nat render must be identity on a normalised text"
    problems = []
    # every rendered span must show the alt form (the all_caps alt twin cannot be re-detected:
    # an all-caps sentence has nat == alt, so the detector drops it — spans are the ground truth)
    for o, (a, b) in zip(opps, spans_alt):
        if text_alt[a:b] != o.alt:
            problems.append(f"alt twin surface not alt at {a}")
            break
    for o, (a, b) in zip(opps, spans_nat):
        if text_nat[a:b] != o.nat:
            problems.append(f"nat twin surface not nat at {a}")
            break
    if prop.name != "all_caps" and len(prop.find_opps(text_alt)) != len(opps):
        problems.append("alt twin opp count differs")
    # identical outside the spans
    def strip(t, spans):
        out, pos = [], 0
        for a, b in spans:
            out.append(t[pos:a]); pos = b
        out.append(t[pos:]); return "".join(out)
    if strip(text_nat, spans_nat) != strip(text_alt, spans_alt):
        problems.append("twins differ outside opportunity spans")
    return text_alt, [{"k": i, "nat": o.nat, "alt": o.alt, "nat_span": list(spans_nat[i]),
                       "alt_span": list(spans_alt[i])} for i, o in enumerate(opps)], problems


def collect():
    rows = []
    for fam in FAMILIES:
        recs = load_out(fam.name)
        for r in recs:
            r["pass"] = decide(r) if (r.get("verify_en") and "text_nat" in r) else None
        save_out(fam.name, recs)
        ver = [r for r in recs if r.get("verify_en")]
        rows.append(dict(
            family=fam.name, n=len(recs), verified=len(ver), n_pass=sum(bool(r["pass"]) for r in recs),
            fail_k=sum(1 for r in ver if r["k_en"] < 5), fail_audit=sum(1 for r in ver if r["audit_nat"]),
            fail_coherent=sum(1 for r in ver if not r["verify_en"]["coherent"]),
            fail_fluent=sum(1 for r in ver if not r["verify_en"]["fluent"]),
            fail_faithful=sum(1 for r in ver if not r["verify_en"]["faithful"]),
            fail_style=sum(1 for r in ver if not r["verify_en"]["style_consistent"]),
            rounds_max=max((r["rounds"] for r in recs), default=0),
            k_en_median=sorted(r["k_en"] for r in recs if "k_en" in r)[len(recs) // 2] if recs else 0))
    with open(STYLE_TRANSLATION_DATA / "english_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{'family':14s} {'n':>4s} {'verif':>5s} {'PASS':>5s} {'k<5':>4s} {'audit':>5s} {'!coh':>4s} {'!flu':>4s} {'!faith':>6s} {'!style':>6s} {'rounds':>6s}")
    for r in rows:
        print(f"{r['family']:14s} {r['n']:4d} {r['verified']:5d} {r['n_pass']:5d} {r['fail_k']:4d} {r['fail_audit']:5d} "
              f"{r['fail_coherent']:4d} {r['fail_fluent']:4d} {r['fail_faithful']:6d} {r['fail_style']:6d} {r['rounds_max']:6d}")


def finalize():
    short = {}
    for fam in FAMILIES:
        # Spanish records: the family's final set plus spare pass texts from the candidate pool
        es = {d["doc_id"]: d for d in json.load(open(STYLE_TRANSLATION_DATA / "spanish" / f"{fam.name}.json")) if d.get("pass")}
        es.update({d["doc_id"]: d for d in json.load(open(FINAL_DIR / f"{fam.name}.json"))})
        recs = [r for r in load_out(fam.name) if r.get("pass")]
        out, twin_problems = [], 0
        for r in sorted(recs, key=lambda r: r["doc_id"]):
            text_alt, opps, problems = build_twins(r)
            if problems:
                twin_problems += 1; continue
            base = dict(es[r["doc_id"]])
            base.update(text_nat=r["text_nat"], text_alt=text_alt, opps=opps, k_en=r["k_en"],
                        verify_en=r["verify_en"], rounds_en=r["rounds"])
            out.append(base)
            if len(out) == TARGET:
                break
        PAIRS_DIR.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(PAIRS_DIR / f"{fam.name}.json", "w"), indent=0, ensure_ascii=False)
        print(f"{fam.name:14s} pass={len(recs):4d} twin-problems={twin_problems:3d} final pairs={len(out):4d}")
        if len(out) < TARGET:
            short[fam.name] = TARGET - len(out)
    print("SHORT:", short if short else "none — all 17 families have 200 pairs")
    return short


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    a = ap.parse_args()
    if a.collect: collect()
    if a.finalize: finalize()
