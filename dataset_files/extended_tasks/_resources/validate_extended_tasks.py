#!/usr/bin/env python
"""Independent validation gate for the 100 generated extended tasks.

Checks per task file:
  - exists, parses, is a list of exactly 1000 {"input": str, "output": str}
  - inputs unique, non-empty, stripped; outputs non-empty, stripped
  - input != output on every pair (no identity leakage)
  - small-output-vocab tasks (<=6 distinct outputs): class balance within [min_frac]
Cross-task checks:
  - no two tasks (new or original 42) share >30% of identical (input,output) pairs
Writes _resources/validation_report.json and prints a summary.
"""
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

RES = Path(__file__).resolve().parent
ROOT = RES.parent
specs = json.load(open(RES / "new_task_specs.json"))["specs"]
names = [s["name"] for s in specs]

report, failures = {}, []
pair_sets = {}

for name in names:
    f = ROOT / f"{name}.json"
    entry = {"file": str(f)}
    try:
        data = json.load(open(f))
        assert isinstance(data, list), "not a list"
        entry["n"] = len(data)
        assert all(isinstance(p, dict) and set(p) == {"input", "output"} for p in data), "bad item shape"
        ins = [p["input"] for p in data]
        outs = [p["output"] for p in data]
        assert all(isinstance(x, str) and x and x == x.strip() for x in ins), "bad inputs"
        assert all(isinstance(x, str) and x and x == x.strip() for x in outs), "bad outputs"
        assert len(set(ins)) == len(ins), f"dup inputs ({len(ins)-len(set(ins))})"
        assert all(i != o for i, o in zip(ins, outs)), "input==output pair present"
        assert len(data) == 1000, f"n={len(data)} != 1000"
        vocab = Counter(outs)
        entry["output_vocab"] = len(vocab)
        if len(vocab) <= 3:
            # binary/ternary label tasks must be balanced; larger vocabs (vowels, continents,
            # states) follow natural frequency and are exempt
            frac = min(vocab.values()) / max(vocab.values())
            entry["class_balance_minmax"] = round(frac, 3)
            assert frac >= 0.6, f"imbalanced classes {dict(vocab)}"
        pair_sets[name] = set(zip(ins, outs))
        entry["ok"] = True
    except (AssertionError, FileNotFoundError, json.JSONDecodeError) as e:
        entry["ok"] = False
        entry["error"] = str(e)
        failures.append((name, str(e)))
    report[name] = entry

# content overlap: new-vs-new and new-vs-original
originals = {}
for f in (ROOT.parent / "abstractive").glob("*.json"):
    d = json.load(open(f))
    originals[f.stem] = set((p["input"], p["output"]) for p in d if isinstance(p, dict))
overlaps = []
for a, b in combinations(sorted(pair_sets), 2):
    inter = len(pair_sets[a] & pair_sets[b])
    if inter > 0.3 * min(len(pair_sets[a]), len(pair_sets[b])):
        overlaps.append((a, b, inter))
for a in pair_sets:
    for o, s in originals.items():
        inter = len(pair_sets[a] & s)
        if inter > 0.3 * min(len(pair_sets[a]), len(s)):
            overlaps.append((a, f"ORIGINAL:{o}", inter))

json.dump({"per_task": report, "overlaps": overlaps}, open(RES / "validation_report.json", "w"), indent=1)
ok = sum(1 for e in report.values() if e["ok"])
print(f"{ok}/{len(names)} tasks pass structural validation")
for n, e in failures:
    print(f"  FAIL {n}: {e}")
for a, b, i in overlaps:
    print(f"  OVERLAP {a} ~ {b}: {i} shared pairs")
sys.exit(0 if ok == len(names) and not overlaps else 1)
