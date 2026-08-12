#!/usr/bin/env python3
"""Generator for compound_join (spec index 99).

Rule: given two words that form a standard closed compound (space
separated), output the compound written as one word, e.g.
"tooth brush" -> "toothbrush".

Reuses the same curated (compound, part1, part2) triples and the same
seed-42 selection as compound_head.py / compound_first.py (per spec:
"same curated compound list, inverted").
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from compound_data import build_compound_pairs  # noqa: E402

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "compound_join.json")
N = 1000

random.seed(42)

triples = build_compound_pairs()
print(f"Verified curated triples available: {len(triples)}")
assert len(triples) >= N, f"only {len(triples)} verified triples, need >= {N}"

random.shuffle(triples)
selected = triples[:N]

examples = [
    {"input": f"{part1} {part2}", "output": compound} for compound, part1, part2 in selected
]

# ---- checks ----
assert len(examples) == N
inputs = [e["input"] for e in examples]
assert len(set(inputs)) == N, "inputs not unique"
for e in examples:
    assert e["input"] == e["input"].strip()
    assert e["output"] == e["output"].strip()
    assert len(e["input"].split()) == 2
    p1, p2 = e["input"].split()
    assert p1.isalpha() and p2.isalpha()
    assert e["output"].isalpha()
    # rule self-check: concatenating the two input words gives the output
    assert p1 + p2 == e["output"]

output_vocab = len(set(e["output"] for e in examples))
print(f"Generated {len(examples)} examples, output vocab size {output_vocab}")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(examples, f, indent=2)
print(f"Wrote {OUT_PATH}")
