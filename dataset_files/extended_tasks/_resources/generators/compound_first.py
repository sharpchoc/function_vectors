#!/usr/bin/env python3
"""Generator for compound_first (spec index 98).

Rule: given a closed compound noun, output its first constituent word
(the modifier), e.g. toothbrush -> tooth.

Reuses the same curated (compound, part1, part2) triples and the same
seed-42 selection as compound_head.py, so both tasks are defined over the
identical 1000-compound domain (per spec: "same curated compound list as
compound_head").
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from compound_data import build_compound_pairs  # noqa: E402

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "compound_first.json")
N = 1000

random.seed(42)

triples = build_compound_pairs()
print(f"Verified curated triples available: {len(triples)}")
assert len(triples) >= N, f"only {len(triples)} verified triples, need >= {N}"

random.shuffle(triples)
selected = triples[:N]

examples = [{"input": compound, "output": part1} for compound, part1, part2 in selected]

# ---- checks ----
assert len(examples) == N
inputs = [e["input"] for e in examples]
assert len(set(inputs)) == N, "inputs not unique"
for e in examples:
    assert e["input"] == e["input"].strip()
    assert e["output"] == e["output"].strip()
    assert e["input"].isalpha()
    assert e["output"].isalpha()
    # rule self-check: output must be a prefix of input
    assert e["input"].startswith(e["output"])
    assert len(e["input"]) > len(e["output"])

# cross-check against the triples table (mechanical re-derivation)
triple_lookup = {c: (p1, p2) for c, p1, p2 in selected}
for e in examples:
    p1, p2 = triple_lookup[e["input"]]
    assert e["output"] == p1

output_vocab = len(set(e["output"] for e in examples))
print(f"Generated {len(examples)} examples, output vocab size {output_vocab}")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(examples, f, indent=2)
print(f"Wrote {OUT_PATH}")
