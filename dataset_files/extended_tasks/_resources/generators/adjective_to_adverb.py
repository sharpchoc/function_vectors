#!/usr/bin/env python3
"""Generator for adjective_to_adverb task.

Rule: given an adjective, output the corresponding -ly adverb.
See _ly_derivation_pairs.py for the shared pool-building, derivation rule,
and manually-audited exclusion list.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _ly_derivation_pairs import build_validated_pairs, derive_adverb

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "adjective_to_adverb.json",
)


def generate():
    pairs = build_validated_pairs()
    print(f"domain size: {len(pairs)}")

    random.seed(42)
    random.shuffle(pairs)
    pairs = pairs[:1000]

    dataset = [{"input": a, "output": v} for a, v in pairs]

    random.seed(42)
    random.shuffle(dataset)

    seen_inputs = set()
    for item in dataset:
        a, v = item["input"], item["output"]
        assert a not in seen_inputs
        seen_inputs.add(a)
        assert a == a.strip() and v == v.strip()
        assert derive_adverb(a) == v

    assert len(dataset) == 1000
    assert len(set(d["input"] for d in dataset)) == 1000
    return dataset


if __name__ == "__main__":
    dataset = generate()
    with open(OUT_PATH, "w") as f:
        json.dump(dataset, f)
    print(f"wrote {len(dataset)} examples to {OUT_PATH}")
