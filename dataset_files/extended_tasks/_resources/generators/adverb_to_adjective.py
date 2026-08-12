#!/usr/bin/env python3
"""Generator for adverb_to_adjective task.

Rule: given an -ly adverb, output the adjective it is derived from.
Recipe: invert the validated (adjective, adverb) pairs from
adjective_to_adverb (see _ly_derivation_pairs.py); keep only adverbs that
map to exactly one adjective (drop collisions, e.g. 'academically' <-
{'academic', 'academical'}).
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _ly_derivation_pairs import build_validated_pairs, derive_adverb

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "adverb_to_adjective.json",
)


def generate():
    pairs = build_validated_pairs()

    adverb_to_adjs = {}
    for adj, adv in pairs:
        adverb_to_adjs.setdefault(adv, set()).add(adj)

    unique_pairs = [
        (adv, next(iter(adjs)))
        for adv, adjs in adverb_to_adjs.items()
        if len(adjs) == 1
    ]
    print(f"domain size (unique adverb->adjective): {len(unique_pairs)}")

    random.seed(42)
    random.shuffle(unique_pairs)
    unique_pairs = unique_pairs[:1000]

    dataset = [{"input": adv, "output": adj} for adv, adj in unique_pairs]

    random.seed(42)
    random.shuffle(dataset)

    seen_inputs = set()
    for item in dataset:
        adv, adj = item["input"], item["output"]
        assert adv not in seen_inputs
        seen_inputs.add(adv)
        assert adv == adv.strip() and adj == adj.strip()
        # self-check: re-deriving the adverb from the adjective must give
        # back exactly this adverb.
        assert derive_adverb(adj) == adv

    assert len(dataset) == 1000
    assert len(set(d["input"] for d in dataset)) == 1000
    return dataset


if __name__ == "__main__":
    dataset = generate()
    with open(OUT_PATH, "w") as f:
        json.dump(dataset, f)
    print(f"wrote {len(dataset)} examples to {OUT_PATH}")
