#!/usr/bin/env python3
"""Generator for plural_to_singular task.

Rule: given a plural noun, output its singular form.
Recipe: invert singular->plural over common_nouns.txt via lemminflect
getInflection(n, 'NNS'). Keep plurals that map to exactly one singular and
differ from it (drops invariants like 'sheep'). Additionally drop function
words / pronouns via the shared STOPWORDS content-word filter, since
common_nouns.txt is a loosely POS-tagged frequency list and includes
non-noun high-frequency words (e.g. 'that', 'which') whose lemminflect
"plurals" (e.g. 'thats') are not real English words.
"""
import json
import os
import random
import sys

from lemminflect import getInflection

sys.path.insert(0, os.path.dirname(__file__))
from _filter_content_words import STOPWORDS

HERE = os.path.dirname(os.path.dirname(__file__))
OUT_PATH = os.path.join(os.path.dirname(HERE), "plural_to_singular.json")


def generate():
    with open(os.path.join(HERE, "common_nouns.txt")) as f:
        nouns = [line.strip() for line in f if line.strip()]

    nouns = [
        n for n in nouns
        if n.isalpha() and n.islower() and len(n) >= 3 and n not in STOPWORDS
    ]

    plural_to_singulars = {}
    for n in nouns:
        infl = getInflection(n, "NNS")
        if not infl:
            continue
        p = infl[0]
        if p == n:
            continue
        plural_to_singulars.setdefault(p, set()).add(n)

    pairs = [
        (p, next(iter(s)))
        for p, s in plural_to_singulars.items()
        if len(s) == 1
    ]
    print(f"domain size (unique plural->singular): {len(pairs)}")

    random.seed(42)
    random.shuffle(pairs)
    pairs = pairs[:1000]

    dataset = [{"input": p, "output": s} for p, s in pairs]

    random.seed(42)
    random.shuffle(dataset)

    # self-check: re-derive via lemminflect, and structural checks
    seen_inputs = set()
    for item in dataset:
        p, s = item["input"], item["output"]
        assert p not in seen_inputs
        seen_inputs.add(p)
        assert p == p.strip() and s == s.strip()
        infl = getInflection(s, "NNS")
        assert infl and infl[0] == p, (p, s, infl)

    assert len(dataset) == 1000
    assert len(set(d["input"] for d in dataset)) == 1000
    return dataset


if __name__ == "__main__":
    dataset = generate()
    with open(OUT_PATH, "w") as f:
        json.dump(dataset, f)
    print(f"wrote {len(dataset)} examples to {OUT_PATH}")
