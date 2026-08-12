#!/usr/bin/env python3
"""Generator for noun_possessive task.

Rule: given a singular noun, output its possessive form (noun + "'s").
Recipe: for each alphabetic noun in common_nouns.txt that does not end in
's' (avoids the s'/s's style variation), output n + "'s". We additionally
drop pure function words / pronouns (that, which, who, ...) via the shared
STOPWORDS content-word filter, since common_nouns.txt is a loosely
POS-tagged frequency list and a possessive of a pronoun/determiner (e.g.
"that's", "which's") is not a real English possessive noun form.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _filter_content_words import STOPWORDS

HERE = os.path.dirname(os.path.dirname(__file__))
OUT_PATH = os.path.join(os.path.dirname(HERE), "noun_possessive.json")


def generate():
    with open(os.path.join(HERE, "common_nouns.txt")) as f:
        nouns = [line.strip() for line in f if line.strip()]

    pool = [
        n for n in nouns
        if n.isalpha() and n.islower() and len(n) >= 3
        and not n.endswith("s") and n not in STOPWORDS
    ]
    pool = sorted(set(pool), key=pool.index)  # de-dup, keep first occurrence order
    print(f"domain size: {len(pool)}")

    random.seed(42)
    sample = random.sample(pool, 1000)

    dataset = [{"input": n, "output": n + "'s"} for n in sample]

    random.seed(42)
    random.shuffle(dataset)

    seen_inputs = set()
    for item in dataset:
        n, poss = item["input"], item["output"]
        assert n not in seen_inputs
        seen_inputs.add(n)
        assert n == n.strip() and poss == poss.strip()
        assert poss == n + "'s"

    assert len(dataset) == 1000
    assert len(set(d["input"] for d in dataset)) == 1000
    return dataset


if __name__ == "__main__":
    dataset = generate()
    with open(OUT_PATH, "w") as f:
        json.dump(dataset, f)
    print(f"wrote {len(dataset)} examples to {OUT_PATH}")
