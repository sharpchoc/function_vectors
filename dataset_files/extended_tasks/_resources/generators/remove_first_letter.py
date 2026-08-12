#!/usr/bin/env python3
"""Generator for remove_first_letter: word -> word with its first letter deleted.

Recipe: common_words.txt (frequency-sorted descending), ASCII alphabetic lowercase
words of length 4-8. output = word[1:]. Keep the 1000 most frequent. Dedup.
Self-check: input[0] + output == input.
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent
OUT = RES.parent / "remove_first_letter.json"

words = [w.strip() for w in (RES / "common_words.txt").read_text().splitlines()]

seen = set()
pairs = []
for w in words:  # frequency order: most common first
    if not (4 <= len(w) <= 8):
        continue
    if not (w.isascii() and w.isalpha() and w.islower()):
        continue
    if w in seen:
        continue
    seen.add(w)
    pairs.append({"input": w, "output": w[1:]})
    if len(pairs) == 1000:
        break

random.seed(42)
random.shuffle(pairs)

# Asserts
assert len(pairs) == 1000, len(pairs)
inputs = [p["input"] for p in pairs]
assert len(set(inputs)) == 1000
for p in pairs:
    assert p["input"][0] + p["output"] == p["input"]
    assert p["input"] == p["input"].strip() and p["output"] == p["output"].strip()

OUT.write_text(json.dumps(pairs, indent=1))
print(f"wrote {OUT} n={len(pairs)} vocab={len(set(p['output'] for p in pairs))}")
