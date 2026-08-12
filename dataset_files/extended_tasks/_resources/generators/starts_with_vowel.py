#!/usr/bin/env python3
"""Generator for starts_with_vowel: word -> 'vowel' if its first letter is one of
a,e,i,o,u, else 'consonant'. NOTE: 'y' counts as a consonant.

Recipe: common_words.txt (frequency-sorted descending), ASCII alphabetic lowercase
words len>=3. Balanced ~50/50: take the 500 most frequent vowel-initial words and
the 500 most frequent consonant-initial words. Dedup. Inputs unique across classes
by construction (a word starts with exactly one letter).
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent
OUT = RES.parent / "starts_with_vowel.json"

VOWELS = set("aeiou")  # 'y' is a consonant for this task

words = [w.strip() for w in (RES / "common_words.txt").read_text().splitlines()]

seen = set()
vowel_pairs = []
cons_pairs = []
for w in words:  # frequency order: most common first
    if len(w) < 3:
        continue
    if not (w.isascii() and w.isalpha() and w.islower()):
        continue
    if w in seen:
        continue
    seen.add(w)
    if w[0] in VOWELS:
        if len(vowel_pairs) < 500:
            vowel_pairs.append({"input": w, "output": "vowel"})
    else:
        if len(cons_pairs) < 500:
            cons_pairs.append({"input": w, "output": "consonant"})
    if len(vowel_pairs) == 500 and len(cons_pairs) == 500:
        break

pairs = vowel_pairs + cons_pairs
random.seed(42)
random.shuffle(pairs)

# Asserts
assert len(pairs) == 1000, len(pairs)
inputs = [p["input"] for p in pairs]
assert len(set(inputs)) == 1000
n_vowel = sum(1 for p in pairs if p["output"] == "vowel")
assert n_vowel == 500, n_vowel  # exact 50/50 balance
for p in pairs:
    expect = "vowel" if p["input"][0] in VOWELS else "consonant"
    assert p["output"] == expect
    assert p["input"] == p["input"].strip()

OUT.write_text(json.dumps(pairs, indent=1))
print(f"wrote {OUT} n={len(pairs)} vocab={len(set(p['output'] for p in pairs))}")
