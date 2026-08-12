#!/usr/bin/env python
"""Build frequency-filtered POS word lists used by the extended_tasks generators.

Sources: `english_words` (web2 list) for the vocabulary, `wordfreq` for frequency
filtering, `lemminflect` lemma tables for POS assignment. Lists contain lowercase
alphabetic lemmas only, sorted by descending frequency.
"""
from pathlib import Path

import lemminflect
import wordfreq
from english_words import get_english_words_set

OUT = Path(__file__).resolve().parent
MIN_ZIPF = 3.0          # ~top 60k English words; common enough for GPT-J
MIN_LEN, MAX_LEN = 3, 12

words = sorted(
    (w for w in get_english_words_set(["web2"], lower=True)
     if w.isalpha() and w.isascii() and MIN_LEN <= len(w) <= MAX_LEN
     and wordfreq.zipf_frequency(w, "en") >= MIN_ZIPF),
    key=lambda w: -wordfreq.zipf_frequency(w, "en"),
)
print("filtered vocabulary:", len(words))

pos_lists = {"noun": [], "verb": [], "adj": []}
tag_of = {"noun": "NOUN", "verb": "VERB", "adj": "ADJ"}
for w in words:
    lemmas = lemminflect.getAllLemmas(w)
    for name, tag in tag_of.items():
        # keep only base-form lemmas so inflection tasks are well-defined
        if tag in lemmas and w in lemmas[tag]:
            pos_lists[name].append(w)

(OUT / "common_words.txt").write_text("\n".join(words[:12000]) + "\n")
for name, lst in pos_lists.items():
    (OUT / f"common_{name}s.txt").write_text("\n".join(lst[:6000]) + "\n")
    print(name, len(lst))
print("wrote lists to", OUT)
