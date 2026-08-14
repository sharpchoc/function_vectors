#!/usr/bin/env python3
"""Generator for syllable_count task.

Rule: word -> its number of syllables.

Method: cmudict.dict() pronunciations; a word's syllable count is the number
of digit-suffixed phones (stress markers 0/1/2) in a pronunciation. Only
words where ALL listed pronunciations agree on the count are kept (gold
labels are otherwise ambiguous). Words are sourced from common_words.txt
(frequency-sorted descending), filtered to alphabetic length>=3, and the
most frequent words within each syllable-count class (1-5) are taken to
build a 1000-example set with no class exceeding 40% of the total.
"""
import json
import random

import cmudict

RESOURCES = "dataset_files/extended_tasks/_resources"
OUT_PATH = "dataset_files/extended_tasks/syllable_count.json"

N = 1000
# Per-class quotas: sum to N, max class 300 (30%) <= 40% cap. Counts 1-5 only
# (6-syllable words exist in the source but are dropped per spec: "include
# 1-5 syllables; it's fine if 5-syllable words are few").
TARGETS = {1: 250, 2: 300, 3: 250, 4: 150, 5: 50}
assert sum(TARGETS.values()) == N
assert max(TARGETS.values()) <= 0.4 * N


def syllable_count(pron):
    """Number of syllables in a single cmudict pronunciation (phone list)."""
    return sum(1 for phone in pron if phone[-1].isdigit())


def main():
    cmu = cmudict.dict()

    with open(f"{RESOURCES}/common_words.txt") as f:
        words = [line.strip().lower() for line in f if line.strip()]
    words = [w for w in words if w.isalpha() and len(w) >= 3]
    print(f"Candidate words after alpha/len filter: {len(words)}")

    # Walk the frequency-sorted list once, keeping first occurrence, and
    # bucket unambiguous words by syllable count (preserves frequency order
    # within each bucket).
    seen = set()
    buckets = {c: [] for c in range(1, 6)}
    n_unambiguous_total = 0
    for w in words:
        if w in seen:
            continue
        seen.add(w)
        prons = cmu.get(w)
        if not prons:
            continue
        counts = {syllable_count(p) for p in prons}
        if len(counts) != 1:
            continue  # pronunciations disagree on syllable count -> ambiguous
        n_unambiguous_total += 1
        c = counts.pop()
        if c in buckets:
            buckets[c].append(w)

    print(f"Total unambiguous words (all counts): {n_unambiguous_total}")
    for c in sorted(buckets):
        print(f"  count={c}: {len(buckets[c])} available")

    examples = []
    for c, target in TARGETS.items():
        available = buckets[c]
        assert len(available) >= target, (
            f"count={c} needs {target} words, only {len(available)} available"
        )
        # Most frequent words within this class (buckets already in
        # frequency order since the source list is frequency-sorted).
        chosen = available[:target]
        for w in chosen:
            examples.append({"input": w, "output": str(c)})

    print(f"Built {len(examples)} examples before final checks")

    random.seed(42)
    random.shuffle(examples)

    # Assertions
    assert len(examples) == N, f"Expected {N} examples, got {len(examples)}"
    inputs = [ex["input"] for ex in examples]
    assert len(set(inputs)) == N, "Duplicate inputs found"
    for ex in examples:
        assert ex["input"] != ex["output"]
        assert ex["input"] == ex["input"].strip()
        assert ex["output"] == ex["output"].strip()
        assert ex["input"].isalpha() and len(ex["input"]) >= 3

    # Re-check gold labels directly against cmudict (independent of the
    # bucket-building logic above).
    for ex in examples:
        prons = cmu[ex["input"]]
        counts = {syllable_count(p) for p in prons}
        assert len(counts) == 1, f"'{ex['input']}' pronunciations disagree"
        assert str(counts.pop()) == ex["output"], (
            f"Label mismatch for '{ex['input']}': expected {ex['output']}"
        )

    final_dist = {}
    for ex in examples:
        final_dist[ex["output"]] = final_dist.get(ex["output"], 0) + 1
    print(f"Final distribution: {dict(sorted(final_dist.items()))}")

    with open(OUT_PATH, "w") as f:
        json.dump(examples, f, indent=1)
    print(f"Wrote {len(examples)} examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
