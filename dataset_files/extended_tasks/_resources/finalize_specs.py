#!/usr/bin/env python
"""Apply the curation decisions to the 126 ideation candidates -> exactly 100 final specs.

Drop reasons (indices into candidates.json order):
  intra-dup:      11, 12 (first/last_digit dup of 32/33), 121 (animal_class dup of 88)
  dup-of-existing/cross-lane dup: 13 (number_parity ~ parity_even_odd 42), 91 (~111 pos_label),
                  112 (~92 singular_or_plural), 93 (~113 verb_tense_label), 115 (~94
                  countable_uncountable), 117 (~79 living_nonliving), 118 (~81
                  concrete_abstract), 90 (~116 word_polarity), 87 (~120 hypernym_category)
  banned:         125 (name_gender - demographic inference)
  extractive-family overlap: 30 (alphabetically_first_word ~ alphabetically_first_3/5)
  determinism/difficulty cuts: 15 (starts_with_s trivial), 20 (word_length_parity),
                  23 (same_first_last_letter), 37 (digit_sum_small), 39 (reverse_digits),
                  46 (halve_even_digits), 52 (two_digit_sum_no_carry), 59 (year_to_century),
                  84 (is_edible fuzzy), 89 (adjective_category fuzzy), 95 (verb_physical_mental
                  fuzzy), 119 (verb_regularity_label metalinguistic)
"""
import json
from collections import Counter
from pathlib import Path

RES = Path(__file__).resolve().parent
DROP = {11, 12, 13, 15, 20, 23, 30, 37, 39, 46, 52, 59,
        84, 87, 89, 90, 91, 93, 95, 112, 115, 117, 118, 119, 121, 125}

specs = json.load(open(RES / "candidates.json"))["specs"]
final = [s for i, s in enumerate(specs) if i not in DROP]
assert len(final) == 100, len(final)

names = Counter(s["name"] for s in final)
assert all(c == 1 for c in names.values()), [n for n, c in names.items() if c > 1]
existing = {p.stem for p in (RES.parent.parent / "abstractive").glob("*.json")}
assert not (set(names) & existing), set(names) & existing
fams = Counter(s.get("family") for s in final if s.get("family"))
assert all(c <= 3 for c in fams.values()), {f: c for f, c in fams.items() if c > 3}

json.dump({"specs": final}, open(RES / "new_task_specs.json", "w"), indent=1)
lanes = Counter(s["lane"] for s in final)
meth = Counter(s["generation_method"] for s in final)
print(f"100 final specs -> new_task_specs.json")
print("by lane:", dict(lanes))
print("by method:", dict(meth))
print("names:", " ".join(sorted(names)))
