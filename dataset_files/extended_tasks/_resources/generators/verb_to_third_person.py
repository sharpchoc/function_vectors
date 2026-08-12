#!/usr/bin/env python3
"""Generator for verb_to_third_person: base verb -> 3rd-person singular present
(-s) form (VBZ).

Recipe (spec idx 78): for each alphabetic verb v in common_verbs.txt, take
f = lemminflect.getInflection(v, 'VBZ'). Keep (v, f[0]) if len(f) == 1 and
f[0] != v (this naturally excludes modals like 'can'/'may' when they truly
have no VBZ form -- lemminflect actually does inflect them mechanically, see
below). Deduplicate inputs.

common_verbs.txt is frequency-sorted descending, so we take the first 1000
verbs (in file order) that pass the filter.

Extra exclusions (same rationale as verb_to_gerund.py): true modal auxiliaries
inflect in lemminflect's tables as if they were regular content verbs (e.g.
'must' -> 'musts', 'can' -> 'cans' as a noun/verb pun rather than the modal),
which does not reflect real English usage of the modal; 'over' has no
standard verb use. A zipf-frequency floor on the produced form additionally
drops lemminflect dictionary artifacts with no real attested use (e.g. rare
denominal-verb entries).
"""
import json
import os
import random

from lemminflect import getInflection
from wordfreq import zipf_frequency

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
OUT = os.path.join(os.path.dirname(RES), "verb_to_third_person.json")

MODAL_EXCLUDE = {
    "can", "could", "may", "might", "must", "shall", "should", "will",
    "would", "ought",
}
EXTRA_EXCLUDE = {"over"}
EXCLUDE = MODAL_EXCLUDE | EXTRA_EXCLUDE

TARGET_N = 1000
MIN_ZIPF = 1.5


def main() -> None:
    with open(os.path.join(RES, "common_verbs.txt")) as f:
        verbs = [w.strip() for w in f if w.strip()]

    seen = set()
    data = []
    for v in verbs:  # frequency-sorted descending
        if not (v.isalpha() and v.islower() and len(v) >= 2):
            continue
        if v in EXCLUDE or v in seen:
            continue
        f = getInflection(v, "VBZ")
        if not f or len(f) != 1:
            continue
        fv = f[0]
        if fv == v or not (fv.isalpha() and fv.islower()):
            continue
        if fv in seen:
            continue
        if zipf_frequency(fv, "en") < MIN_ZIPF:
            continue
        seen.add(v)
        seen.add(fv)
        data.append({"input": v, "output": fv})
        if len(data) >= TARGET_N:
            break

    assert len(data) == TARGET_N, f"only found {len(data)} valid verb pairs"

    random.seed(42)
    random.shuffle(data)

    # Self-checks: re-derive the 3rd-person form from the input and compare.
    assert len(data) == 1000
    inputs = [d["input"] for d in data]
    assert len(set(inputs)) == 1000, "duplicate inputs"
    for d in data:
        v, o = d["input"], d["output"]
        assert v == v.strip() and o == o.strip()
        regen = getInflection(v, "VBZ")
        assert regen and regen[0] == o, (v, o, regen)

    with open(OUT, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {len(data)} to {OUT}")


if __name__ == "__main__":
    main()
