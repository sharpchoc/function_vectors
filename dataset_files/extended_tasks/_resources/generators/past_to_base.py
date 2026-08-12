#!/usr/bin/env python3
"""Generator for past_to_base: past-tense verb form -> base (infinitive) form.

Recipe (spec idx 79): build dict past_form -> set(base verbs) over all
alphabetic v in common_verbs.txt using getInflection(v, 'VBD')[0]. Keep only
past forms that (a) map to exactly one base verb, (b) differ from the base
('put'-type zero-change excluded), (c) are single-form (drop bases whose VBD
result has >1 form, e.g. 'stay' -> ('stayed','staid')).

common_verbs.txt is frequency-sorted descending; among the valid past forms we
keep the ones whose BASE verb ranks earliest (most frequent/best-known) in the
file, for reproducible, high-quality output.

Extra exclusions: true modal auxiliaries (can/could, will/would, may/might,
shall/should) have historically real but synchronically confusing VBD pairs
('could' does not feel like "the past tense of can" to a modern reader/model)
-- excluded from the base-verb side. 'over' is not a standard verb. A
zipf-frequency floor on the past-tense form drops lemminflect dictionary
artifacts with no real attested use.
"""
import json
import os
import random

from lemminflect import getInflection
from wordfreq import zipf_frequency

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
OUT = os.path.join(os.path.dirname(RES), "past_to_base.json")

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

    rank = {v: i for i, v in enumerate(verbs)}

    # Build past_form -> set(bases), restricted to single-form VBD results.
    past_to_bases = {}
    base_of_past = {}
    for v in verbs:
        if not (v.isalpha() and v.islower() and len(v) >= 2):
            continue
        if v in EXCLUDE:
            continue
        vbd = getInflection(v, "VBD")
        if not vbd or len(vbd) != 1:
            continue
        past = vbd[0]
        if past == v or not (past.isalpha() and past.islower()):
            continue
        past_to_bases.setdefault(past, set()).add(v)

    candidates = []
    for past, bases in past_to_bases.items():
        if len(bases) != 1:
            continue  # ambiguous: multiple distinct bases share this past form
        base = next(iter(bases))
        if zipf_frequency(past, "en") < MIN_ZIPF:
            continue
        candidates.append((rank[base], base, past))

    candidates.sort(key=lambda t: t[0])  # most frequent base first

    seen = set()
    data = []
    for _, base, past in candidates:
        if base in seen or past in seen:
            continue
        seen.add(base)
        seen.add(past)
        data.append({"input": past, "output": base})
        if len(data) >= TARGET_N:
            break

    assert len(data) == TARGET_N, f"only found {len(data)} valid pairs"

    random.seed(42)
    random.shuffle(data)

    # Self-checks: re-derive the base's VBD form and confirm it maps back.
    assert len(data) == 1000
    inputs = [d["input"] for d in data]
    assert len(set(inputs)) == 1000, "duplicate inputs"
    for d in data:
        past, base = d["input"], d["output"]
        assert past == past.strip() and base == base.strip()
        regen = getInflection(base, "VBD")
        assert regen and regen[0] == past, (past, base, regen)

    with open(OUT, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {len(data)} to {OUT}")


if __name__ == "__main__":
    main()
