#!/usr/bin/env python3
"""Generator for verb_to_gerund: base verb -> its -ing (VBG) gerund/participle form.

Recipe (spec idx 77): for each alphabetic verb v in common_verbs.txt, take
g = lemminflect.getInflection(v, 'VBG'). Keep (v, g[0]) if g is non-empty,
len(g) == 1 (drop multi-form outputs), and g[0] != v. Deduplicate inputs.

common_verbs.txt is frequency-sorted descending, so we take the first 1000
verbs (in file order) that pass the filter -- this both matches "prefer
well-known words" and gives deterministic, reproducible output.

Extra exclusions (quality, not in the literal mechanical recipe but licensed by
the task instructions to drop "anything ambiguous"): true modal auxiliaries
(can/could/may/.../ought) inflect in lemminflect's tables as if they were
regular content verbs (e.g. 'must' -> 'musting'), which is not real English;
and 'over', which is tagged VERB in lemminflect's static lemma dictionary but
has no standard verb use.
"""
import json
import os
import random

from lemminflect import getInflection
from wordfreq import zipf_frequency

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
OUT = os.path.join(os.path.dirname(RES), "verb_to_gerund.json")

MODAL_EXCLUDE = {
    "can", "could", "may", "might", "must", "shall", "should", "will",
    "would", "ought",
}
EXTRA_EXCLUDE = {"over"}
EXCLUDE = MODAL_EXCLUDE | EXTRA_EXCLUDE

TARGET_N = 1000
MIN_ZIPF = 1.5  # drops lemminflect dictionary artifacts with no real attested
# use as a gerund (e.g. 'candidate' -> 'candidating', zipf 0.0) while keeping
# genuine-but-less-common regular gerunds (e.g. 'steeling' zipf ~2.0)


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
        g = getInflection(v, "VBG")
        if not g or len(g) != 1:
            continue
        gv = g[0]
        if gv == v or not (gv.isalpha() and gv.islower()):
            continue
        if gv in seen:
            continue
        if zipf_frequency(gv, "en") < MIN_ZIPF:
            continue
        seen.add(v)
        seen.add(gv)
        data.append({"input": v, "output": gv})
        if len(data) >= TARGET_N:
            break

    assert len(data) == TARGET_N, f"only found {len(data)} valid verb pairs"

    random.seed(42)
    random.shuffle(data)

    # Self-checks: re-derive the gerund from the input and compare.
    assert len(data) == 1000
    inputs = [d["input"] for d in data]
    assert len(set(inputs)) == 1000, "duplicate inputs"
    for d in data:
        v, o = d["input"], d["output"]
        assert v == v.strip() and o == o.strip()
        regen = getInflection(v, "VBG")
        assert regen and regen[0] == o, (v, o, regen)

    with open(OUT, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {len(data)} to {OUT}")


if __name__ == "__main__":
    main()
