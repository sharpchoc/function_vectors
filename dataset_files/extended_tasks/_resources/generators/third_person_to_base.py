#!/usr/bin/env python3
"""Generator for third_person_to_base task (spec index 81).

Rule: given a verb's third-person singular present (-s) form, output its base
(infinitive) form.

Domain: dataset_files/extended_tasks/_resources/common_verbs.txt, frequency-sorted
descending. For each verb v compute s = getInflection(v, 'VBZ')[0]. Keep -s forms
that map back to exactly one base verb and where the -s form differs from the base
(so the task input is never trivially identical to its own output). Modal auxiliaries
(can, will, shall, may, must, ought) are excluded up front, since lemminflect returns
their suppletive modal past forms for VBD (used by sibling generators sharing this
base list) and their VBZ forms ('cans','wills',...) are not genuinely how those
modals are used -- excluded uniformly for consistency with the rest of the
verb_lemmatize / verb_form_conversion family.

Selection prefers the most frequent verbs (earliest in common_verbs.txt), then
shuffles with random.seed(42) for final example order.
"""
import json
import random
from collections import defaultdict
from lemminflect import getInflection

VERBS_PATH = 'dataset_files/extended_tasks/_resources/common_verbs.txt'
OUT_PATH = 'dataset_files/extended_tasks/third_person_to_base.json'

MODAL_EXCLUDE = {'can', 'will', 'shall', 'may', 'must', 'ought'}

# Same manually identified exclusions as gerund_to_base.py (spot-checked during
# that generator's construction; see its comment for full rationale): base verbs
# whose lemminflect-derived forms are either not real English words or so
# dominated by an unrelated, far more common reading that the mapping is
# misleading rather than merely rare.
AMBIGUOUS_BASE_EXCLUDE = {
    'product', 'self', 'safe', 'king', 'fair',
    'even', 'interest', 'weekend', 'article', 'league',
    'over', 'hot', 'middle',
}


def generate():
    with open(VERBS_PATH) as f:
        verbs = [line.strip() for line in f if line.strip()]
    verbs = [v for v in verbs if v not in MODAL_EXCLUDE and v not in AMBIGUOUS_BASE_EXCLUDE]

    vbz_to_bases = defaultdict(set)
    vbz_for_verb = {}
    for v in verbs:
        infl = getInflection(v, 'VBZ')
        if not infl:
            continue
        s = infl[0]
        vbz_for_verb[v] = s
        vbz_to_bases[s].add(v)

    unique_bases = {
        s: next(iter(b)) for s, b in vbz_to_bases.items()
        if len(b) == 1 and next(iter(b)) != s
    }

    seen = set()
    pairs = []
    for v in verbs:
        s = vbz_for_verb.get(v)
        if s is None or s in seen:
            continue
        if unique_bases.get(s) != v:
            continue
        seen.add(s)
        pairs.append((s, v))

    n_available = len(pairs)
    pairs = pairs[:1000]

    dataset = [{"input": s, "output": v} for s, v in pairs]

    random.seed(42)
    random.shuffle(dataset)

    for item in dataset:
        s, v = item["input"], item["output"]
        infl = getInflection(v, 'VBZ')
        assert infl and infl[0] == s, (v, s, infl)
        assert s != v
        assert unique_bases[s] == v

    assert len(dataset) == min(1000, n_available), (len(dataset), n_available)
    assert len(set(i["input"] for i in dataset)) == len(dataset)
    for item in dataset:
        assert item["input"] == item["input"].strip()
        assert item["output"] == item["output"].strip()

    print(f"domain available: {n_available}, final n: {len(dataset)}")
    return dataset


if __name__ == "__main__":
    dataset = generate()
    with open(OUT_PATH, 'w') as f:
        json.dump(dataset, f, indent=2)
