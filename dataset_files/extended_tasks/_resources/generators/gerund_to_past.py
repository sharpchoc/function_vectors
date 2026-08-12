#!/usr/bin/env python3
"""Generator for gerund_to_past task (spec index 83).

Rule: given a verb's -ing (gerund) form, output the same verb's past tense.

Domain: dataset_files/extended_tasks/_resources/common_verbs.txt, frequency-sorted
descending. For each verb v, g = getInflection(v, 'VBG')[0] (gerund), p =
getInflection(v, 'VBD')[0] (past). Keep (g, p) if:
  - g maps back to exactly one base verb (the same gerund-uniqueness dict used by
    gerund_to_base / gerund_to_third_person),
  - VBD for v returns a single candidate form (no dived/dove-style ambiguity).
  (p == v is allowed for irregular zero-morph verbs like "put"/"read": the mapping
  is still fully deterministic.)
Modal auxiliaries (can, will, shall, may, must, ought) are excluded up front, since
lemminflect returns their suppletive modal past forms (could/would/might/...) which
are not genuinely "the past tense of the sense of the verb whose gerund is
canning/willing/maying/...".

Selection prefers the most frequent verbs (earliest in common_verbs.txt), then
shuffles with random.seed(42) for final example order.
"""
import json
import random
from collections import defaultdict
from lemminflect import getInflection

VERBS_PATH = 'dataset_files/extended_tasks/_resources/common_verbs.txt'
OUT_PATH = 'dataset_files/extended_tasks/gerund_to_past.json'

MODAL_EXCLUDE = {'can', 'will', 'shall', 'may', 'must', 'ought'}

# Manually identified base verbs (see gerund_to_base.py for full rationale) whose
# lemminflect-derived forms are either not real English words or so dominated by
# an unrelated, far more common reading that the mapping is misleading. Applied
# here too for consistency across the shared common_verbs.txt domain.
AMBIGUOUS_BASE_EXCLUDE = {
    'product', 'self', 'safe', 'king', 'fair',
    'even', 'interest', 'weekend', 'article', 'league',
    'over', 'hot', 'middle',
}


def generate():
    with open(VERBS_PATH) as f:
        verbs = [line.strip() for line in f if line.strip()]
    verbs = [v for v in verbs if v not in MODAL_EXCLUDE and v not in AMBIGUOUS_BASE_EXCLUDE]

    gerund_to_bases = defaultdict(set)
    gerund_for_verb = {}
    for v in verbs:
        infl = getInflection(v, 'VBG')
        if not infl:
            continue
        g = infl[0]
        gerund_for_verb[v] = g
        gerund_to_bases[g].add(v)

    unique_gerund_base = {
        g: next(iter(b)) for g, b in gerund_to_bases.items() if len(b) == 1
    }

    seen = set()
    pairs = []
    for v in verbs:
        g = gerund_for_verb.get(v)
        if g is None or g in seen:
            continue
        if unique_gerund_base.get(g) != v:
            continue
        pinfl = getInflection(v, 'VBD')
        if not pinfl or len(pinfl) != 1:
            continue
        p = pinfl[0]
        seen.add(g)
        pairs.append((g, p, v))

    n_available = len(pairs)
    pairs = pairs[:1000]

    dataset = [{"input": g, "output": p} for g, p, v in pairs]

    random.seed(42)
    random.shuffle(dataset)

    base_by_gerund = {g: v for g, p, v in pairs}
    for item in dataset:
        g, p = item["input"], item["output"]
        v = base_by_gerund[g]
        ginfl = getInflection(v, 'VBG')
        pinfl = getInflection(v, 'VBD')
        assert ginfl and ginfl[0] == g, (v, g, ginfl)
        assert pinfl and len(pinfl) == 1 and pinfl[0] == p, (v, p, pinfl)
        assert unique_gerund_base[g] == v

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
