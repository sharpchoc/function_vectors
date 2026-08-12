#!/usr/bin/env python3
"""Generator for past_to_gerund task (spec index 82).

Rule: given a verb in past tense, output the same verb's -ing (gerund) form.

Domain: dataset_files/extended_tasks/_resources/common_verbs.txt, frequency-sorted
descending. For each verb v, p = getInflection(v, 'VBD')[0] (past), g =
getInflection(v, 'VBG')[0] (gerund). Keep (p, g) if:
  - p maps back to exactly one base verb (built the same way as the shared
    past-form uniqueness dict used across the verb_lemmatize family),
  - p != v (past form differs from base -- otherwise input carries no tense info),
  - VBD for v returns a single candidate form (no dived/dove-style ambiguity).
Modal auxiliaries (can, will, shall, may, must, ought) are excluded up front:
lemminflect maps them to suppletive modal pasts (could/would/might/...) whose
"gerund of the same verb" is undefined/wrong (e.g. getInflection('can','VBD') ->
'could', but 'could' is not genuinely the past tense of the sense of "can" whose
gerund is "canning").

Selection prefers the most frequent verbs (earliest in common_verbs.txt), then
shuffles with random.seed(42) for final example order.
"""
import json
import random
from collections import defaultdict
from lemminflect import getInflection

VERBS_PATH = 'dataset_files/extended_tasks/_resources/common_verbs.txt'
OUT_PATH = 'dataset_files/extended_tasks/past_to_gerund.json'

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

    # Shared past-form uniqueness dict (past_to_base construction), reused here.
    past_to_bases = defaultdict(set)
    past_for_verb = {}
    for v in verbs:
        infl = getInflection(v, 'VBD')
        if not infl:
            continue
        past_for_verb[v] = infl
        p = infl[0]
        past_to_bases[p].add(v)

    unique_past_base = {
        p: next(iter(b)) for p, b in past_to_bases.items() if len(b) == 1
    }

    seen = set()
    pairs = []
    for v in verbs:
        infl = past_for_verb.get(v)
        if infl is None or len(infl) != 1:
            continue
        p = infl[0]
        if p == v or p in seen:
            continue
        if unique_past_base.get(p) != v:
            continue
        ginfl = getInflection(v, 'VBG')
        if not ginfl or len(ginfl) != 1:
            continue
        g = ginfl[0]
        seen.add(p)
        pairs.append((p, g, v))

    n_available = len(pairs)
    pairs = pairs[:1000]

    dataset = [{"input": p, "output": g} for p, g, v in pairs]

    random.seed(42)
    random.shuffle(dataset)

    base_by_past = {p: v for p, g, v in pairs}
    for item in dataset:
        p, g = item["input"], item["output"]
        v = base_by_past[p]
        pinfl = getInflection(v, 'VBD')
        ginfl = getInflection(v, 'VBG')
        assert pinfl and len(pinfl) == 1 and pinfl[0] == p, (v, p, pinfl)
        assert ginfl and ginfl[0] == g, (v, g, ginfl)
        assert p != v
        assert unique_past_base[p] == v

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
