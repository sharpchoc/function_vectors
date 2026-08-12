#!/usr/bin/env python3
"""Generator for gerund_to_base task (spec index 80).

Rule: given a verb's -ing (gerund) form, output its base (infinitive) form.

Domain: dataset_files/extended_tasks/_resources/common_verbs.txt, frequency-sorted
descending. For each verb v compute g = getInflection(v, 'VBG')[0]. Keep gerunds
that map back to exactly one base verb (unambiguous). Modal auxiliaries (can, will,
shall, may, must, ought) are excluded up front: lemminflect returns their suppletive
modal forms (e.g. getInflection('can','VBD') -> 'could') for VBD/VBZ, which corrupts
the shared uniqueness dictionaries reused across the verb_form_conversion family of
tasks (past_to_gerund / gerund_to_past / gerund_to_third_person) even though this
particular task only touches VBG. Excluded uniformly for consistency with those
sibling generators, which pull from the same base list.

Selection prefers the most frequent verbs (i.e. earliest in common_verbs.txt) so
GPT-J is more likely to know both the base verb and its gerund, then shuffles with
random.seed(42) for final example order.
"""
import json
import random
from collections import defaultdict
from lemminflect import getInflection

VERBS_PATH = 'dataset_files/extended_tasks/_resources/common_verbs.txt'
OUT_PATH = 'dataset_files/extended_tasks/gerund_to_base.json'

MODAL_EXCLUDE = {'can', 'will', 'shall', 'may', 'must', 'ought'}

# Manually identified during a spot-check of the highest-frequency entries: base
# verbs whose lemminflect-derived -ing form is either not a real English word
# ('product' -> 'producting', 'self' -> 'selfing' is obscure biology jargon,
# 'safe' -> 'safing' is obscure ordnance jargon, 'king' -> 'kinging' is not
# standard, 'fair' -> 'fairing' collides with an unrelated noun rather than being
# derived from the adjective) or so dominated by an unrelated, far more common
# reading that the mapping is misleading rather than merely rare ('even' ->
# 'evening' reads overwhelmingly as the time of day; 'interest' -> 'interesting'
# reads overwhelmingly as the adjective; 'weekend', 'article', 'league' ->
# their gerunds are not genuine common verb usages).
AMBIGUOUS_BASE_EXCLUDE = {
    'product', 'self', 'safe', 'king', 'fair',
    'even', 'interest', 'weekend', 'article', 'league',
    'over', 'hot', 'middle',
}


def generate():
    with open(VERBS_PATH) as f:
        verbs = [line.strip() for line in f if line.strip()]
    verbs = [v for v in verbs if v not in MODAL_EXCLUDE and v not in AMBIGUOUS_BASE_EXCLUDE]

    # Build gerund -> set(base verbs) over the whole list, via getInflection(v,'VBG')[0].
    gerund_to_bases = defaultdict(set)
    gerund_for_verb = {}
    for v in verbs:
        infl = getInflection(v, 'VBG')
        if not infl:
            continue
        g = infl[0]
        gerund_for_verb[v] = g
        gerund_to_bases[g].add(v)

    unique_bases = {g: next(iter(s)) for g, s in gerund_to_bases.items() if len(s) == 1}

    # Walk verbs in frequency order, keep first-seen unambiguous gerunds.
    seen_gerunds = set()
    pairs = []
    for v in verbs:
        g = gerund_for_verb.get(v)
        if g is None or g in seen_gerunds:
            continue
        if unique_bases.get(g) != v:
            continue
        seen_gerunds.add(g)
        pairs.append((g, v))

    n_available = len(pairs)
    pairs = pairs[:1000]

    dataset = [{"input": g, "output": v} for g, v in pairs]

    random.seed(42)
    random.shuffle(dataset)

    # Self-check: re-derive via lemminflect and confirm unambiguous, deterministic mapping.
    for item in dataset:
        g, v = item["input"], item["output"]
        infl = getInflection(v, 'VBG')
        assert infl and infl[0] == g, (v, g, infl)
        assert unique_bases[g] == v

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
