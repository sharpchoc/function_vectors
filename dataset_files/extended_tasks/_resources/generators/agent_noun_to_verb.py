#!/usr/bin/env python3
"""Generator for agent_noun_to_verb task.

Rule: given an -er/-or agent noun, output the verb it is derived from.

This is the inverse of the agent_noun task (spec index 89): we replicate that
task's verb -> agent-noun candidate-generation rule (to get a validated,
deterministic (verb, agent) pair list), then invert it, keeping only agent
nouns that map to exactly one verb.
"""
import json
import random
import re

from lemminflect import getAllLemmas
from wordfreq import top_n_list, zipf_frequency

RESOURCES = "dataset_files/extended_tasks/_resources"
OUT_PATH = "dataset_files/extended_tasks/agent_noun_to_verb.json"

VOWELS = set("aeiou")

# Agent-noun surface forms that pass the mechanical -er/-or + zipf filter but
# are NOT semantically "one/thing that <verb>s" -- coincidental real words
# (false cognates), proper-noun/surname homographs with a dominant non-agentive
# reading, or otherwise too ambiguous out of context. Identified by manual
# audit of all candidates. Comparative-adjective confusables (calmer, faster,
# master, number, proper, ...) are instead caught mechanically below via the
# lemminflect ADJ-lemma check, so they are not repeated here.
SEMANTIC_BLACKLIST = {
    "letter", "flower", "gutter", "honor", "gator", "dolor", "furor",
    "clamor", "corner", "cower", "limber", "luster", "muster", "temper",
    "cocker", "cooper", "corker", "coster", "bugger", "gasser", "soler",
    "conor", "bogor", "acer", "boner", "bower", "viner", "sager", "famer",
    "asher", "bayer", "nestor", "guyer", "buller", "rutter", "kenner",
    "geller", "garber", "keeler", "horner", "weller", "jeter", "cramer",
    "gruber", "pryor", "error", "cater", "coker", "freer",
    "fiber", "visor", "supper", "dimer", "wicker", "larder", "number",
    "razor", "rigor", "rotor", "scatter", "shimmer", "stagger", "summer",
    "twitter", "career", "hammer", "humor", "matter", "mayor", "donor",
    "peer", "caper", "barber", "butter", "lager", "mister", "pepper",
    "tower", "tier", "badger", "crater", "cursor", "demeanor", "equator",
    "gazetteer", "lather", "ledger", "pastor", "splinter", "whisker",
    "narrower",
}


def load_words(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def candidate_agent_nouns(v):
    """Replicate the agent_noun spec's candidate-generation rule for verb v."""
    cands = set()
    cands.add(v + "er")
    cands.add(v + "or")
    if v.endswith("e"):
        cands.add(v + "r")
        cands.add(v[:-1] + "or")
    if len(v) >= 2 and v[-1] == "y" and v[-2] not in VOWELS:
        cands.add(v[:-1] + "ier")
    # monosyllabic, single-vowel, consonant-final -> double final consonant +
    # 'er' (e.g. run -> runner, stop -> stopper, trap -> trapper)
    n_vowels = sum(1 for ch in v if ch in VOWELS)
    if (
        n_vowels == 1
        and len(v) >= 3
        and v[-1] not in VOWELS
        and v[-1] not in "wxy"
        and v[-2] in VOWELS
    ):
        cands.add(v + v[-1] + "er")
    return cands


def is_valid_agent_noun(cand, v):
    """Mechanical filters: real word, not a comparative-adjective confusable
    with a DIFFERENT root, not on the manually-audited semantic blacklist of
    false cognates."""
    if zipf_frequency(cand, "en") < 2.2:
        return False
    lemmas = getAllLemmas(cand)
    if "NOUN" not in lemmas:
        # not recognized as a noun at all (verb-only, e.g. refer/wander/ponder,
        # or altogether unrecognized, e.g. bluer/truer/rainer/dozer) --
        # cannot be a genuine agent noun.
        return False
    adj_lemma = lemmas.get("ADJ")
    if adj_lemma is not None and adj_lemma != (v,):
        # Has a competing adjective reading whose lemma is NOT the same verb
        # we're deriving from -- e.g. master's ADJ lemma is ('master',) not
        # ('mast',); proper's is ('proper',) not ('prop',). That mismatch
        # means the dominant sense is an unrelated word, not this verb's
        # agent noun. (When the ADJ lemma DOES equal v -- e.g. trimmer/trim,
        # flipper/flip, cooler/cool, damper/damp -- the comparative-adjective
        # reading and the agent-noun reading share the same root, so it's
        # harmless ambiguity and we keep it.)
        return False
    if cand in SEMANTIC_BLACKLIST:
        return False
    return True


def build_verb_to_agent(verbs):
    verb_to_agent = {}
    for v in verbs:
        if not v.isalpha() or not v.islower():
            continue
        cands = candidate_agent_nouns(v)
        good = [c for c in cands if is_valid_agent_noun(c, v)]
        if len(good) == 1:
            verb_to_agent[v] = good[0]
    return verb_to_agent


def extra_verb_pool():
    """Base-form verbs from the wordfreq top-100k not already in
    common_verbs.txt, to backfill after semantic filtering (mirrors the
    agent_noun spec's recipe of common_verbs.txt + wordfreq top-100k)."""
    extra = []
    for w in top_n_list("en", 100000):
        if not w.isalpha() or not w.islower() or len(w) < 3:
            continue
        lemmas = getAllLemmas(w)
        if lemmas.get("VERB") == (w,):
            extra.append(w)
    return extra


def main():
    base_verbs = load_words(f"{RESOURCES}/common_verbs.txt")
    verb_pool = list(dict.fromkeys(base_verbs + extra_verb_pool()))
    verb_to_agent = build_verb_to_agent(verb_pool)
    print(f"verbs with unique agent-noun candidate: {len(verb_to_agent)}")

    # Invert; drop agent nouns that (rarely) come from >1 verb.
    agent_to_verbs = {}
    for v, a in verb_to_agent.items():
        agent_to_verbs.setdefault(a, set()).add(v)

    agent_to_verb = {a: next(iter(vs)) for a, vs in agent_to_verbs.items() if len(vs) == 1}
    print(f"agent nouns mapping to exactly one verb: {len(agent_to_verb)}")

    pairs = sorted(agent_to_verb.items())

    random.seed(42)
    random.shuffle(pairs)

    n = min(1000, len(pairs))
    chosen = pairs[:n]

    dataset = [{"input": a, "output": v} for a, v in chosen]

    random.seed(42)
    random.shuffle(dataset)

    # --- self-checks ---
    assert len(dataset) == n
    inputs = [d["input"] for d in dataset]
    assert len(set(inputs)) == n, "duplicate inputs"
    for d in dataset:
        assert d["input"] not in ("", None)
        assert d["output"] not in ("", None)
        assert d["input"] == d["input"].strip()
        assert d["output"] == d["output"].strip()
        # rule self-check: re-derive candidate agent nouns from the output verb
        # and confirm the input is among them (and is the unique validated one)
        cands = candidate_agent_nouns(d["output"])
        assert d["input"] in cands, (d["input"], d["output"], cands)

    with open(OUT_PATH, "w") as f:
        json.dump(dataset, f, indent=1)

    print(f"wrote {n} pairs to {OUT_PATH}")
    return n


if __name__ == "__main__":
    main()
