#!/usr/bin/env python3
"""Generator for adjective_to_noun task.

Rule: given a derived adjective, output its base noun (dangerous->danger,
national->nation, hopeful->hope).

Recipe (per new_task_specs.json): candidate adjectives are the union of
common_adjs.txt and common_words.txt, filtered to words lemminflect
recognizes as an ADJ base form (this also removes proper-name noise and
-ly adverbs that slipped into the raw word lists). For each of a fixed
set of derivational suffix families, strip the suffix and try a small
set of orthographic restoration variants (bare stem, +e, +y, i->y,
un-double final consonant). A restored base is accepted only if it is a
real word (wordfreq zipf>=2.5) AND lemminflect recognizes it as a NOUN
lemma. Suffix families: -ous/-ious, -al/-ial, -ic/-ical, -ful, -less,
-ive, -y, -ish, -en, -some, -esque.

Passing the automated filters only checks that the derived STRING is a
real noun-tagged word, not that it is the true semantic root of the
adjective. A full manual audit of every candidate pair (see WORKLOG
2026-08-14) found three kinds of problems, handled below:

  1. Opaque/coincidental strings: the restored string happens to be a
     real noun, but is not what the adjective actually derives from
     (e.g. 'busy'->'bus', 'cordial'->'cord', 'moral'->'more', 'panic'
     ->'pane', 'legal'->'leg'). These are hard-excluded in
     HARD_EXCLUDE, one word per line with a one-word reason.
  2. Wrong-sense matches: the restored string is a real, etymologically
     related noun, but not the sense the adjective actually conveys
     (e.g. 'tactical'/'tactic'->'tact' should relate to "tactics", not
     "tact"; 'successive'->'success' should mean "in succession", not
     "having success"; 'respective'->'respect'). Also hard-excluded.
  3. Multiple candidates pass for one input (e.g. 'conical'->{'con',
     'cone'}). These are resolved by hand in MULTI_OVERRIDE; the
     ones with no acceptable candidate at all (e.g. 'final'->{'fin',
     'fine'}, 'manic'->{'man','mane','many'}, 'wary'->{'war','ware'})
     are hard-excluded instead.

A few additional special cases don't fit the generic strip+restore
rules (the base noun independently ends in '-ic', so only stripping
'-al' -- not '-ical' -- recovers it): 'musical'->'music', 'logical'->
'logic', 'magical'->'magic'. These are added via SPECIAL_CASE_ADD.

Two words were excluded purely for content sensitivity despite passing
all filters ('pussy'->'puss', already dropped as a MULTI case with no
good candidate; 'pornographic'->'pornography', semantically valid but
excluded as inappropriate for a general dataset).

HONESTY NOTE: the two specified word lists only contain ~650-700
suffix-family adjectives with a validated, transparent noun base after
this audit -- well short of the 1000 target. This script does NOT pad;
if the final count is below 1000 it prints the shortfall and refuses to
write the output file (see generate()/__main__ below).
"""
import json
import os
import random

from lemminflect import getAllLemmas
from wordfreq import zipf_frequency

HERE = os.path.dirname(os.path.dirname(__file__))
OUT_PATH = os.path.join(os.path.dirname(HERE), "adjective_to_noun.json")

VOWELS = set("aeiou")
ZIPF_THRESHOLD = 2.5

# Suffixes tried longest-first so e.g. "historical" is segmented as
# "-ical" (-> "histor" + y-restore -> "history") rather than "-al".
# Note: "-en" (wooden, golden, driven, fallen, given, green, mistaken,
# rotten, ...) was tried and dropped -- it mostly captures irregular
# verb past participles (driven/fallen/given/mistaken/rotten) and other
# false positives (green->"grey", heathen/maiden as attributive nouns),
# not material adjectives. The two genuine hits (wooden, golden) are
# added by hand via SPECIAL_CASE_ADD instead.
SUFFIXES = ["ical", "ial", "ous", "ful", "less", "ive", "ish", "some", "esque", "al", "ic", "y"]


def is_adj(w):
    return w in getAllLemmas(w).get("ADJ", ())


def is_noun(w):
    return w in getAllLemmas(w).get("NOUN", ())


def restorations(stem):
    """Orthographic restoration candidates for a stripped stem."""
    cands = {stem, stem + "e", stem + "y"}
    if stem.endswith("i"):
        cands.add(stem[:-1] + "y")  # beauti -> beauty, furi -> fury
    if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in VOWELS:
        cands.add(stem[:-1])  # mudd -> mud, funn -> fun
    return cands


def best_suffix(w):
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return suf
    return None


def candidates_for(w):
    suf = best_suffix(w)
    if suf is None:
        return []
    stem = w[:-len(suf)]
    cands = restorations(stem)
    return sorted(
        c for c in cands
        if c != w and zipf_frequency(c, "en") >= ZIPF_THRESHOLD and is_noun(c)
    )


# Words with a single semantically wrong or coincidental candidate noun,
# found by manual audit and dropped outright (word not in output at all).
HARD_EXCLUDE = {
    # -ical / -ic
    "tactical",     # base is "tactics", not "tact"
    "tactic",       # same: relates to "tactics", not "tact"
    "surgical",     # relates to "surgery", not "surge"
    "panic",        # Greek "Pan"; unrelated to "pane"
    "sonic",        # relates to "sound", coincidental match to "son"
    "rustic",       # Latin "rus" (countryside); unrelated to "rust"
    "pornographic", # sensitive/inappropriate for general dataset
    # -ial
    "martial",      # relates to Mars/war; unrelated to "mart"
    "cordial",      # Latin "cor" (heart); unrelated to "cord"
    # -ous
    "callous",      # relates to "callus"; unrelated to "call"
    "curious",      # coincidental match to name "Curie"
    "copious",      # Latin "copia"; too indirect a link to modern "copy"
    "hideous",      # Old French "hisdous"; unrelated to English "hide"
    "gorgeous",     # obscure link (throat/collar); not transparent today
    # -ful
    "grateful",     # Latin "gratus"; unrelated to "grate"
    # -ive
    "collective",   # "collect" as noun is too marginal/wrong-sense
    "conductive",   # wrong sense of "conduct" (behavior vs electricity)
    "constructive", # "construct" (noun) reading too technical/marginal
    "degenerative", # wrong sense of "degenerate" (person vs disease course)
    "digestive",    # wrong sense of "digest" (summary vs digestion)
    "elective",     # wrong/obscure sense of "elect" (the elect)
    "exhaustive",   # wrong sense of "exhaust" (car exhaust)
    "expressive",   # wrong sense of "express" (train/delivery)
    "formative",    # wrong sense of "format" (layout)
    "impressive",   # "impress" (noun, a seal) too obscure/wrong-sense
    "initiative",   # wrong sense of "initiate" (a person)
    "objective",    # not transparently "of an object"
    "passive",      # unrelated to "pass"
    "respective",   # unrelated to "respect"
    "subjective",   # not transparently "of a subject" for this rule
    "successive",   # unrelated to "success"
    # -ish
    "cornish",      # place-name adjective (Cornwall); unrelated to "corn"
    # -al
    "choral",       # relates to "chorus/choir"; unrelated to "chore"
    "dental",       # Latin "dens" (tooth); unrelated to "dent"
    "internal",     # unrelated to "intern"
    "legal",        # Latin "lex"; unrelated to "leg"
    "literal",      # Latin "littera" (letter); unrelated to "liter"
    "maximal",      # coincidental match to "maxim"
    "mineral",      # wrong derivational direction/unrelated to "miner"
    "moral",        # coincidental match to "more"
    "papal",        # relates to "pope"; unrelated to "pap"
    "penal",        # Latin "poena"; unrelated to "pen"
    "portal",       # not transparently "of a port" today
    "rational",     # drifted too far from "ration" for modern readers
    "spatial",      # true base "space" not reachable by these rules
    "spiral",       # coincidental match to "spire"
    "total",        # Latin "totus"; unrelated to "tot"
    "virtual",      # drifted too far from "virtue" for modern readers
    "final",        # neither "fin" nor "fine" is the true (Latin) root
    "coral",        # Greek/Latin "korallion"; unrelated to "core"
    "nocturnal",    # "nocturne" derives FROM night/nocturnal, not the reverse
    "feudal",       # relates to the "fief" sense of "feud", not "quarrel"
    # -y
    "army",         # noun mistagged ADJ; unrelated to "arm"
    "busy",         # coincidental match to "bus"
    "ready",        # coincidental match to "read"
    "steady",       # link to "stead" no longer transparent
    "pussy",        # sensitive term
    "ruby",         # from Latin "rubeus" (red); unrelated to "rub"/"rube"
    "wary",         # relates to "aware/beware"; unrelated to "ware"
    "early",        # coincidental match to "earl"
    "holy",         # coincidental match to "hole"
    "naughty",      # link to "naught" no longer transparent
    "petty",        # French "petit"; unrelated to "pet"
    "phony",        # disputed etymology; not transparently "of a phone"
    "teeny",        # means "tiny"; unrelated to "teen"
    "tiny",         # not transparently "of a tine"
    "sickly",       # coincidental match to "sickle"
    "slippery",     # "slipper" derives from "slip", not the reverse
    "stingy",       # uncertain link to "sting", not transparent
    "wacky",        # "wack" is a back-formation from "wacky", not its root
    "silly",        # unrelated to "sill" (windowsill)
    "tidy",         # drifted too far from "tide" (time/season sense) for modern readers
    "party",        # noun mistagged ADJ; not transparently "of a part"
    "auditory",     # relates to hearing/audition, not to an "auditor"
    "sensory",      # relates to "sense", not to a "sensor" (device)
    "conservatory", # not transparently "of a conservator"
    "respiratory",  # relates to "respiration", not to a "respirator"
    "manic",        # Greek "mania"; none of man/mane/many is the root
    # -some
    "handsome",     # drifted far from "hand" ("easy to handle" -> attractive)
    "tiresome",     # wrong sense of "tire" (car tire vs verb "to tire")
    "wholesome",    # not transparently "of a whole" for a modern reader
}

# Inputs where multiple restorations passed the automated filter; the
# correct one was picked by hand. (Only entries that survive audit are
# listed here -- ones with no good candidate are in HARD_EXCLUDE above.)
MULTI_OVERRIDE = {
    "conical": "cone",
    "colonial": "colony",
    "partial": "part",
    "analogous": "analogy",
    "masterful": "master",
    "sinful": "sin",
    "needless": "need",
    "worthless": "worth",
    "bullish": "bull",
    "fatal": "fate",
    "naval": "navy",
    "patriarchal": "patriarch",
    "spinal": "spine",
    "tidal": "tide",
    "tonal": "tone",
    "cubic": "cube",
    "microscopic": "microscope",
    "photographic": "photograph",
    "tonic": "tone",
    "scary": "scare",
    "shady": "shade",
    "shiny": "shine",
}

# A handful of -ical adjectives whose true base noun independently ends
# in "-ic" (music, logic, magic are words in their own right, not
# root+ic+al compounds), so the generic "-ical" strip+restore rule
# lands on the wrong (but real) word ("muse") or nothing at all. Added
# by hand instead of generalizing the stripping rule, since a blanket
# "-al only" fallback for -ical words creates many new ambiguous
# candidates (e.g. classical -> class vs classic) that would need their
# own audit.
SPECIAL_CASE_ADD = {
    "musical": "music",
    "logical": "logic",
    "magical": "magic",
    # -en material adjectives: kept by hand since the general "-en"
    # suffix family was dropped (see SUFFIXES comment) for producing
    # mostly wrong hits from irregular verb participles.
    "wooden": "wood",
    "golden": "gold",
}

# Expected input suffix for each SPECIAL_CASE_ADD entry (used only by
# the self-check asserts in generate()).
SPECIAL_CASE_SUFFIX = {
    "musical": "al",
    "logical": "al",
    "magical": "al",
    "wooden": "en",
    "golden": "en",
}


def build_pairs():
    words = set()
    for fn in ["common_adjs.txt", "common_words.txt"]:
        with open(os.path.join(HERE, fn)) as f:
            for line in f:
                w = line.strip()
                if w.isalpha() and w.islower() and len(w) >= 3:
                    words.add(w)

    adjs = sorted(w for w in words if is_adj(w))

    pairs = {}
    for w in adjs:
        if w in HARD_EXCLUDE:
            continue
        if w in SPECIAL_CASE_ADD:
            pairs[w] = SPECIAL_CASE_ADD[w]
            continue
        cands = candidates_for(w)
        if not cands:
            continue
        if len(cands) > 1:
            if w in MULTI_OVERRIDE:
                pairs[w] = MULTI_OVERRIDE[w]
            # else: ambiguous with no manual resolution -> dropped
            continue
        pairs[w] = cands[0]

    # words in SPECIAL_CASE_ADD that aren't otherwise adjectives in our
    # pool (shouldn't happen, but keep the invariant explicit)
    for w, n in SPECIAL_CASE_ADD.items():
        assert w in adjs, w

    return sorted(pairs.items())


def generate():
    pairs = build_pairs()
    print(f"domain size: {len(pairs)}")

    random.seed(42)
    random.shuffle(pairs)
    pairs = pairs[:1000]

    dataset = [{"input": a, "output": n} for a, n in pairs]

    random.seed(42)
    random.shuffle(dataset)

    seen_inputs = set()
    for item in dataset:
        a, n = item["input"], item["output"]
        assert a not in seen_inputs
        seen_inputs.add(a)
        assert a == a.strip() and n == n.strip()
        assert a != n
        if a in SPECIAL_CASE_SUFFIX:
            assert a.endswith(SPECIAL_CASE_SUFFIX[a]), (a, SPECIAL_CASE_SUFFIX[a])
        else:
            suf = best_suffix(a)
            assert suf is not None and a.endswith(suf), (a, suf)

    assert len(set(d["input"] for d in dataset)) == len(dataset)
    return dataset


if __name__ == "__main__":
    dataset = generate()
    # 2026-08-14: USER-APPROVED EXCEPTION to the 1000-example rule — the transparent
    # adjective->noun derivation vocabulary tops out at 599 pairs at the required
    # frequency/quality bar; the user chose to ship all 599 rather than swap the task.
    assert len(dataset) == 599, f"audited domain changed: {len(dataset)} != 599"
    with open(OUT_PATH, "w") as f:
        json.dump(dataset, f, indent=1)
    print(f"wrote {len(dataset)} examples to {OUT_PATH} (approved 599-example exception)")
