#!/usr/bin/env python3
"""Generator for singular_or_plural: given a noun form, output 'singular' or 'plural'.

Recipe (spec idx 75): for each noun in common_nouns.txt, lemmatize it, derive its
singular (NN) and plural (NNS) forms via lemminflect, and emit both as separate
examples (singular form -> 'singular', plural form -> 'plural'). Using the SAME
underlying set of lemmas for both classes yields exact 50/50 balance by
construction.

Exclusions:
- invariant plurals (plural form == singular form, e.g. 'sheep', 'series')
- mass/rare plurals (wordfreq zipf(plural) < 2.5 -- no naturally-attested plural)
- pluralia tantum (zipf(singular) < zipf(plural) -- singular is rarer than the
  plural, e.g. would catch 'scissor'-type lemmas)
- non-alphabetic / multi-word / ambiguous multi-form outputs (we always take the
  lemminflect-preferred first form, but require lemma to be unambiguous re: NN)

Uses python3.12 (has lemminflect + wordfreq installed system-wide).
"""
import json
import os
import random

from lemminflect import getInflection, getLemma
from wordfreq import zipf_frequency

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
OUT = os.path.join(os.path.dirname(RES), "singular_or_plural.json")

TARGET_PAIRS = 500  # -> 500 singular + 500 plural = 1000 examples

# common_nouns.txt is a lemminflect-lemma-dictionary NOUN filter over a huge web2
# wordlist, not a hand-curated concrete-noun list -- it lets through closed-class
# function words that some lexical resource mistags as NOUN (pronouns/determiners
# etc.), plus a few genuine nouns whose most frequent surface form is a homograph
# of an unrelated common word (e.g. plural of "leave" collides with "leaf"). We
# exclude both classes explicitly since they are a small, enumerable set.
STOPWORDS = {
    "that", "this", "these", "those", "you", "your", "yours", "i", "me", "my",
    "mine", "we", "us", "our", "ours", "they", "them", "their", "theirs", "he",
    "him", "his", "she", "her", "hers", "it", "its", "who", "whom", "whose",
    "which", "what", "all", "any", "some", "none", "one", "other", "others",
    "another", "such", "own", "same", "more", "most", "less", "least", "many",
    "much", "few", "several", "both", "each", "either", "neither", "no", "not",
    "nor", "and", "or", "but", "so", "if", "than", "then", "now", "here",
    "there", "where", "when", "why", "how", "well", "still", "even", "just",
    "also", "only", "very", "too", "quite", "rather", "almost", "already",
    "always", "never", "often", "sometimes", "usually", "up", "down", "out",
    "off", "over", "under", "above", "below", "between", "among", "through",
    "during", "before", "after", "since", "until", "while", "because",
    "although", "though", "unless", "inside", "outside", "within", "without",
    "about", "around", "across", "along", "behind", "beyond", "beside",
    "besides", "against", "toward", "towards", "upon", "onto", "into",
    "throughout", "whatever", "whoever", "whenever", "wherever", "however",
    "yes", "no", "ok", "okay", "please", "thanks", "hello", "yeah",
}
# Homograph / register issues spotted during a manual sample QA pass (plural
# collides with an unrelated word, or the noun sense is dominated by a
# non-count adjective/verb/adverb sense of the same surface form).
AMBIGUOUS_EXCLUDE = {
    "leave", "brown", "military",
    "ass",  # vulgar
    "black", "white",  # racially-loaded plural-as-people-noun sense
    "blood",  # "bloods" is not a standard plural
    "clean", "close",  # -s form reads as 3rd-person verb, not a noun plural
    "due", "good",  # pluralia-tantum leak: "dues"/"goods" have a distinct
    # idiomatic sense unrelated to the ordinary meaning of the singular
    "hell",  # "hells" is not a standard plural
    "keep",  # "keeps" reads as a verb, not a noun plural
    "last",  # "lasts" reads as a verb, not a noun plural
    "old",  # "olds" is not a standard plural
    "today",  # "todays" is not a standard plural
    "gay",  # avoid identity-based noun in a mechanical morphology task
}
EXCLUDE = STOPWORDS | AMBIGUOUS_EXCLUDE
MIN_ZIPF = 3.0  # raise from 2.5: matches the corpus-level MIN_ZIPF used to build
# common_nouns.txt itself, filters out rare/dubious plurals like "informations",
# "fasts", "militaries" that a looser 2.5 threshold let through.


def main() -> None:
    with open(os.path.join(RES, "common_nouns.txt")) as f:
        nouns = [w.strip() for w in f if w.strip()]

    seen_lemmas = set()
    seen_forms = set()
    pairs = []  # (singular, plural)

    for w in nouns:  # frequency-sorted descending -> prefer well-known words
        if not (w.isalpha() and w.islower() and len(w) >= 2):
            continue
        if w in EXCLUDE:
            continue
        lemma_res = getLemma(w, "NOUN")
        lemma = lemma_res[0] if lemma_res else w
        if lemma in seen_lemmas:
            continue

        sg_res = getInflection(lemma, "NN")
        singular = sg_res[0] if sg_res else lemma
        pl_res = getInflection(lemma, "NNS")
        if not pl_res:
            continue
        plural = pl_res[0]

        if not (singular.isalpha() and singular.islower()):
            continue
        if not (plural.isalpha() and plural.islower()):
            continue
        if singular in EXCLUDE or plural in EXCLUDE:
            continue
        if singular == plural:
            continue  # invariant plural (sheep, series, moose, deer, fish-1st-form...)
        if zipf_frequency(plural, "en") < MIN_ZIPF:
            continue  # mass/rare plural, not naturally attested
        if zipf_frequency(singular, "en") < zipf_frequency(plural, "en"):
            continue  # pluralia tantum: singular rarer than plural

        if singular in seen_forms or plural in seen_forms:
            continue

        seen_lemmas.add(lemma)
        seen_forms.add(singular)
        seen_forms.add(plural)
        pairs.append((singular, plural))

        if len(pairs) >= TARGET_PAIRS:
            break

    assert len(pairs) == TARGET_PAIRS, f"only found {len(pairs)} valid noun pairs"

    data = [{"input": s, "output": "singular"} for s, p in pairs]
    data += [{"input": p, "output": "plural"} for s, p in pairs]

    random.seed(42)
    random.shuffle(data)

    # Self-checks
    assert len(data) == 1000
    inputs = [d["input"] for d in data]
    assert len(set(inputs)) == 1000, "duplicate inputs"
    sg_set = {s for s, p in pairs}
    pl_set = {p for s, p in pairs}
    for d in data:
        w, o = d["input"], d["output"]
        assert w == w.strip() and o == o.strip()
        if o == "singular":
            assert w in sg_set
        else:
            assert o == "plural"
            assert w in pl_set
    from collections import Counter

    counts = Counter(d["output"] for d in data)
    assert counts["singular"] == counts["plural"] == 500

    with open(OUT, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {len(data)} to {OUT}; classes={dict(counts)}")


if __name__ == "__main__":
    main()
