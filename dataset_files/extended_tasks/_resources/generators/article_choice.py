#!/usr/bin/env python3
"""Generator for article_choice task.

Rule: given a singular noun, output the correct indefinite article, 'a' or
'an', determined by the noun's initial sound.
"""
import json
import random

from lemminflect import getAllLemmas

RESOURCES = "dataset_files/extended_tasks/_resources"
OUT_PATH = "dataset_files/extended_tasks/article_choice.json"

VOWELS = set("aeiou")

# Vowel-letter words with a CONSONANT initial sound -> 'a' (not 'an').
OVERRIDE_A_PREFIXES = ("uni", "use", "user", "usu", "ute", "uto", "ura", "eu")
OVERRIDE_A_WORDS = {"one", "once", "ewe", "ukulele", "ubiquity", "unicorn",
                     "unicycle", "unify", "union", "unit", "united",
                     "universe", "university", "usable", "usage", "used",
                     "useful", "useless", "user", "usual", "utensil",
                     "uterus", "utility", "utopia", "eucalyptus", "eulogy",
                     "euphemism", "euphoria", "euro"}

# Consonant-letter (h) words with a SILENT h -> 'an' (not 'a').
OVERRIDE_AN_PREFIXES = ("hour", "honest", "honor", "honour", "heir")

# Closed-class function/pronoun words that dominate the top of the
# frequency-sorted common_nouns.txt list but are not usable as "a/an <word>"
# singular common nouns (reused judgment from the pos_label audit), plus a
# few high-frequency non-noun-dominant or plural/ambiguous words.
NON_NOUN_EXCLUDE = {
    "i", "me", "you", "he", "him", "she", "her", "it", "we", "us", "they",
    "them", "this", "that", "these", "those", "who", "whom", "whose",
    "which", "what", "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "yourselves", "themselves", "someone", "something",
    "anyone", "anything", "everyone", "everything", "nobody", "nothing",
    "none", "each", "either", "neither", "both", "all", "some", "any",
    "other", "another", "such", "whatever", "whoever", "whichever",
    "the", "a", "an", "my", "your", "his", "our", "their", "its", "no",
    "every", "several", "many", "few", "much", "more", "most",
    "and", "but", "or", "nor", "for", "yet", "so", "although", "because",
    "since", "unless", "while", "whereas", "if", "though", "whether",
    "of", "in", "on", "at", "by", "with", "from", "to", "into", "onto",
    "upon", "over", "under", "above", "below", "between", "among",
    "through", "during", "before", "after", "until", "despite", "towards",
    "via", "without", "within", "along", "across", "behind", "beside",
    "besides", "except", "plus", "minus", "per", "versus", "about",
    "against", "than", "as",
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "shall", "should", "can",
    "could", "may", "might", "must",
    # high-frequency words that are overwhelmingly not-noun / plural-only /
    # otherwise unnatural as "a/an <word>"
    "now", "well", "good", "first", "back", "still", "two", "people",
    "own", "last", "next", "same", "out", "how", "why", "where", "when",
    "yes", "not", "only", "also", "just", "very", "too", "even", "here",
    "there", "then", "once", "again", "always", "never", "often",
    "sometimes", "usually", "really", "actually", "probably", "already",
    # Second audit pass: words lemminflect tags as (also) NOUN but that are
    # not natural/usable as a countable singular "a/an <word>" -- dominant
    # reading is adjective/interjection/mass-noun, or vulgar/sensitive.
    "ouch", "everlasting", "elect", "elderly", "aging", "affected",
    "abandon", "anterior", "aquatic", "arse", "authentic", "exotic",
    "experimental", "external", "incoming", "indicative", "infantry",
    "insane", "intrinsic", "organic", "overweight", "unemployment",
    "urgency", "urging", "utilization", "utmost", "eighty",
    "colored", "veterinary", "earnings", "eleventh",
}


def load_words(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def classify(w):
    """Return 'a' or 'an', or None if too ambiguous to resolve."""
    if w.startswith(OVERRIDE_AN_PREFIXES):
        return "an"
    if w in OVERRIDE_A_WORDS or w.startswith(OVERRIDE_A_PREFIXES):
        return "a"
    if w[0] in VOWELS:
        return "an"
    return "a"


def main():
    nouns = load_words(f"{RESOURCES}/common_nouns.txt")

    candidates = {}
    seen = set()
    for w in nouns:
        if not w.isalpha() or not w.islower() or len(w) < 3:
            continue
        if w in seen or w in NON_NOUN_EXCLUDE:
            continue
        lemmas = getAllLemmas(w)
        if "NOUN" not in lemmas:
            continue
        seen.add(w)
        candidates[w] = classify(w)

    by_label = {"a": [], "an": []}
    for w, lab in candidates.items():
        by_label[lab].append(w)
    print(f"a: {len(by_label['a'])}, an: {len(by_label['an'])}")

    # Balance to 500/500 (an-words are the scarce class).
    per_class = min(500, len(by_label["an"]), len(by_label["a"]))
    random.seed(42)
    dataset = []
    for lab in ("a", "an"):
        chosen = random.sample(by_label[lab], per_class)
        dataset.extend({"input": w, "output": lab} for w in chosen)

    random.seed(42)
    random.shuffle(dataset)

    n = len(dataset)

    # --- self-checks ---
    inputs = [d["input"] for d in dataset]
    assert len(set(inputs)) == n, "duplicate inputs"
    from collections import Counter
    counts = Counter(d["output"] for d in dataset)
    print("class balance:", counts)
    for lab, cnt in counts.items():
        assert abs(cnt - n / 2) <= 0.10 * (n / 2)

    for d in dataset:
        w, lab = d["input"], d["output"]
        assert w == w.strip() and lab == lab.strip()
        # rule self-check: re-derive the article from the classify() rule
        assert classify(w) == lab

    with open(OUT_PATH, "w") as f:
        json.dump(dataset, f, indent=1)

    print(f"wrote {n} pairs to {OUT_PATH}")
    return n


if __name__ == "__main__":
    main()
