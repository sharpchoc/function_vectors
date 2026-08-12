#!/usr/bin/env python3
"""Generator for pos_label task.

Rule: given a word with only one part of speech, output that part of speech
(noun, verb, adjective, or adverb).
"""
import json
import random

from lemminflect import getAllLemmas

RESOURCES = "dataset_files/extended_tasks/_resources"
OUT_PATH = "dataset_files/extended_tasks/pos_label.json"

POS_TAGS = {"NOUN": "noun", "VERB": "verb", "ADJ": "adjective", "ADV": "adverb"}

# Manual audit: words lemminflect tags as single-POS but that are commonly
# used in a second part of speech in ordinary English (lemminflect's
# dictionary misses the conversion). Excluded so the label is unambiguous.
AMBIGUOUS_OVERRIDE = {
    # tagged NOUN-only by lemminflect, but common verb usage too
    "name", "hand", "head", "house", "water", "book", "eye", "face", "place",
    "school", "study", "state", "voice", "step", "form", "note", "plant",
    "plan", "chair", "fish", "text", "email", "picture", "party", "trust",
    "cause", "corner", "farm", "mind", "master", "host", "chip", "camp",
    "trip", "fear", "silence", "wonder", "channel", "signal", "process",
    "credit", "debt", "profile", "list", "date", "table", "brand", "mask",
    "screen", "stage", "target", "network", "market", "impact", "access",
    "voice", "surface", "author", "pace", "range", "framework", "position",
    "sense", "level", "figure", "field", "ground", "guard", "pair", "map",
    "toast", "fuel", "cash", "fire", "flag", "seat", "queue", "shape",
    "shelter", "smoke", "spy", "storm", "structure", "style", "trade",
    "tune", "vote", "wave", "welcome",
    # tagged VERB-only, but common noun usage too
    "increase", "decrease", "attempt", "hope", "concern", "impact",
    "release", "control", "demand", "offer", "call", "shift", "reach",
    "gain", "loss", "rise", "return", "drop", "cut", "reply", "escape",
    "fight", "help", "kiss", "laugh", "smile", "talk", "walk", "cover",
    "search", "sound", "share", "spread", "start", "study", "watch",
    "work", "wish", "answer", "attack", "change", "check", "close",
    "comment", "contact", "count", "cry", "dance", "design", "drive",
    "end", "exercise", "experience", "feel", "fund", "grade", "guide",
    "hit", "hope", "hug", "hurt", "influence", "interest", "jump",
    "kick", "land", "lead", "look", "love", "match", "mix", "move",
    "need", "order", "plan", "play", "power", "practice", "press",
    "promise", "question", "record", "rest", "result", "risk", "run",
    "sale", "sample", "score", "sign", "smell", "stop", "stress",
    "support", "surprise", "taste", "test", "touch", "trade", "train",
    "trust", "turn", "use", "view", "visit", "vote", "want", "wave",
    "win", "wonder", "worry",
    # tagged ADJ-only, but common noun/verb usage too
    "right", "clear", "clean", "warm", "cool", "dry", "empty", "even",
    "firm", "free", "level", "loose", "open", "round", "slow", "smooth",
    "still", "tight",
}

# Closed-class function words (pronouns, determiners, conjunctions,
# prepositions, auxiliary/modal verbs, question words). lemminflect's
# dictionary mistags many of these as single-POS content words (e.g. "this",
# "that", "something", "none" as NOUN-only) -- exclude them outright, this
# task is about open-class content-word POS, not function words.
FUNCTION_WORDS = {
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
}

# Sensitive terms to exclude regardless of POS-label correctness.
SENSITIVE_EXCLUDE = {"rape", "raping", "raped", "rapist"}


def load_words(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    words = load_words(f"{RESOURCES}/common_words.txt")

    labeled = []
    seen = set()
    for w in words:
        if not w.isalpha() or not w.islower():
            continue
        if w in seen or w in AMBIGUOUS_OVERRIDE or w in FUNCTION_WORDS or w in SENSITIVE_EXCLUDE:
            continue
        lemmas = getAllLemmas(w)
        pos_keys = [k for k in lemmas if k in POS_TAGS]
        if len(pos_keys) == 1:
            seen.add(w)
            labeled.append((w, POS_TAGS[pos_keys[0]]))

    by_label = {"noun": [], "verb": [], "adjective": [], "adverb": []}
    for w, lab in labeled:
        by_label[lab].append(w)

    for lab, ws in by_label.items():
        print(f"{lab}: {len(ws)}")

    # Balance: 250 of each label (1000 total, exactly balanced).
    per_class = 250
    random.seed(42)
    dataset = []
    for lab, ws in by_label.items():
        assert len(ws) >= per_class, f"not enough {lab} words: {len(ws)}"
        chosen = random.sample(ws, per_class)
        dataset.extend({"input": w, "output": lab} for w in chosen)

    random.seed(42)
    random.shuffle(dataset)

    n = len(dataset)

    # --- self-checks ---
    assert n == 1000
    inputs = [d["input"] for d in dataset]
    assert len(set(inputs)) == n, "duplicate inputs"
    from collections import Counter
    counts = Counter(d["output"] for d in dataset)
    print("class balance:", counts)
    for lab, cnt in counts.items():
        assert abs(cnt - n / 4) <= 0.10 * (n / 4)

    for d in dataset:
        w, lab = d["input"], d["output"]
        assert w == w.strip() and lab == lab.strip()
        # rule self-check: re-derive POS from lemminflect and confirm single match
        lemmas = getAllLemmas(w)
        pos_keys = [k for k in lemmas if k in POS_TAGS]
        assert len(pos_keys) == 1 and POS_TAGS[pos_keys[0]] == lab, (w, lab, lemmas)

    with open(OUT_PATH, "w") as f:
        json.dump(dataset, f, indent=1)

    print(f"wrote {n} pairs to {OUT_PATH}")
    return n


if __name__ == "__main__":
    main()
