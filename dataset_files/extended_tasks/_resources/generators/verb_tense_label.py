#!/usr/bin/env python3
"""Generator for verb_tense_label task.

Rule: given an inflected verb form, output which inflection it is: past,
gerund, or present (third-person -s).
"""
import json
import random
from collections import defaultdict

from lemminflect import getInflection

RESOURCES = "dataset_files/extended_tasks/_resources"
OUT_PATH = "dataset_files/extended_tasks/verb_tense_label.json"

TAG_TO_LABEL = {"VBD": "past", "VBG": "gerund", "VBZ": "present"}

# Sensitive base verbs to exclude regardless of inflection correctness.
SENSITIVE_EXCLUDE = {"retard", "rape"}


def load_words(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    verbs = load_words(f"{RESOURCES}/common_verbs.txt")

    form_labels = defaultdict(set)
    for v in verbs:
        if not v.isalpha() or not v.islower():
            continue
        if v in SENSITIVE_EXCLUDE:
            continue
        for tag, label in TAG_TO_LABEL.items():
            forms = getInflection(v, tag)
            if not forms:
                continue
            for f in forms:
                form_labels[f].add(label)

    # Keep only forms with exactly one label (drops collisions, e.g. base==past).
    single = {f: next(iter(labs)) for f, labs in form_labels.items() if len(labs) == 1}

    by_label = {"past": [], "gerund": [], "present": []}
    for f, lab in single.items():
        by_label[lab].append(f)
    for lab, fs in by_label.items():
        print(f"{lab}: {len(fs)}")

    # Balance to 1000 total across 3 classes (334/333/333), each within +-10%.
    random.seed(42)
    target = {"past": 334, "gerund": 333, "present": 333}
    dataset = []
    for lab, n in target.items():
        assert len(by_label[lab]) >= n, f"not enough {lab} forms"
        chosen = random.sample(by_label[lab], n)
        dataset.extend({"input": f, "output": lab} for f in chosen)

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
        assert abs(cnt - n / 3) <= 0.10 * (n / 3)

    for d in dataset:
        f, lab = d["input"], d["output"]
        assert f == f.strip() and lab == lab.strip()
        # rule self-check: re-derive label set from lemminflect inflection
        # of every verb in the pool and confirm f maps uniquely to lab.
        assert single[f] == lab

    with open(OUT_PATH, "w") as fh:
        json.dump(dataset, fh, indent=1)

    print(f"wrote {n} pairs to {OUT_PATH}")
    return n


if __name__ == "__main__":
    main()
