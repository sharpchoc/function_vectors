#!/usr/bin/env python3
"""Generator for ends_with_ing task."""
import random
import json
from lemminflect import getInflection

def generate():
    # yes-class: gerunds from verbs
    with open('dataset_files/extended_tasks/_resources/common_verbs.txt', 'r') as f:
        verbs = [line.strip() for line in f if line.strip()]

    ing_words = []
    for verb in verbs:
        try:
            gerunds = getInflection(verb, 'VBG')
            for g in gerunds:
                if g.endswith('ing'):
                    ing_words.append(g)
        except:
            pass

    # no-class: nouns not ending in ing
    with open('dataset_files/extended_tasks/_resources/common_nouns.txt', 'r') as f:
        nouns = [line.strip() for line in f if line.strip()]
    
    non_ing_words = [w for w in nouns if not w.endswith('ing')]

    print(f"ing_words: {len(ing_words)}, non_ing_words: {len(non_ing_words)}")
    
    ing_words = list(set(ing_words))
    non_ing_words = list(set(non_ing_words))

    min_size = min(len(ing_words), len(non_ing_words))
    target = min(500, min_size)

    random.seed(42)
    y_sample = random.sample(ing_words, min(target, len(ing_words)))
    n_sample = random.sample(non_ing_words, min(target, len(non_ing_words)))

    dataset = [{"input": w, "output": "yes"} for w in y_sample]
    dataset += [{"input": w, "output": "no"} for w in n_sample]

    seen = set()
    deduped = []
    for item in dataset:
        if item["input"] not in seen:
            seen.add(item["input"])
            deduped.append(item)

    random.seed(42)
    random.shuffle(deduped)
    dataset = deduped[:1000]

    for item in dataset:
        w = item["input"]
        expected = "yes" if w.endswith('ing') else "no"
        assert item["output"] == expected

    assert len(dataset) == 1000
    assert len(set(i["input"] for i in dataset)) == 1000
    
    return dataset

if __name__ == "__main__":
    dataset = generate()
    with open('dataset_files/extended_tasks/ends_with_ing.json', 'w') as f:
        json.dump(dataset, f)
