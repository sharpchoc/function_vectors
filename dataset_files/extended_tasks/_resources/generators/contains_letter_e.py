#!/usr/bin/env python3
"""Generator for contains_letter_e task."""
import random
import json

def generate():
    with open('dataset_files/extended_tasks/_resources/common_words.txt', 'r') as f:
        words = [line.strip() for line in f if line.strip()]

    words = [w for w in words if w.isalpha() and 3 <= len(w) <= 9]
    
    yes_words = [w for w in words if 'e' in w]
    no_words = [w for w in words if 'e' not in w]

    print(f"yes_words: {len(yes_words)}, no_words: {len(no_words)}")
    
    min_size = min(len(yes_words), len(no_words))
    target = min(500, min_size)

    random.seed(42)
    y_sample = random.sample(yes_words, min(target, len(yes_words)))
    n_sample = random.sample(no_words, min(target, len(no_words)))

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
        expected = "yes" if 'e' in w else "no"
        assert item["output"] == expected

    assert len(dataset) == 1000
    assert len(set(i["input"] for i in dataset)) == 1000
    
    return dataset

if __name__ == "__main__":
    dataset = generate()
    with open('dataset_files/extended_tasks/contains_letter_e.json', 'w') as f:
        json.dump(dataset, f)
