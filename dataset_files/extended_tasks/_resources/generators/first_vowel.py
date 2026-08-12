#!/usr/bin/env python3
"""Generator for first_vowel task."""
import random
import json

def generate():
    with open('dataset_files/extended_tasks/_resources/common_words.txt', 'r') as f:
        words = [line.strip() for line in f if line.strip()]

    words = [w for w in words if w.isalpha() and len(w) >= 3 and any(c in 'aeiou' for c in w)]
    
    dataset = []
    for w in words:
        first_vowel = next((c for c in w if c in 'aeiou'), None)
        if first_vowel:
            dataset.append({"input": w, "output": first_vowel})

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
        expected = next(c for c in w if c in 'aeiou')
        assert item["output"] == expected
        assert item["output"] in 'aeiou'

    assert len(dataset) == 1000
    assert len(set(i["input"] for i in dataset)) == 1000
    
    return dataset

if __name__ == "__main__":
    dataset = generate()
    with open('dataset_files/extended_tasks/first_vowel.json', 'w') as f:
        json.dump(dataset, f)
