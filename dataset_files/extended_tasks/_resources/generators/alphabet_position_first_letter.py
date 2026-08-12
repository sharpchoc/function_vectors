#!/usr/bin/env python3
"""Generator for alphabet_position_first_letter task."""
import random
import json

def generate():
    with open('dataset_files/extended_tasks/_resources/common_words.txt', 'r') as f:
        words = [line.strip() for line in f if line.strip()]

    words = [w for w in words if w.isalpha() and w[0] in 'abcdefghij']
    
    # Stratify by first letter to ensure all 10 letters appear
    by_letter = {}
    for w in words:
        first = w[0]
        if first not in by_letter:
            by_letter[first] = []
        by_letter[first].append(w)

    dataset = []
    random.seed(42)
    for letter in 'abcdefghij':
        if letter in by_letter:
            samples = random.sample(by_letter[letter], min(100, len(by_letter[letter])))
            for w in samples:
                pos = ord(letter) - 96
                dataset.append({"input": w, "output": str(pos)})

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
        expected = str(ord(w[0]) - 96)
        assert item["output"] == expected
        assert 1 <= int(item["output"]) <= 10

    assert len(dataset) == 1000
    assert len(set(i["input"] for i in dataset)) == 1000
    
    return dataset

if __name__ == "__main__":
    dataset = generate()
    with open('dataset_files/extended_tasks/alphabet_position_first_letter.json', 'w') as f:
        json.dump(dataset, f)
