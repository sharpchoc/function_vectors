#!/usr/bin/env python3
"""
third_letter: Word -> its third letter (lowercase).
"""

import json
import random

# Read common_words.txt
with open('dataset_files/extended_tasks/_resources/common_words.txt') as f:
    all_words = [w.strip() for w in f.readlines() if w.strip()]

# Filter: alphabetic words, len >= 4
words = [
    w for w in all_words
    if w.isalpha() and len(w) >= 4
]

# Generate dataset
dataset = []
seen_inputs = set()

for word in words:
    if word in seen_inputs:
        continue

    output = word[2]

    dataset.append({
        'input': word,
        'output': output,
    })
    seen_inputs.add(word)

    if len(dataset) >= 1000:
        break

# Shuffle with seed 42
random.seed(42)
random.shuffle(dataset)

# Trim to exactly 1000
dataset = dataset[:1000]

# Self-check assertions
assert len(dataset) == 1000, f"Expected 1000 examples, got {len(dataset)}"

inputs = [d['input'] for d in dataset]
assert len(set(inputs)) == 1000, "Inputs are not unique"

for d in dataset:
    word = d['input']
    output = d['output']
    # Check: output is third letter
    assert output == word[2], f"Output mismatch for {word}"
    assert len(output) == 1, f"Output is not a single character for {word}"

# Write output
with open('dataset_files/extended_tasks/third_letter.json', 'w') as f:
    json.dump(dataset, f)

print(f"✓ third_letter: {len(dataset)} examples, {len(set(inputs))} unique inputs, {len(set(d['output'] for d in dataset))} unique outputs")
