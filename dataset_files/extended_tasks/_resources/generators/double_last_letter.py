#!/usr/bin/env python3
"""
double_last_letter: Word -> word with final letter written twice.
Exclude words already ending in doubled letters.
"""

import json
import random

# Read common_words.txt
with open('dataset_files/extended_tasks/_resources/common_words.txt') as f:
    all_words = [w.strip() for w in f.readlines() if w.strip()]

# Filter: alphabetic words, len 3-6, no already-doubled final letter
words = []
for w in all_words:
    if not w.isalpha():
        continue
    if len(w) < 3 or len(w) > 6:
        continue
    # Exclude words already ending in doubled letter
    if len(w) >= 2 and w[-1] == w[-2]:
        continue
    words.append(w)

# Generate dataset
dataset = []
seen_inputs = set()

for word in words:
    if word in seen_inputs:
        continue

    output = word + word[-1]

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
    # Check: output is word + last letter
    assert output == word + word[-1], f"Output mismatch for {word}"
    assert output[:-1] == word, f"Output prefix doesn't match for {word}"
    assert output[-1] == word[-1], f"Output last letter doesn't match for {word}"

# Write output
with open('dataset_files/extended_tasks/double_last_letter.json', 'w') as f:
    json.dump(dataset, f)

print(f"✓ double_last_letter: {len(dataset)} examples, {len(set(inputs))} unique inputs, {len(set(d['output'] for d in dataset))} unique outputs")
