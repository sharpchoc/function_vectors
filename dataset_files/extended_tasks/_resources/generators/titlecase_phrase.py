#!/usr/bin/env python3
"""
titlecase_phrase: Two-word phrase -> title-cased version.
Samples from common_adjs.txt and common_nouns.txt.
"""

import json
import random

# Read common_adjs.txt
with open('dataset_files/extended_tasks/_resources/common_adjs.txt') as f:
    all_adjs = [w.strip() for w in f.readlines() if w.strip()]

# Read common_nouns.txt
with open('dataset_files/extended_tasks/_resources/common_nouns.txt') as f:
    all_nouns = [w.strip() for w in f.readlines() if w.strip()]

# Filter adjectives: top 1500, alphabetic, len 3-8
adjs = []
for adj in all_adjs[:1500]:
    if adj.isalpha() and 3 <= len(adj) <= 8:
        adjs.append(adj)

# Filter nouns: top 2000, alphabetic, len 3-8
nouns = []
for noun in all_nouns[:2000]:
    if noun.isalpha() and 3 <= len(noun) <= 8:
        nouns.append(noun)

# Generate bigrams
dataset = []
seen_inputs = set()

# Try to generate ~20000 random combinations and keep best 1000 after dedup
random.seed(42)
combo_set = set()

for _ in range(20000):
    adj = random.choice(adjs)
    noun = random.choice(nouns)
    pair = (adj, noun)
    if pair not in combo_set:
        combo_set.add(pair)

# Convert to dataset
for adj, noun in list(combo_set):
    input_str = f'{adj} {noun}'
    if input_str in seen_inputs:
        continue

    output = input_str.title()

    dataset.append({
        'input': input_str,
        'output': output,
    })
    seen_inputs.add(input_str)

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
    input_str = d['input']
    output = d['output']
    # Check: output is title-cased input
    assert output == input_str.title(), f"Output mismatch for {input_str}"
    assert output.lower() == input_str, f"Lowercase output doesn't match input for {input_str}"
    # Check both words start with uppercase
    parts = output.split()
    assert len(parts) == 2, f"Expected 2 words in output for {input_str}"
    assert parts[0][0].isupper() and parts[1][0].isupper(), f"Not title-cased: {output}"

# Write output
with open('dataset_files/extended_tasks/titlecase_phrase.json', 'w') as f:
    json.dump(dataset, f)

print(f"✓ titlecase_phrase: {len(dataset)} examples, {len(set(inputs))} unique inputs, {len(set(d['output'] for d in dataset))} unique outputs")
