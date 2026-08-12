#!/usr/bin/env python3
"""
Generator for count_words task.
Phrase of 1-4 space-separated words -> count of words.
"""

import json
import random

# Read resource files
with open('dataset_files/extended_tasks/_resources/common_adjs.txt') as f:
    adjs = [line.strip().lower() for line in f if line.strip() and line.strip().isalpha()]

with open('dataset_files/extended_tasks/_resources/common_nouns.txt') as f:
    nouns = [line.strip().lower() for line in f if line.strip() and line.strip().isalpha()]

# Filter: reasonable length
adjs = [w for w in adjs[:1500] if 3 <= len(w) <= 8]
nouns = [w for w in nouns[:2000] if 3 <= len(w) <= 8]

print(f"Available adjs: {len(adjs)}, nouns: {len(nouns)}")

random.seed(42)
examples = []
seen_inputs = set()

# Generate examples for each length
# Length 1: single noun
length_counts = {1: 0, 2: 0, 3: 0, 4: 0}
target_per_length = 250

while len(examples) < 1000:
    length = random.randint(1, 4)

    if length_counts[length] >= target_per_length:
        # Try another length
        available_lengths = [l for l in range(1, 5) if length_counts[l] < target_per_length]
        if not available_lengths:
            break
        length = random.choice(available_lengths)

    # Generate phrase
    if length == 1:
        # Single noun
        noun = random.choice(nouns)
        input_str = noun
        output = "1"
    elif length == 2:
        # Adj Noun
        adj = random.choice(adjs)
        noun = random.choice(nouns)
        input_str = f"{adj} {noun}"
        output = "2"
    elif length == 3:
        # Adj Adj Noun - ensure distinct adjs
        adj1 = random.choice(adjs)
        adj2 = random.choice(adjs)
        while adj2 == adj1:
            adj2 = random.choice(adjs)
        noun = random.choice(nouns)
        input_str = f"{adj1} {adj2} {noun}"
        output = "3"
    else:  # length == 4
        # Adj Adj Adj Noun - ensure distinct adjs
        adj1 = random.choice(adjs)
        adj2 = random.choice(adjs)
        while adj2 == adj1:
            adj2 = random.choice(adjs)
        adj3 = random.choice(adjs)
        while adj3 in [adj1, adj2]:
            adj3 = random.choice(adjs)
        noun = random.choice(nouns)
        input_str = f"{adj1} {adj2} {adj3} {noun}"
        output = "4"

    if input_str not in seen_inputs:
        examples.append({
            "input": input_str,
            "output": output
        })
        seen_inputs.add(input_str)
        length_counts[length] += 1

print(f"Generated {len(examples)} examples")
print(f"Length distribution: {length_counts}")

# Subsample to exactly 1000
if len(examples) > 1000:
    examples = random.sample(examples, 1000)

# Shuffle with seed
random.seed(42)
random.shuffle(examples)

print(f"Final count: {len(examples)}")

# Assertions
assert len(examples) == 1000, f"Expected 1000, got {len(examples)}"
input_list = [ex['input'] for ex in examples]
assert len(input_list) == len(set(input_list)), "Duplicate inputs found"

# Self-check
for ex in examples:
    word_count = len(ex['input'].split())
    assert str(word_count) == ex['output'], f"Output {ex['output']} doesn't match word count {word_count}"

print("All assertions passed!")

# Write to file
output_file = 'dataset_files/extended_tasks/count_words.json'
with open(output_file, 'w') as f:
    json.dump(examples, f)
print(f"Wrote {len(examples)} examples to {output_file}")
