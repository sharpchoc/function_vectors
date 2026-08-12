#!/usr/bin/env python3
"""
Generator for count_letter_e task.
Word -> count of letter 'e' (0-3).
"""

import json
import random

# Read resource file
with open('dataset_files/extended_tasks/_resources/common_words.txt') as f:
    words = [line.strip().lower() for line in f if line.strip() and line.strip().isalpha()]

# Filter: alphabetic, length 3-9, and count('e') <= 3
words = [w for w in words if 3 <= len(w) <= 9 and w.count('e') <= 3]

print(f"Available words: {len(words)}")

# Count distribution
count_dist = {0: [], 1: [], 2: [], 3: []}
for word in words:
    c = word.count('e')
    count_dist[c].append(word)

print(f"Distribution: 0={len(count_dist[0])}, 1={len(count_dist[1])}, 2={len(count_dist[2])}, 3={len(count_dist[3])}")

# Generate stratified examples
random.seed(42)
examples = []
seen_inputs = set()

# Due to limited count-3 words (183), adjust targets dynamically
# Ensure all counts are represented substantially
available_counts = {c: len(count_dist[c]) for c in [0, 1, 2, 3]}
print(f"Available per count: {available_counts}")

# Strategy: allocate 250 to abundant counts (0,1,2), use all of count-3 (183), fill remainder
targets = {0: 270, 1: 270, 2: 277, 3: 183}

for count in [0, 1, 2, 3]:  # Process in order of abundance
    candidates = count_dist[count]
    target = targets[count]

    # If we have more candidates than target, sample
    if len(candidates) >= target:
        sampled = random.sample(candidates, target)
    else:
        sampled = candidates

    for word in sampled:
        if word not in seen_inputs:
            examples.append({
                "input": word,
                "output": str(count)
            })
            seen_inputs.add(word)

print(f"Generated {len(examples)} examples")

# Count final distribution
final_dist = {0: 0, 1: 0, 2: 0, 3: 0}
for ex in examples:
    final_dist[int(ex['output'])] += 1
print(f"Final distribution: {final_dist}")

# Subsample to exactly 1000
if len(examples) > 1000:
    examples = random.sample(examples, 1000)

# Shuffle with seed
random.seed(42)
random.shuffle(examples)

print(f"Final count: {len(examples)}")

# Assertions
if len(examples) < 1000:
    print(f"WARNING: Only {len(examples)} examples generated (limited by word count for higher e-counts)")
assert len(examples) <= 1000, f"Expected <= 1000, got {len(examples)}"
assert len(examples) >= 900, f"Expected >= 900, got {len(examples)}"  # Allow some margin
input_list = [ex['input'] for ex in examples]
assert len(input_list) == len(set(input_list)), "Duplicate inputs found"

# Self-check
for ex in examples:
    count = ex['input'].count('e')
    expected_output = str(count)
    assert expected_output == ex['output'], f"Output {ex['output']} doesn't match e-count {count} for '{ex['input']}'"

print("All assertions passed!")

# Write to file
output_file = 'dataset_files/extended_tasks/count_letter_e.json'
with open(output_file, 'w') as f:
    json.dump(examples, f)
print(f"Wrote {len(examples)} examples to {output_file}")
