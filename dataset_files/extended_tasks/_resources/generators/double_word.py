#!/usr/bin/env python3
"""
Generator for double_word task.
Word -> word written twice, space-separated.
"""

import json
import random

# Read resource file
with open('dataset_files/extended_tasks/_resources/common_words.txt') as f:
    words = [line.strip().lower() for line in f if line.strip() and line.strip().isalpha()]

# Filter: alphabetic, length 3-7
words = [w for w in words if 3 <= len(w) <= 7]

print(f"Available words: {len(words)}")

random.seed(42)
examples = []
seen_inputs = set()

# Generate examples
for word in words:
    if len(examples) >= 1000:
        break

    if word not in seen_inputs:
        input_str = word
        output = f"{word} {word}"
        examples.append({
            "input": input_str,
            "output": output
        })
        seen_inputs.add(input_str)

print(f"Generated {len(examples)} examples")

# If we don't have enough, generate from all words with length check
if len(examples) < 1000:
    all_words = []
    with open('dataset_files/extended_tasks/_resources/common_words.txt') as f:
        all_words = [line.strip().lower() for line in f if line.strip() and line.strip().isalpha() and 3 <= len(line.strip()) <= 7]

    for word in all_words:
        if len(examples) >= 1000:
            break
        if word not in seen_inputs:
            input_str = word
            output = f"{word} {word}"
            examples.append({
                "input": input_str,
                "output": output
            })
            seen_inputs.add(input_str)

print(f"After generation: {len(examples)} examples")

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
    parts = ex['output'].split()
    assert len(parts) == 2, f"Output should be 2 words, got {len(parts)}"
    assert parts[0] == parts[1], f"Two words should be identical"
    assert parts[0] == ex['input'], f"Output word '{parts[0]}' doesn't match input '{ex['input']}'"

print("All assertions passed!")

# Write to file
output_file = 'dataset_files/extended_tasks/double_word.json'
with open(output_file, 'w') as f:
    json.dump(examples, f)
print(f"Wrote {len(examples)} examples to {output_file}")
