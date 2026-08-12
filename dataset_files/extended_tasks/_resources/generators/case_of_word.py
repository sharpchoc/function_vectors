#!/usr/bin/env python3
"""
Generator for case_of_word task.
Word in upper or lower case -> 'upper' or 'lower' label.
"""

import json
import random

# Read resource file
with open('dataset_files/extended_tasks/_resources/common_words.txt') as f:
    words = [line.strip().lower() for line in f if line.strip() and line.strip().isalpha() and len(line.strip()) >= 3]

print(f"Available words: {len(words)}")

random.seed(42)
examples = []
seen_inputs = set()

# Generate examples: 50/50 upper/lower
# Each word appears in at most one case form
target_upper = 500
target_lower = 500
count_upper = 0
count_lower = 0

for word in words:
    if count_upper >= target_upper and count_lower >= target_lower:
        break

    # Randomly choose case with replacement until we hit targets
    if count_upper < target_upper and count_lower < target_lower:
        choice = random.choice(['upper', 'lower'])
    elif count_upper < target_upper:
        choice = 'upper'
    elif count_lower < target_lower:
        choice = 'lower'
    else:
        break

    if choice == 'upper':
        input_str = word.upper()
        output = 'upper'
        count_upper += 1
    else:
        input_str = word
        output = 'lower'
        count_lower += 1

    if input_str not in seen_inputs:
        examples.append({
            "input": input_str,
            "output": output
        })
        seen_inputs.add(input_str)

print(f"Generated {len(examples)} examples")
print(f"Upper: {count_upper}, Lower: {count_lower}")

# If we don't have enough, generate more
if len(examples) < 1000:
    remaining = 1000 - len(examples)
    for _ in range(remaining):
        word = random.choice(words)
        choice = random.choice(['upper', 'lower'])
        if choice == 'upper':
            input_str = word.upper()
            output = 'upper'
        else:
            input_str = word
            output = 'lower'

        if input_str not in seen_inputs:
            examples.append({
                "input": input_str,
                "output": output
            })
            seen_inputs.add(input_str)

print(f"After padding: {len(examples)} examples")

# Subsample to exactly 1000
if len(examples) > 1000:
    examples = random.sample(examples, 1000)

# Shuffle with seed
random.seed(42)
random.shuffle(examples)

print(f"Final count: {len(examples)}")

# Count balance
upper_count = sum(1 for ex in examples if ex['output'] == 'upper')
lower_count = sum(1 for ex in examples if ex['output'] == 'lower')
print(f"Final balance: upper={upper_count}, lower={lower_count}")

# Assertions
assert len(examples) == 1000, f"Expected 1000, got {len(examples)}"
input_list = [ex['input'] for ex in examples]
assert len(input_list) == len(set(input_list)), "Duplicate inputs found"

# Self-check
for ex in examples:
    is_upper = ex['input'].isupper()
    expected_output = 'upper' if is_upper else 'lower'
    assert expected_output == ex['output'], f"Output {ex['output']} doesn't match input case for {ex['input']}"

print("All assertions passed!")

# Write to file
output_file = 'dataset_files/extended_tasks/case_of_word.json'
with open(output_file, 'w') as f:
    json.dump(examples, f)
print(f"Wrote {len(examples)} examples to {output_file}")
