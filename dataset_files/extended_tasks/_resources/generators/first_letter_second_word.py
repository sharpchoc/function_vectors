#!/usr/bin/env python3
"""
Generator for first_letter_second_word task.
Two-word phrase (adj noun) -> first letter of SECOND word, lowercase.
"""

import json
import random

# Read resource files
with open('dataset_files/extended_tasks/_resources/common_adjs.txt') as f:
    adjs = [line.strip().lower() for line in f if line.strip() and line.strip().isalpha()]

with open('dataset_files/extended_tasks/_resources/common_nouns.txt') as f:
    nouns = [line.strip().lower() for line in f if line.strip() and line.strip().isalpha()]

# Filter: keep top items, reasonable length (3-8 chars)
adjs = [w for w in adjs[:1500] if 3 <= len(w) <= 8]
nouns = [w for w in nouns[:2000] if 3 <= len(w) <= 8]

print(f"Available adjs: {len(adjs)}, nouns: {len(nouns)}")

# Generate bigrams
random.seed(42)
examples = []
seen_inputs = set()

# Generate ~20000 unique bigrams and filter
for _ in range(30000):
    adj = random.choice(adjs)
    noun = random.choice(nouns)
    input_str = f"{adj} {noun}"

    if input_str not in seen_inputs:
        output = noun[0].lower()
        examples.append({
            "input": input_str,
            "output": output
        })
        seen_inputs.add(input_str)

        if len(examples) >= 1200:
            break

print(f"Generated {len(examples)} examples")

# Stratify to ensure >=15 distinct output letters
output_counts = {}
for ex in examples:
    out = ex['output']
    output_counts[out] = output_counts.get(out, 0) + 1

print(f"Output letter coverage: {len(output_counts)} distinct letters")
print(f"Output distribution: {sorted(output_counts.items())}")

if len(output_counts) < 15:
    print(f"WARNING: Only {len(output_counts)} distinct output letters (need >=15)")
    # Try to improve by targeting specific first letters
    needed_letters = set()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        if letter not in output_counts:
            needed_letters.add(letter)

    if needed_letters:
        nouns_by_letter = {}
        for noun in nouns:
            letter = noun[0].lower()
            if letter not in nouns_by_letter:
                nouns_by_letter[letter] = []
            nouns_by_letter[letter].append(noun)

        for needed_letter in needed_letters:
            if needed_letter in nouns_by_letter and len(nouns_by_letter[needed_letter]) > 0:
                for _ in range(5):
                    adj = random.choice(adjs)
                    noun = random.choice(nouns_by_letter[needed_letter])
                    input_str = f"{adj} {noun}"
                    if input_str not in seen_inputs:
                        examples.append({
                            "input": input_str,
                            "output": needed_letter
                        })
                        seen_inputs.add(input_str)
                        break

print(f"After stratification: {len(examples)} examples")

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
    words = ex['input'].split()
    assert len(words) == 2, f"Expected 2 words, got {len(words)} in '{ex['input']}'"
    assert words[1].startswith(ex['output']), f"Output {ex['output']} doesn't match second word {words[1]}"

print("All assertions passed!")

# Write to file
output_file = 'dataset_files/extended_tasks/first_letter_second_word.json'
with open(output_file, 'w') as f:
    json.dump(examples, f)
print(f"Wrote {len(examples)} examples to {output_file}")
