#!/usr/bin/env python3
"""
nato_first_letter: Word -> NATO phonetic-alphabet codeword for first letter.
Exclude 'a' and 'j' to avoid spelling ambiguity.
"""

import json
import random

# NATO phonetic alphabet table (excluding a and j)
nato_table = {
    'b': 'Bravo',
    'c': 'Charlie',
    'd': 'Delta',
    'e': 'Echo',
    'f': 'Foxtrot',
    'g': 'Golf',
    'h': 'Hotel',
    'i': 'India',
    'k': 'Kilo',
    'l': 'Lima',
    'm': 'Mike',
    'n': 'November',
    'o': 'Oscar',
    'p': 'Papa',
    'q': 'Quebec',
    'r': 'Romeo',
    's': 'Sierra',
    't': 'Tango',
    'u': 'Uniform',
    'v': 'Victor',
    'w': 'Whiskey',
    'x': 'X-ray',
    'y': 'Yankee',
    'z': 'Zulu',
}

# Read common_words.txt
with open('dataset_files/extended_tasks/_resources/common_words.txt') as f:
    all_words = [w.strip() for w in f.readlines() if w.strip()]

# Filter: alphabetic words, not starting with a or j
words = [
    w for w in all_words
    if w.isalpha() and len(w) >= 3 and w[0] not in 'aj'
]

# Generate dataset
dataset = []
seen_inputs = set()

for word in words:
    if word in seen_inputs:
        continue
    first_letter = word[0]
    if first_letter not in nato_table:
        continue

    output = nato_table[first_letter]

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
    # Check: first letter NATO lookup
    first_letter = word[0]
    assert first_letter in nato_table, f"First letter {first_letter} not in NATO table"
    assert output == nato_table[first_letter], f"Output mismatch for {word}"
    # Check: output[0].lower() == input[0]
    assert output[0].lower() == word[0], f"Output doesn't match first letter for {word}"

# Write output
with open('dataset_files/extended_tasks/nato_first_letter.json', 'w') as f:
    json.dump(dataset, f)

print(f"✓ nato_first_letter: {len(dataset)} examples, {len(set(inputs))} unique inputs, {len(set(d['output'] for d in dataset))} unique outputs")
