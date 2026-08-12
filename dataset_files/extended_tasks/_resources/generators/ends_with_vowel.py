#!/usr/bin/env python3
"""Generator for ends_with_vowel task."""
import random
import json

def generate():
    with open('/workspace/function_vectors/.claude/worktrees/sparse-heads-fv23/dataset_files/extended_tasks/_resources/common_words.txt', 'r') as f:
        words = [line.strip() for line in f if line.strip()]

    words = [w for w in words if w.isalpha() and len(w) >= 3 and not w.endswith('y')]
    vowel_words = [w for w in words if w[-1] in 'aeiou']
    consonant_words = [w for w in words if w[-1] not in 'aeiou']

    min_size = min(len(vowel_words), len(consonant_words))
    target = min(500, min_size)

    random.seed(42)
    v_sample = random.sample(vowel_words, min(target, len(vowel_words)))
    c_sample = random.sample(consonant_words, min(target, len(consonant_words)))

    dataset = [{"input": w, "output": "vowel"} for w in v_sample]
    dataset += [{"input": w, "output": "consonant"} for w in c_sample]

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
        expected = "vowel" if w[-1] in 'aeiou' else "consonant"
        assert item["output"] == expected

    assert len(dataset) == 1000
    assert len(set(i["input"] for i in dataset)) == 1000
    
    return dataset

if __name__ == "__main__":
    dataset = generate()
    with open('/workspace/function_vectors/.claude/worktrees/sparse-heads-fv23/dataset_files/extended_tasks/ends_with_vowel.json', 'w') as f:
        json.dump(dataset, f)
