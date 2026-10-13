# Accuracy by k on the code-translation prompts (35 Python convention families, Qwen2.5-7B base)

2026-10-14. Model key `qwen25_code_tr` (same weights as `qwen25_code`). Prompt = `Task:` description, the document's accepted source program
in JavaScript / Go / Java / Ruby (`results/code_styles/translation/`, 6,911 of 7,000 documents), then `Python translation:` and the Python twin cut at
the SAME cues as the original prompts (k = 0..4, both conventions; `build_translation_prompts.py`). Rollouts `rollout.py --model qwen25_code_tr`
(one T = 1 sample, 48 tokens), convention scored as before, Gemini judge; `analyze.py --model qwen25_code_tr` → `summary.csv`, `accuracy_by_k.png`;
`translation_prompts_compare.py` → `comparison.csv`, `pooled_by_k.csv`, `per_family_k4.csv`, `accuracy_by_k_compare.png`, `compare_summary.json`.

Pooled over the 35 families (accuracy = convention followed AND judge OK):

| k | alt, source program | alt, original | nat, source program | nat, original |
|---|---|---|---|---|
| 0 | .17 | .14 | .65 | .42 |
| 1 | .77 | .64 | .91 | .74 |
| 2 | .82 | .70 | .93 | .78 |
| 3 | .85 | .73 | .94 | .79 |
| 4 | .85 | .73 | .94 | .78 |

- The source program removes most of the content entropy: at k = 4 the unscorable rate falls from .12 to .04 and judge-OK rises from .89 to .99;
  style_ok (convention followed, ignoring the judge) is .86 / .95 (alt / nat) against .80 / .86.
- Per family at k = 4: 19 families reach ≥ .90 towards the alternative (3 before), 17 reach ≥ .90 on both poles; 23 families improve by ≥ .10
  towards the alternative, none gets worse by ≥ .10. Lowest alternative accuracies: py_not_in .50, py_comprehension .58, comment_language .60.
- 0-shot: the model's default preference is unchanged in direction (alt .17, nat .65) but much more often scorable, as intended: with the program
  pinned down, the completion is almost always one of the two conventions.
