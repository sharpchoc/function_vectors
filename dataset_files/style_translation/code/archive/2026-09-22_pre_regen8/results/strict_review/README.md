# Strict two-reviewer review of every never-reviewed pool document (2026-09-22)

All 7,856 documents of the 55 pool families that no reviewer had seen (the Gemini-2.5-Flash-era pairs; regenerated documents already carry
`reviewed_by`) were judged by Opus 5 and GPT-5 with the regeneration's 14-item checklist (`tmp/regen_common.py` REVIEW), AND rule.
204 came from audit 9's sample, 7,652 from `tmp/review10.py` (cost ≈ $650 + $17). Verdicts: `verdicts.csv` (per doc: ok per reviewer, items cited by
each and by both), `by_family.csv`, `summary.json`, `items_by_family.png`. Raw verdicts with quoted lines: `tmp/review10/verdicts.jsonl`, `tmp/audit9/sample_review.jsonl`.

Accepted by both: 367 / 7,856 (4.7 %); Opus alone 679, GPT-5 alone 1,624. Items cited by BOTH (share of docs): padding 48 %, fake decision
(two occurrences on one line) 13 %, wrong task 7 %, inconsistent convention 6 %, scope creep 5 %, extra change 3.5 %, invalid 2.7 %.
Per-family acceptance ranges from 0 % (18 families, e.g. bash_subst, hex_constants, py2_except, py_literal_ctor) to 23 % (py_indent, comment_language, c_comment_style).
