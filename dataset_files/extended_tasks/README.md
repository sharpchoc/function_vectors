# extended_tasks

142 ICL word-pair tasks: the 42 original abstractive tasks
(copied verbatim from `dataset_files/abstractive/`) plus **100 new tasks with exactly 1000
examples each**, generated 2026-08 (multi-agent ideation -> curation -> generation -> validation).

- Every new task: deterministic single mapping, unique inputs, short outputs, domain chosen so
  1000 examples exist comfortably (small-domain relations like country-capital were excluded by
  design). See `manifest.json` for per-task rule/origin, `_resources/new_task_specs.json` for
  full specs, `_resources/generators/` for the per-task generation scripts (knowledge tasks
  embed their curated fact lists there), and `_resources/validation_report.json` for the
  structural validation results. Curation decisions: `_resources/finalize_specs.py`; post-hoc
  repairs: `_resources/repairs_round1.py`.
- New-task lanes: {'orthographic': 26, 'digit_query': 4, 'counting': 2, 'arithmetic': 6, 'classification': 2, 'formatting': 2, 'comparison': 2, 'dates': 9, 'time': 2, 'translation': 6, 'word_classification': 1, 'linguistic_knowledge': 3, 'world_knowledge': 3, 'semantic_classification': 6, 'grammatical_classification': 2, 'morphology-inflection': 10, 'morphology-derivation': 5, 'word-property': 3, 'lexical-semantic': 5, 'numeric_sequence': 1}
- NOT yet ICL-filtered on GPT-J: run the usual correctness filter before using any task for
  head selection / FV construction.
