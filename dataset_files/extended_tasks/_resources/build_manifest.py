#!/usr/bin/env python
"""Write extended_tasks manifest + README after validation passes."""
import json
from collections import Counter
from pathlib import Path

RES = Path(__file__).resolve().parent
ROOT = RES.parent
specs = {s["name"]: s for s in json.load(open(RES / "new_task_specs.json"))["specs"]}
originals = sorted(p.stem for p in (ROOT.parent / "abstractive").glob("*.json"))

manifest = {"n_tasks": 0, "n_original": 0, "n_new": 0, "tasks": {}}
for f in sorted(ROOT.glob("*.json")):
    if f.stem == "manifest":
        continue
    d = json.load(open(f))
    is_new = f.stem in specs
    manifest["tasks"][f.stem] = {
        "n": len(d),
        "origin": "new" if is_new else "original_abstractive",
        **({"rule": specs[f.stem]["rule"], "lane": specs[f.stem]["lane"],
            "generation_method": specs[f.stem]["generation_method"]} if is_new else {}),
    }
    manifest["n_tasks"] += 1
    manifest["n_new"] += is_new
    manifest["n_original"] += not is_new

assert manifest["n_new"] == 100, manifest["n_new"]
assert manifest["n_original"] == len(originals), (manifest["n_original"], len(originals))
json.dump(manifest, open(ROOT / "manifest.json", "w"), indent=1)

lanes = Counter(specs[n]["lane"] for n in specs)
readme = f"""# extended_tasks

{manifest['n_tasks']} ICL word-pair tasks: the {manifest['n_original']} original abstractive tasks
(copied verbatim from `dataset_files/abstractive/`) plus **100 new tasks with exactly 1000
examples each**, generated 2026-08 (multi-agent ideation -> curation -> generation -> validation).

- Every new task: deterministic single mapping, unique inputs, short outputs, domain chosen so
  1000 examples exist comfortably (small-domain relations like country-capital were excluded by
  design). See `manifest.json` for per-task rule/origin, `_resources/new_task_specs.json` for
  full specs, `_resources/generators/` for the per-task generation scripts (knowledge tasks
  embed their curated fact lists there), and `_resources/validation_report.json` for the
  structural validation results. Curation decisions: `_resources/finalize_specs.py`; post-hoc
  repairs: `_resources/repairs_round1.py`.
- New-task lanes: {dict(lanes)}
- NOT yet ICL-filtered on GPT-J: run the usual correctness filter before using any task for
  head selection / FV construction.
"""
(ROOT / "README.md").write_text(readme)
print(f"manifest: {manifest['n_tasks']} tasks ({manifest['n_new']} new); README written")
