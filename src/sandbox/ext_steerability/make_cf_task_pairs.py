#!/usr/bin/env python
"""Counterfactual task pairing for the bottom-up read-feature ablation baselines.

For the specificity control we ablate task A's prompts with the read direction of an
A-PRIORI clearly different task B (user decision 2026-08-19: human-legible difference,
e.g. numerical vs country->capital — NOT a cosine-similarity criterion). A cheap LLM
(via OpenRouter) assigns each of the 69 tasks one semantic family from a fixed list;
each task is then paired with a deterministically sampled task from a DIFFERENT family.

Output:
  artifacts/69_task_run/bottom_up_ablation/cf_task_pairs.json   (canonical, read by the
                                                                 ablation run script)
  results/69_task_run/bottom_up_read_features/ablation/cf_task_pairs.csv  (reviewable)

Self-checks: every task appears exactly once as A; no pair shares a family; every
family has >= 2 members outside itself to draw from.
"""
import csv
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import requests

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT, TASK69_RUN_DIR  # noqa: E402

FAMILIES = [
    "numerical",              # arithmetic / digit sequences / counting
    "translation",            # between natural languages
    "morphology_linguistic",  # inflection, POS conversion, tense, plurals, synonym/antonym
    "world_knowledge",        # factual lookup: capitals, currencies, landmarks, people
    "text_classification",    # sentiment / topic / category labels
    "string_manipulation",    # letter-level operations, casing, spelling
]
MODEL = "anthropic/claude-haiku-4.5"
SPLIT = REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"
OUT_JSON = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "cf_task_pairs.json"
OUT_CSV = TASK69_RUN_DIR / "bottom_up_read_features" / "ablation" / "cf_task_pairs.csv"


def stable_rng(*parts):
    return np.random.default_rng(zlib.crc32("::".join(map(str, parts)).encode()))


def task_examples(task, k=3):
    recs = json.load(open(REPO_ROOT / "dataset_files" / "isolation_prompts_ext" / task
                          / "train_prompts.json"))
    demos = recs[0]["demos"][:k]
    return [(str(d["input"]), str(d["output"])) for d in demos]


def ask_llm(tasks):
    key = Path("~/.openrouter_key").expanduser().read_text().strip().split("=")[-1].strip()
    lines = []
    for t in tasks:
        ex = "; ".join(f"{i!r} -> {o!r}" for i, o in task_examples(t))
        lines.append(f"- {t}: {ex}")
    prompt = (
        "Below are in-context-learning tasks, each with 3 example input->output pairs. "
        "Assign each task EXACTLY ONE semantic family from this fixed list:\n"
        + ", ".join(FAMILIES) + "\n\n"
        "Judge by what the task fundamentally is about, e.g. next_number_digits is "
        "numerical, country->capital is world_knowledge, antonym is morphology_linguistic, "
        "uppercasing is string_manipulation.\n\nTasks:\n" + "\n".join(lines) +
        "\n\nAnswer with ONLY a JSON object mapping every task name to its family, "
        "no other text."
    )
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": MODEL, "temperature": 0,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=180,
    )
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"].strip()
    txt = txt[txt.index("{"): txt.rindex("}") + 1]
    fam = json.loads(txt)
    missing = [t for t in tasks if t not in fam]
    bad = {t: f for t, f in fam.items() if f not in FAMILIES}
    assert not missing, f"LLM omitted tasks: {missing}"
    assert not bad, f"LLM used unknown families: {bad}"
    return {t: fam[t] for t in tasks}


def main():
    split = json.load(open(SPLIT))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    assert len(tasks) == 69
    fam = ask_llm(tasks)

    from collections import Counter
    counts = Counter(fam.values())
    print("family sizes:", dict(counts))
    pairs = {}
    for t in tasks:
        cands = [b for b in tasks if fam[b] != fam[t]]
        assert len(cands) >= 2, f"{t}: too few out-of-family candidates"
        pairs[t] = str(stable_rng("cf_pair", t).choice(cands))
    assert set(pairs) == set(tasks)
    assert all(fam[a] != fam[b] for a, b in pairs.items())

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"model": MODEL, "families": fam, "pairs": pairs}, f, indent=2)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "family", "cf_task", "cf_family"])
        for t in tasks:
            w.writerow([t, fam[t], pairs[t], fam[pairs[t]]])
    print(f"wrote {OUT_JSON}\n      {OUT_CSV}")
    for t in tasks[:8]:
        print(f"  {t} ({fam[t]})  ->  {pairs[t]} ({fam[pairs[t]]})")


if __name__ == "__main__":
    main()
