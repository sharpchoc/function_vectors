#!/usr/bin/env python
"""Dump readable sampled generations per prompt format (chat-template transfer study)
into results/chat_template_transfer/ext117_6shot_accuracy/debug/sampled_responses.md.

For each selected task: the same queries across all three arms (plain / chat_blank_system /
chat_no_system), favouring prompts where plain succeeded and chat failed (the interesting
disagreements), plus a few chat-wins tasks for contrast. Also renders one full example
prompt per format so the exact model input is visible.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.paths import ARTIFACTS_ROOT, CHAT_TEMPLATE_TRANSFER_DIR  # noqa: E402

IN_ROOT = ARTIFACTS_ROOT / "chat_template_transfer" / "ext117_6shot"
OUT = CHAT_TEMPLATE_TRANSFER_DIR / "ext117_6shot_accuracy" / "debug" / "sampled_responses.md"
ARMS = ["plain", "chat_blank_system", "chat_no_system"]

# tasks pruned (<30%) in both chat arms but kept in plain, plus contrast tasks where chat wins
FAIL_TASKS = ["contains_letter_e", "ends_with_ing", "word_length", "first_vowel", "count_zeros",
              "count_consonants", "last_two_letters", "next_in_group", "person-instrument", "synonym"]
WIN_TASKS = ["last_digit", "french_noun_gender", "capitalize_last_letter"]
N_EX = 6
PROMPT_TASK = "ends_with_ing"


def load(arm, task):
    return {r["i"]: r for r in json.load(open(IN_ROOT / arm / f"{task}.json"))}


def fence(s):
    return s.replace("`", "'")


def example_block(recs_by_arm, i):
    c = recs_by_arm["plain"][i]
    lines = [f"**q = `{fence(c['query'])}`** — gold `{fence(c['gold'][0])}`\n"]
    for arm in ARMS:
        r = recs_by_arm[arm][i]
        mark = "✓" if r["match"] else "✗"
        gen = fence(r["generation"].strip()[:140]) or "(empty)"
        lines.append(f"- {mark} `{arm}`: `{gen}`")
    return "\n".join(lines) + "\n"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    md = ["# Debug: sampled generations per prompt format\n",
          "Qwen2.5-7B-Instruct, 6-shot, T=1.0 sampled, 50 prompts/task; metric = first line,",
          "stripped, exact match. Same demo/query sampling in every arm. Generations truncated",
          "to 140 chars. Regenerate with `src/eval_scripts/dump_chat_template_debug_examples.py`.\n"]

    # full example prompts (tokenizer-rendered, no GPU)
    from eval_scripts.eval_chat_template_ext117 import build_specs, render_prompt
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    spec = build_specs(PROMPT_TASK, 6, 1)[0]
    md.append(f"\n## Full example prompts (task `{PROMPT_TASK}`, prompt i=0)\n")
    for arm in ARMS:
        p = render_prompt(arm, tok, spec["demos"], spec["qt"])
        md.append(f"### {arm}\n\n```\n{p}\n```\n")

    md.append("\n## Tasks pruned (<30%) under BOTH chat arms but kept under plain\n")
    md.append("Examples favour prompts where plain succeeded and chat_no_system failed.\n")
    for task in FAIL_TASKS:
        recs = {arm: load(arm, task) for arm in ARMS}
        accs = " / ".join(f"{arm} {sum(r['match'] for r in recs[arm].values())/50:.2f}" for arm in ARMS)
        md.append(f"\n### {task}  ({accs})\n")
        idx = [i for i in sorted(recs["plain"]) if recs["plain"][i]["match"]
               and not recs["chat_no_system"][i]["match"]][:N_EX]
        idx += [i for i in sorted(recs["plain"]) if i not in idx][: N_EX - len(idx)]
        for i in idx:
            md.append(example_block(recs, i))

    md.append("\n## Contrast: tasks where the chat arms BEAT plain\n")
    md.append("Examples favour prompts where chat_no_system succeeded and plain failed.\n")
    for task in WIN_TASKS:
        recs = {arm: load(arm, task) for arm in ARMS}
        accs = " / ".join(f"{arm} {sum(r['match'] for r in recs[arm].values())/50:.2f}" for arm in ARMS)
        md.append(f"\n### {task}  ({accs})\n")
        idx = [i for i in sorted(recs["plain"]) if not recs["plain"][i]["match"]
               and recs["chat_no_system"][i]["match"]][:N_EX]
        for i in idx:
            md.append(example_block(recs, i))

    OUT.write_text("\n".join(md))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
