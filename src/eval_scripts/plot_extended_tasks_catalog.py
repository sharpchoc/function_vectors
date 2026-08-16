#!/usr/bin/env python
"""Catalog figure of ALL extended_tasks: name, short description, one illustrative example,
6-shot accuracy — grouped by lane, 3 columns, one tall PNG for pasting into a write-up.
Tasks under the 30% pruning threshold are greyed and tagged 'pruned'.

Writes results/general/extended_tasks_nshot_sweep/extended_tasks_catalog.png
"""
import csv
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.paths import GENERAL_DIR  # noqa: E402

ET = REPO_ROOT / "dataset_files" / "extended_tasks"
OUT = GENERAL_DIR / "extended_tasks_nshot_sweep" / "extended_tasks_catalog.png"
PRUNE_AT = 0.30
C_NEW, C_ORIG, C_PRUNED = "#2a78d6", "#9a5b22", "#9aa1a9"
INK, INK2 = "#0b0b0b", "#52514e"

ORIG_DESC = {
    "ag_news": "News snippet -> section (Business/Science/Sports/World).",
    "antonym": "Word -> its antonym.",
    "capitalize": "Word -> same word, first letter capitalized.",
    "capitalize_first_letter": "Word -> its first letter, uppercased.",
    "capitalize_last_letter": "Word -> its last letter, uppercased.",
    "capitalize_second_letter": "Word -> its second letter, uppercased.",
    "commonsense_qa": "Multiple-choice question -> letter of correct option.",
    "count_consonants": "Word -> number of consonants.",
    "count_vowels": "Word -> number of vowels.",
    "country-capital": "Country -> capital city.",
    "country-currency": "Country -> official currency.",
    "east_neighbor": "Country -> neighboring country to its east.",
    "west_neighbor": "Country -> neighboring country to its west.",
    "english-french": "English word -> French translation.",
    "english-german": "English word -> German translation.",
    "english-spanish": "English word -> Spanish translation.",
    "landmark-country": "Landmark/institution -> its country.",
    "lowercase_first_letter": "ALL-CAPS word -> first letter, lowercased.",
    "lowercase_last_letter": "ALL-CAPS word -> last letter, lowercased.",
    "national_parks": "US park/monument -> its state.",
    "next_capital_letter": "Word -> alphabet successor of first letter, uppercased.",
    "next_in_group": "Element -> next element in periodic-table group.",
    "next_in_period": "Element -> next element in its period.",
    "next_item": "Sequence word -> next item in the sequence.",
    "prev_item": "Sequence word -> previous item in the sequence.",
    "next_number_digits": "Digit string -> next integer.",
    "prev_number_digits": "Digit string -> previous integer.",
    "park-country": "National park -> its country.",
    "person-instrument": "Musician -> instrument they play.",
    "person-occupation": "Person -> their occupation.",
    "person-sport": "Athlete -> sport they play.",
    "present-past": "Verb -> its simple past tense.",
    "product-company": "Product -> company behind it.",
    "rhyme": "Word -> a rhyming word.",
    "sentiment": "Review snippet -> positive or negative.",
    "singular-plural": "Singular noun -> its plural.",
    "synonym": "Word -> a synonym.",
    "word_length": "Word -> its number of characters.",
}
ORIG_LANE = {
    "orthographic": ["capitalize", "capitalize_first_letter", "capitalize_last_letter",
                     "capitalize_second_letter", "lowercase_first_letter", "lowercase_last_letter",
                     "next_capital_letter", "word_length", "count_consonants", "count_vowels"],
    "numeric_sequence": ["next_item", "prev_item", "next_number_digits", "prev_number_digits",
                         "next_in_group", "next_in_period"],
    "morphology_lexical": ["antonym", "synonym", "present-past", "singular-plural", "rhyme"],
    "knowledge_translation": ["ag_news", "commonsense_qa", "country-capital", "country-currency",
                              "east_neighbor", "west_neighbor", "english-french", "english-german",
                              "english-spanish", "landmark-country", "national_parks",
                              "park-country", "person-instrument", "person-occupation",
                              "person-sport", "product-company", "sentiment"],
}
LANE_OF = {t: lane for lane, ts in ORIG_LANE.items() for t in ts}
NEW_LANE_MAP = {"digit_query": "numeric_sequence", "counting": "numeric_sequence",
                "arithmetic": "numeric_sequence", "classification": "numeric_sequence",
                "formatting": "numeric_sequence", "comparison": "numeric_sequence",
                "dates": "numeric_sequence", "time": "numeric_sequence",
                "translation": "knowledge_translation", "word_classification": "knowledge_translation",
                "linguistic_knowledge": "knowledge_translation", "world_knowledge": "knowledge_translation",
                "semantic_classification": "knowledge_translation",
                "grammatical_classification": "morphology_lexical",
                "morphology-inflection": "morphology_lexical", "morphology-derivation": "morphology_lexical",
                "word-property": "morphology_lexical", "lexical-semantic": "morphology_lexical",
                "phonology": "morphology_lexical",
                "orthographic": "orthographic", "numeric_sequence": "numeric_sequence",
                "morphology_lexical": "morphology_lexical", "knowledge_translation": "knowledge_translation"}
LANE_LABEL = {"orthographic": "ORTHOGRAPHIC / CHARACTER OPERATIONS",
              "numeric_sequence": "NUMBERS, DATES & SEQUENCES",
              "morphology_lexical": "MORPHOLOGY & LEXICAL RELATIONS",
              "knowledge_translation": "WORLD KNOWLEDGE, TRANSLATION & CLASSIFICATION"}
LANE_ORDER = ["orthographic", "numeric_sequence", "morphology_lexical", "knowledge_translation"]


def pick_example(task):
    data = json.load(open(ET / f"{task}.json"))
    cands = sorted((p for p in data if len(str(p["input"])) <= 40), key=lambda p: len(str(p["input"])))
    p = cands[len(cands) // 3] if cands else data[0]
    i, o = str(p["input"]), str(p["output"])
    return (i[:37] + "…" if len(i) > 40 else i), (o[:22] + "…" if len(o) > 25 else o)


def main():
    specs = {s["name"]: s for s in json.load(open(ET / "_resources/new_task_specs.json"))["specs"]}
    manifest = json.load(open(ET / "manifest.json"))["tasks"]
    acc6 = {r["task"]: float(r["accuracy"]) for r in
            csv.DictReader(open(GENERAL_DIR / "extended_tasks_nshot_sweep" / "nshot_accuracy.csv"))
            if r["n_shots"] == "6"}

    entries = []
    for t in sorted(manifest):
        is_new = manifest[t]["origin"] == "new"
        lane = NEW_LANE_MAP[specs[t]["lane"]] if is_new else LANE_OF[t]
        desc = (specs[t]["rule"] if is_new else ORIG_DESC[t]).replace("->", "→")
        desc = textwrap.shorten(desc, width=118, placeholder="…")
        ex_i, ex_o = pick_example(t)
        entries.append({"task": t, "lane": lane, "new": is_new, "acc": acc6.get(t),
                        "desc": textwrap.wrap(desc, 58)[:2], "ex": f"{ex_i} → {ex_o}"})

    # flatten with lane headers, sorted by accuracy desc inside each lane
    items = []
    for lane in LANE_ORDER:
        sub = sorted((e for e in entries if e["lane"] == lane), key=lambda e: -(e["acc"] or 0))
        items.append({"header": f"{LANE_LABEL[lane]}  ({len(sub)})"})
        items.extend(sub)

    # measure heights (in text lines)
    def h(item):
        return 1.0 if "header" in item else 1.05 + len(item["desc"]) * 0.82 + 0.95

    total = sum(h(i) for i in items)
    ncols = 3
    per_col = total / ncols
    cols, cur, acc_h = [[]], 0, 0.0
    for it in items:
        if acc_h + h(it) > per_col * 1.02 and cur < ncols - 1 and "header" not in it:
            cols.append([])
            cur += 1
            acc_h = 0.0
        cols[cur].append(it)
        acc_h += h(it)

    col_lines = max(sum(h(i) for i in c) for c in cols)
    LINE_IN = 0.155
    fig_h = col_lines * LINE_IN + 1.1
    fig_w = 13.6
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#fcfcfb")
    fig.text(0.5, 1 - 0.25 / fig_h, f"extended_tasks catalog — {len(entries)} ICL tasks", ha="center",
             fontsize=13, fontweight="bold", color=INK)
    fig.text(0.5, 1 - 0.48 / fig_h,
             "rule, one dataset example, and GPT-J 6-shot accuracy (T=1.0 sampled, 50 prompts); "
             "grey = pruned (<30%); blue = new task, brown = original",
             ha="center", fontsize=8.5, color=INK2)

    x0s = [0.015, 0.352, 0.689]
    colw = 0.31
    top = 1 - 0.75 / fig_h
    for ci, col in enumerate(cols):
        y = top
        x = x0s[ci]
        for it in col:
            if "header" in it:
                y -= 0.55 * LINE_IN / fig_h
                fig.text(x, y, it["header"], fontsize=8.2, fontweight="bold", color=INK,
                         family="sans-serif")
                y -= 1.0 * LINE_IN / fig_h
                continue
            pruned = (it["acc"] or 0) < PRUNE_AT
            name_c = C_PRUNED if pruned else (C_NEW if it["new"] else C_ORIG)
            txt_c = C_PRUNED if pruned else INK
            tag = f'{(it["acc"] or 0) * 100:.0f}%' + ("  · pruned" if pruned else "")
            fig.text(x, y, it["task"], fontsize=7.6, fontweight="bold", family="monospace",
                     color=name_c)
            fig.text(x + colw, y, tag, fontsize=7.0, family="monospace", color=txt_c, ha="right")
            y -= 1.05 * LINE_IN / fig_h
            for line in it["desc"]:
                fig.text(x + 0.008, y, line, fontsize=6.9, color=txt_c)
                y -= 0.82 * LINE_IN / fig_h
            fig.text(x + 0.008, y, it["ex"], fontsize=6.9, family="monospace",
                     color=C_PRUNED if pruned else INK2)
            y -= 0.95 * LINE_IN / fig_h

    fig.savefig(OUT, dpi=200, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT} ({len(entries)} tasks)")


if __name__ == "__main__":
    main()
