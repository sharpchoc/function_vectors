#!/usr/bin/env python
"""Presentation visual: catalog of the 17 style properties.

One row per property: family, a short example rendered by the ACTUAL toggler in both
polarities (highlighted spans = what changes), and what one unit of k counts.
Pool status from task_splits/style_properties_pool.json. No measured numbers.

Output: results/style_properties/explainer_visuals/property_catalog.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import REPO_ROOT, STYLE_PROPERTIES_DIR
from src.sandbox.ext_styleprops.properties import PROPS, render

OUT = STYLE_PROPERTIES_DIR / "explainer_visuals"
POOL = REPO_ROOT / "task_splits" / "style_properties_pool.json"
HI = {"nat": "#cfe8ff", "alt": "#ffd9b3"}

# (property, example in the nat convention, what one k counts)
ROWS = [
    ("case", [
        ("sentence_caps", "The market opens early. Vendors arrive first.", "each sentence start"),
        ("all_caps", "The market opens early. Vendors arrive first.", "each sentence"),
    ]),
    ("spelling", [
        ("us_uk", "the color of the harbor wall", "each lexicon word"),
        ("ise_ize", "they organize, then realize", "each lexicon word"),
        ("brit_t_past", "she learned and spelled it", "each lexicon word"),
        ("whilst", "while among friends", "each lexicon word"),
    ]),
    ("typography", [
        ("double_space", "It rained. We stayed in.", "each sentence boundary"),
        ("oxford_comma", "maps, books, and letters", "each 3+ item list"),
        ("curly_quotes", '"Come in," she said.', "each quote mark"),
        ("quote_punct", '"Come in," she said.', "each quote/punct juncture"),
        ("em_dash", "the plan—such as it was—failed", "each dash"),
        ("ellipsis", "It was late... too late.", "each ellipsis"),
    ]),
    ("number", [
        ("num_words", "about 14 boxes on 3 shelves", "each cardinal 2-20"),
        ("percent_sign", "roughly 60% of the boxes", "each percentage"),
        ("ordinal_words", "the 3rd of May, the 1st time", "each ordinal 1st-10th"),
    ]),
    ("lexical", [
        ("contractions", "it's late and we don't mind", "each contraction"),
        ("ampersand", "maps and letters and pens", "each 'and'"),
    ]),
]


def display_spans(text, spans):
    """Widen a toggled span to its whole word when it is a partial alphabetic span
    (sentence_caps toggles one letter; the evidence TOKEN is the whole word)."""
    out = []
    for s, e in spans:
        seg = text[s:e]
        if seg.isalpha() and ((s > 0 and text[s - 1].isalpha()) or
                              (e < len(text) and text[e].isalpha())):
            while s > 0 and text[s - 1].isalpha():
                s -= 1
            while e < len(text) and text[e].isalpha():
                e += 1
        out.append((s, e))
    return out


def draw_hl_text(ax, x, y, text, spans, color, char_w, fs=10.5):
    for s, e in spans:
        ax.add_patch(Rectangle((x + s * char_w, y - 0.95), (e - s) * char_w, 1.9,
                               fc=color, ec="none", zorder=1))
    ax.text(x, y, text.replace("  ", "␣␣"), family="monospace", fontsize=fs, va="center",
            zorder=2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pool = json.load(open(POOL))
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.text(0.5, 0.96, "The 17 style properties", ha="center", fontsize=20,
             fontweight="bold")
    fig.text(0.5, 0.925, "Each property is a binary convention: nat (US-standard / plain) "
             "vs alt (the toggled form). Highlights show what the toggler changes.",
             ha="center", fontsize=11.5)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    probe = ax.text(0, 0, "M" * 50, family="monospace", fontsize=10.5)
    bb = probe.get_window_extent(renderer=rend)
    inv = ax.transData.inverted()
    (xa, _), (xb, _) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
    probe.remove()
    char_w = (xb - xa) / 50

    cols = {"name": 2, "nat": 17, "alt": 47, "k": 76, "pool": 90}
    y = 85
    ax.text(cols["name"], y, "property", fontsize=10, fontweight="bold", color="#555")
    ax.text(cols["nat"], y, "nat example", fontsize=10, fontweight="bold", color="#555")
    ax.text(cols["alt"], y, "alt example", fontsize=10, fontweight="bold", color="#555")
    ax.text(cols["k"], y, "one k =", fontsize=10, fontweight="bold", color="#555")
    ax.text(cols["pool"], y, "in pool", fontsize=10, fontweight="bold", color="#555")
    y -= 3.2
    row_h = 3.55
    for family, rows in ROWS:
        ax.text(cols["name"], y, family.upper(), fontsize=8.5, color="#888", va="center",
                fontweight="bold")
        y -= 2.4
        for name, example, kunit in rows:
            prop = PROPS[name]
            opps = prop.find_opps(example)
            nat, sn = render(example, opps, "nat")
            alt, sa = render(example, opps, "alt")
            ax.text(cols["name"], y, name, fontsize=10.5, va="center", fontweight="bold")
            draw_hl_text(ax, cols["nat"], y, nat, display_spans(nat, sn), HI["nat"], char_w)
            draw_hl_text(ax, cols["alt"], y, alt, display_spans(alt, sa), HI["alt"], char_w)
            ax.text(cols["k"], y, kunit, fontsize=9.5, va="center", color="#333")
            ok = name in pool["pass"]
            if ok:
                label = "yes"
            elif name in pool.get("pruned", {}):
                label = "no (pruned)"
            else:
                label = "no (failed)"
            ax.text(cols["pool"], y, label, fontsize=9.5, va="center",
                    color="#2a9d8f" if ok else "#c0392b")
            y -= row_h
        y -= 1.0
    fig.text(0.5, 0.03, f"Pool = the {len(pool['pass'])} properties GPT-J demonstrably "
             "follows in context (behavioral pre-screen); whilst failed the screen, "
             "ellipsis was pruned (one-sided classifier).", ha="center", fontsize=10,
             color="#444")
    fig.savefig(OUT / "property_catalog.png", dpi=150)
    print(OUT / "property_catalog.png")


if __name__ == "__main__":
    main()
