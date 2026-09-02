#!/usr/bin/env python
"""Presentation visual: what the style-property documents actually look like.

Real excerpts from the base corpus (LLM-generated neutral prose), rendered as the
nat / alt twin pair for three properties, with the toggled spans highlighted.
Purely illustrative — no measured numbers.

Output: results/style_properties/explainer_visuals/corpus_examples.png
"""
import json
import re
import sys
import textwrap
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

BASE = REPO_ROOT / "dataset_files" / "style_properties" / "base_corpus.json"
OUT = STYLE_PROPERTIES_DIR / "explainer_visuals"
SHOW = ["us_uk", "sentence_caps", "curly_quotes", "double_space"]
WRAP = 62          # chars per line (monospace)
MAX_CHARS = 215    # excerpt length target (4 panels must fit one slide)
HI = {"nat": "#cfe8ff", "alt": "#ffd9b3"}


def pick_excerpt(prop, docs, used):
    """First unused doc whose first ~MAX_CHARS chars hold >=3 opportunities and wrap to
    <=5 lines; cut at a sentence end."""
    for d in docs:
        if d["doc_id"] in used:
            continue
        text = d["text"].replace("’", "'").replace("‘", "'")
        m = re.search(r"[.!?][\"”]?\s", text[MAX_CHARS:])
        cut = MAX_CHARS + m.end() if m else len(text)
        ex = text[:cut].strip()
        if len(prop.find_opps(ex)) >= 3 and "\n" not in ex \
                and len(textwrap.wrap(ex, WRAP)) <= 5:
            used.add(d["doc_id"])
            return ex, d["doc_id"]
    raise RuntimeError(prop.name)


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


def draw_text(ax, x0, y0, text, spans, color, char_w, line_h):
    """Monospace wrap; highlight char spans. Returns lines used."""
    lines, line_start = [], 0
    for ln in textwrap.wrap(text, WRAP, break_long_words=False, break_on_hyphens=False):
        idx = text.index(ln, line_start)
        lines.append((idx, ln))
        line_start = idx + len(ln)
    for li, (start, ln) in enumerate(lines):
        y = y0 - li * line_h
        for s, e in spans:
            a, b = max(s, start), min(e, start + len(ln))
            if a < b:
                ax.add_patch(Rectangle((x0 + (a - start) * char_w, y - 0.3 * line_h),
                                       (b - a) * char_w, 0.95 * line_h,
                                       fc=color, ec="none", zorder=1))
        ax.text(x0, y, ln.replace("  ", "␣␣"), family="monospace", fontsize=10.5,
                va="center", zorder=2)
    return len(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    docs = json.load(open(BASE))
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    fig.text(0.5, 0.962, "What the documents look like: one generated text → two twins",
             ha="center", fontsize=20, fontweight="bold")
    fig.text(0.5, 0.925,
             "758 neutral prose documents (~350 words, mundane topics) written by an LLM, then "
             "mechanically re-rendered so every occurrence\nof a property follows ONE "
             "convention. Highlights = the toggled spans (the evidence tokens).",
             ha="center", va="top", fontsize=11.5, linespacing=1.4)

    # measure the monospace char width / line height in axis units (no guessing)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    probe = ax.text(0, 0, "M" * 50, family="monospace", fontsize=10.5)
    bb = probe.get_window_extent(renderer=rend)
    inv = ax.transData.inverted()
    (xa, ya), (xb, yb) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
    probe.remove()
    char_w = (xb - xa) / 50
    line_h = (yb - ya) * 1.45
    col_x = {"nat": 6, "alt": 53}
    y = 82
    used = set()
    for name in SHOW:
        prop = PROPS[name]
        ex, doc_id = pick_excerpt(prop, docs, used)
        opps = prop.find_opps(ex)
        ax.text(2, y + 2.2, f"{name}", fontsize=13, fontweight="bold", va="center")
        ax.text(2 + len(name) * 0.95 + 1.5, y + 2.2,
                f"nat = {prop.nat_label}   |   alt = {prop.alt_label}   ({len(opps)} sites "
                f"in this excerpt; source doc {doc_id})", fontsize=10, va="center", color="#444")
        n_lines = 0
        for pol in ("nat", "alt"):
            txt, spans = render(ex, opps, pol)
            ax.text(col_x[pol], y - 0.3, f"{pol} twin", fontsize=10, style="italic",
                    color="#666", va="center")
            n = draw_text(ax, col_x[pol], y - 2.6, txt, display_spans(txt, spans), HI[pol], char_w, line_h)
            n_lines = max(n_lines, n)
        y -= 2.6 + n_lines * line_h + 3.6

    fig.text(0.5, 0.035,
             "Detectors match BOTH surface forms, so a document is 100% consistent after "
             "rendering whichever form the generator produced.   ␣␣ marks a double space.",
             ha="center", fontsize=10, color="#444")
    fig.savefig(OUT / "corpus_examples.png", dpi=150)
    print(OUT / "corpus_examples.png")


if __name__ == "__main__":
    main()
