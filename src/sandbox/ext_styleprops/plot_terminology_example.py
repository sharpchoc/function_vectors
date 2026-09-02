#!/usr/bin/env python
"""Explainer figure: style-property site terminology on a worked example.

Two panels (us_uk and sentence_caps), each showing the nat/alt minimal-pair twins
tokenized with the real GPT-J tokenizer, with the actual site machinery
(properties.find_opps / render, build_datasets.locate_common / evid) marking
evidence tokens, cue tokens, and background tokens. Purely conceptual —
no measured numbers (mechanism-diagram convention).

Output: results/style_properties/explainer_visuals/terminology_example.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import STYLE_PROPERTIES_DIR
from src.sandbox.ext_styleprops.properties import PROPS, render
from src.sandbox.ext_styleprops.build_datasets import locate_common, evid

from transformers import AutoTokenizer

OUT = STYLE_PROPERTIES_DIR / "explainer_visuals"

EXAMPLES = {
    "us_uk": "They painted the fence gray. The color matched the harbor wall.",
    "sentence_caps": "The market opens early. Vendors arrive first. Buyers follow soon after.",
}

C_EVID = "#f4a261"   # evidence
C_CUE = "#e63946"    # cue
C_BG = "#a8dadc"     # background
C_PLAIN = "#e9ecef"


def token_layout(tok, text):
    enc = tok(text, return_offsets_mapping=True)
    return enc.input_ids, enc.offset_mapping


def draw_panel(ax, tok, prop_name, base_text):
    prop = PROPS[prop_name]
    opps = prop.find_opps(base_text)
    nat_text, nat_spans = render(base_text, opps, "nat")
    alt_text, alt_spans = render(base_text, opps, "alt")
    ids_n, off_n = token_layout(tok, nat_text)
    ids_a, off_a = token_layout(tok, alt_text)

    # site machinery, exactly as the datasets are built
    sites = []
    for i, o in enumerate(opps):
        cue_n, cue_a = locate_common(off_n, nat_spans[i], off_a, alt_spans[i], o.div)
        sites.append(dict(k=i, cue=(cue_n, cue_a),
                          ev=(evid(off_n, nat_spans[i]), evid(off_a, alt_spans[i]))))

    char_w = 0.105
    rows = [("nat twin", nat_text, off_n, 1.0, 0), ("alt twin", alt_text, off_a, 0.0, 1)]
    cue_x = {}
    for label, text, offs, y, twin in rows:
        x = 0.0
        ax.text(-0.25, y + 0.13, label, ha="right", va="center", fontsize=10,
                style="italic")
        for j, (s, e) in enumerate(offs):
            tstr = text[s:e]
            w = max(len(tstr), 1) * char_w + 0.06
            color, lw, edge = C_PLAIN, 0.6, "#666666"
            is_cue = any(st["cue"][twin] == j for st in sites)
            is_ev = any(j in st["ev"][twin] for st in sites)
            if is_ev:
                color = C_EVID
            if is_cue:
                color, lw, edge = "#ffd6d9", 2.2, C_CUE
            ax.add_patch(FancyBboxPatch((x, y), w, 0.26, boxstyle="round,pad=0.015",
                                        fc=color, ec=edge, lw=lw))
            ax.text(x + w / 2, y + 0.13, tstr.replace(" ", "␣"), ha="center",
                    va="center", fontsize=8.5, family="monospace")
            if is_cue:
                k = next(st["k"] for st in sites if st["cue"][twin] == j)
                cue_x.setdefault(k, {})[twin] = x + w
                if twin == 0:
                    ax.text(x + w / 2, y + 0.40, f"k={k}", ha="center", fontsize=8,
                            color=C_CUE, fontweight="bold")
            x += w + 0.03
        if twin == 0:
            total_w = x
    # divergence markers + identical-token tie between twins
    for k, d in cue_x.items():
        if 0 in d and 1 in d:
            ax.plot([d[0] - 0.015, d[1] - 0.015], [1.0 - 0.035, 0.26 + 0.035], ls=":",
                    color=C_CUE, lw=1.2)
    ax.set_xlim(-1.6, max(total_w, 0) + 0.2)
    ax.set_ylim(-0.25, 1.62)
    ax.axis("off")
    ax.set_title(f"{prop_name}   (nat: {prop.nat_label}  /  alt: {prop.alt_label})",
                 fontsize=11, loc="left")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    fig, axes = plt.subplots(2, 1, figsize=(15, 7.8))
    for ax, name in zip(axes, EXAMPLES):
        draw_panel(ax, tok, name, EXAMPLES[name])

    legend = [
        (C_EVID, "evidence token — the property has just manifested here "
                 "(read site; ICL analog: demo label token)"),
        ("#ffd6d9", "cue token — point of no return: the last token that is IDENTICAL in "
                    "both twins before they diverge (write site — the same term as the ICL cue token). "
                    "Dotted line ties the same token across twins."),
        (C_PLAIN, "other tokens (identity-matched background tokens between manifestations "
                  "are sampled from these for state probes)"),
    ]
    y = 0.115
    for color, txt in legend:
        fig.patches.append(plt.Rectangle((0.045, y), 0.012, 0.020, fc=color,
                                         ec="#666666", transform=fig.transFigure))
        fig.text(0.062, y + 0.009, txt, fontsize=8.8, va="center")
        y -= 0.034
    fig.text(0.045, 0.175, "k = number of prior manifestations before this cue token "
                           "(the ICL shot-count analog). ␣ marks token-leading spaces.",
             fontsize=8.8)
    fig.suptitle("Style-property site terminology: minimal-pair twins of one base text",
                 fontsize=13, y=0.98)
    fig.subplots_adjust(top=0.90, bottom=0.22, left=0.03, right=0.99, hspace=0.35)
    fig.savefig(OUT / "terminology_example.png", dpi=170)
    print(OUT / "terminology_example.png")


if __name__ == "__main__":
    main()
