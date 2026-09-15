#!/usr/bin/env python
"""Explainer: what the model saw / wrote / how it was scored, for cue-token steering on k = 0 code prompts (Qwen2.5-7B base).
Rows from the steering confirm records: one success, one success with a judge-rejected arm, one steering failure, one more success.
Output: results/style_translation/explainer/code_steering_examples.png (PNG only)."""
import json, sys, textwrap
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, STYLE_TRANSLATION_RESULTS
CONF = ARTIFACTS_ROOT / "style_translation" / "qwen25_base" / "steering" / "confirm"
OUT = STYLE_TRANSLATION_RESULTS / "explainer" / "code_steering_examples.png"
LANG = {"py_snake_camel": "Python", "js_arrow": "JavaScript", "py_quotes": "Python", "docstring_style": "Python"}
TITLE = {"py_snake_camel": "py_snake_camel — natural: snake_case identifiers · alternative: camelCase",
         "js_arrow": "js_arrow — natural: arrow functions · alternative: function expressions",
         "py_quotes": "py_quotes — natural: single quotes · alternative: double quotes",
         "docstring_style": "docstring_style — natural: Google docstrings (Args:) · alternative: NumPy docstrings (Parameters)"}


def by_doc(fam):
    d = {}
    for r in json.load(open(CONF / f"{fam}.json")):
        d.setdefault(r["doc_id"], {})[r["arm"]] = r
    return d


def wrap(t, w):
    out = []
    for ln in t.split("\n"):
        out += textwrap.wrap(ln, w, replace_whitespace=False, drop_whitespace=False) or [""]
    return "\n".join(out)


rows = []
b = by_doc("py_snake_camel"); rows.append(("py_snake_camel", "py_snake_camel__t001", [("base", "unsteered"), ("alt_top1", "steered → alt"), ("alt_cf", "control: another family's vector")], b["py_snake_camel__t001"]))
b = by_doc("js_arrow"); rows.append(("js_arrow", "js_arrow__t001", [("base", "unsteered"), ("alt_top1", "steered → alt"), ("nat_top2", "steered → nat (2nd setting)")], b["js_arrow__t001"]))
b = by_doc("py_quotes"); d = [d for d, a in b.items() if a["base"]["decision"] == "nat" and a["alt_top1"]["decision"] != "alt"][0]
rows.append(("py_quotes", d, [("base", "unsteered"), ("alt_top1", "steered → alt"), ("alt_top2", "steered → alt (2nd setting)")], b[d]))
b = by_doc("docstring_style"); rows.append(("docstring_style", "docstring_style__t001", [("base", "unsteered"), ("alt_top1", "steered → alt"), ("nat_top1", "steered → nat")], b["docstring_style__t001"]))

LH = 0.135; W = 26.0
prepared = []
for fam, doc, arms, a in rows:
    base = a["base"]; cue = base["cue_tok"]; ctx = base["context_tail"]
    pre = ctx[:-len(cue)] if ctx.endswith(cue) else ctx
    spec = wrap("Task:\n" + base["es_text"][:420] + ("…" if len(base["es_text"]) > 420 else ""), 86)
    code = wrap(pre[-600:], 86)
    left_lines = spec.count("\n") + 1 + 2 + code.count("\n") + 1
    tails = [wrap(a[arm]["tail"][:330], 50) for arm, _ in arms]
    right_lines = max(t.count("\n") + 1 for t in tails)
    box_h = LH * max(left_lines, right_lines) + 1.75
    prepared.append((fam, doc, arms, a, spec, code, tails, box_h))
H = 1.15 + sum(p[-1] + 0.8 for p in prepared)
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
ax.text(W / 2, H - 0.25, "What the model saw, what it wrote, and how we scored it — cue-token steering on k = 0 code prompts (Qwen2.5-7B base)", ha="center", va="top", fontsize=15, fontweight="bold")
ax.text(W / 2, H - 0.68, "Prompt = task spec + the start of the solution, cut right after the CUE token (red); the model's next tokens make the style decision.   "
        "accurate = the continuation uses the TARGET convention (regex classifier at the cue) AND the judge accepts it as a plausible on-task continuation.", ha="center", va="top", fontsize=10)
y = H - 1.15
for fam, doc, arms, a, spec, code, tails, box_h in prepared:
    base = a["base"]; cue = base["cue_tok"]
    ax.text(0.3, y, TITLE[fam] + f"    (task {doc.split('__')[1]})", fontsize=11.5, fontweight="bold", va="top")
    top = y - 0.35
    ax.add_patch(FancyBboxPatch((0.3, top - box_h), 9.0, box_h, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor="#f6f6f6", edgecolor="#999"))
    ax.text(0.45, top - 0.1, "MODEL INPUT", fontsize=9, fontweight="bold", va="top", color="#333")
    ax.text(0.45, top - 0.37, spec, fontsize=7.6, va="top", family="monospace")
    ycode = top - 0.37 - LH * (spec.count("\n") + 1) - 0.12
    ax.text(0.45, ycode, f"{LANG[fam]}:\n" + code, fontsize=7.6, va="top", family="monospace")
    last = code.split("\n")[-1]; ycue = ycode - LH * (code.count("\n") + 1)
    cue_show = cue if cue.strip() else ("␣" * len(cue))
    ax.text(0.45 + 0.0645 * len(last), ycue, cue_show, fontsize=7.6, va="top", family="monospace", color="white", fontweight="bold", bbox=dict(boxstyle="square,pad=0.1", facecolor="#d62728", edgecolor="none"))
    ax.text(0.45, top - box_h + 0.1, f"cue token = {cue!r}   (the last token shared by both conventions)", fontsize=7.5, va="bottom", color="#d62728")
    x = 9.7; w = 5.2
    for (arm, label), tail in zip(arms, tails):
        r = a[arm]; dec = r["decision"]; ok = bool(r["judge"]["ok"]); target_pole = r["style"] if r["style"] != "none" else None
        col = {"nat": "#1f77b4", "alt": "#d62728"}.get(dec, "#888")
        ax.add_patch(FancyBboxPatch((x, top - box_h), w, box_h, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor="white", edgecolor="#999"))
        ax.text(x + 0.15, top - 0.1, label + ("" if r["layer"] == 0 else f"  (L{r['layer']}, α={r['alpha']:g})"), fontsize=9, fontweight="bold", va="top")
        ax.text(x + 0.15, top - 0.37, "MODEL OUTPUT", fontsize=8, va="top", color="#333")
        ax.text(x + 0.15, top - 0.6, tail, fontsize=7.4, va="top", family="monospace")
        yb = top - box_h + 0.95
        ax.text(x + 0.15, yb, f"classifier: {dec if dec else 'no decision (unscorable)'}", fontsize=8.2, va="top", color=col, fontweight="bold")
        ax.text(x + 0.15, yb - 0.27, f"judge: {'OK' if ok else 'NOT OK'} — {r['judge']['notes'][:62]}", fontsize=7.4, va="top", color="#2a7d2a" if ok else "#b22222")
        if target_pole:
            accurate = (dec == target_pole) and ok
            why = "" if accurate else (" (wrong convention)" if dec != target_pole else " (judge rejected)")
            ax.text(x + 0.15, yb - 0.55, f"target = {target_pole}  →  {'ACCURATE' if accurate else 'INACCURATE'}{why}", fontsize=8.6, va="top", fontweight="bold", color="#2a7d2a" if accurate else "#b22222")
        else:
            ax.text(x + 0.15, yb - 0.55, "no target (baseline): counts toward the unsteered rate", fontsize=7.6, va="top", color="#555")
        x += w + 0.25
    y -= box_h + 0.8
OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT, dpi=130); print("->", OUT)
