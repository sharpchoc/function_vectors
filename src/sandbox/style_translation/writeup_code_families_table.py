#!/usr/bin/env python
"""Appendix table (PNG): every coding-convention family — language, category, natural convention, alternative convention, a real example
from the family's own twins (natural rendering → alternative rendering), and status (in pool / dropped with reason).
Output → <results>/writeup/code_families_table.png (+ .csv)"""
import json, re, sys, textwrap, csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import STYLE_TRANSLATION_DATA
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.code_families import CODE_SPECS

CAT = lambda i: "Naming" if i < 10 else "Literals" if i < 18 else "Syntax / dialect" if i < 40 else "Formatting" if i < 47 else "Comments / docs" if i < 52 else "Other languages"
DROP_REASON = {"py_bool_prefix": "dropped: alternative pole .12 at k = 4 after the rebuild", "py_indent": "dropped: alternative pole .24 at k = 4 after the rebuild",
               "js_func_pascal": "dropped: below .30 at the cheap check", "bash_subst": "dropped: below .30 at the cheap check", "css_shorthand": "dropped: below .30 at the cheap check",
               "py_literal_ctor": "dropped: below .30 at the cheap check", "c_braces": "dropped: below .30 at the cheap check"}


def vis(s):
    return s.replace("\n", "⏎").replace("\t", "⇥").replace("    ", "␣␣␣␣") if s.strip() == "" or "\n" in s or "\t" in s else s


def example(fam):
    """shortest clean opportunity: single line, ≤ 24 chars per side, non-whitespace content differs (whitespace families: whitespace differs),
    no generator artefacts."""
    p = STYLE_TRANSLATION_DATA / "pairs" / f"{fam}.json"
    if not p.exists():
        return ""
    ws_fam = fam in ("py_indent", "py_tabs", "blank_lines", "operator_spaces", "comma_space", "line_wrap", "c_braces")
    best = None
    for r in json.load(open(p))[:80]:
        for o in r.get("opps_all") or r["opps"]:
            a, b = o["nat"], o["alt"]
            if a == b or "<ctrl" in a + b or len(a) > 24 or len(b) > 24 or (("\n" in a or "\n" in b) and fam not in ("blank_lines", "py_indent", "py_tabs", "line_wrap")):
                continue
            same_content = re.sub(r"\s+", "", a) == re.sub(r"\s+", "", b)
            if ws_fam != same_content:
                continue
            score = abs(len(a) - 10) + abs(len(b) - 10)
            if best is None or score < best[0]:
                best = (score, a, b)
    return f"{vis(best[1])}  →  {vis(best[2])}" if best else ""


def main():
    R = model_paths("qwen25_base")["results"]; OUT = R / "writeup"; OUT.mkdir(exist_ok=True)
    pool = set(json.load(open(R / "code_pool_full.json"))["pool"])
    rows = []
    for i, s in enumerate(CODE_SPECS):
        name, lang, nat, alt = s[0], s[1], s[2], s[3]
        rows.append(dict(category=CAT(i), family=name, language=lang, natural=nat, alternative=alt, example=example(name),
                         status="in pool" if name in pool else DROP_REASON.get(name, "dropped")))
    with open(OUT / "code_families_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # ---- render
    cols = [("Family", 0.12), ("Language", 0.07), ("Natural convention", 0.17), ("Alternative convention", 0.17), ("Example (natural → alternative)", 0.30), ("Status", 0.17)]
    W = 22.0; wrap = {2: 30, 3: 30, 4: 52, 5: 30}
    cells = []
    for r in rows:
        c = [r["family"], r["language"], r["natural"], r["alternative"], r["example"], r["status"]]
        cells.append(["\n".join(textwrap.wrap(v, wrap[j])) if j in wrap else v for j, v in enumerate(c)])
    lh, pad, hdr_h, cat_h, title_h = 0.16, 0.07, 0.34, 0.30, 0.55
    heights = [max(x.count("\n") + 1 for x in c) * lh + 2 * pad for c in cells]
    cats = [r["category"] for r in rows]; n_cat = len(set(cats))
    H = title_h + hdr_h + sum(heights) + n_cat * cat_h + 0.3
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(W / 2, H - 0.15, "Coding-convention families: the natural and alternative convention of each, with an example opportunity from the twins", ha="center", va="top", fontsize=13, fontweight="bold")
    ax.text(W / 2, H - 0.40, f"{len(rows)} families defined; {len(pool)} in the final pool. In every twin pair the two renderings differ only at the opportunities; the model must continue the code at the next opportunity.", ha="center", va="top", fontsize=9.5, color="#333333")
    xs, x = [], 0.3
    for _, fr in cols:
        xs.append((x, fr * (W - 0.6))); x += fr * (W - 0.6)
    y = H - title_h
    for (xx, w), (name, _) in zip(xs, cols):
        ax.add_patch(Rectangle((xx, y - hdr_h), w, hdr_h, facecolor="#2f2f2f", edgecolor="#2f2f2f")); ax.text(xx + 0.08, y - hdr_h / 2, name, va="center", ha="left", fontsize=9.5, color="white", fontweight="bold")
    y -= hdr_h; prev = None
    for i, (r, c, h) in enumerate(zip(rows, cells, heights)):
        if r["category"] != prev:
            ax.add_patch(Rectangle((0.3, y - cat_h), W - 0.6, cat_h, facecolor="#e8eef5", edgecolor="#c8c8c8", lw=0.6)); ax.text(0.4, y - cat_h / 2, r["category"], va="center", ha="left", fontsize=10, fontweight="bold", color="#1f3b5c"); y -= cat_h; prev = r["category"]
        for j, ((xx, w), v) in enumerate(zip(xs, c)):
            face = "#fbe9e7" if (r["status"] != "in pool" and j == 5) else ("#f7f7f7" if i % 2 else "white")
            ax.add_patch(Rectangle((xx, y - h), w, h, facecolor=face, edgecolor="#d0d0d0", lw=0.5))
            ax.text(xx + 0.08, y - h / 2, v.replace("$", r"\$"), va="center", ha="left", fontsize=8.6, family="DejaVu Sans Mono" if j in (0, 4) else "DejaVu Sans", linespacing=1.15, color="#8a1c1c" if (j == 5 and r["status"] != "in pool") else "black")
        y -= h
    fig.savefig(OUT / "code_families_table.png", dpi=140); print("->", OUT / "code_families_table.png", len(rows))


if __name__ == "__main__":
    main()
