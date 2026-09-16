#!/usr/bin/env python
"""Render results/style_translation/task_inventory.csv (every convention family considered, decision, k = 4 accuracy, reason) as a
collaborator-facing PNG table. Registers the system CJK fonts (fc-list) so Chinese / Japanese / Korean examples render.
Edit the CSV to change content; rerun this script to regenerate the PNG."""
import csv, subprocess, sys, textwrap, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import STYLE_TRANSLATION_RESULTS
CSV = STYLE_TRANSLATION_RESULTS / "task_inventory.csv"; PNG = STYLE_TRANSLATION_RESULTS / "task_inventory.png"
fams = ["DejaVu Sans"]
for lang in ("ja", "zh", "ko"):
    out = subprocess.run(["fc-list", f":lang={lang}", "file"], capture_output=True, text=True).stdout.split("\n")
    for p in [l.strip().rstrip(":") for l in out if l.strip()][:2]:
        try:
            font_manager.fontManager.addfont(p); fams.append(font_manager.FontProperties(fname=p).get_name())
        except Exception:
            pass
plt.rcParams["font.family"] = list(dict.fromkeys(fams))
rows = list(csv.DictReader(open(CSV)))
KEPT, WEAK, DROP, FIX, PRUNE, UNT = "kept (in pool)", "tested: too weak", "tested, then dropped", "lexically identical", "pruned before inference", "not testable on this model"
LATE = "passed the late re-test; not in pool"
order = [KEPT, LATE, WEAK, DROP, UNT, FIX, PRUNE]; rows.sort(key=lambda r: (order.index(r["status"]), r["lang"] != "English", r["name"].lower()))
col = {KEPT: "#c7e9c0", LATE: "#e2f0d9", WEAK: "#f6c6c9", DROP: "#fbd9b0", FIX: "#dcdcdc", PRUNE: "#fff0b3", UNT: "#cfe2f3"}
cols = [("convention", 0.19), ("target", 0.075), ("natural → alternative", 0.17), ("decision", 0.115), ("k=4 nat", 0.05), ("k=4 alt", 0.05), ("texts", 0.045), ("reason / evidence (Qwen2.5-7B base unless stated)", 0.34)]
W = 24.0; wrap_chars = {0: 34, 2: 32, 7: 86}
cells = []
for r in rows:
    c = [r["name"], r["lang"], r["kind"], r["status"], r["nat"], r["alt"], r["n"], r["reason"]]
    cells.append(["\n".join(textwrap.wrap(v, wrap_chars[i])) if i in wrap_chars else v for i, v in enumerate(c)])
line_h, pad = 0.145, 0.09
heights = [max(x.count("\n") + 1 for x in c) * line_h + 2 * pad for c in cells]
hdr_h, title_h = 0.32, 0.75; H = title_h + hdr_h + sum(heights) + 0.2
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
xs, x = [], 0.3
for name, fr in cols:
    xs.append((x, fr * (W - 0.6))); x += fr * (W - 0.6)
counts = {s: sum(1 for r in rows if r["status"] == s) for s in order}
ax.text(W / 2, H - 0.12, "Writing-convention families considered for the read/write-feature study — every candidate and why it was kept or dropped", ha="center", va="top", fontsize=14, fontweight="bold")
ax.text(W / 2, H - 0.42, f"{counts[KEPT]} kept in the pool · {counts[LATE]} passed the late re-test but not in the pool · {counts[WEAK]} tested and too weak · {counts[UNT]} not testable on this model · {counts[FIX]} lexically identical (ignored for this question) · {counts[PRUNE]} pruned before inference.   "
        "Cutoff: k = 4 accuracy ≥ .30 for BOTH conventions (accuracy = uses the context's convention ∧ faithful translation, 200 texts unless stated).", ha="center", va="top", fontsize=9.5)
y = H - title_h
for (xx, w), (name, _) in zip(xs, cols):
    ax.add_patch(Rectangle((xx, y - hdr_h), w, hdr_h, facecolor="#333333", edgecolor="#333333"))
    ax.text(xx + 0.08, y - hdr_h / 2, name, va="center", ha="left", fontsize=9, color="white", fontweight="bold")
y -= hdr_h
for i, (r, c, h) in enumerate(zip(rows, cells, heights)):
    for j, ((xx, w), v) in enumerate(zip(xs, c)):
        ax.add_patch(Rectangle((xx, y - h), w, h, facecolor=col[r["status"]] if j == 3 else ("#f7f7f7" if i % 2 else "white"), edgecolor="#c8c8c8", lw=0.6))
        ax.text(xx + 0.08, y - h / 2, v, va="center", ha="left", fontsize=8.3, fontweight="bold" if j == 3 else "normal", linespacing=1.15)
    y -= h
fig.savefig(PNG, dpi=130); print("->", PNG)
