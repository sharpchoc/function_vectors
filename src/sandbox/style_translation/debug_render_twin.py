#!/usr/bin/env python
"""Render one twin pair side by side (natural / alternative) with the opportunity spans shaded and optional extra highlights.
Usage: debug_render_twin.py --family F --doc D --out PNG [--title T] [--note N] [--mark REGEX] [--lines A-B]"""
import argparse, json, re, subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import STYLE_TRANSLATION_DATA
from src.sandbox.style_translation.code_families import CODE_SPECS


def font(size, mono=True):
    q = "DejaVuSansMono" if mono else "DejaVuSans"
    path = subprocess.run(["fc-list", f":family={q}", "file"], capture_output=True, text=True).stdout.split(":")[0].strip()
    return ImageFont.truetype(path, size)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--family", required=True); ap.add_argument("--doc", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None); ap.add_argument("--note", default=""); ap.add_argument("--mark", default=None, help="regex to box in red in both twins")
    ap.add_argument("--lines", default=None, help="line range A-B to show (1-based, inclusive)"); args = ap.parse_args()
    r = next(x for x in json.load(open(STYLE_TRANSLATION_DATA / "pairs" / f"{args.family}.json")) if x["doc_id"] == args.doc)
    spec = next(s for s in CODE_SPECS if s[0] == args.family); labels = {"nat": spec[2], "alt": spec[3]}
    F, FT, FH = font(15), font(15, mono=False), font(19, mono=False); cw, ch = F.getlength("M"), 21
    lo, hi = (int(x) for x in args.lines.split("-")) if args.lines else (1, 10 ** 6)
    panels = []
    for pole in ("nat", "alt"):
        text = r[f"text_{pole}"]; marks = [(o[f"{pole}_span"][0], o[f"{pole}_span"][1], "opp") for o in (r.get("opps_all") or r["opps"])]
        if args.mark:
            marks += [(m.start(), m.end(), "mark") for m in re.finditer(args.mark, text)]
        lines, pos = [], 0
        for ln in text.split("\n"):
            segs = []
            for j, chr_ in enumerate(ln):
                kind = None
                for a, b, k in marks:
                    if a <= pos + j < b:
                        kind = k if kind != "mark" else kind
                segs.append((chr_, kind))
            lines.append(segs); pos += len(ln) + 1
        panels.append((pole, lines[lo - 1:hi]))
    maxcols = max((len(l) for _, ls in panels for l in ls), default=40) + 2; nl = max(len(ls) for _, ls in panels)
    PW = int(cw * maxcols) + 30; note_lines = [l for l in args.note.split("\n") if l]; y0 = 70 + 22 * len(note_lines) + 40; H = y0 + ch * nl + 40
    img = Image.new("RGB", (2 * PW + 60, H), "white"); d = ImageDraw.Draw(img)
    d.text((20, 12), args.title or f"{args.family}, document {args.doc}", font=FH, fill="black")
    d.text((20, 40), f"natural = {labels['nat']}   |   alternative = {labels['alt']}   |   shaded = opportunity spans (where the twins differ)" + ("   |   red box = highlighted issue" if args.mark else ""), font=FT, fill="#333333")
    for i, l in enumerate(note_lines):
        d.text((20, 64 + 22 * i), l, font=FT, fill="#333333")
    col = {"nat": (31, 119, 180), "alt": (214, 39, 40)}; hl = {"nat": (207, 226, 243), "alt": (248, 208, 208)}
    for pi, (pole, lines) in enumerate(panels):
        x0 = 20 + pi * (PW + 30); d.text((x0, y0 - 28), f"{pole} rendering" + (f"  (lines {lo}–{min(hi, lo + len(lines) - 1)})" if args.lines else ""), font=FT, fill=col[pole])
        for li, segs in enumerate(lines):
            y = y0 + li * ch
            for j, (chr_, kind) in enumerate(segs):
                x = x0 + j * cw
                if kind == "opp": d.rectangle([x, y - 2, x + cw, y + ch - 4], fill=hl[pole])
                if kind == "mark": d.rectangle([x, y - 2, x + cw, y + ch - 4], outline=(200, 0, 0), width=2)
                d.text((x, y), chr_, font=F, fill="black")
    img.save(args.out); print("->", args.out)


if __name__ == "__main__":
    main()
