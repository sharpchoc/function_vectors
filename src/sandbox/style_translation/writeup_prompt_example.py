#!/usr/bin/env python
"""Write-up figure: one document of a family as two k = 4 prompts side by side (nat / alt), with the evidence tokens of the 4 in-context
opportunities highlighted, the cue token boxed, and the 5th opportunity (the decision) shown under each prompt. Rendered with PIL in a
monospace font so character widths are exact. Usage: --family py_indent [--doc <doc_id>] → <results>/writeup/prompt_example_<family>.png"""
import argparse, json, subprocess, sys, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.ml_families import ML_FAMILY

COL = {"nat": (31, 119, 180), "alt": (214, 39, 40)}
WRAP = 78


def font(size, mono=True):
    q = "DejaVuSansMono" if mono else "DejaVuSans"
    path = subprocess.run(["fc-list", f":family={q}", "file"], capture_output=True, text=True).stdout.split(":")[0].strip()
    return ImageFont.truetype(path, size)


def layout(tok, ids, ev_pos, cue_pos):
    """→ list of lines; each line = list of (char, kind) with kind in {None, 'ev', 'cue'}; newlines inside a token shown as ⏎ when it is evidence/cue."""
    ev = set(ev_pos); lines = [[]]
    for j, tid in enumerate(ids):
        kind = "cue" if j == cue_pos else ("ev" if j in ev else None)
        t = tok.decode([tid])
        for ch in t:
            if ch == "\n":
                if kind:
                    lines[-1].append(("⏎", kind))
                lines.append([])
            else:
                lines[-1].append(("␣" if (kind and ch == " ") else ch, kind))
    # soft-wrap long lines (prose in the task header)
    out = []
    for ln in lines:
        while len(ln) > WRAP:
            cut = max((i for i, (c, _) in enumerate(ln[:WRAP]) if c == " "), default=WRAP)
            out.append(ln[:cut]); ln = [(" ", None)] * 4 + ln[cut + 1:]
        out.append(ln)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--family", default="py_indent"); ap.add_argument("--doc", default=None); ap.add_argument("--model", default="qwen25_base")
    args = ap.parse_args()
    MP = model_paths(args.model); fam = args.family
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MP["tokenizer"])
    prompts = [p for p in json.load(open(MP["prompts"] / f"{fam}.json")) if p["k"] == 4]
    ev = {(r["doc_id"], r["pole"]): r for r in json.load(open(MP["evidence"] / f"{fam}.json"))}
    docs = sorted({p["doc_id"] for p in prompts})
    if args.doc is None:
        cand = [d for d in docs if all(all(e["idx"] for e in ev[(d, s)]["instances"]) for s in ("nat", "alt"))]
        args.doc = min(cand, key=lambda d: max(len(p["prompt_ids"]) for p in prompts if p["doc_id"] == d))
    fm = ML_FAMILY[fam]; labels = {"nat": fm.nat, "alt": fm.alt}
    F, FT, FH = font(17), font(19, mono=False), font(24, mono=False)
    cw, ch = F.getlength("M"), 24
    panels = []
    for s in ("nat", "alt"):
        p = next(p for p in prompts if p["doc_id"] == args.doc and p["style"] == s)
        pos = [j for e in ev[(args.doc, s)]["instances"][:4] for j in e["idx"]]
        panels.append((s, p, layout(tok, p["prompt_ids"], pos, len(p["prompt_ids"]) - 1)))
    nlines = max(len(l) for _, _, l in panels)
    PW = int(cw * (WRAP + 2)) + 40; PH = int(ch * (nlines + 3)) + 90
    img = Image.new("RGB", (2 * PW + 60, PH + 130), "white"); d = ImageDraw.Draw(img)
    d.text((20, 12), f"Family {fam}: the same document as a k = 4 prompt in each pole (Qwen2.5-7B base tokenisation)", font=FH, fill="black")
    d.text((20, 48), "shaded = evidence tokens of the 4 in-context opportunities (␣ = one space, ⏎ = newline inside the token)", font=FT, fill=(60, 60, 60))
    d.text((20, 72), "black box = cue token (the last prompt token, where the write vector is added); the model's next token is the 5th opportunity", font=FT, fill=(60, 60, 60))
    for pi, (s, p, lines) in enumerate(panels):
        x0 = 20 + pi * (PW + 30); y0 = 120; col = COL[s]
        d.text((x0, y0), f"{s} context: {labels[s]}   (k = 4 prompt, {len(p['prompt_ids'])} tokens)", font=FT, fill=col)
        y = y0 + 40
        for ln in lines:
            x = x0
            for c, kind in ln:
                if kind == "ev":
                    d.rectangle([x, y - 2, x + cw, y + ch - 4], fill=tuple(int(255 - (255 - v) * 0.35) for v in col))
                d.text((x, y), c, font=F, fill="black")
                if kind == "cue":
                    d.rectangle([x, y - 2, x + cw, y + ch - 4], outline="black", width=2)
                x += cw
            y += ch
        d.text((x0, y + 14), f"→ 5th opportunity: the {s} continuation would be {p[f'next_{s}']!r}", font=FT, fill=col)
    out = MP["results"] / "writeup" / f"prompt_example_{fam}.png"; img.save(out); print("->", out, args.doc)


if __name__ == "__main__":
    main()
