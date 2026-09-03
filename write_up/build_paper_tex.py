#!/usr/bin/env python
"""Build an Overleaf-ready zip of the paper draft from the Markdown source.

  python write_up/build_paper_tex.py            -> write_up/paper_draft_v1.zip  (main.tex + figures/)

Pipeline: (1) copy every referenced PNG into figures/ and rewrite the paths; (2) replace the
handful of non-ASCII symbols the draft uses with LaTeX equivalents (context-aware: inside math
vs. text) so pdflatex compiles; (3) pandoc Markdown -> LaTeX body (no implicit figure
environments, so images appear inline exactly where they are in the .md, with the author's
italic captions kept as text); (4) wrap in a fixed preamble, compile with latexmk to check,
zip. Content is not edited: only markup is translated.
"""
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pypandoc

HERE = Path(__file__).resolve().parent
MD = HERE / "read_and_write_features_for_in_context_learning_paper_draft.md"
BUILD = HERE / "_tex_build"
ZIP = HERE / "paper_draft_v1.zip"

TEXT_MAP = {"–": "--", "—": "---"}   # other symbols: \DeclareUnicodeCharacter fallbacks in the preamble
MATH_MAP = {"–": "-", "—": "-", "…": r"\ldots", "→": r"\rightarrow", "×": r"\times", "⋮": r"\vdots",
            "ρ": r"\rho", "²": "^{2}", "−": "-", "±": r"\pm", "≈": r"\approx", "∈": r"\in",
            "≥": r"\geq", "≤": r"\leq", "‖": r"\|", "σ": r"\sigma", "α": r"\alpha", "λ": r"\lambda",
            "Δ": r"\Delta", "ℓ": r"\ell", "·": r"\cdot", "✓": r"\checkmark"}
MATH_RE = re.compile(r"(```.*?```|`[^`\n]+`|\$\$.*?\$\$|(?<!\\)\$[^$\n]+?\$)", re.S)

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs,longtable,array,calc}
\usepackage{xcolor}
\usepackage{float}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}
\setcounter{secnumdepth}{0}
\DeclareUnicodeCharacter{22EE}{\ensuremath{\vdots}}
\DeclareUnicodeCharacter{2192}{\ensuremath{\rightarrow}}
\DeclareUnicodeCharacter{00D7}{\ensuremath{\times}}
\DeclareUnicodeCharacter{03C1}{\ensuremath{\rho}}
\DeclareUnicodeCharacter{2212}{\ensuremath{-}}
\DeclareUnicodeCharacter{00B1}{\ensuremath{\pm}}
\DeclareUnicodeCharacter{2248}{\ensuremath{\approx}}
\DeclareUnicodeCharacter{2208}{\ensuremath{\in}}
\DeclareUnicodeCharacter{2265}{\ensuremath{\geq}}
\DeclareUnicodeCharacter{2264}{\ensuremath{\leq}}
\DeclareUnicodeCharacter{2016}{\ensuremath{\|}}
\DeclareUnicodeCharacter{2113}{\ensuremath{\ell}}
\renewcommand{\_}{\textunderscore\allowbreak}
\setlength{\emergencystretch}{3em}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}
\makeatother
\setkeys{Gin}{width=\maxwidth,keepaspectratio}
\setlength{\parskip}{4pt}
\title{%(title)s}
\author{}
\date{}
\begin{document}
\maketitle
"""


def normalise_blocks(md):
    """Markdown viewers are lenient; pandoc is not: (a) consecutive image lines become one
    paragraph (side-by-side images), so separate them; (b) a list must be preceded by a blank
    line, else it is glued to the previous paragraph as running text."""
    lines = md.split("\n")
    out = []
    in_code = False
    for k, l in enumerate(lines):
        if l.startswith("```"):
            in_code = not in_code
        if not in_code and out:
            prev = out[-1]
            if l.startswith("![") and prev.startswith("!["):
                out.append("")
            elif re.match(r"^(- |\* |\d+\. )", l) and prev.strip() and not re.match(r"^(\s*- |\s*\* |\s*\d+\. |\s{2,})", prev):
                out.append("")
        out.append(l)
    return "\n".join(out)


def size_table_columns(md):
    """Pipe tables: make the separator-row dash counts proportional to each column's longest cell
    (capped), so pandoc emits relative column widths that fit the content instead of equal widths."""
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|\s*$", lines[i + 1]):
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            rows = [lines[i]] + lines[i + 2:j]
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            ncol = len(cells[0])
            longest = [max(len(r[c]) if c < len(r) else 0 for r in cells) for c in range(ncol)]
            longest = [min(max(l, 4), 60) for l in longest]           # cap so one huge cell cannot starve the rest
            if sum(longest) > 70:                                     # only tables that will need wrapping
                aligns = [c.strip() for c in lines[i + 1].strip().strip("|").split("|")]
                seps = []
                for c, l in enumerate(longest):
                    n = max(8, round(60 * l / sum(longest)))
                    a = aligns[c] if c < len(aligns) else "---"
                    seps.append((":" if a.startswith(":") else "") + "-" * n + (":" if a.endswith(":") else ""))
                lines[i + 1] = "|" + "|".join(seps) + "|"
            i = j
        else:
            i += 1
    return "\n".join(lines)


def replace_unicode(md):
    out, pos = [], 0
    for m in MATH_RE.finditer(md):
        seg = md[pos:m.start()]
        for k, v in TEXT_MAP.items():
            seg = seg.replace(k, v)
        out.append(seg)
        mseg = m.group(0)
        if not mseg.startswith("`"):                 # code spans/blocks: leave as-is (inputenc fallbacks below)
            for k, v in MATH_MAP.items():
                mseg = mseg.replace(k, v)
        out.append(mseg)
        pos = m.end()
    seg = md[pos:]
    for k, v in TEXT_MAP.items():
        seg = seg.replace(k, v)
    out.append(seg)
    return "".join(out)


def main():
    if BUILD.exists():
        shutil.rmtree(BUILD)
    (BUILD / "figures").mkdir(parents=True)
    md = MD.read_text()

    # title
    lines = md.split("\n")
    assert lines[0].startswith("# ")
    title = lines[0][2:].strip()
    md = "\n".join(lines[1:])

    # figures: copy + rewrite paths
    seen = {}
    def fig_sub(m):
        alt, rel = m.group(1), m.group(2)
        src = (HERE / rel).resolve()
        assert src.exists(), f"missing figure {src}"
        if rel not in seen:
            seen[rel] = f"fig{len(seen)+1:02d}_{src.name}"
            shutil.copy(src, BUILD / "figures" / seen[rel])
        return f"![{alt}](figures/{seen[rel]})"
    md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", fig_sub, md)

    # top-level headings other than the title -> raw LaTeX
    md = md.replace("\n# Appendix\n", "\n```{=latex}\n\\clearpage\n\\begin{center}{\\LARGE\\bfseries Appendix}\\end{center}\n```\n")
    md = md.replace("\n# References\n", "\n```{=latex}\n\\section*{References}\n```\n")
    assert "\n# " not in md, "unexpected level-1 heading left"

    md = normalise_blocks(md)
    md = size_table_columns(md)
    md = replace_unicode(md)
    (BUILD / "body.md").write_text(md)

    body = pypandoc.convert_text(
        md, "latex", format="markdown-implicit_figures+tex_math_dollars+pipe_tables+raw_attribute",
        extra_args=["--wrap=none"])
    # bare images on their own paragraph -> centred, full width
    body = re.sub(r"^\\pandocbounded\{\\includegraphics(\[[^\]]*\])?\{([^}]+)\}\}\s*$",
                  lambda m: "\\begin{center}\\includegraphics[width=\\linewidth,height=0.45\\textheight,keepaspectratio]{%s}\\end{center}" % m.group(2),
                  body, flags=re.M)
    body = re.sub(r"^\\includegraphics(\[[^\]]*\])?\{([^}]+)\}\s*$",
                  lambda m: "\\begin{center}\\includegraphics[width=\\linewidth,height=0.45\\textheight,keepaspectratio]{%s}\\end{center}" % m.group(2),
                  body, flags=re.M)
    tex = PREAMBLE % {"title": title} + body + "\n\\end{document}\n"
    (BUILD / "main.tex").write_text(tex)

    # compile check
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                       cwd=BUILD, capture_output=True, text=True)
    log = (BUILD / "main.log").read_text(errors="ignore") if (BUILD / "main.log").exists() else r.stdout
    errs = [l for l in log.split("\n") if l.startswith("!")]
    print("latexmk rc", r.returncode, "| errors:", len(errs), "| undefined refs:", log.count("undefined"),
          "| overfull boxes:", log.count("Overfull"), "| figures:", len(seen))
    for e in errs[:8]:
        print("  ", e)
    if r.returncode != 0:
        sys.exit(1)

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(BUILD / "main.tex", "main.tex")
        for f in sorted((BUILD / "figures").iterdir()):
            z.write(f, f"figures/{f.name}")
    print(f"wrote {ZIP} ({ZIP.stat().st_size/1e6:.1f} MB); pdf at {BUILD/'main.pdf'}")


if __name__ == "__main__":
    main()
