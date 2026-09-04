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
from PIL import Image

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

TEMPLATE = HERE / "iclr2026_template"      # ICLR 2026 Master-Template files (sty/bst), shipped in the zip
BIB = HERE / "references.bib"
TEMPLATE_FILES = ("iclr2026_conference.sty", "fancyhdr.sty", "natbib.sty", "math_commands.tex", "iclr2026_conference.bst")

PREAMBLE = r"""\documentclass{article}
\usepackage{iclr2026_conference,times}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs,longtable,array,calc}
\usepackage{xcolor}
\usepackage{etoolbox}
\usepackage{enumitem}
\usepackage{microtype}
\usepackage{placeins}
\usepackage{flafter}
\usepackage{hyperref}
\usepackage{url}
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
\preto\section{\FloatBarrier}
\setlength{\emergencystretch}{3em}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\setlist{nosep,leftmargin=1.4em}
\AtBeginEnvironment{longtable}{\small}
\setlength{\LTpre}{6pt}\setlength{\LTpost}{6pt}
\AtBeginDocument{\setlength{\abovedisplayskip}{5pt}\setlength{\belowdisplayskip}{5pt}\setlength{\abovedisplayshortskip}{2pt}\setlength{\belowdisplayshortskip}{2pt}}
\title{%(title)s}
\author{Anonymous authors}
%% \iclrfinalcopy   %% uncomment for the camera-ready (de-anonymised) version
\begin{document}
\maketitle
\begin{abstract}
%(abstract)s
\end{abstract}
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

    # citations: the draft cites two works in prose; map every surface form to natbib
    CITES = {"Todd et al.": "todd2024function", "Hu et al.": "hu2025addition"}
    for name, key in CITES.items():
        n = re.escape(name)
        md = re.sub(r"\(" + n + r", (20\d\d)\)", lambda m: "\\citep{%s}" % key, md)          # (Todd et al., 2024)
        md = re.sub(n + r" \((20\d\d)\)", lambda m: "\\citet{%s}" % key, md)                # Todd et al. (2024)
        md = re.sub(n + r" (20\d\d)\b", lambda m: "\\citealt{%s}" % key, md)                 # Todd et al. 2024
        md = re.sub(n + r"(?! ?\()", lambda m: "\\citeauthor{%s}" % key, md)                  # bare "Todd et al."
    # references section -> BibTeX bibliography (same two entries, in references.bib)
    m = re.search(r"\n# References\n.*", md, re.S)
    assert m, "no References section"
    md = md[:m.start()] + "\n```{=latex}\n\\bibliographystyle{iclr2026_conference}\n\\bibliography{references}\n```\n"

    # abstract -> ICLR abstract environment
    m = re.search(r"\n## Abstract\n(.*?)(?=\n## )", md, re.S)
    abstract = m.group(1).strip() if m else ""
    if m:
        md = md[:m.start()] + md[m.end():]

    # figures: copy + rewrite paths
    seen, alt_by_fig = {}, {}
    def fig_sub(m):
        alt, rel = m.group(1), m.group(2)
        src = (HERE / rel).resolve()
        assert src.exists(), f"missing figure {src}"
        if rel not in seen:
            seen[rel] = f"fig{len(seen)+1:02d}_{src.name}"
            shutil.copy(src, BUILD / "figures" / seen[rel])
        alt_by_fig[alt] = f"figures/{seen[rel]}"
        return f"![{alt}](figures/{seen[rel]})"
    md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", fig_sub, md)

    # top-level headings other than the title -> raw LaTeX
    md = md.replace("\n# Appendix\n", "\n```{=latex}\n\\clearpage\n\\appendix\n\\begin{center}{\\LARGE\\bfseries Appendix}\\end{center}\n```\n")
    assert "\n# " not in md, "unexpected level-1 heading left"

    md = normalise_blocks(md)
    md = size_table_columns(md)
    md = replace_unicode(md)
    (BUILD / "body.md").write_text(md)

    body = pypandoc.convert_text(
        md, "latex", format="markdown-implicit_figures+tex_math_dollars+pipe_tables+raw_attribute",
        extra_args=["--wrap=none"])
    # bare images on their own paragraph -> centred; wide multi-panel figures get the full text
    # width, everything else 72% (paper look), all capped in height
    def fig_tex(path, width=None):
        w, h = Image.open(BUILD / path).size
        if width is None:
            width = "\\linewidth" if w / h > 2.2 else "0.6\\linewidth"
        cap = "0.27\\textheight" if w / h > 2.2 else "0.23\\textheight"
        return "\\includegraphics[width=%s,height=%s,keepaspectratio]{%s}" % (width, cap, path)
    IMG = r"^(?:\\pandocbounded\{)?\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}\}?\s*$"
    alt_of = {v: k for k, v in alt_by_fig.items()}
    def label_of(path):
        num, stem = Path(path).stem.split("_", 1)          # figNN_<name>: keep NN, two figures share a name
        return "fig:" + num[3:] + "-" + re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    def caption_after(lines, k):
        """If the next non-empty line is an italic paragraph, it is the author's caption."""
        j = k
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        if j < len(lines) and lines[j].startswith("\\emph{") and lines[j].rstrip().endswith("}"):
            return lines[j].strip()[len("\\emph{"):-1], j
        return None, None
    def add_reference(out, label):
        """Append ' (Figure~\\ref{label})' to the nearest preceding text paragraph or list item."""
        k = len(out) - 1
        while k >= 0 and (out[k].strip() == "" or out[k].startswith(("\\end{", "\\begin{center}", "\\begin{figure"))):
            k -= 1
        if k < 0 or out[k].startswith(("\\section", "\\subsection", "\\begin{", "\\[", "\\end", "%")) or out[k].strip() in ("\\]",):
            unreferenced.append(label); return
        line = out[k].rstrip()
        ref = " (Figure~\\ref{%s})" % label
        if line.endswith(")."):
            line = line[:-1] + ref + "."                      # "...(text)." -> "...(text) (Figure N)."
        elif line.endswith((".", ":", "!")):
            line = line[:-1] + ref + line[-1]                 # "...text." -> "...text (Figure N)."
        else:
            line = line + ref                                 # no terminal punctuation (incl. a closing paren)
        out[k] = line
    def figure_env(paths, widths, caption, label):
        imgs = "\\hfill ".join(fig_tex(p, w) for p, w in zip(paths, widths))
        return ("\\begin{figure}[tbp]\\centering\n%s\n\\caption{%s}\n\\label{%s}\n\\end{figure}" % (imgs, caption, label))
    lines = body.split("\n")
    out, k, unreferenced, figlabels = [], 0, [], []
    while k < len(lines):
        m = re.match(IMG, lines[k])
        if not m:
            out.append(lines[k]); k += 1; continue
        m2 = re.match(IMG, lines[k + 2]) if k + 2 < len(lines) and lines[k + 1].strip() == "" else None
        paths = [m.group(1)] + ([m2.group(1)] if m2 else [])
        k_after = k + (3 if m2 else 1)
        cap, jcap = caption_after(lines, k_after)
        if cap is None:
            cap = " ".join(alt_of.get(p, "") for p in paths).strip() or Path(paths[0]).stem
        else:
            k_after = jcap + 1
        label = label_of(paths[0])
        figlabels.append(label)
        add_reference(out, label)
        widths = ["0.49\\linewidth"] * 2 if m2 else [None]
        out.append(figure_env(paths, widths, cap, label))
        k = k_after
    body = "\n".join(out)
    print(f"figures: {len(figlabels)} environments; unreferenced: {unreferenced or 'none'}")
    abs_tex = pypandoc.convert_text(abstract, "latex", format="markdown", extra_args=["--wrap=none"]).strip() if abstract else ""
    tex = PREAMBLE % {"title": title, "abstract": abs_tex} + body + "\n\\end{document}\n"
    for f in TEMPLATE_FILES:
        shutil.copy(TEMPLATE / f, BUILD / f)
    shutil.copy(BIB, BUILD / "references.bib")
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
        z.write(BUILD / "references.bib", "references.bib")
        for f in TEMPLATE_FILES:
            z.write(BUILD / f, f)
        for f in sorted((BUILD / "figures").iterdir()):
            z.write(f, f"figures/{f.name}")
    shutil.copy(BUILD / "main.tex", HERE / "paper_draft_v1.tex")
    with zipfile.ZipFile(HERE / "paper_draft_v1_update.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(BUILD / "main.tex", "main.tex")
        z.write(BUILD / "references.bib", "references.bib")
        for f in TEMPLATE_FILES:
            z.write(BUILD / f, f)
    print(f"wrote {ZIP} ({ZIP.stat().st_size/1e6:.1f} MB), paper_draft_v1.tex, paper_draft_v1_update.zip; pdf at {BUILD/'main.pdf'}")


if __name__ == "__main__":
    main()
