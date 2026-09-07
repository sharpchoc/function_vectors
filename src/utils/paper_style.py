"""House style for paper figures (2026-09-07).

Matches the vector schematics in write_up/graphics/ (icl_read_write_circuit_v2.tex): Inter text,
Computer-Modern math, thin grey axes, no top/right spines, muted palette keyed to the paper's
concepts (read = green, write = blue, linear map = teal, controls/baselines = grey).

Usage (at the top of any plotting script, before creating figures):
    from utils.paper_style import apply_paper_style, C
    apply_paper_style()
    ax.bar(..., color=C.write)

Only styling lives here — never data, paths, or figure logic.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.font_manager as fm
from cycler import cycler

_INTER_DIRS = (
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/inter",
    "/usr/share/fonts/truetype/inter",
    "/usr/share/fonts/opentype/inter",
    "/usr/local/share/fonts",
)


@dataclass(frozen=True)
class Palette:
    # concept colours (same RGB as the TikZ schematics)
    read: str = "#1B7F6B"      # read feature / targets
    write: str = "#2F55C9"     # write feature / cue / function vector
    map: str = "#1F6C80"       # read -> write linear map
    accent: str = "#B8860B"    # secondary categorical (gold, cf. reference schematic)
    purple: str = "#7B5EA7"    # tertiary categorical
    red: str = "#C9463D"       # sparingly: ablation / failure
    grey: str = "#8C8C8C"      # baselines, unsteered, controls
    lightgrey: str = "#D2D2D2"
    text: str = "#3C3C3C"
    note: str = "#6E6E6E"
    panel: str = "#F5F5F5"

    # light tints (for fills behind lines, hatched no-steering bars, etc.)
    read_light: str = "#BFE0D8"
    write_light: str = "#C5D0F2"
    map_light: str = "#C0DCE4"
    accent_light: str = "#EAD8A6"
    purple_light: str = "#D5CBE6"


C = Palette()

# ordered categorical cycle: write, read, accent, purple, map, grey
CYCLE = [C.write, C.read, C.accent, C.purple, C.map, C.grey]


def _register_inter() -> str | None:
    for d in _INTER_DIRS:
        files = glob.glob(f"{d}/Inter-*.otf") + glob.glob(f"{d}/Inter-*.ttf")
        if files:
            for f in files:
                try:
                    fm.fontManager.addfont(f)
                except Exception:  # pragma: no cover - font registration is best effort
                    pass
            return "Inter"
    return None


def apply_paper_style(base_size: float = 9.0) -> None:
    """Set global rcParams. Call once per script, before any figure is created."""
    family = _register_inter()
    sans = [family] if family else []
    sans += ["Helvetica", "Arial", "DejaVu Sans"]
    mpl.rcParams.update({
        # text
        "font.family": "sans-serif",
        "font.sans-serif": sans,
        "font.size": base_size,
        "axes.titlesize": base_size + 1.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.labelsize": base_size,
        "axes.labelcolor": C.text,
        "xtick.labelsize": base_size - 0.5,
        "ytick.labelsize": base_size - 0.5,
        "legend.fontsize": base_size - 0.5,
        "legend.title_fontsize": base_size - 0.5,
        "figure.titlesize": base_size + 2,
        "figure.titleweight": "semibold",
        "text.color": C.text,
        # math: Computer Modern, as in the TikZ figures
        "mathtext.fontset": "cm",
        "mathtext.default": "it",
        # axes
        "axes.edgecolor": "#9A9A9A",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#E4E4E4",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=CYCLE),
        # ticks
        "xtick.color": "#9A9A9A",
        "ytick.color": "#9A9A9A",
        "xtick.labelcolor": C.text,
        "ytick.labelcolor": C.text,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        # lines / markers
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "patch.linewidth": 0.6,
        "hatch.linewidth": 0.6,
        # legend
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.borderaxespad": 0.4,
        # figure / output
        "figure.facecolor": "white",
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def label_bars(ax, bars, fmt="{:.2f}", pad=0.012, fontsize=None, color=None, weight="semibold"):
    """Put the value above each bar (below for negatives), in the house typography."""
    ymin, ymax = ax.get_ylim()
    off = pad * (ymax - ymin)
    for b in bars:
        h = b.get_height()
        y = h + off if h >= 0 else h - off
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, y),
                    ha="center", va="bottom" if h >= 0 else "top",
                    fontsize=fontsize, color=color or C.text, fontweight=weight)


def despine_left(ax) -> None:
    """For plots where the y axis reads better as grid lines only."""
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
