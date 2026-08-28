"""
Plot the 2-shot FV-projection-ablation task-imitation study (per-token layer sweeps; CPU-only).

Compares two ablation directions at qfinal (both removed at all 29 layers):
  fperp = F' with F projected out (F' − proj_F F')   ·   fdiff = raw FV difference F' − F.

Reads results/exploratory/direction2_label_geometry/twoshot_fv_ablation_imitation/<task_pair>/
<direction>_<token>_alpha{a}_<variant>_layersweep.csv (columns: layer, clean_mean, ablate_mean,
steer_mean, steer_sem, steer_gain, steer_ablate_mean, steer_ablate_sem, steer_ablate_gain, retention).

Outputs (figures/):
  layersweep_compare_alpha{a}.png : grid rows=direction, cols=steer token; each panel overlays
    steer(ℓ) [shared], steer+ablate_fperp(ℓ), steer+ablate_fdiff(ℓ), plus clean + both ablate baselines.
  retention_compare.png : peak-layer retention, rows=(direction,token,α), cols=[fperp,fdiff].
See steer_twoshot_fv_ablation_logitgap.py for how the numbers are produced.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.paths import LABEL_GEOMETRY_DIR

TASK_PAIRS = ["antonym_synonym", "next_number_digits_prev_number_digits"]
TOKENS = ["label1", "label2", "qfinal"]
VARIANTS = ["fperp", "fdiff"]
VAR_COLOR = {"fperp": "#8452a8", "fdiff": "#d1892b"}   # purple / orange
VAR_LABEL = {"fperp": "F'⊥F", "fdiff": "F'−F"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=str(LABEL_GEOMETRY_DIR / "twoshot_fv_ablation_imitation"))
    p.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    return p.parse_args()


def load_csv(path):
    cols = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return {k: np.array(v) for k, v in cols.items()}


def cpath(root, tp, direction, tok, ak, variant):
    return Path(root) / tp / f"{direction}_{tok}_alpha{ak}_{variant}_layersweep.csv"


def discover_directions(root):
    """Ordered [(task_pair, direction)] present, detected from fperp csvs."""
    out = []
    for tp in TASK_PAIRS:
        d = Path(root) / tp
        if not d.is_dir():
            continue
        seen = set()
        for p in sorted(d.glob("*_fperp_layersweep.csv")):
            stem = p.name[:-len("_layersweep.csv")]           # <dir>_<tok>_alpha{a}_fperp
            left = stem.split("_alpha")[0]                    # <dir>_<tok>
            for tok in TOKENS:
                if left.endswith("_" + tok):
                    name = left[:-(len(tok) + 1)]
                    if name not in seen:
                        seen.add(name); out.append((tp, name))
                    break
    return out


def main():
    args = parse_args()
    root = Path(args.root)
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dirs = discover_directions(root)
    if not dirs:
        print(f"no per-token layersweep csvs under {root}")
        return
    print(f"{len(dirs)} directions found")

    # ---- overlay grid per alpha ----
    for alpha in args.alphas:
        ak = f"{alpha:g}"
        nr, nc = len(dirs), len(TOKENS)
        fig, axes = plt.subplots(nr, nc, figsize=(4.5 * nc, 3.0 * nr), squeeze=False)
        any_data = False
        for ri, (tp, name) in enumerate(dirs):
            src, tgt = name.split("_to_")
            for ci, tok in enumerate(TOKENS):
                ax = axes[ri][ci]
                cf = cpath(root, tp, name, tok, ak, "fperp")
                if not cf.exists():
                    ax.set_axis_off(); continue
                any_data = True
                c = {v: load_csv(cpath(root, tp, name, tok, ak, v))
                     for v in VARIANTS if cpath(root, tp, name, tok, ak, v).exists()}
                base = c["fperp"]
                x = base["layer"]
                ax.axhline(base["clean_mean"][0], color="#616161", lw=1.0, ls="--",
                           label=f"clean ({base['clean_mean'][0]:+.2f})")
                ax.plot(x, base["steer_mean"], color="#4c72b0", lw=1.6, label="steer")
                ax.fill_between(x, base["steer_mean"] - base["steer_sem"],
                                base["steer_mean"] + base["steer_sem"], color="#4c72b0", alpha=0.15)
                titbits = []
                for v in VARIANTS:
                    if v not in c:
                        continue
                    cc = c[v]
                    col = VAR_COLOR[v]
                    ax.axhline(cc["ablate_mean"][0], color=col, lw=0.9, ls=":",
                               label=f"ablate {VAR_LABEL[v]} ({cc['ablate_mean'][0]:+.2f})")
                    ax.plot(x, cc["steer_ablate_mean"], color=col, lw=1.5,
                            label=f"steer+abl {VAR_LABEL[v]}")
                    ax.fill_between(x, cc["steer_ablate_mean"] - cc["steer_ablate_sem"],
                                    cc["steer_ablate_mean"] + cc["steer_ablate_sem"], color=col, alpha=0.15)
                    pk = int(np.nanargmax(cc["steer_gain"]))
                    titbits.append(f"{VAR_LABEL[v]} ret {cc['retention'][pk]:.2f}")
                ax.axhline(0, color="k", lw=0.4)
                pk = int(np.nanargmax(base["steer_gain"]))
                ax.set_title(f"{src}→{tgt}  steer@{tok}\npeak L{pk} gain {base['steer_gain'][pk]:+.2f} · "
                             + " · ".join(titbits), fontsize=7.5)
                if ci == 0:
                    ax.set_ylabel("Δlogit = logit(a_tgt)−logit(a_src)", fontsize=7)
                if ri == nr - 1:
                    ax.set_xlabel("injection layer ℓ", fontsize=7)
                ax.tick_params(labelsize=6)
                if ri == 0 and ci == nc - 1:
                    ax.legend(fontsize=5.5, loc="best")
        if not any_data:
            plt.close(fig); continue
        fig.suptitle(f"2-shot FV-projection-ablation — α={ak}  (per steer token; ablation dir comparison)\n"
                     f"steer ONE token at ONE layer ℓ; ablate at qfinal (all layers): "
                     f"{VAR_LABEL['fperp']} (purple) vs {VAR_LABEL['fdiff']} (orange)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(fig_dir / f"layersweep_compare_alpha{ak}.png", dpi=150)
        plt.close(fig)
        print(f"saved layersweep_compare_alpha{ak}.png")

    # ---- peak-retention comparison heatmap: rows=(dir,token,α), cols=[fperp,fdiff] ----
    labels, M = [], []
    for tp, name in dirs:
        src, tgt = name.split("_to_")
        for tok in TOKENS:
            for alpha in args.alphas:
                ak = f"{alpha:g}"
                rowvals = []
                for v in VARIANTS:
                    p = cpath(root, tp, name, tok, ak, v)
                    if p.exists():
                        c = load_csv(p)
                        pk = int(np.nanargmax(c["steer_gain"]))
                        rowvals.append(c["retention"][pk])
                    else:
                        rowvals.append(np.nan)
                M.append(rowvals)
                labels.append(f"{src[:5]}→{tgt[:5]} @{tok} α{ak}")
    if M:
        M = np.array(M)
        fig, ax = plt.subplots(figsize=(3.6, 0.30 * len(labels) + 1.4))
        im = ax.imshow(M, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels([VAR_LABEL[v] for v in VARIANTS], fontsize=9)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=5)
        for i in range(len(labels)):
            for j in range(len(VARIANTS)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5,
                            color="black")
        fig.colorbar(im, ax=ax, shrink=0.5, label="peak-layer retention (low = ablation kills steering)")
        ax.set_title("Steering-gain retention: F'⊥F vs F'−F", fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / "retention_compare.png", dpi=150)
        plt.close(fig)
        print("saved retention_compare.png")
    print(f"DONE -> {fig_dir}")


if __name__ == "__main__":
    main()
