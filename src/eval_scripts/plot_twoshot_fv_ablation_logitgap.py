"""
Plot the 2-shot FV-projection-ablation task-imitation study (per-token layer sweeps; CPU-only).

Reads results/direction2_label_geometry/twoshot_fv_ablation_imitation/<task_pair>/
<direction>_<token>_alpha{a}_layersweep.csv (columns: layer, clean_mean, ablate_mean, steer_mean,
steer_sem, steer_gain, steer_ablate_mean, steer_ablate_sem, steer_ablate_gain, retention).

For each α, a grid of panels rows = direction (4), cols = steer token position (label1, label2, qfinal):
each panel plots, vs injection layer ℓ,
  - steer(t,ℓ)         Δlogit curve (no ablate)      with SEM band
  - steer+ablate(t,ℓ)  Δlogit curve (F'⊥F ablated)   with SEM band
  - clean, ablate      horizontal baseline lines (steering-free)
The localized story, per token: where steer(t,ℓ) rises above clean, does steer+ablate(t,ℓ) fall back
toward ablate? Plus a retention overview (rows = direction×token×α, cols = layer) at effective layers.
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

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.paths import LABEL_GEOMETRY_DIR

TASK_PAIRS = ["antonym_synonym", "next_number_digits_prev_number_digits"]
TOKENS = ["label1", "label2", "qfinal"]


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


def discover_directions(root):
    """Ordered list of direction names present (across both task pairs)."""
    names = []
    for tp in TASK_PAIRS:
        d = Path(root) / tp
        if not d.is_dir():
            continue
        seen = set()
        for p in sorted(d.glob("*_layersweep.csv")):
            # <direction>_<token>_alpha{a}_layersweep.csv ; token is one of TOKENS
            stem = p.name[:-len("_layersweep.csv")]
            for tok in TOKENS:
                marker = f"_{tok}_alpha"
                if marker in stem:
                    name = stem.split(marker)[0]
                    if name not in seen:
                        seen.add(name); names.append((tp, name))
                    break
    return names


def csv_path(root, tp, name, tok, ak):
    return Path(root) / tp / f"{name}_{tok}_alpha{ak}_layersweep.csv"


def panel(ax, c, title, show_ylabel, show_xlabel):
    x = c["layer"]
    clean = c["clean_mean"][0]; ablate = c["ablate_mean"][0]
    ax.axhline(clean, color="#616161", lw=1.0, ls="--", label=f"clean ({clean:+.2f})")
    ax.axhline(ablate, color="#b39ddb", lw=1.0, ls="--", label=f"ablate ({ablate:+.2f})")
    ax.plot(x, c["steer_mean"], color="#4c72b0", lw=1.5, label="steer")
    ax.fill_between(x, c["steer_mean"] - c["steer_sem"], c["steer_mean"] + c["steer_sem"],
                    color="#4c72b0", alpha=0.2)
    ax.plot(x, c["steer_ablate_mean"], color="#8452a8", lw=1.5, label="steer+ablate")
    ax.fill_between(x, c["steer_ablate_mean"] - c["steer_ablate_sem"],
                    c["steer_ablate_mean"] + c["steer_ablate_sem"], color="#8452a8", alpha=0.2)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_title(title, fontsize=8)
    if show_xlabel:
        ax.set_xlabel("injection layer ℓ", fontsize=7)
    if show_ylabel:
        ax.set_ylabel("Δlogit = logit(a_tgt)−logit(a_src)", fontsize=7)
    ax.tick_params(labelsize=6)


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

    ret_rows, ret_labels = [], []
    for alpha in args.alphas:
        ak = f"{alpha:g}"
        nr, nc = len(dirs), len(TOKENS)
        fig, axes = plt.subplots(nr, nc, figsize=(4.3 * nc, 2.9 * nr), squeeze=False)
        any_data = False
        for ri, (tp, name) in enumerate(dirs):
            src, tgt = name.split("_to_")
            for ci, tok in enumerate(TOKENS):
                ax = axes[ri][ci]
                p = csv_path(root, tp, name, tok, ak)
                if not p.exists():
                    ax.set_axis_off(); continue
                any_data = True
                c = load_csv(p)
                peak = int(np.nanargmax(c["steer_gain"]))
                title = (f"{src}→{tgt}\nsteer@{tok}: peak L{peak} gain {c['steer_gain'][peak]:+.2f}, "
                         f"ret {c['retention'][peak]:.2f}")
                panel(ax, c, title, show_ylabel=(ci == 0), show_xlabel=(ri == nr - 1))
                if ri == 0 and ci == nc - 1:
                    ax.legend(fontsize=6, loc="best")
                # collect retention row (effective layers only)
                eff = c["steer_gain"] > 0.5
                ret_rows.append(np.where(eff, c["retention"], np.nan))
                ret_labels.append(f"{src[:5]}→{tgt[:5]} @{tok} α{ak}")
        if not any_data:
            plt.close(fig); continue
        fig.suptitle(f"2-shot FV-projection-ablation — α={ak}  (separate layer-sweep per steer token)\n"
                     f"steer ONE token at ONE layer ℓ; ablate F'⊥F at qfinal (all layers)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(fig_dir / f"layersweep_by_token_alpha{ak}.png", dpi=150)
        plt.close(fig)
        print(f"saved layersweep_by_token_alpha{ak}.png")

    if ret_rows:
        M = np.vstack(ret_rows)
        fig, ax = plt.subplots(figsize=(0.26 * M.shape[1] + 3.5, 0.32 * M.shape[0] + 1.6))
        im = ax.imshow(M, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xlabel("injection layer ℓ", fontsize=8)
        ax.set_yticks(range(len(ret_labels))); ax.set_yticklabels(ret_labels, fontsize=5)
        fig.colorbar(im, ax=ax, shrink=0.7, label="retention = steer_ablate_gain / steer_gain")
        ax.set_title("Steering-gain retention under F'⊥F ablation, per token\n"
                     "(only injection layers with steer_gain>0.5; low=FV direction mediates)", fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / "retention_overview.png", dpi=150)
        plt.close(fig)
        print("saved retention_overview.png")
    print(f"DONE -> {fig_dir}")


if __name__ == "__main__":
    main()
