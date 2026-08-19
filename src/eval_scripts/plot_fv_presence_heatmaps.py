#!/usr/bin/env python
"""Heatmaps: how much FV direction is in the residual stream, unsteered vs steered.

Reads artifacts/69_task_run/fv_presence/<task>.npz (fv_presence_heatmaps.py) and writes
results/69_task_run/fv_presence_heatmaps/<task>.png — a 2x3 grid:
  row 1: cos(residual, v_A)                unsteered | steered | steered - unsteered
  row 2: <residual, v_A/||v_A||>           unsteered | steered | steered - unsteered
x = token position (labelled with the actual tokens), y = layer (block output 0..27).
The injected position (the ' _' label slot) is marked.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR  # noqa: E402

AR = ARTIFACTS_ROOT / "69_task_run" / "fv_presence"
OUT = TASK69_RUN_DIR / "fv_presence_heatmaps"


def panel(ax, M, title, tokens, inj_idx, cmap, vmin=None, vmax=None, center=False):
    if center:
        m = float(np.abs(M).max())
        vmin, vmax = -m, m
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(len(tokens)),
                  [t.replace("\n", "\\n") for t in tokens], rotation=90, fontsize=5.5)
    ax.axvline(inj_idx, color="lime", lw=1.2, ls="--")
    ax.set_ylabel("layer (block output)")
    return im


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(AR.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        task = f.stem
        tokens = [str(t) for t in z["tokens"]]
        inj = int(z["inj_idx"])
        cu, cs = z["cos_unsteered"], z["cos_steered"]
        pu, ps = z["proj_unsteered"], z["proj_steered"]
        fig, axes = plt.subplots(2, 3, figsize=(max(13, 0.42 * len(tokens)), 8.4), dpi=150)
        cmax = max(abs(cu).max(), abs(cs).max())
        pmax = max(abs(pu).max(), abs(ps).max())
        panel(axes[0, 0], cu, "cos(resid, FV) — unsteered", tokens, inj, "RdBu_r", -cmax, cmax)
        panel(axes[0, 1], cs, "cos(resid, FV) — steered (read dir @L3)", tokens, inj,
              "RdBu_r", -cmax, cmax)
        im = panel(axes[0, 2], cs - cu, "cos: steered - unsteered", tokens, inj, "PuOr_r",
                   center=True)
        fig.colorbar(im, ax=axes[0, 2], fraction=0.03)
        panel(axes[1, 0], pu, "projection onto FV — unsteered", tokens, inj, "RdBu_r",
              -pmax, pmax)
        panel(axes[1, 1], ps, "projection onto FV — steered", tokens, inj, "RdBu_r",
              -pmax, pmax)
        im2 = panel(axes[1, 2], ps - pu, "projection: steered - unsteered", tokens, inj,
                    "PuOr_r", center=True)
        fig.colorbar(im2, ax=axes[1, 2], fraction=0.03)
        fig.suptitle(f"{task}: FV content in the residual stream — {int(z['n_prompts'])} "
                     f"prompts, 1-shot '_' scaffold\n"
                     f"steer = {str(z['bracket'])} read direction, alpha={float(z['alpha'])} "
                     f"x natural magnitude, injected at the ' _' slot (green) at L"
                     f"{int(z['inject_layer'])}; ||FV||={float(z['fv_norm']):.0f}, "
                     f"||read dir||={float(z['readdir_norm']):.0f}", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"{task}.png", bbox_inches="tight")
        # quick numbers at the final cue token (where the FV is read out)
        print(f"{task}: cue-token cos unsteered max {cu[:, -1].max():.3f} "
              f"-> steered {cs[:, -1].max():.3f} | cue proj max {pu[:, -1].max():.1f} "
              f"-> {ps[:, -1].max():.1f} | biggest cos gain {np.max(cs - cu):.3f} "
              f"at layer {np.unravel_index(np.argmax(cs - cu), cu.shape)[0]}, "
              f"pos {np.unravel_index(np.argmax(cs - cu), cu.shape)[1]}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
