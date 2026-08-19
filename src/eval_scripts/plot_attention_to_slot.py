#!/usr/bin/env python
"""Attention from the final cue token to the injected ' _' label slot.

Reads artifacts/69_task_run/attn_to_slot/<task>.npz and writes
results/69_task_run/fv_presence_heatmaps/attn_<task>.png:
  (a) layer x head attention to ' _' — unsteered, steered, difference
  (b) L13 per-head bars (FV heads hatched)
  (c) where the cue token actually attends at L13 (mean over heads, by position)
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

AR = ARTIFACTS_ROOT / "69_task_run" / "attn_to_slot"
OUT = TASK69_RUN_DIR / "read_write_relationship" / "top_down" / "fv_presence_heatmaps"
LAYER = 13


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(AR.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        task = f.stem
        au, as_ = z["attn_unsteered"], z["attn_steered"]
        inj = int(z["inj_idx"]); fvm = z["fv_head_mask"]
        toks = [str(t).replace("\n", "\\n") for t in z["tokens"]]
        u2, s2 = au[:, :, inj], as_[:, :, inj]           # (L, H)

        fig = plt.figure(figsize=(15, 8.6), dpi=150)
        gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
        vmax = max(u2.max(), s2.max())
        for j, (M, ttl) in enumerate(((u2, "unsteered"), (s2, "steered (read dir @L3)"))):
            ax = fig.add_subplot(gs[0, j])
            im = ax.imshow(M, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=vmax)
            ax.set_title(f"attention(final cue -> ' _') — {ttl}", fontsize=9)
            ax.set_xlabel("head"); ax.set_ylabel("layer")
            ax.axhline(LAYER, color="cyan", lw=0.8, ls=":")
            fig.colorbar(im, ax=ax, fraction=0.04)
        ax = fig.add_subplot(gs[0, 2])
        d = s2 - u2
        m = float(np.abs(d).max())
        im = ax.imshow(d, aspect="auto", origin="lower", cmap="PuOr_r", vmin=-m, vmax=m)
        ax.set_title("difference (steered - unsteered)", fontsize=9)
        ax.set_xlabel("head"); ax.set_ylabel("layer")
        ax.axhline(LAYER, color="cyan", lw=0.8, ls=":")
        fig.colorbar(im, ax=ax, fraction=0.04)

        ax = fig.add_subplot(gs[1, 0:2])
        x = np.arange(u2.shape[1]); w = 0.4
        b1 = ax.bar(x - w / 2, u2[LAYER], w, color="0.5", label="unsteered")
        b2 = ax.bar(x + w / 2, s2[LAYER], w, color="tab:blue", label="steered")
        for h in range(u2.shape[1]):
            if fvm[LAYER, h]:
                b1[h].set_hatch("///"); b2[h].set_hatch("///")
        ax.set_xticks(x, [f"H{h}" for h in x], fontsize=7)
        ax.set_ylabel(f"attention to ' _' at L{LAYER}")
        ax.set_title(f"L{LAYER} per-head attention from the final cue token to the ' _' slot "
                     f"(hatched = one of the 37 FV heads)", fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")

        ax = fig.add_subplot(gs[1, 2])
        prof_u = au[LAYER].mean(0); prof_s = as_[LAYER].mean(0)
        ax.bar(np.arange(len(toks)) - 0.2, prof_u, 0.4, color="0.5", label="unsteered")
        ax.bar(np.arange(len(toks)) + 0.2, prof_s, 0.4, color="tab:blue", label="steered")
        ax.axvline(inj, color="lime", ls="--", lw=1)
        ax.set_xticks(range(len(toks)), toks, rotation=90, fontsize=6)
        ax.set_title(f"L{LAYER}: where the cue token attends (mean over heads)", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.25, axis="y")

        fig.suptitle(f"{task}: does the cue token look at the injected ' _' slot? "
                     f"({int(z['n_prompts'])} prompts)", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"attn_{task}.png", bbox_inches="tight")
        print(f"{task}: L{LAYER} mean attn to '_' {u2[LAYER].mean():.4f} -> "
              f"{s2[LAYER].mean():.4f} | sink (pos0+pos1) {prof_u[:2].sum():.3f} | "
              f"max head L{LAYER}H{u2[LAYER].argmax()}={u2[LAYER].max():.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
