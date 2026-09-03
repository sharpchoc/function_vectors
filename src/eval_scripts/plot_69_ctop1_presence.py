#!/usr/bin/env python
"""Claim 5 presence figure for the decomposed read feature (carrier c, task-unique v1).

Reads artifacts/69_task_run/ctop1_presence/<task>.npz (capture_69_ctop1_presence.py) and the
write-feature presence (feature_locations/direct_FV_presence/fv_location.npz, cos to the task's
own FV). Averages the 32 structural columns into three token types (input / target / cue) as
plot_69_read_vs_write_presence.py does, then the 69-task mean per layer.

Writes results/69_task_run/feature_locations/ctop1/:
  presence_headline.png   SIMPLE: two panels — READ (cos to v1 at target tokens; carrier as a
                          faint reference) | WRITE (cos to v_A at cue tokens); one line each
  presence_full.png       all three token types × {carrier, v1, write}
  presence.csv            per-layer means for every series
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

import os
_MR = os.environ.get("MEANRESID") == "1"   # mean-residual task-unique part (u_hat_A) instead of SVD v1
AR = ARTIFACTS_ROOT / "69_task_run" / ("meanresid_presence" if _MR else "ctop1_presence")
UL = "$\\hat u_A$" if _MR else "$v_1$"
WRITE = TASK69_RUN_DIR / "feature_locations" / "direct_FV_presence" / "fv_location.npz"
OUT = TASK69_RUN_DIR / "feature_locations" / ("meanresid" if _MR else "ctop1")
CAT = {"cue": "#2a78d6", "target": "#eb6834", "input": "#1baf7a"}


def col_type(name):
    n = str(name)
    if "cue" in n or n.endswith("_pre") or "colon" in n:
        return "cue"
    if "label" in n or "target" in n:
        return "target"
    if "inp" in n or "input" in n:
        return "input"
    return None


def type_profiles(cos_mean, columns):
    out = {}
    for k in CAT:
        idx = [i for i, c in enumerate(columns) if col_type(c) == k]
        out[k] = cos_mean[:, idx].mean(axis=1) if idx else np.full(cos_mean.shape[0], np.nan)
    return out


def main():
    files = sorted(AR.glob("*.npz"))
    assert len(files) == 69, f"expected 69 task files, found {len(files)}"
    # v1's SVD sign is arbitrary; orient each task's v1 along the task's own carrier-removed
    # read feature (sign of n_A = <mbar_A - c, v1>) so "presence" reads as positive alignment.
    import torch
    vecs = torch.load(ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "bankA"
                      / ("carrier_plus_meanresid_vectors.pt" if _MR else "carrier_plus_top1_vectors.pt"), map_location="cpu", weights_only=False)["tasks"]
    acc = {"carrier": {k: [] for k in CAT}, "v1": {k: [] for k in CAT}}
    cols = None
    for f in files:
        z = np.load(f, allow_pickle=True)
        cols = [str(c) for c in z["columns"]]
        sgn = 1.0 if _MR else (1.0 if vecs[f.stem]["n_A"] >= 0 else -1.0)   # u_hat_A is already oriented
        for d in ("carrier", "v1"):
            prof = type_profiles(z[f"cos_{d}"] * (sgn if d == "v1" else 1.0), cols)
            for k in CAT:
                acc[d][k].append(prof[k])
    prof = {(d, k): np.mean(acc[d][k], axis=0) for d in acc for k in CAT}
    zw = np.load(WRITE, allow_pickle=False)
    wcols = [str(c) for c in zw["columns"]]
    wprof = type_profiles(zw["cos"].mean(axis=0), wcols)   # (69,28,32) -> task mean
    n_layers = len(prof[("v1", "target")])
    L = np.arange(n_layers)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "presence.csv", "w") as f:
        keys = [f"{d}_{k}" for d in ("carrier", "v1") for k in CAT] + [f"write_{k}" for k in CAT]
        f.write("layer," + ",".join(keys) + "\n")
        for l in L:
            vals = [prof[(d, k)][l] for d in ("carrier", "v1") for k in CAT] + [wprof[k][l] for k in CAT]
            f.write(f"{l}," + ",".join(f"{v:.5f}" for v in vals) + "\n")

    def style(ax):
        ax.grid(axis="y", color="#e8eae6", lw=0.8, zorder=0)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.set_xticks(range(0, n_layers, 3))
        ax.tick_params(labelsize=10.5)

    # ---- SIMPLE headline: read (v1 @ target) | write (v_A @ cue) ----
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=150)
    fig.patch.set_facecolor("white")
    ax = axes[0]; ax.set_facecolor("white")
    y = prof[("v1", "target")]; pk = int(np.nanargmax(y))
    ax.plot(L, y, color=CAT["target"], lw=2.4, marker="o", ms=4.5, mfc=CAT["target"], mec="white",
            label=f"task-unique direction {UL}")
    ax.plot(L, prof[("carrier", "target")], color="0.6", lw=1.6, ls=(0, (4, 3)),
            label="shared carrier $c$")
    ax.set_title(f"READ: cosine at demonstration target tokens (peak L{pk})", fontsize=12, loc="left")
    ax.set_xlabel("layer"); ax.set_ylabel("mean cosine (69 tasks)")
    ax.legend(frameon=False, fontsize=10); style(ax)
    ax = axes[1]; ax.set_facecolor("white")
    yw = wprof["cue"]; pkw = int(np.nanargmax(yw))
    ax.plot(L, yw, color=CAT["cue"], lw=2.4, marker="o", ms=4.5, mfc=CAT["cue"], mec="white",
            label="function vector $v_A$")
    ax.set_title(f"WRITE: cosine at cue tokens (peak L{pkw})", fontsize=12, loc="left")
    ax.set_xlabel("layer"); ax.legend(frameon=False, fontsize=10); style(ax)
    fig.suptitle("Where the read and write features are present", fontsize=13.5, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "presence_headline.png", facecolor="white")
    plt.close(fig)

    # ---- full: three token types × three directions ----
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4), dpi=150, sharex=True)
    fig.patch.set_facecolor("white")
    for ax, (title, series) in zip(axes, (("shared carrier $c$", {k: prof[("carrier", k)] for k in CAT}),
                                          (f"task-unique direction {UL}", {k: prof[("v1", k)] for k in CAT}),
                                          ("function vector $v_A$ (write)", wprof))):
        ax.set_facecolor("white")
        for k in ("cue", "target", "input"):
            ax.plot(L, series[k], color=CAT[k], lw=2.0, label=k)
        ax.set_title(title, fontsize=12, loc="left"); ax.set_xlabel("layer"); style(ax)
    axes[0].set_ylabel("mean cosine (69 tasks)")
    axes[0].legend(frameon=False, fontsize=10, title="token type")
    fig.tight_layout()
    fig.savefig(OUT / "presence_full.png", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}/presence_headline.png, presence_full.png, presence.csv | "
          f"v1@target peak L{pk} = {y[pk]:.3f}; carrier@target L{pk} = {prof[('carrier','target')][pk]:.3f}; "
          f"write@cue peak L{pkw} = {yw[pkw]:.3f}")


if __name__ == "__main__":
    main()
