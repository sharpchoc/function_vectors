#!/usr/bin/env python
"""Read-feature summary (CPU): per family the prompt counts, evidence tokens per prompt, |diff|/|mean| and split-half cosine by layer of
the read vectors (capture_read.py), and the cosine between the read difference and the write vector (capture_cues.py) at the same layer.
Outputs -> <results>/read_features/: read_vectors.csv, read_summary.png, read_vs_write.png."""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--tag", default="k3_train")
    args = ap.parse_args()
    MP = model_paths(args.model); R = MP["results"]; OUT = R / "read_features"; OUT.mkdir(parents=True, exist_ok=True)
    pool = json.load(open(R / "code_pool.json"))["pool"]; V = MP["read_features"] / f"vectors_{args.tag}"; W = MP["steering"] / f"vectors_{args.tag}"
    rows, cosL, ratioL, rwL = [], {}, {}, {}
    LS = [4, 8, 12, 16, 20, 24]
    for f in pool:
        z = np.load(V / f"{f}.npz"); d = z["diff"]; m = z["mean_nat"]; ratio = np.linalg.norm(d, axis=1) / np.linalg.norm(m, axis=1)
        cosL[f] = z["split_half_cos"]; ratioL[f] = ratio
        rw = None
        if (W / f"{f}.npz").exists():
            w = np.load(W / f"{f}.npz")["v_nat"]; rw = (d * w).sum(1) / (np.linalg.norm(d, axis=1) * np.linalg.norm(w, axis=1) + 1e-9); rwL[f] = rw
        rows.append(dict(family=f, n_nat=int(z["n_nat"]), n_alt=int(z["n_alt"]), tok_nat=round(float(z["tokens_per_prompt_nat"][0]), 1), tok_alt=round(float(z["tokens_per_prompt_alt"][0]), 1),
                         **{f"ratio_L{l}": round(float(ratio[l - 1]), 3) for l in LS}, **{f"cos_L{l}": round(float(cosL[f][l - 1]), 3) for l in LS},
                         **{f"read_write_cos_L{l}": (round(float(rw[l - 1]), 3) if rw is not None else "") for l in LS}))
    with open(OUT / "read_vectors.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5)); x = np.arange(1, 29)
    for f in pool:
        axes[0].plot(x, cosL[f], lw=.8, alpha=.6); axes[1].plot(x, ratioL[f], lw=.8, alpha=.6)
    axes[0].plot(x, np.mean([cosL[f] for f in pool], 0), "k-", lw=2.5, label="mean over families"); axes[1].plot(x, np.mean([ratioL[f] for f in pool], 0), "k-", lw=2.5, label="mean over families")
    axes[0].set_title("split-half cosine of the read difference (halves by document)"); axes[1].set_title("|mean_nat − mean_alt| / |mean_nat| (read feature)")
    for ax in axes: ax.set_xlabel("layer (output of block L)"); ax.grid(alpha=.3); ax.legend()
    fig.suptitle(f"Read features over evidence tokens, {len(pool)} code-convention families, Qwen2.5-7B base (k = 3, 4 correct prompts, training docs)"); fig.tight_layout(); fig.savefig(OUT / "read_summary.png", dpi=130); plt.close(fig)
    if rwL:
        fig, ax = plt.subplots(figsize=(8, 5))
        for f in rwL: ax.plot(x, rwL[f], lw=.8, alpha=.6)
        ax.plot(x, np.mean([rwL[f] for f in rwL], 0), "k-", lw=2.5, label="mean over families"); ax.axvline(24, color="r", ls="--", lw=1, label="write feature layer 24")
        ax.set_xlabel("layer"); ax.set_ylabel("cos(read difference, write vector)"); ax.set_title("Read difference vs write vector at the same layer, per family"); ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(OUT / "read_vs_write.png", dpi=130); plt.close(fig)
    print(f"{len(rows)} families -> {OUT}")


if __name__ == "__main__":
    main()
