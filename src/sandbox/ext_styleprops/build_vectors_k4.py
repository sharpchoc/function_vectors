#!/usr/bin/env python
"""Cue-token steering vectors from HIGH-k, BEHAVIOURALLY VERIFIED sites (user spec 2026-09-05).

v[l] = mean(alt cue act | k >= K, model emitted ALT there) - mean(nat cue act | k >= K, model emitted NAT)

Rationale: at k=0 the twins are character-identical, so their activations are identical and
contribute nothing; the informative contrast is a context that has established the convention.
Restricting to sites where the sampled continuation actually followed that context's own
convention keeps only states where the model demonstrably holds it.

Behavioural labels come from the Stage-A prescreen records (same dataset, same sites),
joined on (doc_id, k, polarity).

Output: artifacts/style_properties/steering_vectors_k4/<prop>.npz {cuediff_k4 [29,4096], counts}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT

ACTS = ARTIFACTS_ROOT / "style_properties" / "site_acts"
PRE = ARTIFACTS_ROOT / "style_properties" / "prescreen"
OLD = ARTIFACTS_ROOT / "style_properties" / "steering_vectors"
OUT = ARTIFACTS_ROOT / "style_properties" / "steering_vectors_k4"
ROLE_CUE = 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kmin", type=int, default=4)
    ap.add_argument("--min_n", type=int, default=30, help="fall back to lower k if too few sites")
    args = ap.parse_args()
    pool = sorted(json.load(open(REPO_ROOT / "task_splits" / "style_properties_pool.json"))["pass"])
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'property':15s} kmin  n_nat  n_alt | ||v_k4|| L16/L20/L24   cos to old-alldiff")
    for name in pool:
        # behavioural labels: (doc_id, k, pol) -> emitted label
        lab = {}
        for r in json.load(open(PRE / f"{name}.json"))["records"]:
            lab[(r["doc_id"], r["k"], r["pol"])] = r["label"]
        sel, kmin = {}, args.kmin
        while kmin >= 1:
            ok = True
            for pol in ("nat", "alt"):
                z = np.load(ACTS / f"{name}__{pol}.npz")
                m = z["role"] == ROLE_CUE
                ids = z["doc_ids"][z["doc"][m]]
                keep = np.array([(lab.get((d, int(k), pol)) == pol) and int(k) >= kmin
                                 for d, k in zip(ids, z["k"][m])])
                sel[pol] = (z, m, keep)
                ok &= keep.sum() >= args.min_n
            if ok:
                break
            kmin -= 1
        means = {}
        for pol in ("nat", "alt"):
            z, m, keep = sel[pol]
            means[pol] = z["acts"][m][keep].astype(np.float32).mean(0)
        v = means["alt"] - means["nat"]
        old = np.load(OLD / f"{name}.npz")["cuediff"]
        cos = [float(v[L] @ old[L] / (np.linalg.norm(v[L]) * np.linalg.norm(old[L]) + 1e-9))
               for L in (16, 20, 24)]
        np.savez(OUT / f"{name}.npz", cuediff_k4=v, kmin=kmin,
                 n_nat=int(sel["nat"][2].sum()), n_alt=int(sel["alt"][2].sum()),
                 norm=np.linalg.norm(v, axis=1))
        nm = np.linalg.norm(v, axis=1)
        print(f"{name:15s} {kmin:4d} {sel['nat'][2].sum():6d} {sel['alt'][2].sum():6d} | "
              f"{nm[16]:5.1f}/{nm[20]:5.1f}/{nm[24]:5.1f}   {cos[0]:.2f}/{cos[1]:.2f}/{cos[2]:.2f}")


if __name__ == "__main__":
    main()
