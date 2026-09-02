#!/usr/bin/env python
"""Build mean-activation steering vectors for the style properties (Stage C/D).

From the Stage B site captures (artifacts/style_properties/site_acts/<prop>__<pol>.npz):
  meandiff[l] = mean evidence-token activation under alt − under nat   (primary vector)
  rawalt[l]   = mean evidence-token activation under alt               (raw-mean arm,
                the ICL raw-mean-vs-mean-diff comparison)
for every capture layer l (0 = embedding, l>=1 = output of block l-1).

Output: artifacts/style_properties/steering_vectors/<prop>.npz
  {meandiff [29,4096] fp32, rawalt [29,4096] fp32, n_nat, n_alt, norms per layer}
"""
import json
import sys
from pathlib import Path

import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT

IN_DIR = ARTIFACTS_ROOT / "style_properties" / "site_acts"
OUT_DIR = ARTIFACTS_ROOT / "style_properties" / "steering_vectors"
POOL_PATH = REPO_ROOT / "task_splits" / "style_properties_pool.json"
ROLE_EVID = 0


def main():
    props = sorted(json.load(open(POOL_PATH))["pass"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in props:
        means, ns = {}, {}
        for pol in ("nat", "alt"):
            z = np.load(IN_DIR / f"{name}__{pol}.npz")
            m = z["role"] == ROLE_EVID
            means[pol] = z["acts"][m].astype(np.float32).mean(0)   # [29, 4096]
            ns[pol] = int(m.sum())
        meandiff = means["alt"] - means["nat"]
        np.savez(OUT_DIR / f"{name}.npz", meandiff=meandiff, rawalt=means["alt"],
                 n_nat=ns["nat"], n_alt=ns["alt"],
                 norm_diff=np.linalg.norm(meandiff, axis=1),
                 norm_rawalt=np.linalg.norm(means["alt"], axis=1))
        nd = np.linalg.norm(meandiff, axis=1)
        print(f"{name}: n={ns['nat']}/{ns['alt']}  |diff| L2={nd[2]:.1f} L6={nd[6]:.1f} "
              f"L13={nd[13]:.1f} L24={nd[24]:.1f}", flush=True)


if __name__ == "__main__":
    main()
