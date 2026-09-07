#!/usr/bin/env python
"""Build steering vectors for every cell of the grid, both directions (CPU only).

Reads the Stage-B cue-token captures (artifacts/style_properties/site_acts/<prop>__<pol>.npz,
role==1) and the prescreen behavioural labels, applies each cell's k / success / pairing
rules, and writes

  artifacts/style_properties/steering_vectors_grid/<cell>/<prop>.npz
    v_alt [29,4096]  vector for alt-steering
    v_nat [29,4096]  vector for nat-steering
    n_alt, n_nat, n_paired, k_min_used, norms

Pairing (mean difference only):
  paired   - sites kept only if the SAME site passes the filter in both twins; the vector is
             the mean of per-site differences (identical context/position on both sides).
  unpaired - each polarity filtered independently, then subtract the two means.
For meanact there is no pairing: v_alt = mean(alt sites), v_nat = mean(nat sites).

Direction for mean difference: v_nat = -v_alt (same object, opposite sign). For meanact the
two directions are different objects (each convention's own mean).
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
from src.sandbox.ext_styleprops.grid import CELLS, BY_NAME

ACTS = ARTIFACTS_ROOT / "style_properties" / "site_acts"
PRE = ARTIFACTS_ROOT / "style_properties" / "prescreen"
OUT = ARTIFACTS_ROOT / "style_properties" / "steering_vectors_grid"
ROLE_CUE = 1
MIN_SITES = 20          # below this a cell/property is recorded as underpowered


def load_sites(prop):
    """-> per polarity: acts [n,29,4096] fp32-ready, k [n], site key [n] (doc_id, k)."""
    out = {}
    labels = {}
    for r in json.load(open(PRE / f"{prop}.json"))["records"]:
        labels[(r["doc_id"], int(r["k"]), r["pol"])] = r["label"]
    for pol in ("nat", "alt"):
        z = np.load(ACTS / f"{prop}__{pol}.npz")
        m = z["role"] == ROLE_CUE
        doc_ids = z["doc_ids"][z["doc"][m]]
        ks = z["k"][m].astype(int)
        keys = list(zip(doc_ids.tolist(), ks.tolist()))
        succ = np.array([labels.get((d, k, pol)) == pol for d, k in keys])
        out[pol] = dict(acts=z["acts"][m], k=ks, keys=keys, succ=succ)
    return out


def select(side, cell):
    keep = side["k"] >= cell.k_min
    if cell.require_success:
        keep &= side["succ"]
    return keep


def build(prop, cell, S):
    """S = load_sites(prop), loaded ONCE per property and reused for all cells (the
    captures are hundreds of MB each on a network volume; reloading per cell is the
    difference between minutes and hours)."""
    sel = {pol: select(S[pol], cell) for pol in ("nat", "alt")}
    info = dict(k_min_used=cell.k_min)
    if cell.technique == "meanact":
        v_alt = S["alt"]["acts"][sel["alt"]].astype(np.float32).mean(0)
        v_nat = S["nat"]["acts"][sel["nat"]].astype(np.float32).mean(0)
        info.update(n_alt=int(sel["alt"].sum()), n_nat=int(sel["nat"].sum()), n_paired=0)
    elif cell.paired:
        # match sites by (doc_id, k); keep only those passing the filter on BOTH sides
        idx_nat = {key: i for i, (key, ok) in enumerate(zip(S["nat"]["keys"], sel["nat"])) if ok}
        idx_alt = {key: i for i, (key, ok) in enumerate(zip(S["alt"]["keys"], sel["alt"])) if ok}
        common = sorted(set(idx_nat) & set(idx_alt))
        ia = np.array([idx_alt[k] for k in common], dtype=int)
        inn = np.array([idx_nat[k] for k in common], dtype=int)
        if len(common) == 0:
            return None, None, dict(info, n_alt=0, n_nat=0, n_paired=0)
        d = (S["alt"]["acts"][ia].astype(np.float32) - S["nat"]["acts"][inn].astype(np.float32))
        v_alt = d.mean(0)
        v_nat = -v_alt
        info.update(n_alt=len(common), n_nat=len(common), n_paired=len(common))
    else:   # unpaired mean difference
        ma = S["alt"]["acts"][sel["alt"]].astype(np.float32).mean(0)
        mn = S["nat"]["acts"][sel["nat"]].astype(np.float32).mean(0)
        v_alt = ma - mn
        v_nat = -v_alt
        info.update(n_alt=int(sel["alt"].sum()), n_nat=int(sel["nat"].sum()), n_paired=0)
    # ---- sanity asserts on the selection itself
    for pol in ("nat", "alt"):
        ks = S[pol]["k"][sel[pol]]
        assert ks.size == 0 or ks.min() >= cell.k_min, f"{cell.name}/{prop}: k filter violated"
        if cell.require_success:
            assert S[pol]["succ"][sel[pol]].all(), f"{cell.name}/{prop}: success filter violated"
    if cell.paired:
        assert info["n_alt"] == info["n_nat"] == info["n_paired"]
    return v_alt, v_nat, info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--force", action="store_true", help="rebuild vectors that already exist")
    args = ap.parse_args()
    props = sorted(json.load(open(REPO_ROOT / "task_splits" / "style_properties_pool.json"))["pass"])
    cells = [BY_NAME[c] for c in args.cells] if args.cells else list(CELLS)
    for cell in cells:
        (OUT / cell.name).mkdir(parents=True, exist_ok=True)
    warn, counts = [], {}
    for prop in props:                      # property-outer: load the captures once
        S = load_sites(prop)
        for cell in cells:
            out_f = OUT / cell.name / f"{prop}.npz"
            if out_f.exists() and not args.force:
                continue
            v_alt, v_nat, info = build(prop, cell, S)
            if v_alt is None or min(info["n_alt"], info["n_nat"]) < MIN_SITES:
                warn.append(f"{cell.name}/{prop}: n_alt={info['n_alt']} n_nat={info['n_nat']}")
            if v_alt is None:
                continue
            np.savez(out_f, v_alt=v_alt, v_nat=v_nat,
                     norm_alt=np.linalg.norm(v_alt, axis=1), norm_nat=np.linalg.norm(v_nat, axis=1),
                     **info)
            counts.setdefault(cell.name, []).append(f"{prop[:8]}:{info['n_alt']}/{info['n_nat']}")
        del S
        print(f"[{prop}] done", flush=True)
    for cn, line in counts.items():
        print(f"{cn:38s} " + " ".join(line[:5]) + (" ..." if len(line) > 5 else ""), flush=True)
    print(f"\n-> {OUT}")
    if warn:
        print(f"\nUNDERPOWERED (< {MIN_SITES} sites), recorded in specs:")
        for w in warn[:20]:
            print("  " + w)
        print(f"  ... {len(warn)} total" if len(warn) > 20 else "")


if __name__ == "__main__":
    main()
