#!/usr/bin/env python
"""Derive a fixed-size head set from a pooled-sparse fit: the top-k heads by final coefficient c.

Reads <fit_root>/pooled_sparse/{selection.json, coeffs_final.pt} and writes
<out_root>/pooled_sparse/selection.json in the same schema (n_selected = k, selected_heads =
[[layer, head, c], ...] sorted by c desc, selected_flat), so eval_ext.py can score it directly.
Inject-layer sweep add-on (user request 2026-10-06): head-count-matched comparison across layers.
"""
import argparse
import json
import os
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit_root", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--k", type=int, default=50)
    a = ap.parse_args()
    sel = json.load(open(a.fit_root / "pooled_sparse" / "selection.json"))
    c = torch.load(a.fit_root / "pooled_sparse" / "coeffs_final.pt", map_location="cpu",
                   weights_only=False)["c"].float().flatten()
    H = {448: 16, 784: 28}[len(c)]        # heads per layer: GPT-J 16x28, Qwen2.5-7B 28x28
    assert all(l * H + h in sel["selected_flat"] for l, h, _ in sel["selected_heads"]), "flat/[l,h] mismatch"
    # rank: c desc, ties (saturated heads) by the first-order dropout lambda -dNLL/dc_h desc
    gp = a.fit_root / "pooled_sparse" / "grad_at_final.pt"
    g = torch.load(gp, map_location="cpu", weights_only=False)["grad"].float().flatten()
    assert g.shape == c.shape
    key = sorted(range(len(c)), key=lambda f: (-round(float(c[f]), 3), float(g[f])))
    order = key[:a.k]
    flat = sorted(order)
    heads = [[int(f // H), int(f % H), round(float(c[f]), 4)] for f in sorted(order, key=lambda f: -float(c[f]))]
    out = dict(sel)
    out.update({"n_selected": a.k, "selected_heads": heads, "selected_flat": flat, "c_high": None,
                "derived": f"top-{a.k} heads by final c of {a.fit_root}/pooled_sparse (parent n_selected={sel['n_selected']})",
                "min_c_included": round(float(c[order[-1]]), 4),
                "tie_break": "saturated heads ranked by -dNLL/dc_h (grad_at_final.pt)",
                "min_dropout_lambda_included": round(float(-g[order[-1]]), 5)})
    (a.out_root / "pooled_sparse").mkdir(parents=True, exist_ok=True)
    # eval_ext.py reads <out_root>/<task>/means.pt: link every task's means from the fit root
    for mp in sorted(a.fit_root.glob("*/means.pt")):
        dst = a.out_root / mp.parent.name / "means.pt"
        dst.parent.mkdir(exist_ok=True)
        if not dst.exists():
            os.symlink(os.path.relpath(mp, dst.parent), dst)
    json.dump(out, open(a.out_root / "pooled_sparse" / "selection.json", "w"), indent=1)
    print(f"top-{a.k}: min c included {out['min_c_included']}, parent had {sel['n_selected']} heads > .8 -> {a.out_root}")


if __name__ == "__main__":
    main()
