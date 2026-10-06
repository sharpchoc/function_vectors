#!/usr/bin/env python
"""Push the identification feature through the FV heads' value-output circuits (USER REQUEST
2026-10-06) and compare with the task FV.

For each task A and input x in {s_A = c + u_A, u_A} (bank A, read band):
    y(x) = sum_{h in FV heads} W_O^h W_V^{kv(h)} Norm_l(h)(x) [+ b_V^{kv(h)} on Qwen]
i.e. what the FV heads would WRITE if they attended entirely to a token carrying x (attention
pattern fixed, one band vector fed to every head regardless of its layer; the head's own input
norm with learned weights applied first). Variants: no skip (y) and skip (y + x).
Readout: cos(y, v_A) with v_A = sum of the selected heads' mean cue outputs (the task FV).
References: input-only cos(x, v_A); cross-task control mean_{B != A} cos(y_A, v_B).
Weights are read lazily from the checkpoint safetensors (CPU, no model load).
Outputs: <out_dir>/{per_task.csv, summary.csv}
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import (ARTIFACTS_ROOT, REPO_ROOT, TASK69_RUN_DIR, QWEN25_FV_DIR,   # noqa: E402
                             QWEN25_FV_ARTIFACTS_DIR, QWEN25_SELECTION_ROOT, QWEN25_SPLIT)

HF = Path("/workspace/.cache/huggingface/hub")
MODELS = {
    "gptj": dict(
        split=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json",
        means_root=ARTIFACTS_ROOT / "sandbox" / "ext_steerability",
        selection=ARTIFACTS_ROOT / "sandbox" / "ext_steerability" / "prunedfail_seed43" / "pooled_sparse" / "selection.json",
        bankA=ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "bankA",
        out=TASK69_RUN_DIR / "read_write_relationship" / "ov_circuit_map",
        ckpt=sorted(glob.glob(str(HF / "models--EleutherAI--gpt-j-6b/snapshots/*/model.safetensors"))),
        n_heads=16, n_kv=16, eps=1e-5, norm="ln", label="GPT-J-6B"),
    "qwen": dict(
        split=QWEN25_SPLIT, means_root=QWEN25_SELECTION_ROOT,
        selection=QWEN25_SELECTION_ROOT / "pooled_sparse" / "selection.json",
        bankA=QWEN25_FV_ARTIFACTS_DIR / "bottom_up_ablation" / "bankA",
        out=QWEN25_FV_DIR / "read_write_relationship" / "ov_circuit_map",
        ckpt=sorted(glob.glob(str(HF / "models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/model-*.safetensors"))),
        n_heads=28, n_kv=4, eps=1e-6, norm="rms", label="Qwen2.5-7B-Instruct"),
}


class Weights:
    """Lazy per-tensor reads across one or more safetensors files."""
    def __init__(self, files):
        self.files = {}
        for f in files:
            with safe_open(f, "pt") as h:
                for k in h.keys():
                    self.files[k] = f

    def get(self, key):
        with safe_open(self.files[key], "pt") as h:
            return h.get_tensor(key).float()


def layer_weights(W, model, l):
    if model == "gptj":
        p = f"transformer.h.{l}."
        return dict(nw=W.get(p + "ln_1.weight"), nb=W.get(p + "ln_1.bias"),
                    wv=W.get(p + "attn.v_proj.weight"), bv=None, wo=W.get(p + "attn.out_proj.weight"))
    p = f"model.layers.{l}."
    return dict(nw=W.get(p + "input_layernorm.weight"), nb=None,
                wv=W.get(p + "self_attn.v_proj.weight"), bv=W.get(p + "self_attn.v_proj.bias"),
                wo=W.get(p + "self_attn.o_proj.weight"))


def norm(x, lw, cfg):
    if cfg["norm"] == "ln":
        mu, var = x.mean(), x.var(unbiased=False)
        return (x - mu) / torch.sqrt(var + cfg["eps"]) * lw["nw"] + lw["nb"]
    return x / torch.sqrt((x * x).mean() + cfg["eps"]) * lw["nw"]


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--no_vbias", action="store_true", help="drop the value bias (Qwen) from the map")
    a = ap.parse_args()
    cfg = MODELS[a.model]
    split = json.load(open(cfg["split"]))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    sel = sorted(int(f) for f in json.load(open(cfg["selection"]))["selected_flat"])
    H, nkv = cfg["n_heads"], cfg["n_kv"]
    W = Weights(cfg["ckpt"])
    by_layer = {}
    for f in sel:
        by_layer.setdefault(f // H, []).append(f % H)
    LW = {l: layer_weights(W, a.model, l) for l in by_layer}
    d_model = LW[next(iter(LW))]["wo"].shape[0]
    hd = d_model // H
    per_kv = H // nkv

    # task FVs: sum over selected heads of W_O^h @ head_mean[l, h]
    fv = {}
    for t in tasks:
        hm = torch.load(cfg["means_root"] / t / "means.pt", map_location="cpu", weights_only=False)["head_means"].float()
        v = torch.zeros(d_model)
        for l, hs in by_layer.items():
            for h in hs:
                v += LW[l]["wo"][:, h * hd:(h + 1) * hd] @ hm[l, h]
        fv[t] = v
    # identification features (bank A): s_A = c + u_A ; u_A = ||u_A|| * u_hat
    cpm = torch.load(cfg["bankA"] / "carrier_plus_meanresid_vectors.pt", map_location="cpu", weights_only=False)
    swap = torch.load(cfg["bankA"] / "meanresid_swap_bases.pt", map_location="cpu", weights_only=False)
    feats = {}
    for t in tasks:
        s = cpm["tasks"][t]["vec"].float()
        u = swap["tasks"][t]["V"].float().flatten() * float(swap["tasks"][t]["s"][0])
        feats[t] = {"s_A": s, "u_A": u}
    c_norm = float(cpm["carrier"].norm())

    def through_heads(x):
        y = torch.zeros(d_model)
        for l, hs in by_layer.items():
            lw = LW[l]
            xn = norm(x, lw, cfg)
            for h in hs:
                g = h // per_kv
                v = lw["wv"][g * hd:(g + 1) * hd] @ xn
                if lw["bv"] is not None and not a.no_vbias:
                    v = v + lw["bv"][g * hd:(g + 1) * hd]
                y += lw["wo"][:, h * hd:(h + 1) * hd] @ v
        return y

    rows = []
    mapped = {}
    for t in tasks:
        r = {"task": t, "group": group[t], "norm_fv": round(float(fv[t].norm()), 2)}
        for name, x in feats[t].items():
            y = through_heads(x)
            mapped[(t, name)] = (y, y + x)
            r[f"{name}_norm"] = round(float(x.norm()), 2)
            r[f"{name}_mapped_norm"] = round(float(y.norm()), 2)
            r[f"{name}_input_only"] = round(cos(x, fv[t]), 4)
            r[f"{name}_noskip"] = round(cos(y, fv[t]), 4)
            r[f"{name}_skip"] = round(cos(y + x, fv[t]), 4)
        rows.append(r)
    for r in rows:   # cross-task controls
        t = r["task"]
        for name in ("s_A", "u_A"):
            y, ys = mapped[(t, name)]
            r[f"{name}_noskip_cross"] = round(float(np.mean([cos(y, fv[b]) for b in tasks if b != t])), 4)
            r[f"{name}_skip_cross"] = round(float(np.mean([cos(ys, fv[b]) for b in tasks if b != t])), 4)

    out = cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(out / "per_task.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    summ = [["metric", "mean_all", "median_all", "mean_train", "mean_heldout"]]
    for k in keys[3:]:
        vals = np.array([r[k] for r in rows]); g = np.array([r["group"] for r in rows])
        summ.append([k, round(float(vals.mean()), 4), round(float(np.median(vals)), 4),
                     round(float(vals[g == "train"].mean()), 4), round(float(vals[g == "heldout"].mean()), 4)])
    with open(out / "summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(summ)
    print(f"{cfg['label']}: {len(tasks)} tasks, {len(sel)} FV heads over {len(by_layer)} layers, "
          f"carrier norm {c_norm:.1f}, vbias={'off' if a.no_vbias else 'on'}")
    for s in summ[1:]:
        print(f"  {s[0]:<22} mean {s[1]:+.3f}  median {s[2]:+.3f}  train {s[3]:+.3f}  heldout {s[4]:+.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
