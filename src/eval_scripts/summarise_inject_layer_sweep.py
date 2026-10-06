#!/usr/bin/env python
"""Summarise the inject-layer sweep of the sparse head selection (2026-10-06).

For each model and candidate injection layer L (fits at fixed lambda under
artifacts/sandbox/ext_steerability_layersweep/<model>/L<L>/, plus the derived top-50 sets in
L<L>/top50/): aggregate the steering eval (aggregate_eval_headset.py schema) and report
  n_heads | held-out zero-shot steering (zs_base -> zs_best) | train zs_best | mixed/shuffled best |
  eval-preferred injection layer (median zs_bestL) | cross-task FV cosine (mean over all task pairs
  of the FV = sum of the selected heads' W_O-projected head means; lower = more task-specific).
Selection rule (fixed in advance): best held-out zs_best; candidates within .02 tie; among ties prefer
lower cross-task cosine, then fewer heads. Writes <out_dir>/{sweep_summary.csv, per_task_<model>.csv}.
"""
import argparse
import csv
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT, TASK69_RUN_DIR, QWEN25_FV_DIR, QWEN25_SPLIT  # noqa: E402

HF = Path("/workspace/.cache/huggingface/hub")
SWEEP = ARTIFACTS_ROOT / "sandbox" / "ext_steerability_layersweep"
MODELS = {
    "gptj": dict(split=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json", H=16,
                 ckpt=sorted(glob.glob(str(HF / "models--EleutherAI--gpt-j-6b/snapshots/*/model.safetensors"))),
                 wo=lambda l: f"transformer.h.{l}.attn.out_proj.weight", out=TASK69_RUN_DIR / "FV_train_test_generalisation" / "inject_layer_sweep"),
    "qwen": dict(split=QWEN25_SPLIT, H=28,
                 ckpt=sorted(glob.glob(str(HF / "models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/model-*.safetensors"))),
                 wo=lambda l: f"model.layers.{l}.self_attn.o_proj.weight", out=QWEN25_FV_DIR / "round2_96_prunedfail" / "inject_layer_sweep"),
}
LAYERS = (9, 12, 16, 20, 24)


class WO:
    def __init__(self, files, key):
        self.key = key; self.where = {}; self.cache = {}
        for f in files:
            with safe_open(f, "pt") as h:
                for k in h.keys():
                    self.where[k] = f
    def __call__(self, l):
        if l not in self.cache:
            with safe_open(self.where[self.key(l)], "pt") as h:
                self.cache[l] = h.get_tensor(self.key(l)).float()
        return self.cache[l]


def cross_task_cos(sel_flat, tasks, root, wo, H):
    by_layer = {}
    for f in sel_flat:
        by_layer.setdefault(f // H, []).append(f % H)
    fvs = []
    for t in tasks:
        hm = torch.load(root / t / "means.pt", map_location="cpu", weights_only=False)["head_means"].float()
        v = torch.zeros(wo(next(iter(by_layer))).shape[0])
        for l, hs in by_layer.items():
            W = wo(l); hd = W.shape[0] // H
            for h in hs:
                v += W[:, h * hd:(h + 1) * hd] @ hm[l, h]
        fvs.append(v / v.norm())
    G = torch.stack(fvs); G = G @ G.T
    n = len(tasks); iu = torch.triu_indices(n, n, 1)
    return float(G[iu[0], iu[1]].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--layers", nargs="+", type=int, default=list(LAYERS))
    a = ap.parse_args()
    for m in a.models:
        cfg = MODELS[m]; split = json.load(open(cfg["split"]))
        tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
        group = {t: "train" for t in split["train_tasks"]}; group.update({t: "heldout" for t in split["heldout_tasks"]})
        wo = WO(cfg["ckpt"], cfg["wo"])
        out = cfg["out"]; out.mkdir(parents=True, exist_ok=True)
        rows = [["model", "inject_layer", "head_set", "n_heads", "min_c", "heldout_zs_base", "heldout_zs_best", "train_zs_best",
                 "heldout_mix_best", "heldout_shuf_best", "eval_bestL_median", "cross_task_fv_cos", "n_tasks_evaluated"]]
        per_task = []
        for L in a.layers:
            for hs, sub in (("full", ""), ("top50", "top50")):
                root = SWEEP / m / f"L{L}" / sub if sub else SWEEP / m / f"L{L}"
                sp = root / "pooled_sparse" / "selection.json"
                if not sp.exists():
                    continue
                sel = json.load(open(sp))
                done = [t for t in tasks if (root / t / "eval_headset.json").exists()]
                if len(done) < len(tasks):
                    print(f"{m} L{L} {hs}: {len(done)}/{len(tasks)} evaluated — skipped (incomplete)")
                    continue
                csvp = root / "train_heldout_summary.csv"
                subprocess.run([sys.executable, str(REPO_ROOT / "src/sandbox/ext_steerability/aggregate_eval_headset.py"),
                                "--split_path", str(cfg["split"]), "--eval_root", str(root), "--out_csv", str(csvp)],
                               check=True, capture_output=True)
                rs = list(csv.DictReader(open(csvp)))
                for r in rs:
                    per_task.append({"inject_layer": L, "head_set": hs, **r})
                def mean(col, g=None):
                    v = [float(r[col]) for r in rs if g is None or r["group"] == g]; return round(float(np.mean(v)), 4)
                bestL = int(np.median([int(r["zs_bestL"]) for r in rs]))
                means_root = SWEEP / m / f"L{L}"
                ctc = cross_task_cos(sel["selected_flat"], tasks, means_root, wo, cfg["H"])
                minc = min(h[2] for h in sel["selected_heads"]) if sel["selected_heads"] else None
                rows.append([m, L, hs, sel["n_selected"], minc, mean("zs_base", "heldout"), mean("zs_best", "heldout"),
                             mean("zs_best", "train"), mean("mix_best", "heldout"), mean("shuf_best", "heldout"), bestL,
                             round(ctc, 4), len(rs)])
                print("  ".join(str(x) for x in rows[-1]), flush=True)
        with open(out / "sweep_summary.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)
        if per_task:
            with open(out / "per_task.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(per_task[0].keys())); w.writeheader(); w.writerows(per_task)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
