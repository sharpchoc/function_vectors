#!/usr/bin/env python
"""SANDBOX: per-prompt FVs under the vanilla_sparse_opt23 head definition (23 heads, c > 0.8).

For every prompt in the sandbox per-prompt capture
(artifacts/sandbox/perprompt_head_acts/gptj_train_varicl_top40/, all 29 tasks x {train,test}
query splits, 170 prompts/task, fixed 10-shot), computes

    v23^j = sum_{(l,h) in H23} W_out^l[:, h*256:(h+1)*256] @ head_activations[j, l, h]

i.e. the per-prompt function vector v^j_A restricted to the 23-head sparse-optimization set
(identical head list to artifacts/function_vectors/gpt-j/sandbox/vanilla_sparse_opt23).

out_proj slices are mmap-read straight from the cached GPT-J pytorch_model.bin (full model
load OOMs under the CPU-pod 16 GB cgroup). HARD GATES before writing anything:
  1. extracted slices == stored top40_outproj_slices.pt on the overlapping heads;
  2. rebuilding the stored top-40 per-prompt targets from head_activations + slices matches
     shard 'targets' (rel L2 < 1e-3).
Output: artifacts/sandbox/sparse_head_selection/perprompt_fv_sparse23/<task>.pt
  {'train'|'test': {'fvs': (N,4096) fp32, 'query_indices': [...]}, 'heads', 'config'}
"""
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.utils.paths import ARTIFACTS_ROOT

CAPTURE_ROOT = ARTIFACTS_ROOT / "sandbox" / "perprompt_head_acts" / "gptj_train_varicl_top40"
COEFFS_PATH = ARTIFACTS_ROOT / "sandbox" / "sparse_head_selection" / "coeffs_final.pt"
OUT_ROOT = ARTIFACTS_ROOT / "sandbox" / "sparse_head_selection" / "perprompt_fv_sparse23"
CKPT = sorted(Path("/workspace/.cache/huggingface/hub/models--EleutherAI--gpt-j-6b/snapshots").glob("*/pytorch_model.bin"))[-1]
HEAD_DIM, N_HEADS = 256, 16


def torch_load(p, **kw):
    return torch.load(p, map_location="cpu", weights_only=False, **kw)


def main():
    c = torch_load(COEFFS_PATH)["c"]
    heads23 = sorted((i // N_HEADS, i % N_HEADS) for i in range(c.numel()) if c[i].item() > 0.8)
    assert len(heads23) == 23, f"expected 23 heads, got {len(heads23)}"
    manifest = json.load(open(ARTIFACTS_ROOT / "function_vectors" / "gpt-j" / "sandbox" / "vanilla_sparse_opt23" / "fv_manifest.json"))
    theirs = sorted((int(l), int(h)) for l, h, *_ in manifest["top_heads"])
    assert theirs == heads23, "c>0.8 set does not match vanilla_sparse_opt23 manifest - HARD STOP"

    # --- out_proj slices straight from the checkpoint (mmap; no full model load) ---
    # The cached snapshot stores fp32 weights; the repo-standard loader (and the capture's
    # stored slices/targets) use fp16, so cast to fp16 first to match that convention exactly.
    sd = torch_load(CKPT, mmap=True)
    slices = {}
    for l, h in heads23:
        w = sd[f"transformer.h.{l}.attn.out_proj.weight"]  # (4096, 4096), row-major
        slices[(l, h)] = w[:, h * HEAD_DIM:(h + 1) * HEAD_DIM].clone().half().float()

    # GATE 1: match stored top-40 slices on overlapping heads (keys are "L{l}H{h}" strings).
    stored = torch_load(CAPTURE_ROOT / "top40_outproj_slices.pt")
    assert stored.get("out_proj_bias") is False, "capture assumed bias-free out_proj"

    def parse_key(k):
        l, h = k[1:].split("H")
        return int(l), int(h)

    stored_slices = {parse_key(k): v for k, v in stored["slices"].items()}
    n_checked = 0
    for lk, val in stored_slices.items():
        if lk in slices:
            rel = (val.float() - slices[lk]).norm() / slices[lk].norm()
            assert rel < 1e-6, f"GATE 1 FAILED at head {lk}: rel={rel:.3e} - HARD STOP, inform user"
            n_checked += 1
    print(f"gate 1 OK: {n_checked} overlapping head slices match stored top-40 slices")

    # GATE 2: rebuild stored top-40 targets for two tasks from head_activations.
    top40 = list(stored_slices.keys())
    slices40 = {}
    for l, h in top40:
        w = sd[f"transformer.h.{l}.attn.out_proj.weight"]
        slices40[(l, h)] = w[:, h * HEAD_DIM:(h + 1) * HEAD_DIM].clone().float()
    for task in ("antonym", "national_parks"):
        sh = torch_load(CAPTURE_ROOT / task / "train" / "shard_00000.pt")
        acts = sh["head_activations"].float()
        rebuilt = torch.zeros(acts.shape[0], 4096)
        for l, h in top40:
            rebuilt += acts[:, l, h] @ slices40[(l, h)].t()
        tgt = sh["targets"].float()
        rel = (rebuilt - tgt).norm() / tgt.norm()
        assert rel < 1e-3, f"GATE 2 FAILED on {task}: rel={rel:.3e} - HARD STOP, inform user"
        print(f"gate 2 OK ({task}): rebuilt targets match stored, rel={rel:.2e}")

    # --- build v23 for every task/split ---
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = sorted(d.name for d in CAPTURE_ROOT.iterdir() if d.is_dir())
    summary = []
    for task in tasks:
        out = {"heads": [list(x) for x in heads23],
               "config": {"sandbox": True, "definition": "vanilla_sparse_opt23 per-prompt FV",
                          "source_capture": str(CAPTURE_ROOT), "n_shots": 10, "dtype": "fp32"}}
        row = {"task": task}
        for split in ("train", "test"):
            idx = json.load(open(CAPTURE_ROOT / task / split / "index.json"))
            fvs, qidx = [], idx["config"]["query_indices"]
            for shard_info in sorted((CAPTURE_ROOT / task / split).glob("shard_*.pt")):
                sh = torch_load(shard_info)
                acts = sh["head_activations"].float()
                v = torch.zeros(acts.shape[0], 4096)
                for l, h in heads23:
                    v += acts[:, l, h] @ slices[(l, h)].t()
                fvs.append(v)
            fvs = torch.cat(fvs)
            assert fvs.shape[0] == len(qidx), f"{task}/{split}: {fvs.shape[0]} fvs vs {len(qidx)} indices"
            out[split] = {"fvs": fvs, "query_indices": qidx}
            row[split] = fvs.shape[0]
            row[f"{split}_mean_norm"] = round(float(fvs.norm(dim=1).mean()), 2)
        # sanity (advisory): cos(mean per-prompt v23, stored vanilla_sparse_opt23 FV)
        fv_path = ARTIFACTS_ROOT / "function_vectors" / "gpt-j" / "sandbox" / "vanilla_sparse_opt23" / task / f"{task}_function_vector.pt"
        if fv_path.exists():
            stored_fv = torch_load(fv_path)
            if isinstance(stored_fv, dict):
                stored_fv = stored_fv["function_vector"]
            stored_fv = torch.as_tensor(stored_fv).float().flatten()
            mean_v = torch.cat([out["train"]["fvs"], out["test"]["fvs"]]).mean(0)
            row["cos_mean_vs_stored23fv"] = round(float(torch.nn.functional.cosine_similarity(mean_v, stored_fv, dim=0)), 4)
        torch.save(out, OUT_ROOT / f"{task}.pt")
        summary.append(row)
        print(f"{task}: train={row['train']} test={row['test']} "
              f"cos_mean_vs_stored23fv={row.get('cos_mean_vs_stored23fv', 'n/a')}")
    with open(OUT_ROOT / "build_summary.json", "w") as f:
        json.dump({"sandbox": True, "heads": [list(x) for x in heads23], "tasks": summary}, f, indent=2)
    print(f"\nwrote {len(tasks)} task files to {OUT_ROOT}")


if __name__ == "__main__":
    main()
