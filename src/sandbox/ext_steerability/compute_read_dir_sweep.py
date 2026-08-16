#!/usr/bin/env python
"""Per-prompt read-direction definition SWEEP for the 69-task pool (37-head set).

Materializes the write_up/read_direction_levers.md crossing under the user's 2026-08-16
decisions, one bracket per (Lever 1 metric) x (Lever 3 aggregation):

  cosine_M        cosine + energy-90-truncated M^+          + summed circuit   [repackaged
                  from artifacts/69_task_run/perprompt_read_dirs (rank90 variant)]
  dot_M           dot product + M^T                          + summed circuit   [repackaged
                  from artifacts/69_task_run/perprompt_dot_read_dirs]
  cosine_perhead  cosine + per-head energy-90-truncated (W_O^h W_V^h)^+ against h^j_A,
                  summed UNNORMALIZED across heads (sub-choice 3'), normalized at the end
  dot_perhead     dot product + per-head (W_O^h W_V^h)^T against h^j_A, summed
                  UNNORMALIZED across heads (3'), normalized at the end

Fixed decisions: Lever 2 = truncated pseudo-inverse at cumulative sigma^2 >= 0.90 of each
circuit's own spectrum (never literal); Lever 4 = BOTH variants stored in one file
(r = unit rows, norm = pre-normalization magnitude, so natural = r * norm[:, None]).
Targets are per-prompt (150/task); a task-level r_task (+ norm) from the mean target is
included for convenience.

Per-head efficiency: W = W_O W_V has rank <= 256, so SVD runs on the 256x4096 factor
(QR of W_O, then SVD of R W_V); the per-prompt solve reduces to a fixed (4096, 256)
matrix per head applied to the stored 256-dim out_proj inputs:
  cosine: r_h = Vh[:k]^T diag(1/S[:k]) (U'^T R a)[:k]     (U = Q U')
  dot   : r_h = W_V^T (R^T R) a                            (W^T W_O a = W_V^T R^T R a)

Gates (hard stop -> user adjudicates): per-head SVD reconstruction rel <= 1e-10; QR
factorization reconstruction rel <= 1e-12; gate (a) stored fv vs sum_h W_O^h raw_h.

Outputs: artifacts/69_task_run/read_dir_sweep/<bracket>/<task>.pt (uniform schema) and
sweep_manifest.json (bracket definitions, per-head k cuts, gate stats).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.utils.paths import ARTIFACTS_ROOT

HEAD_DIM, N_HEADS = 256, 16
BRACKETS = {
    "cosine_M": "cosine + energy90-truncated M^+ + summed circuit (Lever 3.1); repackaged rank90",
    "dot_M": "dot product + M^T + summed circuit (Lever 3.1); repackaged",
    "cosine_perhead": "cosine + per-head energy90-truncated pinv vs h^j_A + unnormalized sum (3') + end-normalize",
    "dot_perhead": "dot product + per-head transpose vs h^j_A + unnormalized sum (3') + end-normalize",
}
COMMON_CONFIG = {
    "lever2": "truncated pseudo-inverse, cum sigma^2 >= 0.90 (user decision 2026-08-16; literal excluded)",
    "lever4": "both stored: unit rows in 'r', pre-normalization magnitude in 'norm' (natural = r * norm)",
    "targets": "per-prompt (150/task); r_task from the mean target included",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fv_dir", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs")
    p.add_argument("--cosM_dir", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "perprompt_read_dirs")
    p.add_argument("--dotM_dir", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "perprompt_dot_read_dirs")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "read_dir_sweep")
    p.add_argument("--selection_path", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox" / "ext_steerability" / "prunedfail_seed43"
                   / "pooled_sparse" / "selection.json")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    return p.parse_args()


def torch_load(p, **kw):
    return torch.load(p, map_location="cpu", weights_only=False, **kw)


def energy_k(s):
    e = np.cumsum(s.numpy() ** 2) / np.sum(s.numpy() ** 2)
    return int(np.searchsorted(e, 0.90) + 1)


def save_bracket(out_root, bracket, task, group, prompt_index, r_unnorm, rt_unnorm, extra_cfg):
    norms = r_unnorm.norm(dim=1)
    rt_norm = rt_unnorm.norm()
    out = {"task": task, "group": group, "bracket": bracket,
           "config": {**COMMON_CONFIG, **extra_cfg, "bracket_definition": BRACKETS[bracket]},
           "prompt_index": prompt_index,
           "r": (r_unnorm / norms[:, None]).float().cpu(),
           "norm": norms.float().cpu(),
           "r_task": (rt_unnorm / rt_norm).float().cpu().squeeze(0),
           "r_task_norm": float(rt_norm)}
    d = out_root / bracket
    d.mkdir(parents=True, exist_ok=True)
    torch.save(out, d / f"{task}.pt")
    return float(norms.median())


def main():
    args = parse_args()
    ckpt = args.ckpt or sorted(Path("/workspace/.cache/huggingface/hub/"
                                    "models--EleutherAI--gpt-j-6b/snapshots").glob("*/pytorch_model.bin"))[-1]
    sel_flat = sorted(json.load(open(args.selection_path))["selected_flat"])
    heads = [(f // N_HEADS, f % N_HEADS) for f in sel_flat]
    assert len(heads) == 37
    split = json.load(open(args.split_path))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})

    # --- per-head factors and solve matrices (all fp64, CPU-cheap) ---
    sd = torch_load(ckpt, mmap=True)
    WO_l, C_cos_l, C_dot_l, k_per_head = [], [], [], []
    for l, h in heads:
        wo = sd[f"transformer.h.{l}.attn.out_proj.weight"][:, h * HEAD_DIM:(h + 1) * HEAD_DIM].clone().half().double()
        wv = sd[f"transformer.h.{l}.attn.v_proj.weight"][h * HEAD_DIM:(h + 1) * HEAD_DIM, :].clone().half().double()
        Q, R = torch.linalg.qr(wo)                    # (4096,256), (256,256)
        rec_qr = (Q @ R - wo).norm() / wo.norm()
        assert rec_qr < 1e-12, f"QR GATE FAILED L{l}H{h}: rel={rec_qr:.3e} - HARD STOP"
        B = R @ wv                                    # (256, 4096); W = Q B
        U2, S, Vh = torch.linalg.svd(B, full_matrices=False)
        rec = (U2 @ torch.diag(S) @ Vh - B).norm() / B.norm()
        assert rec < 1e-10, f"SVD GATE FAILED L{l}H{h}: rel={rec:.3e} - HARD STOP"
        k = energy_k(S)
        k_per_head.append({"layer": l, "head": h, "k_energy90": k,
                           "sigma_max": float(S[0]), "sigma_min": float(S[-1])})
        UtR = U2.T @ R                                # (256,256): a -> U^T (W_O a)
        C_cos_l.append(Vh[:k].T @ (UtR[:k] / S[:k, None]))   # (4096,256)
        C_dot_l.append(wv.T @ (R.T @ R))                     # (4096,256)
        WO_l.append(wo)
    C_cos = torch.stack(C_cos_l)   # (37,4096,256)
    C_dot = torch.stack(C_dot_l)
    WO = torch.stack(WO_l)
    print(f"per-head factors built; k_energy90 range "
          f"{min(x['k_energy90'] for x in k_per_head)}-{max(x['k_energy90'] for x in k_per_head)}",
          flush=True)

    tasks_all = sorted(pth.stem for pth in args.fv_dir.glob("*.pt"))
    assert len(tasks_all) == 69 and set(tasks_all) == set(group)
    tasks = tasks_all[args.shard_idx::args.shard_n]
    rows = []
    for task in tasks:
        d = torch_load(args.fv_dir / f"{task}.pt")
        assert sorted(int(x) for x in d["sel_flat"]) == sel_flat, f"{task}: sel_flat mismatch - HARD STOP"
        v = d["fv"].double()
        raw = d["raw"].double()          # (150, 37, 256)
        v_rebuilt = torch.einsum("hdk,nhk->nd", WO, raw)
        rel_a = ((v_rebuilt - v).norm(dim=1) / v.norm(dim=1)).max().item()
        assert rel_a < 2e-3, f"GATE(a) FAILED {task}: rel={rel_a:.3e} - HARD STOP"
        row = {"task": task, "group": group[task], "gate_a_rel": round(rel_a, 8)}
        raw_mean = raw.mean(0, keepdim=True)

        # per-head brackets (3': sum unnormalized per-head solutions, normalize at end)
        for bracket, C in (("cosine_perhead", C_cos), ("dot_perhead", C_dot)):
            r_unnorm = torch.einsum("hdk,nhk->nd", C, raw)
            rt_unnorm = torch.einsum("hdk,nhk->nd", C, raw_mean)
            row[f"{bracket}_med_norm"] = round(save_bracket(
                args.out_root, bracket, task, group[task], d["prompt_index"],
                r_unnorm, rt_unnorm,
                {"lever1": bracket.split("_")[0], "lever3": "per-head then sum (3.2)",
                 "subchoice_3prime": "sum unnormalized per-head solutions, normalize at end"}), 4)

        # summed-M brackets: repackage existing artifacts into the uniform schema
        cm = torch_load(args.cosM_dir / f"{task}.pt")
        r_unnorm = cm["rank90"]["r"].double() * cm["rank90"]["preinv_norm"].double()[:, None]
        # M^+_tau is linear, so M^+_tau(mean v) = mean of the unnormalized per-prompt solutions
        row["cosine_M_med_norm"] = round(save_bracket(
            args.out_root, "cosine_M", task, group[task], cm["prompt_index"],
            r_unnorm, r_unnorm.mean(0, keepdim=True),
            {"lever1": "cosine", "lever3": "summed circuit M (3.1)",
             "source": str(args.cosM_dir),
             "truncation_k_M": int(cm["config"]["variants"]["rank90"])}), 4)
        dm = torch_load(args.dotM_dir / f"{task}.pt")
        r_unnorm = dm["r"].double() * dm["prenorm_MTv"].double()[:, None]
        row["dot_M_med_norm"] = round(save_bracket(
            args.out_root, "dot_M", task, group[task], dm["prompt_index"],
            r_unnorm, r_unnorm.mean(0, keepdim=True),
            {"lever1": "dot", "lever3": "summed circuit M (3.1)",
             "source": str(args.dotM_dir)}), 4)
        rows.append(row)
        print(f"{task} [{group[task]}]: gate a={rel_a:.2e} | med norms "
              f"cosPH={row['cosine_perhead_med_norm']} dotPH={row['dot_perhead_med_norm']} "
              f"cosM={row['cosine_M_med_norm']} dotM={row['dot_M_med_norm']}", flush=True)

    manifest = {"brackets": BRACKETS, "common_config": COMMON_CONFIG,
                "selection_path": str(args.selection_path), "k_per_head": k_per_head,
                "shard": [args.shard_idx, args.shard_n], "tasks": rows}
    args.out_root.mkdir(parents=True, exist_ok=True)
    with open(args.out_root / f"sweep_manifest_shard{args.shard_idx}of{args.shard_n}.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {len(tasks)} tasks x {len(BRACKETS)} brackets to {args.out_root}")


if __name__ == "__main__":
    main()
