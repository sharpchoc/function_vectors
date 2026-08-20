#!/usr/bin/env python
"""FV presence ABOVE the generic-FV baseline at the query cue (GPU recapture).

Mirrors part (a) of capture_69_presence_vs_acc.py exactly (same prompts, truncation,
right padding, cue = last real token, layers 9..20) but records TWO cosines per prompt
and layer, plus the raw L13 cue activation so future variants need no recapture:

  cos_own = cos(z_l, v_hat_A)    v_hat_A  = unit mean of the task's 150 per-prompt FVs
  cos_gm  = cos(z_l, v_hat_gm)   v_hat_gm = unit EQUAL-TASK-WEIGHTED mean over all 69
                                            task FVs v_A (generic FV)
  cue_L13 = z_13 at the query cue, float16

The plotted quantity downstream is  delta_cos = cos_own - cos_gm  (presence above the
generic-FV baseline). Accuracies are NOT recomputed — they come from the existing
presence_vs_acc/<task>.npz (`match`).

Sanity check: after each task, cos_own is compared to presence_vs_acc/<task>.npz["cos"];
the max abs diff must be fp16-forward-noise small (< 1e-2) or the script aborts.

Writes ARTIFACTS_ROOT/69_task_run/presence_vs_acc_gm/<task>.npz with
  cos_own (7,150,12) f32, cos_gm (7,150,12) f32, cue_L13 (7,150,4096) f16,
  layers, n_shots, group.  Existing outputs are skipped (resumable). Fan out with --tasks.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from baukit import TraceDict  # noqa: E402
from src.sandbox.isolation_upper_bound.run_task import (  # noqa: E402
    auto_batch, load_records, record_to_prompt_data)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed  # noqa: E402
from src.utils.paths import ARTIFACTS_ROOT  # noqa: E402
from src.utils.prompt_utils import create_prompt  # noqa: E402

N_SHOTS = list(range(0, 7))
LAYERS = list(range(9, 21))
L13_IDX = LAYERS.index(13)
SANITY_TOL = 1e-2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs")
    p.add_argument("--ref_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "presence_vs_acc",
                   help="original capture (cos to own FV) used for the sanity check")
    p.add_argument("--out_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "presence_vs_acc_gm")
    p.add_argument("--split", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--capture_batch", type=int, default=16)
    return p.parse_args()


def task_fv(fv_root: Path, task: str) -> torch.Tensor:
    """v_A = mean over the 150 per-prompt FVs, double precision (as in the original capture)."""
    pp = torch.load(fv_root / f"{task}.pt", map_location="cpu", weights_only=False)
    fv = pp["fv"]
    assert fv.shape == (150, 4096), (task, fv.shape)
    return fv.double().mean(dim=0)


def generic_fv(fv_root: Path, all_tasks) -> torch.Tensor:
    """v_gm = equal-task-weighted mean over all 69 task FVs v_A (double precision)."""
    assert len(all_tasks) == 69, len(all_tasks)
    return torch.stack([task_fv(fv_root, t) for t in all_tasks]).mean(dim=0)


def main():
    args = parse_args()
    set_seed(args.seed)
    split = json.load(open(args.split))
    group_of = {t: "train" for t in split["train_tasks"]}
    group_of.update({t: "heldout" for t in split["heldout_tasks"]})
    all_tasks = sorted(group_of)

    v_gm = generic_fv(args.fv_root, all_tasks)
    v_hat_gm = v_gm / v_gm.norm()
    print(f"generic FV over {len(all_tasks)} tasks: ||v_gm|| = {float(v_gm.norm()):.3f}", flush=True)

    model, tokenizer, cfg = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()
    torch.set_grad_enabled(False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hooks = [cfg["layer_hook_names"][l] for l in LAYERS]
    args.out_root.mkdir(parents=True, exist_ok=True)
    v_hat_gm = v_hat_gm.float().to(model.device)

    for task in args.tasks:
        out = args.out_root / f"{task}.npz"
        if out.exists():
            print(f"[{task}] exists, skip", flush=True)
            continue

        v = task_fv(args.fv_root, task)
        v_hat = (v / v.norm()).float().to(model.device)
        print(f"[{task}] cos(v_hat_A, v_hat_gm) = {float(v_hat @ v_hat_gm):.3f}", flush=True)

        recs = load_records(args, task, "train_prompts")
        assert len(recs) == 150
        cos_own = np.zeros((len(N_SHOTS), 150, len(LAYERS)), dtype=np.float32)
        cos_gm = np.zeros((len(N_SHOTS), 150, len(LAYERS)), dtype=np.float32)
        cue_l13 = np.zeros((len(N_SHOTS), 150, 4096), dtype=np.float16)

        for ni, n in enumerate(N_SHOTS):
            sents = []
            for r in recs:
                r_n = dict(r)
                r_n["demos"] = r["demos"][:n]
                sent = create_prompt(record_to_prompt_data(r_n, cfg))
                assert sent.rstrip().endswith("A:")
                sents.append(sent)
            tok_lens = [len(tokenizer(s).input_ids) for s in sents]
            order = sorted(range(150), key=lambda i: tok_lens[i])

            # right-padded forward, cue = last real token (identical to the original part (a))
            tokenizer.padding_side = "right"
            bsz = auto_batch(max(tok_lens), 6000, 64)
            for start in range(0, 150, bsz):
                idx = order[start:start + bsz]
                enc = tokenizer([sents[i] for i in idx], return_tensors="pt", padding=True)
                enc = {k: v_.to(model.device) for k, v_ in enc.items()}
                lens = enc["attention_mask"].sum(dim=1)
                with TraceDict(model, layers=hooks, retain_output=True) as td:
                    model(**enc)
                for li, name in enumerate(hooks):
                    h = td[name].output
                    if isinstance(h, tuple):
                        h = h[0]
                    h = h.float()
                    cue = h[torch.arange(len(idx)), lens - 1]          # (B, 4096)
                    nrm = cue.norm(dim=-1).clamp_min(1e-8)
                    c_own = ((cue @ v_hat) / nrm).cpu().numpy()
                    c_gm = ((cue @ v_hat_gm) / nrm).cpu().numpy()
                    cue_np = cue.half().cpu().numpy() if li == L13_IDX else None
                    for bi, i in enumerate(idx):
                        cos_own[ni, i, li] = c_own[bi]
                        cos_gm[ni, i, li] = c_gm[bi]
                        if cue_np is not None:
                            cue_l13[ni, i] = cue_np[bi]
            print(f"[{task}] n={n} cos_own@L13={cos_own[ni, :, L13_IDX].mean():.3f} "
                  f"cos_gm@L13={cos_gm[ni, :, L13_IDX].mean():.3f} "
                  f"delta={(cos_own[ni, :, L13_IDX] - cos_gm[ni, :, L13_IDX]).mean():.3f}", flush=True)

        # --- sanity check against the original capture ---
        ref = np.load(args.ref_root / f"{task}.npz", allow_pickle=False)
        assert list(ref["layers"]) == LAYERS and list(ref["n_shots"]) == N_SHOTS
        max_diff = float(np.abs(ref["cos"] - cos_own).max())
        print(f"[{task}] SANITY max|cos_own - original cos| = {max_diff:.2e}", flush=True)
        if not (max_diff < SANITY_TOL):
            raise RuntimeError(f"[{task}] sanity check FAILED: max abs diff {max_diff:.3e} "
                               f">= {SANITY_TOL}; stop and investigate")

        np.savez(out, cos_own=cos_own, cos_gm=cos_gm, cue_L13=cue_l13,
                 layers=np.array(LAYERS), n_shots=np.array(N_SHOTS), group=group_of[task],
                 sanity_max_abs_diff=np.float32(max_diff))
        print(f"[{task}] done", flush=True)


if __name__ == "__main__":
    main()
