#!/usr/bin/env python
"""1-shot projection-ablation of the 4D attention_head_payload_subspace (GPT-J; baukit).

Causal test of the payload subspace built by build_payload_subspace.py: the top-4
uncentered-SVD directions of the stacked unit d_payload vectors (d_payload =
unit(W_V^T @ unit(task-mean head activation)), task's top-10 per-task varicl CIE heads).

Protocol mirrors Stream W (ablate_oneshot_preimage_logprob.py; prompt building and chunking
IMPORTED from it, per DECISIONS 2026-07-28): --task's 1-shot prompts (up to 130 train + 40
test queries, seed 42), 3 site rows (cue1 = demo 'A:', target1 = demo label, final_cue =
query 'A:'), start layers L = 0..27. For every block b >= L, at the site token only, the
residual output of transformer.h.b is edited in the arm's FIXED 4D subspace B (layer-
independent, orthonormal rows):

    zero op:  v -= (v @ B^T) @ B                          remove the subspace component
    mean op:  v += (t[site, b] - v @ B^T) @ B             clamp it to the target coords

t[site, b] = B-coordinates of the grand-mean residual activation at that (site, edit layer)
across all 20 varicl train tasks (equal task weighting; capture_train_task_site_means.py).

Arms: payload_{zero,mean} use --task's own subspace; payload_cf_{zero,mean} use --cf_task's
subspace on the SAME prompts (task-specificity control; cf task user-fixed, not cf_map).

Metric, npz schema, gates (decoded sites, batched-vs-unbatched clean log p, no-op hook at
L=28, finite deltas) and resumability are Stream W-identical, plus --debug_invariant: assert
inside the hook that edited rows satisfy |v @ B^T - target| ~ 0.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.eval_scripts.exploratory.ablate_oneshot_preimage_logprob import (
    N_EDIT_LAYERS,
    ROW_SITE,
    SITE_INDEX,
    build_cf_map,
    build_prompts,
    git_commit_hash,
    make_chunks,
)
from src.eval_scripts.regress_activation_to_fv_fulldim_ridge import (
    load_function_vector,
    write_json,
)
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR
from utils.prompt_utils import load_dataset

ROW_NAMES = ["cue1", "target1", "final_cue"]
OPS = ["zero", "mean"]
# payload_* arms edit the 4D payload subspace; fv_* arms project out the unit canonical task
# FV (train_varicl_top40) as a k=1 basis through the same hook (zero op only). All arms run
# all 3 site rows.
ALL_ARMS = ["payload_zero", "payload_mean", "payload_cf_zero", "payload_cf_mean",
            "fv_zero", "fv_cf_zero"]
ARM_ROW_NAMES = {a: ROW_NAMES for a in ALL_ARMS}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", type=str, default="present-past")
    p.add_argument("--cf_task", type=str, default="english-french",
                   help="Whose payload subspace the *_cf arms project out (user-fixed control).")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Multi-task mode: loop these tasks (one model load); each task's cf "
                        "subspace comes from build_cf_map over this same pool (seeded random "
                        "other task, Stream W convention). Overrides --task/--cf_task.")
    p.add_argument("--arms", nargs="+", default=ALL_ARMS, choices=ALL_ARMS)
    p.add_argument("--subspace_root", type=Path, default=ARTIFACTS_ROOT / "payload_subspaces")
    p.add_argument("--subspace_suffix", type=str, default="top10heads_k4")
    p.add_argument("--site_means_path", type=Path,
                   default=ARTIFACTS_ROOT / "payload_subspace_ablation" / "train_task_site_means.pt")
    p.add_argument("--fv_root", type=Path,
                   default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_varicl_top40",
                   help="Canonical FV root for the fv_* arms.")
    p.add_argument("--output_root", type=Path,
                   default=FV_FORMATION_DIR / "ablation/attention_head_mechanisms/train_tasks")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_split", type=float, default=0.3)
    p.add_argument("--n_shots", type=int, default=1)
    p.add_argument("--max_train_prompts", type=int, default=130)
    p.add_argument("--max_test_prompts", type=int, default=40)
    p.add_argument("--start_layers", nargs="+", type=int, default=list(range(N_EDIT_LAYERS)))
    p.add_argument("--batch_size", type=int, default=170)
    p.add_argument("--max_prompts", type=int, default=None, help="Smoke cap on prompts/task.")
    p.add_argument("--prefixes", type=json.loads,
                   default={"input": "Q:", "output": "A:", "instructions": ""})
    p.add_argument("--separators", type=json.loads,
                   default={"input": "\n", "output": "\n\n", "instructions": ""})
    p.add_argument("--mode", choices=["anchor", "propagated"], default="anchor",
                   help="anchor: edit the site token only, blocks b >= L (original study). "
                        "propagated: edit the site token AND every later position (incl. "
                        "newlines/'Q:'/query tokens) at all blocks b >= L; the subspace is "
                        "layer-independent so no anchor-layer fixing is needed. Zero op only "
                        "(no matched mean targets exist for arbitrary downstream positions). "
                        "Use a separate --output_root per mode.")
    p.add_argument("--debug_invariant", action="store_true",
                   help="Assert |B(v_edited) - target| < 1e-3 inside every hook call (smoke).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_subspace(subspace_root, task, suffix, device):
    """-> (B [4, 4096] fp32 orthonormal rows, metadata dict)."""
    path = subspace_root / f"{task}_{suffix}.pt"
    obj = torch.load(path, weights_only=False)
    B = obj["basis"].float()
    gram_dev = (B @ B.T - torch.eye(B.shape[0])).abs().max().item()
    assert gram_dev < 1e-5, f"{path}: basis not orthonormal (dev {gram_dev:.2e})"
    meta = {"path": str(path), "k": int(obj["k"]), "heads": [list(h) for h in obj["heads"]],
            "definition": obj.get("definition", ""), "built": obj.get("built", "")}
    return B.to(device), meta


def main():
    args = parse_args()
    if args.mode == "propagated":
        bad = [a for a in args.arms if a.endswith("_mean")]
        assert not bad, (f"--mode propagated supports zero-op arms only (no matched mean "
                         f"targets for downstream positions); remove {bad}")
    from baukit import TraceDict

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    torch.set_grad_enabled(False)
    print("Loading model...", flush=True)
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = args.device
    layer_names = list(model_config["layer_hook_names"])
    assert len(layer_names) == N_EDIT_LAYERS
    name_to_block = {nm: int(nm.split(".")[2]) for nm in layer_names}

    # task list + cf assignment
    if args.tasks is not None:
        task_list = list(args.tasks)
        cf_map = build_cf_map(task_list, args.seed, pool=task_list)
    else:
        task_list = [args.task]
        cf_map = {args.task: args.cf_task}
    needed = sorted(set(task_list) | set(cf_map.values()))
    print(f"tasks: {task_list}\ncf_map: {cf_map}", flush=True)

    # subspaces + mean-clamp targets
    bases = {}
    subspace_meta = {}
    for t in needed:
        bases[t], subspace_meta[t] = load_subspace(args.subspace_root, t,
                                                   args.subspace_suffix, device)
    sm = torch.load(args.site_means_path, weights_only=False)
    grand_mean = sm["grand_mean"].float().to(device)              # (3, 28, 4096)
    assert grand_mean.shape[:2] == (len(ROW_NAMES), N_EDIT_LAYERS)
    # targets[t] = (3 sites, 28 layers, k) coords of the grand mean in t's basis
    targets = {t: torch.einsum("slr,kr->slk", grand_mean, bases[t]) for t in bases}
    # unit canonical FVs as k=1 bases for the fv_* arms
    fv_bases = {}
    if any(a.startswith("fv") for a in args.arms):
        for t in needed:
            fv = load_function_vector(args.fv_root, t).float().reshape(-1)
            fv_bases[t] = (fv / fv.norm()).unsqueeze(0).to(device)    # (1, 4096)

    config_name = "multi" if args.tasks is not None else args.task
    write_json(out_root / f"run_config_{config_name}.json", {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "subspaces": subspace_meta,
        "mean_target": {
            "source": str(args.site_means_path),
            "population": sm.get("weighting", ""),
            "tasks": sm.get("tasks", []),
            "counts": sm.get("counts", []),
        },
        "cf_map": cf_map,
        "git_commit": git_commit_hash(),
        "metric": "delta log p (ablated - clean) of the first answer token at the final position",
        "edit_layer_mapping": (
            "start layer L edits transformer.h.{b} outputs for all b >= L at the site token; "
            "FIXED subspace (layer-independent basis), mean-clamp target per (site, edit layer)"
            if args.mode == "anchor" else
            "propagated: start layer L edits transformer.h.{b} outputs for all b >= L at the "
            "site token AND every later position (incl. newline/'Q:'/query tokens); FIXED "
            "layer-independent subspace, zero op only"),
    })

    def forward_hidden(ch):
        return model.transformer(input_ids=ch["input_ids"],
                                 attention_mask=ch["attention_mask"]).last_hidden_state

    def run_logp(chunks, hook=None):
        outs = []
        for ch in chunks:
            if hook is not None:
                with TraceDict(model, layers=layer_names, edit_output=hook):
                    hidden = forward_hidden(ch)
            else:
                hidden = forward_hidden(ch)
            logits = model.lm_head(hidden[:, -1, :]).float()
            logp = logits.log_softmax(dim=-1)
            rows = torch.arange(ch["n"], device=logits.device)
            outs.append(logp[rows, ch["answer_ids"]])
        return torch.cat(outs)

    for task in task_list:
        task_cf = cf_map[task]
        task_dir = out_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        todo = [a for a in args.arms
                if args.overwrite or not (task_dir / f"{a}_delta_logp.npz").exists()]
        if not todo:
            print(f"[{task}] all arms exist; nothing to do.", flush=True)
            continue
        t0 = time.time()
        dataset = load_dataset(task, root_data_dir=args.root_data_dir,
                               test_size=args.test_split, seed=args.seed)
        prompts = build_prompts(task, dataset, tokenizer, model_config, args)
        chunks = make_chunks(prompts, tokenizer, args.batch_size, device)
        n = len(prompts)

        st = prompts[0]["site_texts"]
        for role in ("pre_label_token", "last_prompt_token"):
            assert st[role].strip() == ":", f"{task}: {role} decodes to {st[role]!r}, expected ':'"
        print(f"[{task}] n={n} | sites cue1={st['pre_label_token']!r} "
              f"target1={st['last_label_token']!r} final={st['last_prompt_token']!r}", flush=True)

        clean_logp = run_logp(chunks)
        assert torch.isfinite(clean_logp).all(), f"{task}: non-finite clean log p"
        for i in range(min(3, n)):
            single = make_chunks([prompts[i]], tokenizer, 1, device)
            lp = run_logp(single)[0]
            diff = (lp - clean_logp[i]).abs().item()
            assert diff < 0.05, f"{task}: batched vs single log p differs by {diff:.4f} at {i}"

        def noop_hook(output, layer_name):
            if name_to_block[layer_name] < N_EDIT_LAYERS:
                return output
            return output

        noop_logp = run_logp(chunks, hook=noop_hook)
        assert torch.allclose(noop_logp, clean_logp, atol=1e-5), f"{task}: no-op gate != clean"

        for arm in todo:
            op = "zero" if arm.endswith("_zero") else "mean"
            direction_task = task_cf if "_cf_" in arm else task
            is_fv = arm.startswith("fv")
            B = fv_bases[direction_task] if is_fv else bases[direction_task]   # (k, 4096)
            tgt = None if is_fv else targets[direction_task]                   # (3, 28, k)
            arm_rows = ARM_ROW_NAMES[arm]
            delta = np.full((len(arm_rows), N_EDIT_LAYERS, n), np.nan, dtype=np.float32)
            ta = time.time()
            for ri, row in enumerate(arm_rows):
                site = SITE_INDEX[ROW_SITE[row]]
                for L in args.start_layers:
                    s = 0
                    for ch in chunks:
                        pos_vec = ch["pos"][:, site]
                        rows_idx = torch.arange(ch["n"], device=device)

                        if args.mode == "propagated":
                            # Anchor token and every later position; left padding puts all
                            # pads BEFORE the anchor so the mask never touches padding.
                            seq = ch["input_ids"].shape[1]
                            mask = (torch.arange(seq, device=device)[None, :]
                                    >= pos_vec[:, None])                  # (n, seq)

                            def hook(output, layer_name):
                                b = name_to_block[layer_name]
                                if b < L:
                                    return output
                                h = output[0] if isinstance(output, tuple) else output
                                v = h[mask].float()
                                v = v - (v @ B.T) @ B                     # zero op only
                                if args.debug_invariant:
                                    dev = (v @ B.T).abs().max().item()
                                    assert dev < 1e-3, \
                                        f"invariant broken at b={b} L={L}: {dev:.2e}"
                                h[mask] = v.to(h.dtype)
                                return output
                        else:
                            def hook(output, layer_name):
                                b = name_to_block[layer_name]
                                if b < L:
                                    return output
                                h = output[0] if isinstance(output, tuple) else output
                                v = h[rows_idx, pos_vec, :].float()
                                coords = v @ B.T                          # (n, k)
                                t_bk = tgt[site, b] if op == "mean" else 0.0
                                v = v + (t_bk - coords) @ B
                                if args.debug_invariant:
                                    new_coords = v @ B.T
                                    dev = (new_coords - t_bk).abs().max().item()
                                    assert dev < 1e-3, \
                                        f"invariant broken at b={b} L={L}: {dev:.2e}"
                                h[rows_idx, pos_vec, :] = v.to(h.dtype)
                                return output

                        lp = run_logp([ch], hook=hook)
                        delta[ri, L, s:s + ch["n"]] = (lp - clean_logp[s:s + ch["n"]]).cpu().numpy()
                        s += ch["n"]
            assert np.isfinite(delta[:, args.start_layers, :]).all(), f"{task}/{arm}: non-finite delta"
            provenance = (str(args.fv_root / direction_task) if is_fv
                          else subspace_meta[direction_task]["path"])
            np.savez(task_dir / f"{arm}_delta_logp.npz",
                     delta_logp=delta, clean_logp=clean_logp.cpu().numpy(),
                     row_names=np.array(arm_rows),
                     row_cells=np.array([provenance] * len(arm_rows)),
                     start_layers=np.array(args.start_layers),
                     split=np.array([p["split"] for p in prompts]),
                     query=np.array([p["query"] for p in prompts]),
                     answer=np.array([p["answer"] for p in prompts]),
                     answer_token_id=np.array([p["answer_id"] for p in prompts]),
                     op=np.array(op),
                     subspace_task=np.array(direction_task),
                     cf_task=np.array(task_cf if "_cf_" in arm else ""),
                     mode=np.array("perlayer_fixed_subspace" if args.mode == "anchor"
                                   else "propagated_fixed_subspace"))
            print(f"[{task}] {arm}: {len(arm_rows)} rows x {len(args.start_layers)} layers "
                  f"in {time.time() - ta:.0f}s", flush=True)
        print(f"[{task}] done in {time.time() - t0:.0f}s", flush=True)

    # summaries over whatever npz exist under out_root
    combined = []
    summary_tasks = sorted(d.name for d in out_root.iterdir()
                           if d.is_dir() and any(d.glob("*_delta_logp.npz")))
    for task_name in summary_tasks:
        td = out_root / task_name
        rows_out = []
        for arm in ALL_ARMS:
            f = td / f"{arm}_delta_logp.npz"
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=False)
            delta, split = z["delta_logp"], z["split"]
            test_mask = split == "test"
            for ri, row in enumerate(z["row_names"]):
                for L in z["start_layers"]:
                    d = delta[ri, L]
                    d = d[np.isfinite(d)]
                    dt = delta[ri, L][test_mask]
                    dt = dt[np.isfinite(dt)]
                    if d.size == 0:
                        continue
                    rows_out.append({
                        "task": task_name, "arm": arm, "row": str(row), "start_layer": int(L),
                        "mean_delta_all": float(d.mean()),
                        "sem_all": float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else np.nan,
                        "n_all": int(d.size),
                        "mean_delta_test40": float(dt.mean()) if dt.size else np.nan,
                        "sem_test40": float(dt.std(ddof=1) / np.sqrt(dt.size)) if dt.size > 1 else np.nan,
                        "n_test40": int(dt.size)})
        if rows_out:
            with open(td / "summary.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
                w.writeheader()
                w.writerows(rows_out)
            combined.extend(rows_out)
    if combined:
        with open(out_root / "combined_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(combined[0].keys()))
            w.writeheader()
            w.writerows(combined)
        print(f"summaries written under {out_root}")
    print("DONE")


if __name__ == "__main__":
    main()
