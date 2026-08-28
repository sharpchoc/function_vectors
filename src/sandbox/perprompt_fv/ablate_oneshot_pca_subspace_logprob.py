#!/usr/bin/env python
"""SANDBOX (not repo standard): top-k PCA pre-image SUBSPACE ablation (1-shot, GPT-J).

Sibling of the Stream W study (`src/eval_scripts/ablate_oneshot_preimage_logprob.py`, which is
imported for prompts/chunks/cf-assignment and left untouched). Instead of a single direction,
each edit removes the residual-stream component in a per-(task, cell, layer) SUBSPACE built
from the per-prompt-FV pre-images (fit_preimage_pca_subspace_banks.py):

  Q_b   [4096, k+1]  orthonormal basis of span{task-mean pre-image, top-k centered PCs}
                     at edit layer b (k = 0 is the mean-direction-only bridge arm)
  ops   zero:  h[:, pos] -= (h[:, pos] @ Q_b) Q_b^T
        mean:  h[:, pos] -= ((h[:, pos] @ Q_b) - tvec_b) Q_b^T
               with tvec_b = Q_b^T (grand mean over ALL 27 tasks' pre-images at that cell)

perlayer mode only (each block b >= L uses ITS OWN layer's subspace, at the site token only).
Metric, prompts, site tokens, cf assignment, batching identical to Stream W: delta log p
(ablated - clean) of the first answer token at the final position, 170 one-shot prompts per
task, seed 42, site tokens cue1 / target1 / final_cue, start layers 0..27.

Arms (32 per task): pcasub_{matched|icl10}[_cf]_k{0,2,3,4}_{zero|mean}
  matched cells: cue1<-pre_label_icl1, target1<-last_label_icl1, final_cue<-pre_label_icl2
  icl10 cells:   cue1<-pre_label_icl10, target1<-last_label_icl10, final_cue<-last_prompt_icl10
  _cf uses the SAME cf task draw as Stream W (build_cf_map, stable_rng("cf_assignment", 42, t)).

Comparability gate: query/answer/answer_token_id arrays must EXACTLY match the stored Stream W
npz for the task (same prompt construction); clean log p is compared as an ADVISORY (cross-GPU
fp noise ~1.7e-2 is known; same-GPU rule per DECISIONS 2026-07-16).

Resumable per (task, arm) npz. Summaries rebuilt over ALL task dirs on disk.
Output -> results/sandbox/perprompt_ridge_pilot/oneshot_pca_subspace_ablation/
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.eval_scripts.exploratory.ablate_oneshot_preimage_logprob import (  # noqa: E402
    N_EDIT_LAYERS,
    ROW_SITE,
    SITE_INDEX,
    build_cf_map,
    build_prompts,
    make_chunks,
)
from src.eval_scripts.regress_activation_to_fv_fulldim_ridge import (  # noqa: E402
    DEFAULT_TEST_TASKS_EXCLUDE_CC_PC,
    write_json,
)
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed  # noqa: E402
from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR, RESULTS_ROOT  # noqa: E402
from utils.prompt_utils import load_dataset  # noqa: E402

BASE_CELLS = {
    "matched": {"cue1": "pre_label_token_icl1", "target1": "last_label_token_icl1",
                "final_cue": "pre_label_token_icl2"},
    "icl10": {"cue1": "pre_label_token_icl10", "target1": "last_label_token_icl10",
              "final_cue": "last_prompt_token_icl10"},
}
ROW_ORDER = ["cue1", "target1", "final_cue"]
KS = [0, 2, 3, 4]
OPS = ["zero", "mean"]
ALL_ARMS = [f"pcasub_{base}{cf}_k{k}_{op}"
            for base in BASE_CELLS for cf in ("", "_cf") for k in KS for op in OPS]


def parse_arm(arm):
    """'pcasub_matched_cf_k2_zero' -> (base, is_cf, k, op)."""
    body = arm[len("pcasub_"):]
    body, op = body.rsplit("_", 1)
    body, kpart = body.rsplit("_", 1)
    is_cf = body.endswith("_cf")
    base = body[:-3] if is_cf else body
    assert base in BASE_CELLS and op in OPS and kpart.startswith("k"), arm
    return base, is_cf, int(kpart[1:]), op


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TEST_TASKS_EXCLUDE_CC_PC))
    p.add_argument("--arms", nargs="+", default=ALL_ARMS, choices=ALL_ARMS)
    p.add_argument("--banks_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox/perprompt_fv_preimages/gptj_train_varicl_top40_pca_banks")
    p.add_argument("--reference_root", type=Path,
                   default=FV_FORMATION_DIR / "oneshot_preimage_ablation/train_varicl_top40",
                   help="Stream W output root for the prompt-identity comparability gate.")
    p.add_argument("--output_root", type=Path,
                   default=RESULTS_ROOT / "sandbox/perprompt_ridge_pilot/oneshot_pca_subspace_ablation")
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
    p.add_argument("--debug_invariant", action="store_true",
                   help="Assert |Q^T h - target| ~ 0 on edited rows inside the hook (smoke runs).")
    p.add_argument("--skip_reference_gate", action="store_true",
                   help="Allow running when the Stream W reference npz are absent.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:
        return None


def load_subspaces(cell, direction_task, k, banks_root, device):
    """Q [28, 4096, k+1] fp32 + tvec [28, k+1] fp32 + per-layer diag dict."""
    path = banks_root / cell / f"{direction_task}_pca_subspace_bank.pt"
    bank = torch.load(path, map_location="cpu", weights_only=False)["subspaces_by_edit_layer"]
    missing = [b for b in range(N_EDIT_LAYERS) if b not in bank]
    assert not missing, f"{path} missing edit layers {missing}"
    Q = torch.stack([bank[b][k]["Q"] for b in range(N_EDIT_LAYERS)]).float()
    tvec = torch.stack([bank[b][k]["tvec"] for b in range(N_EDIT_LAYERS)]).float()
    diag = {"m_norm": np.array([bank[b]["diag"]["m_norm"] for b in range(N_EDIT_LAYERS)]),
            "g_norm": np.array([bank[b]["diag"]["g_norm"] for b in range(N_EDIT_LAYERS)])}
    assert Q.shape == (N_EDIT_LAYERS, 4096, k + 1), f"{path}: bad Q shape {tuple(Q.shape)} for k={k}"
    return Q.to(device), tvec.to(device), diag


def reference_gate(task, prompts, clean_logp, args):
    """Prompt identity must match Stream W exactly; clean log p advisory only."""
    ref_files = sorted((args.reference_root / task).glob("*_delta_logp.npz"))
    if not ref_files:
        msg = f"[{task}] no Stream W reference npz under {args.reference_root / task}"
        if args.skip_reference_gate:
            print(msg + " (skipped by flag)", flush=True)
            return
        raise FileNotFoundError(msg + " — pass --skip_reference_gate to proceed without it.")
    z = np.load(ref_files[0], allow_pickle=False)
    n = len(prompts)
    for field, mine in (("query", [p["query"] for p in prompts]),
                        ("answer", [p["answer"] for p in prompts]),
                        ("answer_token_id", [p["answer_id"] for p in prompts])):
        ref = z[field][:n]
        same = all(str(a) == str(b) for a, b in zip(mine, ref))
        if not same:
            raise RuntimeError(f"[{task}] COMPARABILITY GATE FAILED: {field} differs from "
                               f"{ref_files[0].name} — STOP, user adjudicates.")
    dmax = float(np.abs(clean_logp.cpu().numpy() - z["clean_logp"][:n]).max())
    level = "OK" if dmax < 0.05 else "WARN (cross-GPU fp noise is ~1.7e-2; investigate if larger)"
    print(f"[{task}] reference gate: prompt identity EXACT vs {ref_files[0].name}; "
          f"clean log p max|diff|={dmax:.4f} [{level}]", flush=True)


def main():
    args = parse_args()
    from baukit import TraceDict   # late import, precedent in the Stream W script

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
    layer_names = list(model_config["layer_hook_names"])     # transformer.h.0..27
    assert len(layer_names) == N_EDIT_LAYERS
    name_to_block = {nm: int(nm.split(".")[2]) for nm in layer_names}

    cf_map = build_cf_map(args.tasks, args.seed)
    write_json(out_root / "run_config.json", {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "sandbox": True, "cf_map": cf_map, "base_cells": BASE_CELLS, "ks": KS, "ops": OPS,
        "git_commit": git_commit_hash(),
        "metric": "delta log p (ablated - clean) of the first answer token at the final position",
        "subspace": "span{task-mean pre-image, top-k centered PCs} of the per-prompt-FV "
                    "pre-images; mean op clamps to the all-27-task grand-mean pre-image's "
                    "subspace component",
        "edit_layer_mapping": "start layer L edits transformer.h.{b} outputs for all b >= L "
                              "with layer b's OWN subspace (edit layers 0..27; embedding never "
                              "touched)",
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

    for task in args.tasks:
        task_dir = out_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        todo = [a for a in args.arms
                if args.overwrite or not (task_dir / f"{a}_delta_logp.npz").exists()]
        if not todo:
            print(f"[{task}] all arms exist; skipping.", flush=True)
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
            if name_to_block[layer_name] < N_EDIT_LAYERS:   # gate at L=28: never fires
                return output
            return output

        noop_logp = run_logp(chunks, hook=noop_hook)
        assert torch.allclose(noop_logp, clean_logp, atol=1e-5), f"{task}: no-op gate != clean"
        reference_gate(task, prompts, clean_logp, args)

        for arm in todo:
            base, is_cf, k, op = parse_arm(arm)
            direction_task = cf_map[task] if is_cf else task
            cells = BASE_CELLS[base]
            delta = np.full((len(ROW_ORDER), N_EDIT_LAYERS, n), np.nan, dtype=np.float32)
            mnorm = np.full((len(ROW_ORDER), N_EDIT_LAYERS), np.nan, dtype=np.float32)
            ta = time.time()
            for ri, row in enumerate(ROW_ORDER):
                Q, tvec, diag = load_subspaces(cells[row], direction_task, k,
                                               args.banks_root, device)
                mnorm[ri] = diag["m_norm"]
                site = SITE_INDEX[ROW_SITE[row]]
                target = tvec if op == "mean" else torch.zeros_like(tvec)
                for L in args.start_layers:
                    s = 0
                    for ch in chunks:
                        pos_vec = ch["pos"][:, site]
                        rows_idx = torch.arange(ch["n"], device=device)

                        def hook(output, layer_name):
                            b = name_to_block[layer_name]
                            if b < L:
                                return output
                            h = output[0] if isinstance(output, tuple) else output
                            v = h[rows_idx, pos_vec, :].float()
                            c = v @ Q[b]                              # [n, k+1]
                            v = v - (c - target[b]) @ Q[b].T
                            if args.debug_invariant:
                                err = (v @ Q[b] - target[b]).abs().max().item()
                                vmax = v.norm(dim=1).max().item()
                                assert err < max(1e-3, 1e-4 * vmax), \
                                    f"{task}/{arm}/{row}/L{L}/b{b}: invariant err {err:.2e} (vmax {vmax:.1f})"
                            h[rows_idx, pos_vec, :] = v.to(h.dtype)
                            return output

                        lp = run_logp([ch], hook=hook)
                        delta[ri, L, s:s + ch["n"]] = (lp - clean_logp[s:s + ch["n"]]).cpu().numpy()
                        s += ch["n"]
            assert np.isfinite(delta[:, args.start_layers, :]).all(), f"{task}/{arm}: non-finite delta"
            np.savez(task_dir / f"{arm}_delta_logp.npz",
                     delta_logp=delta, clean_logp=clean_logp.cpu().numpy(),
                     row_names=np.array(ROW_ORDER),
                     row_cells=np.array([cells[r] for r in ROW_ORDER]),
                     start_layers=np.array(args.start_layers),
                     split=np.array([p["split"] for p in prompts]),
                     query=np.array([p["query"] for p in prompts]),
                     answer=np.array([p["answer"] for p in prompts]),
                     answer_token_id=np.array([p["answer_id"] for p in prompts]),
                     task_mean_preimage_norms=mnorm,
                     mode=np.array("perlayer"), k=np.array(k), op=np.array(op),
                     subspace_dim=np.array(k + 1),
                     cf_task=np.array(cf_map[task] if is_cf else ""))
            print(f"[{task}] {arm}: {len(ROW_ORDER)} rows x {len(args.start_layers)} layers "
                  f"in {time.time() - ta:.0f}s", flush=True)
        print(f"[{task}] done in {time.time() - t0:.0f}s", flush=True)

    # --- summaries over whatever npz exist (ALL task dirs on disk, Stream W convention) ---
    combined = []
    summary_tasks = sorted(d.name for d in out_root.iterdir()
                           if d.is_dir() and any(d.glob("*_delta_logp.npz")))
    for task in summary_tasks:
        task_dir = out_root / task
        rows_out = []
        for arm in ALL_ARMS:
            f = task_dir / f"{arm}_delta_logp.npz"
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
                        "task": task, "arm": arm, "row": str(row),
                        "k": int(z["k"]), "op": str(z["op"]), "start_layer": int(L),
                        "mean_delta_all": float(d.mean()),
                        "sem_all": float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else np.nan,
                        "n_all": int(d.size),
                        "mean_delta_test40": float(dt.mean()) if dt.size else np.nan,
                        "sem_test40": float(dt.std(ddof=1) / np.sqrt(dt.size)) if dt.size > 1 else np.nan,
                        "n_test40": int(dt.size)})
        if rows_out:
            with open(task_dir / "summary.csv", "w", newline="") as fh:
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
