#!/usr/bin/env python
"""Payload-subspace task-switch steering with logit readout (GPT-J-6B; baukit).

Steers a source-task 1-shot prompt toward the target task by REPLACING d_payload-subspace
projections at the demo label token — no paired prompts, no difference-of-means Delta. The
steering targets are unpaired 10-shot task means (last demonstration's label token, train-split
capture prompts) projected into the cached pooled40heads_k4 payload subspaces
(capture_payload_switch_means.py).

The op, at a SINGLE edit layer L (output of transformer.h.L; capture stack row L+1), site =
demo label token, with B_src/B_tgt the (k, 4096) orthonormal bases and c(t->s) task t's mean
coords in task s's basis:

    step 1 (source erase):   v <- v + (c(tgt->src) - v @ B_src^T) @ B_src
    step 2 (target write):   v <- v + (alpha * c(tgt->tgt) - v @ B_tgt^T) @ B_tgt

Step 2 runs LAST so in any subspace overlap the target-task projection wins while the source
projection is minimised (user-specified ordering, 2026-08-04). Arms:
    replace_both        steps 1 + 2 (headline)
    replace_target_only step 2 only (isolates what the source-coord erasure adds)
alpha scales step 2's target coords in both arms; alpha=0 = clean baseline (no hook).

Eval prompts/readout are IDENTICAL to steer_switch_logit.py (same pools, same seed, same
n_test test pairs): queries are shared input words with single-token gold under BOTH tasks,
and we read logit/log p of tgt_gold and src_gold first tokens at the final query position in
one forward pass. The old study's alpha=0 baseline therefore doubles as an advisory
reproduction check.

Sweeps: --layers 0..27 x --alphas {0.5, 1, 2, 4}. Output: per-(direction, arm) npz
(logits + log p, steered and clean), summary.csv, run_config.json. Resumable per npz.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.eval_scripts.exploratory.steer_label_to_query import (  # noqa: E402
    build_prompt_data, extract_positions, get_answer_id,
)
from src.eval_scripts.exploratory.steer_switch_logit import PAIR_FUNCS, build_pools  # noqa: E402
from src.eval_scripts.exploratory.steer_switch_judge import DIRECTIONS  # noqa: E402
from src.eval_scripts.exploratory.ablate_oneshot_preimage_logprob import git_commit_hash  # noqa: E402
from src.utils.prompt_utils import get_token_meta_labels  # noqa: E402
from src.utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR, LABEL_GEOMETRY_DIR  # noqa: E402

SWITCH_DIRECTIONS = ["synonym_to_antonym", "antonym_to_synonym"]
ARMS = ["replace_both", "replace_target_only"]


def parse_args():
    p = argparse.ArgumentParser(description="d_payload subspace-replacement task-switch steering.")
    p.add_argument("--directions", nargs="+", default=SWITCH_DIRECTIONS, choices=SWITCH_DIRECTIONS)
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    p.add_argument("--layers", type=int, nargs="+", default=list(range(28)),
                   help="Edit layers (block indices; single-layer edit each).")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0],
                   help="Strength on the target-task coords (alpha=0 clean baseline is implicit).")
    p.add_argument("--n_test", type=int, default=100, help="Test prompt pairs (as steer_switch_logit).")
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--means_path", type=Path,
                   default=ARTIFACTS_ROOT / "payload_switch_steering" / "tenshot_lastlabel_means.pt")
    p.add_argument("--output_root", type=Path,
                   default=FV_FORMATION_DIR / "ablation" / "attention_head_mechanisms" / "payload_switch_steering")
    p.add_argument("--old_study_root", type=Path, default=LABEL_GEOMETRY_DIR / "oneshot_switch_logit",
                   help="For the advisory clean-baseline reproduction check (may be absent).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--debug_invariant", action="store_true",
                   help="Assert inside the hook that post-edit subspace coords match the targets.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_means(path, source, target, device):
    """-> dict with bases (k,4096) and per-layer coord targets (29,k) for this direction."""
    obj = torch.load(path, weights_only=False)
    for t in (source, target):
        assert t in obj["tasks"], f"{path}: task {t} not captured"
    out = {
        "B_src": obj["bases"][source].to(device),
        "B_tgt": obj["bases"][target].to(device),
        # task t's mean coords in task s's basis are stored as coords["t->s"]
        "c_tgt_in_src": obj["coords"][f"{target}->{source}"].to(device),   # (29, k)
        "c_tgt_in_tgt": obj["coords"][f"{target}->{target}"].to(device),   # (29, k)
        "meta": {k: obj[k] for k in ("basis_files", "counts", "role", "splits",
                                     "activations_root", "layer_convention", "git_commit")},
    }
    for name in ("B_src", "B_tgt"):
        B = out[name]
        dev = (B @ B.T - torch.eye(B.shape[0], device=device)).abs().max().item()
        assert dev < 1e-5, f"{name} not orthonormal (dev {dev:.2e})"
    return out


def build_eval_prompts(direction, args, tok, mc):
    """The SAME n_test test prompt pairs as steer_switch_logit.py (pools + seed + shuffle)."""
    cfg = DIRECTIONS[direction]
    pair, source, target = cfg["pair"], cfg["source"], cfg["target"]
    f1_task, f2_task = PAIR_FUNCS[pair]
    label_pool, query_pool = build_pools(f1_task, f2_task, source, target, args.root_data_dir, tok)
    rng = np.random.default_rng(args.seed)
    lp = label_pool[:]; qp = query_pool[:]
    rng.shuffle(lp); rng.shuffle(qp)
    test_labels, test_queries = lp[:args.n_test], qp[:args.n_test]
    assert len(test_labels) == args.n_test and len(test_queries) == args.n_test, "pool too small for n_test"

    prompts, label_idx, tgt_ids, src_ids, queries, tgt_golds, src_golds = [], [], [], [], [], [], []
    for lab, qy in zip(test_labels, test_queries):
        pd = build_prompt_data(lab["in_src"], lab["label"], qy["q"], lab["label"])
        tk, ps = get_token_meta_labels(pd, tok, query=qy["q"], prepend_bos=mc["prepend_bos"])
        li, _ = extract_positions(tk)
        assert li is not None, f"no demo-label position for prompt {qy['q']!r}"
        prompts.append(ps); label_idx.append(li)
        tgt_ids.append(get_answer_id(ps, " " + qy["tgt_gold"], tok)[0])
        src_ids.append(get_answer_id(ps, " " + qy["src_gold"], tok)[0])
        queries.append(qy["q"]); tgt_golds.append(qy["tgt_gold"]); src_golds.append(qy["src_gold"])
    return {"source": source, "target": target, "prompts": prompts, "label_idx": label_idx,
            "tgt_ids": np.array(tgt_ids), "src_ids": np.array(src_ids),
            "queries": queries, "tgt_golds": tgt_golds, "src_golds": src_golds,
            "n_labels": len(label_pool), "n_queries": len(query_pool)}


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    from baukit import TraceDict
    from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
    set_seed(args.seed)
    print("Loading GPT-J ...")
    model, tok, mc = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    model.eval()
    device = model.device
    hooknames = mc["layer_hook_names"]
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    args.output_root.mkdir(parents=True, exist_ok=True)
    n_layers_sweep, n_alphas = len(args.layers), len(args.alphas)

    for direction in args.directions:
        ev = build_eval_prompts(direction, args, tok, mc)
        source, target = ev["source"], ev["target"]
        sub = load_means(args.means_path, source, target, device)
        n = len(ev["prompts"])
        print(f"\n=== {direction} (source={source} -> target={target}) | {n} test prompts ===")

        # ---- forward pass helper: returns (logits_tgt, logits_src, logp_tgt, logp_src) ----
        tgt_ids_t = torch.tensor(ev["tgt_ids"], device=device)
        src_ids_t = torch.tensor(ev["src_ids"], device=device)

        def run(arm=None, L=None, alpha=None, prompt_slice=None):
            sl = slice(0, n) if prompt_slice is None else prompt_slice
            prompts = ev["prompts"][sl]
            lidx_all = ev["label_idx"][sl]
            o_lt, o_ls, o_pt, o_ps = [], [], [], []
            for s in range(0, len(prompts), args.batch_size):
                enc = tok(prompts[s:s + args.batch_size], return_tensors="pt", padding=True).to(device)
                B_, ilen = enc.input_ids.shape
                if arm is None:
                    lg = model(**enc).logits[:, ilen - 1, :].float()
                else:
                    pad = ilen - enc.attention_mask.sum(1)
                    idx = (pad + torch.tensor(lidx_all[s:s + B_], device=device)).long()
                    rows = torch.arange(B_, device=device)
                    B_src, B_tgt = sub["B_src"], sub["B_tgt"]
                    # capture stack row L+1 = output of transformer.h.L (row 0 = embedding)
                    t_src = sub["c_tgt_in_src"][L + 1]
                    t_tgt = alpha * sub["c_tgt_in_tgt"][L + 1]

                    def hook(output, layer_name):
                        if not (isinstance(output, tuple) and int(layer_name.split(".")[2]) == L):
                            return output
                        h = output[0]
                        v = h[rows, idx].float()
                        if arm == "replace_both":
                            v = v + (t_src - v @ B_src.T) @ B_src
                        v = v + (t_tgt - v @ B_tgt.T) @ B_tgt
                        if args.debug_invariant:
                            dev_t = (v @ B_tgt.T - t_tgt).abs().max().item()
                            assert dev_t < 1e-3, f"invariant fail (tgt coords dev {dev_t:.2e})"
                        h[rows, idx] = v.to(h.dtype)
                        return output
                    with TraceDict(model, layers=hooknames, edit_output=hook):
                        lg = model(**enc).logits[:, ilen - 1, :].float()
                lp = lg.log_softmax(dim=-1)
                r = torch.arange(B_, device=device)
                ti, si = tgt_ids_t[sl][s:s + B_], src_ids_t[sl][s:s + B_]
                o_lt.append(lg[r, ti].cpu().numpy()); o_ls.append(lg[r, si].cpu().numpy())
                o_pt.append(lp[r, ti].cpu().numpy()); o_ps.append(lp[r, si].cpu().numpy())
            return tuple(np.concatenate(x) for x in (o_lt, o_ls, o_pt, o_ps))

        # ---- gates ----
        clean = run()
        assert all(np.isfinite(a).all() for a in clean), "non-finite clean readouts"
        for i in range(min(3, n)):  # batched-vs-unbatched (padding-length fp noise, tol as Stream W)
            single = run(prompt_slice=slice(i, i + 1))
            dev = abs(float(single[2][0]) - float(clean[2][i]))
            assert dev < 0.05, f"batched-vs-unbatched log p dev {dev:.3f} at prompt {i}"

        # no-op gate: a hook that edits nothing must reproduce the same-padding unhooked pass exactly
        clean3 = run(prompt_slice=slice(0, 3))

        def noop_check():
            def hook(output, layer_name):
                return output
            enc = tok(ev["prompts"][:3], return_tensors="pt", padding=True).to(device)
            with TraceDict(model, layers=hooknames, edit_output=hook):
                lg = model(**enc).logits[:, enc.input_ids.shape[1] - 1, :].float()
            lp = lg.log_softmax(dim=-1)
            r = torch.arange(3, device=device)
            return lp[r, tgt_ids_t[:3]].cpu().numpy()
        assert np.allclose(noop_check(), clean3[2], atol=1e-5), "no-op hook != clean"

        clean_diff = float((clean[0] - clean[1]).mean())
        print(f"  gates passed | clean logit(tgt)-logit(src) = {clean_diff:.3f}")
        old_json = args.old_study_root / direction / "logit_diff.json"
        if old_json.exists():  # advisory reproduction check vs the old paired-Delta study
            old_base = json.loads(old_json.read_text())["baseline_alpha0"]["mean_logit_diff"]
            print(f"  advisory: old-study alpha=0 baseline {old_base:.3f} "
                  f"(dev {abs(old_base - clean_diff):.3f}; cross-GPU fp noise ~0.02 expected)")

        # ---- sweep, resumable per (direction, arm) ----
        out_dir = args.output_root / direction
        out_dir.mkdir(parents=True, exist_ok=True)
        for arm in args.arms:
            npz_path = out_dir / f"{arm}_sweep.npz"
            if npz_path.exists() and not args.overwrite:
                print(f"  [skip] {npz_path.name} exists")
                continue
            shape = (n_layers_sweep, n_alphas, n)
            arrs = {k: np.full(shape, np.nan, dtype=np.float32)
                    for k in ("logit_tgt", "logit_src", "logp_tgt", "logp_src")}
            t0 = time.time()
            for li, L in enumerate(args.layers):
                for ai, alpha in enumerate(args.alphas):
                    lt, ls, pt, ps = run(arm=arm, L=int(L), alpha=float(alpha))
                    arrs["logit_tgt"][li, ai] = lt; arrs["logit_src"][li, ai] = ls
                    arrs["logp_tgt"][li, ai] = pt; arrs["logp_src"][li, ai] = ps
                print(f"  {arm}: layer {L} done ({time.time() - t0:.0f}s)")
            assert all(np.isfinite(a).all() for a in arrs.values()), f"non-finite sweep values ({arm})"
            np.savez_compressed(
                npz_path, **arrs,
                clean_logit_tgt=clean[0], clean_logit_src=clean[1],
                clean_logp_tgt=clean[2], clean_logp_src=clean[3],
                layers=np.array(args.layers), alphas=np.array(args.alphas),
                arm=arm, direction=direction, source=source, target=target,
                query=np.array(ev["queries"]), tgt_gold=np.array(ev["tgt_golds"]),
                src_gold=np.array(ev["src_golds"]),
                tgt_token_id=ev["tgt_ids"], src_token_id=ev["src_ids"],
                metric="logit/logp of tgt_gold & src_gold first tokens at final position; "
                       "headline contrast = logit(tgt) - logit(src)")
            print(f"  wrote {npz_path}")

        run_config = {
            "direction": direction, "source": source, "target": target,
            "arms": args.arms, "layers": args.layers, "alphas": args.alphas,
            "n_test": args.n_test, "seed": args.seed, "batch_size": args.batch_size,
            "op": "step1 (replace_both only): v += (c(tgt->src) - v@B_src^T)@B_src; "
                  "step2: v += (alpha*c(tgt->tgt) - v@B_tgt^T)@B_tgt; single edit layer; "
                  "site = demo label token; edit layer L uses capture stack row L+1",
            "means_path": str(args.means_path), "means_meta": {
                k: v for k, v in sub["meta"].items()},
            "clean_logit_diff": clean_diff,
            "pool_sizes": {"labels": ev["n_labels"], "queries": ev["n_queries"]},
            "model_name": args.model_name, "git_commit": git_commit_hash(),
        }
        (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    # ---- summary.csv over all npz on disk ----
    rows = []
    for d_dir in sorted(args.output_root.iterdir()):
        if not d_dir.is_dir():
            continue
        for npz_path in sorted(d_dir.glob("*_sweep.npz")):
            z = np.load(npz_path, allow_pickle=False)
            layers, alphas = z["layers"], z["alphas"]
            contrast = z["logit_tgt"] - z["logit_src"]
            clean_contrast = z["clean_logit_tgt"] - z["clean_logit_src"]
            dpt = z["logp_tgt"] - z["clean_logp_tgt"][None, None, :]
            dps = z["logp_src"] - z["clean_logp_src"][None, None, :]
            for li, L in enumerate(layers):
                for ai, a in enumerate(alphas):
                    c = contrast[li, ai]
                    sem = c.std(ddof=1) / np.sqrt(len(c))
                    rows.append({
                        "direction": str(z["direction"]), "arm": str(z["arm"]),
                        "layer": int(L), "alpha": float(a),
                        "mean_logit_diff": float(c.mean()), "ci95": float(1.96 * sem),
                        "mean_dlogp_tgt": float(dpt[li, ai].mean()),
                        "mean_dlogp_src": float(dps[li, ai].mean()),
                        "flip_rate": float((c > 0).mean()),
                        "clean_logit_diff": float(clean_contrast.mean()),
                        "clean_flip_rate": float((clean_contrast > 0).mean()),
                        "n": int(len(c)),
                    })
    if rows:
        csv_path = args.output_root / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
