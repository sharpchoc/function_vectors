#!/usr/bin/env python
"""(token position x layer) ridge sweep: activation -> per-prompt FV, scored on task FVs.

One invocation handles ONE layer (--layer), so a single forward pass over all 69 tasks x
150 clean 10-shot prompts serves all 31 token positions of that layer; the grid is sharded
across pods by layer.

Token positions (31, matching src/eval_scripts/regress_activation_to_fv_fulldim_ridge.py):
  for each demo n = 1..10:  pre-label (the ':' predictive token), first label token,
                            last label token
  plus the final prompt token (the query cue).

Per (token, layer) cell:
  X = that position's residual activation (block output), one row per (task, prompt)
  Y = the per-prompt FV (artifacts/69_task_run/perprompt_fvs) -- the TRAINING target
  fit ridge on the 55 TRAIN tasks' 8250 rows, alpha from logspace(-1, 8, 19) by 5-fold CV
  over TRAIN TASKS (pooled MSE), refit on all train rows.

Scoring (user spec 2026-08-18): the EVALUATION target is the TASK FV (that task's mean
per-prompt FV), not the per-prompt FV. Per task,
    R^2_t = 1 - sum_i ||pred_i - taskFV_t||^2 / sum_i ||taskFV_t - pool_mean||^2
with pool_mean = the mean task FV of that split (held-out pool for held-out tasks, train
pool for train tasks); dims with zero pooled variance dropped; uniform average over dims.
The reported cell value is the mean of the per-task R^2 values. Train panel uses the same
metric in-sample.

Output: artifacts/69_task_run/token_layer_regressions/layer<L>.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from baukit import TraceDict
from src.utils.model_utils import load_gpt_model_and_tokenizer
from src.utils.prompt_utils import get_token_meta_labels, word_pairs_to_prompt_data
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT

FVS = ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs"
LAB_RE = re.compile(r"^demonstration_(\d+)_label_token$")
INP_RE = re.compile(r"^demonstration_(\d+)_token$")
N_DEMOS = 10
# label-side positions (the original 31): per demo the ':' before the label, and the first
# and last token of the label itself, plus the final query cue.
POS_LABEL = [f"d{n}_{r}" for n in range(1, N_DEMOS + 1)
             for r in ("pre", "first", "last")] + ["query_cue"]
# input-side positions (added 2026-08-18 on user request): per demo the first and last
# token of the INPUT word (after "Q:"), plus the query's own input last token.
POS_INPUT = [f"d{n}_inp_{r}" for n in range(1, N_DEMOS + 1)
             for r in ("first", "last")] + ["query_inp_last"]
POS_SETS = {"label31": POS_LABEL, "input21": POS_INPUT, "all52": POS_LABEL + POS_INPUT}
ALPHAS = list(np.logspace(-1, 8, 19))
KFOLDS, CV_SEED = 5, 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "token_layer_regressions")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--pos_set", choices=("label31", "input21", "all52"), default="label31")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--fp64_check_pos", type=int, default=-1,
                   help="also fit this position in fp64 and report the R^2 delta "
                        "(-1 = off; enable on ONE layer only, it doubles peak GPU memory)")
    return p.parse_args()


def prompt_positions(rec, tokenizer, pos_set="label31"):
    """(prompt_string, [token indices for the chosen position set]); no bos anywhere."""
    wp = {"input": [str(d["input"]) for d in rec["demos"]],
          "output": [str(d["output"]) for d in rec["demos"]]}
    qo = rec["query"]["output"]
    qo = [str(x) for x in qo] if isinstance(qo, list) else str(qo)
    q = {"input": str(rec["query"]["input"]), "output": qo}
    pd_ = word_pairs_to_prompt_data(wp, query_target_pair=q, prepend_bos_token=False,
                                    shuffle_labels=False)
    token_labels, prompt_string = get_token_meta_labels(pd_, tokenizer, query=q["input"],
                                                        prepend_bos=False)
    ids = tokenizer(prompt_string).input_ids
    assert len(ids) == len(token_labels)
    first, last = {}, {}
    inp_first, inp_last, q_inp_last = {}, {}, None
    for i, _, lab in token_labels:
        i = int(i)
        m = LAB_RE.match(lab)
        if m:
            n = int(m.group(1))
            first[n] = min(first.get(n, i), i)
            last[n] = max(last.get(n, -1), i)
            continue
        m = INP_RE.match(lab)          # demonstration_<n>_token = the input word
        if m:
            n = int(m.group(1))
            inp_first[n] = min(inp_first.get(n, i), i)
            inp_last[n] = max(inp_last.get(n, -1), i)
        elif lab == "query_demonstration_token":
            q_inp_last = i if q_inp_last is None else max(q_inp_last, i)
    assert sorted(first) == list(range(1, N_DEMOS + 1)), f"demos found: {sorted(first)}"

    idxs = []
    if pos_set in ("label31", "all52"):
        for n in range(1, N_DEMOS + 1):
            assert first[n] - 1 >= 0
            idxs += [first[n] - 1, first[n], last[n]]
        idxs.append(len(ids) - 1)
    if pos_set in ("input21", "all52"):
        assert sorted(inp_first) == list(range(1, N_DEMOS + 1)), \
            f"input tokens found for demos {sorted(inp_first)}"
        assert q_inp_last is not None, "no query_demonstration_token"
        for n in range(1, N_DEMOS + 1):
            idxs += [inp_first[n], inp_last[n]]
        idxs.append(q_inp_last)
    assert len(idxs) == len(POS_SETS[pos_set])
    return prompt_string, idxs


def r2_vs_taskfv(pred, task_fv, pool_mean):
    """Per-task R^2 with the constant task-FV target and split-pool-mean denominator.

    Summed over dims in BOTH numerator and denominator (vector-norm form):
        R^2_t = 1 - sum_i ||pred_i - taskFV_t||^2 / (n * ||taskFV_t - pool_mean||^2)
    A per-dim R^2 averaged over dims is NOT usable here: the target is constant within a
    task, so a dim where taskFV_t happens to sit near the pool mean has a ~0 denominator
    and explodes (that bug produced R^2 ~ -2.5e4 in the first run of this sweep)."""
    resid = ((pred - task_fv) ** 2).sum()
    tot = ((task_fv - pool_mean) ** 2).sum() * pred.shape[0]
    return float(1 - resid / tot)


def ridge_fit_eval(X, Y, tr_idx, te_slices, tr_slices, task_fv, dtype, device):
    """Returns (mean heldout R^2, mean train R^2, best_alpha, pinned)."""
    Xtr = X[tr_idx].to(device=device, dtype=dtype)
    Ytr = Y[tr_idx].to(device=device, dtype=dtype)
    # ---- CV over train tasks ----
    rng = np.random.RandomState(CV_SEED)
    tasks = sorted(tr_slices)
    order = rng.permutation(len(tasks))
    folds = [sorted(tasks[i] for i in f) for f in np.array_split(order, KFOLDS)]
    cv = torch.zeros(len(ALPHAS), dtype=dtype, device=device)
    row_of = {t: tr_slices[t] for t in tasks}
    for fold in folds:
        m = torch.zeros(len(Xtr), dtype=torch.bool)
        for t in fold:
            s, e = row_of[t]
            m[s:e] = True
        m = m.to(device)
        xf, yf = Xtr[~m], Ytr[~m]
        xbar, ybar = xf.mean(0), yf.mean(0)
        xc = xf - xbar
        evals, evecs = torch.linalg.eigh(xc.T @ xc)
        c = evecs.T @ (xc.T @ (yf - ybar))
        a_val = (Xtr[m] - xbar) @ evecs
        for ai, al in enumerate(ALPHAS):
            cv[ai] += (((a_val / (evals + al)) @ c + ybar - Ytr[m]) ** 2).sum()
    bi = int(torch.argmin(cv))
    best_alpha = float(ALPHAS[bi])

    xbar, ybar = Xtr.mean(0), Ytr.mean(0)
    xc = Xtr - xbar
    evals, evecs = torch.linalg.eigh(xc.T @ xc)
    c = evecs.T @ (xc.T @ (Ytr - ybar))

    def predict(xe):
        return ((xe - xbar) @ evecs / (evals + best_alpha)) @ c + ybar

    out = {}
    for name, slices in (("heldout", te_slices), ("train", tr_slices)):
        fvs = torch.stack([task_fv[t] for t in sorted(slices)]).to(device=device, dtype=dtype)
        pool_mean = fvs.mean(0)
        r2s = []
        for t in sorted(slices):
            s, e = slices[t]
            xe = (X[s:e] if name == "heldout" else Xtr[s:e]).to(device=device, dtype=dtype)
            r2s.append(r2_vs_taskfv(predict(xe), task_fv[t].to(device=device, dtype=dtype),
                                    pool_mean))
        out[name] = float(np.mean(r2s))
        out[name + "_per_task"] = {t: round(v, 4) for t, v in zip(sorted(slices), r2s)}
    return out, best_alpha, bi in (0, len(ALPHAS) - 1)


def main():
    args = parse_args()
    device = "cuda"
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    split = json.load(open(args.split_path))
    train_tasks, test_tasks = sorted(split["train_tasks"]), sorted(split["heldout_tasks"])
    args.out_root.mkdir(parents=True, exist_ok=True)

    pos_names = POS_SETS[args.pos_set]
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    tokenizer.padding_side = "right"
    lname = model_config["layer_hook_names"][args.layer]
    print(f"layer {args.layer} -> {lname} | pos_set {args.pos_set} "
          f"({len(pos_names)} positions)", flush=True)

    # ---- one forward pass per task, keep only this layer's 31 positions ----
    def capture(task):
        recs = json.load(open(args.prompts_root / task / "train_prompts.json"))
        assert len(recs) == 150
        built = [prompt_positions(r, tokenizer, args.pos_set) for r in recs]
        acts = torch.zeros(150, len(pos_names), 4096, dtype=torch.float16)
        for s in range(0, len(built), args.batch_size):
            chunk = built[s:s + args.batch_size]
            idxs = torch.tensor([c[1] for c in chunk], device=model.device)
            inp = tokenizer([c[0] for c in chunk], return_tensors="pt",
                            padding=True).to(model.device)
            with torch.no_grad(), TraceDict(model, layers=[lname], retain_output=True) as td:
                model(**inp)
            o = td[lname].output
            o = o[0] if isinstance(o, tuple) else o
            b = torch.arange(len(chunk), device=model.device).unsqueeze(1)
            acts[s:s + len(chunk)] = o[b, idxs].half().cpu()
        return acts

    Xall, Yall, tr_slices, te_slices, task_fv = [], [], {}, {}, {}
    pos_tr = 0
    for t in train_tasks:
        a = capture(t)
        f = torch.load(FVS / f"{t}.pt", map_location="cpu", weights_only=False)["fv"].float()
        Xall.append(a); Yall.append(f)
        tr_slices[t] = (pos_tr, pos_tr + 150); pos_tr += 150
        task_fv[t] = f.mean(0)
    n_train_rows = pos_tr
    pos_te = n_train_rows
    for t in test_tasks:
        a = capture(t)
        f = torch.load(FVS / f"{t}.pt", map_location="cpu", weights_only=False)["fv"].float()
        Xall.append(a); Yall.append(f)
        te_slices[t] = (pos_te, pos_te + 150); pos_te += 150
        task_fv[t] = f.mean(0)
    X = torch.cat(Xall)                       # (10350, 31, 4096) fp16 cpu
    Y = torch.cat(Yall)                       # (10350, 4096) fp32 cpu
    tr_idx = torch.arange(n_train_rows)
    print(f"captured X {tuple(X.shape)}", flush=True)

    results = {}
    for pi, pname in enumerate(pos_names):
        Xp = X[:, pi].float()
        res, alpha, pinned = ridge_fit_eval(Xp, Y, tr_idx, te_slices, tr_slices, task_fv,
                                            dtype, device)
        entry = {"position": pname, "best_alpha": alpha, "alpha_pinned": pinned,
                 "r2_heldout_mean": round(res["heldout"], 4),
                 "r2_train_mean": round(res["train"], 4),
                 "r2_heldout_per_task": res["heldout_per_task"]}
        if pi == args.fp64_check_pos:
            res64, _, _ = ridge_fit_eval(Xp, Y, tr_idx, te_slices, tr_slices, task_fv,
                                         torch.float64, device)
            entry["fp64_check"] = {"r2_heldout_fp64": round(res64["heldout"], 4),
                                   "delta": round(res64["heldout"] - res["heldout"], 5)}
            print(f"  fp64 check @ {pname}: fp32 {res['heldout']:.4f} vs fp64 "
                  f"{res64['heldout']:.4f}", flush=True)
        results[pname] = entry
        print(f"L{args.layer} {pname:12s} alpha={alpha:g}{' PIN' if pinned else ''} | "
              f"heldout {res['heldout']:.4f} train {res['train']:.4f}", flush=True)
        del Xp
        torch.cuda.empty_cache()

    suffix = "" if args.pos_set == "label31" else f"_{args.pos_set}"
    with open(args.out_root / f"layer{args.layer}{suffix}.json", "w") as f:
        json.dump({"layer": args.layer, "positions": pos_names, "dtype": args.dtype,
                   "scoring": "target = task FV; per-task R^2 vs split-pool mean; "
                              "uniform over dims; mean across tasks",
                   "results": results}, f, indent=1)
    print("layer done", flush=True)


if __name__ == "__main__":
    main()
