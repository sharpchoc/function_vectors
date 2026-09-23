#!/usr/bin/env python
"""Summaries for the execution-side carrier / task-unique split ablation (exec_split_ablation.py), both models.

Adds, per task, the paper's full-FV ablation results on the same prompts (FV_ablation/eval, blocks 9-27) and two geometry
diagnostics computed from the stored cue means: the mean-ablation shift relative to the zero-ablation shift along the full FV
and along the task-unique direction (blocks 9-27, cue block inputs).
Task bootstrap 10,000 resamples, seed 20260923. Writes <results>/FV_ablation/task_unique_split/{per_task.csv, summary.csv}.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT, TASK69_RUN_DIR, QWEN25_FV_DIR  # noqa: E402

MODELS = {"gptj": (ARTIFACTS_ROOT / "69_task_run", ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs",
                   REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json", TASK69_RUN_DIR / "FV_ablation" / "task_unique_split", 69),
          "qwen25": (ARTIFACTS_ROOT / "qwen25_fv", ARTIFACTS_ROOT / "qwen25_read" / "perprompt_fvs",
                     REPO_ROOT / "task_splits" / "qwen25_ext_steerable_96_prunedfail.json", QWEN25_FV_DIR / "FV_ablation" / "task_unique_split", 96)}
CONDS = ["baseline", "uniq_own_mean", "uniq_own_zero", "uniq_cf_mean", "carrier_mean", "carrier_zero", "span2_own_mean", "span2_cf_mean"]
FULL = {"full_own_mean": "own_mean_L9to27", "full_own_zero": "own_zero_L9to27", "full_cf_mean": "cf_mean_L9to27",
        "full_cf_zero": "cf_zero_L9to27", "full_baseline": "real6_baseline"}
RNG = np.random.default_rng(20260923)


def ci(x):
    x = np.asarray(x, float); m = x[RNG.integers(0, len(x), (10000, len(x)))].mean(1)
    return {"mean": float(x.mean()), "lo": float(np.quantile(m, .025)), "hi": float(np.quantile(m, .975))}


def main():
    for model, (art, fvr, sp, out, n_exp) in MODELS.items():
        out.mkdir(parents=True, exist_ok=True)
        s = json.load(open(sp)); tasks = sorted(s["train_tasks"] + s["heldout_tasks"])
        V = {t: torch.load(fvr / f"{t}.pt", weights_only=False)["fv"].double().mean(0) for t in tasks}
        ch = torch.stack([V[t] for t in s["train_tasks"]]).mean(0); ch = ch / ch.norm()
        g = torch.load(art / "FV_ablation" / "grand_mean_cue6.pt", weights_only=False)["mean"].double(); L = list(range(9, 28))
        R = [json.load(open(f)) for f in sorted(glob.glob(str(art / "exec_split_ablation" / "*.json")))]
        assert len(R) == n_exp, (model, len(R))
        rows = []
        for r in R:
            t = r["task"]; e = json.load(open(art / "FV_ablation" / "eval" / f"{t}.json"))
            v = V[t]; fh = v / v.norm(); u = v - (v @ ch) * ch; uh = u / u.norm()
            mu = torch.load(art / "FV_ablation" / "cue_means" / f"{t}.pt", weights_only=False)["mean"].double()[L]

            def shift(d):
                pt, pg = mu @ d, g[L] @ d
                return float(((pt - pg).abs() / pt.abs().clamp_min(1e-9)).median())
            rows.append({"task": t, "group": r["group"], "cf_task": r["cf_task"],
                         **{c: r["conditions"][c]["acc"] for c in CONDS},
                         **{k: e["conditions"][v_]["acc"] for k, v_ in FULL.items()},
                         "cos_uniq_own_cf": r["cos_uniq_own_cf"], "cos_fv_carrier": r["cos_fv_carrier"],
                         "uniq_share_fv_norm2": float((u.norm() / v.norm()) ** 2),
                         "meanabl_shift_ratio_full": shift(fh), "meanabl_shift_ratio_uniq": shift(uh)})
        d = pd.DataFrame(rows); d.round(4).to_csv(out / "per_task.csv", index=False)
        summ = [{"quantity": c, **ci(d[c])} for c in CONDS + list(FULL) + ["cos_uniq_own_cf", "cos_fv_carrier", "uniq_share_fv_norm2",
                                                                            "meanabl_shift_ratio_full", "meanabl_shift_ratio_uniq"]]
        own = d.baseline - d.uniq_own_mean; cf = d.baseline - d.uniq_cf_mean
        summ += [{"quantity": "drop_uniq_own_mean", **ci(own)}, {"quantity": "drop_uniq_cf_mean", **ci(cf)},
                 {"quantity": "drop_uniq_own_minus_cf_mean", **ci(own - cf)},
                 {"quantity": "n_tasks_uniq_own_drop_gt_cf", "mean": int((own > cf).sum()), "lo": np.nan, "hi": np.nan},
                 {"quantity": "n_tasks", "mean": len(d), "lo": np.nan, "hi": np.nan}]
        pd.DataFrame(summ).round(4).to_csv(out / "summary.csv", index=False)
        print(f"== {model}\n{pd.read_csv(out / 'summary.csv').to_string(index=False)}", flush=True)


if __name__ == "__main__":
    main()
