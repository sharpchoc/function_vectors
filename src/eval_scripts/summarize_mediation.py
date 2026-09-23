#!/usr/bin/env python
"""Summaries for the reviewer mediation tests (2026-09-23), both models.

T2.4 (mediation_ablate_readcue.py): cue-FV cosine at the readout block (GPT-J 13, Qwen 24) under no ablation, own u_hat_A
ablation (mean / zero) and other-task ablation, plus the generic-FV cosine.
T2.5 (mediation_steer_fvablate.py): six-dummy-slot identification steering (alpha 2) alone, with the task's own unit FV
zero-ablated at the query cue (blocks 9-27), and with the other-task FV ablated.
Task bootstrap: 10,000 resamples, seed 20260923, percentile 95% intervals (conditional on fitted vectors and settings).

Writes, per model, <bucket>/mediation/{t24_per_task.csv, t24_summary.csv, t25_per_task.csv, t25_summary.csv} with
bucket = results/69_task_run/read_write_relationship (GPT-J) or results/qwen25_fv/read_write_relationship (Qwen).
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
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR, QWEN25_FV_DIR  # noqa: E402

MODELS = {"gptj": (ARTIFACTS_ROOT / "69_task_run" / "mediation", 13, TASK69_RUN_DIR / "read_write_relationship" / "mediation", 69),
          "qwen25": (ARTIFACTS_ROOT / "qwen25_fv" / "mediation", 24, QWEN25_FV_DIR / "read_write_relationship" / "mediation", 96)}
RNG = np.random.default_rng(20260923)


def ci(x):
    ix = RNG.integers(0, len(x), (10000, len(x)))
    m = np.asarray(x)[ix].mean(1)
    return float(np.mean(x)), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def main():
    for model, (art, L, out, n_expected) in MODELS.items():
        out.mkdir(parents=True, exist_ok=True)
        R = [torch.load(f, weights_only=False) for f in sorted(glob.glob(str(art / "ablate_readcue" / "n6shot" / "*.pt")))]
        assert len(R) == n_expected, (model, "t24", len(R))
        C = R[0]["conditions"]
        rows = []
        for r in R:
            row = {"task": r["task"], "group": r["group"], "cf_task": r["cf_task"]}
            for i, c in enumerate(C):
                row[f"cos_task_{c}"] = float(r["cos_task"][i, :, L].mean()); row[f"cos_gen_{c}"] = float(r["cos_gen"][i, :, L].mean())
            rows.append(row)
        d = pd.DataFrame(rows); d.round(4).to_csv(out / "t24_per_task.csv", index=False)
        s = []
        for c in C:
            s.append({"quantity": f"cos_task_{c}", **dict(zip(("mean", "lo", "hi"), ci(d[f"cos_task_{c}"])))})
            s.append({"quantity": f"cos_gen_{c}", **dict(zip(("mean", "lo", "hi"), ci(d[f"cos_gen_{c}"])))})
        for op in ("mean", "zero"):
            own = d[f"cos_task_own_{op}"] - d["cos_task_none"]; cf = d[f"cos_task_cf_{op}"] - d["cos_task_none"]
            s += [{"quantity": f"dcos_own_{op}", **dict(zip(("mean", "lo", "hi"), ci(own)))},
                  {"quantity": f"dcos_cf_{op}", **dict(zip(("mean", "lo", "hi"), ci(cf)))},
                  {"quantity": f"dcos_own_minus_cf_{op}", **dict(zip(("mean", "lo", "hi"), ci(own - cf)))},
                  {"quantity": f"n_tasks_own_below_cf_{op}", "mean": int((own < cf).sum()), "lo": np.nan, "hi": np.nan}]
        s.append({"quantity": "readout_block", "mean": L, "lo": np.nan, "hi": np.nan})
        pd.DataFrame(s).round(4).to_csv(out / "t24_summary.csv", index=False)

        T = [json.load(open(f)) for f in sorted(glob.glob(str(art / "steer_fvablate" / "*.json")))]
        assert len(T) == n_expected, (model, "t25", len(T))
        conds = ["unsteered", "steer", "steer_own_zero", "steer_cf_zero"]
        t = pd.DataFrame([{"task": r["task"], "group": r["group"], "cf_task": r["cf_task"], "cos_own_cf_fv": r["cos_own_cf"],
                           **{c: r["conditions"][c]["acc"] for c in conds}} for r in T])
        t.round(4).to_csv(out / "t25_per_task.csv", index=False)
        gain = t.steer - t.unsteered; own = t.steer - t.steer_own_zero; cf = t.steer - t.steer_cf_zero
        s = [{"quantity": c, **dict(zip(("mean", "lo", "hi"), ci(t[c])))} for c in conds]
        s += [{"quantity": "steering_gain", **dict(zip(("mean", "lo", "hi"), ci(gain)))},
              {"quantity": "drop_own_fv_ablation", **dict(zip(("mean", "lo", "hi"), ci(own)))},
              {"quantity": "drop_cf_fv_ablation", **dict(zip(("mean", "lo", "hi"), ci(cf)))},
              {"quantity": "drop_own_minus_cf", **dict(zip(("mean", "lo", "hi"), ci(own - cf)))},
              {"quantity": "frac_gain_removed_own", "mean": float(own.mean() / gain.mean()), "lo": np.nan, "hi": np.nan},
              {"quantity": "frac_gain_removed_cf", "mean": float(cf.mean() / gain.mean()), "lo": np.nan, "hi": np.nan},
              {"quantity": "n_tasks_own_drop_gt_cf", "mean": int((own > cf).sum()), "lo": np.nan, "hi": np.nan},
              {"quantity": "n_tasks", "mean": len(t), "lo": np.nan, "hi": np.nan}]
        pd.DataFrame(s).round(4).to_csv(out / "t25_summary.csv", index=False)
        print(f"== {model}\n", pd.read_csv(out / "t24_summary.csv").to_string(index=False), "\n",
              pd.read_csv(out / "t25_summary.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
