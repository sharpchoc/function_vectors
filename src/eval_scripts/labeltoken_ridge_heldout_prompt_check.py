import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/workspace/function_vectors/.claude/worktrees/labeltoken-fv-ridge")
sys.path.insert(0, "/workspace/function_vectors/.claude/worktrees/labeltoken-fv-ridge/src")
from src.eval_scripts.regress_activation_to_fv_fulldim_ridge import ridge_eig_prep, ridge_predict

ACTS = Path("/workspace/function_vectors/artifacts/69_task_run/label_all10_L6_acts")
FVS = Path("/workspace/function_vectors/artifacts/69_task_run/perprompt_fvs")
split = json.load(open("/workspace/function_vectors/task_splits/extended_steerable_69_prunedfail.json"))
train_tasks = sorted(split["train_tasks"])
ALPHA = 1000.0
rng = np.random.RandomState(43)

Xf, Yf, Xe, Ye, esl = [], [], [], [], {}
pos = 0
for t in train_tasks:
    a = torch.load(ACTS / f"{t}.pt", map_location="cpu", weights_only=False)["acts"].double().mean(dim=1)
    y = torch.load(FVS / f"{t}.pt", map_location="cpu", weights_only=False)["fv"].double()
    perm = rng.permutation(150)
    fit, ev = perm[:120], perm[120:]
    Xf.append(a[fit]); Yf.append(y[fit]); Xe.append(a[ev]); Ye.append(y[ev])
    esl[t] = (pos, pos + len(ev)); pos += len(ev)
Xf, Yf, Xe, Ye = (torch.cat(v) for v in (Xf, Yf, Xe, Ye))

xbar, ybar, ev_, evec, c = ridge_eig_prep(Xf, Yf)
Pe = ridge_predict(Xe, xbar, ybar, ev_, evec, c, ALPHA)
Pf = ridge_predict(Xf, xbar, ybar, ev_, evec, c, ALPHA)

def r2u(y, p):
    resid = ((y - p) ** 2).sum(0); tot = ((y - y.mean(0)) ** 2).sum(0)
    ok = tot > 0
    return float((1 - resid[ok] / tot[ok]).mean())

print(f"in-sample (120/task fit rows):        R2 = {r2u(Yf, Pf):.4f}")
print(f"HELD-OUT PROMPTS of train tasks:      R2 = {r2u(Ye, Pe):.4f}")

# oracle on the same eval rows: task mean from the FIT rows (fair, no leakage)
Po = torch.cat([Yf[i*120:(i+1)*120].mean(0, keepdim=True).expand(30, -1)
                for i in range(len(train_tasks))])
print(f"oracle (task mean from fit rows) on the same eval rows: R2 = {r2u(Ye, Po):.4f}")

# within-task decomposition on eval rows
Yd = torch.cat([Ye[s:e] - Ye[s:e].mean(0) for s, e in esl.values()])
Pd = torch.cat([Pe[s:e] - Pe[s:e].mean(0) for s, e in esl.values()])
print(f"within-task deviations R2 (eval rows): {float(1 - ((Yd-Pd)**2).sum()/(Yd**2).sum()):.4f}")
