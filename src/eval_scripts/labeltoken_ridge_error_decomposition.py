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
train_tasks, test_tasks = sorted(split["train_tasks"]), sorted(split["heldout_tasks"])
ALPHA = 1000.0  # CV-chosen for the avg variant

def load(t):
    a = torch.load(ACTS / f"{t}.pt", map_location="cpu", weights_only=False)["acts"].double()
    y = torch.load(FVS / f"{t}.pt", map_location="cpu", weights_only=False)["fv"].double()
    return a.mean(dim=1), y

Xtr = []; Ytr = []
for t in train_tasks:
    x, y = load(t); Xtr.append(x); Ytr.append(y)
Xtr = torch.cat(Xtr); Ytr = torch.cat(Ytr)
xbar, ybar, ev, evec, c = ridge_eig_prep(Xtr, Ytr)

def r2u(y, p):
    resid = ((y - p) ** 2).sum(0); tot = ((y - y.mean(0)) ** 2).sum(0)
    ok = tot > 0
    return float((1 - resid[ok] / tot[ok]).mean())

Yte, Pte, tslice = [], [], {}
pos = 0
for t in test_tasks:
    x, y = load(t)
    p = ridge_predict(x, xbar, ybar, ev, evec, c, ALPHA)
    Yte.append(y); Pte.append(p); tslice[t] = (pos, pos + len(y)); pos += len(y)
Yte = torch.cat(Yte); Pte = torch.cat(Pte)
print(f"pooled held-out R2 (avg X, alpha={ALPHA:g}): {r2u(Yte, Pte):.4f}")

# --- between-task: centroids only (each task one point, 14 points) ---
Ym = torch.stack([Yte[s:e].mean(0) for s, e in tslice.values()])
Pm = torch.stack([Pte[s:e].mean(0) for s, e in tslice.values()])
print(f"between-task (14 centroids) R2: {r2u(Ym, Pm):.4f}")

# --- within-task: deviations from each task's own centroid ---
Yd = torch.cat([Yte[s:e] - Yte[s:e].mean(0) for s, e in tslice.values()])
Pd = torch.cat([Pte[s:e] - Pte[s:e].mean(0) for s, e in tslice.values()])
resid = ((Yd - Pd) ** 2).sum(); tot = (Yd ** 2).sum()
print(f"within-task (deviations) R2 pooled: {float(1 - resid / tot):.4f}")

# --- oracle recentring: shift each task's prediction cloud onto the true centroid ---
Prec = torch.cat([Pte[s:e] - Pte[s:e].mean(0) + Yte[s:e].mean(0)
                  for s, e in tslice.values()])
print(f"held-out R2 after ORACLE recentring: {r2u(Yte, Prec):.4f}  "
      f"(oracle task-mean ceiling was 0.675)")

# --- centroid geometry: how far off is each predicted centroid? ---
grand = Ym.mean(0)
for t, (s, e) in tslice.items():
    ym, pm = Yte[s:e].mean(0), Pte[s:e].mean(0)
    off = float((pm - ym).norm()); spread = float((ym - grand).norm())
    cos = float(torch.nn.functional.cosine_similarity(pm - grand, ym - grand, dim=0))
    print(f"  {t:26s} ||pred_c - true_c||={off:6.1f}  ||true_c - grand||={spread:6.1f}  "
          f"cos(dirs from grand)={cos:.3f}")
