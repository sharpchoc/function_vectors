import json, torch
from pathlib import Path

FV = Path("/workspace/function_vectors/artifacts/69_task_run/perprompt_fvs")
split = json.load(open("/workspace/function_vectors/task_splits/extended_steerable_69_prunedfail.json"))

def r2_uniform(y, pred):
    resid = ((y - pred) ** 2).sum(dim=0)
    tot = ((y - y.mean(dim=0)) ** 2).sum(dim=0)
    ok = tot > 0
    return float((1 - resid[ok] / tot[ok]).mean()), float(1 - resid.sum() / tot.sum())

for name, tasks in (("heldout(14)", sorted(split["heldout_tasks"])),
                    ("train(55)", sorted(split["train_tasks"]))):
    Y, P = [], []
    for t in tasks:
        y = torch.load(FV / f"{t}.pt", map_location="cpu", weights_only=False)["fv"].double()
        n = len(y)
        s = y.sum(dim=0, keepdim=True)
        loo_mean = (s - y) / (n - 1)      # leave-one-out task mean per row
        Y.append(y); P.append(loo_mean)
    Y = torch.cat(Y); P = torch.cat(P)
    u, w = r2_uniform(Y, P)
    print(f"{name}: ORACLE LOO task-mean predictor  R2 uniform={u:.4f}  weighted={w:.4f}  "
          f"({len(Y)} pairs)")
