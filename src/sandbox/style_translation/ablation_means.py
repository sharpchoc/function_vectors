#!/usr/bin/env python
"""Write-feature ablation set-up (CPU, 2026-09-23): for every pool family A and EVERY layer l = 1..28 the unit write direction
ŵ_A^l = unit(mean_nat − mean_alt) (steering/vectors_k3_train) and the mean-ablation value m_A^l = corpus-wide mean projection of the cue
residual onto ŵ_A^l — the prompt-count-weighted mean over every pool family's natural- and alternative-pole cue means (the correct k = 3/4
training prompts behind vectors_k3_train); at L24 the activation-store mean over all k = 3/4 prompts is stored alongside for comparison.
Plus a seeded counterfactual partner per family (another pool family, any language).  -> artifacts/.../ablation/{directions,means}_all.npz/json"""
import argparse, json, sys, zlib
from pathlib import Path
import numpy as np
_BOOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(_BOOT))
from src.sandbox.style_translation.models import paths as model_paths


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--pool", default="results/code_styles/code_pool.json"); ap.add_argument("--feature", default="write", choices=["write", "read"]); a = ap.parse_args()
    if a.feature == "read":
        return read_means(a)
    MP = model_paths(a.model); pool = json.load(open(_BOOT / a.pool))["pool"]; L = a.layer
    V = {f: np.load(MP["steering"] / "vectors_k3_train" / f"{f}.npz") for f in pool}; NL = V[pool[0]]["v_nat"].shape[0]
    W = {f: (V[f]["v_nat"].astype(np.float64) / np.linalg.norm(V[f]["v_nat"].astype(np.float64), axis=1, keepdims=True)) for f in pool}   # [NL, D] unit per layer
    n_nat = {f: int(V[f]["n_nat"]) for f in pool}; n_alt = {f: int(V[f]["n_alt"]) for f in pool}; N = sum(n_nat.values()) + sum(n_alt.values())
    corpus_mean = sum(n_nat[f] * V[f]["mean_nat"].astype(np.float64) + n_alt[f] * V[f]["mean_alt"].astype(np.float64) for f in pool) / N   # [NL, D]
    store = {f: np.load(MP["prompt_pairs"] / f"{f}.npz", allow_pickle=True) for f in pool}
    H24 = np.concatenate([store[f]["write"].astype(np.float32) for f in pool]); print(f"vectors population: {N} correct training prompts; activation store (L24 check): {len(H24)} prompts", flush=True)
    out = {}; means = np.zeros((len(pool), NL)); dirs = {}
    for i, f in enumerate(pool):
        m_l = np.einsum("ld,ld->l", corpus_mean, W[f]); means[i] = m_l; dirs[f] = W[f].astype(np.float32)
        w24 = W[f][L - 1].astype(np.float32); z = store[f]; own = z["write"].astype(np.float32) @ w24; pole = z["pole"]
        rng = np.random.default_rng(zlib.crc32(f"ablation|{f}".encode())); others = [g for g in pool if g != f]; cf = others[rng.integers(len(others))]
        out[f] = dict(mean_all_L24_vectors=float(m_l[L - 1]), mean_all_L24_store=float((H24 @ w24).mean()), mean_own_nat_L24=float(own[pole == "nat"].mean()), mean_own_alt_L24=float(own[pole == "alt"].mean()),
                      norm_v_L24=float(np.linalg.norm(V[f]["v_nat"][L - 1])), cf_family=cf, mean_by_layer=[float(x) for x in m_l])
    D = MP["steering"].parent / "ablation"; D.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(D / "directions_all.npz", **dirs)
    json.dump({"layers": list(range(1, NL + 1)), "population": f"{N} correct k=3/4 training prompts of {len(pool)} pool families, both poles", "families": out}, open(D / "means_all.json", "w"), indent=1)
    import pandas as pd; d = pd.DataFrame(out).T
    print("L24 mean-ablation value, vectors-based vs activation-store: mean abs diff %.2f (values span %.0f..%.0f)" % ((d.mean_all_L24_vectors - d.mean_all_L24_store).abs().mean(), d.mean_all_L24_store.min(), d.mean_all_L24_store.max()))
    print("L24 value lies between the family's own two pole means in", int(((d.mean_all_L24_vectors > d[["mean_own_nat_L24", "mean_own_alt_L24"]].min(axis=1)) & (d.mean_all_L24_vectors < d[["mean_own_nat_L24", "mean_own_alt_L24"]].max(axis=1))).sum()), "of", len(d), "families")
    print("->", D / "means_all.json")


def read_means(a):
    """READ feature (2026-09-23, user decision): per family and layer l = 0..27 the unit read direction ŵ_A^l = unit(mean_nat − mean_alt) of
    read_features/vectors_k3_train (evidence-token means; L0 = embedding output), and m_A^l = corpus-wide mean projection of the evidence-token
    activations = prompt-count-weighted mean over every pool family's read mean_nat / mean_alt, projected on ŵ_A^l. cf partner = the write
    ablation's (means_all.json). -> ablation/directions_read_all.npz ([28, D], index = layer) and means_read_all.json"""
    MP = model_paths(a.model); pool = json.load(open(_BOOT / a.pool))["pool"]; D = MP["steering"].parent / "ablation"
    cf = {f: v["cf_family"] for f, v in json.load(open(D / "means_all.json"))["families"].items()}
    V = {f: np.load(MP["read_features"] / "vectors_k3_train" / f"{f}.npz") for f in pool}
    def stack(z, pole):   # [28, D]: index 0 = embedding output, l = block l output
        return np.concatenate([z[f"mean_{pole}_L0"][None].astype(np.float64), z[f"mean_{pole}"][:27].astype(np.float64)])
    n_nat = {f: int(V[f]["n_nat"]) for f in pool}; n_alt = {f: int(V[f]["n_alt"]) for f in pool}; N = sum(n_nat.values()) + sum(n_alt.values())
    corpus = sum(n_nat[f] * stack(V[f], "nat") + n_alt[f] * stack(V[f], "alt") for f in pool) / N
    out = {}; dirs = {}
    for f in pool:
        d = stack(V[f], "nat") - stack(V[f], "alt"); W = d / np.linalg.norm(d, axis=1, keepdims=True); dirs[f] = W.astype(np.float32)
        m = np.einsum("ld,ld->l", corpus, W); own_nat = np.einsum("ld,ld->l", stack(V[f], "nat"), W); own_alt = np.einsum("ld,ld->l", stack(V[f], "alt"), W)
        out[f] = dict(cf_family=cf[f], mean_by_layer=[float(x) for x in m], mean_own_nat_L8=float(own_nat[8]), mean_own_alt_L8=float(own_alt[8]), mean_all_L8=float(m[8]), norm_diff_L8=float(np.linalg.norm(d[8])))
    np.savez_compressed(D / "directions_read_all.npz", **dirs)
    json.dump({"layers": list(range(28)), "layer_index": "0 = embedding output, l = output of block l", "population": f"{N} correct k=3/4 training prompts of {len(pool)} pool families, both poles (evidence-token means)", "families": out}, open(D / "means_read_all.json", "w"), indent=1)
    import pandas as pd; t = pd.DataFrame(out).T
    print(f"read population {N} prompts; L8 corpus-mean projection lies between the family's own pole means in", int(((t.mean_all_L8 > t[["mean_own_nat_L8", "mean_own_alt_L8"]].min(axis=1)) & (t.mean_all_L8 < t[["mean_own_nat_L8", "mean_own_alt_L8"]].max(axis=1))).sum()), "of", len(t), "families")
    print(t[["mean_all_L8", "mean_own_nat_L8", "mean_own_alt_L8", "norm_diff_L8"]].astype(float).describe().loc[["mean", "min", "max"]].round(2).to_string()); print("->", D / "means_read_all.json")


if __name__ == "__main__":
    main()
