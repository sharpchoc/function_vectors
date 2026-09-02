#!/usr/bin/env python
"""Per-head mean outputs at style-property EVIDENCE tokens, per polarity.

Repo convention (capture_label_head_means): store the 256-dim out_proj INPUTS per head;
lift into residual space later with the head's W_O slice
(isolation_upper_bound.run_task.build_contributions_single).

Output: artifacts/style_properties/head_means/<prop>.pt
  {head_mean_nat, head_mean_alt, head_diff (28,16,256) fp32, n_pos per polarity}
"""
import argparse
import json
import sys
from pathlib import Path

import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from baukit import TraceDict
from src.utils.model_utils import load_gpt_model_and_tokenizer
from src.utils.varicl_utils import split_activations_by_head
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT

PROPS_DIR = REPO_ROOT / "dataset_files" / "style_properties" / "props"
POOL_PATH = REPO_ROOT / "task_splits" / "style_properties_pool.json"
OUT_DIR = ARTIFACTS_ROOT / "style_properties" / "head_means"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--props", nargs="*", default=None)
    p.add_argument("--model_name", default="EleutherAI/gpt-j-6b")
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    props = args.props or sorted(json.load(open(POOL_PATH))["pass"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer, mc = load_gpt_model_and_tokenizer(args.model_name)
    n_layers, n_heads = mc["n_layers"], mc["n_heads"]
    head_dim = mc["resid_dim"] // n_heads
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    for name in props:
        out = OUT_DIR / f"{name}.pt"
        if out.exists():
            print(f"{name}: exists, skip", flush=True)
            continue
        data = json.load(open(PROPS_DIR / f"{name}.json"))
        res = {}
        for pol in ("nat", "alt"):
            head_sum = torch.zeros(n_layers, n_heads, head_dim, dtype=torch.float64)
            n_pos = 0
            docs = data["docs"]
            for b0 in range(0, len(docs), args.batch_size):
                chunk = docs[b0:b0 + args.batch_size]
                idlists = [tokenizer(d[f"text_{pol}"]).input_ids for d in chunk]
                L = max(len(x) for x in idlists)
                ids = torch.full((len(chunk), L), tokenizer.eos_token_id, dtype=torch.long)
                att = torch.zeros(len(chunk), L, dtype=torch.long)
                mask = torch.zeros(len(chunk), L, dtype=torch.bool)
                for r, (d, x) in enumerate(zip(chunk, idlists)):
                    ids[r, :len(x)] = torch.tensor(x)
                    att[r, :len(x)] = 1
                    for s in d["sites"]:
                        e0, e1 = s["evid_idx"][pol]
                        mask[r, e0:e1 + 1] = True
                mask_d = mask.to(model.device)
                with torch.no_grad(), TraceDict(model, layers=mc["attn_hook_names"],
                                                retain_input=True) as td:
                    model(input_ids=ids.to(model.device),
                          attention_mask=att.to(model.device))
                for li, lname in enumerate(mc["attn_hook_names"]):
                    inp = td[lname].input
                    inp = inp[0] if isinstance(inp, tuple) else inp
                    heads = split_activations_by_head(inp, mc)      # (B, T, H, hd)
                    head_sum[li] += heads[mask_d].double().sum(0).cpu()
                n_pos += int(mask.sum())
            res[pol] = (head_sum / max(n_pos, 1)).float()
            res[f"n_pos_{pol}"] = n_pos
        torch.save({"property": name, "head_mean_nat": res["nat"],
                    "head_mean_alt": res["alt"],
                    "head_diff": res["alt"] - res["nat"],
                    "n_pos_nat": res["n_pos_nat"], "n_pos_alt": res["n_pos_alt"],
                    "site": "all evidence tokens (all sites, both twins separately)"}, out)
        print(f"{name}: nat_pos={res['n_pos_nat']} alt_pos={res['n_pos_alt']}", flush=True)
    print("head capture done", flush=True)


if __name__ == "__main__":
    main()
