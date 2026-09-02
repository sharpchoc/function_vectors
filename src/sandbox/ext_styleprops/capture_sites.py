#!/usr/bin/env python
"""Stage B capture: residual-stream activations at tagged style-property sites.

One forward pass per (property, polarity, doc) with output_hidden_states=True; gathers
29 layers (index 0 = the token embedding — GPT-J adds no absolute position embeddings,
so layer 0 is the exact identity baseline; 1..28 = block outputs) at three site roles:
  evid — mean over each opportunity's evidence-token span
  cue  — the site's cue token (identity-matched across twins; id re-asserted here)
  bg   — identity-matched background tokens between manifestations (state probes)

Output (resumable per property × polarity):
  artifacts/style_properties/site_acts/<prop>__<pol>.npz
    acts [N, 29, 4096] fp16; role (0/1/2 = evid/cue/bg), doc, k, dist [N]; doc_ids
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT
from src.sandbox.ext_styleprops.prescreen_adherence import load_model

PROPS_DIR = REPO_ROOT / "dataset_files" / "style_properties" / "props"
POOL_PATH = REPO_ROOT / "task_splits" / "style_properties_pool.json"
ROLE = {"evid": 0, "cue": 1, "bg": 2}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--props", nargs="*", default=None,
                   help="default: the pool's passing properties")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "style_properties" / "site_acts")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--batch", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    props = args.props or json.load(open(POOL_PATH))["pass"]
    args.out_root.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(args.model_dir)

    for name in sorted(props):
        data = json.load(open(PROPS_DIR / f"{name}.json"))
        for pol in ("nat", "alt"):
            out_path = args.out_root / f"{name}__{pol}.npz"
            if out_path.exists():
                print(f"{name}/{pol}: exists, skip", flush=True)
                continue
            acts, meta = [], []
            docs = data["docs"]
            for b0 in range(0, len(docs), args.batch):
                chunk = docs[b0:b0 + args.batch]
                idlists = [tok(d[f"text_{pol}"]).input_ids for d in chunk]
                L = max(len(x) for x in idlists)
                ids = torch.full((len(chunk), L), tok.eos_token_id, dtype=torch.long)
                att = torch.zeros(len(chunk), L, dtype=torch.long)
                for r, x in enumerate(idlists):
                    ids[r, :len(x)] = torch.tensor(x)
                    att[r, :len(x)] = 1
                with torch.no_grad():
                    hs = model(input_ids=ids.cuda(), attention_mask=att.cuda(),
                               output_hidden_states=True).hidden_states
                hs = torch.stack(hs, dim=2)          # [B, T, 29, 4096]
                for r, d in enumerate(chunk):
                    di = b0 + r
                    for s in d["sites"]:
                        e0, e1 = s["evid_idx"][pol]
                        acts.append(hs[r, e0:e1 + 1].mean(0).half().cpu())
                        meta.append((ROLE["evid"], di, s["k"], s["dist"][pol]))
                        cue = s["cue_idx"][pol]
                        assert idlists[r][cue] == s["cue_tok_id"]
                        acts.append(hs[r, cue].half().cpu())
                        meta.append((ROLE["cue"], di, s["k"], s["dist"][pol]))
                    for g in d["bg"]:
                        j = g["tok_idx"][pol]
                        assert idlists[r][j] == g["tok_id"]
                        acts.append(hs[r, j].half().cpu())
                        meta.append((ROLE["bg"], di, g["k"], g["dist"][pol]))
                del hs
            m = np.array(meta, dtype=np.int32)
            np.savez(out_path, acts=torch.stack(acts).numpy(),
                     role=m[:, 0], doc=m[:, 1], k=m[:, 2], dist=m[:, 3],
                     doc_ids=np.array([d["doc_id"] for d in docs]))
            print(f"{name}/{pol}: {len(acts)} sites -> {out_path.name}", flush=True)
    print("capture done", flush=True)


if __name__ == "__main__":
    main()
