#!/usr/bin/env python
"""Patch the judge_ok / correct flags of prompt_pairs/<family>.npz from the (now judged) rollouts — for activation stores captured before
the judge ran (regen8 stage A, 2026-09-23). CPU only.   usage: fix_pair_flags.py --families f1 f2 ..."""
import argparse, json, sys
from pathlib import Path
import numpy as np
_BOOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(_BOOT))
from src.sandbox.style_translation.models import paths as model_paths


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--families", nargs="+", required=True); a = ap.parse_args()
    MP = model_paths(a.model)
    for fam in a.families:
        p = MP["prompt_pairs"] / f"{fam}.npz"; z = dict(np.load(p, allow_pickle=True))
        recs = json.load(open(MP["rollouts"] / f"{fam}.json"))
        fl = {(r["doc_id"], r["style"], int(r["k"])): (bool(r["style_ok"]), bool((r.get("judge") or {}).get("ok"))) for r in recs}
        unj = sum(1 for r in recs if r["k"] in (3, 4) and not r.get("judge"))
        so = np.array([fl.get((d, s, int(k)), (False, False))[0] for d, s, k in zip(z["doc_id"], z["pole"], z["k"])])
        jo = np.array([fl.get((d, s, int(k)), (False, False))[1] for d, s, k in zip(z["doc_id"], z["pole"], z["k"])])
        z["style_ok"], z["judge_ok"], z["correct"] = so, jo, so & jo
        np.savez_compressed(p, **z)
        print(f"{fam}: {len(so)} prompts, correct {int((so & jo).sum())} ({(so & jo).mean():.0%}); unjudged k=3/4 rollouts {unj}", flush=True)


if __name__ == "__main__":
    main()
