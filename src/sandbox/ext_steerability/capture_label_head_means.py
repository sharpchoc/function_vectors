#!/usr/bin/env python
"""Per-task mean head outputs at the LAST DEMO LABEL TOKEN of the fixed 10-shot prompts.

Companion capture for the label-slot head selection (read_vector_head_selection): the FV
convention takes head means at the final cue token; here we take them at the last token of
the 10th demonstration's label, because that is the site we steer (the ' _' dummy label of
the 1-shot scaffold).

Stored values are the 256-dim out_proj INPUTS per head (the repo convention; apply the
head's W_O slice to lift into residual space, e.g. via
src/sandbox/isolation_upper_bound/run_task.build_contributions_single).

Output: artifacts/69_task_run/label_head_means/<task>.pt
  {head_means (28,16,256) fp32, n_prompts, label_idx (150,), cue_idx (150,)}

Gates: exactly 150 label positions per task; the captured index must decode to the final
token of that prompt's 10th demo label and must differ from the cue index; per-layer
linearity check (sum of W_O slices @ head acts == attention output) on the first batch.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

# local bootstrap for in-repo runs; a PYTHONPATH-supplied repo also works (staged copies)
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from baukit import TraceDict
from src.utils.model_utils import load_gpt_model_and_tokenizer
from src.utils.prompt_utils import (create_prompt, get_token_meta_labels,
                                    word_pairs_to_prompt_data)
from src.utils.varicl_utils import split_activations_by_head
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT

LAST_LABEL_RE = re.compile(r"^demonstration_10_label_token$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_head_means")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    return p.parse_args()


def prompt_and_label_idx(rec, tokenizer, model_config=None):
    """Build the clean 10-shot prompt and return (prompt_string, last-demo-label index,
    cue index). Number tasks store int labels, which break tokenize_labels -> str-cast.

    NO bos token anywhere (prepend_bos_token=False / prepend_bos=False): GPT-J does not
    need one and every other pipeline in this study (steer_read_dir_*, ablate_pc50_*) builds
    prompts the same way, so token indices line up with a plain tokenizer(prompt_string)."""
    wp = {"input": [str(d["input"]) for d in rec["demos"]],
          "output": [str(d["output"]) for d in rec["demos"]]}
    qo = rec["query"]["output"]
    qo = [str(x) for x in qo] if isinstance(qo, list) else str(qo)
    q = {"input": str(rec["query"]["input"]), "output": qo}
    pd_ = word_pairs_to_prompt_data(wp, query_target_pair=q, prepend_bos_token=False,
                                    shuffle_labels=False)
    token_labels, prompt_string = get_token_meta_labels(
        pd_, tokenizer, query=q["input"], prepend_bos=False)
    ids = tokenizer(prompt_string).input_ids
    assert len(ids) == len(token_labels), "token/label length mismatch"
    idxs = [int(i) for i, _, lab in token_labels if LAST_LABEL_RE.match(lab)]
    assert idxs, "no demonstration_10_label_token found"
    return prompt_string, max(idxs), len(ids) - 1


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])[args.shard_idx::args.shard_n]
    print(f"{len(tasks)} tasks on this shard", flush=True)
    args.out_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    n_layers, n_heads = model_config["n_layers"], model_config["n_heads"]
    resid = model_config["resid_dim"]
    head_dim = resid // n_heads
    tokenizer.padding_side = "right"

    for task in tasks:
        out = args.out_root / f"{task}.pt"
        if out.exists():
            print(f"{task}: exists, skip", flush=True)
            continue
        recs = json.load(open(args.prompts_root / task / "train_prompts.json"))
        assert len(recs) == 150, f"{task}: {len(recs)} records"
        built = [prompt_and_label_idx(r, tokenizer) for r in recs]
        # gate: captured index decodes to the last token of the 10th demo's label
        for (ps, li, ci), rec in zip(built, recs):
            ids = tokenizer(ps).input_ids
            gold_last = tokenizer(" " + str(rec["demos"][9]["output"]).strip()).input_ids[-1]
            assert ids[li] == gold_last, f"{task}: label idx token mismatch at {li}"
            assert li != ci, f"{task}: label idx equals cue idx"

        head_sum = torch.zeros(n_layers, n_heads, head_dim, dtype=torch.float64)
        n_seen, checked = 0, False
        for start in range(0, len(built), args.batch_size):
            chunk = built[start:start + args.batch_size]
            sentences = [c[0] for c in chunk]
            label_idx = torch.tensor([c[1] for c in chunk], device=model.device)
            inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)
            bidx = torch.arange(len(chunk), device=model.device)
            with torch.no_grad(), TraceDict(model, layers=model_config["attn_hook_names"],
                                            retain_input=True, retain_output=True) as td:
                model(**inputs)
            for li_, lname in enumerate(model_config["attn_hook_names"]):
                inp = td[lname].input
                inp = inp[0] if isinstance(inp, tuple) else inp
                heads = split_activations_by_head(inp, model_config)   # (B, seq, H, hd)
                at_label = heads[bidx, label_idx]                      # (B, H, hd)
                head_sum[li_] += at_label.double().sum(dim=0).cpu()
                if not checked:
                    w = model.transformer.h[li_].attn.out_proj.weight.detach()
                    rebuilt = torch.einsum("bhd,ehd->be", at_label.to(w.dtype),
                                           w.view(resid, n_heads, head_dim))
                    outp = td[lname].output
                    outp = outp[0] if isinstance(outp, tuple) else outp
                    ref = outp[bidx, label_idx]
                    rel = (rebuilt - ref).norm() / ref.norm()
                    assert rel < 2e-2, f"{task} L{li_}: linearity gate {rel:.3e}"
            checked = True
            n_seen += len(chunk)
        assert n_seen == 150
        torch.save({"task": task, "head_means": (head_sum / n_seen).float(),
                    "n_prompts": n_seen,
                    "label_idx": torch.tensor([c[1] for c in built]),
                    "cue_idx": torch.tensor([c[2] for c in built]),
                    "site": "last token of the 10th demo label (clean 10-shot prompts)"},
                   out)
        print(f"{task}: captured {n_seen} prompts, label idx range "
              f"{min(c[1] for c in built)}-{max(c[1] for c in built)}", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
