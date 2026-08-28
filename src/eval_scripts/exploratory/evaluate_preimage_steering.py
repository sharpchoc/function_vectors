#!/usr/bin/env python
"""Stage-2 of the pre-image steering experiment (Stream R): the causal test.

For each task and each edit layer l, injects a LAYER-SPECIFIC steering vector (the ridge
pre-image dx_l computed by fit_prelabel_ridge_preimages.py) at the last token of the prompt
(block-output residual add, same mechanism as FV steering) and measures first-token top-1
accuracy under the two standard regimes: zero-shot + vector, 10-shot shuffled-labels + vector.

Arms:
  preimage_raw          inject dx_l as-is (THE pre-image, per user spec)
  preimage_normmatched  inject dx_l * (||fv|| / ||dx_l||)  (controls for norm blow-ups)
  fv_direct             inject the task's train_selected_top40 FV (same vector all layers)

Also computes per task the two no-intervention baselines (zero-shot, 10-shot shuffled) once.

Mirrors evaluate_heldout_multitask_head_fvs.py (filter sets, seeds, summaries); reuses
n_shot_eval / get-filter-set logic. Designed to be sharded one task per process (--tasks X).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import src.utils.eval_utils as eval_utils
from src.utils.eval_utils import n_shot_eval, n_shot_eval_no_intervention
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.prompt_utils import load_dataset
from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR

# Workaround for an upstream bug in n_shot_eval's generate_str path: target is wrapped in a
# list before get_answer_id(query + answer) is called, which crashes on str + list.
_orig_get_answer_id = eval_utils.get_answer_id


def _get_answer_id_listsafe(query, answer, tokenizer):
    if isinstance(answer, list):
        answer = answer[0]
    return _orig_get_answer_id(query, answer, tokenizer)


eval_utils.get_answer_id = _get_answer_id_listsafe

DEFAULT_TASKS = ["next_number", "prev_number", "synonym", "antonym"]
ARMS = ["preimage_damped", "preimage_normmatched", "preimage_raw", "fv_direct"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    p.add_argument("--preimage_root", type=Path,
                   default=ARTIFACTS_ROOT / "preimage_steering/train_selected_top40_icl10_pre")
    p.add_argument("--fv_root", type=Path,
                   default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_selected_top40")
    p.add_argument("--filter_root", type=Path, default=ARTIFACTS_ROOT / "gptj_fv",
                   help="Root of cached fs_results_layer_sweep.json (ICL-correct filter source).")
    p.add_argument("--output_root", type=Path, default=FV_FORMATION_DIR / "preimage_analysis/preimage_steering")
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_split", type=float, default=0.3)
    p.add_argument("--n_shots", type=int, default=10)
    p.add_argument("--max_eval_examples", type=int, default=None,
                   help="Seeded subsample of the (filtered) test queries, for wall-clock control.")
    p.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    p.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    p.add_argument("--filter_to_correct_icl", dest="filter_to_correct_icl", action="store_true")
    p.add_argument("--no_filter_to_correct_icl", dest="filter_to_correct_icl", action="store_false")
    p.set_defaults(filter_to_correct_icl=True)
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="Edit layers to sweep (default: all 0..n_layers-1). For smokes.")
    p.add_argument("--exact_match_best_layer", action="store_true",
                   help="After the sweep, rerun each arm's best layer with generate_str + "
                        "exact_match_score (full multi-token answer; for number-word tasks).")
    p.add_argument("--reuse_arm_results", action="store_true",
                   help="Load existing {arm}_*_results.json instead of re-running the sweeps "
                        "(for finishing a run whose summary stage failed).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def torch_load_trusted(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def write_json(path, data):
    def safe(v):
        if isinstance(v, dict):
            return {str(k): safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [safe(x) for x in v]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().tolist()
        return v
    with open(path, "w") as f:
        json.dump(safe(data), f, indent=2)


def get_filter_set(args, task, dataset, model, model_config, tokenizer, output_dir):
    """ICL-correct filter (cached source) + optional seeded subsample cap."""
    if not args.filter_to_correct_icl:
        filter_set = np.arange(len(dataset["test"]["input"]))
        source = None
    else:
        existing = args.filter_root / task / "fs_results_layer_sweep.json"
        if existing.exists():
            with open(existing) as f:
                fs_results = json.load(f)
            source = str(existing)
        else:
            set_seed(args.seed)
            fs_results = n_shot_eval_no_intervention(
                dataset=dataset, n_shots=args.n_shots, model=model, model_config=model_config,
                tokenizer=tokenizer, compute_ppl=True, prefixes=args.prefixes,
                separators=args.separators)
            out = output_dir / "fs_results_filter_source.json"
            write_json(out, fs_results)
            source = str(out)
        filter_set = np.where(np.array(fs_results["clean_rank_list"]) == 0)[0]
    if args.max_eval_examples is not None and len(filter_set) > args.max_eval_examples:
        rng = np.random.default_rng(args.seed)
        filter_set = np.sort(rng.choice(filter_set, size=args.max_eval_examples, replace=False))
    return filter_set, source


def load_preimage_bank(preimage_root, task):
    path = preimage_root / "preimages" / f"{task}_preimage_bank.pt"
    data = torch_load_trusted(path, map_location="cpu")
    bank = {}
    for k, entry in data["preimages_by_edit_layer"].items():
        bank[int(k)] = {
            "exact": entry["exact"].detach().float().reshape(-1),
            "damped": entry["damped"].detach().float().reshape(-1),
            "damped_gamma": entry.get("damped_gamma"),
            "damped_rel_residual": entry.get("damped_rel_residual"),
        }
    return bank


def load_fv(fv_root, task):
    data = torch_load_trusted(fv_root / task / f"{task}_function_vector.pt", map_location="cpu")
    fv = data["function_vector"] if isinstance(data, dict) else data
    return fv.detach().float().cpu().reshape(-1)


def vector_for(arm, layer, bank, fv):
    if arm == "fv_direct":
        return fv
    entry = bank.get(layer)
    if entry is None:
        return None  # edit layer with no pre-image (none expected for 0..27, but be safe)
    if arm == "preimage_raw":
        return entry["exact"]
    if arm == "preimage_damped":
        return entry["damped"]
    dx = entry["exact"]
    return dx * (fv.norm() / dx.norm().clamp_min(1e-12))


def run_sweep(args, dataset, filter_set, arm, bank, fv, model, model_config, tokenizer,
              generate_str=False, metric=None, layers=None):
    zs, fs = {}, {}
    layers = layers if layers is not None else range(model_config["n_layers"])
    for layer in layers:
        vec = vector_for(arm, int(layer), bank, fv)
        if vec is None:
            continue
        common = dict(dataset=dataset, fv_vector=vec, edit_layer=int(layer), model=model,
                      model_config=model_config, tokenizer=tokenizer, filter_set=filter_set,
                      prefixes=args.prefixes, separators=args.separators,
                      generate_str=generate_str, metric=metric or "f1_score")
        set_seed(args.seed)
        zs[int(layer)] = n_shot_eval(n_shots=0, **common)
        set_seed(args.seed)
        fs[int(layer)] = n_shot_eval(n_shots=args.n_shots, shuffle_labels=True, **common)
        print(f"  [{arm}] L{layer:02d} zs={top1(zs[int(layer)])} fs_shuf={top1(fs[int(layer)])}", flush=True)
    return zs, fs


def top1(result):
    if "intervention_topk" in result:
        return float(result["intervention_topk"][0][1])
    if "intervention_score" in result:
        return float(np.mean(result["intervention_score"]))
    return None


def summarize(zs, fs):
    zs_by = {str(l): top1(r) for l, r in zs.items()}
    fs_by = {str(l): top1(r) for l, r in fs.items()}
    best_zs = max(zs_by, key=zs_by.get) if zs_by else None
    best_fs = max(fs_by, key=fs_by.get) if fs_by else None
    return {
        "zs_intervention_top1_by_layer": zs_by,
        "fs_shuffled_intervention_top1_by_layer": fs_by,
        "best_zs_layer": None if best_zs is None else int(best_zs),
        "best_zs_intervention_top1": None if best_zs is None else zs_by[best_zs],
        "best_fs_shuffled_layer": None if best_fs is None else int(best_fs),
        "best_fs_shuffled_intervention_top1": None if best_fs is None else fs_by[best_fs],
    }


def plot_task(task, summaries, baselines, out_path):
    series = [("Zero-shot + vector", "zs_intervention_top1_by_layer", "zs_top1"),
              ("10-shot shuffled + vector", "fs_shuffled_intervention_top1_by_layer", "fs_shuffled_top1")]
    styles = {"fv_direct": dict(marker="o", color="tab:blue", label="FV direct"),
              "preimage_damped": dict(marker="D", color="tab:green", label="Pre-image (damped)"),
              "preimage_raw": dict(marker="s", color="tab:red", label="Pre-image (exact raw)"),
              "preimage_normmatched": dict(marker="^", color="tab:orange", label="Pre-image (exact, norm-matched)")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (title, key, bkey) in zip(axes, series):
        for arm, summ in summaries.items():
            by = {int(l): v for l, v in summ[key].items()}
            layers = sorted(by)
            ax.plot(layers, [by[l] for l in layers], **styles[arm])
        if baselines.get(bkey) is not None:
            ax.axhline(baselines[bkey], color="gray", ls="--", lw=1, label="No-intervention baseline")
        ax.set_title(title)
        ax.set_xlabel("Edit layer")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Intervention top-1 accuracy (first token)")
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(f"{task} — ridge pre-image steering vs FV steering")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    print(f"Loading model ({args.model_name})")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision)
    model.eval()

    for task in args.tasks:
        print(f"\n=== {task} ===", flush=True)
        out_dir = args.output_root / task
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "summary.json"
        if summary_path.exists() and not args.overwrite:
            raise FileExistsError(f"{summary_path} exists; pass --overwrite.")

        set_seed(args.seed)
        dataset = load_dataset(task, root_data_dir=args.root_data_dir,
                               test_size=args.test_split, seed=args.seed)
        bank = load_preimage_bank(args.preimage_root, task)
        fv = load_fv(args.fv_root, task)
        filter_set, filter_source = get_filter_set(args, task, dataset, model, model_config,
                                                   tokenizer, out_dir)
        print(f"  filter: {len(filter_set)} test queries (source={filter_source})", flush=True)

        # No-intervention baselines, once per task (full test split; restricted to the same
        # filter set post-hoc via clean_rank_list, since the helper takes no filter_set).
        set_seed(args.seed)
        zs_base = n_shot_eval_no_intervention(
            dataset=dataset, n_shots=0, model=model, model_config=model_config,
            tokenizer=tokenizer, compute_ppl=False,
            prefixes=args.prefixes, separators=args.separators)
        set_seed(args.seed)
        fs_base = n_shot_eval_no_intervention(
            dataset=dataset, n_shots=args.n_shots, model=model, model_config=model_config,
            tokenizer=tokenizer, compute_ppl=False, shuffle_labels=True,
            prefixes=args.prefixes, separators=args.separators)
        baselines = {"zs_top1": _filtered_base_top1(zs_base, filter_set),
                     "fs_shuffled_top1": _filtered_base_top1(fs_base, filter_set)}
        print(f"  baselines: zs={baselines['zs_top1']} fs_shuffled={baselines['fs_shuffled_top1']}", flush=True)

        summaries = {}
        for arm in args.arms:
            zs_path = out_dir / f"{arm}_zs_results.json"
            fs_path = out_dir / f"{arm}_fs_shuffled_results.json"
            if args.reuse_arm_results and zs_path.exists() and fs_path.exists():
                print(f" arm={arm} (reusing saved results)", flush=True)
                with open(zs_path) as f:
                    zs = {int(k): v for k, v in json.load(f).items()}
                with open(fs_path) as f:
                    fs = {int(k): v for k, v in json.load(f).items()}
            else:
                print(f" arm={arm}", flush=True)
                zs, fs = run_sweep(args, dataset, filter_set, arm, bank, fv, model, model_config,
                                   tokenizer, layers=args.layers)
                write_json(zs_path, zs)
                write_json(fs_path, fs)
            summaries[arm] = summarize(zs, fs)

        exact_match = None
        if args.exact_match_best_layer:
            exact_match = {}
            for arm, summ in summaries.items():
                exact_match[arm] = {}
                for regime, best_key in (("zs", "best_zs_layer"), ("fs_shuffled", "best_fs_shuffled_layer")):
                    layer = summ[best_key]
                    if layer is None:
                        continue
                    vec = vector_for(arm, int(layer), bank, fv)
                    set_seed(args.seed)
                    res = n_shot_eval(dataset=dataset, fv_vector=vec, edit_layer=int(layer),
                                      n_shots=0 if regime == "zs" else args.n_shots,
                                      shuffle_labels=(regime != "zs"), model=model,
                                      model_config=model_config, tokenizer=tokenizer,
                                      filter_set=filter_set, prefixes=args.prefixes,
                                      separators=args.separators, generate_str=True,
                                      metric="exact_match_score")
                    exact_match[arm][regime] = {"layer": int(layer), "exact_match": top1(res)}
            write_json(out_dir / "exact_match_best_layer.json", exact_match)

        norms = {str(l): {"exact": float(e["exact"].norm()), "damped": float(e["damped"].norm()),
                          "damped_gamma": e["damped_gamma"],
                          "damped_rel_residual": e["damped_rel_residual"]}
                 for l, e in sorted(bank.items())}
        summary = {
            "task": task,
            "fv_root": str(args.fv_root), "preimage_root": str(args.preimage_root),
            "fv_norm": float(fv.norm()), "preimage_norms_by_edit_layer": norms,
            "n_filtered_test_examples": int(len(filter_set)),
            "max_eval_examples": args.max_eval_examples,
            "filter_source": filter_source,
            "baselines_no_intervention": baselines,
            "arms": summaries,
            "exact_match_best_layer": exact_match,
        }
        write_json(summary_path, summary)
        plot_task(task, summaries, baselines, out_dir / f"{task}_preimage_vs_fv_steering.png")
        print(f"  wrote {summary_path}", flush=True)


def _filtered_base_top1(result, filter_set):
    ranks = np.array(result["clean_rank_list"])
    if filter_set is not None:
        ranks = ranks[np.asarray(filter_set, dtype=int)]
    return float(np.mean(ranks == 0)) if len(ranks) else None


if __name__ == "__main__":
    main()
