#!/usr/bin/env python
"""Stamp the GPT-4 judge top-1 verdict into the TWO-shot paired-capture activation metadata.

Two-demo sibling of tag_oneshot_activations_judge.py. Reads
direction2_label_geometry/twoshot_<task>_judge/judged_results.json for each task and writes a
`judge_top1` boolean into every matching activation row's metadata (all 5 roles) in
artifacts/twoshot_paired_graded/<pair>/shard_*.pt, plus into grading.json.
Match key = (function_task, label1, label2, query) — unique per prompt. In-place rewrite.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import ARTIFACTS_ROOT, LABEL_GEOMETRY_DIR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graded_dir", type=Path,
                   default=ARTIFACTS_ROOT / "twoshot_paired_graded" / "antonym_synonym")
    p.add_argument("--judge_root", type=Path, default=LABEL_GEOMETRY_DIR / "twoshot" / "judge")
    p.add_argument("--function_tasks", nargs="+", default=["antonym", "synonym"])
    p.add_argument("--tag_field", type=str, default="judge_top1")
    return p.parse_args()


def main():
    args = parse_args()
    verdict = {}
    for task in args.function_tasks:
        jr = json.loads((args.judge_root / task / "judged_results.json").read_text())
        for r in jr["records"]:
            verdict[(task, r["label1"], r["label2"], r["query_input"])] = bool(r["judge_correct"])
        print(f"{task}: {len(jr['records'])} verdicts loaded")

    def key_meta(m):
        return (m["function_task"], m["label1"], m["label2"], m["query_word"])

    n_rows = n_tagged = 0
    for sp in sorted(glob.glob(str(args.graded_dir / "shard_*.pt"))):
        data = torch.load(sp, map_location="cpu", weights_only=False)
        for m in data["metadata"]:
            n_rows += 1
            v = verdict.get(key_meta(m))
            if v is not None:
                m[args.tag_field] = v
                n_tagged += 1
        torch.save(data, sp)
    print(f"tagged {n_tagged}/{n_rows} activation rows with {args.tag_field} across shards")

    grading_path = args.graded_dir / "grading.json"
    grading = json.loads(grading_path.read_text())
    g_tagged = 0
    for g in grading:
        v = verdict.get((g["function_task"], g["label1"], g["label2"], g["query"]))
        if v is not None:
            g[args.tag_field] = v
            g_tagged += 1
    grading_path.write_text(json.dumps(grading, indent=2))
    print(f"tagged {g_tagged}/{len(grading)} grading.json rows with {args.tag_field}")

    for task in args.function_tasks:
        rows = [g for g in grading if g["function_task"] == task and args.tag_field in g]
        if rows:
            c = sum(g[args.tag_field] for g in rows)
            print(f"  {task}: {args.tag_field} True = {c}/{len(rows)} = {c/len(rows):.3f}")


if __name__ == "__main__":
    main()
