#!/usr/bin/env python
"""Stamp the GPT-4 judge top-1 verdict into the paired-capture activation metadata.

Reads results/oneshot_<task>_judge/judged_results.json for each task and writes a
`judge_top1` boolean into every matching activation row's metadata (both source and
target roles) in results/oneshot_paired_graded/<pair>/shard_*.pt, plus into grading.json.
Match key = (function_task, output_word, query_word) — unique per prompt. In-place rewrite.
"""
import argparse
import glob
import json
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graded_dir", type=Path, default=Path("results/oneshot_paired_graded/antonym_synonym"))
    p.add_argument("--judge_root", type=Path, default=Path("results"))
    p.add_argument("--function_tasks", nargs="+", default=["antonym", "synonym"])
    return p.parse_args()


def main():
    args = parse_args()
    # build verdict map (function_task, output_word, query) -> judge_top1
    verdict = {}
    for task in args.function_tasks:
        jr = json.loads((args.judge_root / f"oneshot_{task}_judge" / "judged_results.json").read_text())
        for r in jr["records"]:
            verdict[(task, r["output_word"], r["query_input"])] = bool(r["judge_correct"])
        print(f"{task}: {len(jr['records'])} verdicts loaded")

    def key_of(m):
        return (m["function_task"], m["output_word"], m["query_word"])

    # tag shards
    n_rows = n_tagged = 0
    for sp in sorted(glob.glob(str(args.graded_dir / "shard_*.pt"))):
        data = torch.load(sp, map_location="cpu", weights_only=False)
        for m in data["metadata"]:
            n_rows += 1
            v = verdict.get(key_of(m))
            if v is not None:
                m["judge_top1"] = v
                n_tagged += 1
        torch.save(data, sp)
    print(f"tagged {n_tagged}/{n_rows} activation rows with judge_top1 across shards")

    # tag grading.json
    grading_path = args.graded_dir / "grading.json"
    grading = json.loads(grading_path.read_text())
    g_tagged = 0
    for g in grading:
        v = verdict.get((g["function_task"], g["output_word"], g["query"]))
        if v is not None:
            g["judge_top1"] = v
            g_tagged += 1
    grading_path.write_text(json.dumps(grading, indent=2))
    print(f"tagged {g_tagged}/{len(grading)} grading.json rows with judge_top1")

    # report counts
    for task in args.function_tasks:
        rows = [g for g in grading if g["function_task"] == task and "judge_top1" in g]
        c = sum(g["judge_top1"] for g in rows)
        print(f"  {task}: judge_top1 True = {c}/{len(rows)} = {c/len(rows):.3f}")


if __name__ == "__main__":
    main()
