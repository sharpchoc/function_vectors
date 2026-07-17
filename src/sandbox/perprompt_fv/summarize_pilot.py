#!/usr/bin/env python
"""SANDBOX: side-by-side summary of the per-prompt-target ridge vs the canonical study.

Aggregates every perprompt_shard_icl{n} under the pilot root and joins, per
(icl_index, token_role, layer):
  * canonical study (combined_metrics_with_r2.csv, fv_root=train_varicl_top40): test_mse, test_r2
  * sandbox per-prompt run: train_mse/r2, test_mse_fv/test_r2_fv (comparable), test_mse_pp/test_r2_pp

Writes summary_vs_canonical_all.csv + .md at the pilot root and prints the headlines.
"""
import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.paths import FV_FORMATION_DIR, RESULTS_ROOT  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--old_csv", type=Path,
                   default=FV_FORMATION_DIR / "fulldim_ridge_activation_to_fv_varicl_top40/combined_metrics_with_r2.csv")
    p.add_argument("--pilot_root", type=Path,
                   default=RESULTS_ROOT / "sandbox/perprompt_ridge_pilot/gptj_train_varicl_top40")
    return p.parse_args()


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    old = {
        (int(r["icl_example_index"]), r["token_role"], int(r["layer"])): r
        for r in read_csv_rows(args.old_csv)
    }

    out_rows = []
    for icl in range(1, 11):
        shard_csv = args.pilot_root / f"perprompt_shard_icl{icl}" / "metrics.csv"
        if not shard_csv.exists():
            print(f"(skipping icl{icl}: no {shard_csv})")
            continue
        for r in read_csv_rows(shard_csv):
            key = (icl, r["token_role"], int(r["layer"]))
            o = old.get(key)
            out_rows.append({
                "icl_index": icl,
                "token_role": key[1],
                "layer": key[2],
                "old_test_mse_fv": float(o["test_mse"]) if o else None,
                "old_test_r2_fv": float(o["test_r2"]) if o else None,
                "new_test_mse_fv": float(r["test_mse_fv"]),
                "new_test_r2_fv": float(r["test_r2_fv"]),
                "new_test_mse_pp": float(r["test_mse_pp"]),
                "new_test_r2_pp": float(r["test_r2_pp"]),
                "new_train_mse": float(r["train_mse"]),
                "new_train_r2": float(r["train_r2"]),
                "new_best_alpha": float(r["best_alpha"]),
                "alpha_pinned": r["alpha_pinned"],
            })
    out_rows.sort(key=lambda x: (x["icl_index"], x["token_role"], x["layer"]))

    out_csv = args.pilot_root / "summary_vs_canonical_all.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    def fmt(v, nd=3):
        return "-" if v is None else f"{v:.{nd}f}"

    def cell_name(r):
        return f"icl{r['icl_index']}/{r['token_role']} L{r['layer']}"

    with_old = [r for r in out_rows if r["old_test_mse_fv"] is not None]
    old_best_mse = min(r["old_test_mse_fv"] for r in with_old)
    n_better = sum(1 for r in with_old if r["new_test_mse_fv"] < r["old_test_mse_fv"])
    n_beat_oldbest = sum(1 for r in out_rows if r["new_test_mse_fv"] < old_best_mse)
    best_new_fv = min(out_rows, key=lambda x: x["new_test_mse_fv"])
    best_new_pp = min(out_rows, key=lambda x: x["new_test_mse_pp"])
    best_old = min(with_old, key=lambda x: x["old_test_mse_fv"])

    lines = [
        "# SANDBOX: per-prompt head-sum targets vs canonical FV-broadcast ridge (all ICL shards)",
        "",
        f"- cells compared: {len(with_old)} (of {len(out_rows)} new cells)",
        f"- new beats old same-cell on test-vs-FV MSE: {n_better}/{len(with_old)}",
        f"- new cells beating the old study's overall best ({old_best_mse:.5f}): {n_beat_oldbest}",
        f"- old (canonical) best cell: {cell_name(best_old)} "
        f"test_mse_fv={fmt(best_old['old_test_mse_fv'],5)} R2={fmt(best_old['old_test_r2_fv'])}",
        f"- NEW best cell by test-vs-FV: {cell_name(best_new_fv)} "
        f"test_mse_fv={fmt(best_new_fv['new_test_mse_fv'],5)} R2={fmt(best_new_fv['new_test_r2_fv'])}",
        f"- NEW best cell by test-vs-per-prompt: {cell_name(best_new_pp)} "
        f"test_mse_pp={fmt(best_new_pp['new_test_mse_pp'],5)} R2={fmt(best_new_pp['new_test_r2_pp'])}",
        "",
        "## Best cell per ICL index (by new test-vs-FV MSE)",
        "",
        "| icl | best cell | old mse(FV) | old R2 | new mse(FV) | new R2(FV) | new mse(pp) | new R2(pp) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for icl in sorted({r["icl_index"] for r in out_rows}):
        rows_icl = [r for r in out_rows if r["icl_index"] == icl]
        b = min(rows_icl, key=lambda x: x["new_test_mse_fv"])
        lines.append(
            f"| {icl} | {b['token_role']} L{b['layer']} | {fmt(b['old_test_mse_fv'],5)} | {fmt(b['old_test_r2_fv'])} "
            f"| {fmt(b['new_test_mse_fv'],5)} | {fmt(b['new_test_r2_fv'])} | {fmt(b['new_test_mse_pp'],5)} "
            f"| {fmt(b['new_test_r2_pp'])} |"
        )
    lines += [
        "",
        "## Top 15 cells overall (by new test-vs-FV MSE)",
        "",
        "| cell | old mse(FV) | old R2 | new mse(FV) | new R2(FV) | new mse(pp) | new R2(pp) | train R2 | alpha |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(out_rows, key=lambda x: x["new_test_mse_fv"])[:15]:
        lines.append(
            f"| {cell_name(r)} | {fmt(r['old_test_mse_fv'],5)} | {fmt(r['old_test_r2_fv'])} "
            f"| {fmt(r['new_test_mse_fv'],5)} | {fmt(r['new_test_r2_fv'])} | {fmt(r['new_test_mse_pp'],5)} "
            f"| {fmt(r['new_test_r2_pp'])} | {fmt(r['new_train_r2'])} | {r['new_best_alpha']:.3g} |"
        )
    out_md = args.pilot_root / "summary_vs_canonical_all.md"
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_csv}\nwrote {out_md}")


if __name__ == "__main__":
    main()
