#!/usr/bin/env python
"""SANDBOX: side-by-side summary of the per-prompt-target pilot vs the canonical study (icl10).

Joins, per (token_role, layer):
  * canonical study (combined_metrics_with_r2.csv, fv_root=train_varicl_top40): test_mse, test_r2
  * sandbox per-prompt run: train_mse/r2, test_mse_fv/test_r2_fv (comparable), test_mse_pp/test_r2_pp

Writes summary.csv + summary.md into the sandbox results dir and prints the headline cells.
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
    p.add_argument("--new_dir", type=Path,
                   default=RESULTS_ROOT / "sandbox/perprompt_ridge_pilot/gptj_train_varicl_top40/perprompt_shard_icl10")
    p.add_argument("--icl_index", type=int, default=10)
    return p.parse_args()


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    old = {
        (r["token_role"], int(r["layer"])): r
        for r in read_csv_rows(args.old_csv)
        if int(r["icl_example_index"]) == args.icl_index
    }
    new = read_csv_rows(args.new_dir / "metrics.csv")

    out_rows = []
    for r in new:
        key = (r["token_role"], int(r["layer"]))
        o = old.get(key)
        out_rows.append({
            "token_role": key[0],
            "layer": key[1],
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
    out_rows.sort(key=lambda x: (x["token_role"], x["layer"]))

    out_csv = args.new_dir / "summary_vs_canonical.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    def fmt(v, nd=3):
        return "-" if v is None else f"{v:.{nd}f}"

    best_new_fv = min(out_rows, key=lambda x: x["new_test_mse_fv"])
    best_new_pp = min(out_rows, key=lambda x: x["new_test_mse_pp"])
    best_old = min((r for r in out_rows if r["old_test_mse_fv"] is not None), key=lambda x: x["old_test_mse_fv"])

    lines = [
        "# SANDBOX pilot: per-prompt head-sum targets vs canonical FV-broadcast ridge (icl10)",
        "",
        f"- old (canonical) best cell: {best_old['token_role']} L{best_old['layer']} "
        f"test_mse_fv={fmt(best_old['old_test_mse_fv'],5)} R2={fmt(best_old['old_test_r2_fv'])}",
        f"- NEW best cell by test-vs-FV: {best_new_fv['token_role']} L{best_new_fv['layer']} "
        f"test_mse_fv={fmt(best_new_fv['new_test_mse_fv'],5)} R2={fmt(best_new_fv['new_test_r2_fv'])}",
        f"- NEW best cell by test-vs-per-prompt: {best_new_pp['token_role']} L{best_new_pp['layer']} "
        f"test_mse_pp={fmt(best_new_pp['new_test_mse_pp'],5)} R2={fmt(best_new_pp['new_test_r2_pp'])}",
        "",
        "| role | layer | old mse(FV) | old R2(FV) | new mse(FV) | new R2(FV) | new mse(pp) | new R2(pp) | new train R2 | alpha |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in out_rows:
        lines.append(
            f"| {r['token_role']} | {r['layer']} | {fmt(r['old_test_mse_fv'],5)} | {fmt(r['old_test_r2_fv'])} "
            f"| {fmt(r['new_test_mse_fv'],5)} | {fmt(r['new_test_r2_fv'])} | {fmt(r['new_test_mse_pp'],5)} "
            f"| {fmt(r['new_test_r2_pp'])} | {fmt(r['new_train_r2'])} | {r['new_best_alpha']:.3g} |"
        )
    out_md = args.new_dir / "summary_vs_canonical.md"
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:8]))
    print(f"\nwrote {out_csv}\nwrote {out_md}")


if __name__ == "__main__":
    main()
