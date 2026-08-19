#!/usr/bin/env bash
# Rewrite TASK69_RUN_DIR sub-paths from the pre-2026-08-19 results/69_task_run layout to the
# reorganised one (see results/69_task_run/README.md for the mapping). Run from anywhere after
# merging an old experiment branch; it edits src/**/*.py in place and then prints any remaining
# old-layout or context-dependent references for manual review (e.g. the split of the old
# FV_dimensionality_{analysis,reduction} contents, and per-study "out" names passed as variables).
set -euo pipefail
cd "$(dirname "$0")/.."

mapfile -t files < <(grep -rl "TASK69_RUN_DIR" src --include='*.py' || true)
for f in "${files[@]}"; do
  sed -i \
    -e 's|TASK69_RUN_DIR / "train_test_generalisation"|TASK69_RUN_DIR / "FV_train_test_generalisation"|g' \
    -e 's|TASK69_RUN_DIR / "FV_location" / "presence_vs_accuracy"|TASK69_RUN_DIR / "write_feature_and_model_accuracy"|g' \
    -e 's|"FV_location" / "read_dir_presence"|"feature_locations" / "top_down_read_dir_presence"|g' \
    -e 's|"FV_location" / "label_mean_L6_presence"|"feature_locations" / "bottom_up_label_mean_L6_presence"|g' \
    -e 's|TASK69_RUN_DIR / "FV_location"|TASK69_RUN_DIR / "feature_locations"|g' \
    -e 's|TASK69_RUN_DIR / "labeltoken_fv_ridge"|TASK69_RUN_DIR / "FV_linear_decodability" / "labeltoken_fv_ridge"|g' \
    -e 's|TASK69_RUN_DIR / "token_layer_regressions"|TASK69_RUN_DIR / "FV_linear_decodability" / "token_layer_regressions"|g' \
    -e 's|TASK69_RUN_DIR / "raw_mean_steering" / "sixshot_dummy"|TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results" / "sixshot_dummy"|g' \
    -e 's|TASK69_RUN_DIR / "raw_mean_steering" / "narrow_patch"|TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results" / "narrow_patch"|g' \
    -e 's|TASK69_RUN_DIR / "raw_mean_steering" / "dimensionality"|TASK69_RUN_DIR / "bottom_up_read_features" / "dimensionality_analysis" / "dimensionality"|g' \
    -e 's|TASK69_RUN_DIR / "raw_mean_steering" / "sparse_pc40"|TASK69_RUN_DIR / "bottom_up_read_features" / "dimensionality_analysis" / "sparse_pc40"|g' \
    -e 's|TASK69_RUN_DIR / "raw_mean_steering"|TASK69_RUN_DIR / "bottom_up_read_features" / "layer_selection"|g' \
    -e 's|TASK69_RUN_DIR / "read_vector_head_selection"|TASK69_RUN_DIR / "bottom_up_read_features" / "head_selection"|g' \
    -e 's|"Read_direction_geometry" / "cross_bracket_overlap"|"top_down_read_features" / "definition_sweep" / "cross_bracket_overlap"|g' \
    -e 's|"Read_direction_geometry" / "unit_vs_natural_containment"|"top_down_read_features" / "definition_sweep" / "unit_vs_natural_containment"|g' \
    -e 's|"Read_direction_geometry" / "writeup_assets"|"top_down_read_features" / "definition_sweep" / "writeup_assets"|g' \
    -e 's|"Read_direction_geometry" / "steering_methods"|"top_down_read_features" / "steering_results" / "steering_methods"|g' \
    -e 's|"Read_direction_geometry" / "steering"|"top_down_read_features" / "steering_results" / "steering"|g' \
    -e 's|"Read_direction_geometry" / "dot_perhead_unit_sparse_optimisation"|"top_down_read_features" / "ablation" / "dot_perhead_unit_sparse_optimisation"|g' \
    -e 's|TASK69_RUN_DIR / "Read_direction_geometry"|TASK69_RUN_DIR / "top_down_read_features" / "dimensionality_analysis"|g' \
    -e 's|TASK69_RUN_DIR / "pc50_ablation"|TASK69_RUN_DIR / "top_down_read_features" / "ablation" / "pc50_ablation"|g' \
    -e 's|TASK69_RUN_DIR / "fv_presence_heatmaps"|TASK69_RUN_DIR / "read_write_relationship" / "top_down" / "fv_presence_heatmaps"|g' \
    -e 's|TASK69_RUN_DIR / "mean_read_steering_effect_on_write"|TASK69_RUN_DIR / "read_write_relationship" / "bottom_up"|g' \
    -e 's|"FV_dimensionality_analysis" / "pc_sparse_summary.csv"|"FV_dimensionality_reduction" / "train_test_split" / "pc_sparse_summary.csv"|g' \
    "$f"
done

echo "Remaining TASK69_RUN_DIR references to review manually (may be valid, e.g. spectra ->"
echo "FV_dimensionality_analysis stays; pc_sparse/dimreduction writers must target the new"
echo "FV_dimensionality_reduction/{train_test_split,train_test_together_50d,low_dim_22d} arms):"
grep -rn 'TASK69_RUN_DIR' src --include='*.py' \
  | grep -E 'TASK69_RUN_DIR / "(FV_dimensionality_analysis|FV_dimensionality_reduction|train_test_generalisation|FV_location|raw_mean_steering|read_vector_head_selection|Read_direction_geometry|mean_read_steering_effect_on_write)"|"read_dir_presence"|"label_mean_L6_presence"|"presence_vs_accuracy"|TASK69_RUN_DIR / [a-zA-Z_]+[\[(]' \
  || echo "  none"
