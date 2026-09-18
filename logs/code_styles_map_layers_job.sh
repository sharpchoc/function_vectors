#!/bin/bash
# v7 recipe for each read layer: pair differences, all prompts, unit norm, test19 split
cd /workspace/function_vectors || exit 1
for L in 6 10 12 14 16 18; do
  T=v9_pairdiff_unitnorm_layers/L$L
  MKL_THREADING_LAYER=GNU /workspace/micromamba/envs/fv/bin/python src/sandbox/style_translation/read_write_map_code.py --tag $T --select all --pair_diff --unit_norm --read_layer $L --split_file results/code_styles/read_write_map/split_2026-09-15_test19.json > logs/read_write_map_v9_L$L.log 2>&1 || { echo "JOB FAILED L$L" >> logs/read_write_map_v9.log; exit 1; }
  echo "L$L done" >> logs/read_write_map_v9.log
done
echo "JOB DONE" >> logs/read_write_map_v9.log
