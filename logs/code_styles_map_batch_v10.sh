#!/bin/bash
# v10: the v7 recipe on the regen8 corpus, 5 random 66/34 and 5 random 80/20 splits (the v8 partitions minus the two dropped families), two at a time
cd /workspace/function_vectors || exit 1
run() { T=$1; S=$2; MKL_THREADING_LAYER=GNU /workspace/micromamba/envs/fv/bin/python src/sandbox/style_translation/read_write_map_code.py --tag $T --select all --pair_diff --unit_norm --split_file $S > logs/read_write_map_$T.log 2>&1 && echo "JOB DONE" >> logs/read_write_map_$T.log || echo "JOB FAILED" >> logs/read_write_map_$T.log; }
for s in 1 2 3 4 5; do
  run v10_pairdiff_unitnorm_split66_s$s results/code_styles/read_write_map/split_v10_66_s$s.json &
  run v10_pairdiff_unitnorm_split80_s$s results/code_styles/read_write_map/split_v10_80_s$s.json &
  wait
done
echo "BATCH DONE" >> logs/read_write_map_v10_batch.log
