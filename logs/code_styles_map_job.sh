#!/bin/bash
# read->write map fit: <tag> <select correct|all>
cd /workspace/function_vectors || exit 1
T="$1"; SEL="$2"; EXTRA="$3"
MKL_THREADING_LAYER=GNU /workspace/micromamba/envs/fv/bin/python src/sandbox/style_translation/read_write_map_code.py --tag $T --select $SEL $EXTRA --split_file ${SPLIT:-results/code_styles/read_write_map/split_2026-09-15_test19.json} > logs/read_write_map_$T.log 2>&1 && echo "JOB DONE" >> logs/read_write_map_$T.log || echo "JOB FAILED" >> logs/read_write_map_$T.log
