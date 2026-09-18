#!/bin/bash
cd /workspace/function_vectors || exit 1
MKL_THREADING_LAYER=GNU /workspace/micromamba/envs/fv/bin/python src/sandbox/style_translation/read_write_map_code.py > logs/read_write_map.log 2>&1 && echo "JOB DONE" >> logs/read_write_map.log || echo "JOB FAILED" >> logs/read_write_map.log
