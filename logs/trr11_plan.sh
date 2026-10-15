#!/bin/bash
cd /workspace/function_vectors || exit 1
PY=/workspace/micromamba/envs/fv/bin/python; A=artifacts/style_translation/qwen25_code_tr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY src/sandbox/style_translation/read_sweep.py --model qwen25_code_tr --families py_class_naming --layers 2 4 6 8 10 12 14 16 20 24 --alphas 1 2 4 8 --with_base --limit 40 --batch 12 --token_budget 12000 --out_dir $A/read_steer/sweep_k3/trr3 >> logs/tr_rsweep_trr11.log 2>&1
$PY src/sandbox/style_translation/read_sweep.py --model qwen25_code_tr --families py_with_open --layers 2 4 6 8 10 12 14 16 20 24 --alphas 1 2 4 8 --with_base --limit 40 --batch 12 --token_budget 12000 --out_dir $A/read_steer/sweep_k3/trr4 >> logs/tr_rsweep_trr11.log 2>&1
$PY src/sandbox/style_translation/read_sweep.py --model qwen25_code_tr --families py2_iter --layers 2 4 6 8 10 12 14 16 20 24 --alphas 1 2 4 8 --with_base --limit 40 --batch 12 --token_budget 12000 --out_dir $A/read_steer/sweep_k3/trr5 >> logs/tr_rsweep_trr11.log 2>&1
echo "JOB DONE trr11" >> logs/tr_rsweep_trr11.log
