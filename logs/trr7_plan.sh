#!/bin/bash
cd /workspace/function_vectors || exit 1
PY=/workspace/micromamba/envs/fv/bin/python; A=artifacts/style_translation/qwen25_code_tr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until [ -f $A/read_features/vectors_k3_train/py2_except.npz ]; do sleep 30; done
$PY src/sandbox/style_translation/read_sweep.py --model qwen25_code_tr --families py_loop_vars --layers 2 4 6 8 10 12 14 16 20 24 --alphas 1 2 4 8 --with_base --limit 40 --batch 12 --token_budget 12000 --out_dir $A/read_steer/sweep_k3/trr1 >> logs/tr_rsweep_trr7.log 2>&1
$PY src/sandbox/style_translation/read_sweep.py --model qwen25_code_tr --families py_join_concat --layers 2 4 6 8 10 12 14 16 20 24 --alphas 1 2 4 8 --with_base --limit 40 --batch 12 --token_budget 12000 --out_dir $A/read_steer/sweep_k3/trr1 >> logs/tr_rsweep_trr7.log 2>&1
$PY src/sandbox/style_translation/read_sweep.py --model qwen25_code_tr --families py_comprehension --layers 2 4 6 8 10 12 14 16 20 24 --alphas 1 2 4 8 --with_base --limit 40 --batch 12 --token_budget 12000 --out_dir $A/read_steer/sweep_k3/trr2 >> logs/tr_rsweep_trr7.log 2>&1
$PY src/sandbox/style_translation/read_sweep.py --model qwen25_code_tr --families py_paren_if --layers 2 4 6 8 10 12 14 16 20 24 --alphas 1 2 4 8 --with_base --limit 40 --batch 12 --token_budget 12000 --out_dir $A/read_steer/sweep_k3/trr2 >> logs/tr_rsweep_trr7.log 2>&1
$PY src/sandbox/style_translation/read_sweep.py --model qwen25_code_tr --families py_private --layers 2 4 6 8 10 12 14 16 20 24 --alphas 1 2 4 8 --with_base --limit 40 --batch 12 --token_budget 12000 --out_dir $A/read_steer/sweep_k3/trr3 >> logs/tr_rsweep_trr7.log 2>&1
echo "JOB DONE trr7" >> logs/tr_rsweep_trr7.log
