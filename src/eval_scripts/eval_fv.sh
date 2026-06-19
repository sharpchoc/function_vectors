#!/bin/bash
ARTIFACTS="${FV_ARTIFACTS_ROOT:-artifacts}"; RESULTS="${FV_RESULTS_ROOT:-results}"; LOGS="${FV_LOGS_ROOT:-logs}"
datasets=('antonym')
# datasets=('antonym' 'capitalize' 'country-capital' 'english-french' 'present-past' 'singular-plural')
cd ../

for d_name in "${datasets[@]}"
do
    echo "Running Script for: ${d_name}"
    python evaluate_function_vector.py --dataset_name="${d_name}" --save_path_root="$ARTIFACTS/gptj" --model_name='EleutherAI/gpt-j-6b'
done