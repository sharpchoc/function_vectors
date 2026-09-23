#!/bin/bash
# re-run: read ablation for the d2 remainder + write ablation for js_var (partner re-drawn: js_hungarian -> js_quotes)
cd /workspace/function_vectors || exit 1
bash logs/code_styles_read_ablate_job.sh d2 js_var float_literals c_comment_style comment_language
bash logs/code_styles_ablate_job.sh c2 js_var
echo "JOB DONE fix" >> logs/ablate_fix.log
