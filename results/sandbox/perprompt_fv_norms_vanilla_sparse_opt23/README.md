# SANDBOX: per-prompt FV norms under `vanilla_sparse_opt23` heads (fixed10)

**Sandbox result — NOT a repo default.** "Function vectors" still means
`train_varicl_top40` (DECISIONS.md 2026-07-10). This folder repeats the part-14 fixed10
per-prompt FV norm study with the head set swapped for the SANDBOX sparse-optimization
pick (`artifacts/sandbox/sparse_head_selection/vanilla_sparse_opt23_heads.pt`: the 23
heads with c > 0.8, unweighted sum). Same captured activations and prompts as part 14
(`artifacts/perprompt_head_activations/gptj_27tasks_170prompts/fixed10/`); only H differs.

Files: `fvnorm_hist_pooled_fixed10.png`, `fvnorm_hist_pertask_fixed10.png`,
`fvnorm_median_top40_vs_sparse23.png` (scatter vs part-14 medians, Spearman ρ = 0.44),
`fvnorm_summary.csv`, `fvnorm_perprompt_fixed10.npz` (git-ignored, volume only).

Produced by `src/eval_scripts/plot_perprompt_fv_norm_hist.py --variants fixed10
--pooled_heads_path <sparse23.pt> --out_dir <here>` and
`src/eval_scripts/plot_fvnorm_headset_comparison.py`. See WORKLOG part 14b.
