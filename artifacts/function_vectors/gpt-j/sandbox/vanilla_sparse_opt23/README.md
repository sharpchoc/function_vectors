# SANDBOX FV definition: `vanilla_sparse_opt23` ("vanilla sparse optimisation FV")

**Sandbox definition — NOT a repo default.** "Function vectors" still means
`train_varicl_top40` (DECISIONS.md 2026-07-10); do not build canonical results on this set
without explicit user promotion.

Definition: for each task, the FV is the **unweighted** sum over the **23 heads with
coefficient c > 0.8** from the sparse-optimization head selection
(`artifacts/sandbox/sparse_head_selection/`, λ=0.01, leave-one-task-out CV; Hu et al. 2025
arXiv:2505.05145 §3.1) of that task's varicl mean head outputs
(`artifacts/multitask_aie_heads_varicl/<task>/<task>_mean_head_activations_varicl.pt`),
projected through each head's attention out_proj — i.e. exactly the canonical train_varicl
top-N construction, with the head set swapped for the sparse-optimization pick. The learned
coefficients select heads only; they do not weight the sum ("vanilla").

Head set (layer, head, c): see `../../../../sandbox/sparse_head_selection/vanilla_sparse_opt23_heads.json`.
Built by `src/eval_scripts/compute_all_task_fvs_varicl.py --heads_path ... --n_top_heads 23`
(see `fv_manifest.json` for exact inputs). Held-out evaluation:
`src/sandbox/sparse_head_selection/eval_vanilla_sparse_fv_heldout.py` → overlay on
`results/steering_vector_comparison/heldout_varicl_nheads_sweep/`.
