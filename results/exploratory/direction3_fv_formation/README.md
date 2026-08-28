# direction3_fv_formation — layout

Reorganized 2026-07-30 (57 flat dirs → 6 folders; see WORKLOG "Stream cue-attn part 8").
Migration script: `logs/direction3_reorg/migrate.sh`. Old paths cited in WORKLOG/DECISIONS
entries predating this date resolve via the mapping below. Script `--output_root` defaults
were repointed the same day; `run_config.json` files inside moved dirs keep their original
(pre-move) paths as historical provenance.

```
ablation/                     causal projection-ablation experiments (Δ log p protocol)
  preimages/                  ridge-map pre-image / FV direction sources (Stream W lineage)
  attention_head_mechanisms/  d_payload subspace sources (head value-channel pullbacks)
attention_head_analysis/      observational head studies (attention rows, d_content,
                              d_payload maps, top-40 geometry) — pre-ablation groundwork
activation_to_fv_decoding/    can the FV be read out of residual activations
                              (fulldim/pca ridge, joint-PCA, OLS sweeps, cosine alignment)
activation_geometry/          activation-space structure (PCA scatters, ICL-evolution
                              PCAs, rank-by-position)
preimage_analysis/            pre-image work not tied to an ablation setup
```

## Old → new mapping

| Old (flat) | New |
|---|---|
| oneshot_preimage_ablation | ablation/preimages/oneshot/main |
| oneshot_preimage_ablation_numbers | ablation/preimages/oneshot/numbers |
| oneshot_preimage_ablation_propagated | ablation/preimages/oneshot/propagated |
| oneshot_preimage_ablation_propagated_numbers | ablation/preimages/oneshot/propagated_numbers |
| fiveshot_preimage_ablation | ablation/preimages/fiveshot |
| payload_subspace_ablation | ablation/attention_head_mechanisms/train_tasks |
| payload_subspace_ablation_test7 | ablation/attention_head_mechanisms/test7 |
| payload_subspace_ablation_test7_k{1,2,8,16} | ablation/attention_head_mechanisms/test7_k_sweep/k{1,2,8,16} |
| payload_subspace_ablation_test7_ciew_k{4,8} | ablation/attention_head_mechanisms/test7_cie_weighted/k{4,8} |
| top10_head_attention_cue_token | attention_head_analysis/top10_head_attention_cue_token |
| dcontent_layer_token | attention_head_analysis/dcontent_layer_token |
| dpayload_layer_token | attention_head_analysis/dpayload_layer_token |
| top40_head_geometry | attention_head_analysis/top40_head_geometry |
| fulldim_ridge_activation_to_fv | activation_to_fv_decoding/fulldim_ridge/main |
| fulldim_ridge_activation_to_fv_qwen3 | activation_to_fv_decoding/fulldim_ridge/qwen3 |
| fulldim_ridge_activation_to_fv_varicl_top40 | activation_to_fv_decoding/fulldim_ridge/varicl_top40 |
| fulldim_ridge_activation_to_fv_varicl_top40_plus_numbers | activation_to_fv_decoding/fulldim_ridge/varicl_top40_plus_numbers |
| fulldim_ridge_activation_to_fv_varicl_top40_plus_number_digits | activation_to_fv_decoding/fulldim_ridge/varicl_top40_plus_number_digits |
| fulldim_ridge_weight_heatmaps | activation_to_fv_decoding/fulldim_ridge/weight_heatmaps |
| fulldim_ridge_activation_to_fv_shuffled[_seed0-2] | activation_to_fv_decoding/fulldim_ridge/controls/shuffled[_seed0-2] |
| fulldim_ridge_activation_to_fv_rowshuffled[_seed0-2] | activation_to_fv_decoding/fulldim_ridge/controls/rowshuffled[_seed0-2] |
| pca_ridge_activation_to_fv | activation_to_fv_decoding/pca_ridge/main |
| pca_ridge_activation_to_fv_varicl_top40 | activation_to_fv_decoding/pca_ridge/varicl_top40 |
| pca_ridge_activation_to_fv_varicl_max4_top40 | activation_to_fv_decoding/pca_ridge/varicl_max4_top40 |
| joint_pca_activation_to_fv_regression | activation_to_fv_decoding/joint_pca/main |
| joint_pca_activation_to_fv_regression_icl | activation_to_fv_decoding/joint_pca/icl |
| joint_pca_activation_to_fv_regression_icl_multitask_top10 | activation_to_fv_decoding/joint_pca/icl_multitask_top10 |
| joint_pca_activation_to_fv_regression_icl_ridge | activation_to_fv_decoding/joint_pca/icl_ridge |
| joint_pca_activation_to_fv_regression_icl_ridge_multitask_top10 | activation_to_fv_decoding/joint_pca/icl_ridge_multitask_top10 |
| layer_sweep_activation_to_fv_ols | activation_to_fv_decoding/ols_layer_sweeps/main |
| layer_sweep_activation_to_fv_ols_full_dim_k16 | activation_to_fv_decoding/ols_layer_sweeps/full_dim_k16 |
| layer_sweep_activation_to_fv_ols_full_dim_k32 | activation_to_fv_decoding/ols_layer_sweeps/full_dim_k32 |
| k_sweeps | activation_to_fv_decoding/ols_layer_sweeps/k_sweeps |
| cosine_activation_to_task_fv | activation_to_fv_decoding/cosine/activation_to_task_fv |
| cosine_activation_to_fv_varicl_top40_pertask[_digits] | activation_to_fv_decoding/cosine/varicl_top40_pertask[_digits] |
| pca_abstractive_fv_activation_scatter | activation_geometry/pca_abstractive_fv_activation_scatter |
| pca_abstractive_icl_examples_fv_activation_scatter[_multitask_top10] | activation_geometry/(same name) |
| pca_cue_token_icl_evolution | activation_geometry/pca_cue_token_icl_evolution |
| pca_lastlabel_token_icl_evolution | activation_geometry/pca_lastlabel_token_icl_evolution |
| activation_rank_by_position | activation_geometry/activation_rank_by_position |
| twoshot_pairdiff_fv_preimage | preimage_analysis/twoshot_pairdiff_fv_preimage |
| preimage_steering | preimage_analysis/preimage_steering |
| joint_pca_activation_to_fv_regression_smoke | DELETED (superseded trial run; user-approved 2026-07-30) |
| pca_abstractive_fv_activation_scatter_smoke | DELETED (superseded trial run; user-approved 2026-07-30) |
```
