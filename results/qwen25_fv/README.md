# results/qwen25_fv — Qwen2.5-7B-Instruct repeat of the read/write-feature study

Sibling of `results/69_task_run/` (GPT-J). Same pipeline, same scripts (parameterised 2026-09-22), base-sampling
`Q:/A:` prompts without a BOS prefix. Pool = `task_splits/qwen25_ext_steerable_96_prunedfail.json` (96 tasks:
77 train / 19 held-out; the 69-pool's 68 shared tasks + 28 new ones), prompts `dataset_files/isolation_prompts_ext_qwen25/`,
write feature = the 140-head pooled sparse selection `artifacts/sandbox/ext_steerability_qwen25_96/pooled_sparse/`,
read feature = block-output residual at the last token of the 10th demo target (`artifacts/qwen25_read/label_resid_means`).
Qwen2.5-7B has 28 blocks, d = 3584, 28 heads. All accuracies are temperature-1 sampled exact match, 150 prompts/task
(52 for next_in_group / next_in_period).

| Folder | Study (GPT-J counterpart) | Contents |
|---|---|---|
| `screen/` | competence screen | 6-shot accuracy of the extended pool (`qwen25_acc6.csv`), threshold ≥ .30 → 104 tasks |
| `round1_104/` | head selection | 104-task pooled sparse selection, head-count sweep, train/heldout summary |
| `round2_96_prunedfail/` | `FV_train_test_generalisation/` | steering-prune to 96 tasks, refit (140 heads), zero-shot heldout .058 → .758 |
| `read_feature_layer_selection/` | `bottom_up_read_features/layer_selection/` | raw label-mean steering layer × α sweep (1-shot dummy scaffold); peak **L12** (.556), shared-mean control ≈ .05 |
| `read_feature_steering_6shot/` | `bottom_up_read_features/steering_results/sixshot_dummy/` | m_A(L12) injected at all six dummy '_' slots: .000 → .543 (α=2), best-α .555 = 70% of real 6-shot (.798); equals the 1-shot dummy level (.555) |
| `read_write_map/` | `FV_linear_decodability/`, `understanding_read_write_linear_map/` | per-prompt ridge read → FV per read layer; heldout centroid R² .554 (L22), per-prompt .29 (L20) |
| `readwrite_msj_transfer/` | — | ICL read→write map applied to the MSJ read feature: no transfer (cos ≈ 0) |
| `FV_ablation/` | `FV_ablation/` | **Study A** — FV-direction ablation at the final cue token, {own, cf} × {zero, mean} × {L9–27, L0–27}, 6-shot and 1-shot |
| `read_feature_ablation/` | `bottom_up_read_features/ablation/task_unique_meanresid/` | **Study C** — task-unique direction û_A (mean carrier-removed **L11–13** residual) ablated at every demo target token, own vs counterfactual task; `cf_task_pairs.csv` |
| `read_write_relationship/` | `read_write_relationship/{bottom_up,meanresid}{,_1shot}/` | **Study D** — read-feature injection at the dummy target slots → cos(cue residual, task FV); `bottom_up` = m_A(L12) at L12, `meanresid` = s_A = c + u_A at L0 |
| `write_feature_and_model_accuracy/` | `write_feature_and_model_accuracy/` | **Study B** — FV presence (cos at the query cue, all 28 layers) vs sampled accuracy, n = 0..6; `baseline_subtracted/`, `per_prompt/` |

## Headline numbers, Qwen vs GPT-J (mean over tasks; 96 vs 69)

| Claim | Qwen2.5-7B-Instruct | GPT-J-6B |
|---|---|---|
| A. own-FV zero-ablation at the cue, 6-shot (unablated → own → cf-task FV) | .798 → **.056** vs .543 | .639 → **.013** vs .476 |
| A. own-FV mean-ablation, 6-shot | .798 → .739 vs .797 (nearly harmless) | .639 → .242 vs .616 |
| A. 1-shot, zero-ablation | .476 → **.014** vs .202 | .211 → .001 vs .107 |
| C. own û_A mean-ablation at demo targets, 6-shot (unablated → own → cf) | .798 → **.205** vs .754 | .630 → **.132** vs .632 |
| C. 1-shot | .476 → **.171** vs .452 | .208 → .044 vs .205 |
| D. cos(cue residual, task FV) after read injection, 6-shot dummy scaffold, α=2 | .388 → **.502** (m_A(L12)@L12), .388 → .475 (s_A@L0); readout L24 | .183 → .365 (m_A(L6)@L6), .183 → .424 (s_A@L0); readout L13 |
| B. within-task Spearman ρ (presence vs accuracy over n = 0..6) | median +.85 at L24 (83/96 positive); +.89 at L20–22 | median +.96 (69/69), L9–20 band |
| B. between-task ρ at fixed n = 6 | +.17 at L24; −.24 for the L9–20 band mean | −.36 for the L9–20 band mean, −.31 at L13 (Simpson pattern) |
| Read steering, 6-shot dummy scaffold (unsteered → six slots steered, best α → real 6-shot) | .000 → .555 → .798 (70%); 1-slot dummy .555 | .000 → .447 → .630 (71%); 1-slot dummy .126 |

Conventions and choices specific to the Qwen port (DECISIONS 2026-09-22):
- **Headline presence/readout layer = L24**, the argmax of the 6-shot mean-presence profile
  (`write_feature_and_model_accuracy/presence_by_layer.{csv,png}`); the profile is a broad plateau L20–26 (.47–.50)
  and collapses at L27. Every table also reports the GPT-J band variants (max / mean over L9–20). Applying the same
  rule to GPT-J's capture gives L15 (.430) vs the quoted L13 (.425) — a tie within .005.
- **Read band L11–13** for û_A / s_A (Qwen read-steering peak L12); ‖u_A‖ median 30.0 vs carrier ‖c‖ 32.5
  (GPT-J: 28 vs 47) — on Qwen the shared carrier is not much larger than the task-unique part.
- **Counterfactual pairs:** GPT-J semantic-family tags for the 68 shared tasks, Haiku tags (same prompt) for the 28
  new tasks; seeded different-family sampling (`read_feature_ablation/cf_task_pairs.csv`).
- **Baselines** for Studies A and C are in-run, seed-matched (`zero_shot`, `real{1,6}_baseline` conditions of the
  FV-ablation eval), since Qwen has no sixshot_dummy CSV.
- Qwen-specific observations: own-FV MEAN-ablation barely hurts (the generic component along the FV direction
  restores the answer; mean cos(v_A, v_generic) = .79 across the 96 FVs), whereas zero-ablation is as specific as on
  GPT-J; the between-task presence/accuracy relation is positive at L24 but negative in the L9–20 band.

Artifacts (gitignored): `artifacts/qwen25_fv/` (grand means, eval JSONs, presence captures, injection captures,
`bottom_up_ablation/bankA/` û_A bank, `cf_task_pairs.json`); run logs and orchestration: `logs/qwen25_fv/port/`.
WORKLOG.md entries dated 2026-09-22 hold the per-study details.
