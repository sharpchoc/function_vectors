# WORKLOG

Coordination log for in-flight experimental work on the Function Vectors repo.
Newest entries at top. One stream per active line of work.

---

## 2026-06-10 — Stream B: mixed-task ICL probe (5 antonym + 5 synonym ICL → antonym vs synonym query)

**Status:** COMPLETE.

**Goal:** Quick experiment. 50 prompts, each with 10 ICL examples (5 antonym + 5
synonym demos). Two query conditions: antonym query vs synonym query. Measure
top-1/2/3 accuracy of the correct answer's first token (rank < k convention,
matching `eval_utils.compute_top_k_accuracy`). Cross-prompt batched (25/forward
pass). Clustered bar chart. Also ran the demo order swapped, and overlaid a
same-task 10-shot reference (10 antonym ICL→antonym, 10 synonym ICL→synonym).

**Files:**
- `src/eval_scripts/mixed_icl_antonym_synonym_topk.py` (new). Flags: `--icl_order
  {antonym_first,synonym_first}`, `--baseline_path` (cached same-task ref, computed
  once and reused), `--refresh_baseline`.
- `results/mixed_icl_antonym_synonym/` (antonym_first), `…_synfirst/` (synonym_first),
  `…_baseline/pure_baseline.json` (shared reference).

**Important fix (after first runs):** the query word was NOT being excluded from the
ICL demos (the `exclude_idx` arg existed but was never passed). Fixed: draw the query
first, then exclude its input word from both tasks' demos (`sample_pairs(...,
exclude_inputs=...)`). Collision rate was low (~0.2%/prompt), but the fix reordered
the RNG stream → effectively a fresh draw. Also bumped n_prompts 50→200 because n=50
SE≈0.06 made the order comparison unreliable. **Numbers below are the n=200,
collision-free run; the earlier n=50 numbers are superseded.**

**Commands:**
- `HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 python
  src/eval_scripts/mixed_icl_antonym_synonym_topk.py --n_prompts 200 --icl_order
  antonym_first --output_dir results/mixed_icl_antonym_synonym --refresh_baseline`
- same with `--icl_order synonym_first --output_dir …_synfirst` (reuses cached baseline)

**Results, n=200 (top-1 / top-2 / top-3):**
- antonym_first ICL (antonym demos far, synonym demos recent):
  antonym query 0.47/0.55/0.61; synonym query 0.20/0.30/0.42
- synonym_first ICL (synonym demos far, antonym demos recent):
  antonym query 0.55/0.66/0.73; synonym query 0.12/0.24/0.30
- same-task 10-shot ref: antonym 0.565/0.685/0.76; synonym 0.365/0.50/0.605

**Findings (n=200, SE≈0.035):**
- **Difficulty:** antonym query ≫ synonym query top-1 in every cut (0.47–0.55 vs
  0.12–0.20). Synonym has many valid answers; the single gold first-token often misses
  top-3, so first-token top-k understates synonym competence.
- **Recency IS real here** (corrects the noisy n=50 read): each task does ~+0.08 top-1
  better when *its* demo block is last (closest to the query). antonym query: 0.47
  (demos far) → 0.55 (demos recent); synonym query: 0.12 (demos far) → 0.20 (recent).
- **Dilution:** mixed-ICL synonym (0.12–0.20) sits far below its same-task 10-shot ref
  (0.365); mixed-ICL antonym (0.47–0.55) is close to its ref (0.565), nearly matching
  it when antonym demos are recent. → synonym pays most for halved demos.

**Env note:** transformers/accelerate not installed locally; `pip install
transformers==4.49.0 accelerate` (matches fv_environment.yml). baukit (transitive
dep of eval_utils) not on PyPI + git install blocked → inlined the 3 pure helpers
(get_answer_id, compute_individual_token_rank, compute_top_k_accuracy). GPT-J
weights cached at /workspace/.cache/huggingface; run with HF_HUB_OFFLINE=1.

**Next:** none. Optional: score synonym against any-acceptable-synonym for a fairer
read (current numbers understate synonym competence).

**Blockers:** None.

---

## 2026-06-11 — Stream D: held-out per-layer steering eval for train_varicl (3-series plots)

**Status:** RUNNING (8 tasks; word_length smoke done). Goal: overlay the `train_varicl` FV as a
3rd line on the existing 9 test-task `*_effectiveness_by_layer.png` steering plots.

**Approach (cheap reuse, not a 4h re-run):** new sibling script
`src/eval_scripts/evaluate_heldout_varicl_fv.py` reuses the existing June-8 per-layer results
(multitask train-only + task-specific) from each task's `comparison_summary.json`, and ONLY
evaluates the prebuilt `train_varicl` FV across all 28 layers (zero-shot + 10-shot-shuffled),
using the IDENTICAL filter set (clean_rank_list==0 from `results/gptj_fv/<task>/fs_results_layer_sweep.json`)
and seed=42 so the new line is directly overlayable. Imports evaluate_fv/get_filter_set/
summarize_results from `evaluate_heldout_multitask_head_fvs.py` (shared file NOT modified).

**Verified (word_length smoke):** filter set reproduced exactly (22 == recorded 22, no warning);
3-series plot renders. Each task ~1–2 min after model load (171 ex/layer × 28 × 2).

**Outputs:** per task → `varicl_heads_{zs,fs_shuffled}_results.json`, `varicl_comparison_summary.json`,
`<task>_effectiveness_by_layer_with_varicl.png` (originals kept). Aggregate →
`results/heldout_multitask_head_eval/heldout_varicl_vs_others_summary.json`. Series labels:
"Multitask heads (train, fixed-ICL)" / "Task-specific heads" / "Variable-ICL multitask (train_varicl)".

**Caveat:** the "Multitask heads" baseline = train-only 20-task FIXED-ICL set (the June-8 run),
so this compares fixed-ICL vs variable-ICL train pooling on the same 9 held-out tasks.

---

## 2026-06-10 — Stream D: variable-ICL train-pooled FV method (COMPLETE on pod)

**Status:** All 4 steps DONE. 29 FVs + heads.pt + fv_manifest.json + 29 selected_heads.json under
`results/function_vectors/gpt-j/train_varicl/`. Sanity passed.

**Step 3 (rebuilt):** after filling the 9 test-task activations (gap below), re-ran
`compute_all_task_fvs_varicl.py --overwrite` → 29 FVs (20 train + 9 test), norms 22–43, all sane.
**Step 4:** `write_fv_head_metadata.py --methods train_varicl --n_top_heads 10` → 29 metadata files.
**Sanity:** varicl top-10 vs `train_selected/heads.pt` = **8/10 overlap** (top-2 identical: (9,14),(15,5);
varicl swaps in (10,0),(21,2) for (9,2),(14,9)). Overlap-not-identity, as expected.

**Leftover dirs:** `results/_smoke_varicl/` (smoke), `results/_varicl_testtasks/` (test-task CIE +
activations + a harmless test-pooled head artifact; the 9 mean-activation .pt were copied into
`results/multitask_aie_heads_varicl/<task>/`). Keep for provenance or delete — not referenced by FVs.

---

## 2026-06-10 — Stream D: variable-ICL train-pooled FV method (RUNNING on pod)

**Status:** GPU available in THIS shell (A100 80GB, torch 2.8+cu128, transformers 4.49.0) —
contradicts the earlier "no transformers/GPU" note. Smoke test PASSED; full stage-1 launched.

**Smoke (step 1):** `compute_multitask_varicl_heads.py --tasks ag_news sentiment country-capital
--abstractive_only --query_split valid --demo_split train --n_top_heads 10 --batch_size 8
--max_prompts_per_task 6 --num_shards 1 --save_path_root results/_smoke_varicl --overwrite` →
exit 0. All 3 checks green: mean tensor shape (28,16,256) per task; fs_results_varicl_valid.json
present per task; cie_result confirms min_shots=1/max_shots=10/cap=170, metadata
n_shots="variable[1,10]"; deterministic sampler spans full 1–10 over 170 queries.

**Stage-1 (step 2): DONE.** `bash src/eval_scripts/run_multitask_varicl_all_tasks.sh 4` — 4 shards
on GPU 0, tasks[shard::4] over 20 train tasks, then --reduce. All 20 per-task cie_results written
(1189 prompts total) → `results/multitask_aie_heads_varicl/multitask_top_aie_heads.pt` (+metadata).
Single A100 = GPU-bound bottleneck; shard 3 drew the 5 heaviest tasks and ran solo for the tail
(commonsense_qa alone took 2h21m — long multiple-choice prompts × up to 10 ICL demos; the 170-cap
DID apply: 22 batches×8≈170 candidates, so the cost is sequence LENGTH not prompt count).

**Pooled top-10 heads (varicl):** L9H14 0.0553, L15H5 0.0546, L8H1 0.0242, L12H10 0.0226,
L11H0 0.0138, L8H0 0.0109, L14H0 0.0106, L24H6 0.0090, L21H2 0.0082, L10H0 0.0081.

**Stage-2 (step 3): PARTIAL → GAP FOUND.** `compute_all_task_fvs_varicl.py` built the **20 train**
FVs (norms ~22–43, sane) then crashed: `FileNotFoundError landmark-country_mean_head_activations_varicl.pt`.
The builder does NOT compute activations on the fly — it requires precomputed varicl mean
activations for ALL 29 tasks, but stage-1 (`--task_split_key train_tasks`) only produced them for
the 20 train tasks. The 9 test-task single-position varicl activations were never computed (gap in
the documented step 1→3 sequence).

**Fix (running):** generate the 9 test-task varicl mean activations via the stage-1 worker on the
test split, isolated dir so `writes_global` can't clobber the train head set:
`compute_multitask_varicl_heads.py --task_split_key test_tasks ... --save_path_root
results/_varicl_testtasks --num_shards 1`. (Computes CIE too — unavoidable, but test tasks are all
short-sequence so it's cheap.) Then: copy the 9 `*_mean_head_activations_varicl.pt` into
`results/multitask_aie_heads_varicl/<task>/`, re-run `compute_all_task_fvs_varicl.py --overwrite`
(rebuilds all 29 + manifest), then step 4 metadata. Sanity: compare pooled top-10 vs
train_selected/heads.pt.

---

## 2026-06-10 — Stream D: variable-ICL train-pooled FV method (IMPLEMENTED, not yet run)

**Status:** Code complete + py_compile/bash -n clean. NOT run (no `transformers`/GPU in the
coordinator shell; user runs the 3 shards on a separate GPU instance).

**Question / method:** A 4th FV head-selection method. Each prompt draws a RANDOM 1–10 ICL
count; keep only prompts the model answers correctly, capped at **170 successful/task**; read
both mean head activations AND the CIE intervention at the **query predictive (last) token**
(T=-1) so activations average over variable-length prompts at one consistent position; CIE uses
variable ICL + shuffled labels; head selection pools CIE across the **20 train tasks** (like
`train_selected`, new regime). FVs built for all 29 tasks → `train_varicl`.

**Files created:**
- `src/utils/varicl_utils.py` — `sample_variable_icl_count`, `build_varicl_prompt_data`,
  `get_last_token_mean_head_activations` (→ shape (n_layers,n_heads,head_dim)),
  `varicl_correctness_filter`, `batch_varicl_last_token_intervention` (single-position).
- `src/eval_scripts/compute_multitask_varicl_heads.py` — stage-1 engine (mirrors
  `compute_multitask_top_aie_heads.py`; reuses select_shard/aggregate/reduce; new args
  `--min_shots --max_shots --max_successful_prompts --cie_seed_offset`; train-only pooling).
- `src/eval_scripts/compute_all_task_fvs_varicl.py` — stage-2 (single-position indexing fix).
- `src/eval_scripts/run_multitask_varicl_all_tasks.sh` — 3 background shards + reduce, batch 8.

**Verified (coordinator):** reused signatures match (`_project_attention_inputs`,
`sample_demo_indices`, `word_pairs_to_prompt_data`, `get_answer_id`,
`compute_individual_token_rank`); `prepend_bos` expr matches engine; FV indexing is
`mean_activations[layer, head]` (no `[-1]`); advanced-indexing for last-token read +
head replacement is correct. One benign deviation: compat attrs set on `args`
(`n_shots="variable[1,10]"`, `shuffle_labels`, `mean_activations_root`) so the reused
`write_global_artifact` metadata writer works.

**Outputs (when run):** heads → `results/multitask_aie_heads_varicl/`; FVs →
`results/function_vectors/gpt-j/train_varicl/`.

**Next (run on GPU box):**
1. Smoke: `python src/eval_scripts/compute_multitask_varicl_heads.py --tasks ag_news sentiment country-capital --abstractive_only --query_split valid --demo_split train --n_top_heads 10 --batch_size 8 --max_prompts_per_task 6 --num_shards 1 --save_path_root results/_smoke_varicl --overwrite` → check mean tensor shape (28,16,256), fs_results_varicl_valid.json, shot-count spans 1–10.
2. Full: `bash src/eval_scripts/run_multitask_varicl_all_tasks.sh` (3 shards + reduce).
3. FVs: `python src/eval_scripts/compute_all_task_fvs_varicl.py` (defaults build all 29).
4. Metadata: `python src/eval_scripts/write_fv_head_metadata.py --model_root results/function_vectors/gpt-j --methods train_varicl --n_top_heads 10`.
5. Sanity: compare pooled top-10 vs `train_selected/heads.pt` (expect overlap, not identity).

**Blockers:** None. Plan: `/root/.claude/plans/immutable-finding-boole.md`.

---

## 2026-06-10 — Stream C: direct full-dim (4096→4096) activation→FV ridge per (token pos, layer)

**Owner:** Coordinator (this session). **Status:** DONE — all 899 cells computed, merged, heatmapped.

**Question:** How linearly recoverable is a task's `train_selected` FV from a *single* residual
activation, as a function of token position and layer? **No PCA** — full 4096→4096 ridge, λ by
leave-one-train-task-out CV, single 20-train standardizer, MSE on 7 test tasks (excl.
country-currency, product-company). 31 token positions (pre/first/last for ICL 1–10 + final
prompt token) × 29 layers = 899 cells. Sharded by ICL index (10 shards) for tmux.

**Commands run:**
- `bash src/eval_scripts/run_fulldim_ridge_shards.sh` (CONCURRENCY=3 → 3 tmux windows in session
  `fvridge`: w0=icl{1,4,7,10}, w1=icl{2,5,8}, w2=icl{3,6,9}).
- icl10 re-run after a fix (see Findings): `... --icl_index 10 --overwrite`.
- `python src/eval_scripts/merge_fulldim_ridge_results.py --input_dir results/fulldim_ridge_activation_to_fv`
  → `combined_metrics.csv` (899 rows), `combined_{test_mse,best_alpha}_heatmap.png`, `combined_summary.json`.

**Files changed:**
- NEW `src/eval_scripts/regress_activation_to_fv_fulldim_ridge.py` (worker; one shard = one ICL idx).
- NEW `src/eval_scripts/run_fulldim_ridge_shards.sh`, `src/eval_scripts/merge_fulldim_ridge_results.py`.
- Output: `results/fulldim_ridge_activation_to_fv/{shard_icl1..10,combined_*}`.

**Findings:**
- **BUG FIXED:** the final prompt token (`last_prompt_token`) is stored with
  `icl_example_index = None` (not 10) in the `4tokens` dir; only the 3 label roles use 10. Loader
  now resolves `None` for that role (`role_load_icl_index`). icl1–9 were unaffected.
- **Best cell: final prompt token @ layer 11, test_mse = 0.1161** (α≈3.2e4). Runner-up:
  query pre-label token @ L11 (0.1172). Both at the query position (ICL 10).
- **Layer profile (mean over all 31 positions):** clean bowl, **min at L11 (0.1325)**, best band
  L10–14; embedding L0 worst (0.196), slow degrade to L28 (0.149). Matches the prior layer≈8–12 result.
- **Best per role:** finaltok 0.1161 (L11) < pre 0.1172 (L11) < first 0.1252 (L11) < last 0.1257 (L13).
- **Query position dominates:** the strongest cells are all at ICL 10 (more accumulated context);
  early ICL demos decode worse. The query's final/pre-label tokens are the most FV-predictive reads.
- **α sanity:** bulk α ∈ 1e3–1e5 (peak 1e4), interior. The 11 "pinned" cells are all L0 (embedding)
  constant-feature positions (the ":" pre-label token / final token embed identically across tasks →
  "predict-the-mean" baseline 0.217, α irrelevant). No grid widening needed.

**Next:** Compare this direct-ridge floor (~0.116) against the joint-PCA reconstruction MSE
(different metric — see Open Q3); decide whether direct full-dim ridge becomes the canonical decoder.

**Blockers:** None. (Single A100 80GB; 3 shells time-shared → ~5s/cell-of-work, ≈ serial; the
split bought load-overlap + restartability, not GPU speedup. Real speedup needs >1 GPU.)

---

## 2026-06-10 — Stream B: direct sweep #3 — k_activations fixed=16, sweep k_FV

**Status:** DONE. Added `--fix_act_k` to `sweep_k_activation_to_fv_direct_log2.py` (mutually
exclusive with `--fix_fv_k`): pins k_activations and reinterprets the doubling grid as k_FV
(capped at fv_k_cap). Ran k_act=16 fixed, k_FV ∈ {1,2,4,8,16}, direct, 7 tasks, ICL 1–5.

**Command:**
- `sweep_k_activation_to_fv_direct_log2.py --output_dir .../activation_to_fv_direct_ols_multitask_top10_log2_fixedactk16_exclude_cc_pc --fix_act_k 16 --k_max 16 --icl_example_indices 1 2 3 4 5 --test_tasks <7>` → exit 0.

**Files changed:**
- `sweep_k_activation_to_fv_direct_log2.py`: NEW `--fix_act_k` mode (plot labels/title +
  run_config now mode-aware: sweep_variable, fix_act_k recorded). py_compile OK.
- NEW `results/k_sweeps/activation_to_fv_direct_ols_multitask_top10_log2_fixedactk16_exclude_cc_pc/`.
  k_sweeps now has 4 direct dirs.

**Findings (k_act=16, sweep k_FV, ICL5):**
- **Test MSE falls monotonically with k_FV; best = k_FV=16 (the cap) in 14/15 series.** No
  overfitting in the k_FV direction (unlike k_act). icl5/last: 0.1805(kFV1)→0.1259(kFV16).
- **Test MSE rides just above the recon floor** (gap ~0.01→0.03); the floor itself drops
  0.172→0.099 as k_FV grows. So k_FV is limited by how much FV variance you discard, NOT by
  overfitting — and the FV PCA caps at 16 (20 train tasks → rank ≤19, capped 16).
- **With 16 activation PCs as input, the regression recovers any k_FV target ~to the floor** —
  the gap only widens modestly as more FV directions are demanded.
- **Combined across all 3 sweep axes: joint optimum = (k_act=16, k_FV=16), the corner.** k_act
  peaks at 16 then overfits; k_FV improves monotonically up to its 16 cap.

**Next:** None pending — the three orthogonal cuts (coupled diagonal, fix k_FV, fix k_act) are done.

**Blockers:** None.

---

## 2026-06-10 — Stream B: all k_sweeps converted to DIRECT method; joint runs deleted

**Status:** DONE. Per user: overwrite every non-direct k-sweep with the direct method. Ran the
two missing direct configs and **deleted all three joint result dirs**. `results/k_sweeps/` is
now 100% direct (regression = project act→k_act PCs, FV→k_FV PCs, OLS between them).

**Commands:**
- `sweep_k_activation_to_fv_direct_log2.py --output_dir .../activation_to_fv_direct_ols_multitask_top10_log2 --icl_example_indices 1 2 3 4 5 --k_max 1024` (coupled, full 9 tasks) → exit 0.
- `sweep_k_activation_to_fv_direct_log2.py --output_dir .../activation_to_fv_direct_ols_multitask_top10_log2_fixedfvk16_exclude_cc_pc --fix_fv_k --icl_example_indices 1 2 3 4 5 --k_max 1024 --test_tasks <7>` → exit 0.
- `rm -rf` the 3 joint dirs (`activation_to_fv_ols_multitask_top10_log2{,_exclude_cc_pc,_fixedfvk16_exclude_cc_pc}`).

**Files changed:**
- NEW direct dirs: `activation_to_fv_direct_ols_multitask_top10_log2` (9-task coupled),
  `..._direct_ols_..._fixedfvk16_exclude_cc_pc` (7-task, k_FV pinned 16). The 7-task coupled
  direct dir already existed. DELETED the 3 joint dirs.
- `results/k_sweeps/` now holds exactly 3 dirs, all direct. (Joint *script*
  `sweep_k_activation_to_fv_ols_log2.py` kept as a tool; only its results were removed.)

**Findings (direct, k_FV fixed=16 — settles the "1 PC is enough?" question):**
- **k_act=1 → test MSE ≈ 0.18–0.19** (≈ predict-the-mean baseline ~0.21; floor@k_FV16 = 0.099),
  **k_act=16 → ≈ 0.126.** So a single activation-PC does NOT recover the 16-dim FV — you need
  ~16. The old joint plot's low-k_act minimum was ENTIRELY the 16 FV-basis features it appended
  to the inputs; with the honest direct features those vanish.
- Direct coupled optima unchanged from before (~0.126 at ICL5, k≈16; first-label edges to 32).

**Next:** None pending. All k-sweeps are direct.

**Blockers:** None.

---

## 2026-06-10 — Stream B: NEW "direct" k_activations→k_FV regression (coupled diagonal log2 sweep)

**Status:** DONE. User redefined the regression: instead of the joint-space setup (project both
activation and FV onto the concatenated [act-PCA | FV-PCA] basis), **project activations → k_act
PCs (input), FVs → k_FV PCs (target), regress R^{k_act}→R^{k_FV} directly**, reconstruct to
4096-d for MSE. First deliverable: the coupled diagonal log2 sweep (1,1),(2,2),(4,4),(8,8),
(16,16),(32,16),… i.e. k_FV = min(k,16). New standalone script; joint scripts untouched.

**Command:**
- `python src/eval_scripts/sweep_k_activation_to_fv_direct_log2.py
   --output_dir results/k_sweeps/activation_to_fv_direct_ols_multitask_top10_log2_exclude_cc_pc
   --icl_example_indices 1 2 3 4 5 --k_max 1024
   --test_tasks landmark-country word_length capitalize_first_letter synonym
   lowercase_first_letter capitalize antonym`  → exit 0; 15 series.

**Files changed:**
- NEW `src/eval_scripts/sweep_k_activation_to_fv_direct_log2.py` (also supports `--fix_fv_k`
  for the next step). py_compile OK.
- NEW `results/k_sweeps/activation_to_fv_direct_ols_multitask_top10_log2_exclude_cc_pc/`.
- WORKLOG (this entry), DECISIONS.

**Findings:**
- **Direct is ~0.002–0.003 HIGHER MSE than joint** everywhere (e.g. icl5/first 0.1261 vs
  0.1232; icl5/last 0.1259 vs 0.1239). Expected & correct: joint's feature vector secretly
  included the activation projected onto the FV basis (extra inputs), so joint numbers were
  mildly optimistic. Direct = the honest activation-space→FV-space regression.
- **Optimal k unchanged: ≈16** (first-label 32); diagonal bowl bottoms at (16,16), rises past.
  More ICL → better (ICL5 best ~0.126). pre-label noisier/flatter (icl2/pre wanders to k=256 on
  a near-flat curve), first/last-label clean minimum at 16–32.

**Next:** Per user's stated plan, sweep k_activations with k_FV fixed (use `--fix_fv_k` on the
new direct script) — the direct analogue of the joint `--fix_fv_k` run.

**Blockers:** None.

---

## 2026-06-10 — Stream B: sweep k_activations with k_FV pinned at 16 (`--fix_fv_k`)

**Status:** DONE. New flag `--fix_fv_k` on `sweep_k_activation_to_fv_ols_log2.py` holds the
FV-side PCs at `fv_k_cap` for EVERY k (instead of `fv_k = min(k, fv_k_cap)`), so the sweep
isolates **k_activations** with **k_FV pinned at 16**. Same k grid (1,2,4,…,1024), same
7-task test set (cc/pc excluded), layer 11, ICL 1–5.

**Command:**
- `python src/eval_scripts/sweep_k_activation_to_fv_ols_log2.py
   --output_dir results/k_sweeps/activation_to_fv_ols_multitask_top10_log2_fixedfvk16_exclude_cc_pc
   --fix_fv_k --icl_example_indices 1 2 3 4 5 --k_max 1024
   --test_tasks landmark-country word_length capitalize_first_letter synonym
   lowercase_first_letter capitalize antonym`  → exit 0; 15 series.

**Files changed:**
- `sweep_k_activation_to_fv_ols_log2.py`: NEW `--fix_fv_k` flag; plot tick labels + title +
  run_config + stdout now reflect the fv_k rule (derive fv_k from rows). py_compile OK.
- NEW `results/k_sweeps/activation_to_fv_ols_multitask_top10_log2_fixedfvk16_exclude_cc_pc/`.
- WORKLOG (this entry), DECISIONS.

**Findings (k_FV fixed = 16; best fv_test_mse per series):**
- **first/last-label tokens:** optimal **k_activations ≈ 16–32** (e.g. icl5/first k=32 0.1232,
  icl5/last k=16 0.1239). Clear bowl; rises past 32 (overfit).
- **pre-label token:** optimal **k_activations is tiny (1–8)** — adding activation PCs *hurts*
  (icl4/pre k=1 0.1299, icl3/pre k=1 0.1356, icl5/pre k=8 0.1248). The pre-label activation
  carries little task signal, so extra PCs are noise.
- **vs the coupled run:** identical for k≥16 (fv_k=16 in both); the only change is k<16, which
  is now *lower* (better) because the FV target is no longer shrunk. So pinning k_FV=16 mainly
  rescues the low-k_activations regime.
- **Headline:** with k_FV=16, the activation side saturates by **k_activations≈16**; first-label
  edges lowest at 32. Marginal value of activation PCs beyond ~16–32 is negative.

**Next:** If a single canonical config is wanted: layer 11, last/first-label token, ICL 5,
k_activations≈16, k_FV=16.

**Blockers:** None.

---

## 2026-06-10 — Stream B: log2 k-sweep excluding country-currency + product-company + k_sweeps reorg

**Status:** DONE. Re-ran `sweep_k_activation_to_fv_ols_log2.py` with the two weakest test
tasks dropped (`country-currency`, `product-company`) — the user's reason: their
train(multitask)-selected FVs perform much worse than task-specific head selection. Also
nested all k-sweep outputs under a new `results/k_sweeps/` parent for repo clarity.

**Command:**
- `python src/eval_scripts/sweep_k_activation_to_fv_ols_log2.py
   --output_dir results/k_sweeps/activation_to_fv_ols_multitask_top10_log2_exclude_cc_pc
   --icl_example_indices 1 2 3 4 5 --k_max 1024
   --test_tasks landmark-country word_length capitalize_first_letter synonym
   lowercase_first_letter capitalize antonym`
  (originally written to the flat path, then moved into k_sweeps/; fv_root, layer 11,
  fv_k_cap 16, k_min 1 all default = same as original 9-task run.) Exit 0; 15 series swept.

**Files changed:**
- NEW `results/k_sweeps/` parent; moved both runs in (orig 9-task + new 7-task), trimmed
  redundant `k_sweep_` prefix, fixed `output_dir`/`metrics_csv`/`plot_png` self-paths in both
  `run_config.json`.
- `sweep_k_activation_to_fv_ols_log2.py` + `sweep_k_activation_to_fv_ols.py`: default
  `--output_dir` now points inside `results/k_sweeps/`.
- WORKLOG.md (this entry), DECISIONS.md (reorg + finding).

**Findings:**
- **Dropping the two tasks barely moves the regression test-MSE** (best fv_test_mse within
  ~±0.001 of the 9-task run; some cells slightly *higher*). Best at ICL5: ~0.123–0.126.
- **Optimal-k structure is UNCHANGED:** bowl minimum at k≈16–32 (fv_k capped 16), overfit
  past 32. Same as the full-9-task run.
- **Key takeaway:** the tasks that are *bad for FV steering* (cc, pc) are NOT the tasks that
  are *hard to regress from activations*. The two performance notions are decoupled — so
  excluding them doesn't clean up the regression aggregate.

**Next:** Decide whether the regression metric is the right lens for the cc/pc weakness
(it isn't — that weakness shows up in steering, not activation→FV reconstruction MSE).

**Blockers:** None.

---

## 2026-06-10 — Coordinator: verified train+test build + corrected degeneracy wording

**Status:** Verified the train+test FV build is complete and sound; corrected two
imprecise claims in the prior entry's degeneracy finding. No new artifacts produced.

**Commands run:**
- `ls`/manifest inspection: 29/29 `train_test_selected/<task>/<task>_function_vector.pt`
  + `fv_manifest.json` (heads_path = `multitask_aie_heads_all_tasks`, n_top=10,
  fv_root=`gptj_fv`) + `heads.pt`/`heads_metadata.json`/per-task `selected_heads.json`.
- torch diff of `train_selected` vs `train_test_selected` over all 29 tasks:
  **global max|Δ| = 0** on the `function_vector` tensors (exactly equal).
- Confirmed head SET identical, rank ORDER differs (train: (9,14) first; train+test:
  (15,5) first). `cmp` on the `.pt` files reports DIFFER — that's dict metadata only.

**Files changed:**
- WORKLOG.md (this entry + tightened degeneracy bullet below).
- DECISIONS.md: table cell train+test = **BUILT**; degeneracy finding wording corrected.

**Findings:**
- Build correct; all three methods now complete for GPT-J.
- Degeneracy is real but the prior "byte-identical FVs" wording was wrong: the FV
  *tensors* are exactly equal; the *files* are not (metadata). Set-identical ≠ list-identical.

**Next:** To make train vs train+test meaningful, rebuild at n>10 (they first differ ~n=11).
Open Q1 (held-out eval head set) is moot at n=10 — train and train+test give the same FVs.

**Blockers:** None.

---

## 2026-06-10 — train+test FVs built; per-task head metadata; degeneracy finding

**Status:** All three methods now have FVs for all 29 tasks under
`results/function_vectors/gpt-j/`. Each task folder has a `selected_heads.json`.

**Commands run (by user + coordinator):**
- Built train+test FVs (user ran the build command) → 29 real
  `train_test_selected/<task>/<task>_function_vector.pt` + `fv_manifest.json`.
- `python src/eval_scripts/write_fv_head_metadata.py --model_root results/function_vectors/gpt-j --n_top_heads 10`
  → 87 `selected_heads.json` (29 × 3 methods).

**Files changed:**
- NEW `src/eval_scripts/write_fv_head_metadata.py` (writes per-task head metadata; idempotent).
- NEW `results/function_vectors/gpt-j/train_test_selected/` (29 FVs + manifest) and
  `selected_heads.json` in every task folder of all three methods.

**Findings:**
- FV norms sane (~30–47), consistent with the FV paper.
- **DEGENERACY at n=10:** `train_selected` and `train_test_selected` select the
  *same set* of top-10 heads (rank order + CIE scores differ slightly — train ranks
  (9,14) first, train+test ranks (15,5) first — but membership is identical through
  n=10). Because the FV is an order-independent sum over the selected heads, the
  `function_vector` tensors are **exactly equal for all 29 tasks** (verified: global
  max|Δ| = 0; norms match). NB: the `.pt` *files* are not byte-identical — `cmp`
  differs on the dict metadata (`top_heads` order, paths) — only the tensors match.
  They diverge only at larger n (34/40 overlap at n=40). So a train-vs-train+test
  comparison at top-10 is meaningless; to study the selection-leakage effect, raise
  n_top_heads or use a per-task metric. `task_specific` is genuinely distinct.

**Next:** When studying train vs train+test, use n>10 (they differ from ~n=11+).
Optionally rerun `write_fv_head_metadata.py` after building FVs for new models.

**Blockers:** None.

---

## 2026-06-10 — Coordinator: FV reorg + provenance resolution

**Status:** Located all three head-selection methods + their FVs; built an organized
FV folder; resolved the split-provenance question; staged (did not run) the
train+test FV build.

**Commands run:**
- Verified mean-activation cache: all 29 tasks have `*_mean_head_activations.pt` in
  `results/gptj_fv` (so train+test FV build needs no forward passes).
- Built `results/function_vectors/{task_specific,train_selected,train_test_selected}/`
  via relative symlinks into existing caches (29/29 FV links + head-set links resolve).

**Files changed:**
- NEW `results/function_vectors/gpt-j/{task_specific,train_selected,train_test_selected}/`
  tree (model-nested for future models) + `README.md`.
- `compute_all_task_fvs_from_multitask_heads.py`: added `--tasks` (subset sharding) and
  `--manifest_name` (so parallel shards don't clobber one manifest). py_compile OK.
- DECISIONS.md: 3-method table, FV-folder decision, provenance RESOLVED, 4-shard build commands.

**Findings:**
- **Three methods located** (see DECISIONS table). task-specific → `gptj_fv/`;
  train → `gptj_fv_multitask_top10/`; train+test → head set exists, **FVs not built**.
- **Provenance RESOLVED:** all-tasks head set was computed on `query_split=valid`
  (per-task files suffixed `_valid`; runner passes `--query_split valid`). Metadata
  `query_split=train` is a stale `--reduce` default. Both head sets share valid/train.
- All-tasks FV build is cheap (activations cached) → single-process, no big GPU needed.

**Next:** Run the train+test FV build command in DECISIONS.md to populate
`function_vectors/train_test_selected/`. Then the 3-way comparison is fully aligned.

**Blockers:** None.

---

## 2026-06-10 — Coordinator: state reconstruction

**Status:** Bootstrapped WORKLOG/DECISIONS from artifacts (both were empty). No
worker streams had registered. Reconstructed the two active experiment lines by
reading the uncommitted scripts in `src/eval_scripts/` and the `results/` tree.

**Findings (verified against artifacts, not just inferred):**

- Two distinct research lines are in flight (see streams below).
- `results/heldout_multitask_head_eval/` contains **all 9 test tasks** (antonym,
  capitalize, capitalize_first_letter, country-currency, landmark-country,
  lowercase_first_letter, product-company, synonym, word_length) + aggregate
  summary. Complete.
- **Split provenance mismatch (needs resolution):**
  - `results/multitask_aie_heads/` (20 train tasks): metadata `query_split=valid`,
    `demo_split=train`.
  - `results/multitask_aie_heads_all_tasks/` (29 tasks): metadata
    `query_split=train`, `demo_split=train` — **but** the runner
    `run_multitask_aie_all_tasks.sh` passes `--query_split valid`. Artifact
    metadata and the shell script disagree → the all-tasks artifact may not have
    been produced by that script, or the script was edited after the run. Confirm
    before trusting the all-tasks head set.
- Linear k-sweep (`sweep_k_activation_to_fv_ols.py`) **was never run** — no
  `results/k_sweep_activation_to_fv_ols_multitask_top10/`. Only the log2 variant ran.
- Two leftover smoke dirs: `results/joint_pca_activation_to_fv_regression_smoke`,
  `results/pca_abstractive_fv_activation_scatter_smoke`.

**Next:** Owners to claim the two streams below and resolve the open questions in
DECISIONS.md. Nothing committed yet — all work is uncommitted/untracked.

**Blockers:** None for coordination. See per-stream blockers.

---

## Stream A — Multitask AIE heads → function vectors

**Owner:** (unclaimed)
**Status:** Core runs complete; held-out eval complete on train-only heads.

**Question:** Do attention heads selected by causal indirect effect (CIE)
aggregated *across many tasks* yield function vectors that transfer to held-out
tasks as well as task-specific head selection?

**Pipeline (inputs → outputs):**
1. `compute_multitask_top_aie_heads.py` (MODIFIED) — computes per-task CIE per
   (layer, head), prompt-weighted aggregate, ranks top-N. New: sharding
   (`--num_shards/--shard_index`), `--reduce`, `--all_split_tasks`,
   `--abstractive_only`, `--save_per_prompt_effects`.
2. `run_multitask_aie_all_tasks.sh` (NEW) — orchestrates sharded run over all 29 tasks.
3. `select_heads_from_cie_subset.py` (NEW) — re-aggregate any task subset from
   cached per-task CIE without recompute (requires per-prompt effects on disk).
4. FV builders: `compute_task_fv_from_multitask_heads.py` (single),
   `compute_all_task_fvs_from_multitask_heads.py` (batch, one model load, manifest),
   `compute_fv_from_selected_heads.py` (flexible head source). **Overlap — pick one
   primary; see DECISIONS open Q.**
5. `evaluate_heldout_multitask_head_fvs.py` (MODIFIED) — multitask vs task-specific
   FV steering effectiveness by layer; now emits per-task PNGs.

**Outputs that exist:**
- `results/multitask_aie_heads/` — 20-task (train) head set, top-40. No per-prompt effects.
- `results/multitask_aie_heads_all_tasks/` — 29-task head set, top-40, per-prompt effects saved.
- `results/gptj_fv_multitask_top10/` — FVs for all 29 tasks from the train-only top-10 heads + manifest.
- `results/heldout_multitask_head_eval/` — 9 test tasks, multitask vs task-specific, + aggregate summary.

**Loose ends:**
- train+test FVs not yet built — run the command in DECISIONS.md (`To run`).
- Held-out eval used the **train-only** head set, not the newer all-tasks set. Re-run to compare?
- 3 overlapping FV builders; train-only baseline can't be re-subset (no per-prompt effects saved).
- (RESOLVED 2026-06-10) split provenance: all-tasks heads are on `query_split=valid`; metadata field is stale.

**Next:** Build train+test FVs into `function_vectors/train_test_selected/`; decide
whether to re-eval against all-tasks heads; consolidate FV builders.

**Blockers:** None known.

---

## Stream B — Regress layer activations → function vectors (joint PCA)

**Owner:** (unclaimed)
**Status:** Baseline, ICL, ICL-ridge, log2 k-sweep, and layer sweeps complete.

**Question:** Can a task's function vector be linearly predicted from intermediate
layer activations in a shared low-rank PCA space? What k, layer, token role, and
ICL position decode best, and does ridge help?

**Pipeline (inputs → outputs):**
- PCA bases: `pca_abstractive_fv_activation_scatter.py` (existing, layer 11),
  `pca_abstractive_icl_examples_fv_activation_scatter.py` (NEW, per-ICL 1–4).
- Regression: `regress_activation_to_fv_joint_pca.py` (existing, OLS, incl.
  last_prompt_token role), `_icl.py` (NEW, per-ICL, drops last_prompt_token),
  `_icl_ridge.py` (NEW, ridge w/ per-cell alpha via LOO-task CV).
- Sweeps: `sweep_k_activation_to_fv_ols.py` (linear k, NOT RUN),
  `sweep_k_activation_to_fv_ols_log2.py` (log2 k, decouples activation-k from
  fv_k cap=16), `sweep_layer_activation_to_fv_ols.py` (all 29 layers; ran at k=5,
  16, 32 as separate dirs).

**Outputs that exist:**
- `results/joint_pca_activation_to_fv_regression{,_icl,_icl_ridge}/`
- `results/k_sweep_activation_to_fv_ols_multitask_top10_log2/`
- `results/layer_sweep_activation_to_fv_ols{,_full_dim_k16,_full_dim_k32}/`
- `results/pca_abstractive_icl_examples_fv_activation_scatter{,_multitask_top10}/`

**Findings (provisional, from result CSVs — re-verify before citing):**
- Best layer ≈ 8–12 (layer 11 near-peak); embedding layer worst; later layers degrade.
- Activation-side k benefit saturates ≈ 16–32; test MSE plateaus above the fv_k=16
  reconstruction floor.
- Joint-PCA baseline: test MSE minimized around k≈6–10 depending on token role.

**Loose ends / inconsistencies to resolve:**
- Two non-comparable MSE metrics in use: joint-PCA-space MSE (regress scripts) vs
  reconstructed 4096-d FV-space MSE (sweep scripts). Pick one primary.
- ICL index ranges differ across scripts (e.g., 1–4 vs 2–5). Standardize.
- Some scripts use `gptj_fv`, others `gptj_fv_multitask_top10` (the `_multitask_top10`
  suffix). Decide which FV target is canonical, or document both as intentional.
- Layer sweep is split across 3 k-specific dirs — no single combined output.
- Linear k-sweep script unrun; decide if log2 coverage suffices.

**Next:** Decide canonical metric + FV target + ICL range; consider one combined
layer×k sweep; clean up smoke dirs.

**Blockers:** None known.
