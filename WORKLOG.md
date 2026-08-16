# WORKLOG

Coordination log for in-flight experimental work on the Function Vectors repo.
Newest entries at top. One stream per active line of work.

---

## 2026-08-15 — SANDBOX ext_steerability phase 1: pooled sparse head selection on extended tasks

**Owner:** Claude Code background session (Train Test Split Works Check), CPU pod + own pods
fv-ext-{1..30} (27× RTX 5090 + 3× 4500; ALL TERMINATED; ~1 h 50 min wall, ≈$55).
**Status:** PHASE 1 DONE — awaiting user review before held-out (phase 2). **SANDBOX.**

**What (user spec, adjudicated in-chat):** filter the 138 extended tasks to the 90 with
6-shot sampled-exact-match acc ≥ 0.30 (stored n-shot sweep) → `task_splits/
extended_steerable_90.json` (seed-42 shuffle, 72 train / 18 heldout). Prompt sets
`dataset_files/isolation_prompts_ext/` (150 fixed-10-shot train prompts + 50 paired test
queries × {zeroshot, sametask_shuffled10, mixedtask10}; example split merge_valid_into_train
79/21; mixed demos from TRAIN tasks only; 5 tasks test-capped 42–47; 90/90 validation
battery passed). Pooled sparse opt (1b), train metric zero-shot, 100 pts/task = 7200,
inject @L9; λ ∈ {0.005,0.01,0.05,0.2} by 5-fold TASK CV (weighted-c fold eval, strict best);
selection c > 0.8. NEW `src/sandbox/ext_steerability/{make_ext_split, generate_isolation_
prompts_ext, run_ext_capture, train_sparse_pooled_ext, eval_ext, plot_train_eval}.py`.

**RESULTS:** λ=0.005 (CV means .570/.552/.474/.302); **39 heads selected** (no fallback;
early stopping fired). Overlap: 13/40 with canonical varicl top-40, **15/23 with
vanilla_sparse_opt23**; canonical trio (9,14),(12,10),(15,5) all at c≈1. Train-task steering
(v_A = Σ h̄_A over the 39 heads, α=1, 50 queries, mean over 72 tasks):
zero-shot base .023 → best-layer **.602** (L9 .490), 56/72 tasks ≥ .4;
mixed-task base .078 → **.587** (L9 .469), 55/72; shuffled-10 base .469 → **.765**, 70/72.
Outputs: `artifacts/sandbox/ext_steerability/` (90 means, pooled_sparse/ 20 folds +
coeffs_final + selection.json, 72 eval_headset.json), `results/sandbox/ext_steerability/`
(train_tasks_summary.csv, train_tasks_bars.png). **Next:** phase 2 (18 held-out tasks)
after user review — eval_ext.py runs unchanged on heldout_tasks.

**PCA-subspace mean-activation trial (2026-08-15, this stream; NULL RESULT):** built the
task subspace from the 72 task-specific FVs (per-task diag_headhunger c>0.8 sets; centered
PCA k90=42; `build_pca_subspace43.py`, gated). User construction v_A(ℓ) = μ_FV +
U42ᵀU42(z̄_A(ℓ) − μ_FV) vs raw-mean control, α∈{1,2,4}, layers 0–27, zs+mixed, 72 train
tasks (`eval_pcasub_ext.py`, pods fv-pcs-{1..24} TERMINATED). Dose-matched (best-α) the
projection is indistinguishable from the raw mean (zs .544 vs .545; mixed .547 vs .583)
and both trail the pooled-39 head FV (.602 @α=1). At α=1 pcasub reads worse only via norm
shrinkage (best-α piles at 4). Side-finding: mean-act is UNDER-DOSED at α=1 — best-α
raw-mean reaches .545/.583; algorithm comparisons involving mean-act need an α sweep.
Third instance of variance-chosen ≠ steering-chosen directions. Also: centered PCA of the
72 task-specific FVs: 50%/80%/90%/95% var at k=11/30/42/52 (`taskspecific_fv_pca.png`).
Outputs: per-task diag_pcasub.json, pcasub_train_summary.csv, pca_subspace43.pt.

**Failing-task diagnosis (2026-08-15, this stream; 16 tasks with pooled-39 zs < 0.4):**
NEW `diag_taskspecific_failing.py` — task-specific sparse (corrected recipe, λ∈{.005,.05},
best kept) per failing task, own-FV zs layer sweep. ***RESULT: it's a pooled-SELECTION
failure, not task unsteerability.*** All 16 improve with their own heads; 13/16 reach ≥0.44
(city-continent .38→.92, hour_after_time .28→.90, days_in_month .22→.76, noun_possessive
.00→.70); only the arithmetic trio stays low even at upper bound (plus_ten .34,
year_to_decade .30, round_down .42). Task-specific sets are LARGE (median ~65, up to 154
heads) with only ~10–25 heads overlapping the pooled 39 → the pooled objective (mean log-prob
+ global L1) never pays for heads that help only minority tasks — easy tasks dominate the
gradient and λ. Norm-energy coverage does NOT separate failing from passing tasks (E39 ≈
0.6–0.9 both), so the failure is functional, not magnitude. Supports the staged/banded
selection trial running in the parallel fork (Batched Train Method). Outputs: per-task
`diag_taskspecific.json`, `results/sandbox/ext_steerability/failing_analysis_taskspecific.csv`.
Pods fv-diag-{1..4} TERMINATED (~30 min, ≈$2).

**Head-hungriness predictors (2026-08-15, fork "Indicators of high number of attention head
tasks"):** NEW `diag_headhunger.py` — task-specific sparse on ALL 72 train tasks + top-k-by-c
sweep at each task's best layer; k90 = min heads for ≥90% of own zs ceiling. 12 fv-hh pods
(~40 min, ≈$8), 72/72 done. ***acc6 does NOT predict head count*** (ρ=−0.02 vs log2 k90;
median k90 = 40 in EVERY competence band) — competence predicts WHETHER a task steers
(ρ=0.62 own-ceiling) but not how many heads it needs, so a 6-shot prune cannot remove
head-hungry tasks. What does predict k90 (a-priori, dataset-only): output entropy ρ=0.42,
label token count ρ=0.44, their product ρ=0.50. k90 is also what the pooled selection
under-serves: ρ(k90, pooled/own ratio) = −0.47. Distribution reality check: k90 median 40,
IQR 40–80 — needing ~40+ heads is the NORM; genuinely few-head tasks are the minority
(18/69 at k90 ≤ 20, isolable a-priori by out_entropy ≤ 3.5 & single-token labels →
median k90 20). Caveats: K-grid coarse {…20,40,80…}, n=69, descriptive. Outputs:
`failing_analysis_{features,headhunger}.csv`, `failing_analysis_headhunger_scatter.png`,
per-task `diag_headhunger{.json,_c.pt}`. Branch `claude-hh-indicators` (ext branch was
checked out by the parent session's worktree — kept off it to avoid a race).

**A-priori prune-rule search CLOSED + micro-probe result (2026-08-15, same fork):**
Exhaustive threshold search over ~15 a-priori features (dataset stats, n-shot curve shape
incl. acc2, label tokens, shuffled-10 unsteered baseline, FV-distinctiveness cos, readout
gap) against BOTH targets (pooled-39 failing 16; big-head∩failing 9): ***no reliable rule
exists*** — best LOO-CV ~9/16 at 3–5 collateral; the last ~5 failures are invisible to
every feature family (incl. single-token-label + high-competence failures). Two informative
NULLS: c-mass coverage of the pooled 39 (from trained task-specific c) does NOT predict
failure (ρ=0.04) — head-identity overlap is not the mechanism; cos(v39, v_own) is the best
mechanistic signal but partial (ρ=0.37). Necessary-condition fact: all 9 big-head∩failing
tasks have label_tokens ≥ 1.36.
***WORKING ALTERNATIVE — micro-probe:*** inject the pooled FV at layers {6,9,12} only, 50
zs queries (~2 s/task GPU): reproduces the full 28-layer verdict with **0/16 failures
missed, 4/56 good false-flagged** (those 4 peak outside the probed layers; rescue = full
sweep on probe-flagged only ⇒ exact failing set at ~1/10 total eval cost). Validated
against stored evals (`failing_analysis_mechanism.csv`); 10-query variant degrades
(miss ~1–2) — keep 50 queries.

**Rule v2 + minus_ten mechanism demo (2026-08-15/16, same fork):** pooling all 90 labeled
tasks (72 train + 18 heldout; 21 pooled-39 zs-best<0.4 failures) + NEW copy-modify feature
`overlap_tok` (mean fraction of answer tokens already present in the input) →
**`label_tokens > 2 (strict) AND overlap_tok ≥ 0.058` prunes 9/21 failures at 0 good lost**
(LOO-CV over the AND-pair family: 9/21 at 0). CAVEATS: feature idea derived from heldout
failures; a clean train-only fit picks a different rule (acc1≤0.12 & label≥1.37) that
catches only 1/5 heldout OOS — rule v2 is a hypothesis pending prospective validation, and
the heldout set is now spent. `failing_analysis_rule_v2.csv`. CPU greedy-generation demo
(minus_ten, 6 zs prompts): steered model outputs a VERBATIM COPY of the query (never −10);
BPE chunks 5-digit numbers arbitrarily; −10 usually edits only the last token (borrow cases
re-chunk entirely) — pooled heads carry copy/format, not the numeric edit. Steering at cue
+ ALL response tokens does NOT fix it (still copies; L12 corrupts digits) — the edit
operation is absent from the vector, not under-injected.

**Pruned-pool refit (2026-08-16, same fork; pods fv-pf-{1..10} RTX PRO 4500, 1 never got
SSH → shard re-routed, ALL TERMINATED ~1.5 h, ≈$12):** removed the 21 failures, re-split
the 69 survivors seed-43 → 55 train / 14 heldout (`task_splits/
extended_steerable_69_prunedfail.json`); reran the FULL pooled sparse protocol (same 4λ ×
5-fold task CV, final retrain, c>0.8) in fresh tree `artifacts/.../prunedfail_seed43/`
(means symlinked). **λ=0.005 again, 37 heads (31/37 shared with the old 39); CV mean .657
(vs .570 on unpruned 72 — pool composition, not better selection). Eval (best-layer):
train 55 zs .748 / mix .684; HELDOUT 14 zs .734 / mix .779; 0/69 tasks below 0.4 zs.**
Same 69 tasks under the old 39-head set: zs .735/.720 — pruning buys only ~+.015 mean but
eliminates the failure tail entirely on a fresh split; biggest per-task movers ±.18
(translations up, animal_plant_object down). `prunedfail_seed43_summary.csv`.

**Pool cleanup (2026-08-16, user decision):** the 21 failures moved OUT of the working pool —
`dataset_files/extended_tasks/head_intensive_pruned_tasks/` (own manifest w/ pooled-zs labels
+ README; main manifest now n_tasks=117 with pruned_note; make_ext_split --expect_n on the
pruned manifest = 69). Working pool treated as if these tasks never existed; revisit later.
Regenerated the CANONICAL figures on the 117 via the repo scripts (patched:
plot_extended_nshot_bar.py now filters to manifest tasks; catalog title uses live count):
`results/general/extended_tasks_nshot_sweep/{nshot_bar_6shot.png, extended_tasks_catalog.png,
task_pool_117_summary.csv}` — 69 pass ≥0.30 (49 new + 20 original; 55 train / 14 heldout
under seed-43), 48 below threshold; mean acc6 .416 pool / .617 passing. Catalog sections
now 32/27/25/33. The 138-task versions live in git history (commit 4048539).

**FV dimensionality analysis (2026-08-16, session "Geometry Analysis of the Function Vectors";
pods fv-pp-{1,2} RTX PRO 4500):** per-prompt FVs captured for ALL 69 tasks (150 fixed 10-shot
train prompts each, 37-head prunedfail_seed43 set) → reusable intermediates
`artifacts/69_task_run/perprompt_fvs/<task>.pt` {fv (150,4096) fp16, raw head acts (150,37,256)
fp16, sel_flat, prompt_index}; NEW `src/eval_scripts/capture_69_perprompt_fvs.py` (linearity
gate kept). Analysis (55 TRAIN tasks only, centered PCA, float64 CPU SVD; consistency gate
raw-mean vs means.pt cos=1.000): task-mean FVs 90%@24 / 95%@32 of 55 PCs; pooled per-prompt
stack (8250×4096) 90%@239 / 95%@556, **stable rank 3.0 raw / 5.7 centered**; within-task
median 90%@85 / 95%@109 of 150. NEW `src/eval_scripts/plot_69_fv_dimensionality.py` →
`results/69_task_run/FV_dimensionality_analysis/{fv_dimensionality.png, spectra.npz,
summary.csv}`. Launch lessons: worktree lacks untracked isolation_prompts_ext (pass
--prompts_root to main checkout); volume script cp is not read-atomic across pods (stage, then
launch); pkill of a python child leaves its wrapper to touch .failed (kill the process group).

**Sparse PC-direction selection (2026-08-16, same session; pods fv-pc69-{1..10} RTX PRO 4500,
ALL TERMINATED, ~2h, ≈$15):** user-adjudicated design: uncentered PCA of the 55-train-task
per-prompt FV stack, TOP-512 dictionary, pooled c∈[0,1]^512 with steering NLL + λ‖c‖₁
(v = Σ cᵢ (v_task·PCᵢ) PCᵢ, inject @L9), same λ-CV protocol as head selection; deployed
vector = unweighted projection onto c>0.8 PCs. NEW `src/eval_scripts/{build_69_pc_basis,
train_sparse_pcs_69, eval_69_pcproj, plot_69_pcproj_results}.py`; artifacts
`artifacts/69_task_run/pc_sparse/`. λ=0.005 (CV .600/.591/.476/.264), **46 PCs selected**
(c_max=1.0; ranks 0–64, median 22 — essentially the TOP of the FV spectrum, unlike prior
variance≠steering precedents, expected since the basis IS FV-derived). **RESULT — train ≈
kept, heldout COLLAPSES:** zs train .723 vs .748 full-FV; heldout .461 vs .734 (mix .487 vs
.779; shuf holds .789 vs .807). Mechanism: heldout FVs stick out of the train-task subspace —
cos(v_proj, v_full) train .992 vs heldout .908 (min .71 ag_news), and heldout zs drop tracks
the lost component at Spearman −.84 (uppercase_word cos .84 → .98→.12). The 46-dim
"steering subspace" is train-task-specific, NOT a universal FV subspace at this dimension.
Figures/tables: `results/69_task_run/FV_dimensionality_analysis/{pc_sparse_summary.csv,
pc_sparse_bars.png, pc_selection.png}`; paired-bars view in
`results/69_task_run/FV_dimensionality_reduction/zeroshot_full_vs_projected.png`.

**Span-coverage debug check (2026-08-16, CPU; confirms the mechanism):** energy of each task
FV inside the top-k train-PC span (`debug_69_span_coverage.py` →
`FV_dimensionality_reduction/debugging/{span_coverage.csv,.png}`). Train FVs: 0.9999 in
top-512 (they define the basis). Held-out: mean 0.94 top-512 / 0.83 in the 46 selected;
the six lowest-coverage tasks (ag_news .80, uppercase_word, person_place_thing, first_digit,
pos_label, initials_two_words ≈.89–.91) are exactly the collapse tasks; coverage-vs-zs-change
Spearman .84. OUTLIER: english-french (top512 .97, sel46 .87) still collapses .60→.08 —
coverage is necessary, not sufficient; the lost 13% appears to carry the payload
(speculation: projection pulls it toward its train translation siblings).

**All-512 oracle probe (2026-08-16, pods ×2 ~25 min ≈$0.6; REVISES the interpretation):**
projecting heldout FVs onto the ENTIRE 512-PC dictionary (the ceiling of any selection)
recovers most of the collapse: heldout zs mean **.677 vs .461 (46 PCs) vs .734 (full FV)**;
uppercase_word .12→.94, english-french .08→.58, first_digit .58→.90. Residual losses sit in
the lowest-coverage tasks (ag_news .48, person_place_thing .48, initials_two_words .46; cov
.80–.91). So the DICTIONARY mostly contains the payload; the 46-PC SELECTION dropped
directions that are ~useless for train yet critical for heldout — not an optimizer bug: a
train-only objective cannot see those directions matter. Two-part conclusion: (1) steering-
sufficient subspace for unseen tasks is closer to a few hundred dims of the train span than
46; (2) the last ~6% off-span energy still costs the lowest-coverage tasks up to half their
accuracy. `debugging/all512_oracle_probe.csv`.

**Promotion out of sandbox (2026-08-16, user decision):** the pruned-pool refit results are the
first entry in NEW tracked bucket `results/69_task_run/` (constant TASK69_RUN_DIR in
utils/paths.py) — sparse opt on the zero-shot train metric is now considered working, no longer
sandbox. `train_test_generalisation/`: train_heldout_summary.csv (renamed from
prunedfail_seed43_summary.csv) + two-panel train/heldout bars per test setting via NEW
`src/eval_scripts/plot_69_task_run_bars.py`: zeroshot (.75/.73 steered vs .03/.09 base),
mixedtask10 (.68/.78 vs .08/.18), shuffled10 (.82/.81 vs .48/.48).

---

## 2026-08-13 — SANDBOX: task-specific isolation upper bound (29 tasks × 3 algos × 3×3 metrics)

**Owner:** Claude Code background session (isolation-methods stream), CPU pod + own pods
fv-iso-ub-{1..10} (10× RTX 5090 $0.99/hr; ALL TERMINATED progressively; ~6 h wall, ≈$60).
**Status:** DONE. **SANDBOX.**

**What (user spec, adjudicated in-chat):** per-task steering upper bound — every isolation
algorithm (write_up/isolation_methods_levers.md) fit task-specifically on that task's 150
train prompts (dataset_files/isolation_prompts/), crossing 1a CIE (top-10 AND top-40) and
1b sparse opt with 3 train metrics {zeroshot, sametask_shuffled10, mixedtask10}; 1c
per-layer mean act; eval on 30 paired test prompts × 3 settings × layers 0–27, α=1,
full-label teacher-forced acc + unsteered baselines. λ by 5-fold CV over 150 prompts.
NEW `src/sandbox/isolation_upper_bound/{generate_train_variants,run_task,
plot_isolation_upper_bound}.py`; train-side metric prompt files added to isolation_prompts/.
Gates: prompt-render, W_O linearity (capture-time assert), sparse smoke, CIE advisory
(sentiment top-10 head overlap 8/10 vs stored gptj_fv indirect effects), baselines sane.

**FINDINGS — CORRECTED 2026-08-14 (sparse stage rerun; results below OVERWRITE the broken
first pass per user instruction, old numbers only in branch history):**
The first pass's sparse collapse was a BUG (user caught it): pooled-run hyperparameters
unscaled — 135 points × batch 128 = ~2 steps/epoch × 30 epochs = ~58 steps, training
truncated (best_epoch=29 everywhere, c_max ~0.8), so c>0.8 selected 0 heads in 61/87 runs
and the largest-λ-within-1pt rule then picked λ=0.5 (73/87). Fix (user-adjudicated): lr
0.03 / 60 epochs / patience 8 (~120 steps, smoke: c separates to 1.0/0.0, es fires);
fold eval on the WEIGHTED c vector; λ strict-best (grid {0.005,0.01,0.05,0.2}); final
product c>0.8 with top-10-by-c non-empty fallback. Rerun on 32 pods (~2.5–3.5 h, ≈$75);
87/87 finals: 0 empty, median 36 heads, 7 fallbacks, λ mostly 0.005 (73/87).

**Corrected results (mean best-layer acc over 29 tasks; baselines zs 0.015 / shuf10 0.410
/ mixed 0.043):**
- **Sparse opt (matched train metric) wins every test setting:** zs 0.625 (train=zs),
  shuf10 0.670 (train=shuf10), mixed 0.623 (train=mixed) — consistent with the pooled
  sparse-vs-top-40 result. Train/test metric MATCHING matters for sparse (mismatched
  drops up to 0.24); CIE is metric-robust by comparison.
- **CIE top-40 second:** zs 0.578, shuf10 0.648, mixed 0.561 (train metric zeroshot ≈
  mixedtask10 > sametask_shuffled10; top-40 > top-10 by ~0.1).
- mean_act mid-pack (zs 0.283, shuf10 0.577, mixed 0.340) — no training, no metric lever.

**Steerability-predictor analysis (2026-08-14, forked session; installed here):** y = best zs
steered acc over products; predictors: clean-10-shot acc (NEW eval `eval_clean10.py` on the
same 30 test queries, correct-label demos), shuffled-10 baseline, npc90/centered-SR of
per-prompt FV stacks (sparse23), max/top-10 task-specific CIE (zs), label token count,
output cardinality/entropy, FV norm. **RESULT: clean10 Spearman ρ=0.96** (fail median 0.20
vs pass 0.80); shuffled-10 baseline ρ=0.47 (a derivative of competence); everything else
|ρ|≤0.24 n.s. Steered-best/clean10 ratio ≈0.8–1.2 for all tasks with clean10 ≥ 0.45: the
single-vector zero-shot upper bound ≈ the model's own 10-shot ceiling; the "unsteerable"
tasks are tasks GPT-J cannot do at 10 shots under the full-label readout (letter-surgery
family + synonym). **A priori rule: clean10 < ~0.4 ⇒ do not expect zero-shot steering.**
Robustness: clean 6-shot competence gives ρ=0.958 (vs 0.960 at 10 shots; clean6~clean10
ρ=0.98, identical failing set) — the competence screen needs only ~6 shots
(`eval_clean6.py`, `clean6_competence.json`).
Files: `analyze_steerability_predictors.py`, `eval_clean10.py`, `eval_clean6.py`,
`results/sandbox/isolation_upper_bound/steerability_predictors_{scatter.png,full.csv}`,
`artifacts/.../clean{10,6}_competence.json` (pods terminated).
- **Infra lessons (bare runpod/pytorch pods):** no HF_HOME → 10 pods each DOWNLOADED gpt-j
  from hub (looked like a lock deadlock); full cache copy overflows 40GB container disk;
  fix = per-pod local HF cache with SYMLINKED blobs to the volume (structure local → locks
  local; weights read from volume). Long-prompt tasks (commonsense_qa ~1200 tok) OOM fixed
  batch sizes → all batching token-length-aware (`auto_batch`).

**Outputs:** `artifacts/sandbox/isolation_upper_bound/<task>/` (means, cie_×3, sparse_×3
incl. fold artifacts, eval_results.json), `results/sandbox/isolation_upper_bound/`
(29 grid PNGs in figures/, summary.csv 870 rows). Branch `claude-sandbox-isolation-upper-bound`.
**Next:** user to direct (candidate: rerun λ selection with strict-best rule / non-empty
constraint — fold artifacts cached, only 87 retrains + evals needed).

---

## 2026-08-12 — Isolation-methods study prep: dataset expansion + prompt sets (CPU)

**Owner:** Claude Code background session (isolation-methods stream). **Status:** DONE.
**What:** (1) `write_up/isolation_methods_levers.md` — reference doc for the study levers
(3 isolation algorithms stated generically over the metric-to-optimise lever, data lever,
success-metric lever, failure hypotheses H1–H3, evidence snapshot). (2) 7 datasets expanded
via parallel agents to give ≥500 train examples under the new no-valid split; country tasks
left as-is; see DECISIONS 2026-08-12 for numbers/backups. (3) `merge_valid_into_train` flag
in `split_icl_dataset`/`load_dataset` (gated: test membership bit-identical, train = old
train ∪ valid). (4) `dataset_files/isolation_prompts/` generated + independently verified
(29/29 tasks pass: pair-membership, query distinctness, paired test queries across settings,
shuffle = exact permutation, mixed-task demos correct in source task; pools deduped by input
— catches product-company dup inputs and national_parks multi-state contradictions).
**Files:** `dataset_files/generate/generate_isolation_prompts.py` (NEW),
`src/utils/prompt_utils.py` (flag), expanded jsons + backups, `write_up/isolation_methods_levers.md`.
**Next:** user to direct (task-specific FV training under different isolation algos on the
new prompt sets).

---

## 2026-08-10 — SANDBOX: sparse-optimization selection over 83 FV-stack PC directions

**Owner:** Claude Code background session (sparse-pcs stream), CPU pod + own pods
fv-sparse-pcs-{1,2,3} (e9gsv928on3ii1, bvgbaed4z5u5jk, uhk8bg6ixzh8j7; 3× RTX 5090 $0.99/hr,
runpod/pytorch 2.4.0 image + volume fv env; ALL TERMINATED after ~2.6 h, ≈$8.5 total).
**Status:** DONE.
**SANDBOX** — builds on the sparse23 head set (chat-scoped FV definition); NOT repo standard.

**What (user spec, gated in-chat 2026-08-10):** Hu et al. §3.1 sparse optimization over PC
COEFFICIENTS instead of heads: basis = top-83 uncentered PCs (≥90% pooled variance; 15 PCs
= 80%) of the 20 train tasks' stacked fixed10 sparse23 per-prompt FVs (3400×4096, fp64);
task FV v_t = fixed10 capture mean (user choice — NOT canonical varicl means); inject
v_t(c) = Σ_i c_i·(v_t·u_i)·u_i once at the zero-shot cue token @ block-9 output; loss =
raw −log p(full label) + λ·Σc, c ∈ [0,1]^83; λ ∈ {0.01,0.02,0.05,0.1,0.2,0.5} (extended
grid, user choice) by LOTO-CV over the 20 train tasks, same selection rule as the head run.
Goal: narrow 83 PCs to a manageable sparse set retaining most zero-shot steering performance.

**Files:** NEW `src/sandbox/sparse_head_selection/build_fv_pc_basis.py` (CPU; gates: npc80/90
== 15/83 recomputed, U orthonormal; per-task FV energy retained by subspace ≥ 0.998) →
`artifacts/sandbox/sparse_pc_selection/pc_basis_83.pt`; NEW
`src/sandbox/sparse_head_selection/train_sparse_pcs.py` (reuses train_sparse_heads.py
machinery; PC-contribution tensor C[t,i,:]=(v_t·u_i)u_i; consistency gate c=1 ≡ fp64 subspace
projection, worst rel_err 1.2e-7; fold seeds keyed to λ VALUE so --lambdas shards across pods
reproducibly). Smoke passed on-pod (grad reaches c, loss falls, c∈[0,1]).
**Run:** 3-way λ shard {0.01,0.1}/{0.02,0.2}/{0.05,0.5} via `run_sparse_pcs_shard.sh`
(120 folds, ~2 h wall); reduce (final retrain + baselines + top-k curve) on pod 1.

**FINDINGS (all 120 folds + final retrain):** chosen λ=0.01 (grid monotone decreasing:
mean LOTO acc 0.376/0.348/0.322/0.240/0.138/0.093 for λ=0.01…0.5; λ=0.5 n_active=83 is the
same early-stop-reverts-to-init artifact as the head run's high-λ rows). Final c: **34 PCs
> 0.2, 29 near 1.0** — mostly the top of the variance ordering (PCs 0–22 nearly all kept)
plus a handful of late PCs (28,34,45,53,55,59,65); 49 of 83 PCs driven to ~0.
References on the same 1720 train-task datapoints (pooled): no-interv 0.016, full fixed10
FV 0.413, 83-PC c=1 projection 0.414 (the ≥99.8%-energy subspace ≈ lossless), final sparse
c 0.392 (~95% of ceiling). Top-k-by-c curve: k=12 → 0.249, k=20 → 0.297, **k=30 → 0.388
(~94% of the c=1 ceiling)**, k=50 unweighted → 0.408; so ~30 PCs carry nearly everything,
consistent with the LOTO-honest 0.376. Sharp per-task structure survives: country-capital
0.812, singular-plural 0.65 vs next_capital_letter 0.025 (full FV is equally bad there —
a property of the fixed10 FV at L9, not of the PC truncation). CAVEAT: pooled table is
train-task (final c trained on those tasks); the held-out number is the LOTO mean 0.376.
Outputs: `artifacts/sandbox/sparse_pc_selection/` (pc_basis_83.pt, fold_results/ 120×,
coeffs_final.pt, selection.json, baselines.json, topk_curve.json, metadata.json),
`results/sandbox/sparse_pc_selection/` (summary.md, lambda_cv_summary.csv, loto_per_fold.csv,
summary PNG). Branch `claude-sandbox-sparse-pcs`. **Next:** user to direct (candidates:
eval selected PCs on the 9 held-out test tasks; compare PC subspace vs head subspace;
interpret the late-PC stragglers).

**Top-29 projection eval addendum (2026-08-11, pod fv-pc-proj-eval jxg9vnqvmlc6gq RTX 5090
~20 min TERMINATED):** NEW `eval_pc_projection.py` — the loto_vs_canonical protocol (same
1720 zero-shot datapoints, single cue-token injection @L9, full-label acc) with arm =
sparse23 fixed10-mean FV orthogonally PROJECTED onto span of the 29 c>0.8 PCs (unweighted).
Per-task means: no-interv 0.015 / full FV 0.4135 / **top-29 proj 0.3921 (94.8% of full)** /
83-PC c=1 0.4109. Reproduction gate vs stored baselines: EXACT on all arms/tasks. 15/20
tasks within ±0.01 of full FV (7 bit-identical); the entire deficit is 3 tasks:
english-french 0.62→0.33, english-spanish 0.53→0.40, english-german 0.20→0.16 (+
next_capital_letter 0.025→0.0125) — the translation tasks, exactly the tasks whose
per-prompt FV stacks need the most PCs (part 14c: 60–64 under sparse23). present-past /
capitalize_second_letter / person-sport slightly IMPROVE under projection. Outputs:
`results/sandbox/sparse_pc_selection/top29pc_projection_vs_fullfv.csv`,
`artifacts/sandbox/sparse_pc_selection/top29pc_projection_eval.json`.

**HELD-OUT test-task eval (2026-08-11, pod fv-pc-testeval 8hwlkxs76322ol RTX 5090 ~25 min
TERMINATED):** the train-task table above is IN-DISTRIBUTION (user flagged); the real
generalization test = the 9 test tasks, never in the basis or optimization.
`capture_perprompt_head_activations.py` gained `--extra_tasks` (appended AFTER the 27 so
query-selection RNG of existing captures is untouched) to capture product-company +
country-currency; `eval_pc_projection.py` gained `--task_split_key/--v_means_capture_dir/
--out_tag` (test v_t built from captures with the exact basis recipe, fp64 + checkpoint W_O).
***HEADLINE: the train-fit PC subspace does NOT transfer.*** Test-task means: no-interv
0.004 / full FV 0.453 / top-29 proj 0.203 (44.7% of full) / 83-PC c=1 0.239. Per task:
capitalize 0.91→0.00 (83-PC only 0.05), lowercase_first_letter 0.74→0.16, product-company
0.50→0.05; survivors landmark-country 0.575 (unchanged), capitalize_first_letter 0.98→0.93.
Energy retained by the 83-PC subspace is 61–95% for held-out FVs (vs ≥99.8% train) and the
steering loss FAR exceeds the energy loss (capitalize keeps 80% of norm, loses ~all
steering) — the causal content of unseen tasks' FVs lives largely OUTSIDE the train-task
PC span. The 29 PCs are a faithful compression of the train FVs, not a universal task
subspace. Outputs: `top29pc_projection_vs_fullfv_testtasks.csv`, `..._pertask_testtasks.png`,
`top29pc_projection_eval_testtasks.json`; claude artifact "Sparse PC selection: narrowing
83 task-subspace directions to 29" updated with the held-out section.

---

## 2026-08-10 — Stream cue-attn part 14c: per-task stable rank of stacked per-prompt FVs (both head sets)

**Owner:** Claude Code background session (cue-attn stream), CPU only. **Status:** DONE.
**What (user request):** per task, stack the 170 fixed10 per-prompt FVs (170×4096) and compute
stable rank, for BOTH FV definitions (canonical top-40 + SANDBOX vanilla_sparse_opt23); bar
chart. NEW `src/eval_scripts/plot_perprompt_fv_stablerank.py` →
`results/sandbox/perprompt_fv_norms_vanilla_sparse_opt23/fvstack_stablerank_pertask.{png,csv}`
(sandbox folder since one definition is sandbox). Definitional choice SURFACED not hidden:
both UNCENTERED (repo SR precedent) and MEAN-CENTERED panels reported.

**FINDINGS:**
- UNCENTERED SR ≈ 1.03–1.94 for every task/definition: within a task the per-prompt FV stack
  is essentially rank-1 — one dominant shared direction (the task-mean FV) plus small
  fluctuations. Ordering: retrieval/classification lowest (commonsense_qa, sentiment,
  person-* ~1.03–1.10), transforms/translation highest (~1.4–1.6); under sparse23 the three
  translation tasks jump to 1.87–1.94 (largest relative prompt-to-prompt variation observed).
- MEAN-CENTERED SR (fluctuation dimensionality) ≈ 5–29 of a possible 170: antonym 28.6 /
  synonym 24.5 / translations 21–22 top the top-40 panel; under sparse23 syn/ant/translation
  fluctuation dims roughly HALVE (13–17) while capitalize_first/lowercase_first rise to 21–23.
  Mirrors part 14b: the tasks whose norms (and now variation dims) stand out are largely the
  ones the head set is tuned to.
- Cross-view note: centered SR correlates loosely with the norm ordering per definition —
  high-norm tasks under a head set also vary along more directions under that head set.
- **Added panel 3 (user request): # UNCENTERED PCs for ≥90% variance (PC1 ≈ task mean).**
  Far more discriminative than SR (which is σ₁²-weighted): range **1 → 64**. Retrieval/
  classification tasks need 1–5 PCs (commonsense_qa/sentiment/person-*/ag_news 1–3;
  PC1 alone ≥90% for several); transforms/translations need 30–64 (synonym 58 top40,
  translations 60–64 under sparse23, antonym 46–49). sparse23 collapses several retrieval
  tasks to 1–2 PCs (country-capital 26→2, national_parks 15→1) while inflating translations
  (42–46→60–64). So "one FV per task" is a ~90%-faithful description for retrieval tasks but
  misses a fat 40-60-dim tail for transform/translation tasks.

**Owner:** Claude Code background session (cue-attn stream), CPU only. **Status:** DONE.
**SANDBOX** — head set is the sparse-optimization pick (NOT repo default); results in
`results/sandbox/perprompt_fv_norms_vanilla_sparse_opt23/` (README there).

**What (user request):** repeat part 14's fixed10 norm plots with H = the 23-head
`vanilla_sparse_opt23` set (c>0.8, unweighted). Pure CPU rerun — the part-14 capture stored
all 16 heads/layer. `plot_perprompt_fv_norm_hist.py` gained `--head_label` (titles) + a
head-list uniqueness assert; NEW `plot_fvnorm_headset_comparison.py` (median-vs-median
scatter top40 vs sparse23).

**FINDINGS — the part-14 task ordering is substantially a property of the HEAD SET, not
just the task:**
- Spearman ρ(top40, sparse23 medians) = **0.44** (Pearson 0.50). Between-task variance share
  stays high (79.5% vs 86%), grand mean similar (61.6 vs 62.1 — with 23 vs 40 heads, so NOT
  proportional to head count), and country-capital stays lowest (43.2) / present-past stays
  high (72.5). But the part-14 headline inverts: **synonym (−20.8) and antonym (−16.2)
  plummet** from ranks 1–2 to below mid-pack (56.0/57.8), while **sentiment (+9.0, new max
  75.8), capitalize_last_letter (+7.7), ag_news (+7.5), next_capital_letter (+7.1) rise**;
  letter-surgery tasks as a block move to the top region.
- Reading: the canonical CIE-selected top-40 contains heads that fire especially hard on
  synonym/antonym-style word-pair transforms (they're also the tasks steering/CIE work
  centered on); the sparse-opt set (10/23 overlap, adds L20-L26 late heads) redistributes
  that. Norm ORDER across tasks is head-set-relative; the coarse structure (norms all same
  order of magnitude, task-typed with ~80%+ between-task share, country-capital lowest)
  is robust. Part-14 point 5b (head-selection bias hypothesis) gains support.

## 2026-08-10 — Stream cue-attn part 14: per-prompt FV norms ||v^j_A|| across 27 tasks

**Owner:** Claude Code background session (cue-attn stream), CPU pod + own GPU pod
fv-perprompt-capture u1amcngwvmafq7 (RTX PRO 4500 Blackwell $0.72/hr, runpod/pytorch 2.4.0
image + volume fv env; TERMINATED after ~1.2 h). **Status:** DONE.

**What (user spec):** compute the per-prompt function vector v^j_A = Σ_{h∈H} W_O·h(p^j_A)
(pooled top-40 varicl heads, final cue token) for 27 tasks (20 train + Stream W TEST7) × 170
prompts, histogram the norms pooled and per task. Key question: do tasks differ in FV norm?
User decisions: queries sampled from the FULL dataset (all examples — NOT the tiny valid split,
NO ICL-correctness filter; the canonical varicl sets have only 3–170 prompts/task because
valid ≈ 9% of data + correctness filter — flagged and adjudicated 2026-08-10); TWO variants:
fixed10 (10 shots) and varicl4to10 (uniform 4–10 shots); supplementary norm-vs-shots figure.

**Files:** NEW `src/eval_scripts/capture_perprompt_head_activations.py` (GPU; per-prompt
out_proj-input head activations (170,28,16,256) fp16 per task/variant →
`artifacts/perprompt_head_activations/gptj_27tasks_170prompts/<variant>/<task>.pt`, resumable;
prompt construction reuses build_varicl_prompt_data with query_split=demo_split="all"),
NEW `src/eval_scripts/plot_perprompt_fv_norm_hist.py` (CPU; mmap W_O slices; figures + npz/csv
→ `results/direction3_fv_formation/attention_head_analysis/perprompt_fv_norms/`).

**Gates:** (1) exact-reduction gate on antonym, BOTH variants: replicating the reference's
fp16-batch-sum reduction on the captured tensors matches get_last_token_mean_head_activations
on identical prompts BITWISE (dev 0.00e+00). Debugging lesson: the naive fp64 per-prompt mean
differs from the reference by ~2.3e-4 — NOT noise (forward is bitwise deterministic on this
GPU, ref-vs-ref dev exactly 0) but the reference's fp16 intra-batch sum (varicl_utils.py:136)
rounding at activation scale ~3.6. (2) count gate 170/task/variant (54 files). (3) advisory:
cos(our fixed10 task mean, stored canonical varicl mean) = 0.94–0.98 across tasks (different
prompt population, as expected). Also: get_last_token_mean_head_activations needs the caller's
global torch.set_grad_enabled(False) — without it the gate call OOMs (31 GB) building graphs.

**FINDINGS — tasks differ strongly and cleanly in FV norm:**
- Medians span 49 → 77 (fixed10: country-capital 49.2 lowest; synonym 76.8, antonym 74.0,
  capitalize 73.6, present-past 72.7 highest). Between-task variance = **86%** of total norm
  variance (fixed10; 84% varicl) — within-task IQRs (~2–4) are tiny vs the 28-point range, so
  the per-task histograms barely overlap across the span. Ordering is nearly identical in both
  variants (fixed10 vs varicl medians agree to ~1).
- Rough pattern: knowledge/relational retrieval tasks (country-capital, park-country,
  national_parks, landmark-country, person-*, ag_news) sit low (49–57); word-TRANSFORM tasks
  (synonym/antonym, capitalize, present-past, singular-plural, next/prev_item) sit high (66–77).
- **Shot count barely matters in 4–10:** corr(norm, n_shots) pooled 0.025, within-task mean
  0.078; per-task median-vs-shots lines are flat; fixed10 ≈ varicl throughout. FV norm is a
  task property, not a context-length artifact (at ≥4 shots).

**Next:** nothing pending; possible follow-ups — extend below 4 shots (0–3) where formation
presumably ramps; norm vs direction (does high-norm correlate with steering efficacy).

**Owner:** Claude Code session (tmux window sparse-heads), CPU pod + own GPU pod fv-sparse-heads
mlxcqy1vtfm5yv (RTX PRO 4500 Blackwell $0.72/hr, runpod/pytorch 2.4.0 image + volume fv env;
TERMINATED after ~9.3 h, ~$6.7). **Status:** DONE.

**What (user spec, choices gated 2026-08-06):** SANDBOX trial (NOT repo default) of arXiv
2505.05145 §3.1 sparse-optimization head selection on GPT-J: learn c ∈ [0,1]^448 over all heads,
v_task(c) = Σ_h c_h · (out_proj-projected varicl mean head output), injected ONCE at the cue
token (output of block 9 = 1/3 depth) of ZERO-SHOT "Q: x\nA:" prompts; loss = raw −log p(full
label) (greedy contextualized label tokens, teacher-forced, label positions NOT intervened)
+ λ‖c‖₁; AdamW lr 0.01 batch 128, clamp [0,1] each step. λ ∈ {0.01,0.02,0.05,0.1,0.2} by
LEAVE-ONE-TASK-OUT CV over the 20 train tasks (rule: largest λ within 1pt of best mean LOTO
full-label accuracy); final retrain on all 20 at chosen λ. Queries: valid split cap 100 / min
80 per task, topped up from train split (user-approved rule). NOTE actual split: repo valid
split is ~9% of data (not 30%) → 1720 datapoints, 8 tasks with substantial train top-up
(country-capital 62/80, singular-plural 61/80, prev/next_item 59/80, present-past 53/80,
person-sport 51/80, national_parks 39/80, person-instrument 34/80).

**Files:** NEW `src/sandbox/sparse_head_selection/train_sparse_heads.py` (modes:
check/smoke/cv/reduce/all; resumable per-(λ,fold) artifacts; consistency gate: indicator-c over
canonical top-40 must rebuild stored train_varicl_top40 FVs, else hard stop), NEW
`src/sandbox/sparse_head_selection/run_sparse_heads.sh`. Outputs →
`artifacts/sandbox/sparse_head_selection/`, `results/sandbox/sparse_head_selection/`,
logs → `logs/sandbox_sparse_heads/`.

**Gates passed:** consistency check (indicator-c over canonical top-40 rebuilds every stored
train_varicl_top40 FV; worst rel_err 2.9e-4, cos 1.000000); smoke (grad reaches c, loss falls,
c in [0,1]). HF gradient checkpointing was INCOMPATIBLE with a grad-carrying vector injected
inside a checkpointed block (CheckpointError: saved-tensor count mismatch) → replaced with
micro-batch 32 gradient accumulation inside each batch-128 AdamW step (identical objective;
only blocks >9 store activations since only v requires grad; 31.8/32.6 GB, ~9 h for 101 runs).

**FINDINGS (all 100 folds + final retrain):** chosen λ=0.01 (rule: largest within 1pt of best
mean LOTO acc; grid was monotone decreasing in acc). Final c: 73 heads > 0.2 (21 near 1.0);
overlap with canonical varicl top-40 = 19/40 — (12,10),(15,5),(9,14),(8,1),(8,0),(11,0) etc.
survive, plus many NEW mid/late-layer heads (16,10),(20,1),(25,3),(13,9),(23,7).
HEADLINE (fair, task-held-out LOTO folds @λ=0.01, same datapoints, zero-shot @L9):
mean acc sparse 0.421 vs canonical top-40 unweighted 0.193 vs no-intervention 0.015; sparse
wins 17/20 tasks, big on sentiment 0.59-vs-0.00, person-sport 0.85-vs-0.01, ag_news
0.56-vs-0.08, singular-plural 0.58-vs-0.08; loses only next_item/prev_item (~0.33-vs-0.38,
0.16-vs-0.23) and ties commonsense_qa. Mirrors Hu et al.'s AIE-vs-optimization gap (0.85 vs
0.31 on add-k). CAVEATS in summary.md: canonical is unweighted and injected at L9 (its best zs
layer is ~11); sparse uses learned weights. λ=0.1/0.2 mean n_active inflated (early stop
reverts toward 0.5 init when strong L1 makes es_nll rise from epoch ~0).

**§3.2 stage 1 (2026-08-08, layer-wise mean-ablation, user-gated: 73-head base, UNWEIGHTED
Eq.-2 FVs, uniform grand mean; pod fv-sparse-heads-ablate 1hz2oq75020qz9 ~1.2h TERMINATED):**
NEW `layer_mean_ablation.py`; 328 conditions on the same 1720 datapoints @L9. References:
all-task-specific (73 unweighted) 0.483, all-mean-ablated 0.017 = no-interv 0.016. Best single
L13 0.060; best pair L9+L13 0.112 (runners-up L9+L12 0.110, L12+L13 0.108 — flat mid-layer
spread). UNLIKE the paper (their best pair kept ~98% of full acc), NO pair comes close: task
content is DISTRIBUTED across many layers for our 20 heterogeneous tasks; the pure-formatting
(all-mean-ablated) FV does nothing. Results in layer_ablation_results.json / _best_pair.json.

**§3.2 stage 1b (2026-08-08, k=4 layers; pod fv-sparse-heads-quad k0pfngt2ststb5 ~1.3h
TERMINATED):** NEW `layer_quad_search.py` (greedy from best pair + exhaustive over 12-layer
pool, 532 cached evals). GREEDY AND EXHAUSTIVE AGREE: best quad = L9+L12+L13+L15, acc 0.278
(vs pair 0.112, full-73 0.483) — 17 task-specific heads incl. the canonical trio
(9,14),(12,10),(15,5) at c~1. k-progression 1/2/4/25 layers = 0.060/0.112/0.278/0.483: no
saturation at k=4, task content stays distributed across layers (unlike add-k's 2-layer story).
Runner-up quads all mid-layer variations (9,12,13,+{19,20,16}). layer_quad_best.json.

**Per-prompt v23 build (2026-08-10, CPU):** NEW `build_perprompt_fv_sparse23.py` — per-prompt
FVs under the vanilla_sparse_opt23 definition (23 heads c>0.8; list verified == fv_manifest)
for all 29 tasks x 170 prompts (130 train- + 40 test-split queries, fixed 10-shot), from the
sandbox 448-head capture + out_proj slices mmap-read from the checkpoint. NOTE: cache snapshot
stores fp32 weights; slices fp16-cast to match the capture/repo convention (gate 1 exact after
cast; without it rel 2e-4 = pure fp16 rounding). Gates: slices==stored top-40 slices (10
overlap), rebuilt targets match stored (rel ~2e-4). Advisory cos(mean v23, stored 23-head FV)
0.94-0.99 except capitalize_last_letter 0.904 (same outlier task as the original top-40
capture, 0.858 — 10-shot vs varicl gap, not a build issue). Output
`artifacts/sandbox/sparse_head_selection/perprompt_fv_sparse23/<task>.pt` + build_summary.json.

**Per-prompt read directions (2026-08-10, pod fv-readdirs-cheap mb0s53tnxur843 RTX2000Ada
$0.24/hr ~15min TERMINATED; driver too old for cu130 GPU -> CPU LAPACK):** NEW
`compute_perprompt_read_dirs_sparse23.py` — glossary Eq. 4-5 r^j = M+ v23^j / ||.|| with
M = sum of the 23 heads' OV circuits (fp16-cast weights, fp64 SVD, recon 8e-15). Two
user-gated variants: literal Eq. 5 (machine-eps) and rank90 (90% sigma^2 energy, k=1072).
M numerically FULL RANK (k_literal=4096, cond 3.7e5, sigma 11.55->3.2e-5). ***HEADLINE: the
two variants are ESSENTIALLY ORTHOGONAL — median cos(r_lit, r_r90) 0.001-0.007 on every
task.*** Literal reconstructs v perfectly (cos(Mr,v)=1.0) but its direction is dominated by
near-kernel 1/sigma amplification (the glossary's own caveat, empirically decisive); rank90
reconstructs at cos 0.92-0.96. Output perprompt_read_dirs_sparse23/<task>.pt (both variants,
unit r + preinv_norm + selfcons cosines + task-level r_task) + M_spectrum.npz + summary.

**Files/outputs:** `artifacts/sandbox/sparse_head_selection/` (selection.json, coeffs_final.pt,
metadata.json, baselines.json, fold_results/ 100×), `results/sandbox/sparse_head_selection/`
(summary.md incl. fair LOTO table, lambda_cv_summary.csv, loto_vs_canonical.csv, summary PNG),
branch `claude-sandbox-sparse-heads` (pushed). **Next:** user to direct (candidates: eval the
73-head set on the 9 held-out test tasks; §3.2-style mean-ablation refinement; layer sweep).

---

## 2026-08-05 — Stream cue-attn part 13: cumulative stable rank of d_payload stacks (quick study)

**Owner:** Claude Code session (CPU pod only). **Status:** DONE.

**What (user request):** stack train tasks' 40 unit d_payloads cumulatively in random order,
watch the stable rank (sum s^2/s1^2) grow. NEW `src/eval_scripts/plot_dpayload_stablerank_growth.py`
(CPU; v_proj slices mmap-read straight from the cached pytorch_model.bin — full fp32 model
load OOMs under the shell's 16 GB cgroup cap; HARD GATE: rebuilt synonym d_payloads match the
stored subspace artifact to 1e-16). Headline seed-42 order + min-max envelope over 20 orders.
Output `attention_head_analysis/top40_head_geometry/dpayload_stablerank_growth.{png,csv}`.

**FINDINGS:** single-task SR 9.7-11.8 (mean 10.7; 40 unit rows). Growth saturates immediately:
mean ~15 at m=2, ~19 by m=4-5, then creeps to 21.4 at m=20 (order-independent endpoint) —
adding 19 tasks (760 rows) merely doubles the SR, and the last 10 tasks add ~1. All 20 tasks'
payload geometry together spans only ~2x one task's effective dimensionality: new tasks mostly
REUSE the same per-head payload directions (part 11's head-dominated geometry), consistent
with task identity being a small within-head offset. (SR stays low because the shared
component absorbs energy proportionally: sigma1^2 ~ 4.7% of total at every m.)

---

## 2026-08-07 — Stream cue-attn part 13b: cumulative stable rank of FV head-OUTPUT stacks (quick study)

**Owner:** Claude Code session (CPU pod only). **Status:** DONE.

**What (user request):** part-13 variation with rows = the out_proj-projected task-mean head
activations o = W_O·z_bar (the per-head FV summands, part 3's "output" stack) instead of the
d_payloads. NEW `src/eval_scripts/plot_headoutput_stablerank_growth.py` (same mmap route +
20-order envelope; ADVISORY anchor: present-past single-task SR reproduces part 3's
4.63 raw / 12.16 unit to 0.002). Two panels: UNIT rows + RAW (FV-weighted) rows. Output
`attention_head_analysis/top40_head_geometry/headoutput_stablerank_growth.{png,csv}`.

**FINDINGS:** same hard saturation as the payload stacks. UNIT: single-task 11.4-13.2
(mean 12.3) -> 16.4 at m=2 -> ~19.4 by m=4-5 -> 20.9 at m=20 (1.7x one task). RAW:
single-task 4.6-6.8 (mean 5.6) -> 12.7 at m=20 (2.3x, absolute SR much lower — the strong
heads' norms, L15H5 ~17 etc., dominate sigma1 at every m). So the OUTPUT geometry across all
20 tasks also spans only ~2x one task's effective dimensionality — new tasks reuse the same
per-head output directions; the FV summand geometry is as head-anchored as the payload
(W_V-pullback) geometry, on both sides of the attention head.

---

## 2026-08-04 — Stream cue-attn part 12: d_payload subspace-REPLACEMENT task-switch steering (syn<->ant)

**Owner:** Claude Code session (CPU pod) + own pod fv-payload-switch 2i2v436ssjgq2a
(RTX PRO 4500 Blackwell $0.74/hr, runpod/pytorch 2.4.0 image + volume fv env,
allowedCudaVersions 13.0; TERMINATED after ~20 min, ~$0.25). **Status:** DONE.

**What (user spec, choices gated 2026-08-04):** steer task identity WITHOUT paired prompts by
replacing d_payload-subspace projections at the demo label token of a source-task 1-shot
prompt. Targets = UNPAIRED 10-shot task means (last demo's label token, 130 train-split
capture prompts from gptj_56tasks_170prompts_4tokens) projected into the cached
pooled40heads_k4 bases. Two-step op at a SINGLE edit layer L (capture stack row L+1),
ordering user-specified so the target-task projection wins in any subspace overlap:
step 1 `v += (c(tgt->src) - v@B_src^T)@B_src` (erase source: set source-basis coords to the
TARGET task's mean coords there); step 2 `v += (alpha*c(tgt->tgt) - v@B_tgt^T)@B_tgt`.
Arms: replace_both (steps 1+2) + replace_target_only (step 2 only). Sweeps: layers 0..27
x alpha {0.5,1,2,4}. Eval prompts/readout IDENTICAL to steer_switch_logit.py (same pools,
seed 42, n_test=100; queries have single-token gold under BOTH tasks); headline metric
logit(tgt_gold)-logit(src_gold) at the final position; Δ log p + flip rate also recorded.

**Files:** NEW `src/eval_scripts/capture_payload_switch_means.py` (CPU; means + coords ->
`artifacts/payload_switch_steering/tenshot_lastlabel_means.pt`; own-subspace captures ~28-30%
of the mean's norm at mid layers), NEW `src/eval_scripts/steer_payload_switch_logit.py`
(gates: batched-vs-unbatched, no-op hook == clean, in-hook coord invariant, finite; advisory
old-study baseline check), NEW `src/eval_scripts/plot_payload_switch_logit.py`. Output root
`results/direction3_fv_formation/ablation/attention_head_mechanisms/payload_switch_steering/`
(per-direction npz volume-only per *.npz gitignore; summary.csv + run_config + 2 figures
tracked). Clean baselines reproduce the old paired-Delta study to 0.001 (same prompts:
syn->ant -1.692 vs -1.690, ant->syn -0.029 vs -0.028).

**FINDINGS (peak mean logit-diff over layers; n=100, ci95 ~0.5; alphas extended to 8 on a
second pod fv-payload-switch-a8 0807dhr1tg2194, full --overwrite rerun, TERMINATED ~$0.15):**
- ant->syn replace_both: alpha=1 +0.20 @L8, alpha=2 +0.89 @L8, alpha=4 +1.75 @L7, alpha=8
  +2.29 @L7 (flip 0.81 from clean 0.52; dlogp tgt +1.38, src -0.93). syn->ant replace_both:
  alpha=2 -0.43, alpha=4 +0.51 @L4, alpha=8 +1.04 @L7 (flip 0.64 from clean 0.25; dlogp tgt
  +2.19).
- vs the old paired-Delta method (same prompts/metric; comparison figure part 12b): per
  matched alpha the old full-rank Delta leads by ~one strength doubling, BUT the strength-free
  ceilings favor different methods per direction — syn->ant: new alpha=8 +1.04 BEATS the old
  ceiling (+0.46 @alpha=4, declining to +0.42 @alpha=8); ant->syn: new alpha=8 +2.29 reaches
  ~87% of old alpha=8's +2.63, neither clearly saturated. A 4-dim unpaired replacement matches
  or exceeds a full-rank paired-difference injection at equal-or-double strength.
- Effective window = layers ~3-13, dead by ~15-16 in BOTH methods — the payload-ablation
  early-L transport window.
- replace_both vs target_only: source-erasure adds a modest gain at gentle alpha (ant->syn
  alpha=2 +0.89 vs +0.71; alpha=1 +0.20 vs +0.09) and nothing at alpha>=8 (+2.29 vs +2.24) —
  the target write dominates at high strength.
- Headline: 4-dim unpaired task-mean subspace replacement steers the task switch at the demo
  label token with the same layer window as the paired-difference method and a comparable or
  better ceiling — no paired prompts needed.

**Next:** open — final-cue site variant, digits task pairs, k sweep, cf-task control arm,
alpha>8 saturation check for ant->syn.

---

## 2026-07-30 — Stream cue-attn part 9: PROPAGATED payload-subspace ablation (ciew k4/k8, test7)

**Owner:** Claude Code session (CPU pod) + own pods fv-prop-k4 v5h0yptqd9zx2c /
fv-prop-k8 xislhghsfaaezc (RTX 5090 $0.99/hr each, run in PARALLEL per user "powerful GPU,
fast"; TERMINATED; ~35 min total incl. one crashed launch — smoke with --tasks antonym
alone trips build_cf_map's no-alternative assertion; use >= 2 tasks in multi-task smoke).
**Status:** DONE.

**What (user spec, gated):** new --mode propagated on
ablate_oneshot_payload_subspace_logprob.py: ablate at the anchor site token AND every
later position (incl. tokens not in the plots: newlines, 'Q:', query word) for all blocks
b >= L (mask = pos >= anchor; left-pad safe). CIE-weighted k4+k8 subspaces, ZERO op only
(no matched mean targets for arbitrary positions; user-gated), own + shuffled-cf arms,
7 test tasks. Roots `ablation/attention_head_mechanisms/test7_propagated/ciew_{k4,k8}`.
plot_avg now renders 2x1 for zero-only runs. GATES passed incl. propagated
debug_invariant; HARD sanity: propagated final_cue == anchor-mode final_cue (same
subspace/op), max |dev| 0.0012 — same-protocol reproduction across GPUs.

**FINDINGS (task-avg min-over-L; anchor-mode zero refs in parens):**
- cue1: k4 -3.62 (-0.06), k8 -4.21 (-0.09) — from ~nothing to the LARGEST effect: the
  anchor cue1 token itself is inert, but cue1-propagated covers the demo label + query +
  everything, i.e. removing the payload subspace across the whole suffix is devastating.
- target1: k4 -3.43 (-2.59), k8 -4.03 (-2.76) — +0.8-1.3 nats over anchor mode from the
  positions downstream of the label.
- final_cue: identical to anchor by construction (-0.27/-0.50).
- cf grows with edit size as expected for zero-op (cue1 cf -0.37/-0.57) but specificity
  holds at ~7-10x. cue1 ~ target1 in propagated mode => the suffix damage is dominated by
  label + downstream content, not the cue token itself.
- Layer profile: propagated damage confined to early starts (gone by L ~ 12-16), same
  window as anchor mode — late-layer suffix content in these subspaces is not load-bearing.

---

## 2026-07-31 — Stream cue-attn part 9b: propagated ablation, UNIFORM (unweighted) k4/k8

**Owner:** Claude Code session + own pods fv-prop-uw-k4 fvdckflnrwkv1z / fv-prop-uw-k8
xuyapdrwrhv9l2 (RTX 5090 $0.99/hr each, parallel; TERMINATED, ~25 min). **Status:** DONE.

**What:** same --mode propagated protocol as part 9 but with the UNIFORM pooled40 k4/k8
subspaces (already cached). Roots `test7_propagated/{k4,k8}`. Gates passed; final_cue ==
anchor-mode consistency max |dev| 0.0023/0.0018.

**FINDINGS (task-avg min-over-L, zero op; own | cf):**
- uniform k4: cue1 -5.31 | -1.57; target1 -5.29 | -1.70; final_cue -1.94 | -0.33
- uniform k8: cue1 -5.66 | -1.83; target1 -5.60 | -1.99; final_cue -2.03 | -0.41
- vs ciew propagated (part 9): own damage larger (-5.3 vs -3.6 at k4) but cf damage grows
  ~4x MORE (-1.6/-2.0 vs -0.37/-0.57) => specificity collapses from ~10x (ciew) to ~3x
  (uniform). Suffix-wide removal of the uniform basis is substantially generic damage
  (its tail directions carry population-shared content), while the CIE-weighted basis
  stays a clean task-content object even under whole-suffix ablation. Same early-L window
  (gone by ~14-16). Consistent with part 7d: tail directions = where the less-specific
  content lives.

---

## 2026-07-31 — Stream cue-attn part 10: comparison figure — Stream W pre-image/FV vs uniform payload subspaces

**Owner:** Claude Code session (CPU pod, plotting only). **Status:** DONE.

**What:** NEW `src/eval_scripts/plot_preimage_vs_payload_comparison.py` -> 2x7 grid (own/cf
x {preimage_matched, preimage_icl10, fv, payload k1,k2,k4,k8 zero-op anchor mode}), shared
scale, 7-task means on identical prompts (both studies import Stream W build_prompts).
Output `ablation/attention_head_mechanisms/comparison/heatmap_preimage_vs_payload_ksweep.png`.

**Min-over-L headline (own):** final_cue: FV -7.19 > icl10 preimage -4.00 > payload k8
-2.03 ~ k4 -1.94 ~ matched preimage -1.95. target1: payload k1 -2.12 already beats BOTH
pre-images (matched -1.78, icl10 -1.12); k8 -3.01; FV only -0.40. Specificity at target1:
payload ~13-18x (cf -0.14..-0.17) vs pre-images ~5-8x; FV final_cue 5.9x.

**Propagated variant (--propagated flag; same 3 Stream W columns vs propagated payload
k1..k8) -> `comparison/heatmap_preimage_vs_payload_ksweep_propagated.png`.** Propagated
payload cue1/target1 damage (-4.0..-5.7) approaches/rivals the FV's final-cue -7.19 in raw
magnitude but from a DIFFERENT site+channel and with weaker specificity (~3-4.6x vs
FV 5.9x); Stream W columns unchanged (single-token by construction).
READING: the two decompositions are COMPLEMENTARY channels — pre-image/FV directions own
the FINAL-CUE (readout) site, payload subspaces own the DEMO-LABEL (transport) site with
the cleanest cf controls; layer windows differ accordingly (payload early-L only, FV
final-cue persists to ~L16-20).

---

## 2026-07-31 — Stream cue-attn part 11: 27-task d_payload 2D PCA

**What:** NEW `src/eval_scripts/plot_dpayload_pca_alltasks.py` (CPU) — the 40 unit
d_payload vectors (pooled top-40 heads, cached task-mean activations; exactly the ablation
SVD rows) for all 27 tasks (20 train + 7 Stream W test), stacked 1080 x 4096, centered
2D PCA, task-colored scatter (circles train / triangles test) ->
`attention_head_analysis/top40_head_geometry/pca2d_dpayload_27tasks.{png,npz}`; raw stack
cached (scratchpad dpayload_27tasks_40heads.npy, session-scoped).

**FINDING:** PC1+PC2 carry only ~5% of variance, and the visible clusters are organized by
HEAD, not task — verified: mean pairwise cos WITHIN-HEAD across tasks 0.445 (range
0.16-0.74) vs WITHIN-TASK across heads 0.062. Geometric reason: each head's d_payload is
confined to its own W_V row space (~orthogonal 256-dim slices of 4096-d), so head identity
dominates any global PCA; task signal is a within-head modulation.

**Head-centered variant (user asked why tasks don't cluster):** subtract each head's
across-task mean, renormalize, PCA -> `pca2d_dpayload_27tasks_headcentered.png`. Residual
cos: same-task/diff-head 0.058 vs diff-task/diff-head -0.002 (weak but real cross-head
task signal). The scatter now organizes by TASK FAMILY: knowledge/location tasks
(national_parks, landmark-country, park-country) upper-left; translation tasks
(en-fr/de/es) right; letter-case tasks (capitalize_*/lowercase_*/next_capital) lower-left
— matching the ablation cf finding that related tasks (lowercase/capitalize_first_letter)
share payload content. Task identity in d_payload = a small family-structured offset on
top of large fixed per-head geometry.

---

## 2026-07-31 — Stream cue-attn part 9c: propagated ablation, uniform k1/k2 (completes the k set)

**Owner:** own pods fv-prop-uw-k1 04yc4alwg2mgvb / fv-prop-uw-k2 so332lnnu4cbp2 (RTX 5090,
parallel, TERMINATED ~20 min). Roots `test7_propagated/{k1,k2}`. Gates + final_cue==anchor
consistency passed (max |dev| 0.0016/0.0020).

**Propagated k-progression (task-avg min-over-L, zero op, own | cf | same/cf):**
- k1: cue1 -4.09 | -0.88 | 4.6x;  target1 -4.04 | -0.96;  final_cue -0.88 | -0.14
- k2: cue1 -4.90 | -1.26 | 3.9x;  target1 -4.89 | -1.38;  final_cue -1.65 | -0.30
- k4: cue1 -5.31 | -1.57 | 3.4x;  k8: cue1 -5.66 | -1.83 | 3.1x
READINGS: (1) even ONE direction per task, removed across the whole suffix, costs ~4 nats
(~72% of the k8 effect) — the propagated regime saturates much earlier in k than anchor
mode; (2) specificity is best at k1 (4.6x) and degrades monotonically with k — each added
uniform-SVD direction contributes proportionally more generic content; (3) cue1 ~ target1
at every k (suffix damage dominated by label+downstream, never the cue token itself).

---

## 2026-07-30 — Stream cue-attn part 8: direction3_fv_formation REORGANIZED (57 dirs -> 6 folders)

**Owner:** Claude Code session (CPU pod). **Status:** DONE.

**What (user-specified through 3 question rounds):** flat 57-dir direction3 restructured
into: `ablation/{preimages/{oneshot/{main,numbers,propagated,propagated_numbers},fiveshot},
attention_head_mechanisms/{train_tasks,test7,test7_k_sweep/k{1,2,8,16},
test7_cie_weighted/k{4,8}}}`, `attention_head_analysis/` (4 observational head dirs),
`activation_to_fv_decoding/{fulldim_ridge/{main,qwen3,varicl_top40,+numbers x2,
weight_heatmaps,controls/{shuffled*,rowshuffled*}},pca_ridge/,joint_pca/,ols_layer_sweeps/,
cosine/}`, `activation_geometry/` (6 dirs), `preimage_analysis/` (twoshot_pairdiff +
preimage_steering). Full old->new table in `results/direction3_fv_formation/README.md`.

**Deletions (user-approved exception to never-delete):** the two superseded smoke trial
dirs (joint_pca_..._smoke, pca_abstractive_fv_activation_scatter_smoke, 37 files ~6MB).

**Mechanics:** migration script `logs/direction3_reorg/migrate.sh` (auditable, plain mv —
tracked+untracked mixed; git working tree left uncommitted); file count 2692 -> 2655
(exactly the 37 deleted). Script defaults repointed: bulk rewrite of 55
FV_FORMATION_DIR-lines across 47 eval scripts + manual fixes (Stream W mode_subdir,
plot_payload_k_sweep.root_for_k -> test7_k_sweep/k{k}, pca_cue_token_icl_evolution dynamic
name, merge_fulldim_ridge_results.run_title heuristic -> substring on full path,
plot_decoded_last3_multitask40's stale pre-bucketing hardcode). run_config.json contents
inside moved dirs untouched (provenance). VERIFIED: py_compile clean on all eval_scripts;
plot_payload_k_sweep regenerates IDENTICAL k-trend numbers from new paths; Stream W
plotter regenerates heatmap_all_arms from ablation/preimages/oneshot/main/; no functional
old-path references remain (grep sweep; remaining mentions are docstrings/WORKLOG which
stay per convention).

---

## 2026-07-30 — Stream cue-attn part 7: payload-subspace ablation on the 7 STREAM W TEST tasks

**Owner:** Claude Code session (CPU pod) + own pod `fv-payload-ablation-test7` 5wy0h1qv5lt0wk
(RTX PRO 4500 Blackwell $0.74/hr; TERMINATED, ~50 min incl. one crashed attempt ~$0.65).
**Status:** DONE.

**What (user spec, gated):** generalize part 6 to the 7 Stream W held-out tasks
(landmark-country, word_length, capitalize_first_letter, synonym, lowercase_first_letter,
capitalize, antonym — user chose the 7 over the 9-task varicl hold-out). Subspace per task =
top-4 uncentered-SVD dirs of the 40 unit d_payloads built from the POOLED top-40
train-selected heads (test tasks played no role in head selection) + that task's cached
varicl mean head activations -> `artifacts/payload_subspaces/<task>_pooled40heads_k4.pt`
(x7; ~55-57% stack energy, coverage spread over many heads — no tail-head domination like
the 10-head variants). Ablation protocol unchanged; mean-clamp target = cached 20-TRAIN-task
grand mean (user-gated reuse); cf = seeded random other test task's subspace via imported
Stream W build_cf_map (pool = the 7; map recorded in run_config_multi.json).

**Script changes:** build_payload_subspace.py: --tasks multi-task loop + --head_source
{pertask,pooled40} (+ torch_dtype fix — GPU pod fv env's older transformers rejects
`dtype=`); ablate_oneshot_payload_subspace_logprob.py: --tasks mode (one model load,
build_cf_map, per-task cf, cf_map in run_config); plot wrapper: --avg_tasks task-averaged
2x2. Output root SEPARATE: `results/direction3_fv_formation/payload_subspace_ablation_test7/`
(+ figures/heatmap_payload_arms_test7avg.png). Gates passed for all 7 tasks (+ smoke w/
debug_invariant).

**FINDINGS (7-task-averaged min-over-L delta log p):**
- Same structure as the train-task studies, now on HELD-OUT tasks with a head set that
  never saw them: target1 zero -2.63 / mean -1.54 (early-L window, gone by ~12-14);
  final_cue zero -1.94 / mean -0.50; cue1 ~0.
- Specificity is even stronger than the train-task pair: cf averages -0.16 zero / -0.05
  mean at target1 => mean-op same/cf ~ 29x (final_cue ~ 36x; zero op 17x / 5.8x).
- Effect present in ALL 7 tasks (target1 mean-op -1.04..-2.64). Only notable cf response:
  lowercase_first_letter under capitalize_first_letter's subspace (zero -1.00/-1.49) — the
  two letter-casing tasks plausibly share payload content; every other cf pair ~0.
- Interpretation: the d_payload transport-content mechanism GENERALIZES to unseen tasks —
  per-task content projected into the pooled heads' value channels is causally necessary
  at the demo label (early layers) and query cue, and is task-specific to ~30x under the
  interpretable mean op.

**Part 7b — k sweep (2026-07-30, pod fv-payload-ksweep 1sidnz8vdkheyq ~75 min ~$0.95,
TERMINATED):** k in {1,2,8,16} added around the existing k=4. Builder extended with --ks
(one SVD per task, one artifact per k; 28 new artifacts). Per-k output roots
`payload_subspace_ablation_test7_k{1,2,8,16}/`; NEW `plot_payload_k_sweep.py` ->
`payload_subspace_ablation_test7/figures/ktrend_summary.png` (min-over-L task-mean vs k,
own solid / cf dashed, per site x op).

Min-over-L task-mean (own, mean op): target1 -1.06 (k=1) -> -1.54 (k4) -> -1.80 (k16);
final_cue -0.22 -> -0.50 -> -0.77. zero op: target1 -2.12 -> -2.63 -> -3.07; final_cue
-0.88 -> -1.94 -> -2.08. cf stays flat and tiny at ALL k (mean op <= -0.09 at k=16).
READINGS: (1) the TOP payload direction alone (k=1) already carries ~60-70% of the
demo-label effect — the label-site content is low-dimensional; (2) the final-cue effect
needs k >= 2 (big jump k1->k2) then saturates ~k=8 — cue-site content spreads over a few
directions; (3) specificity does NOT degrade with k (mean-op same/cf ~ 20x even at k=16) —
even a 16-dim task-payload removal barely touches shuffled tasks; (4) diminishing returns
after k ~ 8 everywhere.

**Part 7c — CIE-weighted payload subspaces, k=4 BUILD ONLY (2026-07-30, CPU pod):**
user spec: scale each unit d_payload row to norm 100 x pooled-train-CIE before the
uncentered SVD (global 100x cosmetic; relative weights matter). Builder gains --cie_weight
(suffix `pooled40heads_ciew_k4`; weights + definition stored). Built for the 7 test tasks
-> `artifacts/payload_subspaces/<task>_pooled40heads_ciew_k4.pt`. NO ablation run yet
(explicitly deferred by user).
Characterization vs unweighted k=4: top-4 energy 86% of the weighted stack (vs ~56% unit);
principal-angle cosines ~[0.85-0.91, 0.4-0.6, 0.2-0.45, ~0] => only ~1 strongly shared
direction, mean sq cos 0.26-0.34; the weighted basis CONTAINS the top-4-CIE heads' payload
directions almost exactly (per-head coverage L9H14/L15H5/L8H1/L12H10: 0.28-0.50 unweighted
-> 0.98-1.00 weighted) — with the 30x CIE range, weighted top-4 SVD ~= span of the 4
strongest heads' payloads, discarding the 36-head tail the unweighted basis partly tracked.

**Part 7d — ciew_k4 ABLATION (2026-07-30, pod fv-payload-ciew y9xwj5t6t5474u ~20 min
~$0.30, TERMINATED):** same 4-arm protocol, 7 test tasks, output root
`payload_subspace_ablation_test7_ciew_k4/` (+ figures/heatmap_payload_arms_test7avg.png).
Task-avg min-over-L (ciew vs unweighted k4):
- target1: zero -2.59 vs -2.63, mean -1.46 vs -1.54 — the demo-label effect is FULLY
  RETAINED by the CIE-weighted basis, with an even cleaner cf (mean-op cf -0.026 =>
  same/cf ~ 56x).
- final_cue: zero -0.27 vs -1.94, mean -0.08 vs -0.50 — the query-cue effect essentially
  VANISHES under CIE weighting.
- READING: the top-CIE heads' payload directions (which the ciew basis ~= spans, part 7c)
  carry the demo-LABEL content; the final-cue payload content lives in the directions the
  weighting discarded (tail heads / lower SVD dims) — consistent with the k-sweep
  (final_cue needed k >= 2 and kept growing) and with cue1-vs-final_cue site asymmetries.
  CIE ranks heads by cue-token swap effect, yet what those heads TRANSPORT is label-site
  content — the cue-site residual content is a distributed, small-head phenomenon.

**Part 7e — ciew_k8 ABLATION (2026-07-30, pod fv-payload-ciew-k8 qovfuikji8x8dm ~25 min
~$0.35, TERMINATED):** ciew k=8 subspaces built (92% weighted stack energy) + same 4-arm
protocol -> `payload_subspace_ablation_test7_ciew_k8/`. Task-avg min-over-L
(ciew k8 | ciew k4 | unweighted k8):
- target1: zero -2.76 | -2.59 | -3.01; mean -1.58 | -1.46 | -1.76.
- final_cue: zero -0.50 | -0.27 | -2.03; mean -0.17 | -0.08 | -0.66.
- cf clean everywhere (worst -0.15).
READING: doubling the weighted basis nudges target1 toward the unweighted level (~92% of
it on mean op) and recovers only a sliver of final_cue (-0.50 vs unweighted k8's -2.03
zero) — the cue-site content genuinely lives in the LOW-CIE-weight directions, not just
beyond k=4. Demo-label content = top-CIE heads' payloads; cue content = distributed tail.
(Also fixed a hardcoded '4D' in plot_avg's suptitle; 3 averaged figures regenerated.)

**Next:** direction3 reorg approved-in-principle (plan in session plan file) — awaiting
user go-ahead; then complement test / 9-task variant / promotion remain open.

---

## 2026-07-29 — Stream cue-attn part 6: 1-shot ablation of the 4D payload subspace (present-past)

**Owner:** Claude Code session (CPU pod) + own pod `fv-payload-ablation` svr9bxxxci0ksl
(RTX PRO 4500 Blackwell $0.74/hr, runpod/pytorch 2.4.0 image + volume fv env,
allowedCudaVersions 13.0; TERMINATED after ~25 min, ~$0.35). **Status:** DONE.

**What (user spec, choices gated):** Stream W-protocol causal test of the present-past
attention_head_payload_subspace (k=4). present-past 1-shot prompts (170, imported Stream W
build_prompts/make_chunks), 3 site rows, start layers 0..27, FIXED layer-independent 4D basis
projected out at the site token for all blocks b >= L. Ops: zero + mean-clamp to the
per-(site, edit layer) grand mean over ALL 20 train tasks (equal task weighting; capture
stage). cf arms = english-french's k=4 subspace on the same prompts (user-fixed cf).
Metric delta log p(first answer token). Gates all passed (decoded sites, batched-vs-single,
no-op L=28, debug_invariant smoke, finite deltas).

**Files:** NEW `src/eval_scripts/capture_train_task_site_means.py` (per-task + grand site
means -> `artifacts/payload_subspace_ablation/train_task_site_means.pt`; needs
--batch_size 32 on 32 GB — 170-prompt chunks with output_hidden_states OOM on long-prompt
tasks), NEW `src/eval_scripts/ablate_oneshot_payload_subspace_logprob.py` (4 arms, Stream W
npz schema, resumable), NEW `src/eval_scripts/plot_payload_subspace_ablation.py` (2x2).
Outputs: `results/direction3_fv_formation/payload_subspace_ablation/` (npz + summaries +
`figures/heatmap_payload_arms.png`); logs `logs/payload_subspace_ablation/`.

**FINDINGS (min-over-L task-mean delta log p; sem <= 0.09):**
- target1 (demo label) is the dominant site: zero -3.22 (best L=1), mean-clamp -2.16 (L=1);
  damage requires starting early (gone for L >= ~12-14) — matches where the core payload
  heads read (L8-15) and where d_payload appears at answer tokens (bnd >= ~6).
- final_cue: zero -1.87 / mean -1.36 (best L=11), damage persists for late starts.
- cue1 (demo 'A:'): ~nothing (-0.08).
- Task-specificity is strong, best on the interpretable MEAN op: same/cf = 2.16/0.27 ~ 7.8x
  at target1 and 1.36/0.07 ~ 19x at final_cue (zero op: 3.1x / 8.2x). So the 4D payload
  subspace carries genuinely present-past-specific content at the demo label + query cue.

**Part 6b (2026-07-30, pod fv-payload-ablation-2 brbuegfue2i2h3, ~15 min, TERMINATED):**
(a) symmetric payload run on ENGLISH-FRENCH prompts (cf = present-past subspace) and
(b) NEW fv_zero/fv_cf_zero arms on BOTH tasks — unit canonical train_varicl_top40 FV
projected out at the FINAL CUE token only (k=1 basis through the same hook; cf = other
task's FV). Script extended (ARM_ROW_NAMES, --fv_root, per-task run_config_<task>.json);
figures now task-suffixed + `figures/heatmap_fv_final_cue.png` (both tasks side by side).

Min-over-L mean delta log p (n=170 each):
- english-french payload: target1 zero -3.07 / mean -1.47 (early-L window, same shape as
  pp); final_cue only -0.67/-0.16 (MUCH weaker than pp's -1.87/-1.36 — asymmetry: the
  payload subspace matters at the query cue for present-past but barely for
  english-french); cf (pp subspace) -0.68 zero / -0.16 mean at target1 => mean-op
  specificity ~9x at target1.
- FV-direction ablation at final_cue: same-task ~ -5.8 BOTH tasks (ef -5.86, pp -5.77;
  best L ~ 4-12, damage persists to L~16-20) but cf FV damage is LARGE too (ef prompts w/
  pp FV -3.02, pp prompts w/ ef FV -1.87) => specificity only 1.9x / 3.1x.
- Headline contrast: the FV direction is more devastating but only weakly task-specific
  under projection ablation; the 4D payload subspace (mean op) is smaller in absolute
  damage but 8-19x task-specific. Consistent with the two tasks' FVs sharing a large
  generic component while the payload subspaces separate task content.

**Part 6c (2026-07-30, pod fv-payload-ablation-3 zqjuz8vmdw5la5, ~10 min, TERMINATED):**
fv_zero/fv_cf_zero extended to ALL 3 site rows (ARM_ROW_NAMES now uniform; fv npz
overwritten via --arms fv_zero fv_cf_zero --overwrite, payload npz untouched). New figure
`figures/heatmap_fv_arms.png` (2x2: task-prompt columns x same/cf rows, 3 sites each; the
single-row heatmap_fv_final_cue.png is superseded). Min-over-L (same-task FV / cf FV):
- final_cue: pp -5.77 / -1.87; ef -5.85 / -3.02 (as in 6b).
- target1: pp -0.32 / -0.36 (FV at the demo label does ~NOTHING for present-past — cf
  indistinguishable); ef -1.85 / -0.58.
- cue1: ~0 everywhere (<= -0.11).
- Contrast with the payload subspace at target1 (pp -2.16 mean / -3.22 zero): the FV
  direction and the payload subspace occupy complementary sites — FV = final-cue readout
  direction, payload subspace = demo-label transport content. english-french's FV does
  carry some demo-label-site effect (-1.85, cf 3.2x weaker); present-past's carries none.

**Next:** open — k sweep, core-heads-only subspace variant (part-5 basis caveat), or
FV mean-clamp arm for a like-for-like op comparison.

---

## 2026-07-29 — Stream cue-attn part 5: attention_head_payload_subspace (present-past) + english-french stats

**Owner:** Claude Code session (CPU pod). **Status:** DONE (english-french subspace deferred
— user deciding how to construct it).

**What:** NEW `src/eval_scripts/build_payload_subspace.py` (--task/--n_heads/--k/
--stats_only): computes the top-N per-task-CIE heads' unit d_payload =
unit(W_V^T @ unit(task-mean z_bar)) directly from cached task means + W_V (prompt-free),
prints dimensionality stats (payload + projected-output stable ranks / pairwise cos), and
caches the top-k UNcentered-SVD right-singular-vector basis to
`artifacts/payload_subspaces/<task>_top<N>heads_k<k>.pt` (fp64, orthonormality gated).

**present-past (cached, k=4):** `artifacts/payload_subspaces/present-past_top10heads_k4.pt`
(basis 4x4096 + svals + the 10 d_payloads + heads + definition). Svals [1.56, 1.11, 1.02,
0.98] = 56.7% of stack energy. Stats reproduce part-4 numbers exactly (payload SR 4.105 /
cos 0.116; outputs raw SR 2.811 / unit 4.557). CAVEAT: per-head coverage shows the k=4 basis
is 1 shared core direction + ~3 directions dominated by the idiosyncratic TAIL heads
(coverage: L21H2 0.90, L18H11 0.93 vs core heads 0.37-0.50) — it is NOT 4 dimensions of
"core past-tense payload"; flagged to user before anything is built on it.

**english-french (stats only, top-10 = L15H5 L9H14 L12H10 L8H1 L14H0 L21H2 L8H0 L11H0 L24H6
L20H0):** payload SR 4.621 / pairwise cos mean 0.110 (max 0.345); outputs raw SR 2.698
(norms 1.1-10.3, top: L9H14 10.3, L15H5 10.2, L8H1 9.0), unit SR 5.391 / cos 0.059.
Same shape as present-past: ~2.7-dim raw FV mass, weakly-shared payload core.

**english-french subspace (built later same day, SAME recipe as pp — uncentered SVD k=4;
user said "PCA" but comparability requires the identical construction, flagged):**
`artifacts/payload_subspaces/english-french_top10heads_k4.pt`, svals [1.47, 1.21, 1.00,
0.97] = 55.6% energy; same coverage caveat (tail heads L21H2 0.86 / L11H0 0.78 covered,
core heads 0.33-0.55). **Subspace overlap pp vs ef:** principal-angle cosines
[0.674, 0.535, 0.389, 0.013] (angles 47.7/57.7/67.1/89.3 deg), mean squared cos 0.223 vs
0.001 random-4D baseline; sum cos^2 ~ 0.89 => ~1 effective shared dimension; top
directions align at cos 0.65. So the two tasks' payload subspaces share a sizable common
core (plausibly generic answer/label-content transport, given 8/10 shared heads) but keep
~3 task-distinct dimensions each.

---

## 2026-07-29 — Stream cue-attn part 4: d_payload (value-channel pullback) for present-past top-10

**Owner:** Claude Code session (CPU pod). **Status:** DONE (single prompt/task).

**What (user spec):** for the PRESENT-PAST-specific top-10 CIE heads (L15H5 L12H10 L13H13*
L11H0 L8H1 L9H14 L24H6 L21H2 L20H0 L18H11*; * = not in pooled top-40 — task mean-acts tensor
covers all heads so they're still available), define the value-channel payload direction
d_payload = unit(W_V^T @ unit(z_bar)), z_bar = task-mean head activation (21 prompts).
Exactly position-free (RoPE never touches V); at fixed attention weights, resid content
along d_payload at an attended token moves the head output along its task-mean direction,
scaled by w_t. NEW `src/eval_scripts/plot_dpayload_layer_token_heatmaps.py`
(--task/--query_idx/--n_heads/--plot_head_rank). HARD GATE: recomposed sum_t w_t*v_t must
match the captured out_proj input per head — passed, max rel dev 3.3e-7.

**Outputs:** `results/direction3_fv_formation/dpayload_layer_token/`:
`present_past_q21_dpayload_top10.npz` (10 unit d_payloads + targets + per-head
cos(cue-output, task-mean) = 0.89-0.96 for the 6 core heads, 0.40/0.55 for L24H6/L20H0)
+ layer x token figure for L15H5 (task rank #1).

**Findings (q21 prompt):** L15H5's d_payload lights up the SAME ten past-tense answer-token
columns as L9H14's d_content, but stronger and deeper: cos up to 0.19 / proj 17-21 at the
read boundary (15), stripes persist through L28 (vs d_content peaking at bnd 8 then fading).
<bos> is strongly ANTI-aligned (proj ~ -20 from bnd ~7 on) — the attention-sink token
carries negative payload along the task-mean direction, so L15H5's partial BOS attention
actively dilutes its task output. Payload content appears at answer tokens from bnd ~6.

**Next:** same analysis for english-french; d_payload vs d_content per-head cos.

---

## 2026-07-29 — Stream cue-attn part 3: top-40 head geometry (stable rank, pairwise cos, d_content x40, PCA)

**Owner:** Claude Code session (CPU pod). **Status:** DONE (exploratory, single prompt/task).

**What:** geometry of the 40 canonical varicl heads for present-past. (a) Stacked the 40
out_proj-projected task-mean activations (the FV summands; task mean = 21 ICL-correct prompts,
cue token) into 40x4096; stable rank (fp64 SVD) = **4.63 raw / 12.16 unit-rows**; row norms
0.99-16.9 (top: L15H5 16.9, L12H10 15.7, L8H1 13.9); mean pairwise cos **0.030** (85% of pairs
|cos|<0.1, max 0.37) — near-orthogonal outputs, stable-rank drop mostly from one weak shared
component + norm imbalance. (b) Built unit **d_content = normalize(W_K_pass^T q_pass)** for all
40 heads from the q21 cue-token queries (per-head gate: manual pipeline reproduces each
attention row, max dev 3.9e-7); q_pass energy fraction splits content heads (top-6 CIE all
0.73-0.84) from rotary/positional heads (L14H0 0.38, L10H11 0.21, L14H9 0.22). d_content
geometry: mean pairwise cos **0.085**, stable rank **7.7**, max pair L24H6-L20H0 0.52 — the
heads' queries are more mutually aligned than their outputs. (c) 2D centered PCA of both sets
-> `results/direction3_fv_formation/top40_head_geometry/pca2d_top40_dcontent_vs_outputs.png`
(+ per-panel npz): d_content 2D captures only 12% var but shows layer organization + a
content-head cluster; raw-output PCA is a norm story (L15H5/L12H10/L8H1/L9H14 outliers, rest a
blob). Caches (40x4096 stacks): scratchpad `top40_projected_activations_present_past.npz`,
`dcontent_unit_top40_present_past_q21.npz` (scratchpad is session-scoped — rebuild scripts
recorded there; regenerate via the WORKLOG commands if needed).

**Next:** possible follow-ups — unit-normalized output PCA, prompt-averaged d_content (21
prompts), cross-task comparison.

---

## 2026-07-29 — Stream cue-attn part 2: L9H14 position-free content direction, layer x token maps

**Owner:** Claude Code session (CPU pod, GPT-J fp32 on CPU).
**Status:** DONE.

**What (user spec):** define the position-free content direction of a head as
`d_content = W_K_pass^T @ q_pass` (GPT-J rotary_dim=64 => head dims 0..63 are the RoPE/
positional channel, dims 64..255 pass through untouched; q = the final-cue-token query).
User-adjudicated: exclude the rotary block entirely (it is the "pretend to be closer"
channel); x = all 97 tokens; compare against the RAW residual stream at all 29 layer
boundaries (no LN), layer-9 read row marked. For L9H14 on the same present-past q21 prompt:
two stacked heatmaps (cos to d_content; projection onto unit d_content).

**Files:** NEW `src/eval_scripts/plot_dcontent_layer_token_heatmaps.py` (--layer/--head
configurable). HARD GATE inside: manual ln_1->q/k->rotary pipeline (imports transformers'
apply_rotary_pos_emb) must reproduce the model's own attention row — passed, max dev 1.7e-7.
Output -> `results/direction3_fv_formation/dcontent_layer_token/present_past_q21_L9H14_*`
(PNG + npz with cos/proj grids, d_content, score decomposition). Projection panel color
scale excludes <bos> (norm outlier, proj ~713 at L9 vs <=10 elsewhere; column saturates).

**Findings (this prompt):** (1) L9H14's answer-token preference is almost entirely
position-free: content term 4.3-6.1 vs rotary term -0.03-0.98 across the top-attended
tokens. (2) d_content is task-content selective, not positional: at the read layer (bnd 9)
mean cos = 0.108 at the 10 past-tense answer tokens vs 0.032 elsewhere. (3) The content
builds up over layers 2-8 at answer tokens, peaks at boundary 8 (mean cos 0.119) right
before the head reads at layer 9, and persists (proj peak ~13 around bnd 16) before fading
by the final layers — consistent with earlier attn/MLP layers WRITING the label-content
signal that L9H14 then reads at the cue token.

**Next:** none scheduled; npz allows other heads (--head/--layer) or renderings.

---

## 2026-07-29 — Stream cue-attn: top-10 varicl head attention at the final cue token (one prompt)

**Owner:** Claude Code session (CPU pod, no GPU — GPT-J fp32 forward on CPU).
**Status:** DONE.

**What (user request):** for ONE exact prompt from the canonical varicl head-selection run —
present-past (global task_index 9), query_idx 21, the only 10-shot draw among its 21 CIE
prompts, target ` converted` — plot where each of the top-10 `train_varicl_top40` heads
attends at the final cue token (the `:` of the last `A:`, the position where CIE selection
and FV mean activations were measured). Heatmap x = 97 prompt tokens, y = heads ordered by
descending pooled CIE (labels `L9H14 (CIE 0.0553)` …), color = raw softmax attention weight.

**Files:** NEW `src/eval_scripts/plot_top10_head_attention_cue_token.py` (rebuilds the prompt
via `build_varicl_prompt_data` seeding, loads GPT-J fp32 on CPU, `output_attentions=True`,
asserts rows sum to 1). Output → `results/direction3_fv_formation/top10_head_attention_cue_token/`
(`present_past_q21_cue_attention.png` + regenerable `.npz` with rows/tokens/head list).

**Command:** `python src/eval_scripts/plot_top10_head_attention_cue_token.py` (~3 min on CPU).

**Findings (this one prompt only):** the six highest-CIE heads (L9H14, L15H5, L8H1, L12H10,
L11H0, L8H0) put their mass on the demos' PAST-TENSE ANSWER tokens (' happened' .19 for L9H14;
' restored' .32 for L15H5; ' imagined' .43 for L11H0), i.e. they aggregate label/output tokens
into the cue position. The deeper/tail heads behave differently: L14H0 and L24H6 split between
BOS and the QUERY word ' convert' (.18/.21); L21H2 and L10H0 are mostly BOS sinks (.60/.43).

**Next:** none scheduled — single-prompt qualitative view; npz allows other renderings.

---

## 2026-07-28 — SANDBOX Stream PP-preimage part 3: top-k PCA pre-image SUBSPACE ablation (1-shot)

**Owner:** Claude Code background session (CPU pod) + own pod `fv-pcasub-ablation`
tntocdsxotlhah (RTX PRO 4500 Blackwell $0.74/hr, public runpod/pytorch
2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 image + volume fv env, allowedCudaVersions 13.0).
**Status:** IN PROGRESS (GPU sweep running).

**What (user spec 2026-07-28; leading-direction REDEFINED 2026-07-29, see below):** are the
per-task pre-image SUBSPACES more causally complete than Stream W's single directions? For
k ∈ {2,3,4} (+ a k=0 bridge arm), ablation subspace = QR-orthonormalized
span{**task-mean − grand-mean** pre-image offset, top-k centered PCs} of the task's 170
per-prompt-FV pre-images (part-2 stacks) at each Stream W cell × edit layer (b ↔ capture b+1).
Two ops:
**zero** (remove the subspace component) and **mean** (clamp to the component of the grand mean
over ALL 27 tasks' 4590 pre-images — user-chosen matched population). Arms
`pcasub_{matched|icl10}[_cf]_k{0,2,3,4}_{zero|mean}` (32/task); everything else = Stream W
perlayer protocol exactly (7 test tasks, 170 one-shot prompts seed 42, cue1/target1/final_cue,
L 0..27, Δ log p of first answer token, same cf_map).

**Files (all sandbox):** `src/sandbox/perprompt_fv/fit_preimage_pca_subspace_banks.py`
(banks → `artifacts/sandbox/perprompt_fv_preimages/gptj_train_varicl_top40_pca_banks/`,
42 files, fp64 SVD/QR on CPU; GATES: orthonormality + mean-in-span < 1e-5 everywhere; Gram-vs-SVD
selfcheck; consistency vs part-2 `task_dimensionality/metrics.csv` — **336/336 overlapping
(task,cell,layer) rows EXACT** on n_pca50/rank90, rel ≤ 1e-8 on stable_rank/PR),
`ablate_oneshot_pca_subspace_logprob.py` (imports Stream W's prompt/chunk/cf machinery; Stream W
script untouched; comparability gate: prompt identity EXACT vs stored Stream W npz, clean-log-p
advisory max|Δ|=0.0073 on smoke), `plot_oneshot_pca_subspace_ablation.py`.
Output → `results/sandbox/perprompt_ridge_pilot/oneshot_pca_subspace_ablation/`;
logs `logs/sandbox_perprompt/pca_subspace_ablation/` (markers RUN.done/RUN.failed).

**RUN 1 (2026-07-28, leading direction = RAW task-mean pre-image) — SUPERSEDED & OVERWRITTEN.**
The sweep itself was clean (224/224 npz, gates EXACT, ~2.4 h/$1.8, pod tntocdsxotlhah
terminated), but the user challenged the results (cf nearly as damaging as same-task on the
zero op; late-start L24 > L0 non-monotonicity) and the diagnosis confirmed a DEFINITIONAL flaw:
raw-space pre-image POINTS carry the reconstruction offsets (x̄·std + μ), so the raw task-mean
direction was ~the population-mean activation (cos(m_t, g) = 0.99, cross-task pairwise cos
0.97–0.98 at early layers; ‖m_t‖ ≈ ‖g‖) — its zero-op removes a task-GENERIC component
(matched final_cue −12.2 vs cf −9.7, vs Stream W's 4–9× specificity; Stream W is immune
because it inverts a target DIRECTION, dx = std·dz, where no mean term belongs). The late-L
blow-up (capitalize_first −5.7 @L0 vs −24.8 @L24) came from ripping the full-norm shared
component out right before readout vs 24 layers of self-repair. Mean-op results were already
sensible (they implicitly subtract the offset). **User decision 2026-07-29: leading direction
redefined to the offset-free task-mean − grand-mean (m_t − g = std·P⁺(ȳ_task − ȳ_all27), exact
Stream W analog), AND run-1 outputs OVERWRITTEN in place** (same banks/results/figure paths;
run-1 summaries/figures survive only in git history ≤ 064cf65). Verified new geometry:
cross-task |cos(d_A, d_B)| mean ≈ 0.16 (was 0.97+); ‖d‖/‖g‖ 0.025 (L0 cues)–2.5. Lesson
recorded in DECISIONS: surface point-vs-direction/offset consequences BEFORE compute; the user
gates such definitional choices.

**RUN 2 (2026-07-29, task-offset direction) — DONE** on pod `fv-pcasub-ablation-2`
u28o7c902hyofg (same recipe): smoke (debug_invariant) passed, full 224/224 npz (~2.4 h), all 7
prompt-identity gates EXACT. Results/figures OVERWRITE run 1 in place (user decision).

**Findings (7-task mean Δ log p, min over start layers L; cf = counterfactual-task subspace):**
- **Task specificity is restored** (validating that run 1's cf-effectiveness was the raw-mean
  flaw, not the protocol): zero op matched target1 −1.09 (k0) vs cf −0.10 (~11×);
  icl10 final_cue −4.46 (k0) vs cf −0.47 (~9×) — Stream W-like ratios. L-profiles are sane
  (minima at L0 or L6–8, decaying late; no late-start blow-up).
- **The per-prompt-map task-offset direction at the 10th-example final-cue cell is the
  strongest single direction**: −4.46 @L8, slightly beating Stream W's canonical-map icl10
  preimage (−4.00) at the same site; matched-cell direction −1.09 at target1 vs Stream W's
  −1.78 (weaker; different centering: ours is offset vs ȳ_all27, Stream W inverted raw fv).
- **k now adds a modest, mostly SPECIFIC increment at the final cue**: icl10 final_cue k0
  −4.46 → k2..4 ≈ −5.9 (cf 0.47 → 0.84) — the first 2 PCs add ~1.4 nats then saturate;
  target1 k0 −1.09 → k4 −1.30 (cf flat). At cue1 the k increment is NON-specific (matched
  −0.04→−0.46 but cf −0.02→−0.49): the top PCs carry population-generic variance directions.
- **Mean op (clamp to grand-mean component)**: icl10 final_cue −1.69..−1.72 (cf ≈ +0.1,
  fully specific), sustained to ~L16; matched target1 −0.39..−0.50 (cf −0.08). Larger than
  run 1's raw-mean version — the (k+1)-dim span is now concentrated on task content.
- Ordering at the query cue: FV direction (−7.1, Stream W) > pp-map task-offset subspace k4
  (−5.9) > k0 single direction (−4.5) > canonical-map preimage (−4.0) ≫ matched-cell subspaces.

**SANITY CHECK (user 2026-07-29, DONE): canonical FV inverted via the PER-PROMPT maps.**
`fit_fv_ppinverse_preimage_banks.py`: rank90 TSVD inverse of the RAW (uncentered, per user —
byte-faithful to Stream W's own convention) canonical FV through the per-prompt ridge maps,
dx = std·dz (no μ/x̄); all 168 repro gates vs stored MSEs passed + rank90 EXACT vs
cells_summary.csv. Ablation + figure via the UNTOUCHED Stream W scripts
(`ablate_oneshot_preimage_logprob.py --tsvd_root`, `plot_oneshot_preimage_ablation.py`) →
`results/sandbox/perprompt_ridge_pilot/oneshot_fv_ppinverse_ablation/` (same pod, then
TERMINATED — verified only the 2 long-lived CPU pods remain).
**Findings (min over L, 7-task mean; vs original Stream W values):**
- **Harness validation: the fv/fv_cf arms reproduce the original Stream W summary to 2
  decimals at every row** (fv final_cue −7.19@L8 both; fv_cf −1.22@L6 both) — same prompts,
  same mechanics, cross-GPU.
- **icl10 cells: the per-prompt-map FV inverse ≈ the canonical-map inverse, slightly stronger
  and MORE specific** (final_cue −4.51@L0 vs −4.00@L0; cf −0.34 vs −0.90) — where the
  canonical map is well-conditioned, the two maps' inverses agree causally. Sanity check PASSES.
- **Matched (icl1/icl2) cells COLLAPSE under the per-prompt map** (target1 −0.38 vs Stream W
  −1.78; final_cue −0.32 vs −1.95): the early-cell per-prompt maps (fat spectra, high rank90,
  weak per-prompt signal at 1–2 shots) send the FV to a much less causally loaded direction
  than the canonical rank-≈16 inverse. The task-offset subspace direction from run 2 sits in
  between at target1 (−1.09). I.e. the early-position demo-label effect is a property of the
  CANONICAL map's low-rank inverse, not recovered by the per-prompt map's FV inverse.

**Next:** awaiting user direction. Candidates: (a) per-task mean-op clamp target (task
transfer); (b) k ≫ 4 (part-2 n_pca50 says ~12–26 dims); (c) paired logit-gap regime;
(d) understand the matched-cell divergence (canonical vs per-prompt inverse at icl1/icl2).

---

## 2026-07-27 — Stream R2: effective rank of raw activations by (layer, token position)

**Owner:** Claude Code background session (CPU pod, ~14 min run). **Status:** DONE.

**What (user spec):** stable rank, PCA rank90, and participation ratio of MEAN-CENTERED
activation matrices at 8 positions (icl {1,2,9,10} × {pre_label_token, last_label_token}) ×
layers 6–20 (29-axis indices), pooled over the 29-manifest tasks ([4,930×4096] per cell) and
per task ([170×4096]). NEW `src/eval_scripts/activation_rank_by_position.py` (repo-standard,
not sandbox) → `results/direction3_fv_formation/activation_rank_by_position/` (2 CSVs, full
spectra npz, pooled 1×3 line figure + three 29-panel per-task grids).

**Verification:** independent numpy-SVD recompute of one pooled + one per-task cell matches
CSVs to 1e-6; per-task rank90 ≤ 169 everywhere; pooled n_rows = 4,930 all cells.

**FINDINGS:**
- **Pooled: label tokens are far higher-rank than cue tokens** (rank90 ≈ 650–800 vs 270–680)
  — cross-task label-content diversity dominates. Label rank is ~flat in layer; icl09/10
  labels sit slightly BELOW icl01/02 (mild compression with depth).
- **Pooled cue tokens: rank90 RISES with layer, and late cues (icl09/10) are the LOWEST-rank
  representations in the whole set, with a dip at L10–11 (~270)** — exactly the mid-layer band
  where FV decodability peaks and the PCA task clusters are crispest. Reading: by the late
  cues the model has compressed prompt variability into a low-dim, task-organized structure;
  ambient dimensionality re-expands after ~L12.
- **Per task the ordering INVERTS: cue tokens (rank90 ≈ 80–110) > label tokens** — within a
  task, label-token rank tracks the label vocabulary (person-sport/instrument ≈ 15–45;
  translation/antonym/synonym ≈ 110–125, above their cue lines), while cue tokens carry rich
  prompt-to-prompt (demo-draw) variability regardless of task.
- **icl01/label is consistently the lowest label line per task** — the first label's
  representation is nearly a pure function of the label word; later labels pick up
  context-dependent modulation (rank grows icl01→09/10 for most tasks).
- Stable rank is tiny everywhere (≈5–18 of 4096; PR 17–90): one dominant direction + modest
  bulk; label > cue on both, cue metrics rising with layer.
- NOT a contradiction of the per-prompt map spectra (rank grows with ICL depth): raw rank
  measures total variance spread; the map spectra measure predictable structure. Late cues
  have LESS total spread but MORE organized/predictable content.

**Next:** none pending.

---

## 2026-07-28 — SANDBOX Stream PP-preimage part 2: per-task dimensionality of pre-image matrices

**Owner:** Claude Code session (CPU pod, tmux `fv-preimages`). **Status: DONE.**

**What (user-specified):** for 4 seed-42-drawn tasks (train: commonsense_qa, national_parks;
test: capitalize, capitalize_first_letter), stack each task's 170 pre-images per (layer, token
position) cell into a [170, 4096] matrix and compute: stable rank, rank90 (σ² energy),
participation ratio, n_pca50 — all on the MEAN-CENTERED matrix — plus mean pairwise cosine of
the raw rows (user adjudicated: rows themselves, not pairwise differences; all 14,365 pairs
exact). Per task, one grid PNG with 5 metric heatmaps (positions × layers, repo style).

**Files:** `src/sandbox/perprompt_fv/analyze_preimage_task_dimensionality.py` (Gram-eigh path,
`--selfcheck` vs direct SVD passed on degenerate AND non-degenerate cells — exact match).
Outputs → `results/sandbox/perprompt_ridge_pilot/preimages_truncsvd/task_dimensionality/`
(metrics.csv 3,596 rows = 899 cells × 4 tasks, metrics.npz grids, heatmaps_<task>.png × 4;
cell_rank90 joined per cell as the inversion-rank cap caveat). Run: 899 file reads, 218 s CPU.

**Findings (all sanity relations hold; no NaNs):**
- **Dimensionality is task-dependent and substantial at label tokens:** at the best ridge cell
  (icl10/pre_label L13), n_pca50 = 12 (commonsense_qa), 18 (national_parks), 26 (both
  capitalize tasks); rank90 70–90 — far above rank-1, i.e. per-task pre-image subspaces are
  genuinely multi-dimensional, but well below the cells' inversion caps (441+).
- **Grows with ICL depth:** mid-layer (L8–18) label-position means roughly double from icl1–2
  to icl9–10 (e.g. capitalize pca50 13.3 → 25.4; commonsense_qa 5.1 → 9.9).
- **The two test (letter-case) tasks have the HIGHEST-dimensional, least-coherent pre-image
  sets** (pca50 ~25–26, mean cos 0.58–0.66 at late-ICL mid layers) while train tasks are more
  compact (commonsense_qa pca50 ~10, cos 0.82). Grid max n_pca50 sits at
  icl10/last_prompt_token mid-late layers (18/24/29/29 per task).
- Early layers (L0–3) and icl1 rows are degenerate (metrics → 1, cos → ~1), consistent with
  the inversion caps there.

STILL SANDBOX — do not build on without user promotion.

---

## 2026-07-27 — SANDBOX Stream PP-preimage: truncated-SVD pre-images of per-prompt FVs

**Owner:** Claude Code session (CPU pod, tmux `fv-preimages`) + own pod `fv-perprompt-preimages`
syplcg7pon4jon (RTX PRO 4500 Blackwell $0.74/hr, public runpod/pytorch 2.4.0 image + volume fv
env, allowedCudaVersions 13.0). **Status:** IN PROGRESS.

**What (user-specified, toward per-task subspaces at label tokens):** for every cell of the
sandbox per-prompt ridge (899 = 31 token positions × 29 layers), refit the forward map at the
stored CV-chosen alpha and invert every per-prompt FV back into activation space via a
**truncated-SVD pseudo-inverse**: k = rank90 by σ² energy (identical definition to
`diagnose_weight_spectrum.spectrum_stats`; user chose this over Tikhonov, 2026-07-27), min-norm
solution, output **raw activation space** fp16. Scope: 27 tasks (20 train + 7 test) × 170
prompts × 899 cells. **User approved ~34 GB fp16 storage** (27×170×899×4096×2 B = 33.8 GB).

**Files (all sandbox):** `src/sandbox/perprompt_fv/invert_perprompt_fvs_truncsvd.py`
(reuses the regression's loaders; per-cell REPRO GATE vs stored test_mse_fv/test_mse_pp rel
≤1e-4, SPECTRUM GATE on icl10/pre_label L13 rank90==441 vs stored spectra.npz, self-consistency
forward(pre-image)==rank-k projection rel ≤1e-3; resumable, skips existing cell files).
Outputs → `artifacts/sandbox/perprompt_fv_preimages/gptj_train_varicl_top40/icl{n}/{role}/L{layer}.pt`
(fp16 [4590,4096] + metadata + sv/standardizer/diagnostics); summaries →
`results/sandbox/perprompt_ridge_pilot/preimages_truncsvd/cells_icl{n}.csv`;
logs `logs/sandbox_perprompt/preimages/`.

**CPU smoke (gate cell icl10/pre_label_token L13): ALL GATES PASSED** — repro rel_fv=7.7e-6 /
rel_pp=1.8e-6, spectrum rank90=441 exact max|Δsv|/σ1=5.2e-5, selfcons 1.6e-6; trunc residual
mean 0.273, file 37.8 MB as budgeted. Resume/skip path verified.

**Full run:** driver `logs/sandbox_perprompt/preimages/run_all_shards.sh` (shards icl10 then
1–9, sequential) launched on the pod. Markers RUN.done/RUN.failed on the volume.

**Self-consistency gate TRIPPED on GPU (icl10/pre_label L2, 1.16e-3 > 1e-3) — root-caused, not
a data issue:** CUDA's default `torch.linalg.svd` driver (gesvdj) has fp32 backend error
‖W−USVʰ‖/σ₁ ≈ 1e-3 on these matrices; `driver="gesvd"` gives 8.7e-6 (= CPU LAPACK level) and is
faster (1.2s vs 3.9s; diagnostic `logs/sandbox_perprompt/preimages/diag_svd_driver.py`, TF32
confirmed off). Fix: pin `driver="gesvd"` on CUDA in the inversion script — **no tolerance or
definition changed**; repro gates were exact throughout. Deleted the 2 gesvdj-computed cells
(icl10/pre_label L00–L01) and relaunched; L13 smoke cell was CPU-computed (accurate), kept.

**RUN DONE (all 10 shards, ~1.7 h GPU):** 899/899 cell files, 32 GB (within approved 34 GB),
merged `cells_summary.csv`, no NaNs. Worst gates across all 899 cells: repro rel_fv 7.7e-6 /
rel_pp 1.8e-6, selfcons 1.5e-5. rank90 spans 1 (degenerate near-zero-signal early-layer cells,
trunc_resid→1.0) to 868; median 258 (p25 150, p75 281); best-cell (icl10/pre_label L13)
trunc_resid mean 0.226–0.273 range across top cells. Spot check at the gate cell: cos(pre-image,
true activation) mean 0.834 (train tasks 0.871, test tasks 0.727), norms 88 vs 81 —
informational, no threshold. Pod terminated (verified gone).

**Status: DONE.** Committed to main (sandbox family lives on main since 57b1e53). STILL SANDBOX —
do not build on without user promotion. Ready for the subspace analyses on top of the pre-images.

---

## 2026-07-26 — SANDBOX Stream PP-steer: steering effectiveness of matched per-prompt FVs

**Owner:** Claude Code background session (CPU pod) + own pod `fv-perprompt-steering`
q7fjo9pg0yc5hj (RTX PRO 4500 Blackwell $0.74/hr, public runpod/pytorch 2.4.0 image +
volume fv env, allowedCudaVersions 13.0). **Status:** IN PROGRESS.

**What (user-specified arm):** benchmark per-prompt FVs in the held-out steering family —
each eval query steered by the FV from a 10-shot CLEAN-ICL prompt with that example as final
query (both conditions: zero-shot+FV, 10-shot-shuffled+FV). Protocol otherwise identical to
`evaluate_heldout_multitask_head_fvs.py` (seed 42, filter=clean-ICL-correct, +FV at last token
of each edit layer 0..27, intervention top-1, 9 test tasks, Σ=1,222 filtered queries).
Existing sandbox capture covers only 40 hash-sampled test queries/task (not ⊇ filter sets) →
new capture stage.

**Files (all sandbox):** `src/sandbox/perprompt_fv/capture_eval_query_fvs.py` (per-query FV
banks for all filter-set queries; stable_rng demo convention; GATE: overlap queries must
reproduce original capture targets), `evaluate_heldout_perprompt_fv.py` (n_shot_eval copy with
per-query fv_bank[j], identical RNG consumption; GATE mode `--gate_constant_varicl`: constant
bank must reproduce cached varicl_top40 per-layer curves exactly on word_length),
`plot_perprompt_steering_comparison.py` (grid + aggregate PNGs + best-layer CSV; baselines
per user: varicl_top40 + task-specific from cached results). Output →
`results/sandbox/perprompt_ridge_pilot/steering_eval/`;
banks → `artifacts/sandbox/perprompt_head_acts/gptj_train_varicl_top40_evalqueries/`;
logs `logs/sandbox_perprompt/steering_eval/`.

**CPU smokes passed:** imports OK; filter sets load (antonym 319 … word_length 22, Σ1,222);
overlap counts vs old capture confirmed; demo RNG deterministic.

**Stage 1 DONE + GATE PASSED:** fv banks for all 9 tasks (1,222 FVs) →
`..._evalqueries/<task>/fv_bank.pt`; all 254 overlap queries reproduce the original capture
targets bit-identically (rel_l2 = 0.0). GPT-J now cached on the volume
(`model_cache/huggingface`, export HF_HOME before runs on fresh pods).

**Stage-2 LOOP GATE: HARD STOP (user adjudicating).** Constant-varicl_top40-bank run on
word_length vs cached `heldout_varicl_nheads_sweep` curves: all 28 zero-shot cells match
EXACTLY; fs-shuffled L0/L1 match EXACTLY (RNG/prompts in sync); fs-shuffled L2 differs by
exactly ONE query (8/22 vs cached 7/22). Rerun reproduces 8/22 bit-for-bit → this stack is
deterministic; discrepancy is cross-stack fp drift vs the mid-June cached run (different
torch/GPU). Logs `gate.log`/`gate_rerun.log`. Pod terminated while awaiting adjudication
(re-provision ~3 min; everything restartable from volume).

**Adjudication (user, 2026-07-26): option 1 — relax gate to isolated single-query flips.**
Gate reworked (≤1 flipped query per cell, ≤6 differing cells of 56); PASSED on word_length:
53/56 exact, 3 single-query flips, all fs-shuffled (zs fully exact). Full sweep run on second
pod `fv-perprompt-steering-2` ixrfu8jnx7i9kh (same recipe; TERMINATED after run), 2 shards,
both exit 0. Plots + best_layer_summary.csv in the steering_eval root (grid-only PNG policy).

**FINDINGS (best-layer intervention top-1, mean over 9 tasks):**
- **Per-prompt FVs steer WORSE than the task-level FVs on aggregate:** zs 0.358 vs varicl_top40
  0.400 (task-specific 0.483); fs-shuffled 0.691 vs 0.784 (0.812). Full-curve aggregate shows
  per-prompt below both baselines at nearly every layer in the fs condition.
- **Heterogeneous per task.** Per-prompt BEATS varicl_top40 zero-shot on antonym (0.668 vs
  0.583), synonym (0.297 vs 0.189), product-company (0.407 vs 0.148); ~ties landmark-country,
  word_length, country-currency; **collapses on the letter-case tasks** (capitalize_first zs
  0.331 vs 0.853; lowercase_first fs 0.451 vs 0.927; capitalize zs 0.692 vs 0.953).
- **Not a scale artifact:** per-prompt norms are LARGER than the task FV (+15–35%), mean
  cos(per-prompt, task FV) 0.77–0.90 (norm/cos control table in this entry's log).
- Interpretation (speculative, flagged as such): the prompt-idiosyncratic component that
  improved ridge decodability does NOT help causally as steering content — averaging into the
  task FV removes it and steers better; per-prompt extra content actively hurts on format-like
  (letter-case) tasks while sometimes helping zero-shot on semantic tasks.

**Status: DONE.** Both pods terminated (verified). STILL SANDBOX.

**Update 2026-07-27 — per-prompt map spectra across token positions (user request):** NEW
`src/sandbox/perprompt_fv/diagnose_weight_spectrum_positions.py` → `.../weight_spectrum_positions/`
(grid PNG + spectra.npz + summary.json). 9 cells (cue + target tokens at icl 1/2/9/10 + final
cue token), each at its best layer by stored test_r2_fv, own CV α; all repro gates passed
(worst rel 3.5e-5, known cross-arch fp tail). FINDINGS:
- **Map dimensionality grows monotonically with ICL depth.** Cue tokens: energy-90 rank
  8 → 74 → 432 → 441 (icl01→02→09→10), PR 7→16→166→178, σ21/σ1 0.04→0.64. Targets:
  81 → 266 → 273 → 423, PR 17→43→59→143.
- **icl01/pre is nearly low-rank** (rank90=8) with a visible knee ~10–20 — the cue token at
  one shot supports only a thin map (matches the PCA blob + R²=0.07 + CV picking α=3.16e5).
- **Targets start structured** (icl01/last rank90=81 vs cue 8) — mirrors the PCA finding that
  label tokens are task-typed from the first example while cue tokens accumulate context.
- **The final cue token has the fattest spectrum of all** (rank90=654, PR=315, α=1e3) and the
  highest per-prompt-target R² (0.632) — expected, since the per-prompt targets are constructed
  at that very position.
- Caveat: per-cell CV α differs (3.16e5 → 1e3) and shrinks tails; endogenous (CV picks big α
  where per-prompt signal is absent), so the trend reflects real signal availability, but the
  spectra are not α-matched across panels.

---

## 2026-07-23 — Stream P2: PCA of L11 target-token (last_label) representations across ICL positions

**Owner:** Claude Code background session (CPU pod). **Status:** DONE (~4 min run + grid replot).

**What:** target-token mirror of Stream P (below): same 20+7 tasks, 170 prompts, two train-fit
PCA bases — but `token_role=last_label_token` at **L11** (user choice; ridge heatmaps: L11 best
for first_label everywhere and tied-best with L13 for last_label, L13 only winning at late
positions). `pca_cue_token_icl_evolution.py` generalized with `--token_role` (+ default output
dir per role); per user decision figures are now GRID PNGs ONLY (per-position figures + PDFs
removed from the script and deleted from the Stream P cue outputs; data npz/csv retained for
fast ad-hoc replots). `plot_pca_cue_pc1pc2_grid.py` made role-aware.

**Outputs:** `results/direction3_fv_formation/pca_lastlabel_token_icl_evolution/` (same layout
as the cue study). Verification: fit shapes exact (34,000/3,400 × 4096); hand-recomputed
projection from pca_model.pt + fresh activation matches npz with diff 0.0 (both variants).

**FINDINGS (vs cue tokens):**
- **Target tokens are already task-organized at icl01** — semantic-family clusters (person-*,
  translation, letter-ops, classification) are visible from the very first label, where the cue
  tokens started as one undifferentiated blob. Makes sense: the label token's literal content is
  task-typed, so no context accumulation is needed.
- Evolution with ICL depth is correspondingly mild: geometry is essentially fixed by icl02–03,
  later shots only tighten clusters slightly. The dramatic blob→clusters transition is specific
  to cue tokens.
- Per-task clouds are visibly broader than at cue tokens (label content varies per prompt), and
  same-family tasks sit closer/overlap more (english-french/spanish; person-instrument/sport;
  park-country/person-* region).
- Explained variance is flatter: pooled top-3 = 7.1/7.1/4.1% (cue: 12.2/9.5/8.2), final-position
  8.7/7.7/4.5% (cue: 15.4/11.7/7.1) — target-token variance is spread over more directions.
- Test tasks again embed inside the train geometry near related train tasks.

**Next:** none pending.

---

> **Results layout (current):** intermediates live in git-ignored `artifacts/`; study deliverables in
> tracked `results/` bucketed by direction (`direction1_ambiguous`, `direction2_label_geometry`,
> `direction3_fv_formation`, `steering_vector_comparison`, `general`); run logs in git-ignored `logs/`.
> Paths come from `src/utils/paths.py` — see README "Repository layout". **Entries below dated before
> 2026-06-19 cite the paths that were current when written.**

## 2026-07-23 — Stream P: PCA of L13 cue-token representations across ICL positions

**Owner:** Claude Code background session (CPU pod). **Status:** DONE (full run ~5 min, CPU-only).

**What:** user-requested visualization of how task representations at layer index 13 (ridge-study
axis: 0 = embeddings → L13 = block-12 output) evolve across ICL positions at cue tokens
(= pre-label tokens ONLY, icl01/pre…icl10/pre; user-confirmed, final prompt token excluded).
Two PCA bases, each fit on the 20 train tasks only, projecting 20 train + 7 test tasks
(ridge task set), 170 prompts/task (train+test splits pooled): (1) `pca_all_positions` — fit on
all 10 positions pooled (34,000×4096); (2) `pca_final_cue` — fit on icl10/pre only (3,400×4096).
Per variant, per position: PC1–PC2/PC1–PC3/PC2–PC3 scatters (per-prompt points + task-mean
markers, colored by task, test = triangles) + a 10×3 positions-by-pairs grid; full top-10-PC
coordinates saved (projections.npz + CSVs) so 3D views can be replotted without reloading.

**Files:** NEW `src/eval_scripts/pca_cue_token_icl_evolution.py` (reuses the fulldim-ridge
loaders). Output → `results/direction3_fv_formation/pca_cue_token_icl_evolution/`;
log `logs/pca_cue_token_icl_evolution.log`.

**Verification:** smoke (2 train + 1 test task, icl 1+10 → `artifacts/sandbox/pca_cue_smoke/`):
fit-row counts exact (680/340), hand-recomputed projection matches npz to 7e-6 (fp32), split
labels correct, figures render correctly.

**Full-run verification:** fit shapes exact (34,000×4096 pooled / 3,400×4096 final-cue);
projections.npz spot-check: hand-recomputed `(x−mean)@Wᵀ` from `pca_model.pt` + a freshly loaded
antonym icl05 activation matches stored coords with diff 0.0 (both variants); 45,900 points
(27 tasks × 10 positions × 170) per variant.

**FINDINGS:**
- Top-3 explained variance: pooled basis 12.2/9.5/8.2% (~30%); final-cue basis 15.4/11.7/7.1%
  (~34%) — task structure at L13 cue tokens is genuinely >3-dimensional, consistent with the
  k≈16 optimum from the k-sweeps.
- **icl01 is one undifferentiated blob** (all 27 task means on top of each other; only
  commonsense_qa, and to a lesser extent sentiment/ag_news — the non-word-level tasks — sit
  apart). In the final-cue basis icl01 also has much smaller spread: early-position reps barely
  project into the final-cue PC subspace.
- **Separation happens fast: by icl03 most per-task clusters are already distinct**; icl05→10 is
  refinement (clusters tighten and drift outward/apart, means stabilize). Matches the ridge-R²
  picture where later ICL positions decode better, and the varicl saturation by mid-shots.
- **Test tasks (triangles) embed inside the train-task geometry, near semantically related train
  tasks**: antonym/synonym/capitalize sit in the extractive/word-transform region;
  lowercase_first_letter/capitalize_first_letter near the other letter-case tasks;
  landmark-country near the person-*/park-country knowledge cluster. Nothing lands outside the
  train hull — consistent with train-fit ridge maps generalizing to test tasks.
- **Clusters organize by task family**: translation (english-french/spanish/german) nearly
  overlapping; letter-case ops together; person-* factual lookups together; classification-ish
  tasks (sentiment, ag_news) and commonsense_qa far out on their own.
- Both PCA variants tell the same story; final-cue axes give slightly crisper icl10 clusters,
  pooled axes make the icl01→10 outward expansion more visible.

**Follow-up (same day, user request):** NEW `src/eval_scripts/plot_pca_cue_pc1pc2_grid.py` —
PC1-vs-PC2-only 2×5 grid (one panel per ICL position) per variant, built purely from
`projections.npz` (no activation reload), same task colors → `figures/grid_pc1_pc2.{png,pdf}`
in both variant dirs. The one-blob→clusters evolution reads cleanly left-to-right; in the
pooled basis the whole cloud also translates (icl01 sits at high PC2 and drifts down/right as
positions advance), i.e. PC2 partly encodes ICL depth; in the final-cue basis clusters expand
radially from the icl01 collapse point with stable geometry from ~icl05.

**Next:** none pending. 3D views can be plotted from `projections.npz` without reloading
(top-10 PCs per point saved). Layer/roles are CLI args for cheap follow-ups.

---

## 2026-07-20 — Stream S: INPUT-matched two-shot token-pair steering (perpair + cumclamp, dircos)

**Owner:** Stream S (CPU pod; 1× RTX PRO 4500 Blackwell pod `fv-inputmatch-tokenpair`, $0.74/hr,
terminated after run). **Status:** DONE. Compute 04:49–06:20 UTC (~91 min, BS=64).

**What:** user-requested mirror of the two-shot token-pair perpair/cumclamp dircos studies:
`--pair_mode input` in `steer_twoshot_tokenpair_cos_heatmap.py` — pairs share the demo INPUT words
(I1, I2) and the query; each function supplies its own labels, so label1/label2 are the differing
(counterfactual-carrying) tokens and input2 is clean (CLEAN flips accordingly). Anchor = every
shared input word whose outputs differ across functions and are single-token under both (all four
demo labels single-token; both choices confirmed by user 2026-07-20). Same seed-42 stable_rng
keying, α∈{0.5,1,2} perpair + cumclamp, both directions, both task pairs. Pairs: ant/syn 987,
digits 200. Outputs → NEW roots
`results/direction2_label_geometry/twoshot_tokenpair_perpair_inputmatch_cos_heatmap/` and
`.../twoshot_tokenpair_perpair_inputmatch_cumclamp_cos_heatmap/` (same file layout; summaries carry
`pair_mode`).

**Files:** `src/eval_scripts/steer_twoshot_tokenpair_cos_heatmap.py` (--pair_mode, CLEAN_BY_PAIR_MODE,
input-mode pair build + label-differ asserts, mode-dependent default roots/summary note),
`src/eval_scripts/plot_twoshot_tokenpair_heatmap_grid.py` (title tag "INPUT-matched pairs (labels
differ)" when root contains `inputmatch`), launcher `logs/run_twoshot_tokenpair_inputmatch.sh`
(gitignored, like its siblings); log `logs/twoshot_tokenpair_inputmatch_full.log`.

**Verification:** label-mode regression smoke (16 pairs, layers 0/6/11): edited script
bit-identical to `git show HEAD` original on all 90 grids. Input-mode perpair + cumclamp smokes:
all structural asserts pass (α=1 patch identity, clamp identity at ℓ=i and ℓ=28, lower-tri≡0,
finiteness); decoded sample pair shows differing label sites (' inability'/' improbable' vs
' capability'/' likely') and identical input2/qinput. Full run: all asserts pass, 90+90 perpair /
30+30 cumclamp grids per root.

**FINDINGS (dircos peaks at α=1; perpair/cumclamp):**
- **Built-in sanity check comes out exact:** label1→input2 and label1→prelabel2 ≡ +1.000 in every
  direction and task pair — for reads BEFORE label2, patching label1 patches the ONLY upstream
  differing token, i.e. a full causal patch, so the displacement equals the counterfactual diff.
- **Label-site patching aligns better than in the label-matched study at the query sites:**
  ant→syn label2→qfinal +0.888/+0.902 (label-matched +0.770/+0.846), syn→ant +0.897/+0.922,
  digits next→prev +0.934/+0.935; label2→qinput ant→syn +0.807/+0.830 (label-matched +0.430/+0.500).
- **The input2 row collapses once it is clean:** peaks ≤ +0.34–0.49 (it was the strongest row when
  it carried the counterfactual: input2→prelabel2 +0.85 label-matched ant→syn).
- label1→qfinal (+0.63 ant/syn, +0.84 digits) < label2→qfinal — the later demo's label transports
  more of the function flip to the query, consistent with the label-matched study's ordering.
- Cumclamp ≥ perpair modestly everywhere, as in the label-matched study.

**Next:** none pending — figures in both new roots' `figures/` (matrix/combined/scalar_overview,
titles tagged INPUT-matched).

**Blockers:** none. Pod terminated and verified gone. NOTE: WORKLOG/DECISIONS left uncommitted
(another active stream has in-flight edits in the shared tree); commit covers code + results only.

---

## 2026-07-20 — Stream Y2: DIGITS number tasks added to FV set + ridge + cosine studies

**Owner:** Claude Code CPU-pod session (tmux `fv-cosine-pertask`) + own RunPod pod
`fv-digits-decodability-2` fnfvfbo0prqov2 (RTX PRO 4500 Blackwell $0.74/hr, ~2.5h, TERMINATED).
First pod attempt died on the known GHCR IMAGE_AUTH_ERROR → public image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1` + the volume's fv env (torch 2.12+cu130 needs a
CUDA-13 driver → `allowedCudaVersions:["13.0"]`). **Status:** DONE.

**Why:** user directive 2026-07-20 — number tasks default to the DIGIT variants from now on
(see DECISIONS same date). Digits had no captures/FVs/ridge coverage, so this mirrors exactly
how the word numbers were added (Stream X capture+ridge, Stream Q FV build), driver:
`logs/pertask_r2_digits/gpu_driver.sh` (restartable, all 4 stages).

**New data:** residual captures for prev/next_number_digits in all 10 gptj_56tasks_170prompts
roots (170×29×4096 fp16 each, loader-verified); max-10 varicl mean head acts
(`artifacts/multitask_aie_heads_varicl_paired_digits_max10/`, copied into the main varicl
root); train_varicl_top40 FVs for both digits tasks (canonical top-40 head set,
`fv_manifest_digits.json`; cos(digits FV, word FV) ≈ 0.75 both pairs).

**New studies:**
- `results/.../fulldim_ridge_activation_to_fv_varicl_top40_plus_number_digits/` — the 10 ridge
  shards re-evaluated with 11 test tasks (7 defaults + word numbers + digits). Fits verified
  unchanged: all 899 alpha choices identical to plus_numbers; per-task MSEs reproduce at
  median 2e-7 rel, p99 1.7e-6. NOTE a small cross-arch (Blackwell-vs-L4) fp tail concentrated
  at layers 2–4 (all >1e-5 cells are L3, max 8.4e-5 rel = 3.7e-5 abs on MSE 0.45) — early-layer
  Gram ill-conditioning; user reviewed and approved proceeding. Merged combined_* + per_task_r2
  outputs (heatmaps, panel, 3 best-over-layers line plots).
- `results/.../cosine_activation_to_fv_varicl_top40_pertask_digits/` — Stream Y cosine metric
  (raw per-prompt cosine vs varicl_top40 FV, mean over 170 prompts) for antonym/synonym/digits.
  CSV 4×899, spot-check exact.

**Findings (digits vs word numbers):**
- Cosine: digits REVERSE the Stream Y anomaly — they saturate at ≈0.50 (above antonym/synonym
  ≈0.47; word numbers sat BELOW at 0.40–0.43), crossing the word tasks by icl3. Best cells
  ≈0.50 at L13 (icl10/finaltok, icl08/pre).
- Ridge R²: digits are LOWER than word numbers (next 0.312 vs 0.375; prev 0.245 vs 0.337),
  consistent with the word numbers' R² being inflated by prev_item/next_item train adjacency
  (digits have no such leakage). Ordering: next_number_digits > antonym ≈ synonym >
  prev_number_digits; saturation ~icl05–07 as before.
- Defaults flipped: `cosine_activation_to_fv_varicl_pertask.py` and
  `plot_pertask_r2_best_lines.py` (--tasks → digits; the latter's --input_csv → the
  _plus_number_digits study). Default invocations now regenerate the digits plots.

**Next:** nothing — awaiting user direction.

---

## 2026-07-20 — Stream Y: per-task cosine(activation, varicl_top40 FV) heatmaps + lines

**Owner:** Claude Code CPU-pod session (tmux `fv-cosine-pertask`); no GPU — pure tensor math over
existing captures. **Status:** DONE.

**Goal:** direct "how visibly does the FV appear in the residual stream as ICL examples accrue"
companion to the per-task ridge R² study (Stream X): per cell (task, icl_example_index,
token_role, layer) the RAW per-prompt cosine(activation, train_varicl_top40 FV), averaged over
ALL 170 prompts (train+test pooled; both choices confirmed by user 2026-07-20, along with raw
= no centering). Tasks: antonym, synonym, prev_number, next_number.

**New:** `src/eval_scripts/cosine_activation_to_fv_varicl_pertask.py` (reuses the pooled cosine
study's shard loader + merge helpers). **Changed:** `cosine_activation_to_task_fv.py` loader got
the same stale-absolute-path fallback the ridge loader already had (index.json paths predating
the results→artifacts reorg); `plot_pertask_r2_best_lines.py` gained backward-compatible
`--value_column/--ylabel/--title_metric/--suptitle` (defaults regenerate the R² plot
pixel-identically — verified via cmp).

**Outputs** (`results/direction3_fv_formation/cosine_activation_to_fv_varicl_top40_pertask/`):
`per_task_cosine.csv` (3596 rows = 4×899; n_prompts=170 everywhere, no NaNs),
`cosine_heatmap_{task}.png` ×4 + `cosine_heatmap_panel.png` (diverging, shared scale),
`best_cosine_by_position_lines{,_prelabel,_label}.png` (label = mean over first/last label roles
per example, matching the R² study's role-filtered variants), `summary.json`.

**Findings:**
- All 3596 cells positive; mid-layer band strongest (≈L10–16), same within-example sawtooth as
  the R² study (pre > first/last label roles at every depth).
- Best cells: antonym 0.479 icl10/finaltok L13; synonym 0.472 icl10/finaltok L16; next_number
  0.433 icl09/pre L13; prev_number 0.406 icl09/pre L13.
- Pre-label lines: steep icl1→3 rise then saturation ~icl05 (antonym slowest, matching its R²
  behavior). Ordering FLIPS vs the ridge study: word tasks sit ABOVE number tasks in raw cosine,
  while number tasks are better ridge-decodable — raw cosine and decodability are not the same
  notion. Verified by manual recomputation of one cell (antonym icl10/finaltok L13: exact match).
- Label-role lines are flatter and lower (~0.29–0.35), nearly saturated by icl03–04.

**Verification:** CSV row/count/NaN checks; hand spot-check exact; R² line plot regenerated
byte-identical with new plotter defaults. **Next:** nothing — awaiting user direction (possible
follow-ups: mean-centered variant; more tasks; Qwen3).

---

## 2026-07-17 — SANDBOX (not repo standard): per-prompt head-sum targets for the full-dim ridge

**Owner:** Claude Code CPU-pod session + own pods `fv-sandbox-perprompt` (A100 ntnl04h2x2ehid — driver too old for env torch 2.12+cu130, terminated unused; RTX PRO 4500 zb7gipjanx16l3 $0.74/hr, terminated after run).
**Status:** DONE (pilot). **Explicitly sandbox per user instruction — do NOT build on this or treat as standard in any capacity unless the user promotes it.**

**Idea (user's):** same X as the canonical full-dim ridge, but per-prompt targets: Y_i = sum over the
train_varicl_top40 heads of out_proj(that prompt's head activation) at the final prompt token (per-task
mean of Y = the FV by linearity; here a fixed-10-shot FV ≈ the stored varicl FV, cos 0.96–0.99).
**New (all under `src/sandbox/perprompt_fv/`):** `capture_perprompt_head_activations.py` (rebuilds the exact
capture prompts from the stable RNGs, asserts stored query_indices; saves all 448 head acts + top-40 head-sum
target per prompt → `artifacts/sandbox/perprompt_head_acts/gptj_train_varicl_top40/`; linearity gate passed
cos=1.000000 all 29 tasks), `regress_activation_to_perprompt_headsum_ridge.py` (canonical protocol, per-prompt Y;
`--target_mode fv_broadcast` repro gate passed: icl10/last L13 test_mse=0.19544 exactly), `summarize_pilot.py`.
**Findings (icl10, 116 cells → `results/sandbox/perprompt_ridge_pilot/gptj_train_varicl_top40/`):** on the
comparable test-vs-stored-FV metric, 68/116 cells improve over the same old cell and 34 beat the old study's
overall best; new best pre_label_token L13 test_mse_fv=0.15284 / R2=0.541 (old best 0.19544 / 0.413,
last_prompt_token L13). vs per-prompt targets: best last_prompt_token L17 test_mse_pp=0.23471 / R2_pp=0.677
(denominator V=0.7376, not comparable to FV R2). Train R2≈0.80; chosen alphas drop ~30x (3.16e4 → 1e3).
**Update 2026-07-17 (later):** user approved extending to the full grid. icl1–9 shards run on own pod
`fv-sandbox-perprompt-2` (ot7w6kjf4qapsj, RTX PRO 4500, terminated after run). All 899 cells:
**482/899 beat the canonical same-cell test-vs-FV MSE; 96 beat the old overall best (0.19544)**; the
advantage grows monotonically with ICL index (icl1 ≈ parity → icl10 best 0.15284/R2 0.541), and
pre_label_token mid-layers (L10–18) dominate the top cells. Full join:
`results/sandbox/perprompt_ridge_pilot/gptj_train_varicl_top40/summary_vs_canonical_all.{csv,md}`.
**Backup:** branch `claude-sandbox-perprompt-ridge` (6074489, pushed; committed via detached worktree — shared
main tree untouched). Logs: `logs/sandbox_perprompt/`.
**Update 2026-07-20:** on user request added `heatmap_test_mse_fv_side_by_side.png` (canonical vs per-prompt,
log10 test-MSE-vs-stored-FV, shared color scale — MSE twin of the R² side-by-side; `plot_pilot_heatmaps.py`
extended, all figures regenerated from summary CSV, byte-changes only to the new file + replots). NOT yet
committed to the sandbox branch (working tree on main; GitHub pushes broken since 2026-07-17 account flag).
**Update 2026-07-26 (weight-spectrum diagnostic, user request):** NEW
`src/sandbox/perprompt_fv/diagnose_weight_spectrum.py` → `.../weight_spectrum/`
(spectra_comparison.png + spectra.npz + summary.json). One cell = sandbox best
(icl10/pre_label_token L13), identical X/standardizer, 4 arms ({canonical, perprompt} ×
{α=3.16e4, 3.16e3}) + centered-target spectra; repro gates vs stored MSEs passed (rel ≲7e-6).
Findings: canonical W keeps the algebraic rank-≈19 cliff at BOTH alphas (sv20/sv1 ≤ 9e-6;
energy-90 rank 15–16). Per-prompt W has NO cliff: sv21/sv1 = 0.64 own-α (0.31 at swapped α),
energy-90 rank 441 (95 at large α), PR 178 (28) — the tail is target-driven, α only modulates
its weight. Target spectra: per-prompt centered Y coincides with canonical Y's spectrum over
the top ~19 (between-task structure) then continues smoothly (σ20/σ1 ≈ 0.16, numrank 3399 =
within-task prompt variation) — exactly the structure the map's extra directions inherit.
Open follow-up if wanted: principal angles between the two maps' top-20 singular subspaces.

**Next:** nothing — awaiting user direction (promote / analyze further / discard). STILL SANDBOX.

## 2026-07-16 — Stream W-5shot: 5-shot preimage/FV ablation + combined-token rows (GPT-J)

**Owner:** Coordinator (tmux `fv-5shot-ablation`; CPU pod + 4 own RunPod pods
`fv-5shot-ablation-{1..4}` rib96bs48mq59r / 9yg6cdfkdghfcn / 5f5og0ywb6yejq / 7zq4301lb8davy,
RTX PRO 4500 $0.74/hr each).
**Status:** IN PROGRESS.

**What:** extends Stream W (1-shot, canonical `train_varicl_top40` FVs) to 5-shot prompts. Same
7 held-out tasks, 6 arms, Δ log p metric, seed-42 sampling, cf_map. 14 rows/arm: cue1..5,
target1..5, final_cue individually + combined simultaneous-ablation rows all_targets, all_cues,
all_cues_incl_final. Matched arm: cue_i←pre_label_icl{i}, target_i←last_label_icl{i},
final_cue←pre_label_icl6; combined rows keep per-position matched cells. icl10 arm: cues←
pre_label_icl10, targets←last_label_icl10; individual final_cue keeps last_prompt_icl10; in
all_cues_incl_final the final cue gets pre_label_icl10 (user decision: same vector at every
cue). 8 NEW ridge+TSVD cells (pre_label_icl{3..6}, last_label_icl{2..5}) fit from the existing
10-shot captures (same provenance as all existing cells; NOTE 5-shot demo draws are a fresh
deterministic sample — seed excludes n_shots — so 1-shot demos are not nested in 5-shot prompts).
**USER-REQUIRED GATE:** new cells' per-layer test MSEs must match the stored full-dim ridge R²
study (`check_ridge_mse_vs_r2_study.py`); on mismatch STOP and inform the user — no
self-adjudication. Gate pre-validated on the 6 existing cells: rel diff exactly 0.0 (140 rows).

**Files:** NEW `src/eval_scripts/ablate_fiveshot_preimage_logprob.py` (standalone sibling of the
1-shot script — 1-shot script untouched, another stream is editing it),
`plot_fiveshot_preimage_ablation.py`, `check_ridge_mse_vs_r2_study.py`,
`logs/fiveshot_preimage_ablation/driver.sh`. Output →
`results/direction3_fv_formation/fiveshot_preimage_ablation/train_varicl_top40/`.
Defaults: batch 85, TraceDict(retain_output=False), use_cache=False (5-shot prompts ~52-64 tok).

**Status update 2026-07-17: DONE.** All 42 (task,arm) npz complete (shape 14x28x170, all finite;
combined_summary.csv independently re-verified against the npz, 392/392 rows match). Gates:
TSVD lincheck <7e-6; **MSE gate: all 8 new cells reproduce the stored R² study test MSEs with
rel diff exactly 0.0 (224 rows)** — no discrepancy to escalate. Pods terminated. Ops note: the
pod-1 driver rolled from smoke straight into an unsharded serial sweep (~2.3h idle on pods 2-4,
~$5); killed at capitalize_first_letter midpoint and resharded — future drivers should gate the
full sweep behind an explicit TASKS_SHARD.

**Findings (7-task mean Δ log p, figures in `.../fiveshot_preimage_ablation/train_varicl_top40/
figures/`; `_targets`/`_cues` variants are rescaled subsets):**
- **Combined target ablation is strongly superadditive.** all_targets (5 demo labels at once,
  matched preimages, L0) = **−2.19** (cf −0.20) vs Σ individual targets ≈ −0.79. Per-token
  redundancy: each individual target is small in 5-shot (target1 −0.08 vs −1.78 in 1-shot;
  target5 −0.29) because the other 4 demos still carry the task, but removing the preimage
  component at ALL labels collapses it. Ordering: later demos matter more (t5>t4>...>t1).
- **Label-token coordinates are still preimage, not FV:** all_targets via matched preimages
  −2.19, icl10 preimages −1.53, raw FV direction only −0.16. Strictly early-layer (≈0 by L20),
  as in 1-shot.
- **final_cue reproduces 1-shot:** FV −7.41 (cf −0.71; 1-shot was −7.13), icl10 −3.42, matched
  (pre_label_icl6) −2.86; still biting at L20 (FV −0.93, icl10 −1.22). FV coords dominate at the
  query cue; preimage coords at the labels.
- **Cue tokens are causally null even jointly:** every individual cue ≈0 AND all_cues (5 at
  once) ≈ −0.01; all_cues_incl_final ≈ final_cue alone (−6.58 vs −7.41 FV) — demo cues add
  nothing on top of the query cue; per icl10-arm design the final cue there uses
  pre_label_icl10, which costs it ~0.7 vs the individual final_cue row's last_prompt_icl10.
- **Per-task concentration matches 1-shot:** all_targets(matched, L0) driven by landmark-country
  −6.09, lowercase_first_letter −5.77 (cf −1.46 — same confounded near-twin cf draw as 1-shot),
  capitalize_first_letter −2.51; ≈−0.2 for antonym/synonym/capitalize/word_length.
- Runtime: stage A 8 cells sharded 4 pods ≈ 25 min; sweep ≈ 45-60 min/task (~5.5h GPU-time
  total across 4x RTX PRO 4500, ~$11 incl. the idle mistake).
**Next:** (a) superadditivity curve — ablate k=1..5 targets to map the redundancy threshold;
(b) same combined-row study in the paired-prompt logit-gap regime; (c) re-draw
lowercase_first_letter's confounded counterfactual; (d) sum-over-answer-tokens metric for
capitalize.

## 2026-07-17 — Stream X5: prev/next_number in the PROPAGATED fixed-direction ablation (separate study)

**Owner:** Coordinator (tmux `pertask-r2-heatmaps`; own RunPod Blackwell 4500 pod
`fv-numbers-prop-ablation` av1de1n4s2wtg3 $0.74/hr — TERMINATED). **Status:** DONE. Committed on
branch `claude-pertask-r2`.

**What:** the two number tasks through `--mode propagated` (fixed anchor-layer direction U[L],
anchor + all later tokens, blocks ≥ L), SEPARATE study →
`results/direction3_fv_formation/oneshot_preimage_ablation_propagated_numbers/train_varicl_top40/`
(banks from Stream X4; driver `logs/numbers_ablation/driver_propagated.sh`; per-task figures
`figures/heatmap_all_arms_{prev,next}_number.png`). fv/fv_cf final_cue rows reproduce the X4
perlayer values exactly (structural equivalence at the last token — free correctness check).

**Findings (leakage caveat as X4):**
- **The X4 inversion (icl10 preimage > FV at final_cue) does NOT survive fixing the direction:**
  fixed U[0] does ≈ NOTHING (−0.02 both tasks) vs the per-layer stack −3.54/−6.20 — the number
  tasks\u2019 preimage advantage lives in the ROTATING per-layer stack, no single layer\u2019s direction
  carries it. Most-damaging single fixed direction is the LAST layer\u2019s (min at L27: −1.28
  prev / −1.81 next; 7-task icl10 peaked at L20 instead).
- **FV propagated from cue1/target1 is near-catastrophic** (prev −3.4, next −5.6 at L9),
  mirroring the 7-task propagated story, with reduced specificity (fv_cf ~−2.5 early).
- position-matched fixed directions stay weak everywhere (min −0.23/−0.27 mid-layers).

---

## 2026-07-17 — Stream X4: prev/next_number in the ORIGINAL 1-shot per-layer ablation (separate study)

**Owner:** Coordinator (tmux `pertask-r2-heatmaps`; own RunPod RTX PRO 4500 Blackwell pod
`fv-numbers-ablation` 79od3s4x9bpjrn $0.74/hr, `allowedCudaVersions` filter per DECISIONS —
TERMINATED). **Status:** DONE. Committed on branch `claude-pertask-r2`.

**What:** user request — run prev_number/next_number through the original per-layer 1-shot
ablation, fully SEPARATE from the 7-task aggregate. New TSVD banks for the two tasks (from the
existing canonical stage-A ridge maps; **TRAP:** `fit_tsvd_preimages_multicell.py` DEFAULTS point
at the max4 DEBUG preimage_root and the two-shot cell set — pass `--preimage_root .../train_
varicl_top40` and the 6 ablation cells explicitly, as the v2 driver did; a first run with
defaults wrote 12 stray banks into the debug tree, deleted). Ablation →
`results/direction3_fv_formation/oneshot_preimage_ablation_numbers/train_varicl_top40/`
(summaries only ever see these 2 tasks). Driver `logs/numbers_ablation/driver.sh`. Per-task
figures via new `--tasks` filter on the plot script: `figures/heatmap_all_arms_{prev,next}_
number.png` (+ per-task grids). 7-task study dir untouched by this run (verified by mtimes;
NOTE: its 12 per-arm PNGs were deleted from the working tree at 2026-07-16 21:28 by something
outside this stream — versions live in git (HEAD pre-retitle; claude-pertask-r2 retitled);
left as-is for the owning stream to adjudicate).

**Findings (task-level; leakage caveat: prev_item/next_item are TRAIN tasks — upper bounds):**
- **The 7-task ordering INVERTS at the final cue: the icl10-regression preimage out-damages the
  FV direction.** prev_number: icl10 preimage −3.54 @ L0 vs FV −2.55 @ L0 (FV min −2.95 @ L9);
  next_number: icl10 preimage −6.20 @ L0 vs FV −4.15 @ L0 (min −4.76 @ L9). In the 7-task study
  FV (−7.1) ≫ icl10 preimage (−4.0). Task-specific: icl10_cf only −0.15 / −0.94.
- position-matched preimage (pre_label_icl2 cell) is weak at final_cue (−0.33 / −0.48), target1
  effects are tiny for both tasks (≈ −0.1..−0.18; cf ≈ 0) — unlike landmark-country/case tasks,
  the number tasks carry almost nothing causally at the demo label token.
- FV arm's max damage at L9 (not L0) for both tasks — the small early-start self-repair shape
  again.

---

## 2026-07-16 — Stream X3: PROPAGATED fixed-direction 1-shot preimage ablation

**Owner:** Coordinator (tmux `pertask-r2-heatmaps`; own RunPod L4 pod `fv-propagated-ablation-3`
xfyifkkbdo4am6 $0.39/hr — TERMINATED; two earlier 4090 pods stalled at boot: the template image
needs **CUDA ≥ 12.8** and those hosts had 12.4 drivers → ALWAYS pass
`allowedCudaVersions: ["12.8","12.9","13.0"]` in podFindAndDeployOnDemand; see DECISIONS).
**Status:** DONE. Committed on branch `claude-pertask-r2`.

**Design (user-specified):** same 6 arms/7 tasks/28 start layers as Stream W's perlayer study,
but (1) the ablated direction is FIXED to U[L], the preimage of the regression at the anchor
layer L (fv arm unchanged in direction); (2) it is projected out at the anchor token AND every
later token position, for all blocks b ≥ L. NEW `--mode propagated` on
`ablate_oneshot_preimage_logprob.py` (default `perlayer` bit-identical — verified); results →
`results/direction3_fv_formation/oneshot_preimage_ablation_propagated/train_varicl_top40/`
(full npz kept per (task, arm); figures: heatmap_all_arms + per_task_grid only, via new
`--skip_per_arm` plot flag). Driver `logs/propagated_ablation/driver.sh`.

**Verification:** (a) perlayer regression — edited script vs HEAD script on the SAME L4 GPU:
max|Δ| = 0.0 across all 6 arms (an earlier check against the stored v2 smoke tripped at 1.7e-2;
that is pure cross-GPU fp16 noise — HEAD-vs-HEAD across GPUs shows the same 1.7e-2; regression
checks must compare same-GPU); (b) cross-mode equivalence — fv arms' final_cue row equal at
every L, all arms' final_cue equal at L=27 (single-position/single-block cases where the modes
coincide by construction): all passed (`EQUIVALENCE_OK` in driver log).

**Findings (task-mean Δ log p; per-arm numbers in per-task summary.csv):**
- **FV direction becomes catastrophic from ANY anchor when propagated:** cue1/target1/final_cue
  all ≈ −7.1..−7.5 at L0–8 (perlayer: only final_cue −7.2; cue1/target1 ≈ −0.1/−0.4). But
  specificity DROPS: fv_cf now −2.2..−2.4 at cue1/target1 (perlayer cf ≈ −0.1/−0.3), i.e.
  propagated FV removal is ~3× task-specific vs ~6× at final_cue and ~8-20× perlayer.
- **The preimage stack is NOT one direction:** at final_cue (where propagated = fixed-direction
  anchor-only), icl10-preimage removal collapses from −4.0 (perlayer, removing each layer's own
  direction) to ≈0.0 at L0 with the fixed U[0]; the only surviving fixed-direction effect is
  late (−1.65 @ L20). Same collapse for matched at final_cue (−1.94 → ≈0 at L0, −0.99 @ L15).
  ⇒ the causal preimage content ROTATES across layers; no single layer's preimage direction
  carries it.
- **target1 (demo label) effect weakens and loses specificity when fixed+propagated:** matched
  −1.78→−0.82 at L0 (min −1.53 @ L1) while its cf grows to −0.81 @ L1 (~1.9× specific vs ~8×
  perlayer).
- **NEW late-layer band for icl10 preimage at cue1** (min −1.62 @ L20; absent perlayer) —
  propagating the late-layer icl10 direction over all downstream tokens (which include the
  final cue) reaches the readout path.

---

## 2026-07-15 — Stream X2: GPT-4.1-judged accuracy-vs-n_shots (antonym/synonym)

**Owner:** Coordinator (tmux `pertask-r2-heatmaps`; generate stage on own RunPod 4090 pod
`fv-judged-accuracy` 2mcpknnn4ncwue $0.69/hr — TERMINATED; judge+plot on CPU pod).
**Status:** DONE. Committed on branch `claude-pertask-r2`.

**What:** gold-first-token top-1 undercounts open-ended tasks (1-shot judge study, WORKLOG
2026-06-30). This re-runs the by_nshots accuracy grid for antonym+synonym (n=0..10, full test
splits, EXACT same prompts: set_seed(seed+n) + per-item np.random.choice in test order) but
stores per-prompt records and scores the top-1 token with the gpt-4.1 judge (JUDGE_SYSTEMS from
judge_oneshot_paired.py; same-word/inflection=false). Judge convention: top-1 TOKEN as-is
(whitespace-trimmed) — word fragments judged false (slightly conservative vs decoding a full
word).

**Files:** NEW `src/eval_scripts/compute_judged_accuracy_by_nshots.py` (stages generate/judge/
summarize, resumable per (task,n)) + `plot_judged_accuracy_by_nshots.py`. Data (TRACKED):
`results/general/task_accuracies/by_nshots_judged/{task}_n{n}.json` (per-prompt: query, gold,
top-5 tokens, gold_rank, judge_correct) + summary.json. Figure:
`.../tenshot_strip_intervention_cos_heatmap/figures/gptj_judged_accuracy_by_nshots_top1.png` (all 4 strip-study tasks: judged solid + gold dashed for antonym/synonym; digit tasks gold-only, exact by construction).

**Findings:**
- antonym: judged ≈ gold + 0.07–0.10 from n=2 on; plateau 0.70–0.74 (gold 0.62–0.66).
- synonym: judged ≈ 1.8–2× gold everywhere; plateau 0.41–0.45 (gold 0.23–0.27). The n=1 cell
  (0.149 vs gold 0.071) reproduces the 1-shot judge study (0.143/0.066) on a different prompt
  sample. Ordering antonym > synonym unchanged; the GAP is larger judged (~0.28) than gold
  (~0.40 vs 0.23 → ~0.40 ratio-wise smaller). Both tasks: correction ≈ constant multiple after
  n≥2, so curve SHAPES (saturation ~n=5) are metric-robust.
- Gold-top1 re-derived from the same records matches the recorded by_nshots numbers to within
  1–2 prompts/cell (≤0.4%; exact at n=0,1) — cross-GPU (L4 vs 4090) logit noise flipping
  near-tied argmaxes, NOT a prompt mismatch. summary.json flags matches_reference at 1e-9
  tolerance, so False there means "±1-2 prompts", read the gold_top1 columns.

**Addendum (2026-07-16):** retitled the Stream W ablation-heatmap arms for clarity — "preimage (matched cells)" → "preimage of position-matched regression", "preimage (icl10 cells)" → "preimage of icl10 regression"; suptitles now state all ablations are applied in 1-shot prompts. plot_oneshot_preimage_ablation.py + all figures regenerated (data untouched).

---

## 2026-07-14 — Stream X: per-test-task R² heatmaps (antonym/synonym/prev_number/next_number)

**Owner:** Coordinator (tmux `pertask-r2-heatmaps`; CPU pod + own RunPod GPU pod
`fv-pertask-r2-capture` bznaqhdemfkl3x, NVIDIA L4 $0.39/hr — no 4090 in EU-RO-1 stock; L4 is
sm_89 so safe with the image's pinned torch, per DECISIONS Blackwell note; pod TERMINATED).
**Status:** DONE. Committed on branch `claude-pertask-r2`.

**Goal:** the pooled `combined_test_r2_heatmap.png` (varicl_top40 study) aggregates the 7 held-out
test tasks; user wants per-task (token position × layer) R² heatmaps for antonym, synonym,
prev_number, next_number.

**Design:**
- NEW `src/eval_scripts/plot_fulldim_ridge_pertask_r2.py`: rescales the stored
  `per_test_task_mse` (present in every shard's metrics.json) into per-task R² with per-task
  denominator V_task = ||fv_task − ȳ_train||²/hidden (train-mean baseline, same convention as
  the pooled R²). No refit. Outputs under `<study>/per_task_r2/`: per-task PNGs (shared color
  scale), combined panel, per_task_r2.csv, summary.json.
- antonym/synonym: already test tasks → plotted directly from the existing
  `fulldim_ridge_activation_to_fv_varicl_top40` shards. Best R²: antonym 0.300, synonym 0.289
  (both icl10/finaltok L13); pooled-study structure (mid-layer band) reproduces per task.
- prev_number/next_number: NOT in the 29-task split and had no captured activations. Captured
  them into the EXISTING activation roots (`gptj_56tasks_170prompts_icl{1..9}_3tokens` +
  `_4tokens`; dir name now understates task count; no consumer globs task subdirs — all pass
  explicit task lists) via `logs/pertask_r2_numbers/capture_driver.sh` (same config: seed 42,
  130/40 prompts, fp16, embeddings). Then re-ran the 10 ridge shards with `--test_tasks
  <7 defaults> prev_number next_number` → NEW study dir
  `fulldim_ridge_activation_to_fv_varicl_top40_plus_numbers` (fits/CV are train-only, so models
  are identical to the existing study; only the eval set grows). Driver:
  `logs/pertask_r2_numbers/ridge_driver.sh`.
- **Caveat for the number tasks:** `prev_item`/`next_item` (number-word tasks, ~10% pair overlap
  with prev/next_number) are among the 20 TRAIN tasks, so prev/next_number are held-out in form
  but leakage-adjacent in content — flag this next to any cross-task comparison.

**Findings (per-task test R², train-mean baseline; `per_task_r2/summary.json` in each study dir):**
- antonym best **0.300**, synonym **0.289**, both at icl10/finaltok **L13** — same late/mid-layer
  band as the pooled heatmap, peaks at the final prompt token.
- prev_number best **0.337** (icl09/pre **L9**), next_number **0.375** (icl08/pre **L9**) — the
  number tasks are BETTER predicted and peak EARLIER (L6–10) and at PRE-LABEL positions rather
  than the final token; their high-R² band starts around L4–6 vs L9–10 for antonym/synonym
  (visible in `test_r2_heatmap_panel.png`, shared scale). Consistent with the leakage-adjacency
  caveat above (train set contains prev_item/next_item), so treat the level as an upper bound;
  the position/layer profile shift is the more interesting observation.
- New-study shard icl1 verified to reproduce the existing varicl_top40 study exactly: identical
  best_alpha in all 87 cells, 7-task per-task MSEs equal to ~1e-6 relative (L4-vs-prior-GPU fp
  noise). Pooled 9-task R² (max 0.391 at icl10/finaltok L13) is NOT comparable to the 7-task
  0.465 — different test-task set → different denominator V.
- Sanity: shard metrics store `per_test_task_mse` per cell (since the varicl_top40 run), which is
  what makes per-task R² a pure CPU post-processing step.

**Addendum:** `src/eval_scripts/plot_pertask_r2_best_lines.py` → `per_task_r2/
best_r2_by_position_lines.png` (plus_numbers study): best-over-layers R² per token position,
one line per task. Shows a within-example sawtooth: every task peaks at the `pre` position of
each cycle (with icl10/finaltok matching the pre-level for antonym/synonym), number tasks sit
~0.05–0.08 above antonym/synonym at every position, and positions saturate by ~icl04–05.
Role-filtered variants (`--roles`): `..._prelabel.png` (pre_label_token only — smooth monotone
saturation, antonym slowest to saturate ~icl05–07) and `..._label.png` (label tokens, one point
per example = mean of first/last best-R² via `--average_roles` — flatter, ~0.03–0.06 lower than
pre positions at every ICL depth).

**Verification:** icl1 activations for both number tasks load through the ridge's own loader with
shape (170, 29, 4096) fp16, matching existing tasks; alpha choices and original-task MSEs
reproduce (above). NOTE layer-0 per-task R² is 0 at the pre-label positions but up to ~0.25 at
label/final positions (token-identity signal in the embeddings) — only the POOLED L0 pre-label
cell is ≈0; don't claim "L0 ≈ 0" per task.

---

## 2026-07-10 — Stream W: 1-shot preimage-ablation causal test (GPT-J, 7 held-out tasks)

**Owner:** Coordinator (tmux `fv-preimage-ablation`; CPU pod + own RunPod GPU pod
`fv-preimage-ablation` nijtdy6z6jzn18, RTX PRO 4500 $0.74/hr).
**Status:** v2 IN PROGRESS on the CANONICAL FVs (`train_varicl_top40`). v1 was fit against the
max4 DEBUG FV set (see DECISIONS 2026-07-10; max4 moved to `.../gpt-j/debug/`) — its results
(`.../oneshot_preimage_ablation/train_varicl_max4_top40/`) are kept only as a debug reference.

**v2 design deltas vs v1:** both preimage arms have exactly 3 rows. `preimage_matched` =
position-matched cells {cue1←pre_label_icl1, target1←last_label_icl1, final_cue←pre_label_icl2
(the 1-shot query token is a "pre label 2" by causal context)}; `preimage_icl10` =
{pre_label_icl10, last_label_icl10, last_prompt_icl10}. Ridge cells REFIT against
train_varicl_top40 targets (fit uses the 20 TRAIN tasks only; 7 test tasks fully held out):
`fit_ridge_preimages_multicell.py --cells <6 cells> --fv_root .../train_varicl_top40
--output_root artifacts/preimage_pairdiff/train_varicl_top40 --pair_specs antonym:synonym`
(digits pair dropped — no digits FVs in this root). TSVD banks →
`artifacts/preimage_pairdiff_tsvdk16/train_varicl_top40/` (linearity-checked). Ablation output →
`results/direction3_fv_formation/oneshot_preimage_ablation/train_varicl_top40/`. NEW Stage E:
first-ever full-dim ridge R² study for this root →
`results/direction3_fv_formation/fulldim_ridge_activation_to_fv_varicl_top40/` (10 shards +
merge + combined_test_r2_heatmap.png, comparable to the train_selected study).
Driver: `logs/oneshot_preimage_ablation/driver.sh` (v2 flags), tmux `streamw` on the pod.
Known pod quirk: matplotlib/numpy clash on the pod (v1 stageD ImportError) — plots are generated
on the CPU pod instead.

**Findings (v2, canonical FVs; mean Δ log p of the correct first answer token over 7 tasks × 170
one-shot prompts; figures `.../oneshot_preimage_ablation/train_varicl_top40/figures/`).**
NOTE: an earlier interim readout quoted antonym-only numbers as 7-task means (per-task
invocations clobbered combined_summary.csv with a subset; script now aggregates all task dirs on
disk). Numbers below are the corrected 7-task means, verified directly against the npz:
- **Early layers at the DEMO LABEL (target1) are causally load-bearing — via the PREIMAGE
  directions, not the FV direction.** Ablating from L0 at target1: matched-cell preimage
  (last_label_icl1) **−1.78** (cf twin −0.23, ~8× task-specific); icl10 preimage −1.12 (cf −0.13);
  raw FV direction only −0.39 (cf −0.26, barely specific). Strictly early-layer (≈0 by L8–12).
  Per-task: driven by landmark-country (−4.1), lowercase_first_letter (−4.5),
  capitalize_first_letter (−2.6); ≈0 for synonym/antonym. At the label token the task-specific
  causal coordinates are the ridge PREIMAGE of the FV, not the FV itself.
- **The final cue (query "A:") carries the largest effects:** FV −7.13 (cf −1.17), icl10 preimage
  −4.00 (cf −0.90), matched (pre_label_icl2) preimage −1.94 (cf −0.27). Late-start ablation still
  bites (L20: FV −2.84, icl10 preimage −2.38, matched −0.89) — "late layers at the cue" holds at
  the query cue. Coordinate flip vs the label token: at the final cue the FV direction itself is
  the most causal; at the label the preimage is.
- **cue1 (demo "A:") ≈ 0 everywhere** (all arms ≤ −0.13).
- **All three same-task arms pass the counterfactual control** (~4–9× same-vs-cf at their active
  sites). Caveat: lowercase_first_letter's random counterfactual drew its near-twin
  capitalize_first_letter, inflating cf baselines somewhat.
- Hypothesis verdict: **supported** — early layers at the target token and late layers at the
  (query) cue token both reduce log p of the correct answer; the label-token effect is carried by
  preimage coordinates rather than the FV direction itself. target1-only rescaled figures:
  `heatmap_all_arms_target1.png`, `per_task_grid_target1.png` (+ `_test40` variants).
- **Stage-1 ridge quality (canonical root):** V(test|train-mean)=0.333; best cells
  last_prompt/pre_label icl10 L13, test_mse 0.195 → R² ≈ 0.41; pre_label_icl1 weakest (R²≈0.11).
- **NEW full-dim ridge R² study for train_varicl_top40**
  (`results/direction3_fv_formation/fulldim_ridge_activation_to_fv_varicl_top40/`, 899 cells):
  best test R² 0.4128 @ icl10/finaltok L13 (train R² 0.969), 888/899 cells beat the train-mean
  baseline; same mid-layer ridge (L9–15) and role banding as the train_selected study (max 0.465).
  11 cells alpha-pinned at a grid endpoint (flagged in the merge log).
- Runtime: Stage A 6 cells ≈ 75 min; sweep ≈ 25 min; R² study 10 shards ≈ 100 min — all on one
  RTX PRO 4500 ($0.74/hr). Pod quirk: matplotlib/numpy import clash → all merge/R²/plot steps
  ran on the CPU pod instead.
**Next:** deeper follow-ups: (a) why the label-token effect lives in preimage coordinates while
the final-cue effect lives in FV coordinates (transport/rotation across layers?); (b)
sum-over-answer-tokens metric for capitalize (52% multi-token); (c) per-task heterogeneity —
target1 causality is concentrated in landmark-country / lowercase_first_letter /
capitalize_first_letter; (d) re-draw lowercase_first_letter's confounded counterfactual;
(e) rank-k ablation at target1; (f) repeat in the paired-prompt logit-gap regime.

**Question:** are the per-layer TSVD-16 ridge preimages of a task's FV causally load-bearing?
On 1-shot prompts over the 7 ridge held-out tasks (landmark-country, word_length,
capitalize_first_letter, synonym, lowercase_first_letter, capitalize, antonym), ablate the
residual-stream component along a candidate direction at one site token, from start layer L
through all downstream layers (at that token only), and measure Δ log p (ablated − clean) of the
first answer token at the final position. Hypothesis (from earlier per-task results): damage
concentrates at EARLY layers at the target token and LATE layers at the cue token.

**Design:** 170 one-shot prompts/task (130 train-split + 40 test-split queries, seed 42,
capture-pipeline-identical construction incl. `'<|endoftext|>'` BOS string prepend). Site tokens:
cue1 (pre_label of demo), target1 (last_label of demo), final_cue (last prompt token). 6 arms:
{matched-cell preimage, icl10-cell preimage, FV direction} × {same task, random counterfactual
task from the other 6}. Matched cells: pre_label_icl1 / last_label_icl1 / pre_label_icl2 +
last_prompt_icl10 (final_cue gets both). Preimages = rank-16 TSVD inverse of the Stream S ridge
maps (fv_root train_varicl_max4_top40), unit-normalized per edit layer. Start-layer sweep over
edit layers 0..27 (= `transformer.h.b` outputs; bank edit_layer b ↔ capture entry b+1; embedding
never ablated). Heatmaps: token-row × start-layer per arm, task-mean Δ log p, shared scale.

**Stages:** A) fit missing ridge cells pre_label_icl10 + last_label_icl10
(`fit_ridge_preimages_multicell.py`, GPU). B) single-task TSVD banks for the 7 task FVs × 6 cells
(extended `fit_tsvd_preimages_multicell.py`, CPU; linearity-validated vs the antonym__synonym
pairdiff bank). C) NEW `src/eval_scripts/ablate_oneshot_preimage_logprob.py` (GPU; resumable
per (task, arm) npz; ~3.9k batched forwards). D) NEW `plot_oneshot_preimage_ablation.py`.
Driver: `logs/oneshot_preimage_ablation/driver.sh`. Outputs:
`results/direction3_fv_formation/oneshot_preimage_ablation/train_varicl_max4_top40/`.

**Findings:** (pending)
**Next:** (pending)

## 2026-07-08 — Stream V: shuffled-label control for the full-dim ridge R²

**Owner:** Coordinator (tmux `fvridge-shuffled-control`; CPU pod + RunPod GPU pod `fv-shuffled-ridge`).
**Status:** DONE — control collapsed to R² ≈ 0 as required; pod terminated; deliverables committed.

**RESULT: the real R² survives the control.** Shuffled-label test R² (3-seed mean over 899 cells):
median −0.021, max 0.0000 (the max is icl01/pre L0 — the trivial train-mean cell); per-seed maxima
1e-7 / 0.016 / 1e-7; only 1.2% of cells microscopically > 0. vs the REAL run's median 0.346 /
max 0.465. Median test MSE 0.222 ≈ V(test|train-mean)=0.2171. Ridge reacts by pinning alpha at/near
the 1e8 grid top (35 pinned cells per seed vs a handful in the real run) → predicts the train-mean
FV. The R² heatmap is structureless (whole grid in [−0.04, 0]); no mid-layer ridge. See DECISIONS.md
2026-07-08 entry. Runtime: ~2.2h on one RTX PRO 4500 ($0.74/hr), 3 seeds concurrent.

**Question:** sanity-check the full-dim ridge R² (max 0.465): permute the train-task→FV assignment
(task-level, test targets untouched), retrain the identical pipeline (incl. LOO-task alpha CV), and
compare test MSE/R² heatmaps. If the real R² is meaningful, the control should collapse to ≈0.

**Design:** task-level permutation over the 20 sorted train tasks via
`np.random.default_rng(seed).permutation`; 3 seeds (0,1,2) averaged; same seed ⇒ same permutation in
every shard, so each seed is one coherent shuffled dataset. R² denominator V is permutation-invariant
(same target set) ⇒ R² directly comparable to the real run.

**Commands:**
- Worker flags: `--shuffle_train_labels --shuffle_seed {s}` (new); mapping recorded in run_config.json.
- Pod: RunPod RTX PRO 4500 Blackwell ($0.74/hr, EU-RO-1, shared volume), pod id `09jf2ydfj597he`.
- Driver: `logs/shuffled_control/run_seeds.sh` (3 concurrent seed processes × 10 serial shards each,
  then merge + R² per seed, then `average_shuffled_ridge_seeds.py`). Logs: `logs/shuffled_control/seed{s}.log`.

**Smoke tests (on pod):** flag OFF reproduces committed icl10/finaltok L11 test_mse=0.11607 exactly;
flag ON (seed 0, a derangement, 0/20 fixed points) → test_mse=0.21741 ≈ V(test|train-mean)=0.2171,
alpha pinned at 1e8 (ridge → train mean), i.e. R² ≈ 0 as predicted.

**Bug found en route:** `residual_activations/*/index.json` shard paths are ABSOLUTE and predate the
2026-06-19 results→artifacts reorg; loader now falls back to the split dir (commit 42044d9).

**Files:** `regress_activation_to_fv_fulldim_ridge.py` (shuffle flags + path fallback),
`average_shuffled_ridge_seeds.py` (new). Outputs (pending):
`results/direction3_fv_formation/fulldim_ridge_activation_to_fv_shuffled_seed{0,1,2}/` + averaged
`fulldim_ridge_activation_to_fv_shuffled/`.

**Extension (2026-07-08/09): ROW-level shuffle variant — DONE.** Second control: permute the
row→FV assignment across all 3400 train rows (`--shuffle_mode row`; without replacement, each FV
keeps 170 rows; fixed per seed across all cells/shards; test rows true). 3 seeds on RunPod pod
`fv-rowshuffle-ridge` (`m99eh1szzwgyd0`, terminated), driver `logs/shuffled_control/run_rowshuffle_seeds.sh`,
outputs `fulldim_ridge_activation_to_fv_rowshuffled{,_seed0,1,2}/`. Smoke: flag-off still 0.11607;
row seed 0 own-task fraction 0.0547, test_mse 0.21618 ≈ V, and train_mse 0.19739 ≈ V_train(0.1978)
— in row mode the fit can't even memorize train (conflicting targets for near-identical activations).

**Row-shuffle RESULT:** even flatter than the task-level control. Seed-mean test R² median 1.6e-5,
max 0.0028 (per-seed maxima 0.0043 / 8e-8 / 0.0084); median test MSE 0.21711 = V to 5 decimal
places; heatmap is noise in ±0.003 with no layer/position structure. Median R² is ~0 (not −0.02 as
in task mode) because the shuffled targets are balanced WITHIN every CV fold, so LOO-CV finds the
global-mean predictor exactly optimal — the cleaner collapse of the two controls. Summary table:
real 0.346/0.465 → task-shuffle −0.021/0.0000 → row-shuffle 0.0000/0.0028 (median/max test R²).

**Extension (2026-07-09): weight-matrix heatmaps for the 6 best cells** (icl08–10/pre L11;
icl08/last L11, icl09–10/last L13). `plot_fulldim_ridge_weight_heatmaps.py` refits at stored
alphas (refit test MSE matches combined_metrics to 5 dp), saves W [4096,4096] + scaler to
`artifacts/fulldim_ridge_weight_matrices/`, renders signed heatmaps + singular spectra to
`results/direction3_fv_formation/fulldim_ridge_weight_heatmaps/`. Findings: W is DENSE and
unstructured in the neuron basis (salt-and-pepper; faint row/col banding from high-norm FV/act
dims); NOTE mean-pooled downsampling renders it blank — use signed max-|.| pooling. Spectra:
~19 large singular values then a 5-decade cliff (rank ≤ 19 after target centering, matching the
rank ≤ #train-tasks bound in DECISIONS). last-label maps have larger weights overall
(|W|_F ≈ 2.83 vs 1.99, alpha 1e4 vs 3.16e4). Full-res per-matrix PNGs (50 MB each) live in
artifacts, not results.

**Extension (2026-07-09): task-space confusion matrices** (`plot_ridge_task_confusion.py`,
`task_confusion_6cells.png`): per cell, each task's mean activation → map → cosine of centered
predicted FV vs all 27 centered true FVs. Train diag ≈ 0.995–0.998 (memorized). Test diag ≈
0.67–0.70 BUT test top-1 = 0/7 in every cell: a held-out task's prediction is always closest to
some related TRAIN task's FV (capitalize_first_letter→capitalize_last_letter, etc.) — predictions
live in span(centered train FVs), the same ~45% reconstruction ceiling as Stream U. Visible
confusion blocks: translation trio (english-french/german/spanish), capitalization family.
Matrices npz in `artifacts/fulldim_ridge_weight_matrices/`.

**Extension (2026-07-09): 3D FV-PCA plots with mapped test predictions**
(`plot_fv_pca3d_predictions.py`, `fv_pca3d_<cell>.png` × 6): the 27 FVs in their top-3 PCs
(27.9/16.1/9.6% = 53.6% var), test FVs red, ridge-mapped test-task mean activations as green X
with dashed connectors to the true FV. Predictions consistently SHRINK toward the train-FV cloud:
test FVs on the periphery (capitalize, antonym, word_length) have predictions pulled well inside;
landmark-country (whose FV already sits inside the train cloud near park-country/national_parks)
is predicted nearly on top of itself. Visual counterpart of the top1=0 confusion result.

**Extension (2026-07-09): ICL trajectories in FV-PCA space** (`plot_fv_pca3d_icl_trajectories.py`;
fitted the 14 missing banks icl01–07 × pre/last at per-position best layers, refits == stored MSE;
bank set now covers all 20 (icl, pre/last) cells). `fv_pca3d_icl_trajectories.png` +
`fv_pred_error_vs_icl.png`: pre-":"-token predictions start near the mean FV at icl01 (error 21–36)
and converge by ~icl4 with small refinements after; last-label predictions are already near final
accuracy at icl01 (first completed demonstration carries most of the FV-relevant info) and stay
flat. Both roles converge to the same shrunk-toward-train-cloud endpoints; landmark-country
remains the best-predicted test task (‖err‖ ≈ 10–11 vs 18–28 for the rest).

**Next:** none — stream complete. (Possible follow-up: same control for the k=16 PCA ridge; expected
to behave identically given the full-dim result.)

---

## 2026-07-06 — Stream U: post-hoc R² for the full-dim ridge (activation→FV)

**Owner:** Coordinator (tmux; CPU pod). **Status:** DONE (untracked; commit pending user request).

**Question:** the full-dim (4096→4096) ridge regressions are only reported as test MSE (heatmaps).
What are the R²?

**Key trick — no re-fit needed.** The target set is identical in every (token position, layer) cell:
the 7 test-task FVs, each broadcast to that task's 170 activation rows; only the input features X
change per cell. So the R² denominator SS_tot is a *single constant* V (per-element target variance
about a baseline mean), and `R²(cell) = 1 − test_mse(cell)/V`. R² is thus a monotone rescaling of the
existing test_mse grid. Sanity check passed: the layer-0/strong-reg cell has test_mse ≈ V exactly (ridge
→ predict train mean), giving R² ≈ 0.

**Baseline choice:** headline uses the TRAIN-target mean (ȳ, the constant ridge centers on → "did the
map beat predicting the mean train FV?"). Also report the sklearn test-mean-baseline variant.

**Numbers (7 test tasks, hidden=4096):**
- V(test|train-mean)=0.2171, V(test|test-mean)=0.1902, V(train|train-mean)=0.1978.
- test R² (train-mean base): min 0.000, median 0.346, **max 0.465** at icl10/last_prompt_token L11.
- test R² (test-mean base / sklearn): min −0.142, median 0.254, max 0.390; negative in 69/899 cells.
- train R²: median 0.978 (severe overfit, expected for 4096→4096 with 3400 rows).
- Best cells cluster at mid-layers (L10–13) at the pre-label and final-prompt-token positions.

**Files:** `src/eval_scripts/compute_fulldim_ridge_r2.py` (new; reuses the FV loader from
regress_activation_to_fv_fulldim_ridge.py and render_heatmap from merge_fulldim_ridge_results.py).
**Outputs:** `results/direction3_fv_formation/fulldim_ridge_activation_to_fv/combined_metrics_with_r2.csv`,
`combined_test_r2_heatmap.png`.

**Extension (2026-07-06): full-dim R² for the PCA ridge.** `pca_ridge_activation_to_fv` (k=16, i.e.
16 act-PCs → 16 FV-PCs) already reports test_mse in the SAME full 4096-d space vs the raw FV (its
layer-0 cell = 0.21712 = V exactly), same 7 test tasks → same denominator V=0.2171. So the identical
R² transform applies (reran compute_fulldim_ridge_r2.py with --input_dir). Result:
- PCA k=16: test R² median 0.325, **max 0.472** (icl10/finaltok L13); train R² median 0.877.
- vs full-dim 4096: test R² median 0.346, max 0.465; train R² median 0.978.
- ⇒ the 16-dim bottleneck TIES full-dim on test while overfitting far less. Extra dims only help train.
- Reconstruction ceiling: the top-16 train-FV-PC subspace captures 54.3% of test-FV variance (about the
  train mean); top-32/64 barely move it (55.2%). So held-out task FVs live ~45% in directions the 20
  training tasks never span — the deep generalization ceiling behind both the ~0.47 R² and the earlier
  20→7-task story. Outputs: pca_ridge_activation_to_fv/combined_metrics_with_r2.csv + heatmap.
- NOTE: the two `pca_ridge_..._varicl_*_top40` variants use a DIFFERENT FV target set (layer-0 MSE
  0.33 / 0.26 ⇒ different V) so are NOT directly comparable to the full-dim ridge; left untouched.

**Next / not done:** joint-PCA(-ICL) ridge/OLS variants report MSE in REDUCED PCA space (per-k), not
full-dim, so need reconstruction before a comparable R². Ask before extending.

---

## 2026-07-04 — Stream T: 2-shot FV-projection-ablation test of FVs as task-imitation machinery

**Owner:** Coordinator (tmux "fv-ablation"; CPU pod + own RTX 4000 Ada GPU pod, terminated).
**Status:** DONE (compute + figures on volume; commit pending user request).

**Question:** are Todd-et-al function vectors the machinery mean-difference steering rides on? On
matched-label 2-shot prompts, steer src→tgt by injecting the mean act-diff vector at a SINGLE
(token position, layer ℓ) — a SEPARATE layer-sweep (ℓ=0..28) per steer site — then ablate the
target-FV-specific direction `F'⊥F = F' − proj_F(F')` at qfinal (all 29 layers, fixed). Steer sites:
{label1, label2, qfinal}. Metric: `Δlogit = logit(a_tgt) − logit(a_src)` at qfinal. Per (direction, α,
site): curves `steer(ℓ)` and `steer+ablate(ℓ)` vs `clean`/`ablate` flat baselines. F, F' = task-specific
top-10 GPT-J FVs (artifacts/gptj_fv).

**Headline metric:** `retention(ℓ) = (steer+ablate − ablate) / (steer − clean)` at effective injection
layers (steer_gain > 0.5) — fraction of the localized steering effect that SURVIVES ablating the FV
direction. retention ≪ 1 ⇒ the FV direction mediates the steering.

**Files:** `src/eval_scripts/steer_twoshot_fv_ablation_logitgap.py` (compute, per-token per-layer sweep),
`plot_twoshot_fv_ablation_logitgap.py` (CPU plot). Results under
`results/direction2_label_geometry/twoshot_fv_ablation_imitation/{antonym_synonym,
next_number_digits_prev_number_digits}/` (`<dir>_<token>_alpha{a}_layersweep.csv`, `_perpair.npz`
[gitignored], summary.json) + `figures/layersweep_by_token_alpha{2,4,8}.png` (4 dir × 3 token grid) +
`retention_overview.png`. Digit FVs computed into `artifacts/gptj_fv/{next,prev}_number_digits`
(top-10, n_shots=10).

**Commands:** `compute_function_vectors.py --dataset_names next_number_digits prev_number_digits`;
`logs/run_fv_ablation_bytoken.sh` (both pairs, α 2/4/8, n=300, batch 300, --overwrite). GPT-J-6B on an
RTX 5090 (~11 min/pair); `model.transformer(...)` + lm_head-on-last-token readout.

**Findings (n=300; separate sweep per steer token):**
- Localized steering works at every site. Peak steer_gain and layer:
  qfinal peaks MID-NET (L10–14), gain +5.1…+7.9 (largest — it's the prediction site); label2 peaks
  EARLY-MID (L5–8), gain +3.0…+6.1; label1 earliest/weakest (L4–8), gain +1.0…+3.8. Steering a demo
  answer token DOES propagate to the query prediction.
- **Ablating F'⊥F at qfinal mediates the steering at ALL THREE token positions** (n=300, α=2 peak-layer
  retention): ant→syn label1 0.43 / label2 0.30 / qfinal 0.35; syn→ant 0.31 / 0.30 / 0.35; next→prev
  0.49 / 0.41 / 0.35; prev→next 0.45 / 0.44 / 0.42. So the ablation removes ~50–70 % of the localized
  steering gain everywhere — including gain that was INJECTED at the demo label tokens, then killed by
  removing one direction at the query token. Strong support for FVs being the shared write-channel.
- **α-dependence:** retention generally rises with α (steering brute-forces other directions as it gets
  stronger; for the small-gain label tokens on digits it → ~0.9 at α=8 where the gain is near-zero and
  the ratio is noisy). The gentle α=2 regime is the clean test and shows FVs carrying half-to-two-thirds
  of the effect at every site.
- cos(FV_src, FV_tgt) = 0.69 (ant/syn), 0.75 (digits) → F'⊥F is a partial slice of the target FV
  (‖F_perp‖ ≈ 28–36); even so it mediates a majority of the α=2 steering.

**Interpretation:** supports the claim that function vectors are (a large part of) the machinery of
task imitation — wherever localized steering installs the target task (demo label tokens or the query),
deleting the target-FV-specific direction at the prediction site removes most of the induced imitation
at natural steering magnitudes.

**Note on iterations:** first pass steered all layers at once (masked the effect, retention≈1); second
pass steered one layer at a time but all 3 tokens together; THIS (final) pass sweeps each token
separately. Earlier same-dir outputs overwritten to avoid confusion (user request).

**Next:** commit on request. Follow-ups: sweep the ablation layer too (2-D site×layer retention map);
project out the full target FV (not just its F-orthogonal part) to bound the effect.

**Blockers:** none.

### Addendum 2026-07-07 — second ablation direction: raw FV difference F'−F (variant `fdiff`)
Added `--ablate_variant {fperp,fdiff}` to the compute script; `fdiff` ablates `u = normalize(F'−F)`
instead of `fperp`'s `normalize(F'−proj_F F')`. Outputs now variant-tagged
(`<dir>_<token>_alpha{a}_<variant>_layersweep.csv`, `<tp>_<variant>_summary.json`); the original fperp
files were `git mv`'d to `_fperp` tags (no recompute). New plot overlays both ablation curves per panel
(`figures/layersweep_compare_alpha{2,4,8}.png`) + a variant retention heatmap (`retention_compare.png`).
Ran fdiff on RTX 5090 (batch 300, both pairs). **Sanity:** the `steer`/`clean` columns are byte-identical
between the fperp and fdiff CSVs (max |Δ|=0 over 36 file pairs) — steering & clean baseline are
ablation-independent, confirming the rename + new run line up.
**Finding:** ablating F'−F removes AS MUCH OR MORE of the localized steering gain than F'⊥F. At α=2 both
give peak-layer retention ~0.28–0.45 (similar); the gap widens with α — where strong steering lets F'⊥F
retention climb to 0.79–0.91 (ablation stops biting), F'−F stays 0.00–0.66. Makes sense: the
mean-difference steer vector aligns with the FV *difference*, and F'−F keeps the F-parallel component
that F'⊥F discards, so it is the more complete anti-steering direction. Both support FVs as the
task-imitation write-channel; F'−F is the stronger knockout.

**Blockers:** none.

---

## 2026-07-04 — Stream S: two-shot pair-diff variance explained by FV pre-image direction

**Owner:** Coordinator (tmux "pairdiff-preimage"; CPU pod — GPU stages on a fresh RunPod pod).
**Status:** DONE (compute + figures on volume; commit pending user request).

**Question:** is the Stream-K two-shot paired-prompt activation difference (per-pair
`d_i = act_f1 − act_f2` at each of the 5 token roles × 28 layers) primarily *FV-related*? At each
cell, invert the direction3 full-dim ridge map on the FV difference `fv_A − fv_B`
(pre-image via the fitted W, Tikhonov-damped + exact variants), unit-normalize → `u`;
report `explained = 1 − var(d − proj_u d)/var(d)` (centered headline, uncentered secondary) as
role × layer heatmaps, vs controls (raw `unit(fv_A − fv_B)` direction, top-PC 1-D upper bound,
random-direction floor). Filters: all pairs + both-judge-correct.

**Consistency rule (user gotcha):** regression target FVs and inverted FVs come from ONE root:
`artifacts/function_vectors/gpt-j/train_varicl_max4_top40` (1–4-shot varicl, varicl-max4 top-40
heads — shot regime matches the 2-shot prompts). Digit-variant FVs (next/prev_number_digits,
used by the two-shot capture) don't exist yet → built in stage 0 on that basis.

**Cell matching** (causal attention ⇒ demo-k tokens of the 10-shot ridge captures see a k-shot
context): demo1_prelabel↔(pre_label_token,icl1), demo1_label↔(last_label_token,icl1),
demo2_prelabel↔(pre_label_token,icl2), demo2_label↔(last_label_token,icl2),
query_final↔(pre_label_token,icl3) context-matched + (last_prompt_token,icl10) secondary view.
Two-shot activations have NO embedding slice: two-shot layer j ↔ capture layer j+1 ↔ bank
edit_layer j.

**Plan:** [0] digit FVs (varicl max-4 capture → isolated root
`artifacts/multitask_aie_heads_varicl_digits_max4`, copy into `…_varicl_max4/<task>/`, FV build
via `compute_all_task_fvs_varicl.py` + new `task_splits/paired_tasks_digits.json`);
[1] NEW `src/eval_scripts/fit_ridge_preimages_multicell.py` (generalizes Stream R's
`fit_prelabel_ridge_preimages.py` to 6 (role,icl) cells × 28 layers; pair-diff pre-images;
validation vs study shard_icl2) → `artifacts/preimage_pairdiff/train_varicl_max4_top40/`;
[2] NEW `src/eval_scripts/analyze_twoshot_pairdiff_fv_preimage.py` (CPU) →
`results/direction3_fv_formation/twoshot_pairdiff_fv_preimage/train_varicl_max4_top40/`.
Driver: `logs/stream_s_pairdiff_preimage/driver.sh`.

**Run:** RunPod pod `mgb6durkq6k0os` ("claude-pairdiff-preimage", RTX PRO 4500 Blackwell,
$0.74/hr) — TERMINATED after ~1.3 h. Driver + per-stage logs in `logs/stream_s_pairdiff_preimage/`.
Smokes before full run: analysis loader reproduced the Stream-K meancos grid (max diff 5.1e-5
antonym / 1.2e-6 digits); fit of the (pre_label, icl2) cell matched saved study metrics to
≤2.9e-7 across all 28 layers.

**Files:** NEW `src/eval_scripts/fit_ridge_preimages_multicell.py`,
`src/eval_scripts/analyze_twoshot_pairdiff_fv_preimage.py`, `task_splits/paired_tasks_digits.json`,
driver. Intermediates (gitignored): digit FV capture `artifacts/multitask_aie_heads_varicl_digits_max4/`
(acts copied into `…_varicl_max4/<task>/`, FVs into `function_vectors/gpt-j/train_varicl_max4_top40/`,
norms 52.1/53.0, manifest `fv_manifest_digits.json`); maps + pre-image banks
`artifacts/preimage_pairdiff/train_varicl_max4_top40/<role>_icl{k}/` (6 cells × 28 layers, ~5.4 GB —
reusable, see DECISIONS). Deliverables (TRACKED):
`results/direction3_fv_formation/twoshot_pairdiff_fv_preimage/train_varicl_max4_top40/<pair>/`
(explained_grid.json + 16 heatmaps + 4 line plots per pair).
2026-07-06: line plots re-drawn WITHOUT the damped arm per user request — the red line
("exact", relabeled **inv(fv_diff)** in the legend) is the plain W⁻¹(fv_A−fv_B) pre-image the
experiment specified. NEW `src/eval_scripts/plot_twoshot_pairdiff_lines.py` replots from
explained_grid.json on CPU (`--directions` chooses lines; damped numbers remain in the
JSON + heatmaps).

**Findings** (explained = fraction of pair-diff variance along the unit pre-image of fv_A−fv_B;
random floor ≈ 2.4e-4; the both-correct filter barely changes any number):
- **Digits pair, uncentered (total energy):** strongly FV-pre-image aligned at predictive
  tokens — query_final L4 **0.43** (top-1-direction bound 0.84 → ~52% of the achievable),
  query_final@lastprompt10 L6 0.39, demo2_label L10 0.27, demo1_label L4 0.18. The damped
  pre-image beats the raw FV-diff direction unit(fv_A−fv_B) by ~10× at query/label views
  (0.43 vs 0.03) and is aligned already at L4–7 where fv_diff only rises from L8 — the ridge
  inversion adds real signal. Exception: demo2_prelabel mid-layers, where raw fv_diff
  (0.34 @L12) beats the pre-image (0.17).
- **Digits, centered (fluctuation around the mean diff):** small — best 0.034 (query_final L4)
  vs centered top-PC bound 0.11: the pre-image direction captures the SYSTEMATIC mean
  difference, not per-pair fluctuation.
- **Antonym/synonym:** much weaker — uncentered damped ≤0.05 (query_final views ~L10; raw
  fv_diff comparable or better there, 0.09–0.10); centered ≤~0.004 at mid layers (~15× floor).
  Late-layer L27 damped spikes (0.02–0.04) look like an end-of-stack map artifact; treat with
  caution.
- **Exact (undamped) pre-image direction ≈ noise everywhere** (mostly ≤4× the random floor,
  ~100× below damped at hot cells) — Stream R's causal finding reproduced geometrically; the
  damped-vs-exact choice matters even for pure directions.
- Headline answer: under the CENTERED definition, no — the FV-pre-image direction explains only
  1–4% of pair-diff variance (though 10–100× chance). Under the UNCENTERED definition the
  digits pair's activation difference IS substantially FV-related at label/query tokens (up to
  ~half of what any single direction could explain); antonym/synonym is not (~3–5%).

**2026-07-07 amendment — anisotropic random baseline (`random_actcov`).** User observation: the
residual stream is highly anisotropic, so the isotropic random floor (~1/4096) understates
chance; a fairer null samples from the activations' covariance. Added per (view, layer, filter):
64 draws `v = unit(Xcᵀ g)`, `g~N(0,I)` — exact samples from the empirical covariance Σ̂ of the
RAW (undiffed) two-shot activations of both functions at that role/layer (rank ≤ 2n−1; separate
RNG stream seed+1 so old numbers reproduce, max |old−new| ≤ 3e-7 = cross-pod BLAS noise).
Modified `analyze_twoshot_pairdiff_fv_preimage.py` (loader now also returns raw acts;
`random_actcov`+`_sd` in explained_grid.json) and `plot_twoshot_pairdiff_lines.py` (orange
dash-dot "random (act-cov)" in default lines; isotropic relabeled "random (isotropic)"). Rerun
entirely on the CPU pod (CPU torch 2.12.1 installed for python3.12); sanity gates re-passed;
log `logs/stream_s_pairdiff_preimage/actcov_baseline_rerun.log`.
Findings: the act-cov null sits **~10–100× above the isotropic floor** (median ~0.4–1.3%
centered, up to 0.15 uncentered at digits query/label views) and always ≤ top_pc. This
**reorders the centered story**: fv_diff's centered medians (0.001–0.004) are mostly BELOW the
act-cov null — i.e. beating the isotropic floor there was largely an anisotropy artifact; only
narrow peaks (antonym query_final L8 0.029 vs null 0.009; digits demo2_prelabel L12) clear it.
The UNCENTERED digit-pair signal survives cleanly: fv_diff 0.34–0.43 at demo2_prelabel/
query_final L8–14 vs null 0.12–0.15 (~3× above a covariance-matched random direction); digits
demo1/demo2_label uncentered fv_diff (~0.01–0.02) drops BELOW the null (~0.07–0.13). Antonym
uncentered query_final peak (0.13 @L9 vs null 0.045) clears it by ~3×; elsewhere comparable to
null. inv(fv_diff)=exact stays at/below the isotropic floor throughout (unchanged).

**2026-07-07 metric change — SIGNED mean cosine (cos_grid.json + cos_lines figures).** User
decided explained-variance is the wrong metric; new headline metric per (view, layer, filter):
`mean_i cos(d_i, x)` for x ∈ {exact pre-image of fv_A−fv_B ("inv_fv_diff"), unit(fv_A−fv_B)
("fv_diff")}, plus `mean_dir = ||mean_i unit(d_i)||` (the analytic MAX of the metric over unit
x; via the resultant identity mean_dir² reproduces the Stream K mean-pairwise-cos grid, checked
to ≤5.1e-5) and the same isotropic + activation-covariance random baselines (signed expectation
~0; the anisotropy appears as band WIDTH — the act-cov sd is ~10–50× the isotropic sd because
covariance-matched draws often align with the dominant direction the diffs concentrate around).
Same RNG draws as the variance metric (explained_grid.json reproduced bit-identically, checked).
Defaults chosen while user was away: SIGNED cos (not |cos|), EXACT pre-image only (matches the
figures' inv(fv_diff) line). Modified analyze script (mean-cos pass + cos_grid.json); NEW
`src/eval_scripts/plot_twoshot_pairdiff_cos_lines.py` → `cos_lines_{all,both_correct}.png` per
pair (linear y, zero line, ±2sd baseline bands). Log:
`logs/stream_s_pairdiff_preimage/cos_metric_rerun.log`.
Findings (filter=all): **inv(fv_diff) ≈ 0 everywhere** (|mean cos| ≤ 0.05) — the exact pre-image
direction has no consistent orientation w.r.t. the diffs, matching its noise character.
**fv_diff is systematically positive** (correct sign at essentially all views/layers past ~L4)
and peaks mid-stack: digits query_final **+0.66 @L12**, demo2_prelabel +0.58 @L12 (~75% of the
mean_dir bound 0.88/0.77); antonym query_final +0.34 @L9, demo2_label/prelabel +0.13 @L12.
Relative to the act-cov band, however, significance is modest: peaks sit at ~1.7–2.2 baseline
sd (band edge = 2 sd), and the digit label tokens (+0.14 vs 2sd≈0.69) are deep inside the band.
The sign consistency itself is informative even where the magnitude is within-band.

**2026-07-07 diagnosis — why the exact pre-image scores ~0 (preimage_diagnostics/).** NEW
`src/eval_scripts/diagnose_pairdiff_preimage_spectrum.py` (cell pre_label_token_icl3 /
query_final, layers 4/8/12/20; SVD of fp16-saved W in fp64; figures + diagnostics.json under
`…/train_varicl_max4_top40/preimage_diagnostics/`). Mechanism, in three measurements:
1. **The FV-informative part of W is only rank ~16–20** — as it must be: the ridge was trained
   on 20 train-task FVs, so W's column space has signal rank ≤ 20. A TRUNCATED inverse keeping
   only the top k=8–16 singular directions is strongly diff-aligned — digits mean cos **0.73 /
   0.66 / 0.62** at L4/8/12 (BEATING the damped pre-image 0.66/0.52/0.44 and raw fv_diff, which
   is ~0.2 at L4); antonym 0.21 @L12 vs damped 0.12. Alignment collapses to ~0 by k=32 and
   never recovers.
2. **The exact inverse buries that 16-dim signal under conditioning noise**: cond(W) =
   0.5–1.3e9, |dz_exact|/|dz_damped| = 2.5e7–1e8, cos(exact, damped) ≈ 0.00–0.03; the exact
   vector's energy is spread ~uniformly over the spectrum (bottom-decile fraction ≈ 0.10 =
   the uniform value), i.e. white-noise-like.
3. **The exact direction is not even well-defined**: recomputing it from the fp16-rounded W
   (~1e-3 relative perturbation) gives cos ≈ 0.001–0.06 with the bank's fp32-derived exact —
   the direction is an artifact of float noise in the smallest singular values (initially
   tripped the script's sanity gate; converted into the recorded instability probe).
Conclusion: "inv(fv_diff) ≈ 0" says nothing about FVs — it's pure numerical conditioning. The
honest inverse is the rank-≲20-truncated one, and it aligns BETTER with the pair diffs than
either the damped pre-image or the raw fv_diff direction at early/mid layers. Candidate
follow-up: replace/augment the cos_lines inv(fv_diff) line with the rank-16 truncated inverse
(cheap: banks + maps already on disk).

**2026-07-07 PCA-k16 ridge inverse (user request: does inverting the k=16 PCA ridge help?).**
NEW `src/eval_scripts/fit_pca_ridge_preimages_multicell.py`: refits the direction3 PCA ridge
(act-PCA k=16 per cell/layer on pooled 20-train rows, FV-PCA k=16 on the 20 train FVs, 16→16
standardized ridge, LOO-task CV, logspace(−2,6,17)) at the 6 Stream-S cells with fv_root
**train_varicl_max4_top40** (consistency rule — the committed pca_ridge study used
train_selected, so it was refit, not reused). Pre-image: t = fv_diff@fv_compᵀ, dz = solve(Aᵀ,t)
(A = fitted 16×16 map, cond(A) ~ 1e2 — well-posed, no damping needed), dx = (dz⊙std)@act_comp.
Banks: `artifacts/preimage_pairdiff_pcak16/train_varicl_max4_top40/<cell>/`. Analyze script +
cos plotter extended: purple "inv(fv_diff) PCA-k16" line in cos_lines figures
(`inv_fv_diff_pcak16` in cos_grid.json); explained_grid untouched (bit-identical rerun).
**Result: NO — it does not improve on fv_diff, and badly trails the TSVD-k16 inverse of the
full-dim map.** Digits query_final: PCA-k16 +0.01/+0.24/+0.10 at L4/8/12 vs TSVD-k16
+0.73/+0.66/+0.62 and fv_diff +0.18/+0.57/+0.66; antonym similar story (peak +0.08 vs fv_diff
+0.34). Partial exceptions: query_final@lastprompt10 reaches +0.46–0.49 at L9–12 (approaching
fv_diff; matches the PCA study's best cells being icl10) and digits demo2_prelabel L27 +0.38.
Why it fails where TSVD succeeds — two bottlenecks the full-dim ridge doesn't have:
(1) TARGET side: the 16 train-FV PCs cover only 24% (antonym−synonym) / 38% (digits) of the
held-out fv_diff energy, so the inversion is aimed at a fraction of the target;
(2) FEATURE side: top-16 activation-VARIANCE PCs ≠ the predictive directions. The full-dim
ridge standardizes all 4096 dims and its top singular directions are regression-chosen; at
early layers (digits L4: TSVD +0.73 vs PCA-k16 +0.01) the task-identity signal evidently lives
in low-variance activation directions that the PCA feature bottleneck discards before the
regression ever sees them.

**2026-07-07 TSVD-k16 line added to the cos figures (user request).** NEW
`src/eval_scripts/fit_tsvd_preimages_multicell.py`: rank-16 truncated-SVD pre-image of the
stage-1 full-dim maps at all 6 cells × 28 layers via torch.svd_lowrank (randomized top-k;
validated against the diagnostics' exact fp64 SVDs at query_final L4/8/12/20 to ≤3.9e-7).
Banks: `artifacts/preimage_pairdiff_tsvdk16/train_varicl_max4_top40/`. Analyze + cos plotter
extended: blue "inv(fv_diff) TSVD-k16" (`inv_fv_diff_tsvdk16` in cos_grid.json). All
reproduction checks unchanged.
**Result: for the digits pair the TSVD-k16 pre-image is the best direction in the whole study.**
Peaks +0.73/+0.76 at L4 (query_final / @lastprompt10; mean_dir bound 0.92, act-cov 2sd 0.62 —
clears the anisotropic band), and — unlike every other direction — it is sustained at
0.4–0.66 across ALL layers at the label tokens, where raw fv_diff is ~0 or negative
(demo1_label L5: TSVD +0.67 vs fv_diff −0.04; demo2_label L6: +0.66 vs −0.03). It also rises
at L3–4, four layers before fv_diff wakes up at L8. Antonym: TSVD-k16 +0.21/+0.25 at L10–12,
slightly below fv_diff's +0.29–0.34 peaks and inside the act-cov band — the weak-pair story is
unchanged. Reading: for a pair the FV system separates well, the ridge's rank-≲20 core reads
the task-identity signal from LOW-VARIANCE activation directions present from L4 at every
label/query token; the raw fv_diff direction only works late and at high-variance tokens.
(demo1_prelabel stays ~0 for every direction — mean_dir ~0.02 says no consistent pair-diff
direction exists there at all.)

**Next:** commit + push on user request. Candidate follow-ups: k-dim pre-image subspace instead
of the 1-D direction; same analysis on the one-shot paired captures; per-layer gamma sweep of
the damped direction.
**Blockers:** none.

---

## 2026-07-03 — Stream Q: TEN-shot intervene-token STRIP cosine-shift heatmaps (read fixed at qfinal)

**Owner:** Coordinator (CPU editing pod; GPU compute on a fresh RTX 4090 pod). **Status:** DONE —
360 grids + scalar overview + 3 strip figures.

**What:** Extends Stream P to 10-shot ICL with a DIFFERENT pairing: **no matched labels** — n=300
ten-shot prompts/task, pairs share ONLY the query (⇒ byte-identical final "A:"); the 10 demos are
independently random per function. steer_vec at each of the **30 demo tokens** (input/pre-label/label
× 10 demos) = `mean_pairs[tgt(t,ℓ)−src(t,ℓ)]` = difference of the two tasks' MEAN activations at that
slot (unmatched ⇒ mixes lexical+function). Inject α·steer_vec at the source prompt; **read only at
qfinal** (query predictive token) across all layers → the token×token matrix collapses to a vertical
STRIP over the 30 intervene tokens. α∈{2,4,8}; 4 combos (2 task pairs × 2 directions).

**Files:** NEW `src/eval_scripts/steer_tenshot_strip_cos_heatmap.py` (port of the Stream-P compute; 10-shot
random-demo builder; 30 intervene tokens, qfinal read; memory-lean; **resumable** skip-if-exists;
`model.transformer(...)` forward to SKIP the unused lm_head — see DECISIONS, this was the OOM fix). NEW
`src/eval_scripts/plot_tenshot_strip_heatmap.py` (CPU-only; global vmax=0.0875): `scalar_overview.png`
(30 tokens × 12 combo·α, peak Δcos, annotated — headline comparable view) + 3 `strip_alpha{a}.png`
(30 tokens × 4 combos of layer heatmaps). Output (TRACKED)
`results/direction2_label_geometry/tenshot_strip_intervention_cos_heatmap/<task_pair>/`: 360 grids
(.npy gitignored / .csv tracked), 2 summaries. Logs `logs/tenshot_strip_full.log`, runner
`logs/run_tenshot_strip.sh`.

**Commands** (GPU pod, volume at /runpod; batch 48 fits the 24 GB 4090 at ~21.7 GB):
`python src/eval_scripts/steer_tenshot_strip_cos_heatmap.py --task_pair {…} --alphas 2 4 8 --n_pairs 300
--batch_size 48` then `python src/eval_scripts/plot_tenshot_strip_heatmap.py`. ~6 h total on one 4090
(the 29-layer retain_output per forward dominates; batch 64 pins mem at the 24 GB ceiling, use 48).

**FINDINGS:**
- **Label tokens carry ~all the steerable signal.** Every top intervene token (all 4 combos) is a demo
  LABEL token (`d{i}_lab`); the input (`_in`) and pre-label (`_pre`) rows are ≈0 in the scalar overview.
- **Mid-layer → late read:** peaks at intervene L5–8 → read L16–26 (words k≈26, digits k≈16–18), same
  band as the 1-/2-shot studies.
- **Directional asymmetry:** `synonym→antonym` peaks **+0.088** but `antonym→synonym` only **+0.019**;
  digits ≈0.05 both ways.
- **Distributed over positions:** unlike 2-shot (where label2 alone dominated), MANY demo-label
  positions contribute (`d2_lab`,`d10_lab`,`d3_lab`,… all near the top) — no single demo owns it.
- α=2 ≳ α=4 ≳ α=8 for digits (cosine saturates); words strengthen a bit up to α=8.

**Verification:** smoke (digits & antonym, n=16, 3 layers) passed asserts; lower-tri≡0 on every grid;
360/360 grids finite; sanity `d10_lab`→qfinal reproduced the label→qfinal band. GPU pod
(`gmfxnpi460x4rn`, RTX 4090) **terminated** after run. **Blockers:** None.

---

## 2026-07-02 — Stream R: causal test of ridge pre-images (per-layer W⁻¹(fv) steering)

**Owner:** Coordinator (tmux "preimage-steer"; GPU on RunPod `ptlrql6hjz6c70` "claude-preimage-steer",
RTX PRO 6000 Blackwell 96GB, $1.89/hr — TERMINATED, ~1.5h total). **Status:** DONE.
**Plan:** `/root/.claude/plans/compiled-singing-falcon.md`.

**Question:** Are the full-dim activation→FV ridge maps causally meaningful? Invert the icl10
pre_label_token regression per layer (target = train_selected_top40 FVs), inject the linear
pre-image Δx_ℓ (σ⊙solve(W_std, fv)) at edit layer ℓ−1 (block-output hook, last token), compare
task top-1 layer sweeps vs direct FV steering. Tasks: next_number, prev_number, synonym, antonym
(all outside the 20 regression-train tasks). Regimes: zero-shot + 10-shot-shuffled. Arms:
preimage_raw / preimage_normmatched / fv_direct (+ no-intervention baselines).

**Files (NEW):** `src/eval_scripts/fit_prelabel_ridge_preimages.py` (refit w/ saved W, fp64 lstsq
pre-images, study-reproduction validation; handles stale absolute shard paths in the capture
index.json), `src/eval_scripts/evaluate_preimage_steering.py` (3-arm layer sweep, cached
ICL-correct filters, --max_eval_examples cap, exact-match best-layer pass),
`logs/stream_r_preimage_steering/driver.sh` (stage 1 then 4 task-parallel evals on one 96GB GPU).
Outputs: `artifacts/preimage_steering/train_selected_top40_icl10_pre/` (maps + banks),
`results/direction3_fv_formation/preimage_steering/<task>/` + `AGGREGATE_preimage_vs_fv_steering.png`.

**Method notes / gotchas hit:**
- Refit reproduces the study's saved shard_icl10 metrics to ≤8e-7 at every layer (validation flag).
- W is severely ill-conditioned (cond 1e9–1e11): the EXACT pre-image W⁻¹(fv) has norm 1e8–1e10
  (~1e7× activation scale). Added a third arm `preimage_damped`: Tikhonov pre-image, per (task,
  layer) the best-residual γ with standardized-space norm ≤ 2√D. Damped rel-residual plateaus at
  ~0.72–0.82 — i.e. only ~20–28% of the FV's norm is reachable from activation-scale
  displacements; damped norms are FV-scale (~30–70) mid-layers, inflating late (numbers ≫ syn/ant).
- Upstream bug found: `n_shot_eval(generate_str=True)` crashes (get_answer_id gets a list target);
  worked around with a documented monkeypatch inside evaluate_preimage_steering.py.
- Number-word tasks: first-token top-1 is heavily inflated by the copy/compound-number artifact
  (zs no-FV baseline already 0.55–0.62 first-token; exact-match at the same layers ~0.1–0.2).
  Trust the EM column for numbers; synonym/antonym EM ≈ first-token (single-token answers).

**FINDINGS — pre-images are PARTIALLY causal; exact inverse is not, damped inverse is.**
Best-layer intervention top-1 (first-token; EM = exact-match generation at that layer):

| task (n) | regime | no-FV | fv_direct | preimage_damped | normmatched | raw |
|---|---|---|---|---|---|---|
| antonym (200) | zs | .010 | **.685** L11 | .095 L11 | .050 | .000 |
| antonym | fs_shuf | .515 | **.900** L11 | **.680** L11 | .505 | .000 |
| synonym (148) | zs | .000 | **.223** L11 | .014 | .014 | .000 |
| synonym | fs_shuf | .095 | **.480** L10 | .189 L8 | .162 | .000 |
| next_number (42) | zs EM | — | .119 L5 | **.143** L9 | .000 | .000 |
| prev_number (42) | zs EM | — | .214 L8 | **.214** L11 | .095 | .000 |

- **Exact pre-image (raw): completely non-causal** — 0.000 at every layer/task (norm ~1e9 destroys
  the forward pass). Norm-matching its direction recovers little (syn/ant ~0.05): the exact
  inverse's direction is regression noise in near-null singular directions.
- **Damped pre-image: substantially causal, layer-aligned with the FV.** On antonym it peaks at
  the SAME layer as FV steering (L11) and recovers 76% of the FV's fs-shuffled effect (.680 vs
  .900, baseline .515); zero-shot it recovers only ~14%. On the number tasks (EM) it EQUALS
  fv_direct (.143/.214 vs .119/.214). Synonym weakest (fs .189 vs .480).
- Interpretation: the ~20-25% FV component reachable at activation scale through the regression's
  well-conditioned subspace carries most of the steerable task signal in-context, but the FV's
  zero-shot punch (esp. syn/ant) lives in components the pre-label-token regression cannot
  produce at natural norms. Consistent with the direction3 story: the map is real but low-rank-ish;
  its usable inverse is the damped one.

**Next (optional):** γ-sweep steering curve (dose-response); repeat with last_prompt_token maps
(position-matched to the injection site); project-out-FV control for the damped arm.
**Blockers:** none.

---

## 2026-07-02 — Stream Q: next/prev_number FVs on the top-40 head bases (fill the gap)

**Owner:** Coordinator (tmux "fv-paired-top40"; CPU pod — GPU compute on a fresh RunPod pod).
**Status:** DONE — 6 new FVs (2 tasks × 3 bases), SANITY PASS on all 12 (3 bases × 4 tasks), pod terminated.

**Goal:** next_number + prev_number FVs on the same top-40 head bases as the ridge studies, so a
combined experiment can use all four of {next_number, prev_number, synonym, antonym} on any basis.
`train_selected` (top-40) already has all four; the gaps are the three top-40 variants:

| basis | heads | acts needed | acts status |
|---|---|---|---|
| train_selected_top40 (= gptj_fv_multitask_top40 symlinks) | `multitask_aie_heads/multitask_top_aie_heads.pt` @40 | fixed-shot `gptj_fv/<task>/` | EXIST |
| train_varicl_top40 | `multitask_aie_heads_varicl/…` @40 | varicl max-10 | EXIST (`multitask_aie_heads_varicl/{next,prev}_number/`) |
| train_varicl_max4_top40 | `multitask_aie_heads_varicl_max4/…` @40 | varicl max-4 | **MISSING — capture needed** |

**Plan (GPU pod):** (1) capture max-4 varicl mean acts: `compute_multitask_varicl_heads.py --tasks
next_number prev_number --min_shots 1 --max_shots 4 --query_split valid --demo_split train
--filter_to_correct_icl --save_path_root artifacts/multitask_aie_heads_varicl_paired_max4` (isolated
root; copy acts into `multitask_aie_heads_varicl_max4/<task>/` after). (2) FV builds:
`compute_all_task_fvs_varicl.py --tasks next_number prev_number --task_manifest
task_splits/paired_tasks_3.json --n_top_heads 40 --manifest_name fv_manifest_paired.json` twice
(heads_path/fv_root/output_root per varicl basis); `compute_all_task_fvs_from_multitask_heads.py`
equivalent for the fixed-shot top-40 basis; then train_selected_top40 symlinks. Mirrors the
2026-06-14 paired-task build (see that entry for the 18-query valid-split caveat; next 18/18,
prev 15/18 ICL-correct there).

**Run:** RunPod GPU pod `s2hti5fmbs15wm` ("claude-fv-paired-top40", RTX PRO 4500 Blackwell,
$0.74/hr, shared volume) — TERMINATED after completion (~15 min total). Driver + logs:
`logs/stream_q_fv_paired_top40/` (driver.sh, driver.log, per-stage logs). Stage order: [1] varicl
max-10 top-40 FV build, [2] fixed-shot top-40 FV build, [3] max-4 varicl capture (long pole),
[4] copy acts + max-4 FV build, [5] train_selected_top40 symlinks + selected_heads.json,
[6] sanity-load all 12 FVs (3 bases × 4 tasks).

**Outputs (all under git-ignored `artifacts/`):**
- FVs: `function_vectors/gpt-j/train_varicl_top40/{next_number,prev_number}/`,
  `function_vectors/gpt-j/train_varicl_max4_top40/{next_number,prev_number}/`,
  `gptj_fv_multitask_top40/{next_number,prev_number}/` + symlinks in
  `function_vectors/gpt-j/train_selected_top40/` (+ refreshed `selected_heads.json`, 31 tasks).
  Manifests: `fv_manifest_paired.json` in each output root (kept distinct from `fv_manifest.json`).
- New max-4 varicl capture: `multitask_aie_heads_varicl_paired_max4/{next_number,prev_number}/`
  (isolated root, incl. CIE + per-prompt effects); mean acts copied into
  `multitask_aie_heads_varicl_max4/<task>/` for the builder.

**Findings:**
- Max-4 ICL-correct filter on the 18-query valid split: next_number 17/18, prev_number 15/18
  (max-10 reference: 18/18, 15/18) — capping demos at 4 costs number tasks ~nothing.
- All 12 FVs across the 3 top-40 bases sane: shape (4096,), finite, norms 53.6–60.5 (the
  established top-40 norm band). next/prev_number norms ≈ synonym/antonym norms on every basis.
- With `train_selected` (top-40) already complete, all four of {next_number, prev_number,
  synonym, antonym} now have FVs on ALL top-40 bases used by the ridge studies.

**Next:** the combined experiment (user to specify). **Blockers:** none.

---

## 2026-06-30 — Stream P: TWO-shot token-pair × layer×layer cosine-shift heatmaps (15 pairs)

**Owner:** Coordinator (CPU editing pod; GPU compute on a fresh RTX 4000 Ada pod). **Status:** DONE —
120 grids + 15 combined figures + 120 individual panels.

**What:** Generalises Stream L's 1-shot {label→query-final} 29×29 heatmap to **every ordered pair of
tokens** in a 2-shot paired ICL prompt, in BOTH src→tgt directions. 6 search-space tokens (sequence
order): `t1 label1`, `t2 input2` (demo-2 last input — the ONE token that differs across functions),
`t3 prelabel2`, `t4 label2`, `t5 qinput`, `t6 qfinal`. All ordered pairs = C(6,2)=**15**; intervention
sources are t1–t5 (qfinal is read-only). For direction src→tgt: `steer_vec(t,ℓ)=mean_pairs[tgt(t,ℓ)−
src(t,ℓ)]`; inject `α·steer_vec(t_i,i)` at t_i in the SOURCE prompt at layer i, read Δcos toward the
unsteered target at every later token t_j and read layer k. One 29×29 grid (x=intervene layer,
y=read layer) per (direction, t_i→t_j, α). **4 combinations** = 2 tasks × 2 directions
(antonym↔synonym, prev↔next digits), **α∈{2,4}**.

**Construction:** Stream-K matched-label paired 2-shot (shared L1,L2 distinct + shared query q; only
the 2 demo INPUTS differ). 5 of 6 tokens byte-identical across f1/f2 (asserted); t2 (input2) differs —
its steer dir mixes lexical+function and its read baseline cos<1 (kept in, flagged, NOT hidden).
n_pairs: ant/syn 544, digits 198.

**Files:** NEW `src/eval_scripts/steer_twoshot_tokenpair_cos_heatmap.py` (ports Stream-K pair build +
Stream-L baukit TraceDict edit-hook/29-entry residual convention; inlines baukit-free
`selected_token_records`/`token_positions`; **memory-lean** — keeps acts fp16, chunked steer/baseline,
per-slice float — fits the 20 GB card at ~18.5 GB). NEW
`src/eval_scripts/plot_twoshot_tokenpair_heatmap_grid.py` (pure plotting; 15 combined 8-panel figs =
4 combos rows × 2 α cols, shared per-figure scale, input-2 caveat in suptitle; + 120 single panels).
**UPDATE 2026-06-30:** per-figure scales aren't comparable across token-pairs (~100× dynamic range), so
added comparable views on ONE global scale (vmax=0.0856 across all 120 grids): **8 `matrix__<dir>_alpha{a}.png`**
— a 5×5 token×token grid (rows=intervene token, cols=read token, triangular) where each cell is that pair's
29×29 layer heatmap, single colorbar — the canonical "every token-pair" view; plus **`scalar_overview.png`**
— 8 small 5×5 matrices (4 combos × 2 α) of per-pair PEAK Δcos, annotated, one shared `Reds` scale (instant
"who drives whom"). Plotting is CPU-only: `pip install numpy matplotlib` on the editing pod and run there —
no GPU needed for figure iteration.
Output (TRACKED) `results/direction2_label_geometry/twoshot_tokenpair_intervention_cos_heatmap/`:
`<task_pair>/<dir>__<ti>_to_<tj>_alpha{2,4}_grid.{npy,csv}` (120 grids), 2 `<task_pair>_summary.json`,
`figures/<ti>_to_<tj>_combined.png` (15). Logs `logs/twoshot_tokenpair_full.log`.

**Commands** (on GPU pod, volume at `/runpod`; HF_HOME=/runpod/.cache/huggingface HF_HUB_OFFLINE=1):
`python src/eval_scripts/steer_twoshot_tokenpair_cos_heatmap.py --task_pair {antonym_synonym|next_number_digits_prev_number_digits} --alphas 2 4 --batch_size 64`
then `python src/eval_scripts/plot_twoshot_tokenpair_heatmap_grid.py`. ~50 min total (both tasks) on
RTX 4000 Ada. Smoke: add `--max_pairs 16 --layers 0 6 11 --batch_size 16`.

**RESULTS — top pairs by peak Δcos (intervene L / read L):**

| task | direction | top pair | α | peak Δcos | i/k |
|---|---|---|---|---|---|
| ant/syn | ant→syn | label2→qfinal | 4 | +0.054 | 7/26 |
| ant/syn | syn→ant | label2→qfinal | 2 | +0.058 | 8/26 |
| digits | next→prev | label2→qfinal | 2 | +0.078 | 4/18 |
| digits | prev→next | label2→qfinal | 2 | **+0.086** | 5/18 |

**FINDINGS:**
- **`label2→qfinal` dominates every combination** (0.048–0.086) — the 2-shot analog of Stream L's
  label→query-final; same mid-layer causal band (intervene ~L5–12 → late reads) and **digits≫words**.
- **`label1→qfinal`** is the clear runner-up (demo-1's label still reaches the prediction site,
  weaker/longer-range), and **`label1→prelabel2`** is a real cross-demo effect (demo-1 label feeds
  demo-2's pre-label region; strong for digits, 0.067).
- **`input2→*` is near-zero** (vmax 4e-4…4e-3): intervening at the demo INPUT token barely moves
  anything downstream — the steerable function signal lives in the LABEL tokens, not the inputs.
- Read-layer peaks land **very late (k≈26)** for words vs k≈15–18 for digits.
- Structural invariants held exactly: lower-triangle (k≤i) ≡ 0 everywhere; embedding column ≡ 0 for the
  5 clean source tokens (input2 the documented exception). α=2 ≳ α=4 at the peak (cosine saturates).

**Verification:** smoke (digits, 16 pairs, 3 layers) passed all asserts; label2→qfinal reproduces the
Stream-L band; 120 grids finite; 15/15 combined figures rendered 8/8 panels. GPU pod
(`8en0fiofwwxcbq`, RTX 4000 Ada, sm_89) **terminated** after run. **Next (optional):** GPU pod was the
upgraded cu128/torch-2.8 image (works on Ada AND Blackwell); could port to Qwen3-8B. **Blockers:** None.

---

## 2026-06-29 — Stream O: attention KNOCKOUT — does qfin read task info directly from demo-2 pre-label?

**Owner:** Coordinator (tmux "qfinal-attn-knockout"). **Status:** DONE — 4 tasks, verified.

**What:** Tests whether the **query-final token** (`qfin`, last `A:`) derives the task from the **demo-2
pre-label token** (the `A:` before L2) or directly from the **label tokens**. Knock out a single attention
edge — `qfin`'s query attending to a chosen key — at **every layer & head**, by setting the pre-softmax
score to `finfo.min` so after softmax that key's weight is 0 and the row renormalizes to sum 1. Only
`qfin`'s query row is edited; all other tokens attend normally.

**Conditions:** `clean`; `ko_demo2_prelabel` (cut qfin→demo2 pre-label — **test**); `ko_both_labels`
(cut qfin→{demo1,demo2 label} — **+control**, the "reads from labels" alternative); `ko_demo2_qcolon`
(cut qfin→demo-2 "Q:" colon — **−control**, a structural token). Metric: first-token top-1 accuracy +
mean gold-token logit at qfin, clean vs each knockout. All prompts per task (matched-label 2-shot;
single-token labels/inputs; gold = task answer). Tasks: antonym, synonym, next_/prev_number_digits.

**Mechanic:** monkeypatch `GPTJAttention._attn` (faithful transformers-4.49.0 copy + per-row knockout
reading `attn._ko`); GPT-J is eager by default so pre-softmax scores are editable; no baukit. Verified
on a live batch (`output_attentions`): knocked-out key weight = 0.0, edited rows sum to 1 (dev 2.5e-4).

**Files:** NEW `src/eval_scripts/ablate_qfinal_attention.py`, `src/eval_scripts/plot_qfinal_attn_knockout.py`.
**Command:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 python src/eval_scripts/ablate_qfinal_attention.py --batch_size 64`.

**RESULTS (Δ vs clean; top-1 / mean gold logit):**

| task (n) | clean top1 / glogit | cut demo2 pre-label (test) | cut both labels (+ctrl) | cut demo2 'Q:' (−ctrl) |
|---|---|---|---|---|
| antonym (1491) | 0.132 / 11.53 | **+0.011 / −0.08** | −0.125 / **−3.17** | +0.003 / −0.03 |
| synonym (1894) | 0.087 / 11.12 | **−0.003 / −0.12** | −0.082 / **−2.43** | +0.001 / −0.02 |
| next_number_digits (200) | 0.305 / 15.24 | **+0.010 / +0.09** | −0.305 / **−6.22** | +0.025 / −0.01 |
| prev_number_digits (200) | 0.270 / 14.70 | **+0.020 / −0.05** | −0.270 / **−5.77** | +0.005 / −0.02 |

**FINDINGS:**
- **qfin does NOT read task information directly from the demo-2 pre-label token.** Cutting that edge is
  indistinguishable from cutting an attention edge to a structural token (the demo-2 `Q:` colon): both
  leave top-1 and the gold logit essentially unchanged (|Δgold_logit| ≤ 0.12, |Δtop1| ≤ 0.02).
- **It reads directly from the label tokens.** Cutting qfin→{both labels} collapses the task: top-1 → ~0
  and gold logit drops 2.4–6.2. So the label tokens are where qfin sources the function at the output.
- Consistent with Streams M/N (the function signal lives at, and is read directly off, the label tokens).
- **Scope:** this blocks only the DIRECT qfin→demo2_prelabel edge; demo2_prelabel could still influence
  qfin indirectly via other tokens — but the direct read it would need is shown to be unnecessary.

**Outputs:** `results/direction2_label_geometry/twoshot/qfinal_attn_knockout/{summary.json, metrics.csv,
figures/qfinal_attn_knockout_bars.png}`. **Verification:** eager-attn assert; knockout zeroes the key &
renormalizes (live check); −control ≈ clean, +control collapses (sanity holds).

**Next:** none queued. Possible: per-layer knockout sweep; also cut qfin→demo1 pre-label; knock out the
qfin→labels edge layer-by-layer to localize where the read happens.

---

## 2026-06-27 — Streams M & N: ALL-LAYERS completeness check + regime folder split

**Owner:** Coordinator (tmux "label-follow-patch"). **Status:** DONE — both experiments, both regimes.

**What:** Re-ran the two patching experiments patching **all residual entries 0..28** (embedding + every
block) instead of just **6..28**, to confirm the findings are robust to the patch onset. Parametrized
both via `--patch_from_entry` (default 6); output now lands in a per-experiment **regime subfolder**
(`<exp>/L6_and_above/` and `<exp>/all_layers/`; name = `all_layers` if onset 0 else `L{n}_and_above`).
Plotters take `--regime`; the interval downstream plot reads its first-read entry from the summary
(`patch_from_entry+1`). Moved the prior L6 results into `L6_and_above/`.

**Files:** edited `patch_interval_sixtoken.py`, `patch_labelset_follow.py` (arg + `global PATCH_FROM_ENTRY`
+ regime out_dir), `plot_patch_interval_sixtoken.py`, `plot_patch_labelset_follow.py` (`--regime`).
**Commands:** `… patch_interval_sixtoken.py --task_pair <tp> --patch_from_entry 0 --batch_size 64`,
`… patch_labelset_follow.py --task_pair <tp> --patch_from_entry 0 --batch_size 64`, then the plotters
with `--regime all_layers`.

**RESULTS — all_layers vs L6_and_above:**

Label-follow (isolated recovery — the headline) is **essentially identical**:

| | L6_and_above | all_layers |
|---|---|---|
| antonym→synonym  both_labels·isolated | 74.0% | 74.0% |
| antonym→synonym  demo2_prelabel·isolated | −0.6% | −0.7% |
| prev→next digits  both_labels·isolated | 95.3% | 96.5% |
| prev→next digits  demo2_prelabel·isolated | −1.0% | −1.0% |

Interval (logit shift) — `demo2 label → query input` essentially unchanged (antonym +2.90→**+2.97**,
digits +2.80→**+2.88**); pre-label rows ≈0; structure (upper-triangle, `j=query pre label`→0) preserved.

**FINDINGS:**
- **Conclusions are robust to patch onset.** The label tokens carry/drive the function in both regimes;
  the pre-label carries ~0; isolated label recovery is unchanged (74% words / ~96% numbers).
- **Why label-follow is identical:** the demo **label** tokens are byte-identical across base/target
  (paired design), so patching their **embedding** (entry 0) is a no-op — only entries ≥1 matter, and
  1..5 add ~nothing on top of 6..28. So `both_labels·isolated` ≈ unchanged.
- **One expected difference (interval grid):** in `all_layers` the **demo2 INPUT** becomes a strong
  carrier (antonym `in→pre2` +1.46→**+3.41**, `in→qin` +1.34→**+3.09**; digits similar). This is because
  the demo inputs DIFFER across base/target, so patching the input's embedding literally swaps the demo
  input word to the target task's — not a contradiction, just the embedding doing the obvious thing.
  The label-token story is unaffected.

**Outputs:** `…/interval_patch_sixtoken/{L6_and_above,all_layers}/…` and
`…/label_follow_patch/{L6_and_above,all_layers}/…` (each with figures/). **Verification:** all_layers
interval `Δcos[...,entry0]≈0` asserted; isolated hook asserted to change logits; L6 data moved + re-plotted
identical (interval vmax 2.898 / 0.071 / 0.078 reproduced).

---

## 2026-06-26 — Stream N: ISOLATED label-token patching — do ONLY the labels drive the output?

**Owner:** Coordinator (tmux "label-follow-patch"). **Status:** DONE — both task pairs + figures.

**What:** Tests whether overwriting the two demo **label** tokens with the other prompt's activations
makes the model follow the OTHER task — and whether the labels do it **directly**. Run base/source
prompt; at residual entries 6..28 overwrite a token set ← **target**, read output `logit_diff =
logit(tgt_gold₁) − logit(src_gold₁)` at query-final. Two modes:
- **open** — overwrite only the patched positions (rest recomputes freely → effect can also be RELAYED
  via in-between/query tokens).
- **isolated** — overwrite patched positions ← target AND **pin every other non-output token to its base
  value** (`h[:, :-1] = base_full[:, :-1, e]` then set patched roles → target). Only the labels carry
  target; only the DIRECT label→output attention path is open.
Direction = Stream M (base antonym/prev, patch-in synonym/next). Metric logit-only; all pairs;
`recovery = (patched − baseline)/(ceiling − baseline)`, baseline = source prompt, ceiling = target prompt.

**Files:** NEW `src/eval_scripts/patch_labelset_follow.py` (reuses Stream-M scaffolding; adds a base
**full** residual-stack capture at all positions for entries 6..28 per chunk, ~3GB at N=544, and an
isolated freeze-then-set hook). NEW `src/eval_scripts/plot_patch_labelset_follow.py` (combined grouped
bars). Output `results/direction2_label_geometry/twoshot/label_follow_patch/`.

**Commands:**
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 python src/eval_scripts/patch_labelset_follow.py --task_pair {antonym_synonym|next_number_digits_prev_number_digits} --batch_size 64`
then `python src/eval_scripts/plot_patch_labelset_follow.py`.

**RESULTS — recovery toward target (baseline → ceiling):**

| condition | antonym→synonym (−1.93→+2.99) | prev→next digits (−1.51→+3.23) |
|---|---|---|
| demo2 pre-label · open | 2.2% | 1.5% |
| demo2 pre-label · **isolated** | −0.6% | −1.0% |
| both labels · open | 94.7% | 96.1% |
| both labels · **isolated** | **74.0%** | **95.3%** |

**FINDINGS:**
- **Yes — the label tokens drive the output, largely directly.** With every other non-output token
  pinned to base (only labels carry target, only the direct label→output path open), patching the two
  labels still recovers **74% (words) / 95% (numbers)** of the full task switch. So the model essentially
  follows the other task from the label tokens alone.
- **Pre-label is ruled out:** demo-2 pre-label recovers ~2% open and ~0% (even slightly negative)
  isolated — it carries essentially none of the function signal to the output.
- **Relay gap (open − isolated):** ~**21%** for words but ~**0.8%** for numbers. For digits the label
  effect is almost entirely *direct*; for words ~a fifth is relayed through the in-between/query tokens
  (which, in the open case, attend to the patched labels and pass the signal on). Consistent with Stream
  M (numbers ≈ rank-1, sharper/earlier; words more distributed).
- Open `both_labels` reproduces the earlier open-path numbers (94.7% / 96.1%); the `> 100%` seen on the
  16-pair smoke is a small-n artifact (baseline −0.93 vs full −1.93).

**Outputs:** `<pair>_summary.json`, `<pair>_logitdiff.csv`, `figures/<pair>_label_follow_bars.png`,
`figures/combined_label_follow_bars.png`. **Verification:** isolated hook asserted to change logits;
baseline/ceiling sanity-matched; the superseded open-only run's files were cleaned before the rerun.

**Next:** none queued. Possible: isolate single demo-1 vs demo-2 label; sweep the pin-onset layer.

---

## 2026-06-26 — Stream M: six-token INTERVAL activation-patching (2-shot; logit flip + downstream cosine)

**Owner:** Coordinator (tmux "sixtoken-interval-patch"). **Status:** DONE — both task pairs, all figures.

**What:** Causal patching on the Stream-K **two-shot** matched-label paired prompts. Study 6 single-token
positions (prompt order): `L1` (demo-1 label), `in2` (demo-2 input), `pre2` (demo-2 pre-label `A:`),
`L2` (demo-2 label), `qin` (query input), `qfin` (query-final `A:`). For every ordered token pair
`(i,j)`, `i<j`: run the **base/source** prompt and at **residual entries 6..28** (29-entry stack;
0 = embedding) OVERWRITE token `i` ← target prompt's activation (switch i to the other function for
L6+) and token `j` ← base's own clean activation (pin j to original; blocks relay). Direction
**antonym→synonym** and **prev→next (digits)** (base = source, patch-in = target), per user.

**Metrics:** (M1) output **logit flip** at `qfin`: `logit(tgt_gold₁) − logit(src_gold₁)`, reported as
mean steered vs no-patch **baseline** and the shift = steered−baseline → **6×6 upper-triangle grid**.
(M2) **downstream cosine** per `k>j` per entry L: `Δcos(k,L) = cos(steered_k[L],tgt_k[L]) −
cos(base_k[L],tgt_k[L])`, valid L≥7 → `[6,6,6,29]` array.

**Files:** NEW `src/eval_scripts/patch_interval_sixtoken.py` (GPU; mirrors Stream-L hook/trace/batching:
2-arg `(output,layer_name)` factory closure, tuple-vs-tensor dispatch, **assign** not `+=`, per-row
left-pad positions). NEW `src/eval_scripts/plot_patch_interval_sixtoken.py` (pure plotting). Pair
construction copies `capture_and_grade_twoshot_paired.py` (matched distinct labels L1,L2), additionally
requiring single-token demo inputs (drops 0 tuples — verified).

**Commands:**
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 python src/eval_scripts/patch_interval_sixtoken.py --task_pair {antonym_synonym|next_number_digits_prev_number_digits} --batch_size 64`
then `python src/eval_scripts/plot_patch_interval_sixtoken.py`.
(batch 256 OOMs on the 4090 — 29-layer `retain_output` over the full batch; 64 is ample. ~17 forwards/task.)

**RESULTS — logit-flip shift (mean steered − baseline; baseline antonym −1.93, digits −1.51):**

| (i switch-on → j pin) | antonym→synonym | prev→next digits |
|---|---|---|
| **L2 → qin** | **+2.90** | **+2.80** |
| in2 → pre2 | +1.46 | +0.08 |
| in2 → qin | +1.34 | +0.05 |
| L1 → in2 | +0.91 | +1.56 |
| L1 → L2 | +0.80 | **+1.83** |
| pre2 → * | ~0 | ~0 |
| * → qfin | +0.00 (pinning output blocks all effect) | +0.00 |

**FINDINGS:**
- **The demo-2 LABEL (`L2`) is the dominant carrier** to the output in both tasks: switching L2 to the
  target while pinning the query input (`qin`) flips the logit gap by ~+2.8–2.9 (from −1.5/−1.9 baseline
  to clearly positive). The **pre-label `A:` (`pre2`) carries ~nothing** (≈0 everywhere) — consistent
  with Stream K (function axis lives at the label token, not the `A:`).
- **Words vs numbers differ at the INPUT/early-label tokens.** For **words**, the demo-2 **input** (`in2`)
  is a strong carrier (+1.3–1.5); the demo-1 label (`L1`) is moderate (~+0.8–0.9). For **numbers**, `in2`
  carries ~0 but **`L1` is strong (+1.5–1.8)** — for ±1 digit maps the input is redundant given the label,
  and a single demo label already pins the function. (Mirrors Stream K: numbers ≈ rank-1, peak early.)
- **Pinning the output token `qfin` to original zeroes the logit effect exactly** (column j=qfin all
  +0.000) — a clean causal boundary: you cannot move the answer if the read position is held at baseline.
- **Downstream cosine** (M2): switching `L2` (pin `qin`) pushes **`qfin` toward the target** (Δcos grows
  monotonically across entries 7→28, peak ~+0.07); switching `in2` (pin `pre2`) pushes **`L2`** strongly.
  Effects are small in magnitude (≤0.08) — the representations are already highly aligned (paired design),
  so patching nudges rather than reorients. `pre2`-source panels are flat (~0), echoing M1.

**Outputs** `results/direction2_label_geometry/twoshot/interval_patch_sixtoken/`:
`<pair>_logit_grid.npy`/`_logit_shift_grid.{npy,csv}`/`_downstream_dcos.npy` (npy git-ignored),
`<pair>_summary.json`, `figures/{combined_logit_shift_heatmap.png, <pair>_logit_shift_heatmap.png,
<pair>_downstream_propagation.png}`. **Verification:** Δcos@entry6 ≈ 0 (asserted <1e-3); upper-triangle
finite; paired role-invariant (non-input tokens byte-identical across functions) asserted per pair;
positions computed per-row (in-context tokenization shifts the query block by ±1 token for some words).

**Next:** none queued. Possible extensions: sweep the patch-onset layer (currently fixed L6); patch only
one token (i without pinning j) to separate "inject" from "block-relay"; bidirectional (synonym→antonym).

---

## 2026-06-25 — Stream L: label→query-final cosine-shift heatmaps (29×29 layer×layer, per task × α)

**Owner:** Coordinator (tmux "label-prelabel-cos-heatmap"). **Status:** DONE — 4 heatmaps rendered.

**What:** 2-D map of the Stream-E label-token steering. For each task pair + fixed source→target
direction, a **29×29** heatmap (x = intervention layer at the demo **label** token, y = read layer at
the **query-final** token; layer 0 = embedding / `transformer.drop`, 1–28 = block outputs). Cell =
mean over prompt pairs of `Δcos = cos(steered_src_final(k), tgt_final(k)) − cos(src_final(k),
tgt_final(k))` — i.e. how much injecting `α·steer_vec(i)` at the source prompt's label token (layer i)
pushes the source's query-final representation toward the (unsteered) target's. One heatmap per
**task × α**, α∈{2,4} → 4 total. Directions: **antonym→synonym**, **prev→next (digits)**.

**Construction:** same overlapping paired-1-shot design as `capture_and_grade_oneshot_paired.py`
(shared single-token label `w` + shared query `q`; only the demo INPUT differs by function, so the
label token AND query-final token are byte-identical across the pair). `steer_vec(i) =
mean_pairs[act_tgt_label(i) − act_src_label(i)]`, added with positive α to the SOURCE prompt. No
correctness filter (blind Δ, matches prior steering). n_pairs: ant/syn 544, digits 198.

**Files:** NEW `src/eval_scripts/steer_label_cos_heatmap.py` (reuses pair construction +
`load_gpt_model_and_tokenizer`, `word_pairs_to_prompt_data`/`create_prompt`/`get_token_meta_labels`;
inlines the baukit-free `selected_token_records`/`extract_positions`; baukit `TraceDict`
edit_output+retain_output hook from `steer_label_to_query.py`). Output (TRACKED)
`results/direction2_label_geometry/oneshot_label_intervention_cos_heatmap/`: 4 `figures/*_cos_shift_heatmap.png`,
4 `<task>_alpha{2,4}_grid.{npy,csv}`, 2 `<task>_summary.json`. Logs `logs/label_cos_heatmap_*.log`.
NEW `src/eval_scripts/plot_label_cos_heatmap_grid.py` (pure plotting from the saved grids; no GPU) →
`figures/combined_2x2_cos_shift_heatmap.png` — all 4 panels (rows=task, cols=α) on ONE figure with a
**shared** symmetric colour scale (vmax 0.043) for direct comparison; the shared scale makes the
digits≫words gap visible at a glance. Cmd: `python src/eval_scripts/plot_label_cos_heatmap_grid.py`.
**Env:** upgraded `matplotlib` 3.7.1→3.11.0 (3.7.1 ABI-incompatible with this box's numpy 2.1.2).

**Commands:**
`HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 python src/eval_scripts/steer_label_cos_heatmap.py --task_pair {antonym_synonym|next_number_digits_prev_number_digits} --alphas 2 4 --batch_size 128`
(smoke: add `--max_pairs 16 --layers 0 6 11`). ~2–4 min/task on the 24GB 4090.

**RESULTS — peak Δcos (intervene L, read L):**

| task | α | peak Δcos | intervene L | read L |
|---|---|---|---|---|
| antonym→synonym | 2 | **+0.0188** | 8 | 18 |
| antonym→synonym | 4 | +0.0166 | 6 | 18 |
| prev→next (digits) | 2 | **+0.0428** | 4 | 16 |
| prev→next (digits) | 4 | +0.0236 | 7 | 18 |

**FINDINGS:**
- **Strictly downstream / upper-triangular:** every grid is **exactly 0 for read k ≤ intervene i**
  (incl. the diagonal) — a label-token edit only reaches the query-final token via later blocks'
  attention. **Embedding row & column are exactly 0** (label + query-final tokens byte-identical
  across f1/f2 ⇒ zero embedding diff, baseline cos=1). Both are built-in sanity checks; they hold to 0.
- **Mid-layer causal band feeding late reads.** Words: intervene ~L5–12 (hot spot L7–9) → read L16–28,
  peak read **L18**; dead by intervene ~L16. Digits: band starts **earlier** (intervene ~L4–13, peak
  L4–8) → read L16–18. Mirrors the prior 1-D causal window (L4–9 plateau, dead by L16) and the
  numbers-steer-earlier / words-mid-layer asymmetry.
- **Digits ≫ words** (peak 0.043 vs 0.019) — the ±1 digit function axis is ~2× more steerable toward
  the partner, consistent with numbers≈rank-1 geometry.
- **α=2 > α=4 at the peak for BOTH** — α=4 already overshoots for this toward-target-cosine metric
  (prior linear regime 0.5–4 was for the logit/flip readout; the cosine-convergence metric saturates
  sooner). Absolute shifts are SMALL because baseline cos sits near ceiling (~0.98–0.99 words, ~0.99
  digits) — little headroom; the *pattern* (where in the layer×layer grid the push lands), not the
  magnitude, is the result.

**Verification:** structural invariants hold exactly (embedding row/col=0, lower-tri incl diag=0, all
finite); per-pair byte-identical label & query-final tokens asserted at build time; peak cells land in
the prior CIE/steering mid-layer band. **Next (optional):** reverse directions (syn→ant, next→prev),
overlay the 1-D `oneshot_steering` profile as a diagonal slice, or a random-direction control.
**Blockers:** None. Plan: `/root/.claude/plans/read-claude-md-and-understand-snuggly-fog.md`.

---

## 2026-06-19 — Stream K: TWO-shot matched-label paired captures (antonym/synonym + digit next/prev)

**Owner:** Coordinator (tmux "twoshot-paired"). **Status:** DONE — Stage A (activation gathering) complete; geometry/judge/steering are follow-on stages (not built).

**What:** New 2-shot prompt style extending the 1-shot paired-difference design. Each prompt has TWO
demos whose labels `(L1,L2)` are matched position-by-position across the two functions and are DISTINCT
within a prompt; demo INPUTS differ by function; query `q` shared (shared-input pool, gold under both).
Per user, additionally capture **demo-2's pre-label token** (the `A:` before L2). Example:
```
antonym: Q: unable\nA: able\n\nQ: scarce\nA: abundant\n\nQ: secular\nA:   (gold religious)
synonym: Q: capable\nA: able\n\nQ: plentiful\nA: abundant\n\nQ: secular\nA: (gold nonreligious)
```
Labels (able, abundant) + query (secular) identical across f1/f2; only the 2 demo inputs differ.

**Construction:** one tuple per shared-output label word `w` → L1=w, L2=random distinct label,
`q` from shared-input pool minus {L1,L2,4 demo inputs}; deterministic per `(seed,task_pair,w)`. Pools
(unchanged from 1-shot): ant/syn 544 labels / 1224 queries; digits 198 labels / 200 queries.

**Roles captured (5/prompt):** `demo1_prelabel`, `demo1_label`, `demo2_prelabel`, `demo2_label`,
`query_final` (from `selected_token_records`; `pre_label_token`/`last_label_token` @ icl idx 1 & 2, plus
`last_prompt_token`). First-token rank grading stamped per row + `grading.json`.

**Files:** NEW `src/eval_scripts/capture_and_grade_twoshot_paired.py` (sibling of
`capture_and_grade_oneshot_paired.py`; reuses `get_residual_stack`/`selected_token_records`/`flush_shard`,
`word_pairs_to_prompt_data`/`create_prompt` — all handle 2 demos with no special-casing). Output (intermediate,
git-ignored) `artifacts/twoshot_paired_graded/<pair>/` (shard_*.pt + index.json + grading.json + scores.json).

**Commands:**
`HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 python src/eval_scripts/capture_and_grade_twoshot_paired.py --task_pair {antonym_synonym|next_number_digits_prev_number_digits}`

**RESULTS — first-token top-1 (2-shot vs 1-shot):**

| task | 2-shot top1 | 1-shot top1 | top2 | top3 |
|---|---|---|---|---|
| antonym | **0.439** | 0.232 | 0.539 | 0.623 |
| synonym | **0.221** | 0.066 | 0.373 | 0.487 |
| next_number_digits | 0.980 | — | 0.995 | 1.000 |
| prev_number_digits | 0.990 | — | 1.000 | 1.000 |

**FINDINGS:** A second matched-label demo roughly **doubles** first-token top-1 for the word tasks
(antonym 0.232→0.439, synonym 0.066→0.221) — two demos identify the function far better than one, and cut
the dominant 1-shot copy-the-query failure. Digits are at ceiling (0.98–0.99). **Verification:** ant/syn
5440 rows (544×2×5), digits 1980 (198×2×5); all finite; balanced 5-role sets; 0 degenerate tuples
(L1≠L2, q∉forbidden); token positions identical across f1/f2 (paired invariant holds at every captured role).

**UPDATE — GPT-4 judge eval DONE + activation rows tagged.** `OPENAI_API_KEY` resolves this session via
`/proc/1/environ` (the repo's `get_openai_key()` fallback), so the judge ran here. NEW
`src/eval_scripts/judge_twoshot_paired.py` (2-demo prompt rebuild, batched greedy gen, `_digits`→base
number-judge mapping; reuses `JUDGE_SYSTEMS`/`judge`/`extract_answer`/`get_openai_key` from
`judge_oneshot_paired.py`) → `results/direction2_label_geometry/twoshot_<task>_judge/judged_results.json`.
NEW `src/eval_scripts/tag_twoshot_activations_judge.py` stamped a `judge_top1` bool into ALL activation
rows (5440 ant/syn + 1980 digits) and grading.json (key = function_task, label1, label2, query).

**GPT-4 judge_top1 (2-shot vs 1-shot), + copy-the-query count:**

| task | 2-shot judge | 1-shot judge | 2-shot copied | 1-shot copied |
|---|---|---|---|---|
| antonym | **0.574** | 0.276 | 125 | 263 |
| synonym | **0.408** | 0.143 | 230 | 339 |
| next_number_digits | 0.980 | — | 2 | — |
| prev_number_digits | 0.980 | — | 1 | — |

A second matched demo ~2–3× the 1-shot semantic accuracy (antonym 0.276→0.574, synonym 0.143→0.408) and
nearly halves the copy-the-query failure (ant 263→125, syn 339→230). Digits at ceiling (0.98), judge ≈
first-token (single-token golds). Every row now filters by first-token `top1/2/3` AND GPT-4 `judge_top1`.

**UPDATE — GEOMETRY: mean-pairwise-cosine heatmap (token position × layer).** NEW
`src/eval_scripts/plot_twoshot_diffcos_heatmap.py` (generalizes `plot_pairwise_cos_hist_byjudge.py` —
collapses each histogram to its MEAN and sweeps the full 5-position × 28-layer grid; reuses the
unit-normalize→Gram→upper-triangle recipe). Cell = mean pairwise cosine of `D = act(f1)−act(f2)` over
all prompt-keys `(label1,label2,query)`; all prompts, independent color scale per pair. Cmd
`python src/eval_scripts/plot_twoshot_diffcos_heatmap.py` → `results/direction2_label_geometry/twoshot_diffcos_heatmap/`
(`<pair>_meancos_heatmap.png` ×2 + `meancos_grid.json`). Verified: grids finite, n=544/198 every cell,
independent cell recompute matches to 5 dp.

**FINDINGS (peak mean cos, role @ layer):**
- **antonym_synonym (words):** peak **0.351 @ query_final, L12**. Coherence ranks
  `query_final > demo2_label (~0.23, L12) > demo1_label (~0.18, L8) ≫ demo2_prelabel (~0.08) > demo1_prelabel (~0)`.
  Mid-layer band (L8–15); embedding/early layers ~0; decays after ~L16.
- **digit next/prev (numbers):** peak **0.840 @ demo2_label, L4**. Label + query positions all high
  (~0.8) and peak **EARLY (L4)** then slowly decay; pre-label positions lower (demo2_prelabel ~0.3–0.6,
  demo1_prelabel ~0).
- **The function axis lives at the LABEL token, not the pre-label `A:`** (label rows ≫ pre-label rows in
  both pairs) — and demo2_prelabel > demo1_prelabel (some axis accumulates by the 2nd demo's pre-label).
- **Accumulation across demos:** `demo2_label ≥ demo1_label` (words clearly; digits ~tied at ceiling), and
  `query_final` is the most coherent for words.
- **numbers ≫ words** (0.84 vs 0.35) — same as 1-shot (numbers ≈ rank-1 ±1 axis); numbers peak early (L4),
  words mid (L12).
- **2-shot > 1-shot coherence:** at L11, 2-shot query_final 0.31 vs 1-shot final 0.12; demo2_label 0.20 vs
  1-shot label 0.15. The second matched demo sharpens the function direction, most at the query.

**UPDATE — GEOMETRY: stable rank of unit-normalised diffs per layer & position.** NEW
`src/eval_scripts/plot_twoshot_stable_rank_by_layer.py` (generalizes `plot_stable_rank_by_layer_byjudge.py`
to the 5-role 2-shot capture; metric `stable rank = Σσ²/σ₁²` of the unit-normalized D matrix). ONE figure,
one panel per task pair (x=layer, one line per role); all prompts. The mean-pairwise-cosine view is
deliberately omitted — it is identical to the cosine heatmap (`twoshot/diffcos_heatmap/`), so plotting it
again added nothing. Cmd `python src/eval_scripts/plot_twoshot_stable_rank_by_layer.py` →
`…/twoshot/stable_rank/stable_rank.png` + `stable_rank_by_layer.json`.

**FINDINGS (stable rank @ L9; lower = more rank-1 / one dominant function axis):**
- **antonym_synonym:** query_final **3.06** < demo2_label 4.35 < demo1_label 4.91 ≪ demo2_prelabel 13.2,
  demo1_prelabel 16.0. So the function axis is most rank-1 at the query, sharper at demo2 than demo1, and
  ~random at the pre-label `A:` (D≈0 there). Dips mid-layer (mirror of the cosine heatmap).
- **digit next/prev:** demo1_label **1.26**, demo2_label 1.36, query_final 1.36 (≈ rank-1, one dominant ±1
  axis); pre-label higher (demo2_prelabel 2.78, demo1_prelabel 7.35).
- Exactly the mirror of the mean-cos heatmap and consistent with 1-shot (numbers ≈ rank-1, words ~3–5).

**RESULTS REORG:** all 2-ICL-example deliverables now live under one umbrella
`results/direction2_label_geometry/twoshot/`:
`judge/{antonym,synonym,next_number_digits,prev_number_digits}/judged_results.json`, `diffcos_heatmap/`,
`stable_rank/`. Default output/read paths updated in `judge_twoshot_paired.py` (writes `…/twoshot/judge/<task>`),
`tag_twoshot_activations_judge.py` (reads same), `plot_twoshot_diffcos_heatmap.py` and
`plot_twoshot_stable_rank_by_layer.py` (write under `…/twoshot/`). Capture intermediates stay in
`artifacts/twoshot_paired_graded/`.

**Next:** optional judge-split variants, 2-shot switch-steering. **Blockers:** None.
Plans: `/root/.claude/plans/validated-fluttering-noodle.md` (capture),
`/root/.claude/plans/playful-weaving-wilkes.md` (heatmap).

---

## 2026-06-19 — Stream E follow-up: LOGIT-READOUT switch-steering, clean train/test split (digits + ant/syn)

**Owner:** Coordinator (tmux "activation-geometry-steering"). **Status:** DONE — 8 logit-diff curves rendered.

**What:** Cheaper, leakage-free readout (replaces sample+judge for single-token pairs). Steer a
source-task 1-shot prompt toward target; in ONE forward read **logit(target_gold) − logit(source_gold)**
at the query final token. α=0 = clean baseline (flat). NO sampling, NO GPT-4 judge, NO API key.

**Clean TRAIN/TEST split (the key methodological fix):** LABEL pool = shared single-token outputs;
QUERY pool = shared inputs with single-token gold under both tasks. Reserve **100 test labels** (ICL
example) + **100 test queries** (final query); everything else = train. **Δ derived from 100 TRAIN
prompt pairs** (label∈train_out × query∈train_in) via our own forward passes reading act(f1)−act(f2)
at the demo-label token (Δ_label) and query-final token (Δ_final) — NOT the contaminated full-pool
capture. n_train=100 for BOTH pairs → comparable. Test label/query tokens never appear in train, so no
overlap in either the ICL example or the final query. Pools: ant/syn label 544 / query 1003 (train
444 / 903 distinct); digits label 198 / query 200 (train 98 / 100 distinct — ~2 labels repeat across
the 100 train pairs, negligible).

**Files:** NEW `src/eval_scripts/steer_switch_logit.py`, `src/eval_scripts/plot_switch_logit.py`.
Output `results/oneshot_switch_logit/<direction>/logit_diff.json`, `delta_meta.json`, `figures/`
(8 panels + `fig_logit_aggregate.png`). Cmd: `python src/eval_scripts/steer_switch_logit.py` then
`python src/eval_scripts/plot_switch_logit.py`. Runs in a few min on the 24GB card.

**RESULTS — logit(target)−logit(source): baseline(α0) → peak (layer, α):**

| direction | site | baseline | peak | @L | @α |
|---|---|---|---|---|---|
| synonym→antonym | label | −1.69 | +0.47 | 8 | 4 |
| synonym→antonym | final | −1.69 | +1.50 | 12 | 8 |
| antonym→synonym | label | −0.03 | +2.63 | 6 | 8 |
| antonym→synonym | final | −0.03 | +3.76 | 12 | 8 |
| prev_digits→next_digits | label | −0.23 | +2.05 | 6 | 2 |
| prev_digits→next_digits | final | −0.23 | **+4.41** | 12 | 8 |
| next_digits→prev_digits | label | −2.12 | +0.38 | 4 | 2 |
| next_digits→prev_digits | final | −2.12 | +1.84 | 12 | 8 |

**FINDINGS:**
- **Causal switch in all 8** — every panel rises from a negative baseline (favors source answer) up
  **across zero into positive** (favors target) = the model now prefers the target-task answer.
- **Two distinct causal windows by site:** demo-label injection peaks **early (L4–8)** and dies by
  ~L14; final-prompt-token injection peaks **later (L10–12)** and reaches **higher** logit-diffs
  (final ≫ label for the peak magnitude). Both dead by ~L16–20.
- **Digits show the cleanest, sharpest curves** (single-token throughout); prev→next digits final hits
  +4.41. Direction asymmetry persists in the baselines (next/synonym prompts strongly favor their own
  answer: −2.12 / −1.69; antonym/prev baselines near 0).
- This corroborates the earlier sample+judge run but with an exact, leakage-free metric and 1/100th
  the cost.

**Note:** the earlier digit *sampling* generation (run_generate_digits) was killed as redundant; the
word next/prev pair is dropped from here on (multi-token). **Blockers:** None.

---

## 2026-06-19 — Stream E follow-up: DIGIT next/prev variant of switch-steering (single-token labels)

**Owner:** Coordinator (tmux "activation-geometry-steering"). **Status:** IN PROGRESS — datasets + capture done, generation running, judge pending (needs OPENAI_API_KEY).

**Why:** the word-form number pair has ~87% MULTI-token gold answers (seventy→seventy-one, one hundred
sixty-one). Digit form is cleaner: **every integer 0–250 is a single GPT-J token** (verified), so labels
AND golds are 100% single-token. Lets first-token metrics work and removes the multi-token confound.

**What:** NEW `dataset_files/abstractive/{next_number_digits,prev_number_digits}.json` (inputs 1–200,
next=i+1 / prev=i−1; mirrors the word pair). Added pair `next_number_digits_prev_number_digits` to
`capture_and_grade_oneshot_paired.py` TASK_PAIRS and to `steer_switch_judge.py` DIRECTIONS
(prev_digits→next_digits sign+1, next_digits→prev_digits sign−1). steer_switch_judge now maps digit
targets to the number judge (`judge_system_task` strips `_digits`) and treats them as multiword.

**Capture done** (`results/oneshot_paired_graded/next_number_digits_prev_number_digits/`): 198 label
words, 200 query pool. First-token grading (now meaningful — single-token): next_digits top1 **0.848**,
prev_digits top1 **0.677** (≈ the word pair's full-answer judge 0.818/0.621). 2 directions × 2 sites
generation running → 2 new plots each. **Next:** judge (key) + `plot_switch_steering.py` (add the 2
digit directions to its DIRECTIONS list) → 4 new figures. **Blockers:** OPENAI_API_KEY (deleted; rotate).

---

## 2026-06-18 — Stream E follow-up: BEHAVIORAL task-switch steering (8 accuracy curves)

**Owner:** Coordinator (tmux "switch-steering" / "activation-geometry-steering"). **Status:** DONE — 564k gens judged, 8 figures rendered.

**RESULTS (peak target-task acc over the layer×α sweep vs α=0 baseline; n=100 prompts ×10 samples):**

| direction | site | baseline | peak | @L | @α | lift |
|---|---|---|---|---|---|---|
| synonym→antonym | label | 0.035 | 0.153 | 6 | 8 | +0.118 |
| synonym→antonym | final | 0.035 | **0.273** | 8 | 8 | +0.238 |
| antonym→synonym | label | 0.081 | 0.142 | 6 | 4 | +0.061 |
| antonym→synonym | final | 0.081 | **0.189** | 14 | 4 | +0.108 |
| prev_number→next_number | label | 0.107 | **0.463** | 4 | 2 | +0.356 |
| prev_number→next_number | final | 0.107 | 0.266 | 8 | 8 | +0.159 |
| next_number→prev_number | label | 0.053 | **0.354** | 4 | 2 | +0.301 |
| next_number→prev_number | final | 0.053 | 0.130 | 8 | 8 | +0.077 |

**FINDINGS:**
- **The task switch is causal in ALL 8 cases** — every direction/site lifts target-task accuracy above
  baseline, peaking **mid-layer (L4–14) and decaying by ~L16** (classic FV causal window).
- **Site asymmetry by pair type:** NUMBERS steer best at the **demo label token** (prev→next 0.46@L4,
  next→prev 0.35@L4 — both early, α=2); WORDS steer best at the **final prompt token** (syn→ant
  0.27@L8, ant→syn 0.19@L14). I.e. the ±1 number function is most steerable upstream at the label;
  the word-meaning function is most steerable at the predictive position.
- **Direction asymmetry persists** (prior-bias): toward the easier target lifts more — prev→**next**
  (0.46) is the strongest panel; syn→**ant** > ant→**syn**.
- Baselines behave as designed (source-task prompt judged for target = LOW: 0.035–0.107), confirming
  headroom. Bigger α (4–8) needed for words; α=2 already peaks for numbers at the label site.

**Files added:** `src/eval_scripts/steer_switch_judge.py`, `src/eval_scripts/plot_switch_steering.py`.
Outputs in `results/oneshot_switch_steering/` (12 dirs × {generations,judged}.jsonl + accuracy.json;
`deltas_used.json`; `figures/` = 8 panels + `fig_switch_aggregate.png`; run_generate.log, run_judge*.log).

**Compute:** generation ~50 min (564k samples, GPT-J, 24GB); judge ~45 min (GPT-4.1, 40-way parallel,
~11k batches). GOTCHA fixed mid-run: number-task judge sometimes returns N+1 verdicts for N pairs
(deterministic at temp 0) → added recursive batch-splitting + skip-if-accuracy.json resume (see DECISIONS).

**SECURITY NOTE:** the user pasted live OPENAI + ANTHROPIC keys into chat; used OpenAI via a gitignored
`.openai_key` (since deleted). Both keys should be ROTATED.

**What:** The first *behavioral* (generate + judge) version of switch-steering. Take a 1-shot prompt
whose single ICL demo is the *source* task, inject `sign·α·Δ_site(L)` at one token position, sample
**n=10 @ temperature 1**, and measure how often GPT-4.1 judges the answer correct **for the target
task**. Produces **8 graphs** = 4 directions × 2 injection sites; each graph x=injection layer,
y=target-task accuracy, one line per α∈{0,0.5,1,2,4,8} (α=0 = unsteered baseline, drawn flat).

- **Directions:** synonym→antonym, antonym→synonym, prev_number→next_number, next_number→prev_number.
- **Sites:** `label` = demo last_label_token (inject Δ_label); `final` = final prompt token /
  query predictive position (inject Δ_final). Contrast with the prior `steer_label_to_query.py` which
  injected ONLY at the demo label and measured logit-diff, never generated/judged/swept the final site.
- **Δ** = `mean_w[act(f1)−act(f2)]` from existing `results/oneshot_paired_graded/<pair>/` shards via
  `load_capture_diffs` (NO new capture). sign=+1 if target is f1 (antonym/next), −1 if f2 (synonym/prev).
- **Scale:** 100 queries/direction (random shared-input subset, seed 42, fixed demo per query reused
  across all conditions); layers 0,2,…,26 (14); α nonzero {0.5,1,2,4,8}; n=10. No random-norm control.
- **Hook:** baukit `edit_output`, per-row position, fires ONLY on the prompt forward (seq_len>1) so the
  perturbation propagates via KV cache during generation. Prompts replicated ×n_samples explicitly
  (prompt-major), left-padded; label idx = pad_len+label_pos, final idx = −1. α=0 baseline = no hook,
  once per direction.

**Files:** NEW `src/eval_scripts/steer_switch_judge.py` (`--stage generate|judge`, both resumable;
reuses helpers from `steer_label_to_query.py` + `judge_oneshot_paired.py`), NEW
`src/eval_scripts/plot_switch_steering.py`. Outputs `results/oneshot_switch_steering/`
(`<direction>__{label,final,baseline}/generations.jsonl,judged.jsonl,accuracy.json`, `deltas_used.json`,
`figures/`).

**Commands:**
- gen: `python src/eval_scripts/steer_switch_judge.py --stage generate --output_root results/oneshot_switch_steering`
- judge: `... --stage judge ...` (needs OPENAI_API_KEY)
- plot: `python src/eval_scripts/plot_switch_steering.py`

**Smoke (synonym→antonym, n=5×2, L6/L26, α4):** hook verified working — α=0 baseline emits
synonyms/copies (advanced→advancement, sorry→no problem), steered shifts toward antonyms
(straight→short at final/L26, straight→tall, hold→frightened). Generation pipeline validated.

**BLOCKER:** judge stage needs `OPENAI_API_KEY`; `get_openai_key()` (env → /proc/1/environ) does not
find it in this session (prior streams read it from /proc/1/environ). Generation runs/caches meanwhile.

**Next:** finish generation sweep; obtain key → run `--stage judge`; render 8 figures + aggregate.

---

## 2026-06-18 — Stream J: variable-ICL FV capped at max 4 demos (top-40 heads) + held-out steering

**Owner:** Coordinator (tmux "varicl-max4"). **Status:** DONE.

**What:** Parallel of the existing `train_varicl` method (random 1–10 ICL demos, top-40 heads pooled
over 20 train tasks) but with the ICL count capped at **1–4 demos** (`--max_shots 4`). Goal: measure
what shortening the ICL window costs the resulting FV's held-out steering, plotted directly against
the max-10 (top-40) varicl line + the task-specific FV reference, on the 9 test tasks.

**Approach:** one behavioral change (`--max_shots 4`); isolated output roots so nothing existing is
overwritten. Reuses all heavy helpers (`evaluate_fv`/`get_filter_set`/`summarize_results` from
`evaluate_heldout_multitask_head_fvs.py`); the method-agnostic no-FV baseline cache
`results/gptj_fv/<task>/fs_results_layer_sweep.json` → no baseline recompute. Max-10 top-40 curve read
from the prebuilt `results/heldout_varicl_nheads_sweep/<task>/nheads_sweep_by_layer.json` (N=40),
task-specific from `results/heldout_multitask_head_eval/<task>/comparison_summary.json` → no recompute
of those lines either. Handles the documented stage-2 gotcha (test-task activations computed
separately + copied in before the 29-task FV build).

**Files (NEW):** `src/eval_scripts/run_varicl_max4_pipeline.sh` (build driver, stages 1a–1c),
`src/eval_scripts/evaluate_heldout_varicl_max4.py` (steering eval + 3-series plots). No edits to
existing source. Outputs: `results/multitask_aie_heads_varicl_max4/`, `results/_varicl_testtasks_max4/`,
`results/function_vectors/gpt-j/train_varicl_max4_top40/`, `results/heldout_varicl_max4_top40/`.

**Commands:** `bash src/eval_scripts/run_varicl_max4_pipeline.sh` then
`python src/eval_scripts/evaluate_heldout_varicl_max4.py --overwrite`.

**Build verification:** sampler draws uniformly over {1,2,3,4} only; pooled head artifact = 40 heads
over 20 train tasks (top-5 = (9,14)(15,5)(8,1)(12,10)(11,0), the canonical varicl FV heads); 29 FVs
built at n_top_heads=40, finite, norms 46–58 (expected top-40 band); 9 test-task activations copied in.
Steering eval: 9/9 test tasks evaluated, no missing-curve warnings (all max-10 N=40 + task-specific
curves read from cache), 9 per-task PNGs + AGGREGATE rendered with 3 lines each, filter sets reused
from cached `fs_results_layer_sweep.json` (no baseline recompute).

**FINDINGS — capping ICL at ≤4 demos costs a little ZERO-SHOT steering, ~nothing FEW-SHOT.**
Mean best-layer intervention top-1 over the 9 test tasks (raw argmax over all 28 layers):

| condition | max-4 (top40) | max-10 (top40) | task-specific |
|---|---|---|---|
| zero-shot + FV | **0.353** | 0.400 | 0.483 |
| 10-shot-shuffled + FV | **0.777** | 0.784 | 0.812 |

- **Zero-shot:** max-4 trails max-10 by ~0.05 (0.353 vs 0.400) — both well below task-specific (0.483).
  Same classic mid-layer (L6–12) steering bump; max-4 sits a notch under max-10 across that band, dies
  by ~L18 (see `AGGREGATE_..._max4_vs_max10.png`). So fewer ICL demos in extraction → a slightly weaker
  but still-real zero-shot FV.
- **Few-shot (10-shot-shuffled):** essentially tied (0.777 vs 0.784, Δ≈0.007) — with real context
  present at inference the shorter extraction window barely matters; both ~task-specific.
- **Per-task:** max-4 < max-10 zero-shot on most tasks but only modestly: antonym .473 vs .583,
  capitalize .917 vs .953, capitalize_first_letter .736 vs .853, product-company .099 vs .148 (both
  near-dead — needs L11+ task-specific to steer). word_length 0.00 for both (FV can't drive it
  zero-shot, matches the canonical finding). country-currency ties at .179. No task where max-4
  collapses relative to max-10. NB many zero-shot argmaxes land at L0 (embedding-norm artifact);
  restricting to L3–27 gives the same max-4 mean (0.349) and the mid-layer picks (L6–12) are stable.

**ADDENDUM — PCA-ridge activation→FV heatmap on the max-4 FVs (k_act=16, k_fv=16).** Reran the
direct PCA-space ridge study (`regress_activation_to_fv_pca_ridge.py` × 10 ICL shards +
`merge_fulldim_ridge_results.py`) with the FV regression TARGET = `train_varicl_max4_top40` (only
`--fv_root` + `--output_dir` change; the cached residual activations in
`results/residual_activations/` are FV-target-agnostic, reused unchanged). 7 test tasks (cc/pc
excluded), 899 cells. Out: `results/pca_ridge_activation_to_fv_varicl_max4_top40/`
(combined_metrics.csv, combined_test_mse_heatmap.png, combined_best_alpha_heatmap.png, combined_summary.json).

Commands:
```
for n in 1..10: python src/eval_scripts/regress_activation_to_fv_pca_ridge.py --icl_index $n \
  --fv_root results/function_vectors/gpt-j/train_varicl_max4_top40 \
  --output_dir results/pca_ridge_activation_to_fv_varicl_max4_top40 --overwrite
python src/eval_scripts/merge_fulldim_ridge_results.py --input_dir results/pca_ridge_activation_to_fv_varicl_max4_top40
```

**Heatmap result:** same canonical structure as the train_selected / max-10 runs — mid-layer (L9–15)
bowl of lowest test MSE, brightening with ICL accumulation (icl08–10) and at pre/finaltok roles; L0
(embedding) worst. Best cell **icl09/pre @ L13 = 0.1596** (FV-PC recon floor 0.1433 → gap 0.0162).
Cross-run (best cell, abs MSE / FV-PC floor / **gap-above-floor** = the FV-target-agnostic regression
quality): train_selected 0.1147/0.0992/**0.0155**; varicl_top40 max-10 0.1994/0.1780/**0.0214**;
varicl_max4_top40 0.1595/0.1433/**0.0162**. Absolute MSE differs only because each FV target has its
own 16-PC variance floor; the gap-above-floor shows the max-4 FVs are **as regressable from activations
as train_selected** and a touch tighter than the max-10 varicl FVs — i.e. shrinking the extraction
window does NOT make the FV harder to decode from the residual stream. 11 alpha-pinned cells, all at
L0 (degenerate predict-the-mean), none at the optimum — benign, matches prior runs.

**Next (optional):** add the max-4 line into a combined nheads-style overlay, or sweep max_shots
∈{2,4,6,8} to trace the extraction-window→zero-shot-steering curve. **Blockers:** None.
Plan: `/root/.claude/plans/can-you-make-a-stateful-blanket.md`.

---

## 2026-06-18 — Stream E follow-up: two MORE correctness notions on the paired 1-shot captures

**Owner:** Coordinator (tmux "oneshot-temp1-judge"). **Status:** DONE.

**What:** Added two new per-prompt correctness labels to BOTH paired 1-shot pairs
(`antonym_synonym`, `next_number_prev_number`), stamped onto every activation row (source =
demo-label, target = final-query) AND `grading.json`, alongside the existing greedy `judge_top1`:
1. **`frac_correct_temp1_n5`** (float in [0,1]) + **`n_correct_temp1_n5`** (int 0..5) — **5
   temperature=1 samples** per prompt (seed 42), each GPT-4.1-judged with the same strict per-task
   prompts; the row stores the FRACTION judged correct (k/5). (Started as a single temp=1 sample
   `judge_top1_temp1`; user upgraded to n=5 fraction for a robust, non-Bernoulli signal — the
   single-sample field was then DROPPED from all rows + grading.json. Mean frac ≈ single-sample acc,
   as expected.)
2. **`greedy_answer_prob`** (float) + **`judge_top1_p50`** (bool) — greedy regeneration capturing
   per-step probs; answer prob = product of greedy per-token probs over the answer span (tokens
   before first "\n"; = first-token prob for single-token answers, joint product for numbers).
   `judge_top1_p50` = greedy `judge_top1` AND prob ≥ 0.50. Raw float stored so the threshold is
   re-tunable for free (no recompute). Reuses existing greedy GPT-4 verdicts → NO extra API calls.
   (Threshold walked 0.70 → 0.60 → 0.50 per user; each step a free instant re-gate from the stored
   float, no model rerun. p70/p60 fields removed; only p50 remains.)

**Files:** MODIFIED `src/eval_scripts/judge_oneshot_paired.py` (`--do_sample`/`--temperature`/
`--output_suffix`; provenance in summary), `src/eval_scripts/tag_oneshot_activations_judge.py`
(`--judge_suffix`/`--tag_field`). NEW `src/eval_scripts/sample_judge_oneshot_paired.py` (n-sample
fraction; reuses judge_oneshot_paired helpers; `num_return_sequences` for the n draws),
`src/eval_scripts/compute_oneshot_greedy_answer_prob.py`. Outputs:
`results/oneshot_{task}_judge_sample5/judged_results.json` (per-sample gens + verdicts);
logs `results/_oneshot_sample5_*.log`, `results/_oneshot_greedyprob_*.log`. Greedy results dirs untouched.

**Commands:**
- `sample_judge_oneshot_paired.py --graded_dir results/oneshot_paired_graded/<pair> --function_tasks <f1 f2> --n_samples 5 --temperature 1.0`
- `compute_oneshot_greedy_answer_prob.py --graded_dir .../<pair> --function_tasks <f1 f2> --threshold 0.50`
- (re-gate at a new threshold = re-derive `judge_top1_pXX` from stored `greedy_answer_prob`; no rerun.)

**Results (n per task: ant/syn 544, next/prev 198):**

| task | greedy judge_top1 | temp1 mean frac (n=5) | any-correct | all-correct | mean greedy prob | judge_top1_p50 |
|---|---|---|---|---|---|---|
| antonym | 0.276 | ~0.24 | ~340/544 | 17 | 0.176 | 19/544 |
| synonym | 0.143 | 0.229 | 323/544 | 3 | 0.163 | 0/544 |
| next_number | 0.818 | 0.485 | 169/198 | 15 | 0.492 | 98/198 |
| prev_number | 0.621 | 0.341 | 144/198 | 6 | 0.392 | 47/198 |

**Findings:**
- **temp=1 sampling vs greedy:** numbers drop hard (next .818→.485, prev .621→.341 mean frac) — greedy
  picks the right argmax, sampling injects errors. synonym RISES (.143→.229) — many valid answers, so
  a sampled non-argmax is often still correct; antonym ~flat (.276→~.24).
- **any-correct ≫ all-correct** (e.g. synonym 323 vs 3; next 169 vs 15): most prompts are *sometimes*
  right under sampling, rarely *always* — the fraction captures this spread the single sample couldn't.
- **High greedy confidence is RARE for open-ended words** — mean greedy answer prob ~0.17 for ant/syn,
  so even p50 keeps few (antonym 19, synonym 0). Numbers far more confident (~0.39–0.49) → p50 keeps 98 / 47.
- **Confidence ≠ correctness (decoupled):** synonym has 8 answers ≥0.50 but 0 judge-correct;
  prev_number 58 ≥0.50, 47 judge-correct. The gate is meaningfully stricter than judge alone.
- **CAVEAT for downstream geometry:** the p50 confident-correct subset for ant/syn is too small (≤19,
  synonym 0) for a correct-vs-incorrect contrast on those tasks; usable on the number pair (98 / 47).
  For ant/syn prefer the continuous `frac_correct_temp1_n5` / `greedy_answer_prob` as covariates.

**Schema now on every row:** `judge_top1` (greedy), `frac_correct_temp1_n5` (+`n_correct_temp1_n5`),
`greedy_answer_prob`, `judge_top1_p50` — plus first-token `top1/2/3`. **Next:** geometry split by the
new labels (e.g. confident-correct vs rest on numbers; frac as a covariate). **Blockers:** None.

---

## 2026-06-18 — Stream I: GPT-J residual activations on magnitude/identity 3+1+1 prompts (labeled by correctness)

**Owner:** Coordinator (tmux "recreate-fig8"). **Status:** DONE.

**What:** Per-prompt residual-stream capture for the `magnitude|identity` ambiguous pair, for later
analysis of where/how the disambiguating function is represented. n=200 prompts/task (3 overlap + 1
differentiator demo of that task + 1 differentiator query), **balanced queries** over all 150
differentiator items. Same seed both tasks ⇒ paired prompts (identical demo/query INPUTS; only the
differentiator demo's output label differs). Residual stream (+embeddings ⇒ 29 layers) captured at
**pre_label / first_label / last_label** tokens of all 4 demos + the **query predictive** position;
every row tagged correct / partner_match (whole-answer exact match) + role/region/metadata.

**Files:** NEW `src/eval_scripts/capture_magnitude_identity_activations.py` (reuses
`get_residual_stack` + `selected_token_records` from `extract_residual_stream_activations.py`,
`split_overlap_differ`/`batched_generate` from `eval_ambiguous_disambiguation.py`). Output
`results/magnitude_identity_activations/gpt-j-6b/{magnitude,identity}.pt` (590MB each, fp16,
shape (2600, 29, 4096)) + `{task}_correctness.json`.

**Command:** `python src/eval_scripts/capture_magnitude_identity_activations.py --n_prompts 200`

**UPDATE 2026-06-18 (decimals):** Doubled both datasets to **600 entries** (300 int + 300 decimal,
≤3 dp) via `dataset_files/generate/add_decimals_magnitude_identity.py` (overwrote canonical
`{magnitude,identity}.json`; int-only backups at `{task}.int_only.json`). Decimal structure mirrors
ints: +d overlap (mag=id=d), −d differentiator (mag=d, id=−d). Re-ran capture; correctness switched
to **numeric equality** (`float(parsed)==float(gold)`, generate-to-newline, max_new_tokens 12) — NOT
normalize_answer (strips '.'). Results: acc magnitude 0.980 / identity 1.000 (unchanged — GPT-J
handles decimals: e.g. `-97.96`→`-97.96`); 200 distinct queries (~half decimal); **multi-token labels
now 400 (magnitude) / 496 (identity)** vs 0/200 before. Regenerated
`figures/magid_query_fv_projection_by_layer.png` (reusing the integer-derived top-20 FVs): **the FVs
still cleanly separate the two tasks on decimals** — separation from ~L6, magnitude below / identity
above the diagonal. Generalization holds int→decimal.

**Findings (original, integer-only):** acc magnitude 0.980 (partner 0.020), identity 1.000 (partner 0.000) — tracks the prior
3+1+1 GPT-J numbers. Dataset: overlap=150, differentiator=150 (each task 300 total). All checks pass
(no NaN; 13 rows/prompt = 4 demos×3 roles + query; join clean; 200 distinct prompts, 150 distinct
queries). **CAVEAT:** the pair is easy ⇒ near-zero incorrect class (magnitude 4 incorrect prompts,
identity 0) — not enough negatives for a correct-vs-incorrect contrast; would need a harder pair or
0-differentiator-demo prompts to induce errors.

---

## 2026-06-16 — Stream H: recreate paper Fig 8 (few-shot ICL curves) for GPT-J + Qwen3-8B-Base

**Owner:** Coordinator (tmux "recreate-fig8"). **Status:** DONE.

**What:** Recreate Figure 8 (few-shot ICL accuracy vs. #shots) with TWO model lines per panel —
GPT-J (`EleutherAI/gpt-j-6B`) and the new Qwen3-8B base (`Qwen/Qwen3-8B-Base`) — plus the dotted
majority-label baseline. Coverage: all 40 `abstractive` tasks (paper Fig 8 = 25 of them) and all
18 `ambiguous` tasks. n=200 trials/task, shots 0..10.

- **Method:** per trial, sample 10 demos + 1 query; for k=0..10 build the k-shot prompt (first k of
  the same demos + query) so shots 1..9 are nested prefixes of the 10-shot draw. **Generate greedily
  until "\n"** (stop_strings=["\n"]), take text before the newline, score with the repo's normalized
  `exact_match_score` (full answer, NOT first-token rank). **Every response is stored** for manual
  review. Batched generation with token-budget batching (left-pad) to stay within 80GB on GPT-J's
  full-MHA KV cache.
- **Env:** upgraded `transformers` 4.49→4.57.6 (Qwen3 support; torch nightly untouched, `pip check`
  clean). Added a `qwen3` branch to `load_gpt_model_and_tokenizer` (broadened the `qwen` match).
  Qwen3-8B-Base is public (no HF_TOKEN needed).

**Commands:**
- `python src/recreate_fig8.py --model_name <M> --folder <abstractive|ambiguous> --n_trials 200 --batch_size 256 --token_budget 48000`
- `python src/plot_fig8_recreation.py --which <abstractive|ambiguous>`

**Files:** NEW `src/recreate_fig8.py`, `src/plot_fig8_recreation.py`. MODIFIED
`src/utils/model_utils.py` (qwen3). Output `results/recreate_fig8/<model>/<task>.json` (summary) +
`<task>_responses.jsonl` (all responses); `figures/fig8_abstractive_2models.png`,
`figures/fig_ambiguous_2models.png`.

**Findings:**
- Both figures rendered: `figures/fig8_abstractive_2models.png` (paper's 25 panels; all 40 computed)
  and `figures/fig_ambiguous_2models.png` (18 panels). Curves rise then plateau, as in the paper.
- Mean acc@10-shot (exact-match): abstractive GPT-J 0.562 vs Qwen3-8B 0.604 (Qwen3 wins 26/40);
  ambiguous GPT-J 0.664 vs Qwen3-8B 0.682 (5 wins, 6 ties — ties are the saturated/near-0 tasks).
- Sanity vs prior work: GPT-J antonym 10-shot 0.56 (Stream G 0.529); country-capital 0.04→0.90;
  count_vowels/consonants stay near baseline (GPT-J zero-competence on counting, matches Stream G).
- Multi-token answers handled (e.g. "Kuala Lumpur"); 200×11=2200 stored responses per task/model.

**GOTCHA (see DECISIONS):** 4 task names live in BOTH `abstractive` and `ambiguous`
(count_consonants, count_vowels, identity, magnitude). Saving results as `<model>/<task>.json`
let the later (ambiguous) run overwrite the abstractive files. Fixed: output is now
`<model>/<folder>/<task>.json`.

**BUG FOUND + FIXED (see DECISIONS):** `round`/`truncate` initially scored ~0.00 — NOT a model
failure. `ICLDataset` int-indexing upcast the int gold to float (`4`→`4.0`) on float-input tasks,
and `normalize_answer` strips '.' (`'4.0'`→`'40'`) so correct `'4'` never matched. Fixed query
extraction to list-indexing; re-ran both tasks. Corrected: round 0.00→~0.77 (models have a mild
truncate bias on the decimal≥0.5 differentiator), truncate 0.00→~1.00. Audit of ALL tasks/models
found **zero** other formatting false-negatives; remaining low scores (rhyme 0.00, next_capital_letter
0.01, capitalize_last_letter ~0.10) are genuine model failures (verified by inspecting responses).

---

## 2026-06-15 — Stream G: cosine(activation, task-specific FV) heatmaps by token position × layer

**Owner:** Coordinator (tmux "cosine-fv-heatmap"). **Status:** DONE.

**What:** Same 31 token-position × 29 layer grid as the Stream C MSE study, but the per-cell metric
is the **raw cosine similarity** between each captured residual activation and that task's **own
task-specific FV** (`results/gptj_fv/<task>/...`), per-example mean, averaged over all 29 tasks.
Two heatmaps: all prompts, and prompts the model answered correctly.

- **Stage 1** `src/eval_scripts/compute_capture_prompt_correctness.py` (loads GPT-J): rebuilds the
  exact 170 ten-shot capture prompts per task (seed 42; reuses the capture's sampling +
  `create_prompt`), greedy-generates, parses at newline via `parse_generation`+`exact_match_score`
  (full answer, NOT first-token rank), writes `correctness/<task>.json` + summary. Cross-checks
  regenerated `query_source_index` against the capture `index.json` (assert) → prompts are identical.
  The cached `fs_results_layer_sweep.json` is NOT reusable (different partition + first-token only).
- **Stage 2** `src/eval_scripts/cosine_activation_to_task_fv.py` (no model): per-example cosine per
  cell, correct-mask from Stage 1, aggregate over tasks → `combined_metrics.csv`,
  `combined_cosine_heatmap_{all,correct}.png` (diverging, shared scale), `combined_summary.json`.

**Commands:**
- `HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 python src/eval_scripts/compute_capture_prompt_correctness.py --overwrite`
- `HF_HUB_OFFLINE=1 python src/eval_scripts/cosine_activation_to_task_fv.py`

**Files:** NEW `src/eval_scripts/compute_capture_prompt_correctness.py`,
`src/eval_scripts/cosine_activation_to_task_fv.py`. Output `results/cosine_activation_to_task_fv/`
(correctness/<task>.json, correctness_summary.json, combined_metrics.csv [899 rows],
combined_cosine_heatmap_{all,correct}.png, combined_summary.json).

**Findings:**
- **Correctness (10-shot, newline-cutoff exact-match):** mean acc 0.593 over 29 tasks. Low for
  char-manip (next_capital_letter 0.035, capitalize_last_letter 0.094); ~1.0 for easy maps
  (singular-plural 0.994, capitalize 0.976, present-past 0.971). antonym 0.529.
- **Cosine is all positive**, range [-0.01, 0.348]. Embedding L0 ≈ 0 (−0.010, ~orthogonal), clean
  **mid-layer bowl peak L10 (0.242 mean over positions)**, fades to 0.127 by L28 — same bowl as the
  Stream C MSE study (its min was L11). Best single cell **icl10/finaltok @ L10 = 0.348**.
- **Token-role banding dominates the picture:** mean cosine `last_prompt_token` 0.221 >
  `pre_label_token` 0.194 ≫ `first_label_token` 0.138 ≈ `last_label_token` 0.136. The query's final
  token and the pre-label (colon/"A:") position — i.e. where the model is about to emit the answer —
  are most aligned with the FV; the label tokens themselves much less so. Alignment also grows with
  more accumulated context (icl08–10/pre brighter than icl01–03/pre).
- **all ≈ correct:** restricting to model-correct prompts barely moves the aggregate cosine
  (correct−all mean +0.0002, median +0.0001, max|Δ| 0.0016). The two heatmaps are near-identical →
  at the task-aggregate level the residual stream points along the FV about as much on wrong answers
  as on right ones. (Per-task differences may be larger; not yet broken out.)

**Verification:** 899 cells (31 pos × 29 layers), n_tasks=29 every cell; independent recompute of
icl10/finaltok L11 matched CSV to 5 dp (0.32217 vs 0.32216); query_source_index join exact (no misses).
**Next (optional):** per-task cosine CSV / per-task all-vs-correct deltas; mean-centered variant.
**Blockers:** None.

---

## 2026-06-15 — Stream E follow-up: stable-rank + pairwise-cos per layer, all vs GPT-4-correct

**Owner:** Coordinator (tmux "oneshot-geometry"). **Status:** DONE.

**What:** `src/eval_scripts/plot_stable_rank_by_layer_byjudge.py`. Per layer, per token position, per task
pair: stable rank (Σσ²/σ₁², unit-normalized rows) and mean pairwise cosine of the stacked
difference-vector matrix D=act(f1)−act(f2). 2 figures (label, final), each [stable rank | mean cos],
with per-pair lines × {all (solid), GPT-4-correct-both-functions (dashed)}. Computed on the CORRECTED
graded capture (shared-input query). Overwrote `fig_compare_{label,final}.png` in BOTH
`results/oneshot_paired_analysis/` and `results/oneshot_paired_diff_geometry/` (the latter's old 4-pair
shared-OUTPUT versions superseded; its CSVs left as provenance). Series json in oneshot_paired_analysis.

**Stable rank, label token (all / correct):** antonym_synonym L9 4.85/3.58, L11 5.38/3.83;
next_number_prev_number L9 1.55/1.42, L11 1.61/1.47. Dips mid-layer (~L9) both pairs; numbers ≈ rank-1
(one dominant ±1 axis), words higher (~5–9). Mean-cos panel is the mirror (peaks mid-layer); correct >
all everywhere.

**CAVEAT (flagged):** stable rank depends on #rows W; correct subset has far fewer words (22 vs 544;
105 vs 198) → stable rank is biased DOWN for the correct line, so "correct < all" in the left panel is
partly a sample-size artifact. The mean-pairwise-cosine panel is W-robust and IS the trustworthy
all-vs-correct signal (correct more co-directional). Fix if needed: size-matched subsampling of all-words.
**Blockers:** None.

---

## 2026-06-15 — Stream E follow-up: pairwise diff-vector cosine histograms by task pair & judge

**Owner:** Coordinator (tmux "oneshot-geometry"). **Status:** DONE.

**What:** `src/eval_scripts/plot_pairwise_cos_hist_byjudge.py`. Histograms of pairwise cosine among the
per-word function-difference vectors D = act(f1) − act(f2) (unit-normalized), per TASK PAIR and per
token position. 4 panels: rows {antonym_synonym, next_number_prev_number} × cols {label, final query}.
Each overlays ALL words (grey) vs the GPT-4-correct subset (green) = words judged top-1 correct under
BOTH functions of the pair. Reads graded+tagged captures. Output
`results/oneshot_paired_analysis/fig_pairwise_diffcos_hist_L11_byjudge.png` (+ stats json).

**Mean pairwise cos (L11): all / correct (n_correct_both):**
- antonym_synonym label 0.152/0.203, final 0.116/0.174 (n=22)
- next_number_prev_number label 0.593/0.671, final 0.452/0.523 (n=105)

**FINDINGS:** (1) numbers ≫ words — the ±1 difference vectors are strongly co-directional (~0.5–0.6)
vs the high-dim word-meaning contrast (~0.12–0.15); echoes the stable-rank result (numbers ≈ rank-1).
(2) label > final in both pairs (function-difference more coherent at the demo label). (3) correct >
all in EVERY panel — correctly-handled words have more consistent function-difference directions
(caveat: antonym_synonym both-correct n=22, noisier). "correct" = both functions judge_top1 (can switch
to either/per-function). **Next:** geometry on judge-correct subsets. **Blockers:** None.

---

## 2026-06-15 — Stream E follow-up: cosine scatter (label vs final) colored by GPT-4 judge

**Owner:** Coordinator (tmux "oneshot-geometry"). **Status:** DONE.

**What:** Remade the cosine scatter on the CORRECTED graded capture (shared-input query, judge-tagged).
New `src/eval_scripts/plot_cos_label_vs_final_byjudge.py`. One point per shared word (n=544): x =
cos(antonym act, synonym act) at the demo LABEL token, y = same cos at the FINAL query token (both
positions already captured). Colored 4 ways by GPT-4 judge_top1 per function: both correct (22),
antonym-only (128), synonym-only (56), neither (338) — reconciles with judge totals (ant 150, syn 78).
Output overwrites `results/oneshot_paired_analysis/antonym_synonym/fig_cos_label_vs_final_L11_byjudge.png`
(+ points `.json`).

**FINDINGS (L11):** (1) ALL points above y=x — f1/f2 activations are MORE aligned at the final query
token (mean cos ~0.95) than at the label token (~0.89); the two functions converge by the query
position. (2) Judge correctness does NOT separate in this cosine space — correct points are
interspersed, per-category means near-identical (both .88/.95, neither .89/.96). So f1–f2 representational
similarity does not predict answer correctness. **Next:** same plot for numbers if wanted; geometry on
judge-correct subset. **Blockers:** None.

---

## 2026-06-15 — Stream E follow-up: SAME pipeline for next_number|prev_number

**Owner:** Coordinator (tmux "oneshot-geometry"). **Status:** DONE.

**What:** Replicated the corrected-capture + grade + GPT-4 judge + activation-tag pipeline for the
`next_number_prev_number` pair (number WORDS). Generalized the scripts:
- `capture_and_grade_oneshot_paired.py`: added the pair to TASK_PAIRS + `--allow_multitoken_label`
  (number words are mostly multi-token: only 26/198 shared outputs are single-token). With it, the
  demo-label pool = all 198 shared outputs and the source is the LAST label token (identical across
  f1/f2); `expected_src_id` now compares the LAST token of " "+w. Query from shared INPUT (200, gold
  under both). → 198 paired units, `results/oneshot_paired_graded/next_number_prev_number/` (2 shards,
  792 rows).
- `judge_oneshot_paired.py`: added `next_number`/`prev_number` judge prompts (answer denotes input±1 in
  ANY notation; SAME number doesn't count) + multi-word answer extraction (`one hundred one`, not just
  the first word). Greedy-generates the FULL answer (max_new_tokens=8).

**Results (n=198/task):**
| task | GPT-4 judge top-1 (full answer) | first-token top-1 | gold exact | copied |
|---|---|---|---|---|
| next_number | **0.818** (162/198) | 0.894 | 0.813 | 24 |
| prev_number | **0.621** (123/198) | 0.859 | 0.621 | 27 |

**KEY DIFFERENCE vs antonym/synonym:** here the GPT-4 judge is LOWER than first-token top-1 (inverted)
because **first-token OVER-counts for numbers** — gold "one hundred one" shares first token "one" with a
wrong "one hundred"/"one hundred two", and the copy-the-query cases (q=fifty, gold=fifty-one, model=fifty)
score first-token-correct but are wrong. So for numbers you MUST grade the full multi-token answer (judge
≈ gold-exact-match, 0.818/0.621); first-token is unreliable here. next > prev (GPT-J better at +1 than −1).

**Tagged** 792/792 activation rows + 396 grading rows with `judge_top1` (via
`tag_oneshot_activations_judge.py`). Activations filter by `top1/2/3` (first-token) and `judge_top1`
(GPT-4 full-answer). Both pairs now share the same on-disk schema. **Next:** geometry on judge-correct
subsets. **Blockers:** None.

---

## 2026-06-15 — Stream E follow-up: GPT-4-judged top-1 (antonym+synonym) + activation tagging

**Owner:** Coordinator (tmux "oneshot-geometry"). **Status:** DONE.

**What:** First-token exact-match undercounts open-ended tasks (many valid antonyms/synonyms). Judged
GPT-J's actual greedy top-1 answer with GPT-4.1 for BOTH tasks. `src/eval_scripts/judge_oneshot_paired.py`
(supersedes the earlier synonym-only script; one model load, `--function_tasks antonym synonym`):
rebuilds the EXACT prompts from the corrected capture (`results/oneshot_paired_graded/antonym_synonym/
grading.json` — shared-input query), greedy-generates the top-1 word, asks GPT-4.1 if it's a valid
antonym/synonym. **Both judge prompts explicitly state the SAME WORD (or cap/plural/inflection) does
NOT count** (per user); also reject the opposite relation, non-words, topical associates. Verdict key
`"correct"`. Key from `/proc/1/environ`, batched 50/request.

**Results (n=544/task):**
| task | GPT-4 judge top-1 | first-token top-1 | gold exact | copied query |
|---|---|---|---|---|
| antonym | **0.276** (150/544) | 0.232 | 0.232 | 263 (0.48) |
| synonym | **0.143** (78/544) | 0.066 | 0.063 | 339 (0.62) |

**FINDING:** judging the real answer lifts both (antonym +.044, synonym >2×: .143 vs .066) — first-token
scoring undercounts, much more so for synonym (many valid alternatives). The dominant failure is
**copying the query word** (antonym 48%, synonym 62%) — in 1-shot with a single `q→w` demo GPT-J largely
echoes the query; all copies judged false (per same-word rule). Judge accepts non-gold answers
(synonym fearless→brave; gold "daring"). Outputs `results/oneshot_{antonym,synonym}_judge/judged_results.json`.

**ACTIVATION TAGGING:** `src/eval_scripts/tag_oneshot_activations_judge.py` stamped a `judge_top1` bool
into EVERY activation row's metadata (match key (function_task, output_word, query_word); both source &
target roles) in `results/oneshot_paired_graded/antonym_synonym/shard_*.pt` AND grading.json — 2176/2176
rows tagged. So activations now filter three ways: `top1/2/3` (first-token rank) and `judge_top1` (GPT-4
semantic). judge_top1 True rows: antonym 300 (=150 prompts×2 roles), synonym 156 (=78×2).
**Next:** geometry on the new capture; compare stable-rank/cosine on judge_top1-correct subsets. **Blockers:** None.

---

## 2026-06-15 — Stream D: GPT-4-judged rhyme eval scaled to ALL 200 pairs

**Owner:** Coordinator. **Status:** DONE.

**What:** Re-ran the GPT-4.1-judged rhyme eval (Stream C, was n=42 test split) over the FULL
200-pair dataset. Added `--all_queries` mode to `src/eval_scripts/judge_rhyme_generations.py`:
each of the 200 examples is the query in turn, its 10-shot ICL demos drawn randomly leave-one-out
from the other 199 (disjoint, no leakage). Also chunked the judge into batches (`--judge_batch_size`,
default 50) so large n doesn't overflow one request. Default Stream-C path unchanged.

**Command:**
`python src/eval_scripts/judge_rhyme_generations.py --all_queries --output_dir results/rhyme_judge_eval_all200 --judge_batch_size 50`

**Result (n=200):** judged-rhyme accuracy **5/200 = 0.025**; gold exact-match **2/200 = 0.010**;
copied the input verbatim **94/200 = 0.47**. The 5 judge-accepted rhymes: array→delay, ban→man,
catch→attach, command→band, cry→eye (only the two exact-match ones also match gold). Confirms the
n=42 finding at full scale: GPT-J essentially cannot rhyme via 10-shot ICL — ~half the time it just
echoes the query word; near-zero true rhymes. Judge ≈ gold (both ~0–2.5%), so the gold-token metric
was NOT understating competence.

**Files:** added flags to `judge_rhyme_generations.py`; artifacts `results/rhyme_judge_eval_all200/`
(generations.json, judged_results.json, run.log). Deleted the superseded n=42 `results/rhyme_judge_eval/`.

---

## 2026-06-15 — Stream E follow-up: CORRECTED paired 1-shot capture (shared-INPUT query) + grading

**Owner:** Coordinator (tmux "oneshot-geometry"). **Status:** DONE (activations + grading).

**Why the redo:** the original Stream-E capture (`capture_oneshot_paired.py`) sampled the QUERY from
the shared OUTPUT-word pool (just a token to read activations from). That makes the query ill-posed —
only 464/389 of 544 queries even had a gold antonym/synonym (50 had neither; an earlier scoring pass,
`score_oneshot_paired_prompts.py`, got ant .280/.431, syn .113/.270 over that biased 853-prompt
subset). CORRECTED: query is now drawn from the shared INPUT space (words that are a valid input under
BOTH tasks → gold defined under both). Demo label `w` still from shared OUTPUT (paired design unchanged).

**Shared-space sizes (antonym/synonym):** shared INPUT = **1224** (all single-token), shared OUTPUT =
555 (544 single-token, used for the demo label). Input vocab > output vocab for both tasks.

**New script** `src/eval_scripts/capture_and_grade_oneshot_paired.py` (supersedes
`capture_oneshot_paired.py` for graded runs): captures source(demo-label)+target(final-query)
activations AND grades each prompt in-line (one extra forward; **first-token rank**, top-k = rank<k —
per user, first-token only, no multi-token gen). The grade (`gold_first_tok_rank`, `top1/2/3`) is
written into EVERY activation row's metadata → activations filter directly to top-1/2/3-correct.

**DELETED old output-space-query run:** `results/oneshot_paired/antonym_synonym/`,
`results/oneshot_paired_analysis/antonym_synonym/` (+ its stale top-level `fig_*.png`),
`results/oneshot_paired_scored/antonym_synonym/`. `landmark_park` + other pairs untouched.

**Outputs** `results/oneshot_paired_graded/antonym_synonym/`: 6 shards, **2176 rows** (544×2 funcs×2
roles), `index.json`, `grading.json` (per-prompt + model top-1 token), `scores.json`.

**Results (1-shot, first-token, all 544/task now scorable):**
| task | n | top1 | top2 | top3 |
|---|---|---|---|---|
| antonym | 544 | 0.232 | 0.324 | 0.401 |
| synonym | 544 | 0.066 | 0.195 | 0.263 |

LOWER than the old output-space-query slice (ant .280/.431, syn .113/.270) — expected: that slice was
an easier/biased subset (queries that happen to be task outputs, gold-having only). These are the
honest well-posed-query numbers. antonym ≫ synonym persists; synonym ~triples top1→top2 (many valid
synonyms, single gold first-token misses). **Next:** re-run geometry (`analyze_oneshot_geometry.py`)
on the new capture; optionally compare stable-rank/cosine when filtered to top-1-correct prompts.
**Blockers:** None.

---

## 2026-06-15 — Stream G: task-specific + train-pooled FVs & steering for 4 ambiguous tasks

**Owner:** Coordinator (tmux "magnitude/identity FV"). **Status:** RUNNING (pipeline launched).

**What:** For the 4 chosen ambiguous tasks (`magnitude`, `identity`, `count_vowels`,
`count_consonants`): (1) task-specific FVs (own CIE top-10) + (2) train-pooled-head FVs at top-10/20/40
(reuse `results/multitask_aie_heads/multitask_top_aie_heads.pt` = original 20-train pool), then steering
eval (per-layer zero-shot + 10-shot-shuffled) for all.

**Setup:** symlinked the 4 JSONs into `dataset_files/abstractive/` (loader hardcodes abstractive/
extractive; same precedent as paired_tasks). NEW manifest `task_splits/ambiguous_4.json`. NEW driver
`src/eval_scripts/run_ambiguous_fv_pipeline.sh`. Installed `bitsandbytes` (intervention_utils imports
it at module top even though the 4-bit path is unused in fp16).

**GPU/parallelism:** single 24GB card fits ONE GPT-J fp16 (~12GB) → GPU stages serialize. Pipeline:
stage1 = compute_function_vectors (GPU, all 4 tasks, batch 16); stage2 = 3× top-N builds on **CPU in
parallel** (out_proj sums, overlap GPU); stage3 = evaluate_heldout ×{10,20,40} (GPU serial; also
re-evals the task-specific FV each run). Logs in `results/_ambiguous_logs/`.

**CAVEAT (expected):** count_vowels/count_consonants have low ICL accuracy (~0.3 first-token) so the
correct-ICL filter leaves few prompts → low-N/noisy FVs (cf. rhyme/geo low-N). magnitude/identity are
high-acc. Outputs: `results/gptj_fv/<task>/` (task-specific), `results/gptj_fv_multitask_top{10,20,40}_
ambiguous/` (train-pooled), `results/heldout_ambiguous_eval_top{10,20,40}/` (steering).

**DONE (with OOM hiccup).** Stage-1 task-specific FVs built for all 4 (count_* survived the filter:
n_filtered_test magnitude/identity=21, count_vowels/count_consonants=**6** → low-N as predicted).
OOM during the run: the 3 parallel CPU model-loads (~12GB RAM each) + GPU steering load tripped the
OOM-killer (killed steer n=10 + 2 builds). **FIX: re-ran the missing pieces SEQUENTIALLY** (no parallel
model loads) — `results/_ambiguous_logs/rerun_missing.sh`. All complete now: steering top-10/20/40 (4
tasks each), persisted train-pooled FVs top-10/20/40 (4 each), task-specific FVs (4).
**Lesson:** GPT-J CPU loads are ~12GB RAM each; don't run 3 in parallel alongside a GPU load on this box.

**STEERING RESULTS (best-layer zero-shot FV top-1; 10-shot-shuffled in parens) —
`summarize_ambiguous_steering.py`:**

| task | n_test | task-specific | train top10 | top20 | top40 |
|---|---|---|---|---|---|
| magnitude | 21 | 0.62@L12 (0.95) | 0.57@L0 (0.90) | 0.57@L0 (0.95) | 0.57@L0 (0.95) |
| identity | 21 | 1.00@L0 (1.00) | 1.00@L0 (1.00) | 1.00@L0 (1.00) | 1.00@L0 (1.00) |
| count_vowels | 6 | 0.00 (0.83) | 0.00 (0.67) | 0.00 (0.67) | 0.00 (0.83) |
| count_consonants | 6 | 0.17@L14 (0.83) | 0.00 (0.83) | 0.00 (0.83) | 0.00 (0.83) |

**CAVEATS / FINDINGS:**
- **By-layer curves are nearly FLAT** (magnitude ≈0.57=12/21 at every layer, identity 1.0 everywhere),
  argmax often at **L0** (large-norm embedding artifact). → the zero-shot numbers largely reflect BASE
  zero-shot behavior, NOT a clean FV effect. Need the **no-FV baseline** to isolate the causal lift
  (evaluate_heldout doesn't compute it).
- **Zero-shot FV steering is weak/absent for these tasks**: magnitude flat ~0.57, count ≈0. Only the
  10-shot-shuffled (context present) numbers are high (0.83–1.0).
- **task-specific ≈ train-pooled, and top-10≈20≈40** — adding heads barely changes these (contrast the
  29-task finding where n40 helped zero-shot). identity is trivially 1.0 (copy).
- **count_* are low-N (n_test=6) + ~0 zero-shot** — competence/low-N caveat confirmed.

**Next:** run `evaluate_function_vector.py --compute_baseline` for the 4 tasks to get the no-FV
zero-shot/10-shot baselines → report the FV's causal steering DELTA (the meaningful metric).
**Blockers:** None.

### 2026-06-15 — Stream G2: CONSTRAINED FV re-extraction (overlap contamination fix)

**Why:** user flagged (confirmed) that the FV extraction AND steering eval were contaminated by the
OVERLAP region. The flat magnitude zero-shot steering 0.57 == the positive (overlap) fraction of the
test set (12/21) — the model just copies, gets overlap positives right regardless of the FV. On
overlap queries magnitude≡identity, so they tell us nothing and dilute the FV.

**Fix (recipe agreed w/ user):** every 10-shot extraction prompt = **5 overlap + 5 differentiator
demos** (order shuffled), **query = differentiator**, keep only **correct** responses. Same for the
steering eval queries (differentiators only — TODO).

**Implemented:**
- Expanded the 4 datasets to **150 overlap + 150 differ** (was 50/50) so train+valid differentiator
  pools are healthy (BIG=150 in `create_ambiguous_datasets.py`). NB the disambiguation-eval results
  above used the 50/50 version.
- Additive `prompt_sampler` hook on `get_mean_head_activations` (extract_utils) and
  `compute_indirect_effect` — default None = original 29-task behavior untouched.
- NEW `src/eval_scripts/compute_ambiguous_constrained_fv.py`: builds the 5+5/differ-query sampler,
  filters valid-differ queries to model-correct, runs mean-act + CIE via the hook, saves task-specific
  FV to `results/gptj_fv_ambiguous_constrained/<task>/` (mirrors gptj_fv layout).
- **GOTCHA fixed:** script must call `torch.set_grad_enabled(False)` (stock compute_function_vectors
  does it in main) — without it the extraction forwards retain the autograd graph → OOM on the 24GB card.

**Smoke (magnitude):** train overlap=102 differ=108, valid differ=12 → 12/12 correct query pool; FV
norm 50.8; top heads (12,10)(15,5)(8,1)(13,13)(20,0) = canonical FV heads (vs the contaminated run).
Full 4-task extraction running.

**DONE — constrained FVs + differentiator-restricted steering.** Built train-pooled FVs (top10/20/40)
from constrained mean acts (`results/gptj_fv_ambiguous_constrained_top{10,20,40}/`); added additive
`--restrict_differentiator`/`--partners` to evaluate_heldout; steering →
`results/heldout_constrained_differ_top{10,20,40}/`, plots →
`results/heldout_constrained_differ_plots/`. Driver `run_constrained_steering.sh`. Differentiator test
queries: magnitude/identity **30**, count_vowels/consonants **12/10**.

**RESULTS — best zero-shot FV-steering top-1 over L3–27 (excl. L0–2 embedding artifact);
10-shot-shuffled in parens. DIFFERENTIATOR queries only:**

| task | nq | task-specific | train10 | train20 | train40 |
|---|---|---|---|---|---|
| magnitude | 30 | **1.00@L14 (1.00)** | 0.20@L12 (0.83) | 0.17@L12 (0.97) | 0.23@L12 (1.00) |
| identity | 30 | 1.00@L3 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| count_vowels | 12 | 0.00 (0.67) | 0.00 (0.67) | 0.00 (0.67) | 0.00 (0.67) |
| count_consonants | 10 | 0.10@L27 (0.70) | 0.00 (0.70) | 0.00 (0.60) | 0.00 (0.60) |

**FINDINGS (the contamination fix changes the conclusions):**
- **magnitude task-specific FV genuinely steers zero-shot** — clean mid-layer plateau **~0.9–1.0 across
  L7–14, dead by L16** (classic FV causal window). The old flat 0.57 was purely overlap-copy; on
  differentiator queries the real effect (≈1.0) is now visible.
- **task-specific ≫ train-pooled for magnitude zero-shot** (1.00 vs ~0.20) — the generic 20-train-task
  pooled heads do NOT encode the magnitude-distinguishing computation; adding heads (10→40) barely
  helps (0.20→0.23). 10-shot-shuffled is high for all (context carries it). Contrast the 29-task result
  where train-pooled ≈ task-specific.
- **identity = trivial-copy confound** — 1.0 everywhere for BOTH FV types (copying a negative is trivial
  zero-shot), so it's not evidence of FV efficacy.
- **count_* = competence failure** — 0.00 zero-shot steering regardless of FV/heads; only weak
  context-present (fs 0.6–0.7) on tiny n (10–12). Constrained extraction gave count FVs tiny CIE
  (~0.003) and 4–5-query pools → no usable steering signal.

**Takeaway:** of the two carried-forward pairs, **magnitude/identity is the one with a real, strong,
cleanly-steerable distinguishing FV (task-specific, mid-layer)**; count vowels/consonants is
competence-limited and does not yield a steerable FV. **Blockers:** None.

---

## 2026-06-14 — Stream F: `ambiguous` task-disambiguation datasets (4 pairs)

**Owner:** Coordinator (tmux "magnitude/identity FV"). **Status:** DONE — all 4 brainstormed pairs generated.

**What:** New dataset family for the task-DISAMBIGUATION study (3 ambiguous ICL demos + 1
differentiating demo + 1 test query). Each pair (f1,f2) AGREES on an overlap region and
DISAGREES on a differentiator region; the two task files share the SAME input set, overlap
entries are byte-identical (input AND output), differ entries share input / split output.

**Pairs (`dataset_files/ambiguous/`):**
- `magnitude.json | identity.json` — n→|n| vs n→n. Overlap = non-neg ints (1..50); differ =
  negatives (-1..-50). **50/50.** Digits ("-5"); switch to words for next/prev_number consistency.
- `past_tense.json | past_participle.json` — verb→past vs verb→participle. Overlap = 50 regular
  verbs (past==participle); differ = 50 irregulars (ate/eaten…). **50/50.** Strongest pair (rich O+D,
  known priors).
- `first_letter.json | last_letter.json` — word→word[0] vs word→word[-1]. Overlap = 50 words with
  first==last char; differ = 50 others. **50/50.** Vocab sourced from repo synonym+antonym inputs
  (4054 words; 184 first==last available), seed=0.
- `capital_city.json | largest_city.json` — country→capital vs →largest. Overlap = 50 (capital IS
  largest); differ = **35** (capital≠largest). **50/35, NOT 50/50** — only ~35 real capital≠largest
  countries exist worldwide and only ~20 are GPT-J-plausible (Naypyidaw/Gitega/Ngerulmud etc. are
  low-freq). RECALL CAVEAT, cf. the geo/element low-N pairs. Trim overlap→35 for a balanced 35/35,
  or restrict differ to the ~20 famous head of the list, in the eval.

**Files:** NEW `dataset_files/generate/create_ambiguous_datasets.py` (one generator, all 4 pairs);
NEW folder `dataset_files/ambiguous/` (8 JSONs).

**Verified:** all pairs share matched inputs; overlap/differ counts as above; differ outputs split
correctly (eat→ate/eaten; vulgar→v/r; United States→Washington/New York).

**3+1+1 EVAL DONE (n=100/task, cross-prompt batched greedy, token-level exact match):** new
`src/eval_scripts/eval_ambiguous_disambiguation.py` (batched generate; reuses
`word_pairs_to_prompt_data`/`create_prompt`; `matches_partner` diagnostic = model produced the OTHER
function's answer). Each prompt = 3 overlap demos + 1 differentiator demo (task's output) + 1
differentiator query (seed 42, paired draws). Results → `results/ambiguous_disambiguation/eval_summary.json`.

| task | acc | partner | neither |
|---|---|---|---|
| magnitude | 0.98 | 0.02 | 0.00 |
| identity | 1.00 | 0.00 | 0.00 |
| past_tense | 0.93 | 0.03 | 0.04 |
| past_participle | **0.36** | 0.60 | 0.04 |
| first_letter | 0.97 | 0.01 | 0.02 |
| last_letter | **0.04** | 0.89 | 0.07 |
| capital_city | 0.57 | 0.37 | 0.06 |
| largest_city | 0.47 | 0.45 | 0.08 |

**FINDING — strong within-pair PRIOR-BIAS asymmetry; one disambiguating demo only redirects the
model in the "easy" direction.** magnitude/identity ~perfect both ways. But the other pairs are
lopsided: the model nails the *prior-aligned* task (past_tense .93, first_letter .97) and largely
ignores the 4th demo for the *anti-prior* task, instead emitting the prior answer (past_participle
.36 with partner .60; last_letter .04 with partner .89). capital/largest both ~chance-ish and noisy
(recall-limited differ tail + capital↔largest confusion). Spot-checks confirm scoring is correct
(last_letter gold 'r' → pred 'f' = first letter; past_participle gold 'shown' → pred 'showed').

**3+2+1 EVAL (n_diff_demos=2; `--n_diff_demos 2`):** → `eval_summary_3plus2plus1.json`. A second
disambiguating demo helps the anti-prior side only partially, and not at all for the strongest prior:

| task | 3+1+1 acc | 3+2+1 acc | partner (3+2+1) |
|---|---|---|---|
| magnitude | 0.98 | 1.00 | 0.00 |
| identity | 1.00 | 1.00 | 0.00 |
| past_tense | 0.93 | 0.98 | 0.02 |
| past_participle | 0.36 | **0.51** | 0.49 |
| first_letter | 0.97 | 0.97 | 0.00 |
| last_letter | 0.04 | **0.02** | 0.88 |
| capital_city | 0.57 | 0.60 | 0.37 |
| largest_city | 0.47 | 0.46 | 0.47 |

**FINDING (k=2 vs k=1 differentiator demos):** the prior-aligned tasks saturate (past_tense→.98,
magnitude→1.0). past_participle improves +.15 (.36→.51, now ~coin-flip vs its prior). **last_letter
does NOT budge (.04→.02, partner .88)** — two demos still can't override the first-letter prior.
capital/largest essentially flat (recall-bound). So a 2nd disambiguating example helps moderate
priors but not the dominant first-letter one; harness now parametrized (`--n_shared_demos`,
`--n_diff_demos`) for a k-sweep.

**3 MORE PAIRS ADDED (2026-06-14, all 50/50):** `round|truncate` (1-dp decimals; overlap frac<.5,
differ frac≥.5), `first_digit|last_digit` (overlap first==last digit; numeric analog of the
first/last-letter prior diagnostic), `american|british` (input=US spelling, american=identity /
british=US→UK convert; overlap=invariant words, differ=50 US/UK variants). All wired into
`eval_ambiguous_disambiguation.py` PAIRS. Not yet eval'd.

**+ reverse|identity_word (50/50):** overlap = 50 palindromes (reverse==identity), differ = 50
ordinary 3–5 letter words (`prose→esorp` vs `prose`). Partner named `identity_word.json` to avoid
clashing with the numeric `identity.json`. Wired into eval PAIRS. Prediction: reversal is hard for
BPE → expect a *competence* failure (high `neither`) not a *prior* failure (high `partner`).

**+ count_vowels|count_consonants (50/50):** overlap = 50 words with #vowels==#consonants
(`able` 2/2), differ = 50 unequal (`silver` 2/4). Vowels=aeiou. Counting task → expect
competence-limited (high `neither`), noisy (small int output space); overlap output also = len/2 so
consistent with several counting rules. Wired into eval PAIRS. (8 pairs total now.)

**3+1+1 EVAL — ALL 8 PAIRS (2026-06-14).** Two scorings, n=100/task, cross-prompt batched:
exact-match (generation) → `eval_summary_3plus1plus1_all.json`; first-answer-token top-1/top-2
(single forward pass, `--scoring topk`) → `eval_topk_3plus1plus1.json`. Added `batched_topk`/
`score_topk` + `--scoring {topk,exact}` to the harness.

| task | top1 | top2 | partner@1 |
|---|---|---|---|
| magnitude | 0.98 | 0.99 | 0.02 |
| identity | 1.00 | 1.00 | 0.00 |
| past_tense | 0.93 | 0.95 | 0.05 |
| past_participle | 0.38 | **0.92** | 0.60 |
| first_letter | 0.97 | 0.99 | 0.01 |
| last_letter | 0.04 | **0.18** | 0.89 |
| capital_city | 0.60 | 0.85 | 0.38 |
| largest_city | 0.48 | 0.76 | 0.48 |
| round | 0.05 | **0.93** | 0.95 |
| truncate | 1.00 | 1.00 | 0.00 |
| first_digit | 0.77 | 0.83 | 0.06 |
| last_digit | 0.16 | **0.37** | 0.51 |
| american | 1.00 | 1.00 | 0.15 |
| british | 0.43 | **0.92** | 0.69 |
| reverse | 0.00 | 0.01 | 0.66 |
| identity_word | 1.00 | 1.00 | 0.00 |
| count_vowels | 0.31 | 0.57 | 0.11 |
| count_consonants | 0.29 | 0.52 | 0.22 |

**FINDINGS (top-2 separates the failure modes):**
- **magnitude/identity** still the only pair solid in BOTH directions (.98/1.00).
- **PRIOR-BIAS pairs (anti-prior side rank-2, not absent):** past_participle .38→**.92**, round .05→**.93**,
  british .43→**.92** all jump at top-2 → the disambiguating demo IS registered (gold is the #2 token)
  but the prior owns #1 (partner@1 .60/.95/.69). The model "knows" the right answer, prior overrides.
- **STRONG-PRIOR pairs (anti-prior side genuinely suppressed):** last_letter .04→.18, last_digit
  .16→.37 — gold not even rank 2. First-position prior is real for BOTH letters and digits, weaker
  for digits (last_digit top2 .37 vs last_letter .18).
- **COMPETENCE failure:** reverse .00/.01 (can't emit the reversed string at all — partner .66, neither
  high in exact-match); count_vowels/consonants ~.3/.55 (counting hard) — distinct from prior bias.
- **RECALL:** capital/largest ~.5–.6 top1, .76–.85 top2 (noisy, model-known-entity bound).
- **american top1=1.0 but partner@1=.15** (identity side occasionally emits the UK form — interesting).

**Next:** 3+2+1 top-k; k-sweep n_diff_demos 0..5 (k=0 control); non-numeric symmetric pairs
(AND|OR, earlier|later, alphabetical-order). **Blockers:** None.

---

## 2026-06-14 — Stream A: train_selected FVs at top-20 and top-40 heads

**Owner:** Coordinator (tmux "FV top20/40"). **Status:** DONE — 29 FVs each, both runs + organized + metadata.

**What:** Added `train_selected` FV runs at n_top_heads=20 and 40 for all 29 tasks (previously only
top-10). No CIE recompute — the train-pooled head artifact `results/multitask_aie_heads/
multitask_top_aie_heads.pt` already stores top-40, mean head activations cached in `results/gptj_fv/`.
Just re-summed `out_proj(mean_act[L,H,-1])` over the top-20 / top-40 heads.

**Commands run (HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1):**
- `compute_all_task_fvs_from_multitask_heads.py --n_top_heads 20 --output_root results/gptj_fv_multitask_top20` → 29 FVs.
- `compute_all_task_fvs_from_multitask_heads.py --n_top_heads 40 --output_root results/gptj_fv_multitask_top40` → 29 FVs.
- `write_fv_head_metadata.py --methods train_selected_top20 --n_top_heads 20` (and `_top40 --n_top_heads 40`).

**Files changed:**
- NEW `results/gptj_fv_multitask_top{20,40}/<task>/<task>_function_vector.pt` (29 each) + `fv_manifest.json`.
- NEW organized views `results/function_vectors/gpt-j/train_selected_top{20,40}/` (FV symlinks +
  `heads.pt`/`heads_metadata.json` → multitask_aie_heads + per-task `selected_heads.json`).
- `write_fv_head_metadata.py`: added `train_selected_top20/40` to `POOL_DESC`.

**Findings:** All 87 FVs (29×3 n) sane. Norms grow with head count: top-10 ~30–47, top-20 ~34–54,
top-40 ~41–66 (more heads add more head-out-proj terms to the sum). The first-10 / first-20 heads are
prefixes of the top-40 list (same ranking truncated), so top-10 ⊂ top-20 ⊂ top-40 head sets.

**Next:** optional — held-out steering eval (`evaluate_heldout_multitask_head_fvs.py`) to see whether
n=20/40 transfers better than n=10. **Blockers:** None.

**UPDATE 2026-06-15 (steering eval done):** ran `evaluate_heldout_multitask_head_fvs.py --n_top_heads
{20,40} --overwrite` (9 held-out test tasks, seed 42, full 28-layer sweep, zero-shot+FV and
10-shot-shuffled+FV; cached `fs_results_layer_sweep.json` filters → no baseline recompute). Logs
`results/_heldout_steering_logs/`.

**Consolidated into ONE folder** `results/heldout_multitask_head_eval_nheads/`:
- `top20/`, `top40/` — the full per-task result JSONs (moved from the old `_top20`/`_top40` dirs).
- `<task>_effectiveness_by_layer_nheads.png` (9) + `AGGREGATE_…png` — combined plots overlaying
  multitask n=10/20/40 + task-specific reference, both conditions. (n=10 data still lives in the
  pre-existing baseline `results/heldout_multitask_head_eval/`.)
- `nheads_comparison.json` — best-layer comparison table.
- NEW plotting script `src/eval_scripts/plot_nheads_steering_comparison.py` (pure; re-renders from
  the JSONs). Redundant per-folder `*_effectiveness_by_layer.png` were deleted (superseded).

**FINDING — more heads helps ZERO-SHOT steering, neutral/slightly-hurts with ICL context:**
- best-layer **zero-shot+FV** mean top-1: n10 **0.376** → n20 **0.381** → n40 **0.446** (task-specific 0.483).
  More heads recovers most of the gap to task-specific. Big n40 winners: capitalize 0.75→0.96,
  capitalize_first_letter 0.70→0.95; product-company recovers at n40 (0.22→0.24 after dipping to 0.12 @n20).
  word_length stays 0.00 at every n (FV can't drive it zero-shot).
- best-layer **10-shot-shuffled+FV** mean: n10 **0.796** → n20 0.785 → n40 0.780 (task-specific 0.812).
  Flat/slightly down — with real ICL context already present, extra heads add nothing.
- **Best zero-shot layer drifts EARLIER as n grows** for some tasks (capitalize n10 L13→n40 L1;
  country-currency L11→L1; product-company L11→L0). The larger-norm 40-head FV scores best injected
  very early — sanity-check before over-reading those cells; the mid-layer (L8–13) optimum is stable
  for antonym/synonym/capitalize_first_letter across all n.
- Takeaway: for zero-shot steering, n=40 train-pooled heads clearly beats n=10; for ICL-context
  steering, n=10 is already saturated.

---

## 2026-06-12 — Stream C: GPT-4-judged rhyme eval — GPT-J truly cannot rhyme (0/42)

**What:** `src/eval_scripts/judge_rhyme_generations.py` — rebuilds the same 10-shot test prompts
(seed 42), greedy-generates GPT-J's actual answer word, and has GPT-4.1 judge whether it truly
rhymes (strict: final stressed vowel + coda match; identical word/non-word/inflection = false).
Tests whether the 0.024 first-token score understates competence via valid alternative rhymes.

**Result: judged accuracy 0/42 = 0.000** (gold exact-match 1/42; the judge rejects even that one —
`apply→ally` isn't a true rhyme → the dataset's CMUdict gold pairs are themselves loose).
Failure modes: copies the input (17/42), semantic associates (black→white, birth→death),
orthographic neighbors (bite→bit, defense→defence). Consistent with BPE models lacking phonology.
**Implication:** rhyme is a genuine zero-competence task for GPT-J — keep only as a known-zero
control, or drop. The OpenAI key is available from the container env
(`/proc/1/environ` → `OPENAI_API_KEY`; pod env vars don't reach interactive shells).
Artifacts: `results/rhyme_judge_eval/`. The judge harness is reusable for lenient scoring of other
tasks (e.g. "is X actually east of Y" for the geo pair).

---

## 2026-06-12 — Stream C (round 2): FVs for 4 MORE paired tasks (east/west_neighbor, next_in_period/group) × 3 methods

**Owner:** Coordinator (tmux "FV derivation"). **Status:** DONE — 12 FVs built + verified; LOW-N CAVEAT below.

**Same recipe as round 1 (below)**: symlinked into `dataset_files/abstractive/`; NEW
`task_splits/paired_tasks_7.json` (supersedes _3); 4 parallel tmux windows (canonical → varicl
stage-1 shard, num_shards=4); stage-2 builders with `fv_manifest_paired2.json`; organized symlinks
+ `selected_heads.json` for all 3 methods. The auto no-filter retry **never triggered** — all 4
tasks pass the ICL filter with nonzero correct counts. Norms sane (24.4–43.9).

**IMPORTANT LOW-N CAVEAT — GPT-J is weak at all 4 tasks, so the correct-only averaging rests on
very few queries:**
- canonical 10-shot ICL-correct: east_neighbor **5/51**, west_neighbor **7/51**,
  next_in_period **2/14**, next_in_group **5/14** (vs next/prev_number 42/42).
- varicl (1–10 shots, valid split): east **2**, west **3**, next_in_period **1**, next_in_group **1**
  correct queries → those varicl FVs average over 1–3 prompts. Treat all four tasks' FVs (esp.
  varicl) as high-variance; consider `--no_filter_to_correct_icl` variants if robustness matters
  (rhyme precedent), or larger datasets / more valid examples.
- Filter kept ON for methodological consistency with the 29 originals (counts were nonzero).

Logs: `results/_paired_fv_logs/{canonical,varicl}_<task>.log`, `build_{varicl,train_selected}2.log`.

---

## 2026-06-12 — Stream B: periodic-table 2D-grid neighbour pair (next_in_period | next_in_group)

**Status:** COMPLETE (datasets generated).

**What:** Chemistry analog of the geography pair — the periodic table is a 2D grid the
model knows. `next_in_period` = element → element to its RIGHT (same row, atomic_number+1);
`next_in_group` = element → element BELOW it (same column, next period down). Both
element→element → matched input AND output marginals; a lone element name reveals nothing
about which grid direction. Generator
`dataset_files/generate/create_periodic_table_neighbor_datasets.py` (data: `mendeleev`, offline).

**Outputs:** `dataset_files/paired_tasks/{next_in_period,next_in_group}.json`, **66 pairs each**,
IDENTICAL input set (66 main-group + transition-block elements that have BOTH a right and a
below neighbour) → matched input marginal. f-block (lanthanides/actinides) excluded (no
group_id in source; also the elements the model knows worst). 0 self-pairs. Element names
mostly multi-token (only ~14 single-token) → allowed, first/last label-token scoring like
countries/numbers. Sane (Nitrogen R→Oxygen/B→Phosphorus; Selenium R→Bromine/B→Tellurium).

**Deps:** installed `mendeleev`.

**Intended pair:** next_in_period | next_in_group. (Skipped the US-presidents ordinal idea —
~46 too small.)

**Next:** capture paired activations (add `next_in_period_next_in_group` to
capture_oneshot_paired_tasks.py TASK_PAIRS) + geometry analysis.

**Blockers:** None.

---

## 2026-06-12 — Stream B: geography spatial-neighbour pair (east_neighbor | west_neighbor)

**Status:** COMPLETE (datasets generated).

**What:** New geographic task pair under the matched-marginal rule. Most geography leaks
(attribute pairs → disjoint output pools; landmark/park → input has "Park" in 747/749);
the only legal family is place→place SPATIAL relations over one shared pool — the geo
analog of next/prev_number. Built `east_neighbor`/`west_neighbor`: country → nearest
country in the E/W bearing quadrant, from most-populous-city coords (geonamescache,
deterministic, offline). Generator `dataset_files/generate/create_geography_neighbor_datasets.py`.

**Outputs:** `dataset_files/paired_tasks/{east_neighbor,west_neighbor}.json`, 244 pairs
each, IDENTICAL input set (244 countries with both an E and W neighbour) → matched input
marginal; both output countries → matched output marginal. 0 input==output collisions.
Multi-token country names allowed (first/last label-token scoring). Sane (Afghanistan
E→Pakistan/W→Turkmenistan; Netherlands E→Germany/W→UK); approximate for islands.

**Deps installed:** pycountry-convert, geonamescache, nltk+WordNet (WordNet unused so far —
semantic co-hyponym/hypernym/meronym pairs remain a validated option if wanted).

**Intended pair:** east_neighbor | west_neighbor (and could add north/south).

**Next:** capture paired activations for this pair (capture_oneshot_paired_tasks.py — add
`east_neighbor_west_neighbor` to TASK_PAIRS) and/or run the geometry analysis.

**Blockers:** None.

---

## 2026-06-12 — Stream C: derive FVs for 3 NEW paired tasks (rhyme, next_number, prev_number) × 3 methods

**Owner:** Coordinator (this session, tmux "FV derivation"). **Status:** DONE — all 9 FVs built + verified.

**RESULTS (all 9 sane: shape [4096]; norms 20–52; correct head-set provenance per method):**
- `task_specific` (own CIE top-10): rhyme 20.1, next_number 49.4, prev_number 51.9 → real files
  `results/gptj_fv/<task>/` + symlinks in `function_vectors/gpt-j/task_specific/<task>/`.
- `train_selected` (existing 20-train pooled heads): norms 37.0/43.5/44.0 → built into
  `results/gptj_fv_multitask_top10/` (`fv_manifest_paired.json`) + organized symlinks.
- `train_varicl` (existing varicl pooled heads): norms 32.1/34.8/37.6 → written directly into
  `function_vectors/gpt-j/train_varicl/<task>/` (`fv_manifest_paired.json`).
- `selected_heads.json` regenerated for all 3 methods (32 tasks each).

**IMPORTANT CAVEAT — rhyme is filter-less:** GPT-J scored **0/18 ICL-correct** on rhyme's valid
split (both pipelines crashed on the empty correctness filter), so rhyme was re-run with
`--no_filter_to_correct_icl` in BOTH the canonical and varicl pipelines. Its FVs average over all
18 (incorrect) queries — "activations while attempting rhyme", not successful execution. Not
methodologically comparable to the other 28+2 tasks (which filter to correct). By contrast
next_number = 18/18 correct, prev_number = 15/18 — number FVs are standard.

**Other notes:** task-specific top heads of next/prev_number overlap heavily with the pooled set
((15,5),(12,10)…) — the number tasks engage the canonical FV circuitry. Per-task GPU stages ran in
3 parallel tmux windows (~25 min total incl. the rhyme re-run); varicl stage-1 used the
`--num_shards 3` trick so writes_global=False protected the existing pooled artifact; isolated
`results/multitask_aie_heads_varicl_paired/` kept for provenance (mean activations copied into
`results/multitask_aie_heads_varicl/<task>/` for the builder). Logs: `results/_paired_fv_logs/`.

**Goal:** FVs for the new `dataset_files/paired_tasks/{rhyme,next_number,prev_number}.json` (200 pairs
each) via (1) **task_specific** (own CIE top-10), (2) **train_selected** (existing 20-train pooled
head set reused), (3) **train_varicl** (existing varicl train-pooled head set reused).

**Setup:**
- Symlinked the 3 JSONs into `dataset_files/abstractive/` (user-approved "treat as abstractive";
  loader hardcodes abstractive/extractive). Split = 140 train / **18 valid** / 42 test (seed 42) —
  NB small valid split → ≤18 query candidates for CIE/mean-activations (trials redraw demos).
- NEW `task_splits/paired_tasks_3.json` (the stage-2 builders validate `--tasks` against a manifest).

**Plan (GPU = 1×A100 80GB; GPT-J fp16 ≈ 12GB → 3 concurrent model instances, NOT 6):**
- 3 tmux windows (one per task), each sequentially: `src/compute_function_vectors.py
  --dataset_names <task> --batch_size 8` (mean acts + CIE + task-specific FV → `results/gptj_fv/<task>/`),
  then `compute_multitask_varicl_heads.py --tasks rhyme next_number prev_number --num_shards 3
  --shard_index <i> --save_path_root results/multitask_aie_heads_varicl_paired ...` (varicl acts+CIE;
  sharded ⇒ writes_global=False ⇒ existing pooled artifact safe).
- Then (cheap, single jobs): copy varicl mean activations into `results/multitask_aie_heads_varicl/<task>/`;
  `compute_all_task_fvs_varicl.py --tasks ... --task_manifest task_splits/paired_tasks_3.json
  --manifest_name fv_manifest_paired.json --overwrite`; `compute_all_task_fvs_from_multitask_heads.py
  --tasks ... --manifest_name fv_manifest_paired.json`; organized-folder symlinks; `write_fv_head_metadata.py`.

**Blockers:** None.

---

## 2026-06-12 — Stream B: PAIRED 1-shot activation capture for paired_tasks (step 1 of rerun)

**Status:** COMPLETE (capture done; downstream experiments next).

**What:** First step of rerunning the experiments on the new `paired_tasks` datasets,
using the PAIRED design (Stream-E style, generalized): for each shared output word `w`
producible under BOTH functions, build two 1-shot prompts that differ ONLY in the ICL
demo INPUT (same demo label `w`, same query) — so the activation difference at the
label token isolates the function, not token identity. New script
`src/eval_scripts/capture_oneshot_paired_tasks.py`, a sibling of (not an edit to)
Stream-E's `capture_oneshot_paired.py`; reuses the canonical capture primitives
(`get_residual_stack`, `selected_token_records`, `make_token_record`, `flush_shard`)
so locations + on-disk format match `results/residual_activations/*`. Reads
`dataset_files/paired_tasks/` directly.

(An earlier single-task capture — `capture_oneshot_singletask.py` + results — was the
wrong reading of the request and has been removed.)

**Design choice:** single-token outputs NOT required (the paired property holds for
multi-token labels too). This lets the number pair reach 100 (only 26 number-words are
single-token, but 198 shared). Captured the full standard location set per prompt:
pre/first/last label_token (+ label alias) and last_prompt_token (+ final alias), 28 layers.

**Pairs + counts (n_target=100, capped by shared-output availability):**
- antonym_synonym: 100 words → 200 prompts (555 shared available)
- synonym_rhyme:    82 words → 164 prompts (82 available — max)
- antonym_rhyme:    60 words → 120 prompts (60 available — max)
- next_number_prev_number: 100 words → 200 prompts (198 available)

**Command:** `for pair in antonym_synonym synonym_rhyme antonym_rhyme
next_number_prev_number; do HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1
python src/eval_scripts/capture_oneshot_paired_tasks.py --pair $pair; done`

**Outputs:** `results/oneshot_paired_tasks/<pair>/{shard_00000.pt, index.json}`. Each
shared word → 2 functions × 6 roles = 12 rows; activations `[12·n_words, 28, 4096]` fp32.

**Verified:** for each pair, the f1/f2 prompts are identical except the demo input — e.g.
`Q: seven\nA: eight…` vs `Q: nine\nA: eight…` (next vs prev of eight, same query);
`Q: unable\nA: able…` vs `Q: can\nA: able…`. Label span decodes to ` w` (assertion in
script, single- and multi-token).

**Next:** the actual experiments on these captures — FV/geometry analysis (function
axis at label vs query token), and/or generalize the `mixed_icl_*` top-k probe to the
four pairs.

**Blockers:** None.

## 2026-06-12 — Stream B: `paired_tasks` datasets for marginal-matched task pairs

**Status:** COMPLETE (datasets generated; experiments not yet run).

**What:** New dataset family for the mixed-task ICL / function-geometry probes, built
on the sharpened criterion that the function must live ONLY in the (input,output)
relation — neither the input token nor the output/label token alone may reveal the
task (matched input AND output marginals). This rules out the leaky pairs
(capitalize_first/last → output-marginal leak; translation → language ID leak;
country-capital/currency → output leak; hypernym/hyponym → input leak).

**Files:** `dataset_files/generate/create_paired_tasks_datasets.py` (new generator),
outputs in `dataset_files/paired_tasks/`:
- `next_number.json` / `prev_number.json` — k→k+1 / k→k−1, integers as WORDS (digits
  risk tokeniser clumping; words tokenise per-word). Same input set {1..200} → matched
  input marginal; outputs {2..201}/{0..199} → near-identical output marginal. 200 each.
- `rhyme.json` — word→rhyming word (CMUdict via `pronouncing`); inputs from syn/ant
  vocab, outputs constrained to that same real-word vocab → shares the general-word
  marginal with synonym & antonym. 200 pairs, single-token real-word outputs.
- `synonym.json`, `antonym.json` — copied verbatim from `abstractive/`.

**Intended pairs:** antonym|synonym, synonym|rhyme, antonym|rhyme, next_number|prev_number.

**Dropped next/prev_letter** (initially generated, then removed): only 26 letters → can't
reach 200, and a deeper review (see DECISIONS) showed the strict criterion +
in-context-learnability is a real double bind — letters added no new legal+learnable axis
the number/word pairs don't already cover. Decision: keep the 3 pairs, no 4th.

**Verification:** input sets identical across next/prev (matched input marginal);
0 input==output collisions; number-words never digit-clump (e.g. 200→201 = "two
hundred"→"two hundred one"); letters & rhyme outputs are single tokens. Compound
number-words >20 are multi-token but word-based (handled by first/last label-token
scoring). Installed `pronouncing` (PyPI, CMUdict).

**Next:** run `mixed_icl_*` probe on the new pairs (will need a task-generic variant
of the script, currently antonym/synonym-specific).

**Blockers:** None.

## 2026-06-11 — Stream E Phase 2: causal steering at the demo label token (COMPLETE, POSITIVE)

**Status:** Full run done on GPU. Plan: `/root/.claude/plans/immutable-finding-boole.md` (Phase-2 rewrite).

**What:** Inject `α·Δ_label(L_steer)` (mean antonym−synonym difference from the Phase-1 capture, 530
held-out-by-construction words) at the demo's label token of a **synonym-context 1-shot prompt**; read
(a) the query-final-token shift vs the natural direction `Δ_final(L_read)` and (b) rank/logit of gold
`ant(q)` vs `syn(q)`. Test queries = 1,003 shared-input words with single-token gold ant+syn (disjoint
from the Δ words). Steer L∈{6,9,11}, α∈{0,0.5,1,2,4,8}, matched-norm random control on every cell.
Geography (landmark↔park, 84 queries) geometry-only.

**Files:** NEW `src/eval_scripts/steer_label_to_query.py` (inlined add-hook/eval helpers; baukit only
inside main). **Two bugs found+fixed during smoke/plots:** (1) hook args must be the exact 2-arg
`(output, layer_name)` closure — extra default-kwargs get mis-bound positionally by baukit's
`invoke_with_optional_args` (a `Tensor += tuple` crash); (2) cos-vs-α plot initially read at
L_read==L_steer where the query-position shift is identically zero → now picks the best-aligned read
layer; added `fig_cos_shift_by_readlayer_L*.png`.

**Results (n=1003, antonym↔synonym):**
- **GEOMETRY — strongly positive:** steering at L6 moves the query token along the natural syn→ant
  direction: mean cos(shift, Δ_final) ≈ **0.71–0.76 at read L8** (α 0.5–2), >0.5 through L15, ~0.36 at
  L27; **random control ≈ 0.01–0.06 everywhere**. L9/L11 steer: peak cos 0.60/0.48 (read L15). Effect
  is α-stable in the linear regime (0.5–4) and degrades at α=8.
- **BEHAVIOR — positive flip:** baseline (α=0) flip rate 0.283, mean logit(ant)−logit(syn) = −1.44.
  Steering L6: flip **0.595 @ α=4, 0.637 @ α=8**, logit-diff crosses zero at α≈2 and reaches **+1.12**.
  L9 similar (peak 0.593 @ α=4, overshoot at 8); L11 weaker (0.48). Random control: flip ≤0.35,
  logit-diff still negative (−0.92) even at α=8 — the effect is direction-specific, not norm.
- **Geography geometry is much weaker** (cos ~0.13–0.23 vs random ~0; n=84). The landmark↔park
  "function" contrast at the label token propagates far less — consistent with its tiny Δ-final
  separation being carried near-entirely by position rather than content in 1-shot? (open question).
- **Reconciliation with Phase 1:** the *mean* function axis DOES propagate causally label→query
  (this run), even though the *per-word* label→query map is not linearly predictable (Phase-1 low R²).
  I.e. the dominant shared axis is causal; the high-dim per-word tail is what defeats the linear map.

**Outputs:** `results/oneshot_steering/<pair>/{per_query.csv, summary.json, fig_cos_shift_vs_alpha_L*.png,
fig_cos_shift_by_readlayer_L*.png, fig_flip_rate_vs_alpha.png, fig_logitdiff_vs_alpha.png}`.

**Steer-layer sweep extension (L1–5, L12/16/20/24; α∈{0,1,2,4}):** outputs in
`results/oneshot_steering_early/` and `_late/`; combined profile
`results/oneshot_steering/fig_steer_layer_profile_L1to24.png`. **A closed causal window [L4, ~L15]:**
L1–2 null (flip ≈ baseline 0.28, despite max downstream depth); sharp onset L3→L4 (flip 0.34→0.57 @α4,
cos 0.58→0.72); plateau L4–9 (flip 0.54–0.60, cos 0.61–0.76); decay L11–12 (0.48→0.44); **completely
dead by L16** (flip 0.282–0.284 = baseline at L16/20/24; cos 0.09→0.00). Natural ‖Δ_label‖ is also tiny
at L1–2 (0.66–0.94) vs L4 (3.3) — the axis doesn't exist yet. So: the function fingerprint is WRITTEN
into the label token ~L3–6 and CONSUMED (moved to the query position via attention) before ~L16 —
matching the CIE top-head band (L8–15) and best read layers (8–15). Caveat: α is scaled to the natural
per-layer ‖Δ_label‖, so the L1–2 null is "the natural-scale axis doesn't exist," not "no perturbation
could act there."

**Next (optional):** multi-shot variant (stronger baseline synonym behavior → cleaner flip test);
steer along top-k SVD directions instead of the mean; geography follow-up on why propagation is weak.

---

## 2026-06-11 — Stream E: 1-shot paired ICL function geometry (Phase 1, COMPLETE)

**Status:** Ran end-to-end on GPU (RTX PRO 4000 Blackwell). Approved plan:
`/root/.claude/plans/immutable-finding-boole.md`.

**What:** Same-output-word / different-function paired 1-shot prompts (demo label token = source;
query final token = target). Substrate: antonym↔synonym (544 shared single-token words) and
landmark↔park (84). Observational geometry only (steering = gated Phase 2).

**Files (new):** `src/eval_scripts/capture_oneshot_paired.py` (model+baukit; per-row source-token-id
assertion passed for all words), `src/eval_scripts/analyze_oneshot_geometry.py` (pure; subspace-projected
the source→target map so per-layer cost is O(m³), m≤2W, not O(4096³) — identical results, ~30× faster).
Installed `baukit` from git (`pip install git+https://github.com/davidbau/baukit.git`; not on PyPI).
**Env:** run with `HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1`.

**Outputs:** `results/oneshot_paired/<pair>/` (activations), `results/oneshot_paired_analysis/<pair>/`
(`{label,final}_geometry.json`, `fv_projection.json`, `source_target_map.json`, `summary.csv`,
`fig_fv_scatter_L*.png`).

**Findings:**
- **STRONG positive — function is linearly decodable at the query token in FV space.** Final-token acts
  separate antonym vs synonym along the FV difference axis `(fv_f1−fv_f2)`: peak **L11 AUC 0.941, d 2.22,
  88.8% acc** (antonym/syn); **L11 AUC 0.894** (geography). Clean mid-layer band L9–15. Confirmed in scatter.
- **NUANCED — label-token difference = one dominant axis + broad tail.** STABLE rank (Σσ²/σ₁²) of the
  L11 `D_label` is **5.2** (antonym/syn 544×4096) / **3.3** (geo 84×4096): σ₁ holds ~19%/~30% of energy →
  a single clear "function axis"; but 90% energy needs k≈315 dims (high-dim residual). Stable rank is
  lowest at mid layers (min ~4.7 @ L9), coinciding with peak FV separation. (Entropy eff-rank ~150 keyed
  on the fat tail; stable rank keys on the dominant axis.) Plots: `fig_Dlabel_svd_L11.png`,
  `fig_Dlabel_stable_rank_by_layer.png`; tensor `D_label.pt`.
- **NEGATIVE — the demo-label → query-token map is NOT low-rank, NOT rotation-like, weakly predictive.**
  Held-out `map_R2` ≤0.16 at mid layers (0.38 at L0, negative by L18); map effective rank ~190–280;
  in-sample rank-8 R² only ~0.25; Procrustes gap large at informative layers. So "label-space arithmetic
  → predictable rotation at the next position" is **not supported in 1-shot**.
- **Caveats:** 1-shot weakly identifies "function"; W(544/84) ≪ d(4096) so full-matrix M structural
  metrics are rank-limited — the trustworthy signals are held-out `map_R2` and the FV-separation AUC.

**Implication for Phase 2:** steering along the FV difference axis at/near the query token is well-motivated
(high separation); steering the demo label token expecting predictable low-rank propagation to the query is
NOT supported by Phase 1. Consider a multi-shot variant (stronger function identification) before concluding.

**Next:** decide whether to run Phase-2 steering (FV-axis at query) and/or a multi-shot variant.

---

## 2026-06-11 — Stream C: PCA-space (direct) activation→FV ridge sweep — DONE + full-dim comparison

**Status:** DONE. 899 cells (31 token positions × 29 layers), merged + heatmapped. Companion to the
full-dim ridge sweep, in a 16-PC bottleneck.

**Method:** per cell, fit activation PCA (k_act=16) on the pooled 20-train rows; FV PCA (k_fv=16)
fit once on the 20 train FVs; **ridge 16→16, λ via leave-one-train-task-out CV, single 20-train
standardizer** (same recipe as full-dim); predict 7 test tasks, **reconstruct to 4096-d**, report
MSE there. Direct projection (not joint). FV target = `train_selected`. cc/pc excluded.

**Commands:**
- `WORKER=src/eval_scripts/regress_activation_to_fv_pca_ridge.py OUTPUT_DIR=results/pca_ridge_activation_to_fv
   SESSION=pcaridge bash src/eval_scripts/run_fulldim_ridge_shards.sh` (3 tmux windows; ~4–5 min total).
- `python src/eval_scripts/merge_fulldim_ridge_results.py --input_dir results/pca_ridge_activation_to_fv`.

**Files changed:**
- NEW `src/eval_scripts/regress_activation_to_fv_pca_ridge.py` (worker).
- EDIT `src/eval_scripts/run_fulldim_ridge_shards.sh`: `WORKER` env override (backward-compatible).
- REUSED `merge_fulldim_ridge_results.py` unchanged (CSV is a superset of the full-dim schema).
- Output: `results/pca_ridge_activation_to_fv/{shard_icl1..10,combined_*}`.

**Findings (PCA 16→16 vs full-dim 4096→4096):**
- **Best cell ~identical:** PCA `icl10/finaltok @ L13 = 0.1147` vs full-dim `icl10/finaltok @ L11 =
  0.1161`. The 16-PC bottleneck is **free at the optimum** — even ~0.0014 better (PCA denoises →
  slightly better test transfer). Best layer flat L11–13.
- **Same structure:** query position (ICL 10) best; clean mid-layer bowl; embedding worst.
- **Where PCA wins vs loses:** in the sweet spot (L8–13) PCA ≈ or < full-dim (L11: 0.1321 vs
  0.1325; L13: 0.1320 vs 0.1331). In **later layers it's worse** (L20: 0.1475 vs 0.1407; L28:
  0.1583 vs 0.1491) and embedding worse (0.2026 vs 0.1960). Net per-cell mean Δ(pca−full)=+0.0032,
  PCA better in 243/899. → top-16 activation PCs capture all recoverable signal *where it's
  concentrated* (mid layers) but discard useful directions where it isn't (late/embedding).
- **Identity verified:** `test_mse = (16/4096)·pca_test_mse + floor` to 4e-8 (orthonormal FV-PCs).
  FV-PC reconstruction floor (test) ≈ 0.099 — the irreducible MSE of the 16-PC FV target.
- **α interior** (peak ~100; grid logspace(-2,6,17)); 11 pinned cells are all L0 embedding
  constant-feature positions (=0.217 predict-the-mean baseline), harmless.

**Next:** Headline comparison ready: a 16-PC activation→FV ridge loses essentially nothing vs full
4096-d at the best read points. Optional: sweep k_act to map the bottleneck cost by layer.

**Blockers:** None.

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

---

## Stream: switch-logit pretty aggregate (2026-06-26)

**Status:** Done.

**What:** Added a "pretty" variant of the logit task-switch aggregate figure for
easier cross-site comparison.

**Files changed:**
- `src/eval_scripts/plot_switch_logit.py`: added `series_by_alpha`, `best_alpha`,
  `plot_axis_pretty`, `plot_pretty_aggregate`; called from `main()`.

**Output:** `results/direction2_label_geometry/oneshot_switch_logit/figures/fig_logit_aggregate_pretty.png`

**Changes vs fig_logit_aggregate.png:**
- Y-axis now shared per row (same task, both steering sites) so left/right panels
  are directly comparable.
- Each panel overlays its horizontal neighbour's *best* curve (alpha with highest
  peak logit diff) as a crimson dotted line — read cross-site comparison in one panel.

**Findings:** Number tasks — label-token best peaks early (~L5-8), final-token best
peaks late (~L12): same peak height, depth-shifted. Synonym/antonym — final-token
site sustains larger/longer logit shift out to ~L25 vs early-concentrated label site.

**Next:** None pending.

**Blockers:** None.

---

## 2026-07-12 — Stream Q addendum: line-graph summary of the 10-shot strip study

**Owner:** Coordinator (CPU pod; plotting only, no GPU). **Status:** DONE.

**What:** 1-D summary of the Stream Q grids requested as a line chart: x = the 30 intervene
tokens in sequence order (d1_in…d10_lab), y = peak Δcos at qfinal (same nanmax-over-grid
reduction as scalar_overview.png), one line per (direction, α) = 12 lines. Hue = direction
(4 fixed categorical slots), line style = α (solid 2 / dashed 4 / dotted 8).

**Files:** NEW `src/eval_scripts/plot_tenshot_strip_lines.py` (CPU-only; reads the saved .npy
grids; `--labels_only` restricts x to the 10 demo label tokens; `--top_k N` reduces each grid to
the mean of its top-N cells instead of the max — top-10 variants saved as `*_top10.png`, nearly
identical shape to the max, values ~7% lower on average (top10/max = 0.93 mean, 0.85–0.98 across
the 100 label-token grids with max>0.01) ⇒ the peak sits on a coherent high-Δcos plateau,
not an isolated spike). Output
`results/direction2_label_geometry/tenshot_strip_intervention_cos_heatmap/figures/scalar_lines.png`
and `scalar_lines_labels_only.png`. DELETED `figures/scalar_overview.png` (superseded by the line
views; `plot_tenshot_strip_heatmap.py` can still regenerate it from the grids if ever needed).

**Findings:** the line view makes the Stream Q claims directly visible — sawtooth with ~all
signal at `lab` tokens (in/pre ≈ 0); syn→ant ≫ ant→syn at every demo; digits show α=2 ≥ α=4 ≥ α=8
(saturation) while syn→ant strengthens with α; contribution is spread across demos 2–10 with a
d10_lab spike for syn→ant α8.

**Next:** none pending. **Blockers:** none.

---

## 2026-07-13 — GPT-J baseline accuracy vs n_shots (0..10), 4 strip-study tasks

**Owner:** Coordinator (CPU pod; GPU compute on a fresh L4 pod — no 4090 in EU-RO-1 stock;
Blackwells excluded per DECISIONS torch incompat). **Status:** DONE, pod `i4eehnuear0yje` terminated.

**What:** Baseline (no-intervention) top-1/2/3 accuracy for antonym / synonym / next_number_digits /
prev_number_digits at n_shots 0..10, full test split, same Q:/A: template as the 10-shot strip study
— companion figure to the scalar_lines strip summaries.

**Files:** NEW `src/eval_scripts/compute_task_accuracy_by_nshots.py` (GPU; resumable per (task,n)
cell; NOTE wraps the eval in torch.no_grad() — `eval_utils.n_shot_eval_no_intervention`'s batched
logit path lacks it and OOMs at batch_size 32 from autograd graph retention, latent bug for any
future batched caller) + `plot_task_accuracy_by_nshots.py` (CPU). Output (TRACKED)
`results/general/task_accuracies/by_nshots/{task}_n{n}.json` (44 cells); figures
`gptj_accuracy_by_nshots_top{1,3}.png` live next to the strip summaries in
`results/direction2_label_geometry/tenshot_strip_intervention_cos_heatmap/figures/` (user request;
JSONs stay in general/). Log `logs/acc_by_nshots.log`.

**FINDINGS:** digit tasks hit 100% top-1 by n=1 (next) / n=2 (prev); n_test only 42 there.
Antonym climbs 0.6%→24.6%→47.7%→56.5% (n=0..3), plateaus ~62-66% by n=5. Synonym saturates
~22-26% from n=5. All four ≈0% at n=0. Shape matches the steering-impact rise over demos 1-3
(task evidence accumulates over the first ~3 demos then saturates).

**Blockers:** none.

---

## 2026-07-13 — Stream R: 10-shot strip activation-MAGNITUDE heatmaps + d_in token BUGFIX

**Owner:** Coordinator (CPU pod; 6× RTX PRO 4500 Blackwell pods, ~$0.74/hr each). **Status:** DONE,
all pods terminated after run.

**BUGFIX (affects Stream Q too):** `token_positions` defined `d{i}_in = pre_label − 1`, which is the
constant "A" template token, NOT the input word ("Q: hot\nA: cold" → Q,:, hot,\n,A,:, cold; the label
carries the leading space, so pre−1 = "A"). Every `*_in_*` grid in the cos study measured steering at
"A". Fixed in BOTH tenshot scripts: `d{i}_in` = LAST token of demo i's input word (via
`demonstration_{i}_token` meta labels; handles multi-token inputs). All old `_in` grids (240 cos,
124 norm) deleted and recomputed with the corrected position; `_pre`/`_lab`/qfinal grids unaffected.
Two-shot study unaffected (never had an `_in` slot). Stream Q's "input rows ≈ 0" finding SURVIVES the
fix (see below), but pre-fix it had only tested the "A" token. Blackwell sm_120 works with the
template image's torch 2.8 cu128 (smoke-validated).

**What (Stream R):** same sweep as Stream Q (identical pairing, steer vecs, point-edits; n_pairs=300,
α∈{2,4,8}, both pairs, both directions) but measuring MAGNITUDE of the qfinal move per (intervene
layer i × read layer k): rel = mean‖Δ‖/‖clean‖ (THE metric of record, per user) and raw = mean‖Δ‖,
both from one steered pass. Split across 6 pods by (task_pair, α); resumable; merged summaries
rebuilt via a final full-α resume-run per pair.

**Files:** NEW `src/eval_scripts/steer_tenshot_strip_norm_heatmap.py`, NEW
`plot_tenshot_strip_norm_heatmap.py` (--metric rel/raw; sequential magma strips + scalar-lines
summaries). Output (TRACKED rel .csv only, 360; rel/raw .npy gitignored)
`results/direction2_label_geometry/tenshot_strip_intervention_norm_heatmap/<pair>/` +
figures (strip_rel_alpha{2,4,8}, scalar_lines_rel[_labels_only]). Cos study figures
re-rendered with corrected `_in` rows (scalar_overview.png stays deleted; top10 line figures deleted —
user: same info as max). Logs `logs/normstrip_*.log`.

**FINDINGS:**
- **Magnitude confirms the label-token story with correct input tokens.** rel peak/median by slot:
  lab 0.317/0.138 ≫ pre 0.141/0.024, in 0.095/0.019. Corrected cos rows: in max 0.009 (median 0.001)
  — REAL input-word steering still produces ~no directional shift toward the target task at qfinal.
- **But input-word edits DO move qfinal.** At α=8 the in-slots reach rel ~0.05–0.12, growing with
  demo index — the perturbation propagates (magnitude) without rotating qfinal toward the paired
  task (cos ≈0): consistent with lexical, non-task-directional transport.
- **A third "gain" metric (mean‖Δ‖/(α·‖steer_vec‖)) was computed and RETIRED same-day** — it was a
  coordinator addition, not requested (user wanted rel = ‖Δ(k)‖/‖clean(k)‖, which exists). It is
  also degenerate (user-caught): where the two tasks' mean activations coincide — digit in-slots
  (identical input distributions), layer 0 — the denominator is sampling noise and the ratio blows
  up (spikes to ~50; e.g. next→prev d7_in α8 "peak" 22 at intervene L0, ‖steer_vec‖=0.065).
  Gain grids deleted from disk, dropped from both scripts and figures; the `gain` peak entries
  still inside the existing summary JSONs are DEPRECATED — do not cite. For the record, masked
  (‖steer_vec‖≥1) gain medians ordered lab 0.92 > in 0.57 > pre 0.18, consistent with rel.
- rel lab-token peaks sit at intervene L5–8 → read L~13–26 (median 7→17), same band as the cos study.
- Norm-growth check: mean clean qfinal norm grows monotonically with read layer (summary
  `mean_clean_norm_by_read_layer`), which is why raw grids skew late and rel is the headline.

**Verification:** smoke (16 pairs, 3 layers) surfaced a 0/0 at layer 0 (constant-token slots — the
observation that led to both the d_in bugfix and, later, the gain retirement); full run: lower-tri≡0
and finiteness asserts pass on all rel/raw norm grids + 360 cos grids; corrected-position sanity
checked on CPU with multi-token inputs ("mountain top" → " top"); rel figures re-render cleanly
after the gain removal and --metric gain is rejected by argparse.

**Blockers:** none.

---

## 2026-07-14 — Stream S: per-pair ("nearest neighbour") steering for the two-shot token-pair heatmaps + input2/qinput position BUGFIX

**Owner:** Stream S (CPU pod; 1× A100 80GB PCIe pod `fv-perpair-steering`, $1.39/hr, terminated after run).
**Status:** DONE. Compute 00:04–00:52 UTC (~48 min, BS=256); figures rendered on the CPU pod
(the GPU container's system python has a numpy/matplotlib ABI mismatch — plot there fails with
`_ARRAY_API not found`; plotting is CPU-only anyway).

**What:** user-requested extension of the two-shot token-pair cosine study: `--steer_mode perpair`
in `steer_twoshot_tokenpair_cos_heatmap.py` — instead of the pair-MEAN steer vector, each source
prompt is steered by its OWN matched counterfactual's activation diff at the edited (token, layer);
α=1 ≡ exact single-site activation patching (asserted: cos(steered site, tgt site) > 0.999).
α∈{0.5,1,2}, both directions, both task pairs; evaluation identical (per-pair Δcos toward the
matched target acts, averaged). Output NEW root
`results/direction2_label_geometry/twoshot_tokenpair_perpair_cos_heatmap/` (same file layout →
`plot_twoshot_tokenpair_heatmap_grid.py --root ... --alphas 0.5 1 2` runs unchanged).

**BUGFIX (corrects Stream R's "two-shot study unaffected" claim):** the twoshot tokenpair script
HAD the d_in bug — `input2 = prelabel2 − 1` and `qinput = qfinal − 1` are the constant "A" template
token, not the input/query words. Conclusive: in the existing mean-study summary JSON, input2's
embedding-layer steer norm ≡ 0 and baseline cos ≡ 1.0 (byte-identical token across f1/f2 despite
differing demo-2 input words). Fixed like the tenshot scripts (last token of the
`demonstration_2_token` / `query_demonstration_token` groups) + startup decoded-token print/asserts.
ALL input2/qinput grids and figures in the EXISTING
`twoshot_tokenpair_intervention_cos_heatmap/` measured the "A" token — left in place (user's call
whether to replace), but a position-correct mean rerun (α∈{2,4}) goes ADDITIVELY to
`twoshot_tokenpair_mean_fixedpos_cos_heatmap/` in the same launch.

**Files:** `src/eval_scripts/steer_twoshot_tokenpair_cos_heatmap.py` (--steer_mode, per-row add_vec
in capture(), position fix, sanity asserts; mean path bit-identical — verified), NEW launcher
`logs/run_twoshot_tokenpair_perpair.sh`; log `logs/twoshot_tokenpair_perpair_full.log`.

**Verification so far:** smoke (16 pairs, layers 0/6/11): (1) edited script in mean mode
bit-identical to `git show HEAD` original on all 12 grids not involving input2/qinput; (2) affected
grids changed as intended; (3) decoded sample tokens f1/f2 = ' square'/' ring' (input2), 'ner'
(qinput); CPU multi-token check "mountain top" → ' top'; (4) perpair α=1 patch assert + structural
asserts (lower-tri≡0, clean embedding col≡0) pass.

**FINDINGS (antonym→synonym unless noted; peaks are max Δcos over the 29×29 grid):**
- **Per-pair patching at α=1 matches mean steering at α=4 on the label tokens.** label2→qfinal:
  perpair α=1 +0.054 @ i8/k18 vs mean α=4 +0.054; label1→qfinal perpair α=1 +0.017 vs mean α=4
  +0.026. Same early-intervene (L7–8) → mid/late-read band as the mean study.
- **Per-pair is non-monotone in α: α=1 (exact patching) ≥ α=2 almost everywhere** (e.g.
  input2→prelabel2 +0.123 @α1 vs +0.104 @α2) — overshooting past the counterfactual hurts,
  unlike mean steering which keeps growing to α=4. α=1 is the natural operating point.
- **The input-word row flips the story vs mean steering.** Mean steering (position-correct) still
  barely moves anything from input2 (peaks +0.006–0.010, echoing Stream Q/R "input rows ≈ 0"),
  but per-pair patching of the demo-2 input word is the STRONGEST cell in the whole matrix:
  input2→prelabel2 +0.123 @ i7/k24, input2→qfinal +0.075 @ i0 (α=1; digits: input2→label2
  +0.144/+0.151 @ i0/k10). Interpretation: per-pair input diffs are lexically idiosyncratic and
  cancel in the mean vector (‖mean diff‖ ≪ mean‖diff‖), so only the matched-counterfactual edit
  transports them; at i0 it is literally swapping the input token embedding. Caveat: input2 edits
  mix lexical+function content by construction.
- **Full-run regression:** in the fixed-pos mean rerun, grids among {label1, prelabel2, label2,
  qfinal} are numerically identical to the ORIGINAL mean study (same peaks to 4 dp) — the position
  fix changed nothing it shouldn't. qinput rows DID change (old qinput was the constant "A"):
  e.g. qinput→qfinal α4 +0.003 (old) → +0.010 (fixed).
- All structural asserts pass on the full run (perpair α=1 patch-identity, lower-tri≡0, clean
  embedding col≡0, finiteness).

**Next:** decide whether the ORIGINAL `twoshot_tokenpair_intervention_cos_heatmap/` (pre-fix
input2/qinput grids) should be replaced by the fixedpos rerun.

**Blockers:** none.

---

## 2026-07-14 — Stream R addendum: signed norm-growth metric (ngrow) + rot

**Owner:** Coordinator (6 fresh RTX PRO 4500 Blackwell pods `fv-ngrow-1..6`, terminated after).
**Status:** DONE.

**What:** user-requested 4th metric ngrow(i,k) = mean_pairs[‖steered(k)‖₂/‖clean(k)‖₂ − 1] — SIGNED
norm growth of qfinal (does the activation grow or shrink), needing a full recompute (‖steered‖ was
never saved). Same sweep/split as Stream R. Also saved (unplotted, future-proofing) rot(i,k) =
mean_pairs[cos(steered, clean)] — with ngrow this decomposes any move into scaling vs rotation
without another GPU run. Resume semantics changed: cell skipped iff ALL metric .npys exist; rel/raw
rewritten identically for consistency. rot lower-tri ≡ 1 within 1e-5 (cos(x,x) float rounding), the
others exactly 0.

**Files:** extended `steer_tenshot_strip_norm_heatmap.py` + `plot_tenshot_strip_norm_heatmap.py`
(--metric ngrow: diverging RdBu strips centred on 0; lines reduce by signed extreme = max |cell|).
NEW figures strip_ngrow_alpha{2,4,8}.png, scalar_lines_ngrow[_labels_only].png. Logs
`logs/ngrow_*.log`. Summary JSONs now carry peak+trough for ngrow/rot.

**FINDINGS:**
- **Label steering GROWS the qfinal activation; word-task α=8 steering SHRINKS it.** Sawtooth at lab
  tokens, amplitude growing with demo index. Extremes per direction: next→prev +0.149 (d10_lab, α=2!)
  / −0.041 (d10_lab α8); prev→next +0.075; ant→syn +0.042 / −0.040 (d10_pre α8); syn→ant +0.042 /
  −0.031 (d10_lab α8).
- **Sign flips with α for the word pair:** small injections grow the norm, α=8 turns growth into
  shrinkage at late-demo lab/pre slots (visible as the negative dips of the dotted lines) — large
  pushes move qfinal off-manifold and the norm contracts.
- ngrow magnitudes (≤0.15) are much smaller than rel (≤0.32): most of the movement is rotation /
  off-axis displacement, not radial scaling (rot grids saved for the full decomposition).

**Ops note:** first fan-out launched with no `cd` (SSH lands in /root) → all 6 died silently while
pgrep liveness checks self-matched; caught by user noticing idle GPUs. Rule now in memory: cd
explicitly, verify via `$!` + `kill -0`, never pgrep. ~$1 idle waste.

**Blockers:** none.

---

## 2026-07-14 — Stream S addendum: cumulative-CLAMP sanity study (trajectory patching)

**Owner:** Stream S (1× RTX PRO 4500 Blackwell pod `fv-cumclamp-steering`, $0.74/hr, terminated).
**Status:** DONE. Compute 18:40–18:55 UTC (~15 min, BS=128); figures rendered on the CPU pod.

**What (user-requested sanity check):** `--layer_mode cumulative` in
`steer_twoshot_tokenpair_cos_heatmap.py` — for intervention token t and start layer i, hard-CLAMP
t's activation to the matched counterfactual's at EVERY layer ℓ∈[i..28] (layer-specific values =
per-pair trajectory patching from layer i on; perpair only, argparse-enforced; NO α sweep — grids
carry the nominal `alpha1` tag so the plot script runs with `--alphas 1`). Same Δcos measurement
and figures; grid x-axis = clamp START layer. Output
`results/direction2_label_geometry/twoshot_tokenpair_perpair_cumclamp_cos_heatmap/`. Launcher
`logs/run_twoshot_tokenpair_cumclamp.sh`; log `logs/twoshot_tokenpair_cumclamp_full.log`.

**FINDINGS (antonym→synonym; digits analogous):**
- **Sanity check PASSES: cumclamp ≥ single-site perpair α=1 on all 15/15 token-pair peaks** (none
  weaker by >0.001). Single-site is the ℓ=i-only special case and behaves like a lower bound.
- **Peaks migrate to start layer i=0** for 13/15 pairs (full-trajectory replacement ≈ counterfactual
  token substitution is maximal): input2→prelabel2 +0.135 @ i0/k24 (vs +0.123 @ i7 single-site),
  label2→qfinal +0.068 @ i0/k26 (vs +0.054 @ i8), qinput→qfinal +0.017 @ i0 (vs +0.010 @ i13).
- Gains over single-site are mostly modest (~1.1–2×), i.e. a single well-placed site already
  captures the bulk of what the full clamped trajectory transports; the biggest relative gains are
  on the weak late-token pairs (prelabel2→label2 +0.033 vs +0.017).
- Clamp-identity asserts (ℓ=i and ℓ=28) and lower-tri≡0 hold on the full run; the clean-token
  embedding-column assert is single-mode-only by design (a clamp starting at i=0 also patches
  layers 1..28, so column 0 is legitimately nonzero).

**Verification:** hook refactor regression — edited script in mean/single mode is bit-identical to
the `claude-perpair-steering` version on all 30 smoke grids; cumulative smoke peaks consistent with
the full run; `--layer_mode cumulative --steer_mode mean` rejected by argparse.

**Blockers:** none.

---

## 2026-07-14 — Stream S: METRIC MIGRATION — all cos steering studies recomputed as dircos

**Owner:** Stream S (7× GPU pods: fv-dircos-twoshot + fv-dircos-10s-{as,dig}{2,4,8}, all RTX PRO
4500 Blackwell $0.74/hr; A100 out of stock at launch). **Status:** DONE (status line updated
2026-07-16 during the main-branch merge — the entry body was already finalised: all shards landed,
asserts pass on all 360+270 grids, all 7 pods terminated and verified gone).

**What:** user discovered the cos studies plotted Δcos-to-target, not the intended
dircos = mean_pairs[cos(act_tgt − act_src, act_src_steered − act_src)] (see DECISIONS 2026-07-14
"METRIC OF RECORD"). Per-pair steered acts were never saved ⇒ full steered-pass re-runs:
- twoshot token-pair (one pod, sequential): perpair α{0.5,1,2} → `twoshot_tokenpair_perpair_cos_heatmap/`,
  cumclamp → `twoshot_tokenpair_perpair_cumclamp_cos_heatmap/`, mean α{2,4} → CANONICAL
  `twoshot_tokenpair_intervention_cos_heatmap/` (supersedes pre-position-fix Δcos data; the interim
  `twoshot_tokenpair_mean_fixedpos_cos_heatmap/` deleted as redundant — in git on branch).
- tenshot strip cos (6 pods, one (task_pair, α) shard each; stale Δcos npy/csv/summaries deleted
  first so resume can't reuse them); merged summaries rebuilt by a full-α resume pass per task_pair
  after shards land (Stream R norm-study pattern).

**Files:** metric swap in `steer_twoshot_tokenpair_cos_heatmap.py` + `steer_tenshot_strip_cos_heatmap.py`
(surgical: diff-vs-diff cosine at the read site; docstrings + summary "metric" keys); label/scale
updates in `plot_twoshot_tokenpair_heatmap_grid.py`, `plot_tenshot_strip_heatmap.py` (scalar
overview → symmetric RdBu, dircos is signed), `plot_tenshot_strip_lines.py`. New launchers
`logs/run_twoshot_tokenpair_dircos.sh`, `logs/run_tenshot_strip_dircos_shard.sh`.

**Verification so far:** synthetic unit-check of the exact metric line (aligned→1, zero→0,
orthogonal→0); twoshot + tenshot smokes pass all structural asserts with correct decoded tokens;
all 90 smoke grids within [-1,1]; smoke peaks behave as dircos should (twoshot perpair α=1
input2→label2 +0.94 — near-perfect alignment, vs the Δcos ceiling ~0.15).

**FINDINGS (all values = dircos peaks over the 29×29 grid unless noted):**
- **Twoshot: steering displacement points almost exactly along the counterfactual direction at the
  strong sites.** antonym→synonym label2→qfinal: perpair α=1 +0.77, cumclamp +0.85, mean α=4 +0.75
  (old Δcos ceiling was ~0.09 headroom); input2→label2 perpair α=1 +0.92, cumclamp +0.94.
- **Tenshot: the Δcos-era "input rows ≈ 0" claim does NOT survive in dircos.** Input-slot steering
  aligns moderately with the counterfactual direction (peaks +0.25…+0.41, medians ~+0.17…+0.24
  across the 4 combos) — the old metric hid this because the input-edit displacement is small in
  MAGNITUDE (norm study rel ~0.02–0.1) and Δcos conflates alignment with proximity. Label slots
  still dominate decisively: peaks +0.47…+0.78, medians up to +0.74 (ant→syn in-peak 0.36 vs
  lab-med 0.44; syn→ant lab 0.78). Qualitative ordering lab ≫ pre/in survives; the "≈ 0" wording
  does not. Stream Q/R should re-quote their cos numbers from the new summaries.
- Tenshot peaks sit in the same intervene-early → read-mid/late band as before (lab peaks
  i4–i9 → k15–k27; digits α=2 shows an early k5 peak).
- Structural asserts (lower-tri≡0, finiteness, twoshot patch/clamp identities) pass on all
  360 tenshot + 270 twoshot grids; all values within [-1,1].

**Next:** none — figures live in the three twoshot `figures/` dirs and
`tenshot_strip_intervention_cos_heatmap/figures/` (strip_alpha{2,4,8}, scalar_overview,
scalar_lines[_labels_only]).

**Blockers:** none. All 7 pods terminated (verified zero remaining).

