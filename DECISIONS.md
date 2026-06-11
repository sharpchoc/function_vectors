# DECISIONS

Reusable conventions, project decisions, and open questions. Append; date entries.
Resolved questions move from "Open" to "Decided" with the rationale.

---

## 2026-06-11 — FINDING: function is decodable at the query token in FV space, but the label→query map is not low-rank/rotation-like (1-shot)

Phase-1 of the function-geometry experiment (Stream E; scripts `capture_oneshot_paired.py` +
`analyze_oneshot_geometry.py`; results in `results/oneshot_paired_analysis/`). Same-output-word,
different-function paired 1-shot prompts; source = demo label token, target = query final token.

- **Function fingerprint is present and linearly decodable at the QUERY token** along the FV difference
  axis `(fv_f1−fv_f2)`: peak **L11 AUC 0.941 / d 2.22** (antonym↔synonym), **0.894** (landmark↔park),
  mid-layer band L9–15. This is the robust, positive result.
- **The label-token difference = one dominant shared axis + a broad high-dim tail.** STABLE rank
  (Σσ²/σ₁²) of the L11 `D_label` matrix is **5.2** (antonym↔syn, 544×4096) / **3.3** (geography, 84×4096):
  σ₁ alone holds ~19%/~30% of the energy → a single clear "function axis." BUT 90% of the energy needs
  k≈315 dims (antonym↔syn) — the residual is high-dimensional/per-word. Stable rank is **lowest at mid
  layers (min ~4.7 @ L9, ~5 through L15)**, coinciding with peak query-token FV separation; rises to ~9
  by L24. (The entropy "effective rank" was ~150; the two disagree precisely because of the fat tail —
  stable rank keys on the dominant axis.) Centered stable rank (mean axis removed) = 18.8 / 9.4.
  **Robust to magnitude:** unit-normalizing each difference vector before stacking leaves stable rank
  essentially unchanged (5.4 / 3.5 vs raw 5.2 / 3.3) — so the dominant axis is DIRECTIONAL, not driven
  by a few high-norm vectors. Magnitudes are themselves tight (CV ~23%/18%) BUT 14/544 antonym↔syn rows
  are exactly zero (degenerate: demo input collided across functions, e.g. borrow←lend in both → identical
  prompts; excluded when normalizing → 530 rows). Artifacts: `D_label.pt`, `fig_Dlabel_svd{,_unit}_L11.png`,
  `fig_Dlabel_magnitude_hist_L11.png`, `fig_Dlabel_stable_rank_by_layer{,_unit}.png`.
- **But the demo-label → query-token linear map is weak / high-rank / not rotation-like** (held-out
  R² ≤0.16 mid-layers, eff rank ~190–280, Procrustes gap large). So the original "low-rank manifold +
  *predictable rotation* from label to next position" hypothesis is **not supported in 1-shot** — even
  though a single dominant function axis DOES exist at the label token.
- **Method notes (reusable):** (1) hold the OUTPUT token fixed and vary function → the source-token
  activation difference is pure contextualization (the label token is literally identical across f1/f2;
  assertion enforced). (2) For a W-sample map into d=4096 with W≪d, the full-matrix M structural metrics
  (eff-rank, reduced-rank, ‖MᵀM−I‖) are rank-limited/regularization-dominated — trust **held-out map_R2**
  and the **FV-separation AUC**, not in-sample reduced-rank R². (3) Project the source→target map into the
  ≤2W-dim data span before any d×d op — all reported metrics are invariant, ~30× faster.
- **Decision for Phase 2 (if pursued):** steer along the FV difference axis at/near the query token
  (well-motivated by the separation); do NOT assume steering the demo label token propagates predictably
  to the query (Phase 1 contradicts it). A multi-shot variant (stronger function identification) is the
  natural next probe before drawing strong conclusions.

## 2026-06-10 — Mixed-task ICL probe: synonym pays for diluted demos, not antonym

Probe (`src/eval_scripts/mixed_icl_antonym_synonym_topk.py`): n=200 prompts, 10 ICL
demos = 5 antonym + 5 synonym, query from one task; top-1/2/3 = correct first token
in model's top-k. Ran both demo orders + a same-task 10-shot reference overlay.
(Query word is excluded from the ICL demos; n=50 SE≈0.06 was too noisy → use n≥200.)

- **Antonym query ≫ synonym query** (top-1 0.47–0.55 vs 0.12–0.20), every cut. This is
  task difficulty — synonyms have many valid answers so the single gold first-token
  frequently misses top-3. First-token top-k *understates* synonym competence; use
  any-acceptable-synonym scoring if synonym is the focus.
- **Recency is real (~+0.08 top-1):** each task scores higher when *its* demo block is
  the last one before the query (antonym 0.47→0.55, synonym 0.12→0.20). An earlier
  n=50 run wrongly called recency "weak/inconsistent" — that was sampling noise.
- **Antonym is fairly robust to halved demos** (mixed 0.47–0.55 vs same-task ref 0.565,
  matching the ref when its demos are recent); **synonym degrades hard** (mixed
  0.12–0.20 vs ref 0.365) → synonym pays most for diluted demos.

Repro convention for model-loading scripts on this box: deps not preinstalled —
`pip install transformers==4.49.0 accelerate` (per fv_environment.yml); baukit is
not on PyPI, so scripts that only need `eval_utils`' pure helpers should inline them
rather than import the baukit-laden module. GPT-J weights cached under
`/workspace/.cache/huggingface`; run with `HF_HOME=… HF_HUB_OFFLINE=1`.

---

## Conventions (observed in the codebase)

- **Model:** GPT-J (29 layers, d=4096). FP16 + Qwen variants exist (see git log).
- **Task split:** `task_splits/abstractive_train_test_tasks_29.json` = 20 train + 9 test.
- **Results layout:** one dir per experiment under `results/`; per-task subdirs hold
  `.pt` artifacts + a `*_metadata.json` / `*_summary.json` describing the run config.
- **Head set artifact:** `multitask_top_aie_heads.pt` (+ `_metadata.json`) — ranked
  (layer, head) list by prompt-weighted CIE. Stored top-40; top-10 typically used.
- **FV artifact:** `{task}_function_vector*.pt` = sum over selected heads of
  out_proj applied to the head's mean last-token activation.
- **`_multitask_top10` suffix** on a result/FV dir means it was built from the
  multitask top-10 head set rather than per-task FVs.
- **Save intermediates (within storage reason).** When a big experiment has an expensive stage
  (forward passes, CIE, activation capture), persist the *general* intermediate — not just the
  final answer — so variations rerun in minutes instead of hours. Concretely: store the full
  per-(layer,head) CIE grid + a generously-sized ranking (top-40, even if top-10 is used) rather
  than only the chosen heads; store mean activations for **all** heads/positions needed by any
  plausible N/k, not just the selected ones; keep per-prompt effects when subset re-aggregation
  might matter; cache residual-stream activations once and reuse across regression targets.
  Payoff observed 2026-06-11: top-20/30/40 varicl FVs + a whole new PCA-ridge heatmap cost
  minutes because stage-1 saved all-head activations and the top-40 ranking. Guardrail: this is
  for O(GB) per-task tensors and grids, not for dumping every forward pass — if an intermediate
  would cost ≫ recomputing it on demand, skip it and note the recompute command in the metadata
  instead.

---

## Decided

### 2026-06-11 — PCA-space (direct) activation→FV ridge sweep + 16-PC vs full-dim comparison

Companion to the full-dim ridge decoder (Stream C below), run in a 16-PC bottleneck across all 31
token positions × 29 layers. Per cell: activation PCA (k_act=16) fit per-cell on 20 train; FV PCA
(k_fv=16) fit once on 20 train FVs; **ridge 16→16, λ by leave-one-train-task-out CV, single
20-train standardizer**; predict 7 test tasks, reconstruct to 4096-d, score there. Direct
projection (not the deprecated joint). FV target `train_selected`. cc/pc excluded.

- **Scripts:** NEW `regress_activation_to_fv_pca_ridge.py`; launcher `run_fulldim_ridge_shards.sh`
  generalized with a `WORKER` env override (drives both sweeps); merge reused unchanged.
- **Output:** `results/pca_ridge_activation_to_fv/` (combined_metrics.csv = 899 rows + heatmaps).
- **HEADLINE:** the 16-PC bottleneck is **free at the optimum** — PCA best `icl10/finaltok @ L13 =
  0.1147` vs full-dim `@ L11 = 0.1161` (PCA marginally *better*; it denoises). In the mid-layer
  sweet spot (L8–13) PCA ≈/< full-dim; in later/embedding layers PCA is worse (L28 0.158 vs 0.149).
  Net mean Δ(pca−full)=+0.003. → 16 activation PCs hold all recoverable activation→FV signal where
  it's concentrated; the regression target genuinely lives in a ~16-dim subspace.
- **Metric identity:** `fv_test_mse = (k_fv/4096)·pca_test_mse + floor` (FV-PCs orthonormal),
  verified to 4e-8. FV-PC reconstruction floor (test) ≈ 0.099. So selecting α on PCA-space CV MSE ==
  selecting on reconstructed MSE (differ by the constant floor).
- **Comparable metric note:** this reconstructed-4096-d MSE is the same unit as the full-dim ridge
  (0.116) and the k/layer sweeps — NOT the joint-PCA-space MSE (Open Q3).

### 2026-06-10 — `train_varicl` RAN to completion; GOTCHA: stage-2 needs test-task activations precomputed

The variable-ICL method is fully built for GPT-J: 29 FVs + heads.pt + manifest + per-task
`selected_heads.json` under `results/function_vectors/gpt-j/train_varicl/`. Pooled top-10 (varicl)
= [(9,14),(15,5),(8,1),(12,10),(11,0),(8,0),(14,0),(24,6),(21,2),(10,0)]; **8/10 overlap with
`train_selected`** (top-2 identical), so the variable-ICL regime selects nearly the same head
subspace as fixed-ICL train pooling — overlap, not identity, as expected.

**GOTCHA (pipeline gap, now handled):** the documented step sequence (stage-1 `--task_split_key
train_tasks` → stage-2 `compute_all_task_fvs_varicl.py`) is INCOMPLETE. Stage-2 builds FVs for all
29 tasks and **requires `<task>_mean_head_activations_varicl.pt` to already exist for every task**
(it raises `FileNotFoundError`, does NOT compute on the fly). Stage-1 over `train_tasks` only writes
the 20 train tasks' activations, so the build crashes on the first test task. **Fix:** run stage-1
once more on the test split to an isolated dir (so `writes_global` can't clobber the train-pooled
head artifact), then copy the 9 activation files in:
```
python src/eval_scripts/compute_multitask_varicl_heads.py --task_split_key test_tasks \
  --abstractive_only --query_split valid --demo_split train --n_top_heads 40 --batch_size 8 \
  --min_shots 1 --max_shots 10 --max_successful_prompts 170 --filter_to_correct_icl \
  --save_per_prompt_effects --save_path_root results/_varicl_testtasks --num_shards 1
# then: cp results/_varicl_testtasks/<task>/<task>_mean_head_activations_varicl.pt
#            results/multitask_aie_heads_varicl/<task>/   (for the 9 test tasks)
python src/eval_scripts/compute_all_task_fvs_varicl.py --overwrite
```
This recomputes CIE for the 9 test tasks too (unavoidable; they're short-sequence so it's cheap and
the resulting test-pooled head artifact is harmless/unused). Future fix option: teach stage-2 to
compute missing activations itself, or have the runner also process `test_tasks` for activations only.

### 2026-06-10 — 4th FV method: `train_varicl` (variable-ICL, train-pooled) — IMPLEMENTED, not yet run

A new head-selection + FV method alongside task_specific / train / train+test. Same two-stage
shape, but the prompt regime changes:
- **Variable ICL:** each prompt draws a random 1–10 demonstration count (deterministic per
  (task_index, query_idx); shard-invariant via global task_index).
- **Correctness filter, capped:** keep only correctly-answered prompts, **≤170 per task**.
- **Read position:** mean head activations AND the CIE intervention both read the **query
  predictive (last) token, T=-1** (user-confirmed). Single, length-independent position → mean
  activations stored as `(n_layers, n_heads, head_dim)` (no token axis), so a new save filename
  `<task>_mean_head_activations_varicl.pt` and adapted indexing are required (the canonical
  builder's `[...,-1]` would grab a scalar — see the three corrections below).
- **CIE:** variable ICL + shuffled labels, intervention at the query token, on the same
  correctly-answered query set; CIE prompts use seed `args.seed + cie_seed_offset` (default
  500000) so they differ from the activation pass but stay reproducible.
- **Pooling:** CIE averaged across the **20 train tasks** (same as `train_selected`), top-N heads.

**New sibling scripts (existing engines untouched):** `src/utils/varicl_utils.py`,
`src/eval_scripts/compute_multitask_varicl_heads.py`,
`src/eval_scripts/compute_all_task_fvs_varicl.py`,
`src/eval_scripts/run_multitask_varicl_all_tasks.sh`. New args: `--min_shots`, `--max_shots`,
`--max_successful_prompts`, `--cie_seed_offset`.

**Three corrections vs the canonical code (handled):** (1) the fixed-n_shots correctness filter
can't vary shots → new `varicl_correctness_filter`; (2) CIE `avg_activations[L,H,token_idx]` →
`avg_activations[L,H]` for single-position; (3) FV builder `mean_activations[L,H,-1]` →
`mean_activations[L,H]`.

**Outputs (when run):** `results/multitask_aie_heads_varicl/` (head set + per-task CIE/activations);
`results/function_vectors/gpt-j/train_varicl/` (FVs + heads.pt + selected_heads.json). Run commands
+ smoke verification: WORKLOG Stream D and plan `/root/.claude/plans/immutable-finding-boole.md`.

### 2026-06-10 — Direct full-dim (4096→4096) activation→FV ridge decoder (Stream C)

New experiment line, **deliberately PCA-free** (distinct from `regress_activation_to_fv_joint_pca*`).
For each (token position, layer) cell, fit one ridge map from the raw 4096-d residual activation
to the raw 4096-d `train_selected` FV. λ chosen by leave-one-train-task-out CV (20 folds); a single
standardizer fit on the pooled 20-train rows is reused everywhere; MSE reported natively in 4096-d
on the 7 test tasks (cc/pc excluded). 31 token positions × 29 layers = 899 cells.

- **Scripts:** `regress_activation_to_fv_fulldim_ridge.py` (worker, one shard = one ICL idx; GPU
  ridge via eigendecomposition reuse so the α grid is ~free), `run_fulldim_ridge_shards.sh` (tmux
  sharding, round-robin), `merge_fulldim_ridge_results.py` (combined CSV + heatmaps + summary).
- **Outputs:** `results/fulldim_ridge_activation_to_fv/`.
- **GOTCHA (now handled):** in the `4tokens` dir the final prompt token (`last_prompt_token`) has
  `icl_example_index = None`; the 3 label roles use `10`. Filter the final-token role on `None`.
- **Result:** best decode = final prompt token @ **layer 11**, test_mse **0.116**; clean layer bowl
  min at L11 (band L10–14); query position (ICL 10) beats earlier ICL demos. Embedding-layer
  pre-label/final tokens are constant across tasks (→ predict-the-mean baseline 0.217).
- **Metric note:** this 4096-d MSE is NOT comparable to the joint-PCA-space MSE (Open Q3); it *is*
  comparable to the reconstructed-4096-d MSE used by the k/layer sweeps.

### 2026-06-10 — Three head-selection methods + organized FV folder

FV derivation is two stages: (1) select top-N heads by CIE; (2) per task, sum
`out_proj(mean_head_activation[L,H,-1])` over the selected heads. The three methods
differ **only in stage 1** — which tasks' per-task CIE is pooled to rank heads.
Stage 2 (mean head activations) is shared. All three use top-10, GPT-J-6B,
`query_split=valid`, `demo_split=train` → directly comparable.

| Method | Stage-1 pool | Head set | Per-task FVs |
|---|---|---|---|
| task-specific | 1 task | `gptj_fv/<task>/<task>_indirect_effect.pt` | `gptj_fv/<task>/<task>_function_vector.pt` |
| train | 20 train tasks | `multitask_aie_heads/multitask_top_aie_heads.pt` | `gptj_fv_multitask_top10/` (29) |
| train+test | 29 tasks | `multitask_aie_heads_all_tasks/multitask_top_aie_heads.pt` | **BUILT** → `function_vectors/gpt-j/train_test_selected/` (29 + manifest) |

**Organized access point:** `results/function_vectors/{task_specific,train_selected,train_test_selected}/`
— a clean view with uniform `<task>_function_vector.pt` naming so any `--fv_root`
points straight at a method. See `results/function_vectors/README.md`.

**Symlinks, not hard moves.** `results/gptj_fv` is dual-role (task-specific FVs +
the shared `*_mean_head_activations.pt` cache + CIE + eval JSONs) and ~15 scripts
default to the old paths. Hard-moving would break the pipeline, so the FV folder
symlinks into the existing caches. Underlying dirs remain source of truth.
Reversible; revisit if we want a true single-source-of-truth migration (would
require rewiring all `--fv_root` defaults).

### 2026-06-10 — k-sweep outputs nested under `results/k_sweeps/`

All activation→FV k-sweep result dirs now live under `results/k_sweeps/` (was flat in
`results/`). Names trimmed of the redundant `k_sweep_` prefix (the parent conveys it):
- `results/k_sweeps/activation_to_fv_ols_multitask_top10_log2/` — original, 9 test tasks.
- `results/k_sweeps/activation_to_fv_ols_multitask_top10_log2_exclude_cc_pc/` — 7 test tasks
  (drops `country-currency`, `product-company`).
Default `--output_dir` in `sweep_k_activation_to_fv_ols_log2.py` and
`sweep_k_activation_to_fv_ols.py` updated to match; `run_config.json` self-paths fixed.

### 2026-06-10 — Direct k-sweep, third axis: fix k_act, sweep k_FV → joint optimum (16,16)

`sweep_k_activation_to_fv_direct_log2.py` now has `--fix_act_k K` (mutually exclusive with
`--fix_fv_k`): pins k_activations=K and sweeps k_FV (doubling grid reinterpreted as k_FV, capped
at fv_k_cap). Run at k_act=16 (7 tasks): **test MSE falls monotonically with k_FV, best at the
k_FV=16 cap (14/15 series), riding just above the recon floor (gap ~0.01–0.03).** No k_FV
overfitting — k_FV is bounded by discarded-FV-variance (floor) and by the FV-PCA rank cap (16),
not by overfitting. With 16 activation PCs the regression recovers any k_FV target nearly to the
floor. **Combined over all three direct cuts (coupled diagonal / fix k_FV / fix k_act): the joint
optimum is (k_act≈16, k_FV=16) — the corner.** k_act peaks ~16 then overfits; k_FV climbs
monotonically to its 16 cap. The four direct dirs in `results/k_sweeps/`:
`..._log2`, `..._log2_exclude_cc_pc`, `..._log2_fixedfvk16_exclude_cc_pc`,
`..._log2_fixedactk16_exclude_cc_pc`.

### 2026-06-10 — All k_sweeps are DIRECT; joint results removed

Per user decision, the k-sweep result dirs are now **direct-method only**. The three joint runs
were deleted and replaced by direct equivalents; `results/k_sweeps/` holds exactly:
- `activation_to_fv_direct_ols_multitask_top10_log2/` — coupled diagonal, full 9 test tasks.
- `activation_to_fv_direct_ols_multitask_top10_log2_exclude_cc_pc/` — coupled diagonal, 7 tasks.
- `activation_to_fv_direct_ols_multitask_top10_log2_fixedfvk16_exclude_cc_pc/` — k_FV pinned 16,
  sweep k_act, 7 tasks.
The joint *script* (`sweep_k_activation_to_fv_ols_log2.py`) is retained as a tool; only its
results were removed. **Settling finding (direct, k_FV=16):** k_act=1 → test MSE ≈ 0.18–0.19
(≈ predict-the-mean ~0.21), k_act=16 → ≈ 0.126. A single activation-PC does NOT recover the
16-dim FV — the joint method's apparent "1 PC suffices" was the 16 appended FV-basis features,
not genuine low-rank recoverability. Optimal remains k_act≈16 (first-label 32), k_FV=16, ICL5.

### 2026-06-10 — Regression definition: "direct" k_activations→k_FV (vs "joint")

Two regression definitions now coexist for the activation→FV PCA decode:
- **joint** (`sweep_k_activation_to_fv_ols_log2.py` + the `regress_*_joint_pca*` family): X and Y
  both projected onto the SAME concatenated [act-PCA (k_act) | FV-PCA (k_fv)] basis; OLS maps the
  (k_act+k_fv)-dim joint projection of the activation → the joint projection of the FV. The FV-PC
  half of the prediction is reconstructed for the 4096-d MSE.
- **direct** (NEW `sweep_k_activation_to_fv_direct_log2.py`): X = activation→act-PCA (k_act) only;
  Y = FV→FV-PCA (k_FV); OLS regresses R^{k_act}→R^{k_FV}; predicted FV-PCs reconstruct to 4096-d.

**Why they differ:** joint's feature vector additionally includes the activation projected onto
the FV basis (k_fv extra inputs), so joint fits slightly better. Measured: direct is
~0.002–0.003 higher test MSE than joint at the optimum (icl5/first 0.1261 vs 0.1232). The
**direct** setup is the cleaner "regress activation space → FV space" and should be preferred as
the headline definition; joint numbers were mildly optimistic. Optimal k (≈16, first-label 32),
ICL monotonicity, and token-role ordering are all unchanged between the two. The new script
mirrors all log2 flags incl. `--fix_fv_k`.

### 2026-06-10 — k is two knobs: k_activations vs k_FV (`--fix_fv_k`)

The joint-PCA regression has two independent dimensionalities: **k_FV** (FV-target PCs) and
**k_activations** (input PCs). The original log2 sweep coupled them below the cap
(`fv_k = min(k, 16)`), conflating the two. New flag `--fix_fv_k` on
`sweep_k_activation_to_fv_ols_log2.py` holds k_FV at `fv_k_cap` for every k, isolating
k_activations. **Finding (k_FV=16):** first/last-label tokens optimal at k_activations≈16–32;
pre-label token optimal at k_activations≈1–8 (more PCs hurt — little task signal there).
Beyond ~16–32 the activation side overfits regardless of token. Pinning k_FV=16 only changes
the k<16 regime vs the coupled run (it's strictly better there since the FV target isn't shrunk).
Canonical config: layer 11, last/first-label, ICL 5, **k_activations≈16, k_FV=16**.

### 2026-06-10 — Excluding cc/pc barely changes the regression k-sweep (FINDING)

Re-ran the log2 sweep without `country-currency` + `product-company` (whose
train-multitask-selected FVs steer much worse than task-specific). The activation→FV
regression test-MSE moved by only ~±0.001 and the optimal-k structure (k≈16–32, fv_k 16,
overfit past 32) was unchanged. **Implication:** "bad for steering" ≠ "hard to regress from
activations" — the two task-quality notions are decoupled, so the cc/pc weakness is not
visible in (and not fixable via) this reconstruction-MSE metric. If their weakness matters,
measure it on the steering/FV-effectiveness side (cf. `evaluate_heldout_multitask_head_fvs.py`).

### 2026-06-10 — Split provenance of all-tasks head set: RESOLVED (was Open Q1)

The all-tasks head set was computed on `query_split=valid` (not `train`). Evidence:
per-task files are suffixed `_valid` (`*_mean_indirect_effect_over_valid.pt`,
`*_per_prompt_indirect_effect_valid.pt`) — the suffix is emitted from the actual
split — and the runner passes `--query_split valid`. The metadata top-level
`query_split=train` is a **stale default**, almost certainly written by the
`--reduce` step (which didn't receive the arg). Both head sets therefore share
`query=valid, demo=train` and are comparable. **Follow-up (low priority):** fix the
`--reduce` path in `compute_multitask_top_aie_heads.py` to record the true split.

### DONE — train+test function vectors built (GPT-J)

Built 2026-06-10 with (reuses cached activations; zero forward passes):

```
python src/eval_scripts/compute_all_task_fvs_from_multitask_heads.py \
  --heads_path results/multitask_aie_heads_all_tasks/multitask_top_aie_heads.pt \
  --n_top_heads 10 --fv_root results/gptj_fv \
  --output_root results/function_vectors/gpt-j/train_test_selected \
  --task_manifest task_splits/abstractive_train_test_tasks_29.json
```

→ `results/function_vectors/gpt-j/train_test_selected/<task>/<task>_function_vector.pt`
(29) + `fv_manifest.json`. All three methods now complete for GPT-J. The builder also
supports `--tasks` (subset shard) and `--manifest_name` for **future models with no
cached activations** (each shard runs prompts at batch size).

### 2026-06-10 — Per-task head metadata (`selected_heads.json`)

Every task folder under `results/function_vectors/<model>/<method>/<task>/` carries a
`selected_heads.json` listing the `[layer, head, mean_indirect_effect]` heads that built
its FV, plus `selection_pool`. Generated (idempotent) by:

```
python src/eval_scripts/write_fv_head_metadata.py --model_root results/function_vectors/gpt-j --n_top_heads 10
```

`task_specific` reads each task's own `top_heads` (unique per task); the two multitask
methods read the shared `heads.pt` (top-N). Rerun after building FVs for a new model.

### 2026-06-10 — IMPORTANT: train vs train+test is degenerate at n=10

At top-10, `train_selected` and `train_test_selected` select the **same head set**
(same membership; rank order and CIE scores differ slightly). Since the FV is an
order-independent sum over selected heads, the `function_vector` tensors are **exactly
equal for all 29 tasks** (verified: global max|Δ|=0; norms match). NB the `.pt` *files*
are not byte-identical — `cmp` differs on dict metadata (`top_heads` order, paths) — only
the tensors match. They diverge only at larger
n (34/40 overlap at n=40; first differences ~n=11+). **Implication for experiments:** a
train-vs-train+test comparison at n=10 measures nothing — raise `n_top_heads` (rebuild
with a larger `--n_top_heads`) or use a per-task metric to study the selection-leakage
effect. `task_specific` is genuinely distinct from both at every n.

---

## Open questions (coordinator-flagged 2026-06-10)

1. **Canonical head set for held-out eval.** `evaluate_heldout_multitask_head_fvs.py`
   used the train-only (20-task) head set. Should held-out eval use the all-tasks
   set instead, or both as a comparison?

2. **One FV builder, not three.** `compute_task_fv_from_multitask_heads.py`,
   `compute_all_task_fvs_from_multitask_heads.py`, and `compute_fv_from_selected_heads.py`
   overlap. Leaning: `compute_all_task_fvs_from_multitask_heads.py` is the canonical
   batch builder (one model load, all 29 tasks, standard `<task>_function_vector.pt`
   filenames + `fv_manifest.json`) — it's the one in the build command above.
   `compute_fv_from_selected_heads.py` is the flexible single/ad-hoc builder (arbitrary
   head source, `--tasks`, non-standard `_fv_<tag>.pt` filename). Mark
   `compute_task_fv_from_multitask_heads.py` (single-task) legacy.

3. **Canonical regression metric.** Joint-PCA-space MSE vs reconstructed 4096-d
   FV-space MSE are not comparable. Pick one as the headline metric; report the other
   as secondary only.

4. **Canonical FV target for regression.** Some regression scripts target `gptj_fv`,
   others `gptj_fv_multitask_top10`. Decide which is the primary target (now also
   reachable as `function_vectors/task_specific` vs `function_vectors/train_selected`).

5. **Standardize ICL index range** across regression/sweep scripts (currently mixes
   1–4 and 2–5).

6. **Save per-prompt CIE effects on the baseline too?** The train-only head set lacks
   per-prompt effects, so `select_heads_from_cie_subset.py` cannot re-subset it. If
   subset analysis matters for the baseline, re-run with `--save_per_prompt_effects`.

7. **Smoke-test dirs.** `results/joint_pca_activation_to_fv_regression_smoke` and
   `results/pca_abstractive_fv_activation_scatter_smoke` — keep or remove?

---

## Process notes

- Workers: register a stream in WORKLOG.md before editing; don't co-edit the same
  source file without coordinating (per CLAUDE.md). `compute_multitask_top_aie_heads.py`
  and `evaluate_heldout_multitask_head_fvs.py` are the currently-modified shared files.
- Nothing is committed yet; all experimental scripts + results are uncommitted.
