# results/style_translation/read_write_map — is there a shared read→write linear map across conventions?

Question (user, 2026-09-11): in the 69-task study a ridge map from the read feature (label-token mean) to the
write feature (function vector) generalised to held-out tasks (held-out R² ≈ .5, cos ≈ .8–.9). Does the same
hold here — from the evidence-token read activation at layer 0 (embeddings) to the cue-token write activation
at layer 24 — when the map is fit on the lexically identical families and scored on the lexically diverse ones?

**Verdict: no.** Three tests, all negative.

## Data
`capture_prompt_pairs.py` (GPU, one pass per k = 4 prompt, 200 texts × 2 poles × 17 families): per prompt the mean
residual over all evidence tokens at layers {0,2,4,8,12,24} and the cue-token (last position) residual at layers
{12,16,20,24} → `artifacts/style_translation/prompt_pairs/<family>.npz` (fp16). Consistency: the paired k = 4 cue
difference has cos .93–1.00 with the stored layer-24 steering vector in every family.

## Tests (conventions as in results/69_task_run/read_write_relationship/linear_mapping: dual ridge with intercept,
train-mean centring, λ by leave-one-family-out CV; R² vs test mean and vs train mean; within-family R² removes each
held-out family's mean; "convention vector" = held-out nat centroid − alt centroid, predicted vs true)

| test | held-out R² (train-mean) | within-family R² | cos convention vector |
|---|---|---|---|
| 69-task map applied off the shelf (block-1 read → FV, λ = 17.8) — cos to the L24 write vector | — | — | .09 (identical) / .09 (diverse); only sentence_caps .41, all_caps .30 above a shuffled-task null of ±.02–.04 |
| ridge fit here: 11 identical → 6 diverse, CV λ = 31.6 | −.03 (test-mean −.14) | .00 | .21 (ise_ize .35, brit_t_past .29, us_uk .26, others .07–.16) |
| same, λ ×10 / ×100 / ×1000 | .01 / .00 / .00 | .00 | .17 / .12 / .12 |
| shuffled pairing control (X rows permuted in training) | .00 | .00 | .02 |
| reverse: 6 diverse → 11 identical | .00 | .00 | .06 |
| identical + 3 diverse → other 3 diverse, all 20 splits (`read_write_ridge_mixed.py`) | +.01 (test-mean −.11) | .00 | .26 (us_uk .39, ise_ize .40, brit_t_past .29, num_words .26, ordinal_words .19, contractions .02) vs identical-only on the same splits .21 |

Files: `ridge_summary.csv` (main run; parsed from `logs/read_write_ridge.log` — the run's leave-one-family-out-over-17
and layer-sweep stages were stopped for CPU and are not reported), `mixed_splits.csv`, `mixed_splits_per_family.csv`.

## Reading
- Prompt-level prediction is absent in every setting: the map cannot say which text or which pole a held-out prompt
  is (within-family R² = 0), and CV drives λ to heavy shrinkage (100–1000 in the mixed splits).
- The small convention-vector cosines (.2–.4) come from near-duplicate axes already visible in the raw read vectors
  (us_uk–ise_ize cos .72 at L4, num_words–ordinal_words .49): the map memorises a neighbouring direction rather than
  learning a general read→write relation; contractions, which shares an axis with nothing, gets .02.
- Why it differs from the 69-task result: there both features are task-identity codes related by one rotation. Here the
  layer-0 read feature is a token-identity code (for fixed-marker families literally the embedding difference of two
  tokens), and cos(write L24, read L0) is .00 in all 17 families before any regression; even same-layer read/write
  differences overlap only at .17 (identical) / .29 (diverse).
- Not run: the read side at layers 4–8 (where evidence-token steering showed information beyond token identity). Given the
  same-layer cosines it is not expected to change the verdict.

## Layer sweep (2026-09-12, `read_write_ridge_layers.py`, identical → diverse, λ by leave-one-family-out CV on 1e-1…1e5; `layer_sweep.csv`)

| read → write | held-out R² (train-mean) | within-family R² | cos convention vector (mean of 6) | per family (us_uk / ise_ize / brit_t_past / num_words / ordinal_words / contractions) |
|---|---|---|---|---|
| L0 → L24 | −.03 | .00 | .21 | .26 / .35 / .29 / .12 / .16 / .07 |
| L2 → L24 | −.05 | .00 | .26 | .43 / .37 / .22 / .17 / .20 / .18 |
| L4 → L24 | −.04 | .00 | .30 | .47 / .44 / .34 / .22 / .23 / .10 |
| L8 → L24 | −.03 | .00 | .33 | .49 / .43 / .37 / .31 / .26 / .14 |
| L12 → L24 | −.02 | .00 | .38 | .51 / .50 / .45 / .35 / .29 / .18 |
| L24 → L24 | .00 | .01 | .38 | .51 / .43 / .41 / .38 / .37 / .16 |
| L4 → L12 / L16 / L20 | −.15 / −.09 / −.05 | .00 | .22 / .31 / .29 | — |
| L8 → L12 / L16 / L20 | −.13 / −.08 / −.04 | .00 | .26 / .34 / .32 | — |
| L0 → L12 / L16 / L20 | −.17 / −.11 / −.06 | .00 | .13 / .20 / .18 | — |

CV picked λ = 10 in every setting. Deeper read layers raise the transferred convention *direction* monotonically
(.21 → .38, plateau from L12), concentrated in the families that share an axis with a training neighbour; earlier
write layers are uniformly worse than L24. Prompt-level R² stays at zero everywhere, and the R² of the predicted
convention vector is ≤ 0 in every setting. The layer choice does not change the verdict.

## Why the map fails: coverage, not inconsistency (2026-09-12)

Family-centroid differences (nat − alt) from `prompt_pairs`, read at L12 and write at L24:
- **Each convention is its own axis.** Participation ratio of the 17 unit read differences 15.1 / 17, of the 17 write
  differences 15.0 / 17; mean |cos| between families .13 (read L12), .05 (read L0), .11 (write L24).
- **Held-out directions lie only partly in the training span** (fraction of norm inside the span of the 11 identical
  families' differences; read L12 / write L24): us_uk .68/.52, ise_ize .62/.54, brit_t_past .56/.62, contractions .49/.35,
  num_words .43/.46, ordinal_words .39/.44 (read L0: .11–.34). The ridge's convention cosines (.51/.50/.45/.18/.35/.29)
  track this ceiling; predicted norms are .3–.5 of the true ones. Within the training set the leave-one-family-out CV
  R² is only .06–.09 — the map fails among the identical families for the same reason.
- **Where families overlap, read and write agree.** corr(cos read, cos write) over the 136 family pairs = .84 (L12) / .68
  (L0): ise_ize–us_uk .83 → .88, brit_t_past–ise_ize .62 → .72, brit_t_past–us_uk .51 → .67, num_words–ordinal_words
  .57 → .65, us_uk–whilst .58 → .42. The read→write relation is angle-preserving where it can be observed; with 11
  near-orthogonal training directions in a ≥ 15-dimensional code it cannot be learned as a linear map.
- Contrast with the 69-task line: 55 training tasks whose read features had cross-task cosines ≈ .7, so held-out tasks lay
  inside the training span. Same circuit stage (layers), different coverage. The 27-family corpus (21 training
  directions) is the natural next test.

### Coverage, measured against the training PROMPTS (not just centroids), and the 69-task comparison (2026-09-12)

Fraction of a held-out convention direction (nat − alt centroid difference) inside the top-k principal components of
the 4400 training prompts (centred), range over the 6 diverse families; chance for a random direction = √(k/4096):

| k | read L12 | read L0 | write L24 | chance |
|---|---|---|---|---|
| 11 | .35–.49 | .11–.28 | .10–.15 | .05 |
| 50 | .47–.69 | .23–.39 | .31–.43 | .11 |
| 200 | .53–.75 | .35–.49 | .48–.68 | .22 |
| 1000 | .69–.86 | .66–.80 | .72–.81 | .49 |
| span of the 11 training centroid differences | .39–.68 | .11–.34 | .35–.62 | .05 |

Same statistic for the 69-task line (55 train / 14 held-out tasks, features centred on the train mean): held-out read
(L6) inside the span of the 55 training task means .78 (min .48); held-out task FV inside the span of the 55 training
FVs .86 (min .64); inside the top-50 / top-200 PCs of the 8250 training per-prompt FVs .86 / .91. Chance .11.
NB the 69 tasks are NOT more mutually similar once centred: mean |cos| between training tasks .15 (read L6) / .21 (FV),
participation ratios 39 / 34 of 55 — the earlier "cross-task cos ≈ .7" was the uncentred (shared-mean) figure. The
difference is sampling density: 55 tasks in a ~35-dim code vs 11 conventions in a ~15-dim code. A ridge recovers the
in-span part of a held-out direction in both studies; here that part is .4–.6 and the observed cosines (.2–.5) follow it.
Before fitting on the 27-family corpus, compute this coverage number for each candidate held-out family first.
