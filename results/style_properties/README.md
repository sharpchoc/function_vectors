# results/style_properties — free-form style-property read/write study

Extension of the 69-task read/write-feature line (results/69_task_run) to a new
substrate: **free-form text with a binary stylistic property** (all-lowercase sentence
starts, UK vs US spelling, double space after periods, ...). Property manifestations play
the role of demo *labels* (evidence / read sites); the last token before a
manifestation's nat/alt divergence — the *point of no return*, identity-matched across
the minimal-pair twins — plays the role of the *cue* (write site).
Full staged design + user decisions of 2026-09-01: see `adjudication_memo.md` here and
WORKLOG stream "style-properties".

Model: GPT-J-6B. Readout: **sampled adherence only** (T=1 seeded generation classified
nat / alt / unscorable at cue tokens) — user decision 2026-09-01; no logit-diff
readouts anywhere in reported results.

| Folder / file | Contents |
|---|---|
| `dataset_audit.csv` | Stage A2 tokenization audit: eligible/used docs, sites, cue-token mismatch drops, nat/alt token-count deltas, k coverage. |
| `behavioral_prescreen/` | Stage A4: sampled adherence by (context polarity, k prior manifestations) and by token distance since last manifestation; `prescreen_summary.csv` with the proposed pass/fail gate. 16/17 properties pass (whilst fails); then PRUNED 2026-09-02: ellipsis (one-sided classifier), brit_t_past + ise_ize (<15% scorable floor) → pool of 13. |
| `decodability/` | Stage B: polarity logistic probes over (site role × 29 layers), doc-level splits. Evidence tokens are L0-decodable (token identity — the quantified shortcut); cue and background tokens are exactly 0.5 at L0 (identity-matched control) yet 0.80–1.0 decodable in context. Decision decodability jumps 0.5 → 0.8–1.0 after ONE manifestation. Background decodability persists ≥ 90 tokens for typography/spelling properties and decays to ~0.6 for others — persistent style-state with property-dependent half-life. |

Data: `dataset_files/style_properties/` (base corpus + per-property minimal-pair
datasets); scripts: `src/sandbox/ext_styleprops/`; run artifacts:
`artifacts/style_properties/` (gitignored). Pool artifact after the gate:
`task_splits/style_properties_pool.json`.
| `steering/` | **Start with `cue/headline_cue.png`** — the write-feature analog: a per-property vector (mean cue-token activation under the ALT convention minus under the standard one) added at the CUE token alone flips GPT-J's convention for 12/13 properties (0.90–1.00 from ≤0.21 baselines; counterfactual-property controls at baseline; reverse direction works). `cue/steering_by_layer_cue.png`: injection layer, peaks L16–20. Cue-injected sparse head gates are genuinely sparse (19–167 heads) and steer 0.77–1.0 (`cue/sparse_heads_cue_summary.csv`; overlap with the 37 ICL heads ≈ 2× chance). Evidence-token (read-side) steering: `headline.png`, `steering_by_layer.png`. Appendix: `appendix_all_conditions.png`, `cue/appendix_cue_all_conditions.png`, `sweep_heatmaps.png`. |
