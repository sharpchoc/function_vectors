
## Stream: is the read->write map just a rotation? (2026-08-23, branch worktree-readwrite-map-understanding)

Status: done (rotation_vs_ridge.py, CPU fp64; same task-level X=label_resid_means[L6,L13],
Y=task FV as the centered-cossim step). Question: given similar centered cos histograms,
what does the linear map actually do?

Findings (rotation_vs_ridge_summary.csv, rotation_vs_ridge.png, crossfamily_cos_hists.png):
- Internal geometry is nearly CONGRUENT: centered pairwise cosines correlate pair-by-pair
  Pearson .93 (L6) / .96 (L13), gram-CKA .93/.95, centered-norm corr .79.
- But the two families occupy nearly ORTHOGONAL subspaces of activation space:
  feature-side alignment .014/.051; max principal cosine between the 90%-variance
  subspaces .26 (L6) / .41 (L13), median .09/.17.
- Cross-family cosines (matched cos(m_A,v_A), centered): L6 mean .076, L13 mean .195,
  mismatched pairs ~0; matched above mismatched-p95 for 40/69 (L6), 57/69 (L13).
- Procrustes vs ridge, held-out testmean R2: L13 pure rotation .625 vs ridge .657 (95%);
  L6 rotation .482, rotation+global-scale .586, ridge .642 (91%). Train-mean baseline -.08.
- Ridge-map spectrum on the train span decays gradually, not flat (s10/s1 .68-.72,
  s40/s1 ~.42) -> the residual ~5-9% is anisotropic gain, but the dominant component
  is a rigid rotation (+ global scale at L6).

ANSWER: yes, to first order the map re-embeds an almost-unchanged task geometry into a
nearly orthogonal subspace — a rotation; ridge adds modest direction-dependent gain.
Note: cross-family cosines answer the user's "distribution of cos(read, write) per task"
question — stored in rotation_vs_ridge_spectra.npz + crossfamily_cos_hists.png.
