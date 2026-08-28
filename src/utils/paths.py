"""Centralized result/artifact paths for the Function Vectors repo.

Two roots keep recomputable intermediates out of git while the study deliverables are committed:
  * ARTIFACTS_ROOT  (gitignored) — activations, captures, function vectors, head selections, scratch.
  * RESULTS_ROOT    (tracked)    — study deliverables, bucketed by research direction.
  * LOGS_ROOT       (gitignored) — run logs.

All roots are anchored to the repo root (NOT the current working directory), so scripts resolve the
same paths regardless of where they are launched from. Override any root via an env var, e.g.
`FV_ARTIFACTS_ROOT=/scratch/fv_artifacts python ...`.

Usage:
    from utils.paths import ARTIFACTS_ROOT, LABEL_GEOMETRY_DIR
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "gptj_fv")
    p.add_argument("--output_root", type=Path, default=LABEL_GEOMETRY_DIR / "oneshot_switch_logit")
"""
import os
from pathlib import Path

# repo root = two levels up from this file (src/utils/paths.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _root(env_var: str, default_name: str) -> Path:
    val = os.environ.get(env_var)
    return Path(val) if val else REPO_ROOT / default_name


ARTIFACTS_ROOT = _root("FV_ARTIFACTS_ROOT", "artifacts")   # gitignored intermediates
RESULTS_ROOT = _root("FV_RESULTS_ROOT", "results")         # tracked study deliverables
LOGS_ROOT = _root("FV_LOGS_ROOT", "logs")                  # gitignored run logs

# Mainstream buckets under results/ (tracked) — the settled research line (DECISIONS 2026-08-28):
# the 69-task read/write-feature study and steering-vector-method comparisons.
TASK69_RUN_DIR = RESULTS_ROOT / "69_task_run"  # canonical 69-task pool studies (DECISIONS 2026-08-16)
STEERING_COMPARISON_DIR = RESULTS_ROOT / "steering_vector_comparison"

# Live new research branch (2026-08-28): chat-template transfer of the ICL line
# (Qwen2.5-7B-Instruct, demos as user/assistant chat turns instead of Q:/A:).
# Deliberately separate from 69_task_run — do not mix results between the two.
CHAT_TEMPLATE_TRANSFER_DIR = RESULTS_ROOT / "chat_template_transfer"

# Exploratory buckets — research directions that did not pan out, quarantined under
# results/exploratory/ on 2026-08-28 (DECISIONS entry of that date). Kept for possible
# later revisits; do NOT build new mainstream results on them without user promotion.
EXPLORATORY_ROOT = RESULTS_ROOT / "exploratory"
AMBIGUOUS_DIR = EXPLORATORY_ROOT / "direction1_ambiguous"
LABEL_GEOMETRY_DIR = EXPLORATORY_ROOT / "direction2_label_geometry"
FV_FORMATION_DIR = EXPLORATORY_ROOT / "direction3_fv_formation"
GENERAL_DIR = EXPLORATORY_ROOT / "general"

__all__ = [
    "REPO_ROOT", "ARTIFACTS_ROOT", "RESULTS_ROOT", "LOGS_ROOT",
    "TASK69_RUN_DIR", "STEERING_COMPARISON_DIR", "CHAT_TEMPLATE_TRANSFER_DIR", "EXPLORATORY_ROOT",
    "AMBIGUOUS_DIR", "LABEL_GEOMETRY_DIR", "FV_FORMATION_DIR", "GENERAL_DIR",
]
