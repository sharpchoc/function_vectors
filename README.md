# Function Vectors in Large Language Models
### [Project Website](https://functions.baulab.info) | [Arxiv Preprint](https://arxiv.org/abs/2310.15213) | [OpenReview](https://openreview.net/forum?id=AwyxtyMwaG)

This repository contains data and code for the paper: [Function Vectors in Large Language Models](https://arxiv.org/pdf/2310.15213).

<p align="left">
<img src="https://functions.baulab.info/images/Paper/fv-demonstrations.png" style="width:100%;"/>
</p> 

## Setup

We recommend using conda as a package manager. 
The environment used for this project can be found in the `fv_environment.yml` file.
To install, you can run: 
```
conda env create -f fv_environment.yml
conda activate fv
```

## Demo Notebook
Checkout `notebooks/fv_demo.ipynb` for a jupyter notebook with a demo of how to create a function vector and use it in different contexts.

## Data
The datasets used in our project can be found in the `dataset_files` folder.

## Repository layout

Results are split into two roots, configured centrally in `src/utils/paths.py` (override with
`FV_ARTIFACTS_ROOT` / `FV_RESULTS_ROOT` / `FV_LOGS_ROOT`):

- **`artifacts/`** (git-ignored) — recomputable intermediates: residual-activation captures, function
  vectors, attention-head selections, paired-task captures, scratch. The *only* tracked files here are the
  small head-selection metadata (`multitask_top_aie_heads*`, `heads.pt`, `heads_metadata.json`,
  `fv_manifest*.json`, `selected_heads.json`) — the output of the expensive CIE head-ranking, kept so
  function vectors can be rebuilt without recomputing it.
- **`results/`** (tracked) — study deliverables (figures + summary tables), organized by research direction:
  - `direction1_ambiguous/` — FV development on ambiguous ICL tasks (e.g. magnitude/identity).
  - `direction2_label_geometry/` — label- vs pre-label-token geometry and task-switch steering.
  - `direction3_fv_formation/` — FV formation across layers/positions (activation→FV regression,
    decodability and cosine heatmaps).
  - `steering_vector_comparison/` — steering effectiveness of different FV-construction methods.
  - `general/` — exploratory / uncategorized results.
- **`logs/`** (git-ignored) — run logs.

Scripts never hardcode these paths: import `ARTIFACTS_ROOT`, `RESULTS_ROOT`, `LOGS_ROOT` and the bucket
constants (`AMBIGUOUS_DIR`, `LABEL_GEOMETRY_DIR`, `FV_FORMATION_DIR`, `STEERING_COMPARISON_DIR`,
`GENERAL_DIR`) from `src/utils/paths.py`.

## Code
Our main evaluation scripts are contained in the `src` directory with sample script wrappers in `src/eval_scripts`.

Other main code is split into various util files:
- `eval_utils.py` contains code for evaluating function vectors in a variety of contexts
- `extract_utils.py`  contains functions for extracting function vectors and other relevant model activations.
- `intervention_utils.py` contains main functionality for intervening with function vectors during inference
- `model_utils.py` contains helpful functions for loading models & tokenizers from huggingface
- `prompt_utils.py` contains data loading and prompt creation functionality

## Citing our work
This work appeared at ICLR 2024. The paper can be cited as follows:

```bibtex
@inproceedings{todd2024function,
    title={Function Vectors in Large Language Models}, 
    author={Eric Todd and Millicent L. Li and Arnab Sen Sharma and Aaron Mueller and Byron C. Wallace and David Bau},
    booktitle={The Twelfth International Conference on Learning Representations},
    url={https://openreview.net/forum?id=AwyxtyMwaG},
    note={arXiv:2310.15213},
    year={2024},
}
