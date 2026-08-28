# results/chat_template_transfer — chat-template ICL transfer (live research branch)

**New research branch, started 2026-08-28 (user decision).** Question: does in-context task
learning transfer when the demonstrations are given through a **chat template** — each demo a
`user` turn (input) + `assistant` turn (label), the query a final `user` turn — instead of the
plain `Q:`/`A:` text format the rest of the repo uses?

This folder is deliberately **separate from `results/69_task_run/`** (the mainstream read/write-
feature line): same research programme, different model + prompt regime — never mix results
between the two. It is *not* exploratory/retired work either; it is live and forward-looking.

Model: **Qwen/Qwen2.5-7B-Instruct** (bf16; no thinking mode exists for this model family;
system-prompt handling is an experimental variable, see arms below).

## ext117_6shot_accuracy/

6-shot accuracy over the **117-task extended working pool** (`dataset_files/extended_tasks/
manifest.json`), mirroring the GPT-J extended n-shot sweep protocol exactly
(`src/eval_scripts/compute_extended_nshot_sampled.py`, user-locked 2026-08-13): identical
per-(task, n=6, i) demo/query sampling (seed `sha256(f"{task}|6|{i}")[:12]`, 50 prompts/task),
identical generation (T=1.0 pure ancestral sampling, `top_k=0, top_p=1.0, max_new_tokens=12`)
and metric (continuation cut at first newline, stripped, exact match vs stripped gold).

Three arms (exact rendered strings user-confirmed 2026-08-28 on the country-capital example):

| arm | prompt |
|---|---|
| `chat_blank_system` | `apply_chat_template` with an explicit EMPTY system message (Qwen2.5 otherwise auto-inserts its default "You are Qwen, …" system prompt) |
| `chat_no_system` | same, with the leading `<|im_start|>system\n<|im_end|>\n` stripped — no system block at all |
| `plain` | classic `Q:`/`A:` format (`word_pairs_to_prompt_data`, `prepend_bos_token=False`) — the model-change control |

GPT-J 6-shot reference joined from
`results/exploratory/general/extended_tasks_nshot_sweep/nshot_accuracy.csv`.

Files: `accuracy_6shot.csv` (per task), `summary.csv` (per arm), `bar_6shot_<arm>.png`
(ranked ascending bars, style of the GPT-J `nshot_bar_6shot.png`), `arm_comparison_6shot.png`.

## Scripts

- compute (GPU): `src/eval_scripts/eval_chat_template_ext117.py` (sharded `--shard_idx/--shard_n`,
  resumable per (format, task); raw generation records in
  `artifacts/chat_template_transfer/ext117_6shot/<format>/<task>.json`)
- aggregate + plots (CPU): `src/eval_scripts/plot_chat_template_ext117.py`
- pod driver: `logs/chat_template_transfer/pod_run.sh`
