# ICL Read & Write Features — results write-up

Standalone copy of the Claude-artifact write-up of the 69-task GPT-J study
(read feature = L6 task-mean label-token activation; write feature = 37-head function
vector). Mirrors https://claude.ai/code/artifact/21031c63-73ae-48a1-b36e-6c04f17eeac2
as of 2026-08-20.

- `icl_read_write_features.html` — the complete document with all figures inlined as
  base64 (≈5 MB). Open it directly in a browser; no server or network needed.
- `writeup_template.html` — the source: same content with `{{IMG:<path>}}` placeholders
  that point into `results/69_task_run/`.
- `build_writeup.py` — regenerates the HTML from the template
  (`python3 write_up/icl_read_write_features/build_writeup.py`). Fails loudly if a
  referenced figure is missing.

To change the write-up, edit the template (not the built HTML) and rebuild.
Terminology follows `write_up/task_id_im_subspaces.md`.
