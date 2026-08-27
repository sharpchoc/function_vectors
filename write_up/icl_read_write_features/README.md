# ICL Read & Write Features — results write-up

Standalone copy of the Claude-artifact write-up of the 69-task GPT-J study
(read feature = L6 task-mean label-token activation; write feature = 37-head function
vector). Mirrors https://claude.ai/code/artifact/157ad0cd-e3aa-48a4-bcfe-cb9e327951df
as of 2026-08-24 (the pre-2026-08-24 artifact URL 21031c63 became inaccessible; the
write-up was republished at the new URL). Sections 1-10: full story including the
task-unique read code (S4-5), the carrier-gap hypothesis tests (App. I), and the
read->write rotation result (S9, App. J).

- `icl_read_write_features.html` — the complete document with all figures inlined as
  base64 (≈5 MB). Open it directly in a browser; no server or network needed.
- `writeup_template.html` — the source: same content with `{{IMG:<path>}}` placeholders
  that point into `results/69_task_run/`.
- `build_writeup.py` — regenerates the HTML from the template
  (`python3 write_up/icl_read_write_features/build_writeup.py`). Fails loudly if a
  referenced figure is missing.

To change the write-up, edit the template (not the built HTML) and rebuild.
Terminology follows `write_up/task_id_im_subspaces.md`.
