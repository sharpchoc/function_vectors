# head_intensive_pruned_tasks

21 extended tasks moved out of the working pool on 2026-08-16. They pass the 6-shot >= 0.30
competence filter but the pooled sparse-optimization FV (39 heads, extended_steerable_90 train
set) fails to steer them zero-shot (best-layer acc < 0.4). Diagnostics showed they are
head-intensive and/or copy-and-modify tasks (the answer largely copies input tokens with a small
edit the pooled head set cannot encode; task-specific heads rescue most of them). Excluded to keep
the working pool simple; to be revisited separately.

Provenance: results/sandbox/ext_steerability/ (train_tasks_summary.csv, failing_analysis_*.csv),
task_splits/extended_steerable_69_prunedfail.json. NOTE: make_ext_split.py --expect_n was 90 on
the pre-move manifest; on the pruned manifest the equivalent filter yields 69.
