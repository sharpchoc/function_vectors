#!/bin/bash
# Generate the full base corpus: general batch + 4 seeded batches.
set -e
cd "$(dirname "$0")/../../.."
python src/sandbox/ext_styleprops/gen_corpus.py --per_topic 2 --workers 8
for b in num dlg dsh ukv; do
    python src/sandbox/ext_styleprops/gen_corpus.py --batch "$b" --n_topics 30 --per_topic 4 --workers 8
done
echo ALL_BATCHES_DONE
