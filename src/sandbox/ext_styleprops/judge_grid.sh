#!/bin/bash
# Judge every judged-stage rollout in the grid (CPU/API, resumable: verdicts cache in-file).
cd "$(dirname "$0")/../../.."
W=${1:-32}
for stage in refs headline bylayer; do
  for d in artifacts/style_properties/steering/grid/$stage/*/; do
    [ -d "$d" ] || continue
    echo "=== judging $d"
    python src/sandbox/ext_styleprops/judge_coherence.py --dir "$d" --workers "$W" || exit 1
  done
done
echo JUDGE_GRID_DONE
