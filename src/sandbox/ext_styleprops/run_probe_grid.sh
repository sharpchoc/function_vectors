#!/bin/bash
# Drive probe_grid.py to completion: per-property caching makes each invocation resume
# where the previous one stopped (guards against silent kills of long runs).
cd "$(dirname "$0")/../../.."
for i in $(seq 1 20); do
    python src/sandbox/ext_styleprops/probe_grid.py 2>&1 | grep -vE "ConvergenceWarning|warnings.warn|STOP: TOTAL"
    if [ -f results/style_properties/decodability/probe_grid.csv ]; then
        echo PROBE_GRID_DONE
        exit 0
    fi
    echo "attempt $i incomplete, resuming..."
done
echo PROBE_GRID_FAILED
