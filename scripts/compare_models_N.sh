#!/usr/bin/env bash
# Drive scripts/compare_models_N.jl over the phantom sizes of the N sweep —
# strictly serially, one Julia process per size, never two GPU workloads at once.
#
#   ./scripts/compare_models_N.sh [output-directory]
#   MRSTAT_N_LIST="120 200" ./scripts/compare_models_N.sh /tmp/sweep
#
# Sizes default to 120 100 150 200: N=120 first, because it is the size
# scripts/compare_models.jl reports and therefore the run that says whether this
# machine is behaving as it did when the baseline numbers were taken. The sizes
# are exactly those that divide the checkpoint's 600 readouts into an integral
# number K = 600/N of phase-encode cycles.
#
# Each size gets its own process (see the script's header) and its own log; a
# size that fails does not stop the sweep, so the surviving sizes still produce
# a complete comparison.

set -u
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source env.sh

OUT="${1:-$PWD/scripts/output/compare_N}"
mkdir -p "$OUT"
echo "sweep output: $OUT"

for N in ${MRSTAT_N_LIST:-120 100 150 200}; do
    echo "=== N=$N started $(date -Is) ==="
    MRSTAT_N="$N" MRSTAT_OUT="$OUT" \
        julia --project=. scripts/compare_models_N.jl > "$OUT/N$N.log" 2>&1
    echo "=== N=$N exit=$? finished $(date -Is) ==="
done

echo "sweep complete $(date -Is)"
ls -l "$OUT"
