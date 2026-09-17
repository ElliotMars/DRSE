#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SETTINGS="ETTm1:24 WTH:48 ECL:48"
METHODS="${METHODS:-fsnet onenet dyname}"
SETTINGS="${SETTINGS:-$DEFAULT_SETTINGS}"
SEEDS="${SEEDS:-0 1 2}"
GPU_IDS="${GPU_IDS:-0}"
MAX_PER_GPU="${MAX_PER_GPU:-1}"
EXECUTE="${EXECUTE:-0}"

if [[ "$EXECUTE" != "1" ]]; then
    echo "[DRY RUN] Printing the 27 main-text fairness-control commands."
fi

for seed in $SEEDS; do
    METHODS="$METHODS" \
    SETTINGS="$SETTINGS" \
    SEED="$seed" \
    GPU_IDS="$GPU_IDS" \
    MAX_PER_GPU="$MAX_PER_GPU" \
    EXECUTE="$EXECUTE" \
    bash "$SCRIPT_DIR/run_progressive_baseline_control.sh"
done
