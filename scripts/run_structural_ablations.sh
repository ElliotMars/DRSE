#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
EXECUTE="${EXECUTE:-0}"

if [[ "$#" -eq 0 ]]; then
    echo "Usage: $0 <shared main.py arguments>"
    echo "Default is dry-run; set EXECUTE=1 to launch the generated commands."
    exit 2
fi

BASE_ARGS=("$@")

run_variant() {
    local label="$1"
    shift
    local -a command=(
        "$PYTHON_BIN" -u main.py
        "${BASE_ARGS[@]}"
        "$@"
        --checkpoint_tag "structural_${label}"
    )

    printf "[%s] " "$label"
    printf "%q " "${command[@]}"
    printf "\n"
    if [[ "$EXECUTE" == "1" ]]; then
        "${command[@]}"
    fi
}

run_variant composition_fsnet --expert_composition fsnet
run_variant composition_fsnet_time --expert_composition fsnet_time
run_variant composition_mixed --expert_composition mixed

run_variant tsb_smooth_off_filter_off \
    --expert_update_strategy tsb \
    --disable_tsb_smoothing \
    --disable_tsb_conflict_filter
run_variant tsb_smooth_on_filter_off \
    --expert_update_strategy tsb \
    --disable_tsb_conflict_filter
run_variant tsb_smooth_off_filter_on \
    --expert_update_strategy tsb \
    --disable_tsb_smoothing
run_variant tsb_smooth_on_filter_on --expert_update_strategy tsb

run_variant controller_fixed --adaptive_controller fixed
run_variant controller_dynamic --adaptive_controller dynamic
