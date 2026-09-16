#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
EXECUTE="${EXECUTE:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-result/recurring_drift_maintext}"
SEEDS="${SEEDS:-0 1 2}"

SEQ_LEN="${SEQ_LEN:-60}"
PRED_LEN="${PRED_LEN:-24}"
CHANNELS="${CHANNELS:-3}"
A1_LENGTH="${A1_LENGTH:-300}"
B_LENGTH="${B_LENGTH:-300}"
A2_LENGTH="${A2_LENGTH:-300}"
NOISE_STD="${NOISE_STD:-0.05}"
NUM_EXPERTS="${NUM_EXPERTS:-4}"
REGIME_SEPARATION="${REGIME_SEPARATION:-1.0}"
STRATEGY="${STRATEGY:-subspace}"
RECOVERY_TOLERANCE="${RECOVERY_TOLERANCE:-0.2}"
RECOVERY_HOLD_STEPS="${RECOVERY_HOLD_STEPS:-4}"
ROLLING_WINDOW="${ROLLING_WINDOW:-12}"

for seed in $SEEDS; do
    for variant in full no_direction no_recovery; do
        variant_args=()
        if [[ "$variant" == "no_direction" ]]; then
            variant_args+=(--disable_directional_recovery)
        elif [[ "$variant" == "no_recovery" ]]; then
            variant_args+=(--disable_recovery)
        fi
        command=(
            "$PYTHON_BIN" -m utils.synthetic_drift_benchmark
            --output_dir "$OUTPUT_ROOT"
            --drift_type recurring
            --strategy "$STRATEGY"
            --seq_len "$SEQ_LEN"
            --pred_len "$PRED_LEN"
            --channels "$CHANNELS"
            --a1_length "$A1_LENGTH"
            --b_length "$B_LENGTH"
            --a2_length "$A2_LENGTH"
            --noise_std "$NOISE_STD"
            --num_experts "$NUM_EXPERTS"
            --regime_separation "$REGIME_SEPARATION"
            --recovery_tolerance "$RECOVERY_TOLERANCE"
            --recovery_hold_steps "$RECOVERY_HOLD_STEPS"
            --rolling_window "$ROLLING_WINDOW"
            --seed "$seed"
            "${variant_args[@]}"
        )
        printf "[seed=%s variant=%s] " "$seed" "$variant"
        printf "%q " "${command[@]}"
        printf "\n"
        if [[ "$EXECUTE" == "1" ]]; then
            "${command[@]}"
        fi
    done
done

if [[ "$EXECUTE" != "1" ]]; then
    echo "Dry-run only. Set EXECUTE=1 to run the 3-seed main-text experiment."
fi
