#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
EXECUTE="${EXECUTE:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-result/synthetic_drift}"
STRATEGIES="${STRATEGIES:-plain tsb subspace hybrid}"
DRIFT_TYPES="${DRIFT_TYPES:-recurring}"
SEED="${SEED:-0}"
SEQ_LEN="${SEQ_LEN:-12}"
PRED_LEN="${PRED_LEN:-3}"
CHANNELS="${CHANNELS:-3}"
TOTAL_LENGTH="${TOTAL_LENGTH:-120}"
NOISE_STD="${NOISE_STD:-0.05}"
TRANSITION_WINDOW="${TRANSITION_WINDOW:-}"
SHOCK_DURATION="${SHOCK_DURATION:-}"
NUM_EXPERTS="${NUM_EXPERTS:-3}"
SYNTHETIC_DRIFT_CHANNELS="${SYNTHETIC_DRIFT_CHANNELS:-}"

optional_args=()
if [[ -n "$TRANSITION_WINDOW" ]]; then
    optional_args+=(--transition_window "$TRANSITION_WINDOW")
fi
if [[ -n "$SHOCK_DURATION" ]]; then
    optional_args+=(--shock_duration "$SHOCK_DURATION")
fi
if [[ -n "$SYNTHETIC_DRIFT_CHANNELS" ]]; then
    optional_args+=(--synthetic_drift_channels "$SYNTHETIC_DRIFT_CHANNELS")
fi
if [[ "${DISABLE_RECOVERY:-0}" == "1" ]]; then
    optional_args+=(--disable_recovery)
fi
if [[ "${DISABLE_VERSION_AWARENESS:-0}" == "1" ]]; then
    optional_args+=(--disable_version_awareness)
fi
if [[ "${DISABLE_Z_CORRECTION:-0}" == "1" ]]; then
    optional_args+=(--disable_z_correction)
fi

for drift_type in $DRIFT_TYPES; do
    for strategy in $STRATEGIES; do
        command=(
            "$PYTHON_BIN" -m utils.synthetic_drift_benchmark
            --output_dir "$OUTPUT_ROOT"
            --drift_type "$drift_type"
            --strategy "$strategy"
            --seq_len "$SEQ_LEN"
            --pred_len "$PRED_LEN"
            --channels "$CHANNELS"
            --total_length "$TOTAL_LENGTH"
            --noise_std "$NOISE_STD"
            --num_experts "$NUM_EXPERTS"
            --seed "$SEED"
            "${optional_args[@]}"
        )
        printf "[%s/%s] " "$drift_type" "$strategy"
        printf "%q " "${command[@]}"
        printf "\n"
        if [[ "$EXECUTE" == "1" ]]; then
            "${command[@]}"
        fi
    done
done

if [[ "$EXECUTE" != "1" ]]; then
    echo "Dry-run only. Set EXECUTE=1 to run the short synthetic configurations."
fi
