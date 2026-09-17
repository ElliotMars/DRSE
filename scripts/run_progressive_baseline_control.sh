#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON:-python}"
ROOT_PATH="${ROOT_PATH:-$PROJECT_ROOT/data/}"
METHODS="${METHODS:-fsnet onenet dyname}"
DATASETS="${DATASETS:-ETTm1 WTH ECL}"
LENS="${LENS:-24 48}"
SETTINGS="${SETTINGS:-}"
SEED="${SEED:-0}"
GPU_IDS="${GPU_IDS:-0}"
MAX_PER_GPU="${MAX_PER_GPU:-1}"
EXECUTE="${EXECUTE:-1}"
PRETRAIN_MODE="${PRETRAIN_MODE:-retrain}"
MAX_ONLINE_STEPS="${MAX_ONLINE_STEPS:--1}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/log/progressive_baseline_control_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

read -r -a methods <<< "$METHODS"
read -r -a gpu_ids <<< "$GPU_IDS"
if (( ${#gpu_ids[@]} == 0 )); then
    echo "GPU_IDS must contain at least one id" >&2
    exit 2
fi
if (( MAX_PER_GPU < 1 )); then
    echo "MAX_PER_GPU must be at least 1" >&2
    exit 2
fi

pairs=()
if [[ -n "${SETTINGS//[[:space:]]/}" ]]; then
    read -r -a pairs <<< "$SETTINGS"
else
    read -r -a datasets <<< "$DATASETS"
    read -r -a horizons <<< "$LENS"
    for dataset in "${datasets[@]}"; do
        for pred_len in "${horizons[@]}"; do
            pairs+=("${dataset}:${pred_len}")
        done
    done
fi

slots=()
for gpu in "${gpu_ids[@]}"; do
    for ((i = 0; i < MAX_PER_GPU; i++)); do
        slots+=("$gpu")
    done
done
pids=()
job_index=0

for method in "${methods[@]}"; do
    case "$method" in
        fsnet|onenet|dyname) ;;
        *)
            echo "Unsupported progressive baseline method: $method" >&2
            exit 2
            ;;
    esac
    for pair in "${pairs[@]}"; do
        if [[ "$pair" != *:* ]]; then
            echo "Invalid setting '$pair'; expected DATASET:PRED_LEN" >&2
            exit 2
        fi
        dataset="${pair%%:*}"
        pred_len="${pair##*:}"
        learning_rate="1e-3"
        if [[ "$dataset" == "ECL" ]]; then
            learning_rate="3e-3"
        fi

        slot_index=$((job_index % ${#slots[@]}))
        gpu="${slots[$slot_index]}"
        log="$LOG_DIR/${method}_progfb_${dataset}_pl${pred_len}_seed${SEED}.out"
        cmd=(
            "$PYTHON_BIN" -u main.py
            --method "$method"
            --root_path "$ROOT_PATH"
            --data "$dataset"
            --features M
            --seq_len 60
            --label_len 0
            --pred_len "$pred_len"
            --test_bsz 1
            --batch_size 32
            --itr 1
            --seed "$SEED"
            --train_epochs 15
            --patience 3
            --learning_rate "$learning_rate"
            --online_learning full
            --progressive_baseline_fb
            --baseline_online_lr 1e-3
            --dyname_period_num 2
            --dyname_krr_lambda 1e-4
            --dyname_krr_train_num 8
            --dyname_past_num 672
            --dyname_beta 0.3
            --dyname_delta 0.01
            --dyname_online_lr 1e-3
            --pretrain_mode "$PRETRAIN_MODE"
            --max_online_steps "$MAX_ONLINE_STEPS"
        )

        printf '[COMMAND] CUDA_VISIBLE_DEVICES=%q ' "$gpu"
        printf '%q ' "${cmd[@]}"
        printf '> %q 2>&1\n' "$log"

        if [[ "$EXECUTE" == "1" ]]; then
            previous_pid="${pids[$slot_index]:-}"
            if [[ -n "$previous_pid" ]]; then
                wait "$previous_pid"
            fi
            printf '[COMMAND] CUDA_VISIBLE_DEVICES=%q ' "$gpu" > "$log"
            printf '%q ' "${cmd[@]}" >> "$log"
            printf '\n' >> "$log"
            CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >> "$log" 2>&1 &
            pids[$slot_index]=$!
        fi
        job_index=$((job_index + 1))
    done
done

if [[ "$EXECUTE" == "1" ]]; then
    for pid in "${pids[@]}"; do
        if [[ -n "$pid" ]]; then
            wait "$pid"
        fi
    done
    echo "Progressive baseline runs finished. log_dir=$LOG_DIR"
else
    echo "Dry run only. Set EXECUTE=1 to launch commands."
fi
