#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "/opt/data/private/envs/Online/bin/python" ]]; then
    PYTHON_BIN="/opt/data/private/envs/Online/bin/python"
elif [[ -x "/root/miniconda3/envs/online/bin/python" ]]; then
    PYTHON_BIN="/root/miniconda3/envs/online/bin/python"
else
    PYTHON_BIN="python"
fi

DATASET="${DATASET:-ETTh2}"
PRED_LEN="${PRED_LEN:-24}"
GPU_ID="${GPU_ID:-0}"
MAX_ONLINE_STEPS="${MAX_ONLINE_STEPS:-200}"
STABLE_BUFFER_SIZE="${STABLE_BUFFER_SIZE:-8}"
RECOVERY_BUFFER_SIZE="${RECOVERY_BUFFER_SIZE:-4}"
RECOVERY_BATCH_SIZE="${RECOVERY_BATCH_SIZE:-1}"
SUBSPACE_RANK="${SUBSPACE_RANK:-4}"
SUBSPACE_MAX_RANK="${SUBSPACE_MAX_RANK:-8}"
SUBSPACE_REFRESH_INTERVAL="${SUBSPACE_REFRESH_INTERVAL:-20}"
EXPERT_UPDATE_STRATEGY="${EXPERT_UPDATE_STRATEGY:-subspace}"
ROUTER_GRANULARITY="${ROUTER_GRANULARITY:-horizon_channel}"
TOP_K="${TOP_K:-4}"
NUM_EXPERTS="${NUM_EXPERTS:-4}"
STRICT_ONLINE_CHECKS="${STRICT_ONLINE_CHECKS:-1}"
ONLINE_LOG_INTERVAL="${ONLINE_LOG_INTERVAL:-20}"
ITR="${ITR:-1}"
SEED="${SEED:-0}"
PRETRAIN_MODE="${PRETRAIN_MODE:-load}"
CHECKPOINT="${CHECKPOINT:-}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/log/smoke_progressive}"
ROOT_PATH="${ROOT_PATH:-$PROJECT_ROOT/data}"
DATA_FILE="$ROOT_PATH/$DATASET.csv"
LOG_FILE="$LOG_DIR/${DATASET}_pl${PRED_LEN}.out"

echo "[SMOKE CONFIG] dataset=$DATASET pred_len=$PRED_LEN gpu=$GPU_ID"
echo "[SMOKE CONFIG] max_steps=$MAX_ONLINE_STEPS itr=$ITR seed=$SEED"
echo "[SMOKE CONFIG] stable=$STABLE_BUFFER_SIZE recovery=$RECOVERY_BUFFER_SIZE recovery_batch=$RECOVERY_BATCH_SIZE"
echo "[SMOKE CONFIG] subspace_rank=$SUBSPACE_RANK max_rank=$SUBSPACE_MAX_RANK refresh=$SUBSPACE_REFRESH_INTERVAL"
echo "[SMOKE CONFIG] strategy=$EXPERT_UPDATE_STRATEGY router=$ROUTER_GRANULARITY top_k=$TOP_K strict=$STRICT_ONLINE_CHECKS"
echo "[SMOKE CONFIG] pretrain_mode=$PRETRAIN_MODE python=$PYTHON_BIN"

if [[ ! -f "$DATA_FILE" ]]; then
    echo "Smoke test data file does not exist: $DATA_FILE" >&2
    exit 2
fi

PRETRAIN_ARGS=(--pretrain_mode "$PRETRAIN_MODE")
if [[ "$PRETRAIN_MODE" == "load" ]]; then
    if [[ -z "$CHECKPOINT" ]]; then
        shopt -s nullglob
        if [[ "$ROUTER_GRANULARITY" == "horizon_channel" ]]; then
            candidates=(
                "$PROJECT_ROOT"/checkpoints/multi_expert*"$ROUTER_GRANULARITY"*_"$DATASET"_pl"$PRED_LEN"_*/checkpoint.pth
            )
        else
            candidates=(
                "$PROJECT_ROOT"/checkpoints/multi_expert*_"$DATASET"_pl"$PRED_LEN"_*/checkpoint.pth
            )
        fi
        shopt -u nullglob
        if [[ ${#candidates[@]} -gt 0 ]]; then
            CHECKPOINT="$(printf '%s\n' "${candidates[@]}" | sort | tail -n 1)"
        fi
    fi
    if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
        echo "No pretrained checkpoint found for $DATASET pred_len=$PRED_LEN." >&2
        echo "Set CHECKPOINT=/path/to/checkpoint.pth or PRETRAIN_MODE=retrain." >&2
        exit 2
    fi
    PRETRAIN_ARGS+=(--pretrained_checkpoint "$CHECKPOINT")
    echo "[SMOKE CONFIG] checkpoint=$CHECKPOINT"
elif [[ "$PRETRAIN_MODE" == "retrain" ]]; then
    echo "[SMOKE CONFIG] no checkpoint: smoke test will retrain"
else
    echo "PRETRAIN_MODE must be load or retrain, got: $PRETRAIN_MODE" >&2
    exit 2
fi

STRICT_ARGS=()
if [[ "$STRICT_ONLINE_CHECKS" == "1" ]]; then
    STRICT_ARGS+=(--strict_online_checks)
fi

mkdir -p "$LOG_DIR"
echo "[SMOKE RUN] log=$LOG_FILE"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u main.py \
    --method multi_expert \
    --root_path "$ROOT_PATH/" \
    --data "$DATASET" \
    --features M \
    --seq_len 60 \
    --label_len 0 \
    --pred_len "$PRED_LEN" \
    --test_bsz 1 \
    --batch_size 32 \
    --itr "$ITR" \
    --seed "$SEED" \
    --train_epochs "${TRAIN_EPOCHS:-1}" \
    --patience "${PATIENCE:-1}" \
    --learning_rate "${LEARNING_RATE:-1e-3}" \
    --online_lr_expert "${ONLINE_LR_EXPERT:-1e-4}" \
    --online_lr_router "${ONLINE_LR_ROUTER:-1e-5}" \
    --online_learning full \
    --delay_fb \
    --progressive_fb \
    --router_granularity "$ROUTER_GRANULARITY" \
    --correction_lr "${CORRECTION_LR:-0.1}" \
    --local_credit_weight "${LOCAL_CREDIT_WEIGHT:-0.1}" \
    --stable_buffer_size "$STABLE_BUFFER_SIZE" \
    --recovery_buffer_size "$RECOVERY_BUFFER_SIZE" \
    --recovery_batch_size "$RECOVERY_BATCH_SIZE" \
    --memory_refresh_interval "${MEMORY_REFRESH_INTERVAL:-20}" \
    --subspace_scope regressor \
    --subspace_rank "$SUBSPACE_RANK" \
    --subspace_max_rank "$SUBSPACE_MAX_RANK" \
    --subspace_refresh_interval "$SUBSPACE_REFRESH_INTERVAL" \
    --subspace_min_samples "${SUBSPACE_MIN_SAMPLES:-2}" \
    --subspace_lambda "${SUBSPACE_LAMBDA:-10000}" \
    --expert_update_strategy "$EXPERT_UPDATE_STRATEGY" \
    --num_experts "$NUM_EXPERTS" \
    --top_k "$TOP_K" \
    --online_log_interval "$ONLINE_LOG_INTERVAL" \
    --max_online_steps "$MAX_ONLINE_STEPS" \
    "${STRICT_ARGS[@]}" \
    "${PRETRAIN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

RESULT_DIR="$(
    sed -n 's/^RESULT_DIR: //p' "$LOG_FILE" | tail -n 1
)"
if [[ -z "$RESULT_DIR" || ! -d "$RESULT_DIR" ]]; then
    echo "Smoke run completed without a valid RESULT_DIR in $LOG_FILE" >&2
    exit 3
fi
"$PYTHON_BIN" scripts/check_smoke_results.py "$RESULT_DIR"
echo "[SMOKE DONE] result_dir=$RESULT_DIR"
