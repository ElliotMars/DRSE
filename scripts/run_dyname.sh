#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

ROOT_PATH="$PROJECT_ROOT/data/"
LOG_DIR="$PROJECT_ROOT/log/dyname_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

read -r -a datasets <<< "${DATASETS:-ETTh2 ETTm1 WTH ECL}"
read -r -a horizons <<< "${LENS:-1 24 48}"
GPU_ID="${GPU_IDS:-0}"

for data in "${datasets[@]}"; do
  for pred_len in "${horizons[@]}"; do
    learning_rate=1e-3
    if [[ "$data" == "ECL" ]]; then learning_rate=3e-3; fi
    log="$LOG_DIR/dyname_${data}_${pred_len}_full.out"
    echo "[RUN] data=$data pred_len=$pred_len gpu=$GPU_ID log=$log"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u main.py \
      --method dyname --root_path "$ROOT_PATH" --data "$data" --features M \
      --seq_len 60 --label_len 0 --pred_len "$pred_len" \
      --test_bsz 1 --batch_size 32 --itr 1 --train_epochs 15 --patience 3 \
      --learning_rate "$learning_rate" --online_learning full --delay_fb \
      --dyname_period_num 2 --dyname_krr_lambda 1e-4 \
      --dyname_krr_train_num 8 --dyname_past_num 672 \
      --dyname_beta 0.3 --dyname_delta 0.01 --dyname_online_lr 1e-3 \
      > "$log" 2>&1
  done
done

echo "All DynaME runs finished. log_dir=$LOG_DIR"
