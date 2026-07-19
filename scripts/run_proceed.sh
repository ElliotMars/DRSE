#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python}"; SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; cd "$ROOT"
LOG_DIR="$ROOT/log/proceed_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG_DIR"
read -r -a datasets <<< "${DATASETS:-ETTh2 ETTm1 WTH ECL}"; read -r -a horizons <<< "${LENS:-1 24 48}"; GPU_ID="${GPU_IDS:-0}"
for data in "${datasets[@]}"; do for pred_len in "${horizons[@]}"; do
  lr=1e-3; [[ "$data" == ECL ]] && lr=3e-3; log="$LOG_DIR/proceed_${data}_${pred_len}_full.out"
  echo "[RUN] PROCEED data=$data pred_len=$pred_len gpu=$GPU_ID log=$log"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u main.py --method proceed --root_path "$ROOT/data/" --data "$data" --features M \
    --seq_len 60 --label_len 0 --pred_len "$pred_len" --test_bsz 1 --batch_size 32 --itr 1 --train_epochs 15 --patience 3 \
    --learning_rate "$lr" --online_learning full --delay_fb --patch_len 16 --stride 8 --padding_patch end --d_model 32 --n_heads 8 \
    --e_layers 2 --d_ff 128 --revin 1 --individual 1 --proceed_concept_dim 200 --proceed_bottleneck_dim 32 \
    --proceed_online_lr 1e-3 > "$log" 2>&1
done; done
echo "PROCEED runs finished. log_dir=$LOG_DIR"
