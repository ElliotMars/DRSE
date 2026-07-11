#!/usr/bin/env bash

# Prefer the same environment as run.sh; PYTHON can still override it.
if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "/opt/data/private/envs/Online/bin/python" ]]; then
    PYTHON_BIN="/opt/data/private/envs/Online/bin/python"
else
    PYTHON_BIN="python"
fi

# 固定在项目根目录执行，避免脚本迁移后相对路径失效
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
ROOT_PATH="$PROJECT_ROOT/data/"
LOG_DIR="$PROJECT_ROOT/log/pretrain_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 运行前检查数据文件是否存在
required_files=("ETTh2.csv" "ETTm1.csv" "WTH.csv" "ECL.csv")
missing=0
for f in "${required_files[@]}"; do
    if [ ! -f "${ROOT_PATH}${f}" ]; then
        echo "Missing data file: ${ROOT_PATH}${f}" >&2
        missing=1
    fi
done
if [ $missing -ne 0 ]; then
    echo "Please put required CSV files into ${ROOT_PATH}" >&2
    exit 1
fi

## Multi-expert pretraining only: 4 datasets x 3 horizons = 12 jobs
online_learning='full'
i=1
ns=(1)
bszs=(1)
methods=('multi_expert')
read -r -a lens <<< "${LENS:-1 24 48}"
read -r -a datasets <<< "${DATASETS:-ETTh2 ETTm1 WTH ECL}"

num_experts="${NUM_EXPERTS:-4}"
top_k="${TOP_K:-$num_experts}"
lambda_div="${LAMBDA_DIV:-0.0}"
tsb_alpha="${TSB_ALPHA:-0.5}"
tsb_eps=1e-8
tsb_buffer_size="${TSB_BUFFER_SIZE:-8}"
expert_grad_clip="${EXPERT_GRAD_CLIP:-1.0}"
router_grad_clip="${ROUTER_GRAD_CLIP:-0.5}"
router_temperature="${ROUTER_TEMPERATURE:-2.0}"
router_entropy_weight="${ROUTER_ENTROPY_WEIGHT:-0.001}"
robust_fallback_threshold="${ROBUST_FALLBACK_THRESHOLD:-25.0}"
online_log_interval="${ONLINE_LOG_INTERVAL:-500}"

learning_rate_expert=1e-3
learning_rate_router=1e-3
online_lr_expert=1e-4
online_lr_router=1e-5
patience=3
checkpoint_tag="stateful_pc_v2_ne${num_experts}_tk${top_k}"

# Stable tuning default: one job per GPU.
GPU_IDS_STR="${GPU_IDS:-0}"
MAX_PER_GPU="${MAX_PER_GPU:-1}"
IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_STR"
echo "[CONFIG] GPU_IDS_STR=${GPU_IDS_STR} parsed_gpus=${GPU_IDS[*]} MAX_PER_GPU=${MAX_PER_GPU}"
echo "[CONFIG] pretrain only, skip_test enabled, log_dir=${LOG_DIR}"

declare -a RUN_PIDS=()
declare -A PID_GPU=()
next_gpu_idx=0

cleanup_finished() {
  local alive=()
  local pid
  for pid in "${RUN_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    else
      unset 'PID_GPU[$pid]'
    fi
  done
  RUN_PIDS=("${alive[@]}")
}

running_on_gpu() {
  local gpu="$1"
  local cnt=0
  local pid
  for pid in "${RUN_PIDS[@]}"; do
    if [[ "${PID_GPU[$pid]:-}" == "$gpu" ]] && kill -0 "$pid" 2>/dev/null; then
      cnt=$((cnt + 1))
    fi
  done
  echo "$cnt"
}

wait_for_gpu_slot() {
  local gpu="$1"
  while true; do
    cleanup_finished
    local cnt
    cnt=$(running_on_gpu "$gpu")
    if [ "$cnt" -lt "$MAX_PER_GPU" ]; then
      return 0
    fi
    sleep 2
  done
}

submit_job() {
  local gpu="$1"
  local log="$2"
  shift 2

  echo "[WAIT] gpu=$gpu log=$log"
  wait_for_gpu_slot "$gpu"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u main.py "$@" > "$log" 2>&1 &
  local pid=$!
  RUN_PIDS+=("$pid")
  PID_GPU[$pid]="$gpu"
  echo "[SUBMIT] pid=$pid gpu=$gpu log=$log"
}

pick_next_gpu() {
  PICKED_GPU="${GPU_IDS[$next_gpu_idx]}"
  next_gpu_idx=$(((next_gpu_idx + 1) % ${#GPU_IDS[@]}))
}

for n in "${ns[@]}"; do
for bsz in "${bszs[@]}"; do
for len in "${lens[@]}"; do
for m in "${methods[@]}"; do
for data in "${datasets[@]}"; do
chosen_lr="$learning_rate_expert"
chosen_router_lr="$learning_rate_router"
chosen_online_lr="$online_lr_expert"
chosen_online_router_lr="$online_lr_router"
chosen_lambda_div="$lambda_div"
case "$data" in
  ECL)
    chosen_lr=3e-3
    chosen_router_lr=3e-3
    chosen_online_lr=1e-5
    chosen_online_router_lr=1e-6
    ;;
  WTH)
    chosen_online_lr=5e-5
    ;;
esac
pick_next_gpu
gpu="$PICKED_GPU"
log_file="$LOG_DIR/pretrain_multi_expert_${data}_${len}.out"

echo "[PRETRAIN] data=${data} pred_len=${len} gpu=${gpu} expert_lr=${chosen_lr} router_lr=${chosen_router_lr} lambda_div=${chosen_lambda_div}"
submit_job "$gpu" "$log_file" \
    --method "$m" \
    --root_path "$ROOT_PATH" \
    --n_inner "$n" \
    --test_bsz "$bsz" \
    --data "$data" \
    --features M \
    --seq_len 60 \
    --label_len 0 \
    --pred_len "$len" \
    --des 'PretrainOnly' \
    --itr "$i" \
    --train_epochs 15 \
    --patience "$patience" \
    --learning_rate "$chosen_lr" \
    --learning_rate_expert "$chosen_lr" \
    --learning_rate_router "$chosen_router_lr" \
    --online_lr_expert "$chosen_online_lr" \
    --online_lr_router "$chosen_online_router_lr" \
    --online_learning "$online_learning" \
    --num_experts "$num_experts" \
    --top_k "$top_k" \
    --lambda_div "$chosen_lambda_div" \
    --tsb_alpha "$tsb_alpha" \
    --tsb_eps "$tsb_eps" \
    --tsb_buffer_size "$tsb_buffer_size" \
    --expert_grad_clip "$expert_grad_clip" \
    --router_grad_clip "$router_grad_clip" \
    --router_temperature "$router_temperature" \
    --router_entropy_weight "$router_entropy_weight" \
    --robust_fallback_threshold "$robust_fallback_threshold" \
    --online_log_interval "$online_log_interval" \
    --checkpoint_tag "$checkpoint_tag" \
    --skip_test
done
done
done
done
done

wait
echo "All pretraining runs finished. log_dir=$LOG_DIR"
