#!/usr/bin/env bash

# Prefer the same environment as run.sh; PYTHON can still override it.
if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "/opt/data/private/envs/Online/bin/python" ]]; then
    PYTHON_BIN="/opt/data/private/envs/Online/bin/python"
elif [[ -x "/root/miniconda3/envs/online/bin/python" ]]; then
    PYTHON_BIN="/root/miniconda3/envs/online/bin/python"
else
    PYTHON_BIN="python"
fi

# 固定在项目根目录执行，避免脚本迁移后相对路径失效
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
ROOT_PATH="$PROJECT_ROOT/data/"
LOG_DIR="$PROJECT_ROOT/log/$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints"
mkdir -p "$LOG_DIR"

# 运行前检查数据文件是否存在
required_files=("ECL.csv")
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

## Multi-expert
online_learning='full'
i=1
ns=(1)
bszs=(1)
methods=('multi_expert')
lens=(24 48)
datasets=(ECL)

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
online_log_interval="${ONLINE_LOG_INTERVAL:-500}"

learning_rate_expert=1e-3
learning_rate_router=1e-3
online_lr_expert="${ONLINE_LR_EXPERT:-1e-4}"
online_lr_router="${ONLINE_LR_ROUTER:-1e-5}"
patience=3
opt_name="adam"
pretrained_checkpoint_tag="${PRETRAINED_CHECKPOINT_TAG:-stateful_pc_v2_ne${num_experts}_tk${top_k}}"
checkpoint_tag="${CHECKPOINT_TAG:-${pretrained_checkpoint_tag}_online_mse_v3}"
online_variant="online_mse_v3"

# 所有实验均重新预训练，不加载已有 checkpoint。
PRETRAIN_MODE="retrain"

find_latest_checkpoint() {
  local method="$1"
  local data="$2"
  local pred_len="$3"
  local online="$4"
  local opt="$5"
  local bsz="$6"
  local pattern="$CHECKPOINT_ROOT/${method}_${data}_pl${pred_len}_ol${online}_opt${opt}_tb${bsz}_"
  local latest=""
  local ckpt

  shopt -s nullglob
  for ckpt in "${pattern}"*/checkpoint.pth; do
    if [[ -z "$latest" || "$ckpt" > "$latest" ]]; then
      latest="$ckpt"
    fi
  done
  shopt -u nullglob

  echo "$latest"
}

# Run the two ECL horizons concurrently by default.
GPU_IDS_STR="${GPU_IDS:-0}"
MAX_PER_GPU="${MAX_PER_GPU:-2}"
IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_STR"
echo "[CONFIG] GPU_IDS_STR=${GPU_IDS_STR} parsed_gpus=${GPU_IDS[*]} MAX_PER_GPU=${MAX_PER_GPU}"
echo "[CONFIG] PRETRAIN_MODE=${PRETRAIN_MODE} CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
echo "[CONFIG] PRETRAINED_TAG=${pretrained_checkpoint_tag} RUN_TAG=${checkpoint_tag}"

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
    chosen_online_lr="${ECL_ONLINE_LR_EXPERT:-${ONLINE_LR_EXPERT:-1e-5}}"
    chosen_online_router_lr="${ECL_ONLINE_LR_ROUTER:-${ONLINE_LR_ROUTER:-1e-5}}"
    ;;
  WTH)
    chosen_online_lr="${WTH_ONLINE_LR_EXPERT:-${ONLINE_LR_EXPERT:-5e-5}}"
    chosen_online_router_lr="${WTH_ONLINE_LR_ROUTER:-${ONLINE_LR_ROUTER:-1e-5}}"
    ;;
esac
pick_next_gpu
gpu="$PICKED_GPU"
log_file="$LOG_DIR/multi_expert_${online_variant}_${data}_${len}_${online_learning}.out"

extra_args=(--pretrain_mode "$PRETRAIN_MODE" --checkpoint_tag "$checkpoint_tag")
if [[ "$PRETRAIN_MODE" == "load" ]]; then
    latest_ckpt=$(find_latest_checkpoint "${m}_${pretrained_checkpoint_tag}" "$data" "$len" "$online_learning" "$opt_name" "$bsz")
    if [[ -z "$latest_ckpt" ]]; then
        echo "No checkpoint found for data=${data} pred_len=${len} under ${CHECKPOINT_ROOT}" >&2
        exit 1
    fi
    extra_args+=(--pretrained_checkpoint "$latest_ckpt")
    echo "[LOAD] data=${data} pred_len=${len} gpu=${gpu} checkpoint=${latest_ckpt}"
else
    echo "[TRAIN] data=${data} pred_len=${len} gpu=${gpu} expert_lr=${chosen_lr} router_lr=${chosen_router_lr} online_expert_lr=${chosen_online_lr} online_router_lr=${chosen_online_router_lr} lambda_div=${chosen_lambda_div}"
fi

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
    --des 'Exp' \
    --itr "$i" \
    --train_epochs 15 \
    --patience "$patience" \
    --learning_rate "$chosen_lr" \
    --learning_rate_expert "$chosen_lr" \
    --learning_rate_router "$chosen_router_lr" \
    --online_lr_expert "$chosen_online_lr" \
    --online_lr_router "$chosen_online_router_lr" \
    --online_learning "$online_learning" \
    --delay_fb \
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
    --online_log_interval "$online_log_interval" \
    "${extra_args[@]}"
done
done
done
done
done

wait
echo "All runs finished. log_dir=$LOG_DIR"
