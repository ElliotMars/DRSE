#!/usr/bin/env bash

# 环境变量相关
export PATH=~/git-2.41.0/git:$PATH
export PATH=/opt/data/private/ollama/bin:$PATH
export PATH=/opt/conda/bin:$PATH
export OLLAMA_MODELS=/opt/data/private/ollama-models/
export OLLAMA_HOST=0.0.0.0:11434
export HF_ENDPOINT=https://hf-mirror.com

# conda相关
# >>> conda initialize >>>
if [ -x "/opt/conda/bin/conda" ]; then
    _CONDA_BIN="/opt/conda/bin/conda"
    _CONDA_SH="/opt/conda/etc/profile.d/conda.sh"
elif [ -x "/root/miniconda3/bin/conda" ]; then
    _CONDA_BIN="/root/miniconda3/bin/conda"
    _CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
else
    echo "conda not found in /opt/conda or /root/miniconda3" >&2
    exit 1
fi

__conda_setup="$("$_CONDA_BIN" 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
elif [ -f "$_CONDA_SH" ]; then
    . "$_CONDA_SH"
else
    export PATH="$(dirname "$_CONDA_BIN"):$PATH"
fi
unset __conda_setup
unset _CONDA_BIN
unset _CONDA_SH
# <<< conda initialize <<<

conda activate /opt/data/private/envs/Online

# 固定在项目根目录执行，避免脚本迁移后相对路径失效
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
ROOT_PATH="$PROJECT_ROOT/data/"
LOG_DIR="$PROJECT_ROOT/log/$(date +%Y%m%d_%H%M%S)"
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

## Multi-expert
online_learning='full'
i=1
ns=(1)
bszs=(1)
lens=(1 24 48)
methods=('multi_expert')
datasets=(ETTh2 ETTm1 WTH ECL)

num_experts=4
top_k=2
lambda_div=0.03
tsb_alpha=0.5
tsb_eps=1e-8
tsb_buffer_size=32

learning_rate_expert=2e-3
learning_rate_router=2e-3
patience=3

# 与 run_com.sh 一致的限流并行：默认4张卡，每张卡并行2个任务
GPU_IDS_STR="${GPU_IDS:-0,1,2}"
MAX_PER_GPU="${MAX_PER_GPU:-2}"
IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_STR"
echo "[CONFIG] GPU_IDS_STR=${GPU_IDS_STR} parsed_gpus=${GPU_IDS[*]} MAX_PER_GPU=${MAX_PER_GPU}"

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

  CUDA_VISIBLE_DEVICES="$gpu" python -u main.py "$@" > "$log" 2>&1 &
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
chosen_lambda_div="$lambda_div"
pick_next_gpu
gpu="$PICKED_GPU"
log_file="$LOG_DIR/multi_expert_${data}_${len}_${online_learning}.out"

echo "[TRAIN] data=${data} pred_len=${len} gpu=${gpu} expert_lr=${chosen_lr} router_lr=${chosen_router_lr} lambda_div=${chosen_lambda_div}"
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
    --online_learning "$online_learning" \
    --num_experts "$num_experts" \
    --top_k "$top_k" \
    --lambda_div "$chosen_lambda_div" \
    --tsb_alpha "$tsb_alpha" \
    --tsb_eps "$tsb_eps" \
    --tsb_buffer_size "$tsb_buffer_size"
done
done
done
done
done

wait
echo "All runs finished. log_dir=$LOG_DIR"
