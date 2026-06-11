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

online_learning='full'
i=1
n=1
bsz=1
m='onenet_fsnet'
LOG_DIR='log/run'
mkdir -p "$LOG_DIR"

# Comma-separated GPU ids. Default: 3 GPUs, 2 concurrent jobs per GPU.
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

run_job() {
  local data="$1"
  local pred_len="$2"
  local lr="$3"
  local adbfgs="$4"
  local log="$LOG_DIR/${data}${pred_len}${online_learning}.out"
  local gpu
  pick_next_gpu
  gpu="$PICKED_GPU"

  if [ "$adbfgs" = "1" ]; then
    submit_job "$gpu" "$log" \
      --method "$m" --root_path ./data/ --n_inner "$n" --test_bsz "$bsz" \
      --data "$data" --features M --seq_len 60 --label_len 0 --pred_len "$pred_len" \
      --des 'Exp' --itr "$i" --train_epochs 15 --learning_rate "$lr" \
      --online_learning "$online_learning" --use_adbfgs
  else
    submit_job "$gpu" "$log" \
      --method "$m" --root_path ./data/ --n_inner "$n" --test_bsz "$bsz" \
      --data "$data" --features M --seq_len 60 --label_len 0 --pred_len "$pred_len" \
      --des 'Exp' --itr "$i" --train_epochs 15 --learning_rate "$lr" \
      --online_learning "$online_learning"
  fi
}

run_job ECL 24 3e-3 1
run_job ECL 48 3e-3 1
run_job ETTh2 48 1e-3 1
run_job ETTm1 24 1e-3 0
run_job ETTm1 48 1e-3 0

wait
echo "All runs finished."
