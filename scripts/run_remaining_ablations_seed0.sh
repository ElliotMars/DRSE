#!/usr/bin/env bash
set -euo pipefail

mkdir -p log/ablations_seed0

COMMON=(
  DATASETS="ETTh2 ETTm1 WTH ECL"
  LENS="1 24 48"
  MAX_PER_GPU=1
  PRETRAIN_MODE=load
  CHECKPOINT_TAG=directional_recovery_v1
  RECOVERY_DEGRADATION_MARGIN=0.0
  ITR=1
  SEED=0
)

echo "===== Wave 1: w/o Version Awareness + w/o Recovery ====="

env "${COMMON[@]}" \
  GPU_IDS=0 \
  DISABLE_VERSION_AWARENESS=1 \
  bash scripts/run_progressive_credit_subspace.sh \
  > log/ablations_seed0/wo_version_awareness.out 2>&1 &
p1=$!

env "${COMMON[@]}" \
  GPU_IDS=1 \
  DISABLE_RECOVERY=1 \
  bash scripts/run_progressive_credit_subspace.sh \
  > log/ablations_seed0/wo_recovery.out 2>&1 &
p2=$!

wait "$p1"
wait "$p2"

echo "===== Wave 2: Unweighted Subspace + w/o Subspace ====="

env "${COMMON[@]}" \
  GPU_IDS=0 \
  DISABLE_CREDIT_WEIGHTED_SUBSPACE=1 \
  bash scripts/run_progressive_credit_subspace.sh \
  > log/ablations_seed0/unweighted_subspace.out 2>&1 &
p1=$!

env "${COMMON[@]}" \
  GPU_IDS=1 \
  EXPERT_UPDATE_STRATEGY=plain \
  bash scripts/run_progressive_credit_subspace.sh \
  > log/ablations_seed0/wo_subspace.out 2>&1 &
p2=$!

wait "$p1"
wait "$p2"

echo "===== ALL FOUR REMAINING ABLATIONS FINISHED ====="
