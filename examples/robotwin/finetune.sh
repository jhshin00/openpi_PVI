#!/bin/bash

set -euo pipefail

train_config_name=${1:?train_config_name is required}
model_name=${2:?model_name is required}
gpu_use=${3:?gpu_use is required}
repo_id=${4:?repo_id is required}
lerobot_root=${5:-./datasets}
asset_id=${6:-}

export CUDA_VISIBLE_DEVICES="${gpu_use}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

IFS=',' read -r -a gpu_array <<< "${gpu_use}"
gpu_count=${#gpu_array[@]}

cmd=(
  examples/robotwin/workflow/finetune.py
  --train-config-name "${train_config_name}"
  --model-name "${model_name}"
  --repo-id "${repo_id}"
  --lerobot-root "${lerobot_root}"
  --overwrite
)

if [ -n "${asset_id}" ]; then
  cmd+=(--asset-id "${asset_id}")
fi

if [ "${gpu_count}" -gt 1 ]; then
  uv run torchrun --standalone --nnodes=1 --nproc_per_node="${gpu_count}" "${cmd[@]}"
else
  uv run python "${cmd[@]}"
fi
