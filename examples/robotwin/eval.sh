#!/bin/bash

set -euo pipefail

task_name=${1:?task_name is required}
task_config=${2:?task_config is required}
train_config_name=${3:?train_config_name is required}
model_name=${4:?model_name is required}
seed=${5:?seed is required}
gpu_id=${6:?gpu_id is required}
checkpoint_id=${7:-30000}

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python examples/robotwin/eval.py \
  --train-config "${train_config_name}" \
  --task-name "${task_name}" \
  --task-config "${task_config}" \
  --exp-name "${model_name}" \
  --seed "${seed}" \
  --checkpoint-id "${checkpoint_id}"
