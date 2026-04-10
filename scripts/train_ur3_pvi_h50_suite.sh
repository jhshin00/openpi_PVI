#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,5,6}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train_pytorch_PVI.py}"

DEFAULT_CONFIGS=(
  "pi05_ur3_pvi_dinov2_h50"
  "pi05_ur3_pvi_hpr_h50"
  "pi05_ur3_pvi_siglip_h50"
  "pi05_ur3_pvi_clip_h50"
  "pi05_ur3_pvi_r3m_h50"
)

if [ "$#" -gt 0 ]; then
  CONFIGS=("$@")
else
  CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Using nnodes=${NNODES}, nproc_per_node=${NPROC_PER_NODE}"
echo "Training order: ${CONFIGS[*]}"

for config_name in "${CONFIGS[@]}"; do
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${config_name}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    uv run torchrun --standalone --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" \
    "${TRAIN_SCRIPT}" "${config_name}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${config_name}"
done

echo
echo "All training jobs completed."
