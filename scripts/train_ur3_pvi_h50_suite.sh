#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train_pytorch_PVI.py}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${REPO_ROOT}/.cache/openpi}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${REPO_ROOT}/datasets}"

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

mkdir -p "${HF_HOME}" "${OPENPI_DATA_HOME}" "${HF_LEROBOT_HOME}"

echo "Using repo root=${REPO_ROOT}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Using nnodes=${NNODES}, nproc_per_node=${NPROC_PER_NODE}"
echo "Using HF_HOME=${HF_HOME}"
echo "Using OPENPI_DATA_HOME=${OPENPI_DATA_HOME}"
echo "Using HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
echo "Training order: ${CONFIGS[*]}"

for config_name in "${CONFIGS[@]}"; do
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${config_name}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    HF_HOME="${HF_HOME}" \
    OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" \
    HF_LEROBOT_HOME="${HF_LEROBOT_HOME}" \
    uv run torchrun --standalone --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" \
    "${TRAIN_SCRIPT}" "${config_name}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${config_name}"
done

echo
echo "All training jobs completed."
