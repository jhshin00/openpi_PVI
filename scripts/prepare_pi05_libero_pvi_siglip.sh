#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

CONFIG="${CONFIG:-pi05_libero_pvi_from_pi05_libero_siglip}"
RAW_DIR="${LIBERO_RAW_DIR:-$ROOT/datasets/libero_rlds_raw}"
LEROBOT_ROOT="${HF_LEROBOT_HOME:-$ROOT/datasets}"
LEROBOT_DIR="$LEROBOT_ROOT/physical-intelligence/libero"
PYTORCH_CKPT="${PYTORCH_CKPT:-$ROOT/checkpoints/pytorch/pi05_libero}"
EXPECTED_NORM_DIR="$ROOT/assets/pi05_libero_pvi/physical-intelligence/libero"
GENERATED_NORM_DIR="$ROOT/assets/$CONFIG/physical-intelligence/libero"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2}}"

if [[ -z "${TRAIN_NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -ra train_gpu_ids <<< "$TRAIN_CUDA_VISIBLE_DEVICES"
  TRAIN_NPROC_PER_NODE="${#train_gpu_ids[@]}"
fi

required_raw_dirs=(
  "libero_10_no_noops"
  "libero_goal_no_noops"
  "libero_object_no_noops"
  "libero_spatial_no_noops"
)

have_raw=1
for d in "${required_raw_dirs[@]}"; do
  if [[ ! -f "$RAW_DIR/$d/1.0.0/dataset_info.json" ]]; then
    have_raw=0
  fi
done

if [[ "$have_raw" != "1" ]]; then
  mkdir -p "$RAW_DIR"
  uv run python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openvla/modified_libero_rlds",
    repo_type="dataset",
    local_dir="./datasets/libero_rlds_raw",
    max_workers=8,
)
PY
else
  echo "Raw LIBERO RLDS already exists at $RAW_DIR"
fi

if [[ ! -f "$LEROBOT_DIR/meta/info.json" ]]; then
  uv run examples/libero/convert_libero_data_to_lerobot.py \
    --data-dir "$RAW_DIR" \
    --output-root "$LEROBOT_ROOT"
else
  echo "LeRobot LIBERO dataset already exists at $LEROBOT_DIR"
fi

if [[ ! -f "$EXPECTED_NORM_DIR/norm_stats.json" ]]; then
  uv run scripts/compute_norm_stats.py \
    --config-name "$CONFIG" \
    --lerobot-root "$LEROBOT_ROOT"

  if [[ ! -f "$GENERATED_NORM_DIR/norm_stats.json" ]]; then
    echo "Expected generated norm stats were not found at $GENERATED_NORM_DIR/norm_stats.json" >&2
    exit 1
  fi

  mkdir -p "$EXPECTED_NORM_DIR"
  cp -a "$GENERATED_NORM_DIR/." "$EXPECTED_NORM_DIR/"
else
  echo "Norm stats already exist at $EXPECTED_NORM_DIR"
fi

if [[ ! -f "$PYTORCH_CKPT/model.safetensors" ]]; then
  uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint-dir gs://openpi-assets/checkpoints/pi05_libero \
    --config-name pi05_libero \
    --output-path "$PYTORCH_CKPT"
else
  echo "PyTorch pi05_libero checkpoint already exists at $PYTORCH_CKPT"
fi

echo
echo "Ready. Train with:"
echo "CUDA_VISIBLE_DEVICES=$TRAIN_CUDA_VISIBLE_DEVICES HF_LEROBOT_HOME=$LEROBOT_ROOT uv run torchrun --standalone --nnodes=1 --nproc_per_node=$TRAIN_NPROC_PER_NODE scripts/train_pytorch_PVI.py $CONFIG"
