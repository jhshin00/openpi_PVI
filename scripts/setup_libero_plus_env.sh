#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/5] Syncing the root uv environment with LIBERO-plus dependencies..."
GIT_LFS_SKIP_SMUDGE=1 uv sync --group libero-plus
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

echo "[2/5] Generating .libero-plus-config/config.yaml..."
mkdir -p "$ROOT_DIR/.libero-plus-config"
mkdir -p "$ROOT_DIR/third_party/libero-plus/libero/datasets"
cat > "$ROOT_DIR/.libero-plus-config/config.yaml" <<EOF
benchmark_root: $ROOT_DIR/third_party/libero-plus/libero/libero
bddl_files: $ROOT_DIR/third_party/libero-plus/libero/libero/bddl_files
init_states: $ROOT_DIR/third_party/libero-plus/libero/libero/init_files
datasets: $ROOT_DIR/third_party/libero-plus/libero/datasets
assets: $ROOT_DIR/third_party/libero-plus/libero/libero/assets
EOF

echo "[3/5] Applying the transformers_replace patch..."
TRANSFORMERS_DIR="$("$ROOT_DIR/.venv/bin/python" - <<'PY'
import pathlib
import transformers

print(pathlib.Path(transformers.__file__).resolve().parent)
PY
)"
cp -r "$ROOT_DIR"/src/openpi/models_pytorch/transformers_replace/* "$TRANSFORMERS_DIR"/

echo "[4/5] Verifying the root environment..."
LIBERO_CONFIG_PATH="$ROOT_DIR/.libero-plus-config" \
PYTHONPATH="$ROOT_DIR/third_party/libero-plus:${PYTHONPATH:-}" \
"$ROOT_DIR/.venv/bin/python" - <<'PY'
import openpi
import robosuite
import bddl
import robomimic
import wand
import skimage
from libero.libero import benchmark

print("Verified root environment for LIBERO-plus.")
print("Available suites:", sorted(benchmark.get_benchmark_dict().keys()))
PY

echo "[5/5] Done."
echo
echo "Next steps:"
echo "  - Download LIBERO-plus assets into: $ROOT_DIR/third_party/libero-plus/libero/libero/assets"
echo "  - Put the LeRobot dataset under: $ROOT_DIR/lerobot/physical-intelligence/libero"
echo "  - Use the root .venv for PVI training and local LIBERO-plus evaluation."
