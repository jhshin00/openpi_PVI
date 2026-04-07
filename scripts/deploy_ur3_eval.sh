#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/deploy_ur3_eval.sh \
    --target user@ur3-pc:/remote/openpi \
    --variant dinov2_h50 \
    [--copy-hf-cache] \
    [--local-hf-home /local/hf_home] \
    [--remote-hf-home /remote/hf_home] \
    [--dry-run]

Variants:
  dinov2_h50
  hpr_h50

Examples:
  ./scripts/deploy_ur3_eval.sh \
    --target user@192.168.5.200:/data/jhshin/openpi \
    --variant dinov2_h50 \
    --copy-hf-cache

  ./scripts/deploy_ur3_eval.sh \
    --target user@192.168.5.200:/data/jhshin/openpi \
    --variant hpr_h50 \
    --copy-hf-cache
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

TARGET=""
VARIANT=""
COPY_HF_CACHE=0
DRY_RUN=0
LOCAL_HF_HOME="${SOURCE_ROOT}/.cache/huggingface"
REMOTE_HF_HOME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="${2:-}"
      shift 2
      ;;
    --variant)
      VARIANT="${2:-}"
      shift 2
      ;;
    --copy-hf-cache)
      COPY_HF_CACHE=1
      shift
      ;;
    --local-hf-home)
      LOCAL_HF_HOME="${2:-}"
      shift 2
      ;;
    --remote-hf-home)
      REMOTE_HF_HOME="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${TARGET}" || -z "${VARIANT}" ]]; then
  usage >&2
  exit 1
fi

if [[ "${TARGET}" != *:* ]]; then
  echo "--target must be in the form user@host:/absolute/path" >&2
  exit 1
fi

TARGET_HOST="${TARGET%%:*}"
TARGET_ROOT="${TARGET#*:}"

if [[ -z "${REMOTE_HF_HOME}" ]]; then
  REMOTE_HF_HOME="${TARGET_ROOT}/.cache/huggingface"
fi

case "${VARIANT}" in
  dinov2_h50|dinov2-h50)
    POLICY_CONFIG="pi05_ur3_pvi_dinov2_h50_infer"
    POLICY_DIR_REL="checkpoints/pi05_ur3_pvi_dinov2_h50/pi05_ur3_pvi_dinov2_h50_run1/1600"
    EXTRA_PATHS=()
    ;;
  hpr_h50|hpr-h50)
    POLICY_CONFIG="pi05_ur3_pvi_hpr_h50_infer"
    POLICY_DIR_REL="checkpoints/pi05_ur3_pvi_hpr_h50/pi05_ur3_pvi_hpr_h50_run1/1600"
    EXTRA_PATHS=(
      "hpr_checkpoints/hpr_fullfinetune_base_lang_trace_negative_mod.ckpt"
    )
    ;;
  *)
    echo "Unsupported --variant: ${VARIANT}" >&2
    exit 1
    ;;
esac

COMMON_PATHS=(
  ".python-version"
  "pyproject.toml"
  "uv.lock"
  "README.md"
  "ur3_readme.md"
  "packages/openpi-client"
  "src/openpi"
  "examples/ur3"
  "_external/gello_software"
  "assets/pi05_ur3_pvi"
)

run_remote() {
  local command="$1"
  if (( DRY_RUN )); then
    echo "[dry-run] ssh ${TARGET_HOST} ${command}"
  else
    ssh "${TARGET_HOST}" "${command}"
  fi
}

sync_path() {
  local rel="$1"
  local src="${SOURCE_ROOT}/${rel}"
  if [[ ! -e "${src}" ]]; then
    echo "Missing required path: ${src}" >&2
    exit 1
  fi

  local rsync_args=(
    -av
    --info=progress2
    --relative
  )
  if (( DRY_RUN )); then
    rsync_args+=(--dry-run)
  fi

  if (( DRY_RUN )); then
    echo "[dry-run] rsync ${rsync_args[*]} ${SOURCE_ROOT}/./${rel} ${TARGET_HOST}:${TARGET_ROOT}/"
  else
    rsync "${rsync_args[@]}" "${SOURCE_ROOT}/./${rel}" "${TARGET_HOST}:${TARGET_ROOT}/"
  fi
}

echo "Source root : ${SOURCE_ROOT}"
echo "Target root : ${TARGET_HOST}:${TARGET_ROOT}"
echo "Variant     : ${VARIANT}"
echo "Policy cfg  : ${POLICY_CONFIG}"
echo "Policy dir  : ${POLICY_DIR_REL}"
echo

run_remote "mkdir -p '${TARGET_ROOT}'"

for rel in "${COMMON_PATHS[@]}"; do
  sync_path "${rel}"
done

sync_path "${POLICY_DIR_REL}"

for rel in "${EXTRA_PATHS[@]}"; do
  sync_path "${rel}"
done

if (( COPY_HF_CACHE )); then
  if [[ ! -d "${LOCAL_HF_HOME}" ]]; then
    echo "HF cache directory does not exist: ${LOCAL_HF_HOME}" >&2
    exit 1
  fi

  run_remote "mkdir -p '${REMOTE_HF_HOME}'"

  HF_RSYNC_ARGS=(
    -av
    --info=progress2
  )
  if (( DRY_RUN )); then
    HF_RSYNC_ARGS+=(--dry-run)
  fi

  if (( DRY_RUN )); then
    echo "[dry-run] rsync ${HF_RSYNC_ARGS[*]} ${LOCAL_HF_HOME}/ ${TARGET_HOST}:${REMOTE_HF_HOME}/"
  else
    rsync "${HF_RSYNC_ARGS[@]}" "${LOCAL_HF_HOME}/" "${TARGET_HOST}:${REMOTE_HF_HOME}/"
  fi
fi

cat <<EOF

Sync complete.

Next steps on the target machine:

  cd ${TARGET_ROOT}
  GIT_LFS_SKIP_SMUDGE=1 uv sync
  GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
  uv pip install pyrealsense2
  SITE_PACKAGES=\$(uv run python -c 'import site; print(next(p for p in site.getsitepackages() if p.endswith("site-packages")))')
  cp -r ./src/openpi/models_pytorch/transformers_replace/* "\${SITE_PACKAGES}/transformers/"

Validation:

  test -f ${TARGET_ROOT}/${POLICY_DIR_REL}/model.safetensors
  test -f ${TARGET_ROOT}/${POLICY_DIR_REL}/assets/ur3_dataset/norm_stats.json

Run:

  HF_HOME=${REMOTE_HF_HOME} uv run examples/ur3/main.py \\
    --policy-config ${POLICY_CONFIG} \\
    --policy-dir ./${POLICY_DIR_REL} \\
    --robot-mode direct \\
    --robot-ip 192.168.5.102 \\
    --base-camera-serial 335222074820 \\
    --wrist-camera-serial 335522070336 \\
    --hz 30 \\
    --replan-steps 8 \\
    --debug-action-stats \\
    --debug-log-every 1 \\
    --default-prompt "pick up the pear and place it in the sink" \\
    --max-steps 5000 \\
    --deadband 0.003 \\
    --kp 10 \\
    --max-joint-velocity 0.35 \\
    --max-joint-accel 0.8 \\
    --speedj-accel 0.8 \\
    --chunk-execution chunk_endpoint \\
    --target-smoothing-alpha 0.2
EOF
