#!/bin/bash

set -euo pipefail

train_config_name=${1:?train_config_name is required}
repo_id=${2:?repo_id is required}
lerobot_root=${3:-./datasets}
asset_id=${4:-}

cmd=(
  uv run python examples/robotwin/workflow/compute_norm_stats.py
  --train-config-name "${train_config_name}"
  --repo-id "${repo_id}"
  --lerobot-root "${lerobot_root}"
)

if [ -n "${asset_id}" ]; then
  cmd+=(--asset-id "${asset_id}")
fi

"${cmd[@]}"
