#!/bin/bash

set -euo pipefail

input_dir=${1:?input_dir is required}
repo_id=${2:?repo_id is required}
output_root=${3:-./datasets}

uv run examples/robotwin/convert_robotwin_to_lerobot.py \
  --input-dir "${input_dir}" \
  --repo-id "${repo_id}" \
  --output-root "${output_root}"
