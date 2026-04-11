#!/bin/bash

set -euo pipefail

task_name=${1:?task_name is required}
task_config=${2:?task_config is required}
expert_data_num=${3:?expert_data_num is required}
raw_root=${4:-./datasets/robotwin_raw}
processed_root=${5:-./datasets/robotwin_processed}

input_dir="${raw_root}/${task_name}/${task_config}"
output_dir="${processed_root}/${task_name}-${task_config}-${expert_data_num}"

mkdir -p "${processed_root}"

uv run examples/robotwin/process_robotwin_data.py \
  --input-dir "${input_dir}" \
  --output-dir "${output_dir}" \
  --limit "${expert_data_num}"
