#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

gpu_use=${1:-2}
task_config=${2:-demo_clean}
checkpoint_id=${3:-30000}
encoder_selector=${4:-dinov2,siglip,clip,r3m}
seed=${5:-0}
test_num=${6:-100}

shift $(( $# >= 6 ? 6 : $# ))

task_selectors=()
extra_eval_args=()
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--" ]; then
    shift
    extra_eval_args=("$@")
    break
  fi
  task_selectors+=("$1")
  shift
done

if [ "${#task_selectors[@]}" -eq 0 ]; then
  task_selectors=(
    robotwin_lift_pot_aloha_agilex_clean_50
    robotwin_move_can_pot_aloha_agilex_clean_50
    robotwin_beat_hammer_block_aloha_agilex_clean_50
  )
fi

trim_selector() {
  local value=$1
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  echo "${value}"
}

resolve_train_config() {
  local selector
  selector="$(trim_selector "$1")"
  case "${selector}" in
    dino|dinov2|dino_v1|dinov2_v1|er_dino|pi05_robotwin_encoder_replace_dino)
      echo "pi05_robotwin_encoder_replace_dino"
      ;;
    siglip|siglip_v1|er_siglip_v1|pi05_robotwin_encoder_replace_siglip)
      echo "pi05_robotwin_encoder_replace_siglip"
      ;;
    clip|clip_v1|er_clip|pi05_robotwin_encoder_replace_clip)
      echo "pi05_robotwin_encoder_replace_clip"
      ;;
    r3m|r3m_v1|er_r3m|pi05_robotwin_encoder_replace_r3m)
      echo "pi05_robotwin_encoder_replace_r3m"
      ;;
    hpr|hpr_v1|er_hpr_v1|pi05_robotwin_encoder_replace_hpr)
      echo "pi05_robotwin_encoder_replace_hpr"
      ;;
    *)
      echo "Unknown v1 encoder-replacement selector: ${selector}" >&2
      echo "Expected comma-separated values from: dinov2, siglip, clip, r3m, hpr" >&2
      exit 1
      ;;
  esac
}

encoder_label_from_config() {
  local train_config_name=$1
  case "${train_config_name}" in
    pi05_robotwin_encoder_replace_dino)
      echo "dinov2"
      ;;
    pi05_robotwin_encoder_replace_siglip)
      echo "siglip"
      ;;
    pi05_robotwin_encoder_replace_clip)
      echo "clip"
      ;;
    pi05_robotwin_encoder_replace_r3m)
      echo "r3m"
      ;;
    pi05_robotwin_encoder_replace_hpr)
      echo "hpr"
      ;;
    *)
      echo "${train_config_name#pi05_robotwin_encoder_replace_}"
      ;;
  esac
}

resolve_model_name() {
  local selector
  selector="$(trim_selector "$1")"
  case "${selector}" in
    lift_pot|robotwin_lift_pot_aloha_agilex_clean_50)
      echo "robotwin_lift_pot_aloha_agilex_clean_50"
      ;;
    move_can_pot|robotwin_move_can_pot_aloha_agilex_clean_50)
      echo "robotwin_move_can_pot_aloha_agilex_clean_50"
      ;;
    beat_hammer_block|robotwin_beat_hammer_block_aloha_agilex_clean_50)
      echo "robotwin_beat_hammer_block_aloha_agilex_clean_50"
      ;;
    beat_block_hammer|robotwin_beat_block_hammer_aloha_agilex_clean_50)
      echo "robotwin_beat_block_hammer_aloha_agilex_clean_50"
      ;;
    robotwin_*_aloha_agilex_clean_50)
      echo "${selector}"
      ;;
    *)
      echo "Unknown task selector: ${selector}" >&2
      echo "Expected one of: lift_pot, move_can_pot, beat_hammer_block, beat_block_hammer, or a full robotwin_* task id" >&2
      exit 1
      ;;
  esac
}

resolve_eval_task_name() {
  local model_name=$1
  case "${model_name}" in
    robotwin_lift_pot_aloha_agilex_clean_50)
      echo "lift_pot"
      ;;
    robotwin_move_can_pot_aloha_agilex_clean_50)
      echo "move_can_pot"
      ;;
    robotwin_beat_hammer_block_aloha_agilex_clean_50|robotwin_beat_block_hammer_aloha_agilex_clean_50)
      echo "beat_block_hammer"
      ;;
    robotwin_*_aloha_agilex_clean_50)
      local task_name=${model_name#robotwin_}
      task_name=${task_name%_aloha_agilex_clean_50}
      echo "${task_name}"
      ;;
    *)
      echo "${model_name}"
      ;;
  esac
}

sanitize_label() {
  local value=$1
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value// /_}"
  value="${value//,/_}"
  echo "${value}"
}

IFS=',' read -r -a encoder_selectors <<< "${encoder_selector}"
train_configs=()
for selector in "${encoder_selectors[@]}"; do
  train_configs+=("$(resolve_train_config "${selector}")")
done

tasks=()
eval_task_names=()
for selector in "${task_selectors[@]}"; do
  model_name="$(resolve_model_name "${selector}")"
  tasks+=("${model_name}")
  eval_task_names+=("$(resolve_eval_task_name "${model_name}")")
done

IFS=',' read -r -a gpu_array <<< "${gpu_use}"
if [ "${#gpu_array[@]}" -eq 0 ]; then
  echo "At least one GPU id is required." >&2
  exit 1
fi

checkpoint_base_dir=${ROBOTWIN_EVAL_CHECKPOINT_BASE_DIR:-./checkpoints_local}
result_root=${ROBOTWIN_EVAL_RESULT_ROOT:-./eval_result/robotwin}
sweep_root=${ROBOTWIN_EVAL_SWEEP_ROOT:-./eval_result/robotwin_sweeps}
parallel_jobs=${ROBOTWIN_EVAL_PARALLEL_JOBS:-${#gpu_array[@]}}
if [ "${parallel_jobs}" -lt 1 ]; then
  echo "ROBOTWIN_EVAL_PARALLEL_JOBS must be >= 1." >&2
  exit 1
fi
if [ "${parallel_jobs}" -gt "${#gpu_array[@]}" ]; then
  parallel_jobs=${#gpu_array[@]}
fi

sweep_id="$(date +"%Y-%m-%d_%H-%M-%S")"
sweep_dir="${sweep_root}/${sweep_id}"
summary_dir="${sweep_dir}/summaries"
log_dir="${sweep_dir}/logs"
record_dir="${sweep_dir}/records"
mkdir -p "${summary_dir}" "${log_dir}" "${record_dir}"

default_eval_args=(--disable-video --disable-torch-compile)
total_runs=$(( ${#train_configs[@]} * ${#tasks[@]} ))

echo "Python: $(command -v python)"
echo "GPU pool: ${gpu_array[*]}"
echo "Parallel eval jobs: ${parallel_jobs}"
echo "Selected v1 configs: ${train_configs[*]}"
echo "Selected tasks: ${tasks[*]}"
echo "RoboTwin task_config=${task_config}"
echo "checkpoint_id=${checkpoint_id}"
echo "seed=${seed}"
echo "test_num=${test_num}"
echo "checkpoint_base_dir=${checkpoint_base_dir}"
echo "sweep_dir=${sweep_dir}"
echo "Total eval runs: ${total_runs}"

run_eval() {
  local run_index=$1
  local train_config_name=$2
  local encoder_label=$3
  local model_name=$4
  local eval_task_name=$5
  local gpu_id=$6
  local summary_file=$7
  local log_file=$8
  local record_file=$9

  local start_ts
  local end_ts
  local exit_code
  local status
  local summary_values
  local success_count
  local summary_test_num
  local success_rate
  local result_dir

  echo "=== [${run_index}/${total_runs}] Eval ${eval_task_name} (model=${model_name}, config=${train_config_name}, gpu=${gpu_id}) ==="
  start_ts="$(date -Iseconds)"

  cmd=(
    python examples/robotwin/eval.py
    --train-config "${train_config_name}"
    --task-name "${eval_task_name}"
    --task-config "${task_config}"
    --exp-name "${model_name}"
    --seed "${seed}"
    --checkpoint-id "${checkpoint_id}"
    --checkpoint-base-dir "${checkpoint_base_dir}"
    --result-root "${result_root}"
    --summary-output "${summary_file}"
    --test-num "${test_num}"
    "${default_eval_args[@]}"
    "${extra_eval_args[@]}"
  )

  {
    echo "Started: ${start_ts}"
    echo "CUDA_VISIBLE_DEVICES=${gpu_id}"
    printf "Command:"
    printf " %q" "${cmd[@]}"
    printf "\n\n"
  } > "${log_file}"

  set +e
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" >> "${log_file}" 2>&1
  exit_code=$?
  set -e

  end_ts="$(date -Iseconds)"
  status="ok"
  if [ "${exit_code}" -ne 0 ]; then
    status="failed"
  elif [ ! -f "${summary_file}" ]; then
    status="missing_summary"
  fi

  success_count="NA"
  summary_test_num="NA"
  success_rate="NA"
  result_dir="NA"
  if [ -f "${summary_file}" ]; then
    summary_values="$(
      python -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    summary = json.load(f)

rate = summary.get("success_rate")
rate_text = "NA" if rate is None else f"{float(rate):.4f}"
print(
    "\t".join(
        [
            str(summary.get("success_count", "NA")),
            str(summary.get("test_num", "NA")),
            rate_text,
            str(summary.get("result_dir", "NA")),
        ]
    )
)
' "${summary_file}" 2>/dev/null
    )" || summary_values=""
    if [ -n "${summary_values}" ]; then
      IFS=$'\t' read -r success_count summary_test_num success_rate result_dir <<< "${summary_values}"
    elif [ "${status}" = "ok" ]; then
      status="summary_parse_failed"
    fi
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${run_index}" \
    "${status}" \
    "${encoder_label}" \
    "${train_config_name}" \
    "${model_name}" \
    "${eval_task_name}" \
    "${task_config}" \
    "${checkpoint_id}" \
    "${seed}" \
    "${gpu_id}" \
    "${success_count}" \
    "${summary_test_num}" \
    "${success_rate}" \
    "${summary_file}" \
    "${result_dir}" \
    "${log_file}" \
    "${end_ts}" > "${record_file}"

  echo "=== [${run_index}/${total_runs}] Done status=${status} rate=${success_rate} log=${log_file} ==="
  return 0
}

wait_batch() {
  local pid
  for pid in "${batch_pids[@]}"; do
    wait "${pid}"
  done
  batch_pids=()
}

batch_pids=()
run_record_files=()
run_index=0
batch_slot=0

for train_config_name in "${train_configs[@]}"; do
  encoder_label="$(encoder_label_from_config "${train_config_name}")"
  for task_idx in "${!tasks[@]}"; do
    run_index=$((run_index + 1))
    model_name="${tasks[task_idx]}"
    eval_task_name="${eval_task_names[task_idx]}"
    gpu_id="${gpu_array[batch_slot]}"
    run_label="$(printf "%03d_%s_%s" "${run_index}" "$(sanitize_label "${train_config_name}")" "$(sanitize_label "${model_name}")")"
    summary_file="${summary_dir}/${run_label}.json"
    log_file="${log_dir}/${run_label}.log"
    record_file="${record_dir}/${run_label}.tsv"
    run_record_files+=("${record_file}")

    run_eval \
      "${run_index}" \
      "${train_config_name}" \
      "${encoder_label}" \
      "${model_name}" \
      "${eval_task_name}" \
      "${gpu_id}" \
      "${summary_file}" \
      "${log_file}" \
      "${record_file}" &
    batch_pids+=("$!")

    batch_slot=$((batch_slot + 1))
    if [ "${batch_slot}" -ge "${parallel_jobs}" ]; then
      wait_batch
      batch_slot=0
    fi
  done
done
wait_batch

results_tsv="${sweep_dir}/results.tsv"
{
  echo -e "run_index\tstatus\tencoder\ttrain_config\tmodel_name\teval_task_name\ttask_config\tcheckpoint_id\tseed\tgpu_id\tsuccess_count\ttest_num\tsuccess_rate\tsummary_file\tresult_dir\tlog_file\tended_at"
  for record_file in "${run_record_files[@]}"; do
    if [ -f "${record_file}" ]; then
      cat "${record_file}"
    else
      echo -e "NA\tmissing_record\tNA\tNA\tNA\tNA\t${task_config}\t${checkpoint_id}\t${seed}\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t$(date -Iseconds)"
    fi
  done
} > "${results_tsv}"

python - "${results_tsv}" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

columns = ["status", "encoder", "eval_task_name", "success", "success_rate", "gpu_id", "log_file"]
display_rows = []
for row in rows:
    success_count = row.get("success_count") or "NA"
    test_num = row.get("test_num") or "NA"
    display_rows.append(
        {
            "status": row.get("status", "NA"),
            "encoder": row.get("encoder", "NA"),
            "eval_task_name": row.get("eval_task_name", "NA"),
            "success": f"{success_count}/{test_num}",
            "success_rate": row.get("success_rate", "NA"),
            "gpu_id": row.get("gpu_id", "NA"),
            "log_file": row.get("log_file", "NA"),
        }
    )

widths = {
    column: max(len(column), *(len(row[column]) for row in display_rows)) if display_rows else len(column)
    for column in columns
}
print("\nEval sweep results")
print("  ".join(column.ljust(widths[column]) for column in columns))
print("  ".join("-" * widths[column] for column in columns))
for row in display_rows:
    print("  ".join(row[column].ljust(widths[column]) for column in columns))
PY

echo
echo "Wrote sweep results: ${results_tsv}"
