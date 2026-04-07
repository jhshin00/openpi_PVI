#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_PYTHON="$ROOT_DIR/.venv/bin/python"

POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero_base_infer}"
POLICY_DIR="${POLICY_DIR:-/data/jhshin/openpi/checkpoints/pytorch/pi05_libero}"
POLICY_PYTORCH_DEVICE="${POLICY_PYTORCH_DEVICE:-}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
OUTPUT_TAG="${OUTPUT_TAG:-$(basename "$POLICY_DIR")_local}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/data/libero_plus_eval/$OUTPUT_TAG}"
TASK_CATEGORY="${TASK_CATEGORY:-}"
DIFFICULTY_LEVEL="${DIFFICULTY_LEVEL:-}"
TASK_NAME_PATTERN="${TASK_NAME_PATTERN:-}"
TASK_START_INDEX="${TASK_START_INDEX:-}"
TASK_LIMIT="${TASK_LIMIT:-}"
MUJOCO_GL_VALUE="${MUJOCO_GL:-}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
SUITE_CUDA_VISIBLE_DEVICES_MAP="${SUITE_CUDA_VISIBLE_DEVICES_MAP:-}"
SUITE_MUJOCO_EGL_DEVICE_ID_MAP="${SUITE_MUJOCO_EGL_DEVICE_ID_MAP:-}"
SUITE_POLICY_DEVICE_MAP="${SUITE_POLICY_DEVICE_MAP:-}"
SUITE_OUTPUT_DIR_MAP="${SUITE_OUTPUT_DIR_MAP:-}"

if [[ "$#" -gt 0 ]]; then
  SUITES=("$@")
else
  SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")
fi

mkdir -p "$OUTPUT_ROOT"

if [[ ! -x "$CLIENT_PYTHON" ]]; then
  echo "Missing root python at $CLIENT_PYTHON" >&2
  exit 1
fi

validate_egl_device_id() {
  local egl_device="$1"

  if [[ -z "$egl_device" ]]; then
    return
  fi

  if [[ ! "$egl_device" =~ ^[0-9]+$ ]]; then
    echo "Invalid EGL configuration: MUJOCO_EGL_DEVICE_ID must be a non-negative integer, got '$egl_device'" >&2
    exit 1
  fi
}

validate_egl_device_id "${MUJOCO_EGL_DEVICE_ID:-}"

if ! LIBERO_CONFIG_PATH="$ROOT_DIR/.libero-plus-config" \
  PYTHONPATH="$ROOT_DIR/third_party/libero-plus:${PYTHONPATH:-}" \
  "$CLIENT_PYTHON" -c "import openpi, robosuite, bddl, robomimic, wand, skimage; from libero.libero import benchmark" >/dev/null 2>&1; then
  echo "The root environment is missing LIBERO-plus local-eval dependencies." >&2
  echo "Run the setup script, then retry:" >&2
  echo "  ./scripts/setup_libero_plus_env.sh" >&2
  exit 1
fi

resolve_suite_value() {
  local suite="$1"
  local mapping="$2"
  local default_value="$3"

  if [[ -z "$mapping" ]]; then
    echo "$default_value"
    return
  fi

  IFS=',' read -ra entries <<< "$mapping"
  for entry in "${entries[@]}"; do
    [[ -z "$entry" ]] && continue
    local key="${entry%%:*}"
    local value="${entry#*:}"
    if [[ "$key" == "$suite" ]]; then
      echo "$value"
      return
    fi
  done

  echo "$default_value"
}

run_suite() {
  local suite="$1"
  local suite_cuda
  local suite_egl
  local suite_policy_device
  local suite_output_name

  suite_cuda="$(resolve_suite_value "$suite" "$SUITE_CUDA_VISIBLE_DEVICES_MAP" "${CUDA_VISIBLE_DEVICES:-}")"
  suite_egl="$(resolve_suite_value "$suite" "$SUITE_MUJOCO_EGL_DEVICE_ID_MAP" "${MUJOCO_EGL_DEVICE_ID:-}")"
  suite_policy_device="$(resolve_suite_value "$suite" "$SUITE_POLICY_DEVICE_MAP" "$POLICY_PYTORCH_DEVICE")"
  suite_output_name="$(resolve_suite_value "$suite" "$SUITE_OUTPUT_DIR_MAP" "$suite")"

  validate_egl_device_id "$suite_egl"

  local suite_output_dir="$OUTPUT_ROOT/$suite_output_name"
  local suite_log="$suite_output_dir/eval.log"
  local suite_summary="$suite_output_dir/summary.json"

  mkdir -p "$suite_output_dir"

  local -a env_cmd=(
    env
    "LIBERO_CONFIG_PATH=$ROOT_DIR/.libero-plus-config"
    "PYTHONPATH=$ROOT_DIR/third_party/libero-plus:${PYTHONPATH:-}"
  )
  if [[ -n "$suite_cuda" ]]; then
    env_cmd+=("CUDA_VISIBLE_DEVICES=$suite_cuda")
  fi
  if [[ -n "$suite_egl" ]]; then
    env_cmd+=("MUJOCO_EGL_DEVICE_ID=$suite_egl")
  fi
  if [[ -n "$MUJOCO_GL_VALUE" ]]; then
    env_cmd+=("MUJOCO_GL=$MUJOCO_GL_VALUE")
  fi

  local -a cmd=(
    "$CLIENT_PYTHON"
    "$ROOT_DIR/examples/libero/main.py"
    --args.task-suite-name "$suite"
    --args.num-trials-per-task "$NUM_TRIALS_PER_TASK"
    --args.summary-json-path "$suite_summary"
    --args.video-out-path "$suite_output_dir/videos"
    --args.no-save-videos
    --args.policy-config "$POLICY_CONFIG"
    --args.policy-dir "$POLICY_DIR"
  )

  if [[ -n "$suite_policy_device" ]]; then
    cmd+=(--args.policy-pytorch-device "$suite_policy_device")
  fi
  if [[ -n "$TASK_CATEGORY" ]]; then
    cmd+=(--args.task-category "$TASK_CATEGORY")
  fi
  if [[ -n "$DIFFICULTY_LEVEL" ]]; then
    cmd+=(--args.difficulty-level "$DIFFICULTY_LEVEL")
  fi
  if [[ -n "$TASK_NAME_PATTERN" ]]; then
    cmd+=(--args.task-name-pattern "$TASK_NAME_PATTERN")
  fi
  if [[ -n "$TASK_START_INDEX" ]]; then
    cmd+=(--args.task-start-index "$TASK_START_INDEX")
  fi
  if [[ -n "$TASK_LIMIT" ]]; then
    cmd+=(--args.task-limit "$TASK_LIMIT")
  fi

  echo "Evaluating suite: $suite (local policy config=$POLICY_CONFIG dir=$POLICY_DIR cuda=${suite_cuda:-inherit})"
  "${env_cmd[@]}" "${cmd[@]}" 2>&1 | tee "$suite_log"
}

if [[ "$PARALLEL_JOBS" -le 1 ]]; then
  for suite in "${SUITES[@]}"; do
    run_suite "$suite"
  done
else
  pids=()
  for suite in "${SUITES[@]}"; do
    run_suite "$suite" &
    pids+=("$!")
    while [[ "$(jobs -pr | wc -l)" -ge "$PARALLEL_JOBS" ]]; do
      sleep 1
    done
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
fi

"$CLIENT_PYTHON" - <<'PY' "$OUTPUT_ROOT" "$SUITE_OUTPUT_DIR_MAP" "${SUITES[@]}"
import json
import pathlib
import sys

output_root = pathlib.Path(sys.argv[1])
output_dir_map = sys.argv[2]
suites = sys.argv[3:]
aggregate = {}
known_suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
candidate_suites = []

for suite in known_suites + list(suites):
    if suite not in candidate_suites:
        candidate_suites.append(suite)


def resolve_output_dir_name(suite: str) -> str:
    if not output_dir_map:
        return suite

    for entry in output_dir_map.split(","):
        if not entry:
            continue
        key, value = entry.split(":", 1)
        if key == suite:
            return value
    return suite

for suite in candidate_suites:
    summary_path = output_root / resolve_output_dir_name(suite) / "summary.json"
    if not summary_path.exists():
        continue
    with open(summary_path, "r") as f:
        summary = json.load(f)
    aggregate[suite] = {
        "total_success_rate": summary["total_success_rate"],
        "total_episodes": summary["total_episodes"],
        "total_successes": summary["total_successes"],
        "num_completed_tasks": summary.get("num_completed_tasks"),
        "num_tasks_in_suite": summary.get("num_tasks_in_suite"),
        "per_category": summary["per_category"],
        "per_difficulty": summary["per_difficulty"],
    }

summary_index_path = output_root / "per_suite_category_summary.json"
summary_index_path.write_text(json.dumps(aggregate, indent=2))
print(f"Wrote aggregate summary to {summary_index_path}")
PY

"$CLIENT_PYTHON" "$ROOT_DIR/scripts/export_libero_plus_summary_csv.py" \
  --input-json "$OUTPUT_ROOT/per_suite_category_summary.json" \
  --output-csv "$OUTPUT_ROOT/per_suite_category_summary.csv" \
  --output-long-csv "$OUTPUT_ROOT/per_suite_category_summary_long.csv"
