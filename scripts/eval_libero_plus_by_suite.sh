#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_PYTHON="$ROOT_DIR/examples/libero/.venv/bin/python"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
START_SERVER="${START_SERVER:-1}"
SERVER_CONFIG="${SERVER_CONFIG:-pi05_libero_base_infer}"
SERVER_CHECKPOINT_DIR="${SERVER_CHECKPOINT_DIR:-/data/jhshin/openpi/checkpoints/pytorch/pi05_libero}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
OUTPUT_TAG="${OUTPUT_TAG:-$(basename "$SERVER_CHECKPOINT_DIR")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/data/libero_plus_eval/$OUTPUT_TAG}"
TASK_CATEGORY="${TASK_CATEGORY:-}"
DIFFICULTY_LEVEL="${DIFFICULTY_LEVEL:-}"
TASK_NAME_PATTERN="${TASK_NAME_PATTERN:-}"
TASK_LIMIT="${TASK_LIMIT:-}"
MUJOCO_GL_VALUE="${MUJOCO_GL:-}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
SUITE_PORT_MAP="${SUITE_PORT_MAP:-}"
SUITE_HOST_MAP="${SUITE_HOST_MAP:-}"

if [[ "$#" -gt 0 ]]; then
  SUITES=("$@")
else
  SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")
fi

export LIBERO_CONFIG_PATH="$ROOT_DIR/.libero-plus-config"
export PYTHONPATH="$ROOT_DIR/third_party/libero-plus:${PYTHONPATH:-}"

mkdir -p "$OUTPUT_ROOT"

if [[ ! -x "$CLIENT_PYTHON" ]]; then
  echo "Missing client python at $CLIENT_PYTHON" >&2
  exit 1
fi

if ! "$CLIENT_PYTHON" -c "import libero" >/dev/null 2>&1; then
  echo "The examples/libero client environment does not have LIBERO-plus installed." >&2
  echo "Run: uv pip install -e third_party/libero-plus" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}

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

wait_for_server() {
  local host="$1"
  local port="$2"
  local retries="${3:-60}"
  local delay_s="${4:-2}"

  for _ in $(seq 1 "$retries"); do
    if (echo >"/dev/tcp/$host/$port") >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay_s"
  done
  return 1
}

if [[ "$START_SERVER" == "1" ]]; then
  if [[ -n "$SUITE_PORT_MAP" || -n "$SUITE_HOST_MAP" ]]; then
    echo "START_SERVER=1 only supports a single server. For per-suite host/port routing, start servers manually and use START_SERVER=0." >&2
    exit 1
  fi
  SERVER_LOG="$OUTPUT_ROOT/server.log"
  (
    cd "$ROOT_DIR"
    uv run scripts/serve_policy.py \
      policy:checkpoint \
      --policy.config "$SERVER_CONFIG" \
      --policy.dir "$SERVER_CHECKPOINT_DIR" \
      --port "$PORT"
  ) >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  trap cleanup EXIT

  if ! wait_for_server "$HOST" "$PORT"; then
    echo "Server did not become ready on $HOST:$PORT. Check $SERVER_LOG" >&2
    exit 1
  fi
fi

run_suite() {
  local suite="$1"
  local suite_host
  local suite_port
  suite_host="$(resolve_suite_value "$suite" "$SUITE_HOST_MAP" "$HOST")"
  suite_port="$(resolve_suite_value "$suite" "$SUITE_PORT_MAP" "$PORT")"
  suite_output_dir="$OUTPUT_ROOT/$suite"
  suite_log="$suite_output_dir/eval.log"
  suite_summary="$suite_output_dir/summary.json"

  mkdir -p "$suite_output_dir"

  cmd=(
    "$CLIENT_PYTHON"
    "$ROOT_DIR/examples/libero/main.py"
    --args.host "$suite_host"
    --args.port "$suite_port"
    --args.task-suite-name "$suite"
    --args.num-trials-per-task "$NUM_TRIALS_PER_TASK"
    --args.summary-json-path "$suite_summary"
    --args.video-out-path "$suite_output_dir/videos"
    --args.no-save-videos
  )

  if [[ -n "$TASK_CATEGORY" ]]; then
    cmd+=(--args.task-category "$TASK_CATEGORY")
  fi
  if [[ -n "$DIFFICULTY_LEVEL" ]]; then
    cmd+=(--args.difficulty-level "$DIFFICULTY_LEVEL")
  fi
  if [[ -n "$TASK_NAME_PATTERN" ]]; then
    cmd+=(--args.task-name-pattern "$TASK_NAME_PATTERN")
  fi
  if [[ -n "$TASK_LIMIT" ]]; then
    cmd+=(--args.task-limit "$TASK_LIMIT")
  fi

  echo "Evaluating suite: $suite (server ${suite_host}:${suite_port})"
  if [[ -n "$MUJOCO_GL_VALUE" ]]; then
    MUJOCO_GL="$MUJOCO_GL_VALUE" "${cmd[@]}" 2>&1 | tee "$suite_log"
  else
    "${cmd[@]}" 2>&1 | tee "$suite_log"
  fi
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

"$CLIENT_PYTHON" - <<'PY' "$OUTPUT_ROOT" "${SUITES[@]}"
import json
import pathlib
import sys

output_root = pathlib.Path(sys.argv[1])
suites = sys.argv[2:]
aggregate = {}

for suite in suites:
    summary_path = output_root / suite / "summary.json"
    if not summary_path.exists():
        continue
    with open(summary_path, "r") as f:
        summary = json.load(f)
    aggregate[suite] = {
        "total_success_rate": summary["total_success_rate"],
        "total_episodes": summary["total_episodes"],
        "total_successes": summary["total_successes"],
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
