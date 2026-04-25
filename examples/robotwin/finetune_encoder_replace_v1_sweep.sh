#!/bin/bash

set -euo pipefail

gpu_use=${1:-0,1,6,7}
lerobot_root=${2:-./datasets}
mode=${3:-overwrite}
num_train_steps=${4:-30000}
encoder_selector=${5:-dinov2,siglip,clip,r3m}

shift $(( $# >= 5 ? 5 : $# ))

if [ "$#" -gt 0 ]; then
  task_selectors=("$@")
else
  task_selectors=(
    robotwin_lift_pot_aloha_agilex_clean_50
    robotwin_move_can_pot_aloha_agilex_clean_50
    robotwin_beat_hammer_block_aloha_agilex_clean_50
  )
fi

case "${mode}" in
  overwrite|resume)
    ;;
  *)
    echo "Invalid mode: ${mode}"
    echo "Expected one of: overwrite, resume"
    exit 1
    ;;
esac

resolve_train_config() {
  local selector=$1
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

resolve_task() {
  local selector=$1
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
    robotwin_*_aloha_agilex_clean_50)
      echo "${selector}"
      ;;
    *)
      echo "Unknown task selector: ${selector}" >&2
      echo "Expected one of: lift_pot, move_can_pot, beat_hammer_block, or a full robotwin_* task id" >&2
      exit 1
      ;;
  esac
}

IFS=',' read -r -a encoder_selectors <<< "${encoder_selector}"
train_configs=()
for selector in "${encoder_selectors[@]}"; do
  train_configs+=("$(resolve_train_config "${selector}")")
done

tasks=()
for selector in "${task_selectors[@]}"; do
  tasks+=("$(resolve_task "${selector}")")
done

export CUDA_VISIBLE_DEVICES="${gpu_use}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

IFS=',' read -r -a gpu_array <<< "${gpu_use}"
gpu_count=${#gpu_array[@]}
total_runs=$(( ${#train_configs[@]} * ${#tasks[@]} ))

echo "Selected v1 configs: ${train_configs[*]}"
echo "Selected tasks: ${tasks[*]}"
echo "num_train_steps=${num_train_steps}"
echo "mode=${mode}"
echo "Total training runs: ${total_runs}"

run_index=0
for train_config_name in "${train_configs[@]}"; do
  for task in "${tasks[@]}"; do
    run_index=$((run_index + 1))
    model_name="${task}"

    echo "=== [${run_index}/${total_runs}] Training ${task} (config=${train_config_name}, steps=${num_train_steps}, mode=${mode}) ==="
    date

    cmd=(
      examples/robotwin/workflow/finetune_encoder_replace.py
      --train-config-name "${train_config_name}"
      --model-name "${model_name}"
      --repo-id "${task}"
      --lerobot-root "${lerobot_root}"
      --asset-id "${task}"
      --num-train-steps "${num_train_steps}"
    )

    if [ "${mode}" = "overwrite" ]; then
      cmd+=(--overwrite)
    else
      cmd+=(--resume)
    fi

    if [ "${gpu_count}" -gt 1 ]; then
      uv run torchrun --standalone --nnodes=1 --nproc_per_node="${gpu_count}" "${cmd[@]}"
    else
      uv run python "${cmd[@]}"
    fi
  done
done
