#!/bin/bash

set -euo pipefail

config_selector=${1:-all}
gpu_use=${2:-2,4,5,6}
lerobot_root=${3:-/data/shkim/RoboTwin/lerobot_data}
mode=${4:-overwrite}

shift $(( $# >= 4 ? 4 : $# ))

if [ "$#" -gt 0 ]; then
  tasks=("$@")
else
  tasks=(
    robotwin_lift_pot_aloha_agilex_clean_50
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
    hpr_v1|er_hpr_v1|pi05_robotwin_encoder_replace_hpr)
      echo "pi05_robotwin_encoder_replace_hpr"
      ;;
    hpr_v2|er_hpr_v2|pi05_robotwin_encoder_replace_hpr_v2)
      echo "pi05_robotwin_encoder_replace_hpr_v2"
      ;;
    hpr_v3|er_hpr_v3|pi05_robotwin_encoder_replace_hpr_v3)
      echo "pi05_robotwin_encoder_replace_hpr_v3"
      ;;
    siglip_v1|er_siglip_v1|pi05_robotwin_encoder_replace_siglip)
      echo "pi05_robotwin_encoder_replace_siglip"
      ;;
    siglip_v2|er_siglip_v2|pi05_robotwin_encoder_replace_siglip_v2)
      echo "pi05_robotwin_encoder_replace_siglip_v2"
      ;;
    siglip_v3|er_siglip_v3|pi05_robotwin_encoder_replace_siglip_v3)
      echo "pi05_robotwin_encoder_replace_siglip_v3"
      ;;
    dino|dinov2|er_dino|pi05_robotwin_encoder_replace_dino)
      echo "pi05_robotwin_encoder_replace_dino"
      ;;
    clip|er_clip|pi05_robotwin_encoder_replace_clip)
      echo "pi05_robotwin_encoder_replace_clip"
      ;;
    r3m|er_r3m|pi05_robotwin_encoder_replace_r3m)
      echo "pi05_robotwin_encoder_replace_r3m"
      ;;
    *)
      echo "Unknown encoder-replacement config selector: ${selector}" >&2
      echo "Expected one of: all, hpr_v1, hpr_v2, hpr_v3, siglip_v1, siglip_v2, siglip_v3," >&2
      echo "dino, clip, r3m, or a pi05_robotwin_encoder_replace_* config name" >&2
      exit 1
      ;;
  esac
}

resolve_num_train_steps() {
  local task=$1
  case "${task}" in
    robotwin_lift_pot_aloha_agilex_clean_50)
      echo "30000"
      ;;
    robotwin_pick_dual_bottles_aloha_agilex_clean_50|robotwin_place_phone_stand_aloha_agilex_clean_50)
      echo "5000"
      ;;
    robotwin_place_empty_cup_aloha_agilex_clean_50|robotwin_handover_mic_aloha_agilex_clean_50)
      echo "10000"
      ;;
    robotwin_blocks_ranking_rgb_aloha_agilex_clean_50|robotwin_put_bottles_dustbin_aloha_agilex_clean_50)
      echo "40000"
      ;;
    *)
      echo "Unknown task for num_train_steps: ${task}" >&2
      exit 1
      ;;
  esac
}

if [ "${config_selector}" = "all" ]; then
  train_configs=(
    pi05_robotwin_encoder_replace_hpr
    pi05_robotwin_encoder_replace_hpr_v2
    pi05_robotwin_encoder_replace_hpr_v3
  )
else
  IFS=',' read -r -a selectors <<< "${config_selector}"
  train_configs=()
  for selector in "${selectors[@]}"; do
    train_configs+=("$(resolve_train_config "${selector}")")
  done
fi

export CUDA_VISIBLE_DEVICES="${gpu_use}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

IFS=',' read -r -a gpu_array <<< "${gpu_use}"
gpu_count=${#gpu_array[@]}
total_runs=$(( ${#train_configs[@]} * ${#tasks[@]} ))

echo "Selected configs: ${train_configs[*]}"
echo "Selected tasks: ${tasks[*]}"
echo "Total training runs: ${total_runs}"

for task in "${tasks[@]}"; do
  num_train_steps="$(resolve_num_train_steps "${task}")"
  for train_config_name in "${train_configs[@]}"; do
    model_name="${task}"

    echo "=== Training ${task} (config=${train_config_name}, steps=${num_train_steps}, mode=${mode}) ==="

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
