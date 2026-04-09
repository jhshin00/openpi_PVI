# PVI Setup and Recipes

This document focuses on the repo-specific setup and command recipes for this fork.
Implementation details for the PyTorch PVI model live in [implementation.md](implementation.md).

## Repository Setup

Use the repo root `.venv` for both PVI training and local LIBERO-plus evaluation.

### Submodules

After cloning, initialize submodules:

```bash
git submodule update --init --recursive
```

### uv Environment

From the repo root:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --group libero-plus
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Recommended one-command setup:

```bash
./scripts/setup_libero_plus_env.sh
```

This script:

- syncs the root `.venv` with the extra LIBERO-plus dependencies
- generates `.libero-plus-config/config.yaml` for the current clone path
- applies the `transformers_replace` patch required for PyTorch checkpoints
- verifies that the root environment can import the LIBERO-plus benchmark stack

After setup, place the required local data at:

- LIBERO-plus assets: `third_party/libero-plus/libero/libero/assets`
- LIBERO-plus LeRobot dataset: `lerobot/physical-intelligence/libero`

### LIBERO-plus Local Patch

`third_party/libero-plus/` is intentionally gitignored in the parent repo because it is a separate checkout. The
current local checkout also needs a small PyTorch 2.6 compatibility patch in
`third_party/libero-plus/libero/libero/benchmark/__init__.py`, because `torch.load(...)` now defaults to
`weights_only=True` and that breaks LIBERO init-state loading.

Keep this helper in that file:

```python
def _load_init_states(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)
```

Then route the benchmark init-state loads through `_load_init_states(init_states_path)` instead of calling
`torch.load(init_states_path)` directly. If `third_party/libero-plus/` is re-cloned or reset, re-apply this patch.

## Quick Start

These commands assume the environment above is already prepared.

### Convert JAX Checkpoints to PyTorch

```bash
# pi0_base
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir gs://openpi-assets/checkpoints/pi0_base \
    --config_name pi0_libero \
    --output_path ./checkpoints/pytorch/pi0_base
```

```bash
# pi05_base
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir gs://openpi-assets/checkpoints/pi05_base \
    --config_name pi05_libero \
    --output_path ./checkpoints/pytorch/pi05_base
```

### Train PVI on LIBERO

```bash
# pi0 + PVI
CUDA_VISIBLE_DEVICES=0 HF_LEROBOT_HOME=/data/jhshin/openpi/datasets \
  uv run scripts/train_pytorch_PVI.py pi0_libero_pvi \
    --exp_name pi0_libero_pvi \
    --pytorch_weight_path ./checkpoints/pytorch/pi0_base
```

```bash
# pi05 + PVI
CUDA_VISIBLE_DEVICES=0 HF_LEROBOT_HOME=/data/jhshin/openpi/datasets \
  uv run scripts/train_pytorch_PVI.py pi05_libero_pvi \
    --exp_name pi05_libero_pvi \
    --pytorch_weight_path ./checkpoints/pytorch/pi05_base
```

## LIBERO-plus Recipes

### Train PI0.5 + PVI on the LIBERO-plus LeRobot Dataset

These commands assume:

- the dataset lives at `./lerobot/physical-intelligence/libero`
- the starting PyTorch checkpoint is `/data/jhshin/openpi/checkpoints/pytorch/pi05_libero`
- you want to fine-tune from the LIBERO-finetuned PI0.5 checkpoint, not from `pi05_base`

One-time setup for the root environment:

```bash
cd /data/jhshin/openpi-libero-plus

./scripts/setup_libero_plus_env.sh
```

This prepares the root `.venv`, generates `.libero-plus-config/config.yaml`, and applies the required
`transformers_replace` patch for PyTorch checkpoints.

Copy the dataset normalization stats into the config-specific assets directory:

```bash
cd /data/jhshin/openpi-libero-plus

mkdir -p assets/pi05_libero_plus_pvi_from_pi05_libero/physical-intelligence/libero
cp \
  lerobot/physical-intelligence/libero/norm_stats.json \
  assets/pi05_libero_plus_pvi_from_pi05_libero/physical-intelligence/libero/norm_stats.json
```

Run training:

```bash
cd /data/jhshin/openpi-libero-plus

CUDA_VISIBLE_DEVICES=2,3,4,5 \
HF_LEROBOT_HOME=$PWD/lerobot \
./.venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_pytorch_PVI.py pi05_libero_plus_pvi_from_pi05_libero \
  --exp_name pi05_libero_plus_pvi_from_pi05_libero_run1 \
  --pytorch_weight_path /data/jhshin/openpi/checkpoints/pytorch/pi05_libero
```

Notes:

- `pi05_libero_plus_pvi_from_pi05_libero` uses the LIBERO-plus LeRobot schema.
- `HF_LEROBOT_HOME` must point to `./lerobot`, not the old `/data/jhshin/openpi/datasets`.
- `--pytorch_weight_path` is passed explicitly here so the run does not depend on a local copy under `./checkpoints/pytorch/pi05_libero`.
- If you previously activated `/data/jhshin/openpi/.venv`, run `deactivate` first or use the explicit `./.venv/bin/python` command above.

### Evaluate on LIBERO-plus Without `serve_policy.py`

For long LIBERO-plus runs, local in-process evaluation is safer than websocket evaluation because there is no separate
policy server to disconnect mid-run.

One-time root-environment setup:

```bash
cd /data/jhshin/openpi-libero-plus

./scripts/setup_libero_plus_env.sh
```

Run one suite locally:

```bash
cd /data/jhshin/openpi-libero-plus

CUDA_VISIBLE_DEVICES=1 \
MUJOCO_EGL_DEVICE_ID=1 \
POLICY_CONFIG=pi05_libero_base_infer \
POLICY_DIR=/data/jhshin/openpi/checkpoints/pytorch/pi05_libero \
./scripts/eval_libero_plus_local_by_suite.sh libero_goal
```

Run all four main suites in parallel, one suite per GPU:

```bash
cd /data/jhshin/openpi-libero-plus

POLICY_CONFIG=pi05_libero_base_infer \
POLICY_DIR=/data/jhshin/openpi/checkpoints/pytorch/pi05_libero \
POLICY_PYTORCH_DEVICE=cuda:0 \
PARALLEL_JOBS=4 \
SUITE_CUDA_VISIBLE_DEVICES_MAP="libero_10:0,libero_spatial:1,libero_goal:2,libero_object:3" \
SUITE_MUJOCO_EGL_DEVICE_ID_MAP="libero_10:0,libero_spatial:1,libero_goal:2,libero_object:3" \
./scripts/eval_libero_plus_local_by_suite.sh
```

Resume a crashed suite from task index `N`:

```bash
cd /data/jhshin/openpi-libero-plus

CUDA_VISIBLE_DEVICES=1 \
MUJOCO_EGL_DEVICE_ID=1 \
POLICY_CONFIG=pi05_libero_base_infer \
POLICY_DIR=/data/jhshin/openpi/checkpoints/pytorch/pi05_libero \
TASK_START_INDEX=N \
OUTPUT_TAG=pi05_libero_spatial_retry \
./scripts/eval_libero_plus_local_by_suite.sh libero_spatial
```

Run an inclusive task range `[start, end]`:

```bash
cd /data/jhshin/openpi-libero-plus

CUDA_VISIBLE_DEVICES=3 \
MUJOCO_EGL_DEVICE_ID=0 \
POLICY_PYTORCH_DEVICE=cuda:0 \
NUM_TRIALS_PER_TASK=1 \
POLICY_CONFIG=pi05_libero_pvi_infer_hpr \
POLICY_DIR=/data/jhshin/openpi/checkpoints/pi05_libero_pvi_from_pi05_libero_hpr/pi05_libero_pvi_from_pi05_libero_hpr/40000 \
OUTPUT_TAG=pi05_libero_pvi_hpr_from_pi05_libero_ckpt_40000 \
SUITE_OUTPUT_DIR_MAP="libero_10:libero_10_resume" \
TASK_START_INDEX=1611 \
TASK_END_INDEX=1800 \
./scripts/eval_libero_plus_local_by_suite.sh libero_10
```

Outputs are written under:

- `data/libero_plus_eval/<output_tag>/<suite>/summary.json`
- `data/libero_plus_eval/<output_tag>/per_suite_category_summary.json`
- `data/libero_plus_eval/<output_tag>/per_suite_category_summary.csv`
- `data/libero_plus_eval/<output_tag>/per_suite_category_summary_long.csv`

`summary.json` is updated after each completed task, so partial progress survives process crashes.

CUDA_VISIBLE_DEVICES=2,3,4,5\
  HF_LEROBOT_HOME=$PWD/lerobot \
  ./.venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch_PVI.py pi05_libero_plus_pvi_from_pi05_libero \
    --exp_name pi05_libero_plus_pvi_bs128_160k_from_pi05_libero \
    --resume

CUDA_VISIBLE_DEVICES=4 \
  MUJOCO_EGL_DEVICE_ID=6 \
  POLICY_PYTORCH_DEVICE=cuda:0 \
  NUM_TRIALS_PER_TASK=1 \
  POLICY_CONFIG=pi05_libero_pvi_infer \
  POLICY_DIR=/data/jhshin/openpi/checkpoints/pi05_libero_pvi_from_pi05_libero/pi05_libero_pvi_bs128_40k_from_pi05_libero/40000 \
  OUTPUT_TAG=pi05_libero_pvi_dino_bs128_40k_from_pi05_libero_ckpt \
  ./scripts/eval_libero_plus_local_by_suite.sh libero_object

  CUDA_VISIBLE_DEVICES=5 \
  MUJOCO_EGL_DEVICE_ID=5 \
  POLICY_PYTORCH_DEVICE=cuda:0 \
  NUM_TRIALS_PER_TASK=1 \
  POLICY_CONFIG=pi05_libero_pvi_infer \
  POLICY_DIR=/data/jhshin/openpi/checkpoints/pi05_libero_pvi_from_pi05_libero/pi05_libero_pvi_bs128_40k_from_pi05_libero/40000 \
  OUTPUT_TAG=pi05_libero_pvi_dino_bs128_40k_from_pi05_libero_ckpt \
  ./scripts/eval_libero_plus_local_by_suite.sh libero_10

  CUDA_VISIBLE_DEVICES=6 \
  MUJOCO_EGL_DEVICE_ID=4 \
  POLICY_PYTORCH_DEVICE=cuda:0 \
  NUM_TRIALS_PER_TASK=1 \
  POLICY_CONFIG=pi05_libero_pvi_infer \
  POLICY_DIR=/data/jhshin/openpi/checkpoints/pi05_libero_pvi_from_pi05_libero/pi05_libero_pvi_bs128_40k_from_pi05_libero/40000 \
  OUTPUT_TAG=pi05_libero_pvi_dino_bs128_40k_from_pi05_libero_ckpt \
  ./scripts/eval_libero_plus_local_by_suite.sh libero_goal

  참고로 spatial은 그대로 이겁니다:

  CUDA_VISIBLE_DEVICES=3 \
  MUJOCO_EGL_DEVICE_ID=0 \
  POLICY_PYTORCH_DEVICE=cuda:0 \
  NUM_TRIALS_PER_TASK=1 \
  POLICY_CONFIG=pi05_libero_pvi_infer \
  POLICY_DIR=/data/jhshin/openpi/checkpoints/pi05_libero_pvi_from_pi05_libero/pi05_libero_pvi_bs128_40k_from_pi05_libero/40000 \
  OUTPUT_TAG=pi05_libero_pvi_dino_bs128_40k_from_pi05_libero_ckpt \
  SUITE_OUTPUT_DIR_MAP="libero_spatial:libero_spatial_resume" \
  TASK_START_INDEX=1691 \
  ./scripts/eval_libero_plus_local_by_suite.sh libero_spatial
