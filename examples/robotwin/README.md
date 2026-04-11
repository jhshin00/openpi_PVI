# RoboTwin Integration

This directory integrates RoboTwin into this `openpi` repo without copying RoboTwin's vendored `policy/pi05` fork.

The intended split is:

- train/data conversion in the repo root `openpi` environment
- same-process RoboTwin simulator eval in a separate RoboTwin-capable environment

The workflow shape follows the official RoboTwin `Pi0.5` docs, but training uses this repo's PVI PyTorch trainer instead of RoboTwin's vendored JAX trainer.

References:

- RoboTwin install guide: https://robotwin-platform.github.io/doc/usage/robotwin-install.html
- RoboTwin Pi0.5 guide: https://robotwin-platform.github.io/doc/usage/Pi05.html
- RoboTwin repo: https://github.com/RoboTwin-Platform/RoboTwin
- RoboTwin pre-collected dataset release: https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset

## What Lives Here

- `third_party/robotwin`: full RoboTwin simulator/runtime as a git submodule
- `examples/robotwin/process_robotwin_data.py`: raw RoboTwin HDF5 -> processed Aloha-style HDF5
- `examples/robotwin/convert_robotwin_to_lerobot.py`: processed RoboTwin folders -> local LeRobot dataset
- `examples/robotwin/eval.py`: same-process RoboTwin eval using local `openpi` checkpoints
- `examples/robotwin/process_data_pi05.sh`
- `examples/robotwin/generate.sh`
- `examples/robotwin/compute_norm_stats.sh`
- `examples/robotwin/finetune.sh`
- `examples/robotwin/eval.sh`
- `examples/robotwin/bootstrap_eval_env.sh`: creates the RoboTwin eval env

## Directory Convention

Use the following paths.

- raw RoboTwin demos: `./datasets/robotwin_raw/<task_name>/<task_config>/`
- processed RoboTwin demos: `./datasets/robotwin_processed/<task_name>-<task_config>-<N>/`
- RoboTwin training staging root: `./datasets/robotwin_training/<model_name>/`
- final LeRobot dataset: `./datasets/<repo_id>/`

Recommended naming:

- `model_name`: the training bundle name, for example `demo_clean_mt10`
- `repo_id`: keep it simple and match `model_name`, for example `demo_clean_mt10`
- `exp_name`: the training run name; the wrappers set this to `model_name`

Using a simple `repo_id` like `demo_clean_mt10` is recommended.

## 0. Initialize the Repo

```bash
git submodule update --init --recursive
```

If you do not already have the base PyTorch `pi0.5` checkpoint locally, create it once:

```bash
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir gs://openpi-assets/checkpoints/pi05_base \
  --output_path ./checkpoints/pytorch/pi05_base
```

The RoboTwin PVI configs in [src/openpi/training/config.py](/data/jhshin/openpi/src/openpi/training/config.py) expect `./checkpoints/pytorch/pi05_base`.

## 1. Train Env

Use the repo root `uv` environment for:

- raw -> processed conversion
- processed -> LeRobot conversion
- norm stats
- PVI training

Create or refresh it from the repo root:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
```

Activate it when you want to run the shell wrappers manually:

```bash
source .venv/bin/activate
```

If your Hugging Face or datasets cache should live somewhere with more disk space, set it before conversion:

```bash
export HF_HOME=/path/to/hf-home
export HF_DATASETS_CACHE=/path/to/hf-home/datasets
export XDG_CACHE_HOME=/path/to/xdg-cache
```

## 2. Eval Env

Use a separate RoboTwin-capable env for same-process simulator eval.

This env needs:

- RoboTwin simulator deps
- `sapien`, `mplib`, `open3d`, `pytorch3d`, `curobo`
- this repo installed so `openpi` checkpoints can be loaded in the same process

Create it with:

```bash
bash examples/robotwin/bootstrap_eval_env.sh --download-assets
```

This bootstrap script defaults to Python `3.11`.
That is intentional for this repo because the local `openpi` branch and the current RoboTwin `policy/pi05` stack are easiest to keep aligned there.

Default path:

- `examples/robotwin/.venv`

Activate it before RoboTwin eval:

```bash
source examples/robotwin/.venv/bin/activate
```

This script follows the official RoboTwin `Pi0.5` flow:

1. create a dedicated `uv` env
2. install this repo into that env
3. install RoboTwin simulator extras
4. patch `sapien` and `mplib`
5. install `curobo`
6. optionally download RoboTwin assets

If `av` build fails or RoboTwin warns about ffmpeg, check the official docs and make sure `ffmpeg` works:

```bash
ffmpeg -version
```

## 3. Prepare Raw RoboTwin Data

The official RoboTwin `Pi0.5` docs assume raw RoboTwin demos already exist in the standard layout:

```text
<raw_root>/<task_name>/<task_config>/
├── data/
│   ├── episode0.hdf5
│   ├── episode1.hdf5
│   └── ...
└── instructions/
    ├── episode0.json
    ├── episode1.json
    └── ...
```

In this repo, the recommended raw root is:

```text
./datasets/robotwin_raw
```

You have two normal ways to get raw data.

### Option A: Download Pre-Collected RoboTwin Data

RoboTwin publishes a large pre-collected dataset here:

- https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset

After downloading and unpacking, place the task folder so it looks like:

```text
./datasets/robotwin_raw/beat_block_hammer/demo_clean/
├── data/
└── instructions/
```

### Option B: Collect with RoboTwin

Run RoboTwin's official data collection command inside the RoboTwin env:

```bash
cd third_party/robotwin
bash collect_data.sh beat_block_hammer demo_clean 0
cd ../..
```

That writes raw data under:

```text
third_party/robotwin/data/beat_block_hammer/demo_clean/
```

You can either keep it there and pass that path explicitly, or move/copy it into `./datasets/robotwin_raw`.

Example:

```bash
mkdir -p ./datasets/robotwin_raw/beat_block_hammer
cp -R third_party/robotwin/data/beat_block_hammer/demo_clean ./datasets/robotwin_raw/beat_block_hammer/
```

## 4. End-to-End Command Order

This is the intended execution order if raw RoboTwin demos are already under `./datasets/robotwin_raw`.

### Single-Task Example

Example task:

- `task_name=beat_block_hammer`
- `task_config=demo_clean`
- `expert_data_num=50`
- `model_name=demo_clean_single`
- `repo_id=demo_clean_single`

#### 4.1 Process raw RoboTwin data

```bash
bash examples/robotwin/process_data_pi05.sh \
  beat_block_hammer \
  demo_clean \
  50
```

This reads:

```text
./datasets/robotwin_raw/beat_block_hammer/demo_clean
```

and writes:

```text
./datasets/robotwin_processed/beat_block_hammer-demo_clean-50
```

#### 4.2 Stage the training root

```bash
mkdir -p ./datasets/robotwin_training/demo_clean_single
ln -sfn \
  "$(pwd)/datasets/robotwin_processed/beat_block_hammer-demo_clean-50" \
  "./datasets/robotwin_training/demo_clean_single/beat_block_hammer-demo_clean-50"
```

You can copy instead of symlink if you prefer.

#### 4.3 Generate the local LeRobot dataset

```bash
bash examples/robotwin/generate.sh \
  ./datasets/robotwin_training/demo_clean_single \
  demo_clean_single
```

This writes the final LeRobot dataset under:

```text
./datasets/demo_clean_single
```

#### 4.4 Compute norm stats

```bash
bash examples/robotwin/compute_norm_stats.sh \
  pi05_robotwin_pvi_from_base \
  demo_clean_single
```

This writes norm stats under:

```text
./assets/pi05_robotwin_pvi/demo_clean_single
```

#### 4.5 Train PVI

Single GPU:

```bash
bash examples/robotwin/finetune.sh \
  pi05_robotwin_pvi_from_base \
  demo_clean_single \
  0 \
  demo_clean_single
```

Multi GPU:

```bash
bash examples/robotwin/finetune.sh \
  pi05_robotwin_pvi_from_base \
  demo_clean_single \
  0,1 \
  demo_clean_single
```

This uses:

- config: `pi05_robotwin_pvi_from_base`
- dataset: `./datasets/demo_clean_single`
- norm stats: `./assets/pi05_robotwin_pvi/demo_clean_single`
- checkpoint dir: `./checkpoints/pi05_robotwin_pvi_from_base/demo_clean_single`

#### 4.6 Evaluate in RoboTwin

Activate the RoboTwin eval env first:

```bash
source examples/robotwin/.venv/bin/activate
```

Then run official-style task-by-task eval:

```bash
bash examples/robotwin/eval.sh \
  beat_block_hammer \
  demo_clean \
  pi05_robotwin_pvi_from_base \
  demo_clean_single \
  0 \
  0
```

If you want to evaluate a checkpoint other than the default `30000`, add it as the seventh argument:

```bash
bash examples/robotwin/eval.sh \
  beat_block_hammer \
  demo_clean \
  pi05_robotwin_pvi_from_base \
  demo_clean_single \
  0 \
  0 \
  10000
```

To test robustness on the harder randomized setting:

```bash
bash examples/robotwin/eval.sh \
  beat_block_hammer \
  demo_randomized \
  pi05_robotwin_pvi_from_base \
  demo_clean_single \
  0 \
  0
```

### Multi-Task Example

This matches the official RoboTwin `training_data/<model_name>/` idea.

Example:

- `model_name=demo_clean_mt3`
- `repo_id=demo_clean_mt3`

Process each task first:

```bash
bash examples/robotwin/process_data_pi05.sh beat_block_hammer demo_clean 50
bash examples/robotwin/process_data_pi05.sh stack_blocks_two demo_clean 50
bash examples/robotwin/process_data_pi05.sh place_bread_basket demo_clean 50
```

Build a RoboTwin-style training root:

```bash
mkdir -p ./datasets/robotwin_training/demo_clean_mt3
ln -sfn "$(pwd)/datasets/robotwin_processed/beat_block_hammer-demo_clean-50" \
  "./datasets/robotwin_training/demo_clean_mt3/beat_block_hammer-demo_clean-50"
ln -sfn "$(pwd)/datasets/robotwin_processed/stack_blocks_two-demo_clean-50" \
  "./datasets/robotwin_training/demo_clean_mt3/stack_blocks_two-demo_clean-50"
ln -sfn "$(pwd)/datasets/robotwin_processed/place_bread_basket-demo_clean-50" \
  "./datasets/robotwin_training/demo_clean_mt3/place_bread_basket-demo_clean-50"
```

Generate one LeRobot dataset from the whole training root:

```bash
bash examples/robotwin/generate.sh \
  ./datasets/robotwin_training/demo_clean_mt3 \
  demo_clean_mt3
```

Then continue as usual:

```bash
bash examples/robotwin/compute_norm_stats.sh \
  pi05_robotwin_pvi_from_base \
  demo_clean_mt3

bash examples/robotwin/finetune.sh \
  pi05_robotwin_pvi_from_base \
  demo_clean_mt3 \
  0,1 \
  demo_clean_mt3
```

Eval is still task-by-task:

```bash
source examples/robotwin/.venv/bin/activate

bash examples/robotwin/eval.sh \
  beat_block_hammer \
  demo_clean \
  pi05_robotwin_pvi_from_base \
  demo_clean_mt3 \
  0 \
  0
```

## 5. Configs Used by This Flow

The relevant configs are in [src/openpi/training/config.py](/data/jhshin/openpi/src/openpi/training/config.py):

- `pi05_robotwin`
- `pi05_robotwin_pvi_from_base`

The PVI flow normally uses:

- `pi05_robotwin_pvi_from_base`

These configs expect:

- `lerobot_root=./datasets`
- RoboTwin camera keys `cam_high`, `cam_left_wrist`, `cam_right_wrist`
- `prompt_from_task=True`
- `adapt_to_pi=False`

The RoboTwin shell wrappers override `repo_id`, `lerobot_root`, and `exp_name` so you do not need to edit the config for every run.

## 6. Notes and Failure Modes

- `examples/robotwin/eval.py` is same-process RoboTwin eval, not remote inference.
- Run `eval.sh` only inside the RoboTwin eval env.
- Run `process_data_pi05.sh`, `generate.sh`, `compute_norm_stats.sh`, and `finetune.sh` in the repo root `openpi` env.
- Conversion uses Hugging Face datasets/parquet internally, so cache disk space matters even though the final dataset is written to `./datasets/<repo_id>`.
- If you keep raw RoboTwin data under `third_party/robotwin/data`, pass it explicitly:

```bash
bash examples/robotwin/process_data_pi05.sh \
  beat_block_hammer \
  demo_clean \
  50 \
  ./third_party/robotwin/data
```

- Checkpoints are saved under:

```text
./checkpoints/<train_config_name>/<model_name>/<checkpoint_id>
```

- Evaluation results are written under:

```text
./eval_result/robotwin/
```
