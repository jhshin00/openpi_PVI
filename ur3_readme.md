# UR3 Integration Notes

This branch adds a first-class UR3 path to the current OpenPI codebase:

- `src/openpi/policies/ur3_policy.py`
  - UR3-specific observation/action transforms.
- `examples/ur3/convert_ur3_data_to_lerobot.py`
  - Converts raw UR3 hdf5 episodes to LeRobot format.
- `examples/ur3/main.py`
  - Single-process evaluation entrypoint that loads the policy directly instead of going through the websocket server.
- `src/openpi/training/config.py`
  - Adds `LeRobotUR3DataConfig`, `pi05_ur3_pvi`, and `pi05_ur3_pvi_infer`.

## 1. Raw Data Assumption

The converter auto-detects two layouts.

Primary layout: the current `run_env_ik.py` output.

```text
<traj_dir>/data.hdf5
|- data
   |- joint_positions        # (T, 7)
   |- joint_actions          # (T, 7)
   |- base_rgb               # (T, H, W, C)
   |- wrist_rgb              # (T, H, W, C)
   |- joint_velocities       # optional
   |- effort                 # optional
   |- task                   # optional
```

Legacy layout is still supported:

```text
episode_000000.hdf5
|- observation
|  |- qpos
|  |- qvel                  # optional
|  |- effort                # optional
|  |- image
|     |- base_image
|     |- wrist_image
|- action
|- task                     # optional
```

The integrated training path assumes:

- state dim = 7
- action dim = 7
- first 6 action dims are joint targets
- 7th action dim is gripper
- raw UR3 actions are absolute, so training converts only the first 6 dims to delta actions

## 2. Convert To LeRobot

`run_env_ik.py` does not save a task string by default, so the converter first tries to infer one from the directory name above the timestamp folder.
For example:

```text
./datasets/ur3/pick_and_place/pick_up_the_pear/0403_153000/data.hdf5
```

becomes the prompt:

```text
pick up the pear
```

If your directory layout does not encode the task cleanly, fall back to `--default-task` or `--task-map-json`.

Example:

```bash
uv run examples/ur3/convert_ur3_data_to_lerobot.py \
  --raw-dir ./datasets/ur3_raw \
  --repo-id ur3_dataset \
  --root ./datasets \
  --fps 30
```

This creates a local LeRobot dataset at:

```text
./datasets/ur3_dataset
```

The converted dataset stores:

- `state`
- `actions`
- `base_image`
- `wrist_image`
- `velocity` and `effort` when present

Episode tasks are saved as LeRobot task metadata and later reused as prompts with `prompt_from_task=True`.

If you need custom per-trajectory prompts, pass a JSON mapping with `--task-map-json`.
The mapping may use the relative `data.hdf5` path, the filename, or the parent trajectory directory name as the key.

## 3. Norm Stats

Compute UR3 normalization statistics before training:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_ur3_pvi
```

With the default config, this reads from:

```text
./datasets/ur3_dataset
```

and writes stats to:

```text
./assets/pi05_ur3_pvi/ur3_dataset
```

## 4. Training

UR3 `pi0.5` training is wired as a PVI run.

Primary config:

```text
pi05_ur3_pvi
```

Default assumptions in the integrated config:

- base model: `pi05_base`
- action horizon: `10`
- `use_pvi=True`
- `discrete_state_input=False`
- local dataset root: `./datasets`
- dataset repo id: `ur3_dataset`

Example:

```bash
uv run scripts/train_pytorch_PVI.py pi05_ur3_pvi \
  --exp_name pi05_ur3_pvi_run1
```

If the dataset location or repo id differs, override them from CLI:

```bash
uv run scripts/train_pytorch_PVI.py pi05_ur3_pvi \
  --exp_name pi05_ur3_pvi_run1 \
  --data.repo_id my_ur3_dataset \
  --data.lerobot_root /path/to/datasets
```

## 5. Single-Process Eval

The old pattern was:

- terminal 1: websocket policy server
- terminal 2: env client script

The new UR3 path supports a single-process loop in `examples/ur3/main.py`.

It directly:

1. loads the trained checkpoint with `policy_config.create_trained_policy(...)`
2. constructs the UR3 environment in-process
3. runs `policy.infer(...)`
4. feeds actions back into the env in the same process

The built-in runtime supports:

- `robot_mode=direct`
  - one-process evaluation using `gello.robots.ur.URRobot`
- `robot_mode=zmq`
  - connects to an already-running robot server from `experiments/launch_nodes.py`
- `env_factory=...`
  - optional override when you want to provide your own env object

The built-in env expects two RealSense cameras. If you do not pass serials, it uses the first detected camera as `base` and the second as `wrist`.

Example with the built-in direct UR3 runtime:

```bash
uv run examples/ur3/main.py \
  --policy-config pi05_ur3_pvi_infer \
  --policy-dir ./checkpoints/pi05_ur3_pvi/pi05_ur3_pvi_run1/30000 \
  --robot-mode direct \
  --robot-ip 192.168.5.102 \
  --default-prompt "pick up the object" \
  --replan-steps 5 \
  --max-steps 200
```

If you still want to provide a custom env factory:

```bash
uv run examples/ur3/main.py \
  --policy-config pi05_ur3_pvi_infer \
  --policy-dir ./checkpoints/pi05_ur3_pvi/pi05_ur3_pvi_run1/30000 \
  --env-factory your_package.your_env:create_env \
  --env-kwargs-json '{"robot_ip":"192.168.0.10"}' \
  --replan-steps 5 \
  --max-steps 200
```

`env.step(action)` may return either:

- `(obs, reward, done, info)`
- `(obs, reward, terminated, truncated, info)`

## 6. Current Scope

Integrated now:

- UR3 data conversion
- UR3 policy transforms
- UR3 training config for `pi0.5 + PVI`
- direct single-process eval loop with a concrete real-UR3 runtime

Still environment-specific on your side:

- success criteria in `info`
- camera serial assignment / crop tuning for your setup
- any task-specific reset routine beyond the default joint reset


DINO:

CUDA_VISIBLE_DEVICES=0 \
HF_LEROBOT_HOME=/data/jhshin/openpi/datasets \
uv run scripts/train_pytorch_PVI.py pi05_ur3_pvi_dinov2

HPR:

CUDA_VISIBLE_DEVICES=1 \
HF_LEROBOT_HOME=/data/jhshin/openpi/datasets \
uv run scripts/train_pytorch_PVI.py pi05_ur3_pvi_hpr