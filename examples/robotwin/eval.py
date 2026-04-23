"""
Evaluate an openpi checkpoint inside the RoboTwin simulator.

This script is intended to run in a RoboTwin-capable simulator environment, not the default openpi
training environment. The RoboTwin repo should already be available at `third_party/robotwin`.

Example:
    python examples/robotwin/eval.py \
        --train-config pi05_robotwin_pvi_from_base \
        --task-name beat_block_hammer \
        --task-config demo_clean \
        --exp-name my_robotwin_run \
        --checkpoint-id 30000
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
import importlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import tyro
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass(frozen=True)
class Args:
    train_config: str
    task_name: str
    task_config: str
    exp_name: str | None = None
    checkpoint_id: str = "30000"
    checkpoint_dir: Path | None = None
    checkpoint_base_dir: Path = REPO_ROOT / "checkpoints_local"
    robotwin_root: Path = Path("third_party/robotwin")
    result_root: Path = Path("./eval_result/robotwin")
    instruction_type: str = "unseen"
    seed: int = 0
    test_num: int = 100
    pi05_step: int = 50
    default_prompt: str | None = None
    pytorch_device: str | None = None
    asset_id_override: str | None = None
    clear_cache_freq_override: int | None = None
    disable_video: bool = False
    disable_torch_compile: bool = False
    log_validation_tracebacks: bool = False


class OpenPIRobotwinPolicy:
    def __init__(
        self,
        train_config: _config.TrainConfig,
        checkpoint_dir: Path,
        *,
        default_prompt: str | None,
        pytorch_device: str | None,
        asset_id_override: str | None,
        pi05_step: int,
    ) -> None:
        self._policy = _policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            default_prompt=default_prompt,
            pytorch_device=pytorch_device,
            asset_id_override=asset_id_override,
        )
        self._instruction: str | None = None
        self.pi05_step = pi05_step

    def reset(self) -> None:
        self._instruction = None

    def _encode_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        head_image = np.transpose(observation["observation"]["head_camera"]["rgb"], (2, 0, 1)).copy()
        right_image = np.transpose(observation["observation"]["right_camera"]["rgb"], (2, 0, 1)).copy()
        left_image = np.transpose(observation["observation"]["left_camera"]["rgb"], (2, 0, 1)).copy()
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)

        return {
            "state": state,
            "images": {
                "cam_high": head_image,
                "cam_left_wrist": left_image,
                "cam_right_wrist": right_image,
            },
            "prompt": self._instruction,
        }

    def run_policy_step(self, task_env: Any, observation: dict[str, Any]) -> None:
        if self._instruction is None:
            self._instruction = task_env.get_instruction()

        policy_observation = self._encode_observation(observation)
        actions = self._policy.infer(policy_observation)["actions"][: self.pi05_step]

        for action in actions:
            task_env.take_action(action)
            if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
                break


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _load_robotwin_modules(robotwin_root: Path):
    _prepend_sys_path(robotwin_root)
    _prepend_sys_path(robotwin_root / "description" / "utils")

    from envs import CONFIGS_PATH
    from envs.utils.create_actor import UnStableError
    from generate_episode_instructions import generate_episode_descriptions

    return CONFIGS_PATH, UnStableError, generate_episode_descriptions


def _instantiate_task_env(task_name: str) -> Any:
    env_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(env_module, task_name)
    return env_class()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _get_embodiment_file(embodiment_type: str, embodiment_types: dict[str, Any]) -> str:
    robot_file = embodiment_types[embodiment_type]["file_path"]
    if robot_file is None:
        raise ValueError(f"No embodiment file configured for {embodiment_type}")
    return robot_file


def _prepare_robotwin_args(args: Args, config_path: Path, configs_path: str) -> dict[str, Any]:
    task_args = _load_yaml(config_path)
    task_args["task_name"] = args.task_name
    task_args["task_config"] = args.task_config

    embodiment_types = _load_yaml(Path(configs_path) / "_embodiment_config.yml")
    camera_config = _load_yaml(Path(configs_path) / "_camera_config.yml")

    head_camera_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = camera_config[head_camera_type]["h"]
    task_args["head_camera_w"] = camera_config[head_camera_type]["w"]

    embodiment = task_args.get("embodiment")
    if len(embodiment) == 1:
        task_args["left_robot_file"] = _get_embodiment_file(embodiment[0], embodiment_types)
        task_args["right_robot_file"] = _get_embodiment_file(embodiment[0], embodiment_types)
        task_args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        task_args["left_robot_file"] = _get_embodiment_file(embodiment[0], embodiment_types)
        task_args["right_robot_file"] = _get_embodiment_file(embodiment[1], embodiment_types)
        task_args["embodiment_dis"] = embodiment[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise ValueError("RoboTwin embodiment must have length 1 or 3.")

    task_args["left_embodiment_config"] = _load_yaml(Path(task_args["left_robot_file"]) / "config.yml")
    task_args["right_embodiment_config"] = _load_yaml(Path(task_args["right_robot_file"]) / "config.yml")
    if args.clear_cache_freq_override is not None:
        task_args["clear_cache_freq"] = args.clear_cache_freq_override

    return task_args


def _infer_checkpoint_dir(args: Args, train_config: _config.TrainConfig, repo_root: Path) -> Path:
    if args.checkpoint_dir is not None:
        return args.checkpoint_dir.resolve()
    if args.exp_name is None:
        raise ValueError("Either checkpoint_dir or exp_name must be provided for RoboTwin evaluation.")
    checkpoint_base_dir = args.checkpoint_base_dir
    checkpoint_dir = checkpoint_base_dir / args.train_config / args.exp_name / str(args.checkpoint_id)
    return checkpoint_dir.resolve()


def _infer_checkpoint_asset_id(checkpoint_dir: Path) -> str | None:
    assets_dir = checkpoint_dir / "assets"
    if not assets_dir.exists():
        return None
    norm_stats_files = sorted(assets_dir.rglob("norm_stats.json"))
    if len(norm_stats_files) == 1:
        return norm_stats_files[0].parent.relative_to(assets_dir).as_posix()
    asset_dirs = sorted(path.name for path in assets_dir.iterdir() if path.is_dir())
    if len(asset_dirs) == 1:
        return asset_dirs[0]
    return None


def _select_instruction(
    generate_episode_descriptions,
    *,
    task_name: str,
    episode_info: dict[str, Any],
    instruction_type: str,
) -> str:
    generated = generate_episode_descriptions(task_name, [episode_info], 100)
    if not generated:
        raise ValueError(f"Failed to generate RoboTwin instructions for task {task_name}")

    candidates = generated[0].get(instruction_type) or generated[0].get("seen") or generated[0].get("unseen")
    if not candidates:
        raise ValueError(f"No instructions available for task {task_name} ({instruction_type=})")
    return str(np.random.choice(candidates))


def _create_video_writer(task_env: Any, video_size: str) -> subprocess.Popen[bytes]:
    assert task_env.eval_video_path is not None
    output_path = Path(task_env.eval_video_path) / f"episode{task_env.test_num}.mp4"
    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            video_size,
            "-framerate",
            "10",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    repo_root = Path.cwd().resolve()
    robotwin_root = args.robotwin_root.resolve()
    train_config = _config.get_config(args.train_config)
    if args.disable_torch_compile and hasattr(train_config.model, "pytorch_compile_mode"):
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, pytorch_compile_mode=None),
        )
    checkpoint_dir = _infer_checkpoint_dir(args, train_config, repo_root)

    os.chdir(robotwin_root)
    configs_path, unstable_error_cls, generate_episode_descriptions = _load_robotwin_modules(robotwin_root)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    asset_id_override = args.asset_id_override or _infer_checkpoint_asset_id(checkpoint_dir)
    task_config_path = Path("task_config") / f"{args.task_config}.yml"
    task_args = _prepare_robotwin_args(args, task_config_path, configs_path)
    task_args["eval_mode"] = True
    if args.disable_video:
        task_args["eval_video_log"] = False

    checkpoint_label = checkpoint_dir.relative_to(repo_root) if checkpoint_dir.is_relative_to(repo_root) else checkpoint_dir.name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_root = (repo_root / args.result_root).resolve()
    result_dir = result_root / args.task_name / args.train_config / args.task_config / str(checkpoint_label).replace("/", "_") / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    if task_args["eval_video_log"]:
        camera_config = _load_yaml(Path(configs_path) / "_camera_config.yml")
        camera_type = task_args["camera"]["head_camera_type"]
        task_args["eval_video_save_dir"] = result_dir
        video_size = f"{camera_config[camera_type]['w']}x{camera_config[camera_type]['h']}"
    else:
        video_size = None

    policy = OpenPIRobotwinPolicy(
        train_config,
        checkpoint_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
        asset_id_override=asset_id_override,
        pi05_step=args.pi05_step,
    )

    task_env = _instantiate_task_env(args.task_name)
    initial_seed = 100000 * (1 + args.seed)
    current_seed = initial_seed
    valid_seed_count = 0
    episode_id = 0
    success_count = 0
    clear_cache_freq = int(task_args["clear_cache_freq"])
    task_env.suc = 0
    task_env.test_num = 0

    while valid_seed_count < args.test_num:
        render_freq = task_args["render_freq"]
        task_args["render_freq"] = 0

        try:
            task_env.setup_demo(now_ep_num=episode_id, seed=current_seed, is_test=True, **task_args)
            episode_info = task_env.play_once()
            task_env.close_env()
        except unstable_error_cls:
            task_env.close_env()
            current_seed += 1
            task_args["render_freq"] = render_freq
            continue
        except Exception:
            task_env.close_env()
            current_seed += 1
            task_args["render_freq"] = render_freq
            if args.log_validation_tracebacks:
                logging.exception("RoboTwin expert rollout failed during seed validation")
            else:
                logging.warning("RoboTwin expert rollout failed during seed validation; skipping seed=%s", current_seed - 1)
            continue

        if not (task_env.plan_success and task_env.check_success()):
            current_seed += 1
            task_args["render_freq"] = render_freq
            continue

        valid_seed_count += 1
        task_args["render_freq"] = render_freq

        task_env.setup_demo(now_ep_num=episode_id, seed=current_seed, is_test=True, **task_args)
        instruction = _select_instruction(
            generate_episode_descriptions,
            task_name=args.task_name,
            episode_info=episode_info["info"],
            instruction_type=args.instruction_type,
        )
        task_env.set_instruction(instruction=instruction)

        ffmpeg = None
        if video_size is not None and task_env.eval_video_path is not None:
            ffmpeg = _create_video_writer(task_env, video_size)
            task_env._set_eval_video_ffmpeg(ffmpeg)

        policy.reset()
        succeeded = False
        while task_env.take_action_cnt < task_env.step_lim:
            observation = task_env.get_obs()
            policy.run_policy_step(task_env, observation)
            if task_env.eval_success:
                succeeded = True
                break

        if task_env.eval_video_path is not None:
            task_env._del_eval_video_ffmpeg()

        if succeeded:
            success_count += 1
            task_env.suc += 1

        task_env.close_env(clear_cache=(valid_seed_count % clear_cache_freq == 0))
        if task_env.render_freq:
            task_env.viewer.close()

        task_env.test_num += 1
        logging.info(
            "task=%s config=%s checkpoint=%s success=%s running_success=%d/%d seed=%d",
            args.task_name,
            args.task_config,
            checkpoint_label,
            succeeded,
            success_count,
            task_env.test_num,
            current_seed,
        )

        episode_id += 1
        current_seed += 1

    summary = {
        "task_name": args.task_name,
        "task_config": args.task_config,
        "train_config": args.train_config,
        "checkpoint_dir": str(checkpoint_dir),
        "asset_id_override": asset_id_override,
        "instruction_type": args.instruction_type,
        "seed": args.seed,
        "test_num": args.test_num,
        "success_count": success_count,
        "success_rate": success_count / args.test_num,
    }
    with (result_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
