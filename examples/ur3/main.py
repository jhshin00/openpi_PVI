import collections
import dataclasses
import importlib
import importlib.util
import json
import logging
import pathlib
import sys
from typing import Any, Protocol

import imageio.v2 as imageio
import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class UR3Env(Protocol):
    def reset(self) -> Any:
        raise NotImplementedError

    def step(self, action: np.ndarray) -> Any:
        raise NotImplementedError


@dataclasses.dataclass
class Args:
    policy_config: str = "pi05_ur3_pvi_dinov2_infer"
    policy_dir: str = tyro.MISSING
    default_prompt: str | None = None
    prompt: str | None = None
    replan_steps: int = 5
    max_steps: int = 200
    num_episodes: int = 1
    pytorch_device: str | None = None
    video_out_dir: str | None = None
    debug_action_stats: bool = False
    debug_log_every: int = 1
    chunk_execution: str = "per_step"

    # Optional override if you want to provide a custom env factory instead of the built-in UR3 env.
    env_factory: str | None = None
    env_kwargs_json: str = "{}"

    # Built-in UR3 env parameters.
    gello_root: str = str(pathlib.Path(__file__).resolve().parents[2] / "_external" / "gello_software")
    robot_mode: str = "direct"
    robot_ip: str = "192.168.5.102"
    hostname: str = "127.0.0.1"
    robot_port: int = 6001
    hz: int = 30
    image_size: int = 224
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    base_camera_serial: str | None = None
    wrist_camera_serial: str | None = None
    base_crop_box: tuple[int, int, int, int] | None = (0, 500, 0, 480)
    wrist_crop_box: tuple[int, int, int, int] | None = None
    base_crop_center: tuple[int, int] | None = None
    wrist_crop_center: tuple[int, int] | None = None
    crop_size: tuple[int, int] = (400, 400)
    kp: float = 8.0
    deadband: float = 0.003
    max_joint_velocity: float = 1.0
    max_joint_accel: float = 1.0
    speedj_accel: float = 1.0
    target_smoothing_alpha: float = 1.0
    reset_joints_deg: tuple[float, ...] | None = (0.0, -90.0, -90.0, -90.0, 90.0, 90.0)
    reset_gripper: float = 0.0
    reset_steps: int = 120
    reset_max_delta: float = 0.05
    camera_warmup_sec: float = 5.0
    mock: bool = False


def _load_factory(factory_spec: str):
    module_name, separator, attr_name = factory_spec.partition(":")
    if not separator:
        raise ValueError("env_factory must be in the form 'package.module:create_env'")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _load_builtin_factory():
    module_path = pathlib.Path(__file__).with_name("real_env.py")
    spec = importlib.util.spec_from_file_location("openpi_examples_ur3_real_env", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load built-in UR3 env module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.create_env


def _normalize_reset(reset_result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, info = reset_result
        return obs, info
    return reset_result, {}


def _normalize_step(step_result: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    if isinstance(step_result, tuple) and len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, bool(terminated or truncated), info
    if isinstance(step_result, tuple) and len(step_result) == 4:
        obs, reward, done, info = step_result
        return obs, reward, bool(done), info
    raise ValueError("env.step(action) must return a 4-tuple or 5-tuple")


def _first_present(data: dict[str, Any], *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"None of the keys are present: {keys}")


def _to_policy_observation(obs: dict[str, Any], prompt: str | None) -> dict[str, Any]:
    prompt_value = prompt if prompt is not None else obs.get("prompt")
    policy_obs = {
        "observation/state": _first_present(obs, "observation/state", "state"),
        "observation/base_image": _first_present(obs, "observation/base_image", "base_image", "image"),
        "observation/wrist_image": _first_present(obs, "observation/wrist_image", "wrist_image"),
    }
    if prompt_value is not None:
        policy_obs["prompt"] = prompt_value
    return policy_obs


def _get_frame(obs: dict[str, Any]) -> np.ndarray:
    image = np.asarray(_first_present(obs, "observation/base_image", "base_image", "image"))
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def _build_action_plan(chunk: np.ndarray, current_state: np.ndarray, mode: str) -> np.ndarray:
    if mode == "per_step":
        return chunk
    if mode == "chunk_endpoint":
        if len(chunk) == 0:
            return chunk

        plan = chunk.copy()
        endpoint = chunk[-1]
        weights = np.linspace(1.0 / len(chunk), 1.0, len(chunk), dtype=np.float32)
        plan[:, :6] = current_state[None, :6] + (endpoint[None, :6] - current_state[None, :6]) * weights[:, None]
        # Keep the model's original gripper schedule.
        plan[:, 6:] = chunk[:, 6:]
        return plan

    raise ValueError(f"Unsupported chunk_execution: {mode}")


def create_policy(args: Args) -> _policy.Policy:
    config = _config.get_config(args.policy_config)
    return _policy_config.create_trained_policy(
        config,
        args.policy_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
    )


def _create_builtin_env(args: Args) -> UR3Env:
    factory = _load_builtin_factory()
    return factory(
        gello_root=args.gello_root,
        robot_mode=args.robot_mode,
        robot_ip=args.robot_ip,
        hostname=args.hostname,
        robot_port=args.robot_port,
        hz=args.hz,
        prompt=args.prompt or args.default_prompt,
        image_size=args.image_size,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        base_camera_serial=args.base_camera_serial,
        wrist_camera_serial=args.wrist_camera_serial,
        base_crop_box=args.base_crop_box,
        wrist_crop_box=args.wrist_crop_box,
        base_crop_center=args.base_crop_center,
        wrist_crop_center=args.wrist_crop_center,
        crop_size=args.crop_size,
        kp=args.kp,
        deadband=args.deadband,
        max_joint_velocity=args.max_joint_velocity,
        max_joint_accel=args.max_joint_accel,
        speedj_accel=args.speedj_accel,
        target_smoothing_alpha=args.target_smoothing_alpha,
        reset_joints_deg=args.reset_joints_deg,
        reset_gripper=args.reset_gripper,
        reset_steps=args.reset_steps,
        reset_max_delta=args.reset_max_delta,
        camera_warmup_sec=args.camera_warmup_sec,
        mock=args.mock,
    )


def _create_env(args: Args) -> UR3Env:
    if args.env_factory is None:
        return _create_builtin_env(args)

    env_kwargs = json.loads(args.env_kwargs_json)
    env_factory = _load_factory(args.env_factory)
    return env_factory(**env_kwargs)


def run(args: Args) -> None:
    policy = create_policy(args)
    env = _create_env(args)

    video_dir = pathlib.Path(args.video_out_dir) if args.video_out_dir is not None else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    total_successes = 0
    try:
        for episode_index in range(args.num_episodes):
            obs, info = _normalize_reset(env.reset())
            prompt = args.prompt or info.get("prompt") or obs.get("prompt") or args.default_prompt
            if prompt is None:
                raise ValueError(
                    "A prompt is required for UR3 evaluation. Pass --prompt/--default-prompt or return one from env.reset()."
                )
            action_plan: collections.deque[np.ndarray] = collections.deque()
            frames = [] if video_dir is not None else None

            if frames is not None:
                frames.append(_get_frame(obs))

            success = False
            for step in range(args.max_steps):
                if not action_plan:
                    policy_obs = _to_policy_observation(obs, prompt)
                    policy_result = policy.infer(policy_obs)
                    action_chunk = policy_result["actions"]
                    if len(action_chunk) < args.replan_steps:
                        raise ValueError(
                            f"replan_steps={args.replan_steps} but policy only predicted {len(action_chunk)} actions"
                        )
                    raw_chunk = np.asarray(action_chunk[: args.replan_steps], dtype=np.float32)
                    current_state = np.asarray(policy_obs["observation/state"], dtype=np.float32)
                    chunk = _build_action_plan(raw_chunk, current_state, args.chunk_execution)

                    if args.debug_action_stats:
                        joint_delta = chunk[:, :6] - current_state[None, :6]
                        logging.info(
                            "policy_chunk step=%d infer_ms=%.2f plan_mode=%s "
                            "first_target=%s last_target=%s first_delta=%s last_delta=%s "
                            "chunk_max_abs_delta=%.5f chunk_mean_abs_delta=%.5f",
                            step,
                            float(policy_result.get("policy_timing", {}).get("infer_ms", -1.0)),
                            args.chunk_execution,
                            np.array2string(chunk[0], precision=4, suppress_small=True),
                            np.array2string(chunk[-1], precision=4, suppress_small=True),
                            np.array2string(joint_delta[0], precision=4, suppress_small=True),
                            np.array2string(joint_delta[-1], precision=4, suppress_small=True),
                            float(np.max(np.abs(joint_delta))),
                            float(np.mean(np.abs(joint_delta))),
                        )

                    action_plan.extend(chunk)

                obs, reward, done, info = _normalize_step(env.step(np.asarray(action_plan.popleft())))
                success = bool(info.get("success", done))

                logging.info(
                    "episode=%d step=%d reward=%s done=%s success=%s",
                    episode_index,
                    step,
                    reward,
                    done,
                    success,
                )

                if args.debug_action_stats and step % max(args.debug_log_every, 1) == 0:
                    current = np.asarray(info.get("current_joint_positions", []), dtype=np.float32)
                    target = np.asarray(info.get("target_action", []), dtype=np.float32)
                    err = np.asarray(info.get("joint_error", []), dtype=np.float32)
                    qd = np.asarray(info.get("joint_velocity_cmd", []), dtype=np.float32)
                    if current.size and target.size and err.size and qd.size:
                        logging.info(
                            "control step=%d current=%s target=%s err=%s qd=%s "
                            "max_abs_err=%.5f max_abs_qd=%.5f",
                            step,
                            np.array2string(current, precision=4, suppress_small=True),
                            np.array2string(target, precision=4, suppress_small=True),
                            np.array2string(err, precision=4, suppress_small=True),
                            np.array2string(qd, precision=4, suppress_small=True),
                            float(np.max(np.abs(err[:6]))),
                            float(np.max(np.abs(qd[:6]))),
                        )

                if frames is not None:
                    frames.append(_get_frame(obs))

                if done:
                    break

            total_successes += int(success)
            if frames is not None:
                imageio.mimwrite(video_dir / f"episode_{episode_index:04d}.mp4", frames, fps=max(args.hz, 1))

            logging.info("episode=%d success=%s", episode_index, success)

        logging.info("success_rate=%.3f", total_successes / max(args.num_episodes, 1))
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    run(tyro.cli(Args))
