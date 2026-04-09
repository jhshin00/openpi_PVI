import collections
import dataclasses
import importlib
import importlib.util
import json
import logging
import pathlib
from queue import Empty
from queue import Full
from queue import Queue
import re
import sys
import threading
from typing import Any, Literal, Protocol

import imageio.v2 as imageio
import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

DEFAULT_VIDEO_ROOT = pathlib.Path("/data/jhshin/openpi/video")


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
    replan_steps: int = 8
    async_inference: bool = True
    async_prefetch_steps: int = 6
    async_plan_guard_steps: int = 1
    max_steps: int = 200
    num_episodes: int = 1
    pytorch_device: str | None = None
    video_out_dir: str | None = None
    video_filename: str | None = None
    save_video: Literal["auto", "on", "off"] = "auto"
    video_frame_source: Literal["cropped", "uncropped"] = "cropped"
    debug_action_stats: bool = False
    debug_log_every: int = 1
    chunk_execution: str = "chunk_endpoint"

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
    base_crop_box: tuple[int, int, int, int] | None = (0, 550, 0, 480)
    wrist_crop_box: tuple[int, int, int, int] | None = None
    base_crop_center: tuple[int, int] | None = None
    wrist_crop_center: tuple[int, int] | None = None
    crop_size: tuple[int, int] = (400, 400)
    kp: float = 10.0
    deadband: float = 0.003
    max_joint_velocity: float = 0.35
    max_joint_accel: float = 0.8
    speedj_accel: float = 0.8
    target_smoothing_alpha: float = 0.2
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


def _get_frame(obs: dict[str, Any], *, source: Literal["cropped", "uncropped"]) -> np.ndarray:
    if source == "uncropped":
        image = np.asarray(
            _first_present(
                obs,
                "observation/base_image_uncropped",
                "base_image_uncropped",
                "observation/base_image_raw",
                "base_image_raw",
            )
        )
    else:
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


def _current_state_from_obs(obs: dict[str, Any]) -> np.ndarray:
    return np.asarray(_first_present(obs, "observation/state", "state"), dtype=np.float32)


def _extract_action_window(
    actions: np.ndarray,
    *,
    start: int,
    size: int,
    strict_size: bool,
) -> np.ndarray:
    if actions.ndim != 2:
        raise ValueError(f"Expected a 2D action chunk, got shape {actions.shape}")
    if strict_size and len(actions) < start + size:
        raise ValueError(f"Need {start + size} actions but policy only predicted {len(actions)}")
    return np.asarray(actions[start : start + size], dtype=np.float32)


def _replace_action_plan(
    action_plan: collections.deque[np.ndarray],
    new_plan: np.ndarray,
    *,
    guard_steps: int,
) -> None:
    preserved = []
    preserve_count = min(max(guard_steps, 0), len(action_plan))
    for _ in range(preserve_count):
        preserved.append(np.asarray(action_plan.popleft(), dtype=np.float32))

    action_plan.clear()
    action_plan.extend(preserved)
    action_plan.extend(np.asarray(action, dtype=np.float32) for action in new_plan)


def _drain_queue(q: Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except Empty:
            return


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


def _slugify_path_component(value: str, *, fallback: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not slug:
        return fallback
    if len(slug) <= max_length:
        return slug
    return slug[:max_length].rstrip("_") or fallback


def _policy_video_dirname(policy_config: str) -> str:
    return _slugify_path_component(policy_config.removesuffix("_infer"), fallback="policy")


def _checkpoint_video_dirname(policy_dir: str) -> str:
    checkpoint_path = pathlib.Path(policy_dir)
    if checkpoint_path.name.isdigit() and checkpoint_path.parent.name:
        return _slugify_path_component(checkpoint_path.parent.name, fallback="checkpoint")
    return _slugify_path_component(checkpoint_path.name, fallback="checkpoint")


def _prompt_video_dirname(prompt: str) -> str:
    return _slugify_path_component(prompt, fallback="task")


def _resolve_video_dir(args: Args, *, prompt: str | None) -> pathlib.Path | None:
    if args.save_video == "off":
        return None
    if args.video_out_dir is not None:
        return pathlib.Path(args.video_out_dir)
    prompt_dir = _prompt_video_dirname(prompt) if prompt is not None else "task"
    return DEFAULT_VIDEO_ROOT / _policy_video_dirname(args.policy_config) / _checkpoint_video_dirname(args.policy_dir) / prompt_dir


def _normalize_video_filename(video_filename: str) -> pathlib.Path:
    filename = pathlib.Path(video_filename).name
    if not filename or filename in {".", ".."}:
        raise ValueError("video_filename must be a valid filename")

    path = pathlib.Path(filename)
    if path.suffix == "":
        return path.with_suffix(".mp4")
    if path.suffix.lower() != ".mp4":
        raise ValueError("video_filename must end with .mp4")
    return path


def _resolve_video_path(
    video_dir: pathlib.Path,
    *,
    video_filename: str | None,
    episode_index: int,
    num_episodes: int,
) -> pathlib.Path:
    if video_filename is None:
        return video_dir / f"episode_{episode_index:04d}.mp4"

    filename = _normalize_video_filename(video_filename)
    if num_episodes == 1:
        return video_dir / filename
    return video_dir / f"{filename.stem}_episode_{episode_index:04d}{filename.suffix}"


def _write_video(video_path: pathlib.Path, frames: list[np.ndarray], *, fps: int) -> None:
    if not frames:
        logging.warning("Skipping empty video buffer: %s", video_path)
        return
    imageio.mimwrite(video_path, frames, fps=max(fps, 1))
    logging.info("saved_video=%s num_frames=%d", video_path, len(frames))


def run(args: Args) -> None:
    policy = create_policy(args)
    env = _create_env(args)

    if args.save_video == "off":
        logging.info("video_saving=disabled")
    elif args.video_out_dir is not None:
        logging.info(
            "video_saving=enabled video_dir=%s video_filename=%s video_frame_source=%s",
            pathlib.Path(args.video_out_dir),
            args.video_filename or "<auto>",
            args.video_frame_source,
        )
    else:
        logging.info(
            "video_saving=enabled video_dir_root=%s policy=%s checkpoint=%s prompt=<auto> video_frame_source=%s",
            DEFAULT_VIDEO_ROOT,
            _policy_video_dirname(args.policy_config),
            _checkpoint_video_dirname(args.policy_dir),
            args.video_frame_source,
        )

    if args.replan_steps <= 0:
        raise ValueError("replan_steps must be positive")
    if args.async_prefetch_steps < 0:
        raise ValueError("async_prefetch_steps must be non-negative")
    if args.async_plan_guard_steps < 0:
        raise ValueError("async_plan_guard_steps must be non-negative")

    inference_queue: Queue[tuple[int, dict[str, Any]]] | None = None
    result_queue: Queue[tuple[int, dict[str, Any]]] | None = None
    error_queue: Queue[BaseException] | None = None
    stop_event: threading.Event | None = None
    infer_thread: threading.Thread | None = None

    if args.async_inference:
        inference_queue = Queue(maxsize=1)
        result_queue = Queue(maxsize=1)
        error_queue = Queue(maxsize=1)
        stop_event = threading.Event()

        def inference_worker() -> None:
            assert inference_queue is not None
            assert result_queue is not None
            assert error_queue is not None
            assert stop_event is not None

            while not stop_event.is_set():
                try:
                    request_step, request_obs = inference_queue.get(timeout=0.1)
                except Empty:
                    continue

                try:
                    policy_result = policy.infer(request_obs)
                    _drain_queue(result_queue)
                    result_queue.put_nowait((request_step, policy_result))
                except Exception as exc:
                    logging.exception("Inference worker failed at step=%d", request_step)
                    _drain_queue(error_queue)
                    error_queue.put_nowait(exc)
                    return

        infer_thread = threading.Thread(target=inference_worker, name="ur3-policy-infer", daemon=True)
        infer_thread.start()

    total_successes = 0
    try:
        for episode_index in range(args.num_episodes):
            obs, info = _normalize_reset(env.reset())
            prompt = args.prompt or info.get("prompt") or obs.get("prompt") or args.default_prompt
            if prompt is None:
                raise ValueError(
                    "A prompt is required for UR3 evaluation. Pass --prompt/--default-prompt or return one from env.reset()."
                )
            video_dir = _resolve_video_dir(args, prompt=prompt)
            if video_dir is not None:
                video_dir.mkdir(parents=True, exist_ok=True)
                logging.info(
                    "episode=%d video_dir=%s video_filename=%s prompt=%s",
                    episode_index,
                    video_dir,
                    args.video_filename or "<auto>",
                    prompt,
                )
                video_path = _resolve_video_path(
                    video_dir,
                    video_filename=args.video_filename,
                    episode_index=episode_index,
                    num_episodes=args.num_episodes,
                )
            else:
                video_path = None
            action_plan: collections.deque[np.ndarray] = collections.deque()
            frames = [] if video_dir is not None else None
            pending_request_step: int | None = None
            hold_steps = 0
            step = -1

            success = False
            pending_exception: BaseException | None = None
            try:
                if frames is not None:
                    frames.append(_get_frame(obs, source=args.video_frame_source))

                for step in range(args.max_steps):
                    if args.async_inference:
                        assert inference_queue is not None
                        assert result_queue is not None
                        assert error_queue is not None
                        assert infer_thread is not None

                        try:
                            worker_exc = error_queue.get_nowait()
                        except Empty:
                            worker_exc = None
                        if worker_exc is not None:
                            raise RuntimeError("Inference worker failed.") from worker_exc
                        if not infer_thread.is_alive():
                            raise RuntimeError("Inference worker thread stopped unexpectedly.")

                        try:
                            request_step, policy_result = result_queue.get_nowait()
                        except Empty:
                            policy_result = None
                        else:
                            pending_request_step = None
                            delay_steps = max(step - request_step, 0)
                            current_state = _current_state_from_obs(obs)
                            raw_actions = np.asarray(policy_result["actions"], dtype=np.float32)
                            raw_chunk = _extract_action_window(
                                raw_actions,
                                start=delay_steps,
                                size=args.replan_steps,
                                strict_size=False,
                            )
                            if len(raw_chunk) == 0:
                                logging.warning(
                                    "Dropping stale policy result at step=%d: delay_steps=%d chunk_len=%d",
                                    step,
                                    delay_steps,
                                    len(raw_actions),
                                )
                            else:
                                chunk = _build_action_plan(raw_chunk, current_state, args.chunk_execution)
                                _replace_action_plan(
                                    action_plan,
                                    chunk,
                                    guard_steps=args.async_plan_guard_steps,
                                )

                                if args.debug_action_stats:
                                    joint_delta = chunk[:, :6] - current_state[None, :6]
                                    logging.info(
                                        "policy_chunk step=%d req_step=%d delay_steps=%d infer_ms=%.2f plan_mode=%s "
                                        "first_target=%s last_target=%s first_delta=%s last_delta=%s "
                                        "chunk_max_abs_delta=%.5f chunk_mean_abs_delta=%.5f queue_len=%d",
                                        step,
                                        request_step,
                                        delay_steps,
                                        float(policy_result.get("policy_timing", {}).get("infer_ms", -1.0)),
                                        args.chunk_execution,
                                        np.array2string(chunk[0], precision=4, suppress_small=True),
                                        np.array2string(chunk[-1], precision=4, suppress_small=True),
                                        np.array2string(joint_delta[0], precision=4, suppress_small=True),
                                        np.array2string(joint_delta[-1], precision=4, suppress_small=True),
                                        float(np.max(np.abs(joint_delta))),
                                        float(np.mean(np.abs(joint_delta))),
                                        len(action_plan),
                                    )

                        if pending_request_step is None and len(action_plan) <= args.async_prefetch_steps:
                            policy_obs = _to_policy_observation(obs, prompt)
                            try:
                                inference_queue.put_nowait((step, policy_obs))
                            except Full:
                                pass
                            else:
                                pending_request_step = step

                    else:
                        if not action_plan:
                            policy_obs = _to_policy_observation(obs, prompt)
                            policy_result = policy.infer(policy_obs)
                            raw_actions = np.asarray(policy_result["actions"], dtype=np.float32)
                            raw_chunk = _extract_action_window(
                                raw_actions,
                                start=0,
                                size=args.replan_steps,
                                strict_size=True,
                            )
                            current_state = _current_state_from_obs(obs)
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

                    if action_plan:
                        action = np.asarray(action_plan.popleft(), dtype=np.float32)
                        hold_steps = 0
                    else:
                        # Keep tracking the measured current pose while waiting for a fresh plan.
                        action = _current_state_from_obs(obs).copy()
                        hold_steps += 1
                        if args.debug_action_stats and hold_steps % max(args.debug_log_every, 1) == 0:
                            logging.info(
                                "holding_current_pose step=%d hold_steps=%d pending_request=%s",
                                step,
                                hold_steps,
                                pending_request_step,
                            )

                    obs, reward, done, info = _normalize_step(env.step(action))
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
                        frames.append(_get_frame(obs, source=args.video_frame_source))

                    if done:
                        break
            except BaseException as exc:
                pending_exception = exc
                if isinstance(exc, KeyboardInterrupt):
                    logging.info(
                        "KeyboardInterrupt at episode=%d step=%d. Saving partial video before shutdown.",
                        episode_index,
                        step,
                    )
                raise
            finally:
                if frames is not None and video_path is not None:
                    try:
                        _write_video(video_path, frames, fps=args.hz)
                    except Exception:
                        logging.exception("Failed to save video: %s", video_path)
                        if pending_exception is None:
                            raise

            total_successes += int(success)
            logging.info("episode=%d success=%s", episode_index, success)

        logging.info("success_rate=%.3f", total_successes / max(args.num_episodes, 1))
    finally:
        if stop_event is not None:
            stop_event.set()
        if infer_thread is not None:
            infer_thread.join(timeout=1.0)
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    run(tyro.cli(Args))
