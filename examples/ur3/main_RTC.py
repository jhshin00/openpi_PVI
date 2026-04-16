from __future__ import annotations

import collections
import dataclasses
import importlib.util
import logging
import math
import pathlib
from queue import Empty
from queue import Queue
import sys
import threading
import time
from typing import Any
from typing import Literal

import jax
import numpy as np
import torch
import tyro

from openpi.models import model as _model


def _load_base_module():
    module_path = pathlib.Path(__file__).with_name("main_modify.py")
    spec = importlib.util.spec_from_file_location("openpi_examples_ur3_main_modify", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base UR3 module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_base_module()

UR3Env = _base.UR3Env
EpisodeResult = _base.EpisodeResult
TerminalUI = _base.TerminalUI
DEFAULT_VIDEO_ROOT = _base.DEFAULT_VIDEO_ROOT
create_policy = _base.create_policy
_build_action_plan = _base._build_action_plan
_checkpoint_video_dirname = _base._checkpoint_video_dirname
_create_env = _base._create_env
_current_state_from_obs = _base._current_state_from_obs
_drain_queue = _base._drain_queue
_extract_action_window = _base._extract_action_window
_first_present = _base._first_present
_get_frame = _base._get_frame
_normalize_reset = _base._normalize_reset
_normalize_step = _base._normalize_step
_policy_video_dirname = _base._policy_video_dirname
_prompt_video_dirname = _base._prompt_video_dirname
_replace_action_plan = _base._replace_action_plan
_resolve_video_path = _base._resolve_video_path
_set_env_prompt = _base._set_env_prompt
_to_policy_observation = _base._to_policy_observation


@dataclasses.dataclass
class Args(_base.Args):
    rtc_inference_delay_steps: int = 2
    rtc_dynamic_inference_delay: bool = True
    rtc_delay_history: int = 8
    rtc_prefix_attention_schedule: Literal["linear", "exp", "ones", "zeros"] = "exp"
    rtc_max_guidance_weight: float = 5.0


@dataclasses.dataclass(frozen=True)
class _RTCChunkResult:
    model_actions: np.ndarray
    actions: np.ndarray
    infer_ms: float


@dataclasses.dataclass(frozen=True)
class _RTCInferenceRequest:
    episode_index: int
    iteration_index: int
    request_step: int
    inference_delay_steps: int
    policy_obs: dict[str, Any]
    prev_model_actions: np.ndarray


@dataclasses.dataclass(frozen=True)
class _RTCInferenceResult:
    episode_index: int
    iteration_index: int
    request_step: int
    inference_delay_steps: int
    chunk: _RTCChunkResult


class _RTCDelayTracker:
    def __init__(self, *, enabled: bool, bootstrap_steps: int, history_size: int):
        self._enabled = enabled
        self._bootstrap_steps = max(int(bootstrap_steps), 0)
        self._recent_observed_steps: collections.deque[int] = collections.deque(maxlen=max(int(history_size), 1))

    def estimate(self, *, execute_horizon: int) -> tuple[int, int]:
        raw_delay_steps = self._bootstrap_steps
        if self._enabled and self._recent_observed_steps:
            raw_delay_steps = max(raw_delay_steps, max(self._recent_observed_steps))
        clamped_delay_steps = min(raw_delay_steps, max(int(execute_horizon), 0))
        return clamped_delay_steps, raw_delay_steps

    def observe(self, *, infer_ms: float, actual_delay: int, hz: int) -> tuple[int, int]:
        control_period_ms = max(1000.0 / max(hz, 1), 1e-6)
        timing_delay_steps = max(int(math.ceil(infer_ms / control_period_ms)), 0)
        observed_delay_steps = max(actual_delay, timing_delay_steps)
        if self._enabled:
            self._recent_observed_steps.append(observed_delay_steps)
        return timing_delay_steps, observed_delay_steps


class _RTCPolicyAdapter:
    def __init__(self, policy: Any):
        self._policy = policy
        self._supports_policy_internals = all(
            hasattr(policy, attr)
            for attr in ("_model", "_input_transform", "_output_transform", "_sample_kwargs")
        )
        self._supports_rtc = bool(
            self._supports_policy_internals
            and getattr(policy, "_is_pytorch_model", False)
            and hasattr(policy._model, "realtime_action")
        )
        if self._supports_policy_internals:
            self._action_horizon = int(policy._model.action_horizon)
        else:
            self._action_horizon = -1

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    @property
    def supports_rtc(self) -> bool:
        return self._supports_rtc

    def infer_initial(self, obs: dict[str, Any]) -> _RTCChunkResult:
        if not self._supports_policy_internals:
            start_time = time.monotonic()
            outputs = self._policy.infer(obs)
            infer_ms = (time.monotonic() - start_time) * 1000.0
            actions = np.asarray(outputs["actions"], dtype=np.float32)
            return _RTCChunkResult(
                model_actions=actions.copy(),
                actions=actions,
                infer_ms=infer_ms,
            )

        outputs_np, infer_ms = self._infer_model_actions(obs, realtime_kwargs=None)
        return _RTCChunkResult(
            model_actions=np.asarray(outputs_np["actions"], dtype=np.float32),
            actions=np.asarray(self._policy._output_transform(outputs_np)["actions"], dtype=np.float32),
            infer_ms=infer_ms,
        )

    def infer_realtime(
        self,
        obs: dict[str, Any],
        prev_model_actions: np.ndarray,
        *,
        inference_delay: int,
        execute_horizon: int,
        prefix_attention_schedule: Literal["linear", "exp", "ones", "zeros"],
        max_guidance_weight: float,
    ) -> _RTCChunkResult:
        if not self._supports_rtc:
            raise RuntimeError("The loaded policy does not expose the PyTorch RTC sampler.")

        prefix_attention_horizon = self._action_horizon - execute_horizon
        if prefix_attention_horizon < 0:
            raise ValueError(
                f"execute_horizon={execute_horizon} exceeds model action horizon={self._action_horizon}"
            )

        outputs_np, infer_ms = self._infer_model_actions(
            obs,
            realtime_kwargs={
                "prev_action_chunk": np.asarray(prev_model_actions, dtype=np.float32),
                "inference_delay": inference_delay,
                "prefix_attention_horizon": prefix_attention_horizon,
                "prefix_attention_schedule": prefix_attention_schedule,
                "max_guidance_weight": max_guidance_weight,
            },
        )
        return _RTCChunkResult(
            model_actions=np.asarray(outputs_np["actions"], dtype=np.float32),
            actions=np.asarray(self._policy._output_transform(outputs_np)["actions"], dtype=np.float32),
            infer_ms=infer_ms,
        )

    def _infer_model_actions(
        self,
        obs: dict[str, Any],
        *,
        realtime_kwargs: dict[str, Any] | None,
    ) -> tuple[dict[str, np.ndarray], float]:
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._policy._input_transform(inputs)
        if not self._policy._is_pytorch_model:
            raise NotImplementedError("RTC integration in this example currently supports PyTorch UR3 checkpoints only.")

        batched_inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(self._policy._pytorch_device)[None, ...],
            inputs,
        )
        observation = _model.Observation.from_dict(batched_inputs)
        sample_kwargs = dict(self._policy._sample_kwargs)
        start_time = time.monotonic()
        if realtime_kwargs is None:
            model_actions = self._policy._model.sample_actions(
                self._policy._pytorch_device,
                observation,
                **sample_kwargs,
            )
        else:
            prev_action_chunk = torch.from_numpy(realtime_kwargs.pop("prev_action_chunk")).to(self._policy._pytorch_device)
            model_actions = self._policy._model.realtime_action(
                self._policy._pytorch_device,
                observation,
                prev_action_chunk,
                **realtime_kwargs,
                **sample_kwargs,
            )
        infer_ms = (time.monotonic() - start_time) * 1000.0
        outputs = {
            "state": batched_inputs["state"],
            "actions": model_actions,
        }
        return (
            jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs),
            infer_ms,
        )


def _validate_args(args: Args) -> None:
    _base._validate_args(args)
    if not args.async_inference:
        raise ValueError("RTC mode requires --async-inference true.")
    if args.rtc_inference_delay_steps < 0:
        raise ValueError("rtc_inference_delay_steps must be non-negative")
    if args.rtc_delay_history <= 0:
        raise ValueError("rtc_delay_history must be positive")
    if args.rtc_max_guidance_weight <= 0:
        raise ValueError("rtc_max_guidance_weight must be positive")


def _validate_rtc_policy(args: Args, adapter: _RTCPolicyAdapter) -> None:
    if not adapter.supports_rtc:
        raise ValueError(
            "The loaded policy does not support RTC. Use a PyTorch pi0/pi0.5 checkpoint with the new realtime sampler."
        )
    if args.replan_steps > adapter.action_horizon:
        raise ValueError(
            f"replan_steps={args.replan_steps} exceeds policy action_horizon={adapter.action_horizon}"
        )
    if args.rtc_inference_delay_steps > args.replan_steps:
        raise ValueError(
            f"RTC requires replan_steps >= rtc_inference_delay_steps, got {args.replan_steps} < {args.rtc_inference_delay_steps}"
        )


def _shift_chunk(chunk: np.ndarray, shift: int) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2:
        raise ValueError(f"Expected a 2D action chunk, got shape {chunk.shape}")
    if len(chunk) == 0 or shift <= 0:
        return chunk.copy()
    if shift >= len(chunk):
        return np.repeat(chunk[-1:], len(chunk), axis=0)
    tail = np.repeat(chunk[-1:], shift, axis=0)
    return np.concatenate([chunk[shift:], tail], axis=0)


def _build_window_plan(window_actions: np.ndarray, current_state: np.ndarray, mode: str) -> np.ndarray:
    window_actions = np.asarray(window_actions, dtype=np.float32)
    if len(window_actions) == 0:
        return window_actions
    return _base._build_action_plan(window_actions, current_state, mode)


def _replace_remaining_plan(action_plan: collections.deque[np.ndarray], new_plan: np.ndarray) -> None:
    action_plan.clear()
    action_plan.extend(np.asarray(action, dtype=np.float32) for action in new_plan)


def _start_inference_worker(
    args: Args,
    adapter: _RTCPolicyAdapter,
    request_queue: Queue[_RTCInferenceRequest],
    result_queue: Queue[_RTCInferenceResult],
    error_queue: Queue[BaseException],
    stop_event: threading.Event,
    *,
    name: str,
) -> threading.Thread:
    def inference_worker() -> None:
        while not stop_event.is_set():
            try:
                request = request_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                chunk = adapter.infer_realtime(
                    request.policy_obs,
                    request.prev_model_actions,
                    inference_delay=request.inference_delay_steps,
                    execute_horizon=args.replan_steps,
                    prefix_attention_schedule=args.rtc_prefix_attention_schedule,
                    max_guidance_weight=args.rtc_max_guidance_weight,
                )
                _base._drain_queue(result_queue)
                result_queue.put_nowait(
                    _RTCInferenceResult(
                        episode_index=request.episode_index,
                        iteration_index=request.iteration_index,
                        request_step=request.request_step,
                        inference_delay_steps=request.inference_delay_steps,
                        chunk=chunk,
                    )
                )
            except Exception as exc:
                logging.exception(
                    "RTC inference worker failed at episode=%d iteration=%d step=%d",
                    request.episode_index,
                    request.iteration_index,
                    request.request_step,
                )
                _base._drain_queue(error_queue)
                error_queue.put_nowait(exc)
                return

    thread = threading.Thread(target=inference_worker, name=name, daemon=True)
    thread.start()
    return thread


class _RTCPlanner:
    def __init__(
        self,
        args: Args,
        adapter: _RTCPolicyAdapter,
        *,
        prompt: str,
        episode_index: int,
        request_queue: Queue[_RTCInferenceRequest],
        result_queue: Queue[_RTCInferenceResult],
        error_queue: Queue[BaseException],
        infer_thread: threading.Thread,
    ):
        self._args = args
        self._adapter = adapter
        self._prompt = prompt
        self._episode_index = episode_index
        self._request_queue = request_queue
        self._result_queue = result_queue
        self._error_queue = error_queue
        self._infer_thread = infer_thread

        self._action_plan: collections.deque[np.ndarray] = collections.deque()
        self._current_model_chunk: np.ndarray | None = None
        self._current_env_chunk: np.ndarray | None = None
        self._next_model_chunk: np.ndarray | None = None
        self._next_env_chunk: np.ndarray | None = None
        self._iteration_index = 0
        self._iteration_progress = 0
        self._pending_iteration_index: int | None = None
        self._last_request_step: int | None = None
        self._delay_tracker = _RTCDelayTracker(
            enabled=args.rtc_dynamic_inference_delay,
            bootstrap_steps=args.rtc_inference_delay_steps,
            history_size=args.rtc_delay_history,
        )
        self._last_clamped_delay_warning: int | None = None
        self._last_observed_delay_warning: int | None = None

    def bootstrap(self, obs: dict[str, Any], *, step: int) -> None:
        policy_obs = _base._to_policy_observation(obs, self._prompt)
        chunk = self._adapter.infer_initial(policy_obs)
        self._current_model_chunk = np.asarray(chunk.model_actions, dtype=np.float32)
        self._current_env_chunk = np.asarray(chunk.actions, dtype=np.float32)
        logging.info(
            "rtc_bootstrap episode=%d infer_ms=%.2f action_horizon=%d execute_horizon=%d initial_delay=%d dynamic_delay=%s",
            self._episode_index,
            chunk.infer_ms,
            self._adapter.action_horizon,
            self._args.replan_steps,
            self._args.rtc_inference_delay_steps,
            self._args.rtc_dynamic_inference_delay,
        )
        self._prepare_iteration(obs, step=step, fallback_used=False)

    def maybe_refresh(self, obs: dict[str, Any], *, step: int) -> None:
        self._ensure_worker()

        while True:
            try:
                result = self._result_queue.get_nowait()
            except Empty:
                break

            if result.episode_index != self._episode_index:
                continue
            if result.iteration_index != self._iteration_index:
                logging.info(
                    "Dropping stale RTC result at step=%d: iteration=%d current_iteration=%d",
                    step,
                    result.iteration_index,
                    self._iteration_index,
                )
                continue

            self._pending_iteration_index = None
            self._last_request_step = None
            actual_delay = max(step - result.request_step, 0)
            remaining_start = min(actual_delay, self._args.replan_steps)
            self._next_model_chunk = _shift_chunk(result.chunk.model_actions, self._args.replan_steps)
            self._next_env_chunk = _shift_chunk(result.chunk.actions, self._args.replan_steps)

            if remaining_start < self._args.replan_steps:
                current_state = _base._current_state_from_obs(obs)
                remaining_plan = _build_window_plan(
                    result.chunk.actions[remaining_start : self._args.replan_steps],
                    current_state,
                    self._args.chunk_execution,
                )
                _replace_remaining_plan(self._action_plan, remaining_plan)

            timing_delay_steps, observed_delay_steps = self._delay_tracker.observe(
                infer_ms=result.chunk.infer_ms,
                actual_delay=actual_delay,
                hz=self._args.hz,
            )
            next_delay_steps, raw_next_delay_steps = self._delay_tracker.estimate(execute_horizon=self._args.replan_steps)
            if observed_delay_steps > self._args.replan_steps and observed_delay_steps != self._last_observed_delay_warning:
                logging.warning(
                    "rtc_delay_exceeds_horizon episode=%d iteration=%d step=%d observed_delay=%d execute_horizon=%d; "
                    "consider increasing replan_steps or reducing inference latency.",
                    self._episode_index,
                    self._iteration_index,
                    step,
                    observed_delay_steps,
                    self._args.replan_steps,
                )
                self._last_observed_delay_warning = observed_delay_steps
            logging.info(
                "rtc_update episode=%d iteration=%d step=%d req_step=%d request_delay=%d actual_delay=%d timing_delay=%d "
                "observed_delay=%d next_delay_estimate=%d raw_next_delay_estimate=%d infer_ms=%.2f schedule=%s queue_len=%d",
                self._episode_index,
                self._iteration_index,
                step,
                result.request_step,
                result.inference_delay_steps,
                actual_delay,
                timing_delay_steps,
                observed_delay_steps,
                next_delay_steps,
                raw_next_delay_steps,
                result.chunk.infer_ms,
                self._args.rtc_prefix_attention_schedule,
                len(self._action_plan),
            )

    def next_action(self, obs: dict[str, Any], *, step: int) -> tuple[np.ndarray, int]:
        self.maybe_refresh(obs, step=step)

        if self._action_plan:
            return np.asarray(self._action_plan.popleft(), dtype=np.float32), 0

        hold_steps = 1
        action = _base._current_state_from_obs(obs).copy()
        if self._args.debug_action_stats and step % max(self._args.debug_log_every, 1) == 0:
            logging.info(
                "rtc_holding_pose episode=%d iteration=%d step=%d hold_steps=%d pending_iteration=%s",
                self._episode_index,
                self._iteration_index,
                step,
                hold_steps,
                self._pending_iteration_index,
            )
        return action, hold_steps

    def after_step(self, obs: dict[str, Any], *, step: int) -> None:
        self._iteration_progress += 1
        if self._iteration_progress < self._args.replan_steps:
            return

        fallback_used = self._next_model_chunk is None or self._next_env_chunk is None
        if fallback_used:
            assert self._current_model_chunk is not None
            assert self._current_env_chunk is not None
            self._current_model_chunk = _shift_chunk(self._current_model_chunk, self._args.replan_steps)
            self._current_env_chunk = _shift_chunk(self._current_env_chunk, self._args.replan_steps)
            logging.warning(
                "rtc_fallback episode=%d iteration=%d step=%d: next chunk was not ready; reusing shifted current chunk.",
                self._episode_index,
                self._iteration_index,
                step,
            )
        else:
            self._current_model_chunk = self._next_model_chunk
            self._current_env_chunk = self._next_env_chunk

        self._next_model_chunk = None
        self._next_env_chunk = None
        self._iteration_index += 1
        self._prepare_iteration(obs, step=step + 1, fallback_used=fallback_used)

    def _prepare_iteration(self, obs: dict[str, Any], *, step: int, fallback_used: bool) -> None:
        assert self._current_env_chunk is not None
        current_state = _base._current_state_from_obs(obs)
        window = self._current_env_chunk[: self._args.replan_steps]
        plan = _build_window_plan(window, current_state, self._args.chunk_execution)
        self._iteration_progress = 0
        self._action_plan.clear()
        self._action_plan.extend(np.asarray(action, dtype=np.float32) for action in plan)
        self._dispatch_request(obs, step=step)

        if self._args.debug_action_stats and len(plan) > 0:
            joint_delta = plan[:, :6] - current_state[None, :6]
            logging.info(
                "rtc_iteration episode=%d iteration=%d step=%d fallback=%s plan_mode=%s "
                "first_target=%s last_target=%s first_delta=%s last_delta=%s "
                "chunk_max_abs_delta=%.5f chunk_mean_abs_delta=%.5f",
                self._episode_index,
                self._iteration_index,
                step,
                fallback_used,
                self._args.chunk_execution,
                np.array2string(plan[0], precision=4, suppress_small=True),
                np.array2string(plan[-1], precision=4, suppress_small=True),
                np.array2string(joint_delta[0], precision=4, suppress_small=True),
                np.array2string(joint_delta[-1], precision=4, suppress_small=True),
                float(np.max(np.abs(joint_delta))),
                float(np.mean(np.abs(joint_delta))),
            )

    def _dispatch_request(self, obs: dict[str, Any], *, step: int) -> None:
        assert self._current_model_chunk is not None
        inference_delay_steps, raw_inference_delay_steps = self._delay_tracker.estimate(
            execute_horizon=self._args.replan_steps
        )
        if (
            raw_inference_delay_steps > inference_delay_steps
            and raw_inference_delay_steps != self._last_clamped_delay_warning
        ):
            logging.warning(
                "rtc_delay_clamped episode=%d iteration=%d step=%d raw_delay=%d execute_horizon=%d; "
                "conditioning will use the clamped value.",
                self._episode_index,
                self._iteration_index,
                step,
                raw_inference_delay_steps,
                self._args.replan_steps,
            )
            self._last_clamped_delay_warning = raw_inference_delay_steps
        policy_obs = _base._to_policy_observation(obs, self._prompt)
        request = _RTCInferenceRequest(
            episode_index=self._episode_index,
            iteration_index=self._iteration_index,
            request_step=step,
            inference_delay_steps=inference_delay_steps,
            policy_obs=policy_obs,
            prev_model_actions=self._current_model_chunk,
        )
        _base._drain_queue(self._request_queue)
        self._request_queue.put_nowait(request)
        self._pending_iteration_index = self._iteration_index
        self._last_request_step = step

    def _ensure_worker(self) -> None:
        try:
            worker_exc = self._error_queue.get_nowait()
        except Empty:
            worker_exc = None

        if worker_exc is not None:
            raise RuntimeError("RTC inference worker failed.") from worker_exc
        if not self._infer_thread.is_alive():
            raise RuntimeError("RTC inference worker thread stopped unexpectedly.")


def _run_rtc_episode(
    args: Args,
    *,
    env: UR3Env,
    adapter: _RTCPolicyAdapter,
    initial_obs: dict[str, Any],
    prompt: str,
    episode_index: int,
    request_queue: Queue[_RTCInferenceRequest],
    result_queue: Queue[_RTCInferenceResult],
    error_queue: Queue[BaseException],
    infer_thread: threading.Thread,
    manual_command_fn,
    video_path: pathlib.Path | None,
) -> EpisodeResult:
    _base._set_env_prompt(env, prompt)
    _base._drain_queue(request_queue)
    _base._drain_queue(result_queue)

    frames = [_base._get_frame(initial_obs, source=args.video_frame_source)] if video_path is not None else None
    obs = initial_obs
    success = False
    executed_steps = 0
    stop_reason: Literal["stop", "reset", "quit", "done", "max_steps"] = "max_steps"
    pending_exception: BaseException | None = None

    planner = _RTCPlanner(
        args,
        adapter,
        prompt=prompt,
        episode_index=episode_index,
        request_queue=request_queue,
        result_queue=result_queue,
        error_queue=error_queue,
        infer_thread=infer_thread,
    )
    planner.bootstrap(obs, step=0)

    try:
        for step in range(args.max_steps):
            manual_command = manual_command_fn() if manual_command_fn is not None else None
            if manual_command is not None:
                stop_reason = manual_command
                logging.info("episode=%d manual_command=%s step=%d", episode_index, manual_command, step)
                break

            action, _ = planner.next_action(obs, step=step)
            obs, reward, done, info = _base._normalize_step(env.step(action))
            executed_steps = step + 1
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
                frames.append(_base._get_frame(obs, source=args.video_frame_source))

            if done:
                stop_reason = "done"
                break

            planner.after_step(obs, step=step)
    except BaseException as exc:
        pending_exception = exc
        if isinstance(exc, KeyboardInterrupt):
            logging.info(
                "KeyboardInterrupt at episode=%d. Saving partial video before shutdown.",
                episode_index,
            )
        raise
    finally:
        if frames is not None and video_path is not None:
            try:
                _base._write_video(video_path, frames, fps=args.hz)
            except Exception:
                logging.exception("Failed to save video: %s", video_path)
                if pending_exception is None:
                    raise

    return EpisodeResult(
        final_obs=obs,
        reason=stop_reason,
        success=success,
        steps=executed_steps,
        video_path=video_path,
    )


def _run_single_episode_mode(args: Args) -> None:
    _validate_args(args)
    policy = create_policy(args)
    adapter = _RTCPolicyAdapter(policy)
    _validate_rtc_policy(args, adapter)
    env = _base._create_env(args)

    _base._log_video_config(args)
    logging.info(
        "rtc_mode=enabled initial_delay=%d dynamic_delay=%s delay_history=%d execute_horizon=%d action_horizon=%d "
        "schedule=%s max_guidance_weight=%.2f",
        args.rtc_inference_delay_steps,
        args.rtc_dynamic_inference_delay,
        args.rtc_delay_history,
        args.replan_steps,
        adapter.action_horizon,
        args.rtc_prefix_attention_schedule,
        args.rtc_max_guidance_weight,
    )
    logging.info(
        "rtc_scheduler=enabled execute_horizon_is_replan_steps async_prefetch_steps_ignored=%d async_plan_guard_steps_ignored=%d",
        args.async_prefetch_steps,
        args.async_plan_guard_steps,
    )

    request_queue: Queue[_RTCInferenceRequest] = Queue(maxsize=1)
    result_queue: Queue[_RTCInferenceResult] = Queue(maxsize=1)
    error_queue: Queue[BaseException] = Queue(maxsize=1)
    stop_event = threading.Event()
    infer_thread = _start_inference_worker(
        args,
        adapter,
        request_queue,
        result_queue,
        error_queue,
        stop_event,
        name="ur3-policy-rtc-infer",
    )

    total_successes = 0
    try:
        for episode_index in range(args.num_episodes):
            obs, info = _base._normalize_reset(env.reset())
            prompt = args.prompt or info.get("prompt") or obs.get("prompt") or args.default_prompt
            if prompt is None:
                raise ValueError(
                    "A prompt is required for UR3 evaluation. Pass --prompt/--default-prompt or return one from env.reset()."
                )

            video_dir = _base._resolve_video_dir(args, prompt=prompt)
            if video_dir is not None:
                video_dir.mkdir(parents=True, exist_ok=True)
                logging.info(
                    "episode=%d video_dir=%s video_filename=%s prompt=%s",
                    episode_index,
                    video_dir,
                    args.video_filename or "<auto>",
                    prompt,
                )
                video_path = _base._resolve_video_path(
                    video_dir,
                    video_filename=args.video_filename,
                    episode_index=episode_index,
                    num_episodes=args.num_episodes,
                )
            else:
                video_path = None

            episode_result = _run_rtc_episode(
                args,
                env=env,
                adapter=adapter,
                initial_obs=obs,
                prompt=prompt,
                episode_index=episode_index,
                request_queue=request_queue,
                result_queue=result_queue,
                error_queue=error_queue,
                infer_thread=infer_thread,
                manual_command_fn=None,
                video_path=video_path,
            )
            total_successes += int(episode_result.success)
            logging.info("episode=%d success=%s", episode_index, episode_result.success)

        logging.info("success_rate=%.3f", total_successes / max(args.num_episodes, 1))
    finally:
        stop_event.set()
        infer_thread.join(timeout=1.0)
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _run_interactive_mode(args: Args) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("Interactive mode requires a TTY on stdin. Pass --no-interactive for one-shot mode.")

    _validate_args(args)
    policy = create_policy(args)
    adapter = _RTCPolicyAdapter(policy)
    _validate_rtc_policy(args, adapter)
    env = _base._create_env(args)

    _base._log_video_config(args)
    if args.video_filename is not None and args.save_video != "off":
        logging.info("interactive mode ignores --video-filename and uses episode_XXXX.mp4 names.")
    logging.info(
        "rtc_mode=enabled initial_delay=%d dynamic_delay=%s delay_history=%d execute_horizon=%d action_horizon=%d "
        "schedule=%s max_guidance_weight=%.2f",
        args.rtc_inference_delay_steps,
        args.rtc_dynamic_inference_delay,
        args.rtc_delay_history,
        args.replan_steps,
        adapter.action_horizon,
        args.rtc_prefix_attention_schedule,
        args.rtc_max_guidance_weight,
    )
    logging.info(
        "rtc_scheduler=enabled execute_horizon_is_replan_steps async_prefetch_steps_ignored=%d async_plan_guard_steps_ignored=%d",
        args.async_prefetch_steps,
        args.async_plan_guard_steps,
    )

    request_queue: Queue[_RTCInferenceRequest] = Queue(maxsize=1)
    result_queue: Queue[_RTCInferenceResult] = Queue(maxsize=1)
    error_queue: Queue[BaseException] = Queue(maxsize=1)
    stop_event = threading.Event()
    infer_thread = _start_inference_worker(
        args,
        adapter,
        request_queue,
        result_queue,
        error_queue,
        stop_event,
        name="ur3-policy-rtc-infer",
    )

    try:
        last_prompt = args.prompt or args.default_prompt
        _base._set_env_prompt(env, last_prompt)
        current_obs, info = _base._normalize_reset(env.reset())
        if last_prompt is None:
            last_prompt = info.get("prompt") or current_obs.get("prompt") or args.default_prompt

        logging.info("interactive_rtc_session=ready")
        _base._print_interactive_help(args, last_prompt=last_prompt)

        with TerminalUI() as terminal:
            episode_index = 0

            while True:
                next_prompt = _base._read_next_prompt(args, terminal, last_prompt=last_prompt)
                if next_prompt == "/quit":
                    logging.info("interactive_session=quit")
                    break
                if next_prompt == "/reset":
                    logging.info("idle_reset_requested")
                    _base._set_env_prompt(env, last_prompt)
                    current_obs, _ = _base._normalize_reset(env.reset())
                    logging.info("idle_reset_complete")
                    continue

                last_prompt = next_prompt
                video_dir = _base._resolve_video_dir(args, prompt=last_prompt)
                if video_dir is not None:
                    video_dir.mkdir(parents=True, exist_ok=True)
                    video_path = _base._resolve_interactive_video_path(
                        video_dir,
                        episode_index=episode_index,
                    )
                    logging.info(
                        "episode=%d video_dir=%s video_path=%s prompt=%s",
                        episode_index,
                        video_dir,
                        video_path,
                        last_prompt,
                    )
                else:
                    video_path = None

                episode_result = _run_rtc_episode(
                    args,
                    env=env,
                    adapter=adapter,
                    initial_obs=current_obs,
                    prompt=last_prompt,
                    episode_index=episode_index,
                    request_queue=request_queue,
                    result_queue=result_queue,
                    error_queue=error_queue,
                    infer_thread=infer_thread,
                    manual_command_fn=lambda: _base._poll_manual_command(args, terminal),
                    video_path=video_path,
                )
                current_obs = episode_result.final_obs

                logging.info(
                    "episode=%d reason=%s steps=%d success=%s video=%s",
                    episode_index,
                    episode_result.reason,
                    episode_result.steps,
                    episode_result.success,
                    episode_result.video_path or "<disabled>",
                )

                episode_index += 1

                if episode_result.reason == "quit":
                    logging.info("interactive_session=quit")
                    break
                if episode_result.reason == "reset":
                    _base._set_env_prompt(env, last_prompt)
                    current_obs, _ = _base._normalize_reset(env.reset())
                    logging.info("post_episode_reset_complete")
    finally:
        stop_event.set()
        infer_thread.join(timeout=1.0)
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def run(args: Args) -> None:
    if args.interactive:
        _run_interactive_mode(args)
        return
    _run_single_episode_mode(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    run(tyro.cli(Args))
