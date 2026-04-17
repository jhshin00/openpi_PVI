from __future__ import annotations

import dataclasses
import importlib.util
import logging
import pathlib
from queue import Queue
import sys
import threading
import traceback
from typing import Literal

import numpy as np
import tyro


def _load_module(filename: str, module_name: str):
    module_path = pathlib.Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_gui = _load_module("main_interactive.py", "openpi_examples_ur3_main_interactive_base")
_backend = _load_module("main_RTC.py", "openpi_examples_ur3_main_RTC")
_gui._backend = _backend

PendingVideo = _gui.PendingVideo
SessionLogHandler = _gui.SessionLogHandler
TempVideoRecorder = _gui.TempVideoRecorder


@dataclasses.dataclass
class Args(_backend.Args):
    preview_source: Literal["cropped", "uncropped"] = "uncropped"
    mock_policy: bool = False
    server_host: str = "127.0.0.1"
    server_port: int = 7860
    server_debug: bool = False
    ui_title: str = "OpenPI UR3 RTC Interactive Control"
    log_buffer_lines: int = 200
    temp_video_dir: str = "/tmp/openpi_ur3_interactive"


def _validate_args(args: Args) -> None:
    _backend._validate_args(args)
    if args.mock_policy:
        raise ValueError("RTC interactive mode does not support --mock-policy.")
    if not args.policy_dir:
        raise ValueError("policy_dir is required.")


_gui._validate_args = _validate_args


class InteractivePolicySession(_gui.InteractivePolicySession):
    def _initialize(self) -> None:
        self.append_log("Loading trained policy.")
        self._policy = _backend.create_policy(self._args)
        self._rtc_adapter = _backend._RTCPolicyAdapter(self._policy)
        _backend._validate_rtc_policy(self._args, self._rtc_adapter)
        execute_horizon = _backend._resolve_execute_horizon(self._args)
        chunk_execution_mode = _backend._normalize_chunk_execution_mode(self._args.chunk_execution)

        self.append_log("Creating UR3 environment.")
        self._env = _backend._create_env(self._args)
        _gui._log_video_config(self._args)
        if self._args.video_filename is not None and self._args.save_video != "off":
            logging.info("GUI mode ignores --video-filename until a manual Save action is requested.")
        logging.info(
            "rtc_mode=enabled configured_delay=%d execute_horizon=%d overlap_horizon=%d action_horizon=%d "
            "schedule=%s max_guidance_weight=%.2f plan_mode=%s",
            self._args.rtc_inference_delay_steps,
            execute_horizon,
            self._rtc_adapter.action_horizon - execute_horizon,
            self._rtc_adapter.action_horizon,
            self._args.rtc_prefix_attention_schedule,
            self._args.rtc_max_guidance_weight,
            chunk_execution_mode,
        )

        self._start_inference_worker_if_needed()
        self._set_phase("resetting", "Resetting robot and cameras.")
        _backend._set_env_prompt(self._env, self._last_prompt)
        obs, info = _backend._normalize_reset(self._env.reset())

        if self._last_prompt is None:
            self._last_prompt = info.get("prompt") or obs.get("prompt") or self._args.default_prompt

        self._update_runtime(obs=obs, step=0, success=False, info={})
        self._set_phase("idle", "Ready. Enter a task and press Start.")
        logging.info("interactive_rtc_gui_session=ready")

    def _start_inference_worker_if_needed(self) -> None:
        if not self._args.async_inference:
            return

        self._inference_queue = Queue(maxsize=1)
        self._result_queue = Queue(maxsize=1)
        self._error_queue = Queue(maxsize=1)
        self._infer_stop_event = threading.Event()
        self._infer_thread = _backend._start_inference_worker(
            self._args,
            self._rtc_adapter,
            self._inference_queue,
            self._result_queue,
            self._error_queue,
            self._infer_stop_event,
            name="ur3-policy-gui-rtc-infer",
        )

    def _run_episode(self, episode_index: int, initial_obs: dict[str, object], prompt: str) -> None:
        assert self._env is not None
        assert self._inference_queue is not None
        assert self._result_queue is not None
        assert self._error_queue is not None
        assert self._infer_thread is not None

        recorder = TempVideoRecorder(self._args, prompt=prompt, episode_index=episode_index)
        obs = initial_obs
        success = False
        stop_reason = "max_steps"
        executed_steps = 0

        planner = _backend._RTCPlanner(
            self._args,
            self._rtc_adapter,
            prompt=prompt,
            episode_index=episode_index,
            request_queue=self._inference_queue,
            result_queue=self._result_queue,
            error_queue=self._error_queue,
            infer_thread=self._infer_thread,
        )

        try:
            _backend._set_env_prompt(self._env, prompt)
            _backend._drain_queue(self._inference_queue)
            _backend._drain_queue(self._result_queue)

            recorder.append_obs(obs)
            self._update_runtime(obs=obs, step=0, success=False, info={})
            planner.bootstrap(obs, step=0)

            for step in range(self._args.max_steps):
                if self._stop_requested.is_set():
                    stop_reason = "stop"
                    break

                action, _ = planner.next_action(obs, step=step)
                obs, reward, done, info = _backend._normalize_step(self._env.step(action))
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

                if self._args.debug_action_stats and step % max(self._args.debug_log_every, 1) == 0:
                    current = np.asarray(info.get("current_joint_positions", []), dtype=np.float32)
                    target = np.asarray(info.get("target_action", []), dtype=np.float32)
                    err = np.asarray(info.get("joint_error", []), dtype=np.float32)
                    qd = np.asarray(info.get("joint_velocity_cmd", []), dtype=np.float32)
                    if current.size and target.size and err.size and qd.size:
                        logging.info(
                            "control step=%d current=%s target=%s err=%s qd=%s max_abs_err=%.5f max_abs_qd=%.5f",
                            step,
                            np.array2string(current, precision=4, suppress_small=True),
                            np.array2string(target, precision=4, suppress_small=True),
                            np.array2string(err, precision=4, suppress_small=True),
                            np.array2string(qd, precision=4, suppress_small=True),
                            float(np.max(np.abs(err[:6]))),
                            float(np.max(np.abs(qd[:6]))),
                        )

                recorder.append_obs(obs)
                self._update_runtime(obs=obs, step=executed_steps, success=success, info=info)

                if done:
                    stop_reason = "done"
                    break

                planner.after_step(obs, step=step)
        except Exception:
            recorder.discard()
            with self._lock:
                self._running_prompt = None
                self._last_episode_reason = "error"
                self._last_episode_success = False
                self._episode_thread = None
                self._phase = "error"
                self._message = "Episode failed. Check logs."
            self.append_log(traceback.format_exc())
            return
        finally:
            self._stop_requested.clear()

        pending_video = recorder.finish()
        with self._lock:
            self._episode_index += 1
            self._running_prompt = None
            self._last_prompt = prompt
            self._last_episode_reason = stop_reason
            self._last_episode_success = success
            self._last_episode_steps = executed_steps
            self._current_step = executed_steps
            self._current_success = success
            self._current_obs = obs
            self._preview_revision += 1
            self._episode_thread = None

            if pending_video is not None:
                self._pending_video = pending_video
                self._phase = "pending_save"
                self._message = f"Episode stopped ({stop_reason}). Save or discard the recording."
            else:
                self._phase = "idle"
                self._message = f"Episode stopped ({stop_reason}). Ready for the next task."

        logging.info(
            "episode=%d reason=%s steps=%d success=%s",
            episode_index,
            stop_reason,
            executed_steps,
            success,
        )


def run(args: Args) -> None:
    session = InteractivePolicySession(args)
    app = _gui.create_app(session, args)

    logging.info(
        "interactive_gui_url=http://%s:%d host_override=use --server-host 0.0.0.0 for LAN access",
        args.server_host,
        args.server_port,
    )
    try:
        app.run(
            host=args.server_host,
            port=args.server_port,
            debug=args.server_debug,
            use_reloader=False,
            threaded=True,
        )
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    run(tyro.cli(Args))
