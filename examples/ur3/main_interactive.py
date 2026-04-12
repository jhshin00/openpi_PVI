from __future__ import annotations

import collections
import dataclasses
import importlib.util
import io
import logging
import os
import pathlib
from queue import Empty
from queue import Full
from queue import Queue
import shutil
import sys
import tempfile
import threading
import time
import traceback
from typing import Any
from typing import Literal

from flask import Flask
from flask import Response
from flask import jsonify
from flask import render_template_string
from flask import request
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import tyro


def _load_backend_module():
    module_path = pathlib.Path(__file__).with_name("main_modify.py")
    spec = importlib.util.spec_from_file_location("openpi_examples_ur3_main_modify", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load backend module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_backend = _load_backend_module()


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <style>
    :root {
      --bg: #f2ede4;
      --panel: rgba(255, 252, 245, 0.9);
      --panel-strong: #fffaf0;
      --line: rgba(70, 52, 32, 0.14);
      --ink: #261b12;
      --muted: #6f6255;
      --accent: #c96e1e;
      --accent-strong: #9f4e0c;
      --danger: #a2352b;
      --ok: #2f7a48;
      --shadow: 0 20px 60px rgba(40, 28, 16, 0.14);
      --mono: "JetBrains Mono", "Fira Code", monospace;
      --sans: "IBM Plex Sans", "Segoe UI", sans-serif;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(201, 110, 30, 0.18), transparent 28%),
        radial-gradient(circle at bottom right, rgba(47, 122, 72, 0.14), transparent 25%),
        linear-gradient(180deg, #f8f3ea 0%, #efe5d6 100%);
      min-height: 100vh;
    }

    .shell {
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      gap: 20px;
    }

    .hero {
      background: linear-gradient(135deg, rgba(255, 250, 240, 0.95), rgba(245, 234, 214, 0.92));
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 28px;
      display: grid;
      gap: 14px;
    }

    .eyebrow {
      font-family: var(--mono);
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--accent-strong);
    }

    h1 {
      margin: 0;
      font-size: clamp(28px, 5vw, 44px);
      line-height: 1;
      letter-spacing: -0.04em;
    }

    .status-line {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }

    .badge {
      border-radius: 999px;
      padding: 8px 14px;
      background: rgba(38, 27, 18, 0.08);
      border: 1px solid rgba(38, 27, 18, 0.08);
      font-family: var(--mono);
      font-size: 12px;
    }

    .grid {
      display: grid;
      gap: 20px;
      grid-template-columns: minmax(320px, 430px) minmax(0, 1fr);
    }

    .column {
      display: grid;
      gap: 20px;
      align-content: start;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      padding: 20px;
      display: grid;
      gap: 14px;
    }

    .panel h2 {
      margin: 0;
      font-size: 18px;
      letter-spacing: -0.02em;
    }

    .subtle {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }

    textarea,
    input[type="text"] {
      width: 100%;
      border: 1px solid rgba(38, 27, 18, 0.14);
      background: var(--panel-strong);
      border-radius: 16px;
      padding: 14px 16px;
      color: var(--ink);
      font: inherit;
      resize: vertical;
      min-height: 56px;
    }

    input[type="text"] {
      min-height: 0;
    }

    textarea:focus,
    input[type="text"]:focus {
      outline: 2px solid rgba(201, 110, 30, 0.35);
      outline-offset: 0;
      border-color: rgba(201, 110, 30, 0.45);
    }

    .button-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease, background 120ms ease;
      background: rgba(38, 27, 18, 0.08);
      color: var(--ink);
    }

    button:hover:enabled {
      transform: translateY(-1px);
    }

    button:disabled {
      opacity: 0.38;
      cursor: not-allowed;
    }

    .primary {
      background: linear-gradient(135deg, var(--accent), var(--accent-strong));
      color: white;
    }

    .danger {
      background: linear-gradient(135deg, #d3584b, var(--danger));
      color: white;
    }

    .secondary {
      background: linear-gradient(135deg, #3f8a58, var(--ok));
      color: white;
    }

    .stats {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    }

    .stat {
      padding: 14px;
      border-radius: 16px;
      background: rgba(255, 250, 240, 0.95);
      border: 1px solid var(--line);
    }

    .stat .label {
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }

    .stat .value {
      font-family: var(--mono);
      font-size: 15px;
      word-break: break-word;
    }

    .preview-grid {
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    }

    .preview-card {
      display: grid;
      gap: 10px;
    }

    .preview-card img {
      width: 100%;
      aspect-ratio: 4 / 3;
      border-radius: 18px;
      object-fit: cover;
      background: linear-gradient(135deg, #24170d, #6f6255);
      border: 1px solid rgba(38, 27, 18, 0.18);
    }

    .preview-label {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      font-family: var(--mono);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }

    pre {
      margin: 0;
      padding: 16px;
      border-radius: 18px;
      border: 1px solid rgba(38, 27, 18, 0.12);
      background: #20150d;
      color: #f7ead5;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.45;
      max-height: 360px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .save-panel[hidden] {
      display: none;
    }

    .error {
      color: var(--danger);
      font-weight: 600;
      min-height: 1.2em;
    }

    @media (max-width: 1024px) {
      .grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">Persistent UR3 Policy Session</div>
      <h1>{{ title }}</h1>
      <div class="status-line">
        <span class="badge" id="phase-pill">phase: booting</span>
        <span class="badge" id="message-pill">starting session</span>
        <span class="badge" id="recording-pill">recording: unknown</span>
      </div>
      <div class="subtle">
        Model and checkpoint stay loaded. Change the task text, press Start, then Stop, Save, Discard, or Reset without
        restarting the process.
      </div>
    </section>

    <section class="grid">
      <div class="column">
        <div class="panel">
          <h2>Controls</h2>
          <div class="subtle" id="control-hint">Leave the task box empty to reuse the last prompt.</div>
          <textarea id="prompt" rows="4" placeholder="pick up apple and place it in sink"></textarea>
          <div class="button-row">
            <button class="primary" id="start-btn">Start</button>
            <button class="danger" id="stop-btn">Stop</button>
            <button id="reset-btn">Reset</button>
          </div>
          <div class="error" id="error-box"></div>
        </div>

        <div class="panel save-panel" id="save-panel" hidden>
          <h2>Recording</h2>
          <div class="subtle" id="save-summary">Recording is ready to save.</div>
          <input type="text" id="video-filename" placeholder="episode_0000.mp4">
          <div class="button-row">
            <button class="secondary" id="save-btn">Save Video</button>
            <button id="discard-btn">Discard</button>
          </div>
        </div>

        <div class="panel">
          <h2>Run State</h2>
          <div class="stats">
            <div class="stat">
              <div class="label">Episode</div>
              <div class="value" id="episode-value">-</div>
            </div>
            <div class="stat">
              <div class="label">Step</div>
              <div class="value" id="step-value">-</div>
            </div>
            <div class="stat">
              <div class="label">Prompt</div>
              <div class="value" id="prompt-value">-</div>
            </div>
            <div class="stat">
              <div class="label">Last Reason</div>
              <div class="value" id="reason-value">-</div>
            </div>
            <div class="stat">
              <div class="label">Success</div>
              <div class="value" id="success-value">-</div>
            </div>
            <div class="stat">
              <div class="label">Saved Video</div>
              <div class="value" id="saved-video-value">-</div>
            </div>
          </div>
          <div class="subtle" id="state-value"></div>
        </div>
      </div>

      <div class="column">
        <div class="panel">
          <h2>Camera Preview</h2>
          <div class="preview-grid">
            <div class="preview-card">
              <div class="preview-label">
                <span>Base Camera</span>
                <span id="base-source-label">{{ preview_source }}</span>
              </div>
              <img id="base-preview" alt="Base camera preview">
            </div>
            <div class="preview-card">
              <div class="preview-label">
                <span>Wrist Camera</span>
                <span id="wrist-source-label">{{ preview_source }}</span>
              </div>
              <img id="wrist-preview" alt="Wrist camera preview">
            </div>
          </div>
        </div>

        <div class="panel">
          <h2>Logs</h2>
          <pre id="logs"></pre>
        </div>
      </div>
    </section>
  </div>

  <script>
    const promptEl = document.getElementById("prompt");
    const filenameEl = document.getElementById("video-filename");
    const errorEl = document.getElementById("error-box");
    const savePanel = document.getElementById("save-panel");
    let previewRevision = -1;

    filenameEl.addEventListener("input", () => {
      filenameEl.dataset.dirty = "1";
    });

    async function callApi(path, body) {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        cache: "no-store",
        body: JSON.stringify(body || {}),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "request failed");
      }
      return data;
    }

    function setError(message) {
      errorEl.textContent = message || "";
    }

    function formatArray(values) {
      if (!Array.isArray(values) || values.length === 0) {
        return "-";
      }
      return values.map((value) => Number(value).toFixed(4)).join(", ");
    }

    function updatePreview(status) {
      if (status.preview_revision === previewRevision) {
        return;
      }
      previewRevision = status.preview_revision;
      const token = `rev=${status.preview_revision}&t=${Date.now()}`;
      document.getElementById("base-preview").src = `/api/preview/base.jpg?${token}`;
      document.getElementById("wrist-preview").src = `/api/preview/wrist.jpg?${token}`;
    }

    function applyStatus(status) {
      document.getElementById("phase-pill").textContent = `phase: ${status.phase}`;
      document.getElementById("message-pill").textContent = status.message;
      document.getElementById("recording-pill").textContent = status.pending_video
        ? `recording: ready (${status.pending_video.frames} frames)`
        : `recording: ${status.video_enabled ? "armed" : "disabled"}`;
      document.getElementById("episode-value").textContent = status.episode_index;
      document.getElementById("step-value").textContent = status.current_step;
      document.getElementById("prompt-value").textContent = status.running_prompt || status.last_prompt || "-";
      document.getElementById("reason-value").textContent = status.last_episode_reason || "-";
      document.getElementById("success-value").textContent =
        status.last_episode_success === null ? "-" : String(status.last_episode_success);
      document.getElementById("saved-video-value").textContent = status.last_saved_video || "-";
      document.getElementById("state-value").textContent = `joint state: ${formatArray(status.current_state)}`;
      document.getElementById("logs").textContent = (status.logs || []).join("\\n");

      promptEl.placeholder = status.last_prompt || promptEl.placeholder;

      document.getElementById("start-btn").disabled = !status.can_start;
      document.getElementById("stop-btn").disabled = !status.can_stop;
      document.getElementById("reset-btn").disabled = !status.can_reset;

      savePanel.hidden = !status.pending_video;
      document.getElementById("save-btn").disabled = !status.can_save;
      document.getElementById("discard-btn").disabled = !status.can_discard;

      if (status.pending_video) {
        document.getElementById("save-summary").textContent =
          `Prompt "${status.pending_video.prompt}" stopped after ${status.pending_video.duration_sec.toFixed(1)}s / ${status.pending_video.frames} frames.`;
        if (!filenameEl.dataset.dirty && status.pending_video.suggested_filename) {
          filenameEl.value = status.pending_video.suggested_filename;
        }
      } else if (!filenameEl.dataset.dirty) {
        filenameEl.value = "";
      }

      updatePreview(status);
    }

    async function refreshStatus() {
      try {
        const response = await fetch("/api/status", { cache: "no-store" });
        const data = await response.json();
        applyStatus(data);
      } catch (error) {
        setError(error.message);
      }
    }

    document.getElementById("start-btn").addEventListener("click", async () => {
      try {
        setError("");
        const data = await callApi("/api/start", { prompt: promptEl.value });
        filenameEl.dataset.dirty = "";
        applyStatus(data);
      } catch (error) {
        setError(error.message);
      }
    });

    document.getElementById("stop-btn").addEventListener("click", async () => {
      try {
        setError("");
        const data = await callApi("/api/stop", {});
        applyStatus(data);
      } catch (error) {
        setError(error.message);
      }
    });

    document.getElementById("reset-btn").addEventListener("click", async () => {
      try {
        setError("");
        const data = await callApi("/api/reset", {});
        filenameEl.dataset.dirty = "";
        applyStatus(data);
      } catch (error) {
        setError(error.message);
      }
    });

    document.getElementById("save-btn").addEventListener("click", async () => {
      try {
        setError("");
        const data = await callApi("/api/save_video", { filename: filenameEl.value });
        filenameEl.dataset.dirty = "";
        applyStatus(data);
      } catch (error) {
        setError(error.message);
      }
    });

    document.getElementById("discard-btn").addEventListener("click", async () => {
      try {
        setError("");
        const data = await callApi("/api/discard_video", {});
        filenameEl.dataset.dirty = "";
        applyStatus(data);
      } catch (error) {
        setError(error.message);
      }
    });

    refreshStatus();
    setInterval(refreshStatus, 500);
  </script>
</body>
</html>
"""


@dataclasses.dataclass
class Args:
    policy_config: str = "pi05_ur3_pvi_dinov2_infer"
    policy_dir: str | None = None
    default_prompt: str | None = None
    prompt: str | None = None
    replan_steps: int = 8
    async_inference: bool = True
    async_prefetch_steps: int = 6
    async_plan_guard_steps: int = 1
    max_steps: int = 200
    pytorch_device: str | None = None
    save_video: Literal["auto", "on", "off"] = "auto"
    video_out_dir: str | None = None
    video_filename: str | None = None
    video_frame_source: Literal["cropped", "uncropped"] = "cropped"
    preview_source: Literal["cropped", "uncropped"] = "uncropped"
    debug_action_stats: bool = False
    debug_log_every: int = 1
    chunk_execution: str = "chunk_endpoint"
    mock_policy: bool = False

    env_factory: str | None = None
    env_kwargs_json: str = "{}"

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
    gripper_open_threshold: float = 0.3
    gripper_close_threshold: float = 0.7
    reset_steps: int = 120
    reset_max_delta: float = 0.05
    camera_warmup_sec: float = 5.0
    mock: bool = False

    server_host: str = "127.0.0.1"
    server_port: int = 7860
    server_debug: bool = False
    ui_title: str = "OpenPI UR3 Interactive Control"
    log_buffer_lines: int = 200
    temp_video_dir: str = "/tmp/openpi_ur3_interactive"


@dataclasses.dataclass
class PendingVideo:
    prompt: str
    episode_index: int
    temp_path: pathlib.Path
    frames_written: int
    started_at: float
    ended_at: float


class _MockPolicy:
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(_backend._first_present(obs, "observation/state", "state"), dtype=np.float32)
        if state.shape != (7,):
            raise ValueError(f"Expected a 7D state for mock policy, got {state.shape}")

        repeated = np.repeat(state[None, :], 32, axis=0)
        return {
            "actions": repeated,
            "policy_timing": {"infer_ms": 0.0},
        }


class SessionLogHandler(logging.Handler):
    def __init__(self, session: "InteractivePolicySession") -> None:
        super().__init__(level=logging.INFO)
        self._session = session
        self.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._session.append_log(self.format(record))
        except Exception:
            self.handleError(record)


class TempVideoRecorder:
    def __init__(self, args: Args, *, prompt: str, episode_index: int) -> None:
        self._args = args
        self._prompt = prompt
        self._episode_index = episode_index
        self._temp_path: pathlib.Path | None = None
        self._writer = None
        self._frames_written = 0
        self._started_at = time.time()

        if args.save_video == "off":
            return

        temp_dir = pathlib.Path(args.temp_video_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f"openpi_ur3_episode_{episode_index:04d}_",
            suffix=".mp4",
            dir=temp_dir,
        )
        os.close(fd)
        self._temp_path = pathlib.Path(temp_name)
        self._writer = imageio.get_writer(self._temp_path, fps=max(args.hz, 1))

    def append_obs(self, obs: dict[str, Any]) -> None:
        if self._writer is None:
            return
        frame = _backend._get_frame(obs, source=self._args.video_frame_source)
        self._writer.append_data(frame)
        self._frames_written += 1

    def finish(self) -> PendingVideo | None:
        if self._writer is None or self._temp_path is None:
            return None

        writer = self._writer
        self._writer = None
        writer.close()

        if self._frames_written == 0:
            self.discard()
            return None

        return PendingVideo(
            prompt=self._prompt,
            episode_index=self._episode_index,
            temp_path=self._temp_path,
            frames_written=self._frames_written,
            started_at=self._started_at,
            ended_at=time.time(),
        )

    def discard(self) -> None:
        if self._writer is not None:
            writer = self._writer
            self._writer = None
            writer.close()

        if self._temp_path is not None:
            try:
                self._temp_path.unlink(missing_ok=True)
            except TypeError:
                if self._temp_path.exists():
                    self._temp_path.unlink()


def _validate_args(args: Args) -> None:
    if not args.mock_policy and not args.policy_dir:
        raise ValueError("policy_dir is required unless --mock-policy is enabled.")
    if args.replan_steps <= 0:
        raise ValueError("replan_steps must be positive")
    if args.async_prefetch_steps < 0:
        raise ValueError("async_prefetch_steps must be non-negative")
    if args.async_plan_guard_steps < 0:
        raise ValueError("async_plan_guard_steps must be non-negative")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.server_port <= 0:
        raise ValueError("server_port must be positive")
    if args.log_buffer_lines <= 0:
        raise ValueError("log_buffer_lines must be positive")


def _image_to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _extract_preview_image(
    obs: dict[str, Any],
    *,
    camera: Literal["base", "wrist"],
    source: Literal["cropped", "uncropped"],
) -> np.ndarray | None:
    if camera == "base":
        uncropped_keys = (
            "observation/base_image_uncropped",
            "base_image_uncropped",
            "observation/base_image_raw",
            "base_image_raw",
        )
        cropped_keys = ("observation/base_image", "base_image", "image")
    else:
        uncropped_keys = ("observation/wrist_image_uncropped", "wrist_image_uncropped")
        cropped_keys = ("observation/wrist_image", "wrist_image")

    ordered_keys = uncropped_keys + cropped_keys if source == "uncropped" else cropped_keys + uncropped_keys
    for key in ordered_keys:
        if key in obs:
            return _image_to_uint8(np.asarray(obs[key]))
    return None


def _encode_jpeg(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _unique_path(path: pathlib.Path) -> pathlib.Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}_{index:02d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a unique path for {path}")


def _default_video_filename(prompt: str, episode_index: int) -> str:
    prompt_slug = _backend._prompt_video_dirname(prompt)
    return f"{prompt_slug}_{episode_index:04d}.mp4"


def _effective_policy_dir(args: Args) -> str:
    return args.policy_dir or "mock_policy"


def _resolve_video_dir(args: Args, *, prompt: str | None) -> pathlib.Path | None:
    if args.save_video == "off":
        return None
    if args.video_out_dir is not None:
        return pathlib.Path(args.video_out_dir)

    prompt_dir = _backend._prompt_video_dirname(prompt) if prompt is not None else "task"
    return (
        _backend.DEFAULT_VIDEO_ROOT
        / _backend._policy_video_dirname(args.policy_config)
        / _backend._checkpoint_video_dirname(_effective_policy_dir(args))
        / prompt_dir
    )


def _log_video_config(args: Args) -> None:
    if args.save_video == "off":
        logging.info("video_saving=disabled")
    elif args.video_out_dir is not None:
        logging.info(
            "video_saving=enabled video_dir=%s video_filename=%s video_frame_source=%s",
            pathlib.Path(args.video_out_dir),
            args.video_filename or "<manual>",
            args.video_frame_source,
        )
    else:
        logging.info(
            "video_saving=enabled video_dir_root=%s policy=%s checkpoint=%s prompt=<auto> video_frame_source=%s",
            _backend.DEFAULT_VIDEO_ROOT,
            _backend._policy_video_dirname(args.policy_config),
            _backend._checkpoint_video_dirname(_effective_policy_dir(args)),
            args.video_frame_source,
        )


class InteractivePolicySession:
    def __init__(self, args: Args) -> None:
        _validate_args(args)
        self._args = args
        self._lock = threading.RLock()
        self._logs: collections.deque[str] = collections.deque(maxlen=args.log_buffer_lines)
        self._phase = "booting"
        self._message = "Loading policy and environment."
        self._current_obs: dict[str, Any] | None = None
        self._last_info: dict[str, Any] = {}
        self._last_prompt = args.prompt or args.default_prompt
        self._running_prompt: str | None = None
        self._current_step = 0
        self._current_success = False
        self._episode_index = 0
        self._last_episode_reason: str | None = None
        self._last_episode_success: bool | None = None
        self._last_episode_steps = 0
        self._last_saved_video: str | None = None
        self._preview_revision = 0
        self._pending_video: PendingVideo | None = None
        self._episode_thread: threading.Thread | None = None
        self._stop_requested = threading.Event()
        self._closed = False

        self._inference_queue: Queue[tuple[int, int, dict[str, Any]]] | None = None
        self._result_queue: Queue[tuple[int, int, dict[str, Any]]] | None = None
        self._error_queue: Queue[BaseException] | None = None
        self._infer_stop_event: threading.Event | None = None
        self._infer_thread: threading.Thread | None = None

        self._policy = None
        self._env = None

        self._log_handler = SessionLogHandler(self)
        logging.getLogger().addHandler(self._log_handler)

        try:
            self._initialize()
        except Exception:
            self._set_phase("error", "Initialization failed.")
            self.append_log(traceback.format_exc())
            self.close()
            raise

    def append_log(self, message: str) -> None:
        with self._lock:
            self._logs.append(message)

    def _set_phase(self, phase: str, message: str) -> None:
        with self._lock:
            self._phase = phase
            self._message = message

    def _update_runtime(
        self,
        *,
        obs: dict[str, Any],
        step: int,
        success: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._current_obs = obs
            self._last_info = info or {}
            self._current_step = step
            self._current_success = success
            self._preview_revision += 1

    def _initialize(self) -> None:
        if self._args.mock_policy:
            self.append_log("mock_policy=enabled")
            self._policy = _MockPolicy()
        else:
            self.append_log("Loading trained policy.")
            self._policy = _backend.create_policy(self._args)

        self.append_log("Creating UR3 environment.")
        self._env = _backend._create_env(self._args)
        _log_video_config(self._args)
        if self._args.video_filename is not None and self._args.save_video != "off":
            logging.info("GUI mode ignores --video-filename until a manual Save action is requested.")

        self._start_inference_worker_if_needed()
        self._set_phase("resetting", "Resetting robot and cameras.")
        _backend._set_env_prompt(self._env, self._last_prompt)
        obs, info = _backend._normalize_reset(self._env.reset())

        if self._last_prompt is None:
            self._last_prompt = info.get("prompt") or obs.get("prompt") or self._args.default_prompt

        self._update_runtime(obs=obs, step=0, success=False, info={})
        self._set_phase("idle", "Ready. Enter a task and press Start.")
        logging.info("interactive_gui_session=ready")

    def _start_inference_worker_if_needed(self) -> None:
        if not self._args.async_inference:
            return

        self._inference_queue = Queue(maxsize=1)
        self._result_queue = Queue(maxsize=1)
        self._error_queue = Queue(maxsize=1)
        self._infer_stop_event = threading.Event()

        def inference_worker() -> None:
            assert self._policy is not None
            assert self._inference_queue is not None
            assert self._result_queue is not None
            assert self._error_queue is not None
            assert self._infer_stop_event is not None

            while not self._infer_stop_event.is_set():
                try:
                    request_episode, request_step, request_obs = self._inference_queue.get(timeout=0.1)
                except Empty:
                    continue

                try:
                    policy_result = self._policy.infer(request_obs)
                    _backend._drain_queue(self._result_queue)
                    self._result_queue.put_nowait((request_episode, request_step, policy_result))
                except Exception as exc:
                    logging.exception(
                        "Inference worker failed at episode=%d step=%d",
                        request_episode,
                        request_step,
                    )
                    _backend._drain_queue(self._error_queue)
                    self._error_queue.put_nowait(exc)
                    return

        self._infer_thread = threading.Thread(
            target=inference_worker,
            name="ur3-policy-gui-infer",
            daemon=True,
        )
        self._infer_thread.start()

    def _ensure_inference_worker(self) -> None:
        if not self._args.async_inference:
            return

        assert self._error_queue is not None
        assert self._infer_thread is not None

        try:
            worker_exc = self._error_queue.get_nowait()
        except Empty:
            worker_exc = None

        if worker_exc is not None:
            raise RuntimeError("Inference worker failed.") from worker_exc
        if not self._infer_thread.is_alive():
            raise RuntimeError("Inference worker thread stopped unexpectedly.")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            current_state_source = self._last_info.get("current_joint_positions")
            if current_state_source is None and self._current_obs is not None:
                current_state_source = self._current_obs.get("state")
            if current_state_source is None:
                current_state_source = []
            current_state = np.asarray(current_state_source, dtype=np.float32)
            pending = self._pending_video
            return {
                "phase": self._phase,
                "message": self._message,
                "episode_index": self._episode_index,
                "current_step": self._current_step,
                "running_prompt": self._running_prompt,
                "last_prompt": self._last_prompt,
                "last_episode_reason": self._last_episode_reason,
                "last_episode_success": self._last_episode_success,
                "last_episode_steps": self._last_episode_steps,
                "last_saved_video": self._last_saved_video,
                "current_state": current_state.tolist(),
                "preview_revision": self._preview_revision,
                "video_enabled": self._args.save_video != "off",
                "logs": list(self._logs),
                "can_start": self._phase == "idle" and self._pending_video is None,
                "can_stop": self._phase in {"running", "stopping"},
                "can_reset": self._phase == "idle" and self._pending_video is None,
                "can_save": self._phase == "pending_save" and pending is not None,
                "can_discard": self._phase == "pending_save" and pending is not None,
                "pending_video": None
                if pending is None
                else {
                    "prompt": pending.prompt,
                    "frames": pending.frames_written,
                    "duration_sec": max(pending.ended_at - pending.started_at, 0.0),
                    "suggested_filename": _default_video_filename(pending.prompt, pending.episode_index),
                },
            }

    def _get_obs_for_episode_start(self) -> dict[str, Any]:
        with self._lock:
            if self._current_obs is None:
                raise RuntimeError("No observation is available yet.")
            return self._current_obs

    def start(self, prompt: str | None) -> dict[str, Any]:
        prompt_text = (prompt or "").strip()
        with self._lock:
            if self._phase == "error":
                raise RuntimeError("Session is in an error state. Restart the process.")
            if self._pending_video is not None:
                raise RuntimeError("Save or discard the previous recording before starting a new task.")
            if self._episode_thread is not None and self._episode_thread.is_alive():
                raise RuntimeError("An episode is already running.")

            if not prompt_text:
                prompt_text = (self._last_prompt or "").strip()
            if not prompt_text:
                raise ValueError("A task prompt is required.")

            episode_index = self._episode_index
            initial_obs = self._current_obs
            if initial_obs is None:
                raise RuntimeError("The environment is not ready yet.")

            self._stop_requested.clear()
            self._running_prompt = prompt_text
            self._last_prompt = prompt_text
            self._current_step = 0
            self._current_success = False
            self._last_episode_reason = None
            self._last_episode_success = None
            self._set_phase("running", f"Running task: {prompt_text}")

            self._episode_thread = threading.Thread(
                target=self._run_episode,
                args=(episode_index, initial_obs, prompt_text),
                name=f"ur3-gui-episode-{episode_index:04d}",
                daemon=True,
            )
            self._episode_thread.start()

        logging.info("episode=%d prompt=%s", episode_index, prompt_text)
        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._episode_thread is None or not self._episode_thread.is_alive():
                raise RuntimeError("No episode is running.")
            self._stop_requested.set()
            self._phase = "stopping"
            self._message = "Stop requested. Waiting for the current control step to finish."
        logging.info("manual_stop_requested")
        return self.snapshot()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            if self._pending_video is not None:
                raise RuntimeError("Save or discard the recording before resetting.")
            if self._episode_thread is not None and self._episode_thread.is_alive():
                raise RuntimeError("Stop the current episode before resetting.")
            if self._env is None:
                raise RuntimeError("Environment is not initialized.")
            self._phase = "resetting"
            self._message = "Resetting robot and cameras."
            prompt = self._last_prompt

        try:
            _backend._set_env_prompt(self._env, prompt)
            obs, info = _backend._normalize_reset(self._env.reset())
        except Exception:
            self._set_phase("error", "Reset failed.")
            self.append_log(traceback.format_exc())
            raise

        if prompt is None:
            prompt = info.get("prompt") or obs.get("prompt") or self._args.default_prompt

        with self._lock:
            self._last_prompt = prompt
            self._running_prompt = None
            self._current_step = 0
            self._last_episode_reason = None
            self._last_episode_success = None
            self._last_episode_steps = 0
            self._update_runtime(obs=obs, step=0, success=False, info={})
            self._phase = "idle"
            self._message = "Reset complete. Ready for the next task."

        logging.info("manual_reset_complete")
        return self.snapshot()

    def save_video(self, filename: str | None) -> dict[str, Any]:
        with self._lock:
            pending = self._pending_video
            if pending is None:
                raise RuntimeError("There is no pending recording to save.")
            self._phase = "saving"
            self._message = "Saving video."

        try:
            video_dir = _resolve_video_dir(self._args, prompt=pending.prompt)
            if video_dir is None:
                raise RuntimeError("Video saving is disabled.")
            video_dir.mkdir(parents=True, exist_ok=True)

            resolved_path = _backend._resolve_video_path(
                video_dir,
                video_filename=(filename or "").strip() or None,
                episode_index=pending.episode_index,
                num_episodes=1,
            )
            final_path = _unique_path(resolved_path)
            shutil.move(str(pending.temp_path), str(final_path))
        except Exception:
            self._set_phase("pending_save", "Save failed. Choose another filename or discard the recording.")
            self.append_log(traceback.format_exc())
            raise

        with self._lock:
            self._pending_video = None
            self._last_saved_video = str(final_path)
            self._phase = "idle"
            self._message = f"Saved video: {final_path.name}"

        logging.info("saved_video=%s", final_path)
        return self.snapshot()

    def discard_video(self) -> dict[str, Any]:
        with self._lock:
            pending = self._pending_video
            if pending is None:
                raise RuntimeError("There is no pending recording to discard.")

        pending.temp_path.unlink(missing_ok=True)

        with self._lock:
            self._pending_video = None
            self._phase = "idle"
            self._message = "Recording discarded."

        logging.info("discarded_video=%s", pending.temp_path)
        return self.snapshot()

    def get_preview_bytes(self, camera: Literal["base", "wrist"]) -> bytes | None:
        with self._lock:
            obs = self._current_obs
        if obs is None:
            return None

        image = _extract_preview_image(obs, camera=camera, source=self._args.preview_source)
        if image is None:
            return None
        return _encode_jpeg(image)

    def _run_episode(self, episode_index: int, initial_obs: dict[str, Any], prompt: str) -> None:
        assert self._env is not None
        assert self._policy is not None

        recorder = TempVideoRecorder(self._args, prompt=prompt, episode_index=episode_index)
        obs = initial_obs
        success = False
        stop_reason = "max_steps"
        executed_steps = 0
        action_plan: collections.deque[np.ndarray] = collections.deque()
        pending_request_step: int | None = None
        hold_steps = 0

        try:
            _backend._set_env_prompt(self._env, prompt)
            if self._inference_queue is not None:
                _backend._drain_queue(self._inference_queue)
            if self._result_queue is not None:
                _backend._drain_queue(self._result_queue)

            recorder.append_obs(obs)
            self._update_runtime(obs=obs, step=0, success=False, info={})

            for step in range(self._args.max_steps):
                if self._stop_requested.is_set():
                    stop_reason = "stop"
                    break

                if self._args.async_inference:
                    self._ensure_inference_worker()
                    assert self._inference_queue is not None
                    assert self._result_queue is not None

                    policy_result: dict[str, Any] | None = None
                    request_step: int | None = None
                    while True:
                        try:
                            result_episode, request_step_candidate, policy_result_candidate = self._result_queue.get_nowait()
                        except Empty:
                            break

                        if result_episode != episode_index:
                            continue
                        request_step = request_step_candidate
                        policy_result = policy_result_candidate

                    if policy_result is not None and request_step is not None:
                        pending_request_step = None
                        delay_steps = max(step - request_step, 0)
                        current_state = _backend._current_state_from_obs(obs)
                        raw_actions = np.asarray(policy_result["actions"], dtype=np.float32)
                        raw_chunk = _backend._extract_action_window(
                            raw_actions,
                            start=delay_steps,
                            size=self._args.replan_steps,
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
                            chunk = _backend._build_action_plan(raw_chunk, current_state, self._args.chunk_execution)
                            _backend._replace_action_plan(
                                action_plan,
                                chunk,
                                guard_steps=self._args.async_plan_guard_steps,
                            )

                            if self._args.debug_action_stats:
                                joint_delta = chunk[:, :6] - current_state[None, :6]
                                logging.info(
                                    "policy_chunk step=%d req_step=%d delay_steps=%d infer_ms=%.2f plan_mode=%s "
                                    "first_target=%s last_target=%s first_delta=%s last_delta=%s "
                                    "chunk_max_abs_delta=%.5f chunk_mean_abs_delta=%.5f queue_len=%d",
                                    step,
                                    request_step,
                                    delay_steps,
                                    float(policy_result.get("policy_timing", {}).get("infer_ms", -1.0)),
                                    self._args.chunk_execution,
                                    np.array2string(chunk[0], precision=4, suppress_small=True),
                                    np.array2string(chunk[-1], precision=4, suppress_small=True),
                                    np.array2string(joint_delta[0], precision=4, suppress_small=True),
                                    np.array2string(joint_delta[-1], precision=4, suppress_small=True),
                                    float(np.max(np.abs(joint_delta))),
                                    float(np.mean(np.abs(joint_delta))),
                                    len(action_plan),
                                )

                    if pending_request_step is None and len(action_plan) <= self._args.async_prefetch_steps:
                        policy_obs = _backend._to_policy_observation(obs, prompt)
                        try:
                            self._inference_queue.put_nowait((episode_index, step, policy_obs))
                        except Full:
                            pass
                        else:
                            pending_request_step = step

                else:
                    if not action_plan:
                        policy_obs = _backend._to_policy_observation(obs, prompt)
                        policy_result = self._policy.infer(policy_obs)
                        raw_actions = np.asarray(policy_result["actions"], dtype=np.float32)
                        raw_chunk = _backend._extract_action_window(
                            raw_actions,
                            start=0,
                            size=self._args.replan_steps,
                            strict_size=True,
                        )
                        current_state = _backend._current_state_from_obs(obs)
                        chunk = _backend._build_action_plan(raw_chunk, current_state, self._args.chunk_execution)

                        if self._args.debug_action_stats:
                            joint_delta = chunk[:, :6] - current_state[None, :6]
                            logging.info(
                                "policy_chunk step=%d infer_ms=%.2f plan_mode=%s "
                                "first_target=%s last_target=%s first_delta=%s last_delta=%s "
                                "chunk_max_abs_delta=%.5f chunk_mean_abs_delta=%.5f",
                                step,
                                float(policy_result.get("policy_timing", {}).get("infer_ms", -1.0)),
                                self._args.chunk_execution,
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
                    action = _backend._current_state_from_obs(obs).copy()
                    hold_steps += 1
                    if self._args.debug_action_stats and hold_steps % max(self._args.debug_log_every, 1) == 0:
                        logging.info(
                            "holding_current_pose step=%d hold_steps=%d pending_request=%s",
                            step,
                            hold_steps,
                            pending_request_step,
                        )

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

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._episode_thread
            pending = self._pending_video

        self._stop_requested.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

        if pending is not None:
            pending.temp_path.unlink(missing_ok=True)

        if self._infer_stop_event is not None:
            self._infer_stop_event.set()
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=1.0)

        if self._env is not None:
            close_fn = getattr(self._env, "close", None)
            if callable(close_fn):
                close_fn()

        logging.getLogger().removeHandler(self._log_handler)
        self._log_handler.close()


def create_app(session: InteractivePolicySession, args: Args) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return render_template_string(
            HTML_TEMPLATE,
            title=args.ui_title,
            preview_source=args.preview_source,
        )

    @app.get("/api/status")
    def api_status():
        return jsonify(session.snapshot())

    @app.get("/api/preview/<camera>.jpg")
    def api_preview(camera: str):
        if camera not in {"base", "wrist"}:
            return jsonify({"error": f"Unsupported camera: {camera}"}), 404

        payload = session.get_preview_bytes(camera)  # type: ignore[arg-type]
        if payload is None:
            return Response(status=204)
        return Response(payload, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/start")
    def api_start():
        body = request.get_json(silent=True) or {}
        try:
            return jsonify(session.start(body.get("prompt")))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/stop")
    def api_stop():
        try:
            return jsonify(session.stop())
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/reset")
    def api_reset():
        try:
            return jsonify(session.reset())
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/save_video")
    def api_save_video():
        body = request.get_json(silent=True) or {}
        try:
            return jsonify(session.save_video(body.get("filename")))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/discard_video")
    def api_discard_video():
        try:
            return jsonify(session.discard_video())
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "phase": session.snapshot()["phase"]})

    return app


def run(args: Args) -> None:
    session = InteractivePolicySession(args)
    app = create_app(session, args)

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
