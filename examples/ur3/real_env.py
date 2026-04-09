from __future__ import annotations

import dataclasses
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

DEFAULT_GELLO_ROOT = Path(__file__).resolve().parents[2] / "_external" / "gello_software"


def _ensure_gello_root(gello_root: str | Path) -> Path:
    root = Path(gello_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"gello root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


class _MockURRobot:
    def __init__(self) -> None:
        self._joint_state = np.zeros(7, dtype=np.float32)

    def num_dofs(self) -> int:
        return 7

    def get_joint_state(self) -> np.ndarray:
        return self._joint_state.copy()

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        self._joint_state = np.asarray(joint_state, dtype=np.float32).copy()

    def command_joint_velocity(self, joint_velocity: np.ndarray, a: float, t: float, gripper: float | None = None) -> None:
        joint_velocity = np.asarray(joint_velocity, dtype=np.float32)
        self._joint_state[:6] = self._joint_state[:6] + joint_velocity * float(t)
        if gripper is not None:
            self._joint_state[6] = float(gripper)

    def get_observations(self) -> dict[str, np.ndarray]:
        return {"joint_positions": self.get_joint_state()}


class _RealSenseDriver:
    def __init__(self, serial: str, *, width: int, height: int, fps: int) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ModuleNotFoundError("pyrealsense2 is required for real UR3 evaluation.") from exc

        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._pipeline.start(config)

    def read(self) -> np.ndarray | None:
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if color is None:
            return None
        return np.asanyarray(color.get_data())

    def close(self) -> None:
        try:
            self._pipeline.stop()
        except RuntimeError:
            pass


class _AsyncCamera:
    def __init__(
        self,
        driver: _RealSenseDriver,
        *,
        target_size: tuple[int, int],
        crop_box: tuple[int, int, int, int] | None,
        crop_center: tuple[int, int] | None,
        crop_size: tuple[int, int],
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise ModuleNotFoundError("opencv-python is required for real UR3 evaluation.") from exc

        self._cv2 = cv2
        self._driver = driver
        self._target_size = target_size
        self._crop_box = crop_box
        self._crop_center = crop_center
        self._crop_size = crop_size
        self._frame: np.ndarray | None = None
        self._uncropped_frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def _crop(self, frame: np.ndarray) -> np.ndarray:
        if self._crop_box is not None:
            # Keep compatibility with the data-collection script's (x1, x2, y1, y2) crop_box convention.
            x1, x2, y1, y2 = self._crop_box
            x1 = max(int(x1), 0)
            x2 = min(int(x2), frame.shape[1])
            y1 = max(int(y1), 0)
            y2 = min(int(y2), frame.shape[0])
            if x1 >= x2 or y1 >= y2:
                raise ValueError(f"Invalid crop_box {self._crop_box} for frame shape {frame.shape}")
            return frame[y1:y2, x1:x2]

        if self._crop_center is None:
            return frame

        crop_h, crop_w = self._crop_size
        center_y, center_x = self._crop_center
        y1 = max(center_y - crop_h // 2, 0)
        y2 = min(center_y + crop_h // 2, frame.shape[0])
        x1 = max(center_x - crop_w // 2, 0)
        x2 = min(center_x + crop_w // 2, frame.shape[1])
        return frame[y1:y2, x1:x2]

    def _update_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._driver.read()
            except Exception:
                if self._stop_event.is_set():
                    break
                raise
            if frame is None:
                continue

            uncropped_frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            frame = self._crop(frame)
            frame = self._cv2.resize(frame, self._target_size, interpolation=self._cv2.INTER_LINEAR)
            frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

            with self._lock:
                self._frame = frame
                self._uncropped_frame = uncropped_frame

    def read(self) -> np.ndarray | None:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def read_pair(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            uncropped_frame = None if self._uncropped_frame is None else self._uncropped_frame.copy()
            return frame, uncropped_frame

    def close(self) -> None:
        self._stop_event.set()
        self._driver.close()
        self._thread.join(timeout=1.0)


def _discover_camera_serials(base_serial: str | None, wrist_serial: str | None) -> tuple[str, str]:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise ModuleNotFoundError("pyrealsense2 is required for real UR3 evaluation.") from exc

    context = rs.context()
    available_serials = [device.get_info(rs.camera_info.serial_number) for device in context.query_devices()]
    if len(available_serials) < 2 and (base_serial is None or wrist_serial is None):
        raise RuntimeError(
            "Could not auto-discover two RealSense cameras. Pass both --base-camera-serial and --wrist-camera-serial."
        )

    resolved_base = base_serial or available_serials[0]
    if wrist_serial is not None:
        resolved_wrist = wrist_serial
    else:
        remaining = [serial for serial in available_serials if serial != resolved_base]
        if not remaining:
            raise RuntimeError(
                "Only one RealSense camera was discovered. Pass a second serial or run with --mock for dry-runs."
            )
        resolved_wrist = remaining[0]

    if resolved_base == resolved_wrist:
        raise ValueError("base and wrist camera serials must be different")

    logging.info(
        "Resolved RealSense serials: base=%s wrist=%s available=%s",
        resolved_base,
        resolved_wrist,
        available_serials,
    )

    return resolved_base, resolved_wrist


@dataclasses.dataclass
class OpenPIUR3Env:
    gello_root: str = str(DEFAULT_GELLO_ROOT)
    robot_mode: str = "direct"
    robot_ip: str = "192.168.5.102"
    hostname: str = "127.0.0.1"
    robot_port: int = 6001
    hz: int = 30
    prompt: str | None = None
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
    kp: float = 8.0
    deadband: float = 0.003
    max_joint_velocity: float = 1.5
    max_joint_accel: float = 10.0
    speedj_accel: float = 10.0
    target_smoothing_alpha: float = 1.0
    reset_joints_deg: tuple[float, ...] | None = (0.0, -90.0, -90.0, -90.0, 90.0, 90.0)
    reset_gripper: float = 0.0
    reset_steps: int = 120
    reset_max_delta: float = 0.05
    camera_warmup_sec: float = 5.0
    mock: bool = False

    def __post_init__(self) -> None:
        if self.hz <= 0:
            raise ValueError("hz must be positive")
        if self.reset_joints_deg is not None and len(self.reset_joints_deg) != 6:
            raise ValueError("reset_joints_deg must contain exactly 6 joint angles")
        if not (0.0 < self.target_smoothing_alpha <= 1.0):
            raise ValueError("target_smoothing_alpha must be in the interval (0, 1].")

        self._control_dt = 1.0 / float(self.hz)
        self._last_qd = np.zeros(6, dtype=np.float32)
        self._last_target_action: np.ndarray | None = None
        self._next_step_time: float | None = None
        self._zero_image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        self._zero_uncropped_image = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        self._cameras: dict[str, _AsyncCamera] = {}

        if self.mock:
            self._robot = _MockURRobot()
        else:
            _ensure_gello_root(self.gello_root)
            self._robot = self._create_robot()
            self._cameras = self._create_cameras()
            self._wait_for_cameras()

        if self.reset_joints_deg is not None:
            self._reset_target = np.asarray(
                [*np.deg2rad(np.asarray(self.reset_joints_deg, dtype=np.float32)), self.reset_gripper],
                dtype=np.float32,
            )
        else:
            self._reset_target = None

    def _create_robot(self):
        if self.robot_mode == "direct":
            from gello.robots.ur import URRobot

            return URRobot(robot_ip=self.robot_ip)
        if self.robot_mode == "zmq":
            from gello.zmq_core.robot_node import ZMQClientRobot

            return ZMQClientRobot(port=self.robot_port, host=self.hostname)
        raise ValueError(f"Unsupported robot_mode: {self.robot_mode}")

    def _create_cameras(self) -> dict[str, _AsyncCamera]:
        base_serial, wrist_serial = _discover_camera_serials(self.base_camera_serial, self.wrist_camera_serial)
        target_size = (self.image_size, self.image_size)
        logging.info(
            "Resolved camera crops: base_crop_box=%s base_crop_center=%s wrist_crop_box=%s wrist_crop_center=%s",
            self.base_crop_box,
            self.base_crop_center,
            self.wrist_crop_box,
            self.wrist_crop_center,
        )

        return {
            "base": _AsyncCamera(
                _RealSenseDriver(
                    base_serial,
                    width=self.camera_width,
                    height=self.camera_height,
                    fps=self.camera_fps,
                ),
                target_size=target_size,
                crop_box=self.base_crop_box,
                crop_center=self.base_crop_center,
                crop_size=self.crop_size,
            ),
            "wrist": _AsyncCamera(
                _RealSenseDriver(
                    wrist_serial,
                    width=self.camera_width,
                    height=self.camera_height,
                    fps=self.camera_fps,
                ),
                target_size=target_size,
                crop_box=self.wrist_crop_box,
                crop_center=self.wrist_crop_center,
                crop_size=self.crop_size,
            ),
        }

    def _wait_for_cameras(self) -> None:
        deadline = time.time() + self.camera_warmup_sec
        while time.time() < deadline:
            if all(camera.read() is not None for camera in self._cameras.values()):
                return
            time.sleep(0.05)
        raise TimeoutError("Timed out waiting for RealSense frames.")

    def _sleep_to_rate(self) -> None:
        now = time.perf_counter()
        if self._next_step_time is None:
            self._next_step_time = now + self._control_dt
            return

        sleep_time = self._next_step_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)
            self._next_step_time += self._control_dt
        else:
            self._next_step_time = now + self._control_dt

    def _read_raw_obs(self) -> dict[str, np.ndarray]:
        observations = {"joint_positions": np.asarray(self._robot.get_observations()["joint_positions"], dtype=np.float32)}

        for name, camera in self._cameras.items():
            frame, uncropped_frame = camera.read_pair()
            if frame is not None:
                observations[f"{name}_rgb"] = frame
            if uncropped_frame is not None:
                observations[f"{name}_rgb_uncropped"] = uncropped_frame

        return observations

    def _format_obs(self, raw_obs: dict[str, np.ndarray]) -> dict[str, np.ndarray | str]:
        obs = {
            "state": raw_obs["joint_positions"],
            "base_image": np.asarray(raw_obs.get("base_rgb", self._zero_image)),
            "base_image_uncropped": np.asarray(raw_obs.get("base_rgb_uncropped", self._zero_uncropped_image)),
            "wrist_image": np.asarray(raw_obs.get("wrist_rgb", self._zero_image)),
            "wrist_image_uncropped": np.asarray(raw_obs.get("wrist_rgb_uncropped", self._zero_uncropped_image)),
        }
        if self.prompt is not None:
            obs["prompt"] = self.prompt
        return obs

    def _move_to_reset(self) -> None:
        assert self._reset_target is not None

        current = self._read_raw_obs()["joint_positions"]
        for alpha in np.linspace(0.0, 1.0, self.reset_steps):
            command = (1.0 - alpha) * current + alpha * self._reset_target
            latest = self._read_raw_obs()["joint_positions"]
            delta = command - latest
            max_delta = np.max(np.abs(delta[:6]))
            if max_delta > self.reset_max_delta:
                command[:6] = latest[:6] + delta[:6] / max_delta * self.reset_max_delta
                command[6] = self._reset_target[6]

            self._robot.command_joint_state(command.astype(np.float32))
            self._sleep_to_rate()

    def reset(self):
        self._last_qd[:] = 0.0
        self._next_step_time = time.perf_counter() + self._control_dt

        if self._reset_target is not None:
            self._move_to_reset()

        raw_obs = self._read_raw_obs()
        self._last_target_action = raw_obs["joint_positions"].copy()
        obs = self._format_obs(raw_obs)
        info = {"prompt": self.prompt} if self.prompt is not None else {}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(f"Expected a 7D UR3 action, got shape {action.shape}")

        current = self._read_raw_obs()["joint_positions"]
        target_action = action.copy()
        if self.target_smoothing_alpha < 1.0:
            if self._last_target_action is None:
                self._last_target_action = current.copy()
            target_action[:6] = (
                (1.0 - self.target_smoothing_alpha) * self._last_target_action[:6]
                + self.target_smoothing_alpha * action[:6]
            )
            target_action[6] = action[6]
        self._last_target_action = target_action.copy()

        err = target_action[:6] - current[:6]
        err[np.abs(err) < self.deadband] = 0.0

        qd = self.kp * err
        qd = np.clip(qd, -self.max_joint_velocity, self.max_joint_velocity)

        accel_limit = self.max_joint_accel * self._control_dt
        qd_delta = np.clip(qd - self._last_qd, -accel_limit, accel_limit)
        qd = self._last_qd + qd_delta
        self._last_qd = qd

        self._robot.command_joint_velocity(qd, a=self.speedj_accel, t=self._control_dt, gripper=float(target_action[6]))
        self._sleep_to_rate()

        raw_obs = self._read_raw_obs()
        info = {
            "current_joint_positions": current.copy(),
            "requested_action": action.copy(),
            "target_action": target_action.copy(),
            "joint_error": err.copy(),
            "joint_velocity_cmd": qd.copy(),
        }
        return self._format_obs(raw_obs), 0.0, False, info

    def close(self) -> None:
        for camera in self._cameras.values():
            camera.close()

        close_candidates = [
            getattr(self._robot, "close", None),
            getattr(getattr(self._robot, "robot", None), "close", None),
        ]
        for close_fn in close_candidates:
            if callable(close_fn):
                close_fn()
                break


def create_env(**kwargs) -> OpenPIUR3Env:
    return OpenPIUR3Env(**kwargs)
