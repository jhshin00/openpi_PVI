"""Convert UR3 demonstrations to LeRobot format.

Supports two raw layouts:

1. The current `run_env_ik.py` output:

   <traj_dir>/data.hdf5
   |- data
      |- joint_positions      # (T, 7)
      |- joint_actions        # (T, 7)
      |- base_rgb             # (T, H, W, C)
      |- wrist_rgb            # (T, H, W, C)
      |- joint_velocities     # optional
      |- effort               # optional
      |- task                 # optional

2. The older "episode_*.hdf5" layout:

   episode_000000.hdf5
   |- observation
   |  |- qpos
   |  |- qvel                # optional
   |  |- effort              # optional
   |  |- image
   |     |- base_image
   |     |- wrist_image
   |- action
   |- task                   # optional

Typical usage with `run_env_ik.py` data:

uv run examples/ur3/convert_ur3_data_to_lerobot.py \
  --raw-dir ./datasets/ur3_raw \
  --repo-id ur3_dataset \
  --root ./datasets \
  --fps 30

If the raw episodes are stored as:

./datasets/ur3/pick_and_place/pick_up_the_pear/0403_153000/data.hdf5

the converter will infer the task prompt as:

pick up the pear
"""

import dataclasses
import json
from pathlib import Path
import re
import shutil
from typing import Any, Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro

EpisodeFormat = Literal["gello", "legacy"]
CAMERAS = ("base_image", "wrist_image")
LEGACY_CAMERA_KEYS = {
    "base_image": "/observation/image/base_image",
    "wrist_image": "/observation/image/wrist_image",
}
GELLO_CAMERA_KEYS = {
    "base_image": "/data/base_rgb",
    "wrist_image": "/data/wrist_rgb",
}


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    fps: int = 30
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def _decode_task(task_value: Any) -> str:
    if isinstance(task_value, bytes):
        return task_value.decode()
    if isinstance(task_value, np.bytes_):
        return task_value.decode()
    if isinstance(task_value, np.ndarray) and task_value.shape == ():
        return _decode_task(task_value.item())
    return str(task_value)


def _load_task_mapping(task_map_json: Path | None) -> dict[str, str]:
    if task_map_json is None:
        return {}

    mapping = json.loads(task_map_json.read_text())
    if not isinstance(mapping, dict):
        raise ValueError(f"Task mapping must be a JSON object, got: {type(mapping).__name__}")

    return {str(key): str(value) for key, value in mapping.items()}


def _find_hdf5_files(raw_dir: Path) -> list[Path]:
    if raw_dir.is_file():
        return [raw_dir]

    legacy_paths = sorted(raw_dir.glob("episode_*.hdf5"))
    if legacy_paths:
        return legacy_paths

    gello_paths = sorted(raw_dir.glob("**/data.hdf5"))
    if gello_paths:
        return gello_paths

    raise FileNotFoundError(f"No UR3 hdf5 episodes found under {raw_dir}")


def _detect_format(episode_path: Path) -> EpisodeFormat:
    with h5py.File(episode_path, "r") as episode:
        if "/data/joint_positions" in episode and "/data/joint_actions" in episode:
            return "gello"
        if "/observation/qpos" in episode and "/action" in episode:
            return "legacy"
    raise ValueError(f"Unsupported UR3 episode format: {episode_path}")


def _load_images(dataset: h5py.Dataset) -> np.ndarray:
    if dataset.ndim == 4:
        images = dataset[:]
    else:
        import cv2

        images = []
        for encoded_image in dataset:
            decoded = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
            images.append(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
        images = np.asarray(images)

    if images.ndim != 4:
        raise ValueError(f"Expected image tensor with shape (T, H, W, C), got {images.shape}")

    if images.shape[-1] != 3 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))

    return images


def _normalize_task_from_path_name(name: str) -> str:
    normalized = re.sub(r"[_-]+", " ", name.strip())
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _looks_like_metadata_dir(name: str) -> bool:
    lowered = name.lower()
    if lowered in {
        "success",
        "failure",
        "train",
        "val",
        "valid",
        "validation",
        "test",
        "raw",
        "dataset",
        "datasets",
        "demo",
        "demos",
        "recording",
        "recordings",
    }:
        return True

    timestamp_patterns = (
        r"\d{4}_\d{6}",
        r"\d{8}_\d{6}",
        r"\d{8}-\d{6}",
        r"\d{4}-\d{2}-\d{2}",
        r"\d{2}-\d{2}-\d{2}",
        r"\d{8}",
        r"\d{4}",
    )
    return any(re.fullmatch(pattern, name) for pattern in timestamp_patterns)


def _infer_task_from_path(episode_path: Path, raw_dir: Path) -> str | None:
    if episode_path.is_relative_to(raw_dir):
        candidate_parts = list(episode_path.relative_to(raw_dir).parts[:-1])
        raw_dir_name = raw_dir.name
    else:
        candidate_parts = list(episode_path.parts[:-1])
        raw_dir_name = None

    for part in reversed(candidate_parts):
        if _looks_like_metadata_dir(part):
            continue
        return _normalize_task_from_path_name(part)

    if raw_dir_name is not None and not _looks_like_metadata_dir(raw_dir_name):
        return _normalize_task_from_path_name(raw_dir_name)

    return None


def _resolve_task(
    episode: h5py.File,
    episode_path: Path,
    raw_dir: Path,
    episode_format: EpisodeFormat,
    *,
    default_task: str | None,
    task_mapping: dict[str, str],
) -> str:
    task_keys = ["/task"] if episode_format == "legacy" else ["/data/task", "/task"]
    for task_key in task_keys:
        if task_key in episode:
            return _decode_task(episode[task_key][()])

    relative_path = episode_path.relative_to(raw_dir).as_posix() if episode_path.is_relative_to(raw_dir) else episode_path.name
    lookup_candidates = (
        relative_path,
        episode_path.name,
        episode_path.parent.name,
        episode_path.parent.relative_to(raw_dir).as_posix() if episode_path.parent.is_relative_to(raw_dir) else None,
    )
    for candidate in lookup_candidates:
        if candidate is not None and candidate in task_mapping:
            return task_mapping[candidate]

    if (path_task := _infer_task_from_path(episode_path, raw_dir)) is not None:
        return path_task

    if default_task is not None:
        return default_task

    raise ValueError(
        "No task prompt was found in the raw episode, task map, or path structure. "
        "Pass --default-task or --task-map-json."
    )


def _has_feature(hdf5_files: list[Path], episode_format: EpisodeFormat, key: str) -> bool:
    feature_keys = {
        "velocity": {
            "gello": ("/data/joint_velocities", "/data/qvel"),
            "legacy": ("/observation/qvel",),
        },
        "effort": {
            "gello": ("/data/effort",),
            "legacy": ("/observation/effort",),
        },
    }
    with h5py.File(hdf5_files[0], "r") as episode:
        return any(path in episode for path in feature_keys[key][episode_format])


def _peek_image_shapes(episode_path: Path, episode_format: EpisodeFormat) -> dict[str, tuple[int, int, int]]:
    image_keys = LEGACY_CAMERA_KEYS if episode_format == "legacy" else GELLO_CAMERA_KEYS
    shapes = {}
    with h5py.File(episode_path, "r") as episode:
        for camera_name, key in image_keys.items():
            if key not in episode:
                raise KeyError(f"Missing required camera stream {key} in {episode_path}")
            images = _load_images(episode[key])
            shapes[camera_name] = tuple(images[0].shape)
    return shapes


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    root: str | Path | None = None,
    mode: Literal["video", "image"] = "video",
    *,
    image_shapes: dict[str, tuple[int, int, int]],
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    features = {
        "state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["actions"],
        },
    }

    if has_velocity:
        features["velocity"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": ["velocity"],
        }

    if has_effort:
        features["effort"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": ["effort"],
        }

    for camera_name, image_shape in image_shapes.items():
        features[camera_name] = {
            "dtype": mode,
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }

    target_dir = (Path(root) if root is not None else Path(HF_LEROBOT_HOME)) / repo_id
    if target_dir.exists():
        shutil.rmtree(target_dir)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=dataset_config.fps,
        root=target_dir,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_raw_episode_data(
    episode_path: Path,
    raw_dir: Path,
    episode_format: EpisodeFormat,
    *,
    default_task: str | None,
    task_mapping: dict[str, str],
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, str, torch.Tensor | None, torch.Tensor | None]:
    image_keys = LEGACY_CAMERA_KEYS if episode_format == "legacy" else GELLO_CAMERA_KEYS

    with h5py.File(episode_path, "r") as episode:
        if episode_format == "gello":
            state = torch.from_numpy(episode["/data/joint_positions"][:].astype(np.float32))
            actions = torch.from_numpy(episode["/data/joint_actions"][:].astype(np.float32))

            velocity = None
            for key in ("/data/joint_velocities", "/data/qvel"):
                if key in episode:
                    velocity = torch.from_numpy(episode[key][:].astype(np.float32))
                    break

            effort = None
            if "/data/effort" in episode:
                effort = torch.from_numpy(episode["/data/effort"][:].astype(np.float32))
        else:
            state = torch.from_numpy(episode["/observation/qpos"][:].astype(np.float32))
            actions = torch.from_numpy(episode["/action"][:].astype(np.float32))

            velocity = None
            if "/observation/qvel" in episode:
                velocity = torch.from_numpy(episode["/observation/qvel"][:].astype(np.float32))

            effort = None
            if "/observation/effort" in episode:
                effort = torch.from_numpy(episode["/observation/effort"][:].astype(np.float32))

        images_per_camera = {
            camera_name: _load_images(episode[key]) for camera_name, key in image_keys.items() if key in episode
        }
        missing_cameras = set(CAMERAS) - set(images_per_camera)
        if missing_cameras:
            raise KeyError(f"Missing required camera streams {sorted(missing_cameras)} in {episode_path}")
        task = _resolve_task(
            episode,
            episode_path,
            raw_dir,
            episode_format,
            default_task=default_task,
            task_mapping=task_mapping,
        )

    expected_frames = state.shape[0]
    if actions.shape[0] != expected_frames:
        raise ValueError(f"Mismatched state/action lengths in {episode_path}: {state.shape[0]} vs {actions.shape[0]}")
    for camera_name, images in images_per_camera.items():
        if images.shape[0] != expected_frames:
            raise ValueError(
                f"Mismatched image/frame lengths in {episode_path} for {camera_name}: {images.shape[0]} vs {expected_frames}"
            )

    return images_per_camera, state, actions, task, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    raw_dir: Path,
    episode_format: EpisodeFormat,
    *,
    default_task: str | None,
    task_mapping: dict[str, str],
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = list(range(len(hdf5_files)))

    for episode_index in tqdm.tqdm(episodes, desc="Converting UR3 episodes"):
        episode_path = hdf5_files[episode_index]
        images_per_camera, state, actions, task, velocity, effort = load_raw_episode_data(
            episode_path,
            raw_dir,
            episode_format,
            default_task=default_task,
            task_mapping=task_mapping,
        )

        for frame_index in range(state.shape[0]):
            frame = {
                "state": state[frame_index],
                "actions": actions[frame_index],
            }

            for camera_name, image_array in images_per_camera.items():
                frame[camera_name] = image_array[frame_index]

            if velocity is not None:
                frame["velocity"] = velocity[frame_index]
            if effort is not None:
                frame["effort"] = effort[frame_index]

            dataset.add_frame(frame)

        dataset.save_episode(task=task)

    return dataset


def main(
    raw_dir: Path,
    repo_id: str = "ur3_dataset",
    root: Path | None = None,
    raw_repo_id: str | None = None,
    default_task: str | None = None,
    task_map_json: Path | None = None,
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    raw_dir = raw_dir.expanduser()
    if not raw_dir.exists():
        if raw_repo_id is not None:
            raise ValueError(
                "raw_repo_id is not supported with the installed lerobot version. Download the raw UR3 data locally first."
            )
        raise FileNotFoundError(f"Raw UR3 directory does not exist: {raw_dir}")

    hdf5_files = _find_hdf5_files(raw_dir)
    episode_format = _detect_format(hdf5_files[0])
    task_mapping = _load_task_mapping(task_map_json)
    image_shapes = _peek_image_shapes(hdf5_files[0], episode_format)

    dataset = create_empty_dataset(
        repo_id,
        root=root,
        robot_type="UR3",
        mode=mode,
        image_shapes=image_shapes,
        has_velocity=_has_feature(hdf5_files, episode_format, "velocity"),
        has_effort=_has_feature(hdf5_files, episode_format, "effort"),
        dataset_config=dataset_config,
    )
    populate_dataset(
        dataset,
        hdf5_files,
        raw_dir,
        episode_format,
        default_task=default_task,
        task_mapping=task_mapping,
        episodes=episodes,
    )
    dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(main)
