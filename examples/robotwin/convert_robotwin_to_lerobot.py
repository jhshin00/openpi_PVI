"""
Convert processed RoboTwin HDF5 episodes into a local LeRobot dataset for openpi training.

Example:
    uv run examples/robotwin/convert_robotwin_to_lerobot.py \
        --input-dir data/robotwin_processed/beat_block_hammer-demo_clean \
        --repo-id robotwin/beat_block_hammer_demo_clean

Multi-task example:
    uv run examples/robotwin/convert_robotwin_to_lerobot.py \
        --input-dir data/robotwin_training/demo_clean \
        --repo-id robotwin/beat_block_hammer_demo_clean
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro
import json


MOTORS = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]

CAMERAS = [
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
]


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


@dataclasses.dataclass(frozen=True)
class Args:
    # Directory containing either:
    # 1) a single processed RoboTwin task directory with `episode_<n>/...`
    # 2) a RoboTwin multi-task training root with nested task folders
    input_dir: Path
    repo_id: str
    # Deprecated alias for `input_dir`. Kept to avoid breaking single-task usage added earlier.
    processed_dir: Path | None = None
    output_root: Path = Path("./datasets")
    robot_type: str = "aloha"
    mode: Literal["video", "image"] = "image"
    fps: int = 50
    overwrite: bool = True
    instruction_strategy: Literal["random", "first"] = "random"
    seed: int = 0
    episode_indices: list[int] | None = None
    dataset_config: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)


def _create_empty_dataset(args: Args) -> LeRobotDataset:
    dataset_path = args.output_root / args.repo_id
    if dataset_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset already exists: {dataset_path}")
        shutil.rmtree(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": args.mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=dataset_path,
        robot_type=args.robot_type,
        features=features,
        use_videos=args.dataset_config.use_videos,
        tolerance_s=args.dataset_config.tolerance_s,
        image_writer_processes=args.dataset_config.image_writer_processes,
        image_writer_threads=args.dataset_config.image_writer_threads,
        video_backend=args.dataset_config.video_backend,
    )


def _load_images(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    images = {}
    for camera in cameras:
        camera_dataset = ep[f"/observations/images/{camera}"]
        if camera_dataset.ndim == 4:
            images[camera] = camera_dataset[:]
            continue

        decoded_images = []
        for data in camera_dataset:
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to decode {camera} image in processed RoboTwin dataset.")
            decoded_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        images[camera] = np.asarray(decoded_images)
    return images


def _load_episode(
    episode_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
    with h5py.File(episode_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])
        images = _load_images(ep, CAMERAS)
    return images, state, action


def _load_instruction(instruction_path: Path, *, rng: np.random.Generator, strategy: str) -> str:
    with instruction_path.open("r", encoding="utf-8") as f:
        instruction_data = json.load(f)

    instructions = instruction_data.get("instructions", [])
    if not instructions:
        raise ValueError(f"No instructions found in {instruction_path}")
    if strategy == "first":
        return str(instructions[0])
    return str(rng.choice(instructions))


def _resolve_input_dir(args: Args) -> Path:
    if args.processed_dir is not None:
        return args.processed_dir.resolve()
    return args.input_dir.resolve()


def _discover_episode_dirs(input_dir: Path) -> list[Path]:
    episode_dirs = set()
    for episode_path in input_dir.rglob("episode_*.hdf5"):
        if episode_path.parent.name.startswith("episode_") and (episode_path.parent / "instructions.json").exists():
            episode_dirs.add(episode_path.parent)
    return sorted(episode_dirs)


def main(args: Args) -> None:
    input_dir = _resolve_input_dir(args)
    output_root = args.output_root.resolve()

    episode_dirs = _discover_episode_dirs(input_dir)
    if args.episode_indices is not None:
        requested = set(args.episode_indices)
        episode_dirs = [
            episode_dir for episode_dir in episode_dirs if int(episode_dir.name.removeprefix("episode_")) in requested
        ]
    if not episode_dirs:
        raise ValueError(f"No processed RoboTwin episodes found in {input_dir}")

    dataset = _create_empty_dataset(dataclasses.replace(args, output_root=output_root))
    rng = np.random.default_rng(args.seed)

    for episode_dir in tqdm.tqdm(episode_dirs, desc="Converting RoboTwin episodes"):
        episode_index = episode_dir.name.removeprefix("episode_")
        episode_path = episode_dir / f"episode_{episode_index}.hdf5"
        instruction_path = episode_dir / "instructions.json"

        images, state, action = _load_episode(episode_path)
        instruction = _load_instruction(instruction_path, rng=rng, strategy=args.instruction_strategy)

        for frame_index in range(state.shape[0]):
            frame = {
                "observation.state": state[frame_index],
                "action": action[frame_index],
                "task": instruction,
            }
            for camera_name, image_array in images.items():
                frame[f"observation.images.{camera_name}"] = image_array[frame_index]
            dataset.add_frame(frame)

        dataset.save_episode()

    print(f"Saved LeRobot dataset to {output_root / args.repo_id}")


if __name__ == "__main__":
    main(tyro.cli(Args))
