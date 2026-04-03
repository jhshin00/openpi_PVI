import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro

"""
hdf5 dataset format

episode_000000.hdf5
├── observation
│   ├── qpos                    # (n_frames, 7)
│   ├── qvel                    # (n_frames, 7)
│   ├── effort                  # (n_frames, 7)
│   └── image
|       ├── base_image          # (n_frames, H, W, C)
|       └── wrist_image         # (n_frames, H, W, C)
├── action                      # (n_frames, 7)
└── task                        # str

usage
uv run convert_ur3_data_to_lerobot.py --raw_dir='./datasets/ur3_raw' --repo_id=ur3_lerobot --root='./datasets'

output LeRobot dataset format
observation.state
observation.image.base_image
observation.image.wrist_image
observation.velocity
observation.effort
actions

TODO -> 아래것들은 다 자동저장이겠지?
task index
timestamp
frame_index
episode_index
index
-------------
task는 저장 안되는듯

"""


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    root: str | Path | None = None,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    cameras = ["base_image", "wrist_image"]
    features = {
        "observation/state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["actions"],
        },
    }

    if has_velocity:
        features["observation/velocity"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": ["velocity"],
        }

    if has_effort:
        features["observation/effort"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": ["effort"],
        }

    for cam in cameras:
        features[f"observation/{cam}"] = {
            "dtype": mode,
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        }

    target_dir = (Path(root) if root else Path(LEROBOT_HOME)) / repo_id
    if target_dir.exists():
        shutil.rmtree(target_dir)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        root=target_dir,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        return [key for key in ep["/observation/image"].keys() if "depth" not in key]


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observation/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observation/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for cam in cameras:
        uncompressed = ep[f"/observation/image/{cam}"].ndim == 4

        if uncompressed:
            imgs_array = ep[f"/observation/image/{cam}"][:]
        else:
            import cv2

            imgs_array = []
            for data in ep[f"/observation/image/{cam}"]:
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BAYER_BGGR2RGB))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[cam] = imgs_array

    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, str, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observation/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])
        task = ep["/task"]

        velocity = None
        if "/observation/qvel" in ep:
            velocity = torch.from_numpy(ep["/observation/qvel"][:])

        effort = None
        if "/observation/effort" in ep:
            effort = torch.from_numpy(ep["/observation/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            ["base_image", "wrist_image"],
        )

    return imgs_per_cam, state, action, task, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, task, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation/state": state[i],
                "actions": action[i],
            }

            for cam, img_array in imgs_per_cam.items():
                frame[f"observation/{cam}"] = img_array[i]

            if velocity is not None:
                frame["observation/velocity"] = velocity[i]

            if effort is not None:
                frame["observation/effort"] = effort[i]

            dataset.add_frame(frame)

        dataset.save_episode(task=task.decode())
        # dataset.save_episode(task=task)

    return dataset


def main(
    raw_dir: Path,
    repo_id: str,
    root: Path | None = None,
    raw_repo_id: str | None = None,
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        download_raw(raw_dir, repo_id=raw_repo_id)

    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))

    dataset = create_empty_dataset(
        repo_id,
        root=root,
        robot_type="UR3",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        episodes=episodes,
    )
    dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(main)
