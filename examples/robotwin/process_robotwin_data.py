"""
Convert RoboTwin-collected episodes into the Aloha-style HDF5 layout expected by openpi/LeRobot conversion.

Example:
    uv run examples/robotwin/process_robotwin_data.py \
        --input-dir third_party/robotwin/data/beat_block_hammer/demo_clean \
        --output-dir data/robotwin_processed/beat_block_hammer-demo_clean
"""

import dataclasses
from pathlib import Path

import cv2
import h5py
import numpy as np
import tqdm
import tyro
import json


CAMERA_NAME_MAP = {
    "head_camera": "cam_high",
    "right_camera": "cam_right_wrist",
    "left_camera": "cam_left_wrist",
}


@dataclasses.dataclass(frozen=True)
class Args:
    # RoboTwin directory that contains `data/episode*.hdf5` and `instructions/episode*.json`.
    input_dir: Path
    # Output directory that will contain `episode_<n>/episode_<n>.hdf5`.
    output_dir: Path
    # Optional explicit episode indices to process. If omitted, every episode in `input_dir/data` is used.
    episode_indices: list[int] | None = None
    # Optional cap on how many discovered episodes to process.
    limit: int | None = None
    # Which instruction split to keep from RoboTwin's generated descriptions.
    instruction_split: str = "seen"
    image_width: int = 640
    image_height: int = 480


def _episode_path(input_dir: Path, episode_index: int) -> Path:
    return input_dir / "data" / f"episode{episode_index}.hdf5"


def _instruction_path(input_dir: Path, episode_index: int) -> Path:
    return input_dir / "instructions" / f"episode{episode_index}.json"


def _discover_episode_indices(input_dir: Path) -> list[int]:
    data_dir = input_dir / "data"
    episode_indices = []
    for path in sorted(data_dir.glob("episode*.hdf5")):
        suffix = path.stem.removeprefix("episode")
        if suffix.isdigit():
            episode_indices.append(int(suffix))
    return episode_indices


def _select_instructions(instruction_path: Path, split: str) -> list[str]:
    with instruction_path.open("r", encoding="utf-8") as f:
        instruction_data = json.load(f)

    candidates = instruction_data.get(split)
    if candidates is None:
        candidates = instruction_data.get("instructions")
    if candidates is None:
        candidates = instruction_data.get("seen")
    if not candidates:
        raise ValueError(f"No instructions found in {instruction_path}")
    return [str(instruction) for instruction in candidates]


def _load_episode(episode_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(episode_path, "r") as root:
        left_gripper = root["/joint_action/left_gripper"][()]
        left_arm = root["/joint_action/left_arm"][()]
        right_gripper = root["/joint_action/right_gripper"][()]
        right_arm = root["/joint_action/right_arm"][()]
        images = {camera_name: root[f"/observation/{camera_name}/rgb"][()] for camera_name in CAMERA_NAME_MAP}

    return left_gripper, left_arm, right_gripper, right_arm, images


def _encode_images(images: list[np.ndarray]) -> tuple[list[bytes], int]:
    encoded_images: list[bytes] = []
    max_len = 0

    for image in images:
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError("Failed to JPEG-encode processed RoboTwin frame.")
        image_bytes = encoded.tobytes()
        encoded_images.append(image_bytes)
        max_len = max(max_len, len(image_bytes))

    padded = [image_bytes.ljust(max_len, b"\0") for image_bytes in encoded_images]
    return padded, max_len


def _decode_and_resize_frame(image_bytes: bytes, image_width: int, image_height: int) -> np.ndarray:
    if isinstance(image_bytes, np.ndarray) and image_bytes.ndim == 3:
        return cv2.resize(image_bytes, (image_width, image_height))
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode RoboTwin image bytes.")
    return cv2.resize(image, (image_width, image_height))


def _process_episode(
    input_dir: Path,
    output_dir: Path,
    episode_index: int,
    *,
    instruction_split: str,
    image_width: int,
    image_height: int,
) -> None:
    episode_path = _episode_path(input_dir, episode_index)
    instruction_path = _instruction_path(input_dir, episode_index)
    left_gripper, left_arm, right_gripper, right_arm, images = _load_episode(episode_path)
    instructions = _select_instructions(instruction_path, instruction_split)

    episode_output_dir = output_dir / f"episode_{episode_index}"
    episode_output_dir.mkdir(parents=True, exist_ok=True)

    with (episode_output_dir / "instructions.json").open("w", encoding="utf-8") as f:
        json.dump({"instructions": instructions}, f, indent=2)

    qpos = []
    actions = []
    camera_frames = {mapped_name: [] for mapped_name in CAMERA_NAME_MAP.values()}
    left_arm_dims = []
    right_arm_dims = []

    num_steps = int(left_gripper.shape[0])
    for step in range(num_steps):
        state = np.array(
            left_arm[step].tolist()
            + [left_gripper[step]]
            + right_arm[step].tolist()
            + [right_gripper[step]],
            dtype=np.float32,
        )

        if step != num_steps - 1:
            qpos.append(state)
            for raw_camera_name, mapped_camera_name in CAMERA_NAME_MAP.items():
                camera_frames[mapped_camera_name].append(
                    _decode_and_resize_frame(images[raw_camera_name][step], image_width, image_height)
                )

        if step != 0:
            actions.append(state)
            left_arm_dims.append(left_arm[step].shape[0])
            right_arm_dims.append(right_arm[step].shape[0])

    output_hdf5_path = episode_output_dir / f"episode_{episode_index}.hdf5"
    with h5py.File(output_hdf5_path, "w") as f:
        f.create_dataset("action", data=np.asarray(actions, dtype=np.float32))
        observations = f.create_group("observations")
        observations.create_dataset("qpos", data=np.asarray(qpos, dtype=np.float32))
        observations.create_dataset("left_arm_dim", data=np.asarray(left_arm_dims, dtype=np.int32))
        observations.create_dataset("right_arm_dim", data=np.asarray(right_arm_dims, dtype=np.int32))
        image_group = observations.create_group("images")

        for camera_name, frames in camera_frames.items():
            encoded_frames, max_len = _encode_images(frames)
            image_group.create_dataset(camera_name, data=encoded_frames, dtype=f"S{max_len}")


def main(args: Args) -> None:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not (input_dir / "data").is_dir():
        raise ValueError(f"{input_dir} must contain a data/ directory.")
    if not (input_dir / "instructions").is_dir():
        raise ValueError(f"{input_dir} must contain an instructions/ directory.")

    episode_indices = args.episode_indices or _discover_episode_indices(input_dir)
    if args.limit is not None:
        episode_indices = episode_indices[: args.limit]
    if not episode_indices:
        raise ValueError(f"No RoboTwin episodes found in {input_dir / 'data'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for episode_index in tqdm.tqdm(episode_indices, desc="Processing RoboTwin episodes"):
        _process_episode(
            input_dir,
            output_dir,
            episode_index,
            instruction_split=args.instruction_split,
            image_width=args.image_width,
            image_height=args.image_height,
        )

    print(f"Processed {len(episode_indices)} episodes into {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
