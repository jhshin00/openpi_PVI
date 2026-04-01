"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, install the repo's RLDS dependency group:
`uv sync --group rlds`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to <output_root>/<repo_id>.
Running this conversion script will take approximately 30 minutes.
"""

import importlib.metadata
import pathlib
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

DEFAULT_REPO_ID = "physical-intelligence/libero"
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def _validate_raw_data_dir(data_dir: pathlib.Path) -> None:
    if not data_dir.exists():
        raise FileNotFoundError(f"Raw LIBERO data directory does not exist: {data_dir}")

    missing_builders = [name for name in RAW_DATASET_NAMES if not (data_dir / name).exists()]
    if missing_builders:
        available_dirs = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
        raise FileNotFoundError(
            "Raw LIBERO TFDS builders were not found under the provided data_dir.\n"
            f"Provided data_dir: {data_dir}\n"
            f"Missing builders: {missing_builders}\n"
            f"Available directories: {available_dirs}"
        )


def _load_raw_dataset(raw_dataset_name: str, data_dir: str):
    try:
        return tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
    except ImportError as exc:
        tensorflow_version = None
        tensorflow_cpu_version = None
        protobuf_version = None
        try:
            tensorflow_version = importlib.metadata.version("tensorflow")
        except importlib.metadata.PackageNotFoundError:
            pass
        try:
            tensorflow_cpu_version = importlib.metadata.version("tensorflow-cpu")
        except importlib.metadata.PackageNotFoundError:
            pass
        try:
            protobuf_version = importlib.metadata.version("protobuf")
        except importlib.metadata.PackageNotFoundError:
            pass

        error_text = str(exc)
        if "runtime_version" in str(exc) and "google.protobuf" in str(exc):
            raise RuntimeError(
                "TensorFlow and protobuf versions in the current virtualenv are incompatible.\n"
                f"Detected tensorflow={tensorflow_version!r}, tensorflow-cpu={tensorflow_cpu_version!r}, "
                f"protobuf={protobuf_version!r}.\n"
                "This repo expects the pinned RLDS stack from pyproject.toml.\n"
                "Fix it with:\n"
                "  uv pip uninstall tensorflow\n"
                "  uv sync --group rlds\n"
                "If you previously installed `tensorflow`, remove it or let `uv sync` replace it with "
                "`tensorflow-cpu==2.15.0`."
            ) from exc
        if "libtensorflow_cc.so.2" in error_text or "undefined symbol" in error_text:
            raise RuntimeError(
                "TensorFlow binary import failed, which usually means incompatible TensorFlow wheels are mixed in "
                "the same virtualenv.\n"
                f"Detected tensorflow={tensorflow_version!r}, tensorflow-cpu={tensorflow_cpu_version!r}, "
                f"protobuf={protobuf_version!r}.\n"
                "This repo should use only the pinned CPU stack for RLDS conversion.\n"
                "Fix it with:\n"
                "  uv pip uninstall tensorflow tensorflow-cpu tensorflow-estimator keras tensorflow-io-gcs-filesystem\n"
                "  uv sync --group rlds\n"
                "Then rerun the conversion script."
            ) from exc
        raise


def main(
    data_dir: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    output_root: str | None = None,
    push_to_hub: bool = False,
    overwrite: bool = False,
):
    data_dir_path = pathlib.Path(data_dir).expanduser()
    _validate_raw_data_dir(data_dir_path)

    output_base = pathlib.Path(output_root).expanduser() if output_root is not None else HF_LEROBOT_HOME
    output_path = output_base / repo_id

    # Clean up any existing dataset in the output directory
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_path} already exists. Pass --overwrite to replace it or change --repo-id/--output-root."
            )
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = _load_raw_dataset(raw_dataset_name, str(data_dir_path))
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": step["language_instruction"].decode(),
                    }
                )
            dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
