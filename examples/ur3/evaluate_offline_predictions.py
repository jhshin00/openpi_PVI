import dataclasses
import json
import pathlib
from typing import Any

import numpy as np
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


@dataclasses.dataclass
class Args:
    policy_config: str
    checkpoint_dir: str
    pytorch_device: str | None = "cpu"
    episodes_per_task: int = 1
    pairs_per_episode: int = 2
    pairs_per_task: int | None = None


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


def _load_episode_meta(dataset_root: pathlib.Path) -> list[dict[str, Any]]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    episodes = []
    for line in episodes_path.read_text().splitlines():
        if line.strip():
            episodes.append(json.loads(line))
    return episodes


def _select_candidate_positions(window_count: int, num_samples: int) -> list[int]:
    if window_count <= 0 or num_samples <= 0:
        return []
    positions = np.linspace(0, window_count - 1, num=min(num_samples, window_count))
    return [int(pos) for pos in np.unique(np.round(positions).astype(int))]


def _select_eval_pair_starts(
    episodes: list[dict[str, Any]],
    action_horizon: int,
    episodes_per_task: int,
    pairs_per_episode: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    starts = []
    selected = []
    cumulative = 0
    episode_entries_by_task: dict[str, list[dict[str, Any]]] = {}

    for episode in episodes:
        task = episode["tasks"][0]
        length = int(episode["length"])
        episode_entries_by_task.setdefault(task, []).append(
            {
                "episode": episode,
                "length": length,
                "global_start": cumulative,
            }
        )
        cumulative += length

    for task, task_entries in episode_entries_by_task.items():
        valid_entries = [entry for entry in task_entries if entry["length"] - action_horizon > 0]
        selected_episode_positions = _select_candidate_positions(len(valid_entries), episodes_per_task)

        for task_episode_position in selected_episode_positions:
            entry = valid_entries[task_episode_position]
            episode = entry["episode"]
            valid_pair_count = entry["length"] - action_horizon
            candidate_positions = _select_candidate_positions(valid_pair_count, pairs_per_episode)

            starts.extend(entry["global_start"] + pos for pos in candidate_positions)
            selected.append(
                {
                    "episode_index": int(episode["episode_index"]),
                    "task": task,
                    "length": entry["length"],
                    "task_episode_position": int(task_episode_position),
                    "pair_local_indices": candidate_positions,
                }
            )

    return starts, selected


def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main(args: Args) -> None:
    if args.episodes_per_task < 1:
        raise ValueError("--episodes-per-task must be >= 1")

    # Keep the old flag working for existing commands.
    resolved_pairs_per_episode = args.pairs_per_task if args.pairs_per_task is not None else args.pairs_per_episode
    if resolved_pairs_per_episode < 1:
        raise ValueError("--pairs-per-episode must be >= 1")

    train_config = _config.get_config(args.policy_config)
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        pytorch_device=args.pytorch_device,
    )

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = _data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    dataset_root = pathlib.Path(data_config.lerobot_root) / data_config.repo_id
    episodes = _load_episode_meta(dataset_root)

    pair_starts, selected_episodes = _select_eval_pair_starts(
        episodes,
        train_config.model.action_horizon,
        args.episodes_per_task,
        resolved_pairs_per_episode,
    )
    eval_indices = sorted({idx for base in pair_starts for idx in (base, base + 1)})

    predictions: dict[int, np.ndarray] = {}
    ground_truth: dict[int, np.ndarray] = {}
    infer_times_ms: list[float] = []

    for index in eval_indices:
        item = dataset[index]
        obs = _to_policy_observation(item, item.get("prompt"))
        result = policy.infer(obs)
        predictions[index] = np.asarray(result["actions"])
        ground_truth[index] = _to_numpy(item["actions"])
        infer_times_ms.append(float(result["policy_timing"]["infer_ms"]))

    first_joint_errors = []
    first_gripper_errors = []
    chunk_joint_errors = []
    chunk_gripper_errors = []
    overlap_joint_errors = []
    overlap_gripper_errors = []

    for index in eval_indices:
        pred = predictions[index]
        gt = ground_truth[index]
        first_joint_errors.append(np.abs(pred[0, :6] - gt[0, :6]))
        first_gripper_errors.append(np.abs(pred[0, 6] - gt[0, 6]))
        chunk_joint_errors.append(np.abs(pred[:, :6] - gt[:, :6]))
        chunk_gripper_errors.append(np.abs(pred[:, 6] - gt[:, 6]))

    for base in pair_starts:
        pred_now = predictions[base]
        pred_next = predictions[base + 1]
        overlap_joint_errors.append(np.abs(pred_now[1:, :6] - pred_next[:-1, :6]))
        overlap_gripper_errors.append(np.abs(pred_now[1:, 6] - pred_next[:-1, 6]))

    summary = {
        "policy_config": args.policy_config,
        "checkpoint_dir": str(pathlib.Path(args.checkpoint_dir).resolve()),
        "dataset_repo_id": data_config.repo_id,
        "action_horizon": train_config.model.action_horizon,
        "episodes_per_task": args.episodes_per_task,
        "pairs_per_episode": resolved_pairs_per_episode,
        "num_eval_windows": len(eval_indices),
        "num_overlap_pairs": len(pair_starts),
        "selected_episodes": selected_episodes,
        "metrics": {
            "first_step_joint_mae": float(np.mean(np.concatenate(first_joint_errors))),
            "first_step_gripper_mae": float(np.mean(np.asarray(first_gripper_errors))),
            "chunk_joint_mae": float(np.mean(np.concatenate(chunk_joint_errors))),
            "chunk_gripper_mae": float(np.mean(np.concatenate(chunk_gripper_errors))),
            "overlap_joint_mae": float(np.mean(np.concatenate(overlap_joint_errors))),
            "overlap_gripper_mae": float(np.mean(np.concatenate(overlap_gripper_errors))),
            "infer_ms_mean": float(np.mean(infer_times_ms)),
            "infer_ms_p95": float(np.quantile(np.asarray(infer_times_ms), 0.95)),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
