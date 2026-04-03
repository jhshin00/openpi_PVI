import collections
import dataclasses
import json
import logging
import math
import pathlib
import re
from typing import Any
from typing import Dict
from typing import Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.1.1"
    port: int = 8000
    policy_config: Optional[str] = None  # Optional local checkpoint config name for in-process evaluation.
    policy_dir: Optional[str] = None  # Optional local checkpoint directory for in-process evaluation.
    policy_pytorch_device: Optional[str] = None  # Optional device for local PyTorch policy, e.g. cuda:0.
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_object"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    task_category: Optional[str] = None  # Filter by classification category, e.g. "Camera Viewpoints"
    difficulty_level: Optional[int] = None  # Filter by difficulty level 1-5
    task_name_pattern: Optional[str] = None  # Optional regex filter on task names
    task_start_index: int = 0  # Optional offset into the filtered task list, useful for manual resume.
    task_limit: Optional[int] = None  # Optional cap after filtering
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 10  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/pi05_libero_pvi_dino_base_debug1_30000/libero_object/videos"  # Path to save videos
    save_videos: bool = True  # Disable for large suite-level evaluations.
    summary_json_path: Optional[str] = None  # Optional JSON summary output path.

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    _filter_task_suite(task_suite, args)
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    logging.info(f"Filtered tasks: {num_tasks_in_suite}")
    if num_tasks_in_suite == 0:
        raise ValueError("No tasks remain after applying the requested filters.")

    if args.save_videos:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name in {"libero_90", "libero_100", "libero_mix"}:
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _create_policy_client(args)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_classification = _load_task_classification()
    category_stats = collections.defaultdict(lambda: {"episodes": 0, "successes": 0})
    difficulty_stats = collections.defaultdict(lambda: {"episodes": 0, "successes": 0})
    task_stats = []
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)
        task_meta = _get_task_metadata(task_suite.name, task.name, task_classification)
        task_category = task_meta.get("category") or "Unclassified"
        task_difficulty = task_meta.get("difficulty_level")

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        try:
            # Start episodes
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                logging.info(f"\nTask: {task_description}")

                # Reset environment and policy state
                env.reset()
                client.reset()
                action_plan = collections.deque()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Setup
                t = 0
                done = False
                replay_images = []

                logging.info(f"Starting episode {task_episodes+1}...")
                while t < max_steps + args.num_steps_wait:
                    try:
                        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                        # and we need to wait for them to fall
                        if t < args.num_steps_wait:
                            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        # Get preprocessed image
                        # IMPORTANT: rotate 180 degrees to match train preprocessing
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                        )
                        wrist_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                        )

                        # Save preprocessed image for replay video
                        replay_images.append(img)

                        if not action_plan:
                            # Finished executing previous action chunk -- compute new chunk
                            # Prepare observations dict
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": np.concatenate(
                                    (
                                        obs["robot0_eef_pos"],
                                        _quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"],
                                    )
                                ),
                                "prompt": str(task_description),
                            }

                            # Query model to get action
                            action_chunk = client.infer(element)["actions"]
                            assert (
                                len(action_chunk) >= args.replan_steps
                            ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                            action_plan.extend(action_chunk[: args.replan_steps])

                        action = action_plan.popleft()

                        # Execute action in environment
                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        logging.error(f"Caught exception: {e}")
                        break

                task_episodes += 1
                total_episodes += 1
                category_stats[task_category]["episodes"] += 1
                if task_difficulty is not None:
                    difficulty_stats[task_difficulty]["episodes"] += 1
                if done:
                    category_stats[task_category]["successes"] += 1
                    if task_difficulty is not None:
                        difficulty_stats[task_difficulty]["successes"] += 1

                # Save a replay video of the episode
                if args.save_videos:
                    suffix = "success" if done else "failure"
                    task_segment = task_description.replace(" ", "_")
                    imageio.mimwrite(
                        pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                        [np.asarray(x) for x in replay_images],
                        fps=10,
                    )

                # Log current results
                logging.info(f"Success: {done}")
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            # Log final results
            logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
            task_stats.append(
                {
                    "task_name": task.name,
                    "task_description": task_description,
                    "category": task_category,
                    "difficulty_level": task_difficulty,
                    "episodes": task_episodes,
                    "successes": task_successes,
                    "success_rate": (float(task_successes) / float(task_episodes)) if task_episodes else 0.0,
                }
            )
        finally:
            env.close()

        _write_summary_if_requested(
            args,
            total_episodes=total_episodes,
            total_successes=total_successes,
            category_stats=category_stats,
            difficulty_stats=difficulty_stats,
            task_stats=task_stats,
            num_tasks_in_suite=num_tasks_in_suite,
        )

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info("Per-category success rates:")
    for category in sorted(category_stats):
        episodes = category_stats[category]["episodes"]
        successes = category_stats[category]["successes"]
        logging.info(
            "  %s: %d/%d (%.4f)",
            category,
            successes,
            episodes,
            float(successes) / float(episodes) if episodes else 0.0,
        )

    _write_summary_if_requested(
        args,
        total_episodes=total_episodes,
        total_successes=total_successes,
        category_stats=category_stats,
        difficulty_stats=difficulty_stats,
        task_stats=task_stats,
        num_tasks_in_suite=num_tasks_in_suite,
    )


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _create_policy_client(args: Args):
    using_local_policy = args.policy_config is not None or args.policy_dir is not None
    if using_local_policy and (args.policy_config is None or args.policy_dir is None):
        raise ValueError("Both policy_config and policy_dir must be set to run local in-process evaluation.")

    if not using_local_policy:
        return _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    try:
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Local policy loading requires the root openpi environment (Python >=3.11). "
            "Run this script with `uv run` and `PYTHONPATH=$PWD/third_party/libero-plus`."
        ) from e

    logging.info("Loading local policy from %s with config %s", args.policy_dir, args.policy_config)
    train_config = _config.get_config(args.policy_config)
    return _policy_config.create_trained_policy(
        train_config,
        args.policy_dir,
        pytorch_device=args.policy_pytorch_device,
    )


def _build_summary(
    args: Args,
    *,
    total_episodes: int,
    total_successes: int,
    category_stats,
    difficulty_stats,
    task_stats,
    num_tasks_in_suite: int,
) -> Dict[str, Any]:
    return {
        "suite_name": args.task_suite_name,
        "filters": {
            "task_category": args.task_category,
            "difficulty_level": args.difficulty_level,
            "task_name_pattern": args.task_name_pattern,
            "task_start_index": args.task_start_index,
            "task_limit": args.task_limit,
        },
        "policy": {
            "mode": "local" if args.policy_config is not None else "websocket",
            "host": args.host if args.policy_config is None else None,
            "port": args.port if args.policy_config is None else None,
            "policy_config": args.policy_config,
            "policy_dir": args.policy_dir,
            "policy_pytorch_device": args.policy_pytorch_device,
        },
        "num_trials_per_task": args.num_trials_per_task,
        "num_tasks_in_suite": num_tasks_in_suite,
        "num_completed_tasks": len(task_stats),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": (float(total_successes) / float(total_episodes)) if total_episodes else 0.0,
        "per_category": {
            category: {
                "episodes": stats["episodes"],
                "successes": stats["successes"],
                "success_rate": (float(stats["successes"]) / float(stats["episodes"])) if stats["episodes"] else 0.0,
            }
            for category, stats in sorted(category_stats.items())
        },
        "per_difficulty": {
            str(level): {
                "episodes": stats["episodes"],
                "successes": stats["successes"],
                "success_rate": (float(stats["successes"]) / float(stats["episodes"])) if stats["episodes"] else 0.0,
            }
            for level, stats in sorted(difficulty_stats.items())
        },
        "tasks": task_stats,
    }


def _write_summary_if_requested(
    args: Args,
    *,
    total_episodes: int,
    total_successes: int,
    category_stats,
    difficulty_stats,
    task_stats,
    num_tasks_in_suite: int,
) -> None:
    if args.summary_json_path is None:
        return
    summary = _build_summary(
        args,
        total_episodes=total_episodes,
        total_successes=total_successes,
        category_stats=category_stats,
        difficulty_stats=difficulty_stats,
        task_stats=task_stats,
        num_tasks_in_suite=num_tasks_in_suite,
    )
    summary_path = pathlib.Path(args.summary_json_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    logging.info("Wrote summary JSON to %s", summary_path)


def _load_task_classification() -> Dict[str, Dict[str, Dict[str, Any]]]:
    classification_path = pathlib.Path(__file__).resolve().parents[2] / "third_party/libero-plus/libero/libero/benchmark/task_classification.json"
    with open(classification_path, "r") as f:
        raw = json.load(f)
    return {
        suite_name: {entry["name"]: entry for entry in entries}
        for suite_name, entries in raw.items()
    }


def _filter_task_suite(task_suite, args: Args) -> None:
    if (
        args.task_category is None
        and args.difficulty_level is None
        and args.task_name_pattern is None
        and args.task_start_index == 0
        and args.task_limit is None
    ):
        return

    task_classification = _load_task_classification()
    suite_classification = task_classification.get(task_suite.name, {})
    task_name_pattern = re.compile(args.task_name_pattern) if args.task_name_pattern is not None else None

    filtered_tasks = []
    for task in task_suite.tasks:
        task_meta = suite_classification.get(task.name)
        if args.task_category is not None:
            task_category = task_meta.get("category") if task_meta is not None else _infer_task_category(task.name)
            if task_category != args.task_category:
                continue
        if args.difficulty_level is not None:
            if task_meta is None or task_meta.get("difficulty_level") != args.difficulty_level:
                continue
        if task_name_pattern is not None and not task_name_pattern.search(task.name):
            continue
        filtered_tasks.append(task)

    if args.task_start_index > 0:
        filtered_tasks = filtered_tasks[args.task_start_index :]

    if args.task_limit is not None:
        filtered_tasks = filtered_tasks[: args.task_limit]

    task_suite.tasks = filtered_tasks
    task_suite.n_tasks = len(filtered_tasks)


def _get_task_metadata(
    suite_name: str,
    task_name: str,
    task_classification: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    suite_classification = task_classification.get(suite_name, {})
    task_meta = dict(suite_classification.get(task_name, {}))
    if "category" not in task_meta or task_meta["category"] is None:
        task_meta["category"] = _infer_task_category(task_name)
    if "difficulty_level" not in task_meta:
        task_meta["difficulty_level"] = None
    return task_meta


def _infer_task_category(task_name: str) -> Optional[str]:
    if "_noise_" in task_name:
        return "Sensor Noise"
    if "_language_" in task_name:
        return "Language Instructions"
    if "_light_" in task_name:
        return "Light Conditions"
    if "_add_" in task_name or "moved_level" in task_name:
        return "Objects Layout"
    if "_table_" in task_name or "_tb_" in task_name:
        return "Background Textures"
    if "_view_" in task_name:
        match = re.search(r"_initstate_(\d+)", task_name)
        if match is not None and match.group(1) != "0":
            return "Robot Initial States"
        return "Camera Viewpoints"
    return None


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
