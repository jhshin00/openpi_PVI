import dataclasses
from pathlib import Path
import sys

import numpy as np
import tqdm
import tyro

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from examples.robotwin.workflow.common import prepare_robotwin_train_config
import openpi.shared.normalize as normalize
from scripts import compute_norm_stats as _compute_norm_stats


@dataclasses.dataclass(frozen=True)
class Args:
    train_config_name: str
    repo_id: str
    lerobot_root: str = "./datasets"
    asset_id: str | None = None
    max_frames: int | None = None


def main(args: Args) -> None:
    config = prepare_robotwin_train_config(
        args.train_config_name,
        repo_id=args.repo_id,
        lerobot_root=args.lerobot_root,
        asset_id=args.asset_id,
    )
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = _compute_norm_stats.create_rlds_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            args.max_frames,
        )
    else:
        data_loader, num_batches = _compute_norm_stats.create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            args.max_frames,
        )

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))

    assets_root = Path(config.data.assets.assets_dir or config.assets_dirs)
    output_path = assets_root / data_config.asset_id
    normalize.save(output_path, {key: value.get_statistics() for key, value in stats.items()})
    print(f"Writing stats to: {output_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
