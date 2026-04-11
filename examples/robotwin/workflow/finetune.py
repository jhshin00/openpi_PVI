import dataclasses
from pathlib import Path
import sys

import tyro

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.robotwin.workflow.common import prepare_robotwin_train_config
from scripts import train_pytorch_PVI as _train


@dataclasses.dataclass(frozen=True)
class Args:
    train_config_name: str
    model_name: str
    repo_id: str
    lerobot_root: str = "./datasets"
    asset_id: str | None = None
    overwrite: bool = True
    resume: bool = False


def main(args: Args) -> None:
    _train.init_logging()
    config = prepare_robotwin_train_config(
        args.train_config_name,
        repo_id=args.repo_id,
        exp_name=args.model_name,
        lerobot_root=args.lerobot_root,
        asset_id=args.asset_id,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    _train.train_loop(config)


if __name__ == "__main__":
    main(tyro.cli(Args))
