import dataclasses
from pathlib import Path
import sys

import tyro

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from examples.robotwin.workflow.common import prepare_robotwin_train_config  # noqa: E402
from scripts import train_pytorch_encoder_replace as _train  # noqa: E402


@dataclasses.dataclass(frozen=True)
class Args:
    train_config_name: str
    model_name: str
    repo_id: str
    lerobot_root: str = "./datasets"
    asset_id: str | None = None
    checkpoint_base_dir: str = str(REPO_ROOT / "checkpoints_local")
    num_train_steps: int | None = None
    overwrite: bool = False
    resume: bool = False


def main(args: Args) -> None:
    _train.init_logging()
    config = prepare_robotwin_train_config(
        args.train_config_name,
        repo_id=args.repo_id,
        exp_name=args.model_name,
        lerobot_root=args.lerobot_root,
        asset_id=args.asset_id,
        checkpoint_base_dir=args.checkpoint_base_dir,
        num_train_steps=args.num_train_steps,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    if not getattr(config.model, "use_encoder_replace", False):
        raise ValueError("finetune_encoder_replace.py requires a config with model.use_encoder_replace=True")
    _train.train_loop(config)


if __name__ == "__main__":
    main(tyro.cli(Args))
