import dataclasses

from openpi.training import config as _config


def prepare_robotwin_train_config(
    train_config_name: str,
    *,
    repo_id: str,
    exp_name: str | None = None,
    lerobot_root: str = "./datasets",
    asset_id: str | None = None,
    overwrite: bool | None = None,
    resume: bool | None = None,
) -> _config.TrainConfig:
    config = _config.get_config(train_config_name)

    data_factory_kwargs = {
        "repo_id": repo_id,
        "lerobot_root": lerobot_root,
    }
    if asset_id is not None and hasattr(config.data, "assets"):
        data_factory_kwargs["assets"] = dataclasses.replace(config.data.assets, asset_id=asset_id)

    data_factory = dataclasses.replace(config.data, **data_factory_kwargs)

    config_kwargs = {
        "data": data_factory,
    }
    if exp_name is not None:
        config_kwargs["exp_name"] = exp_name
    if overwrite is not None:
        config_kwargs["overwrite"] = overwrite
    if resume is not None:
        config_kwargs["resume"] = resume

    return dataclasses.replace(config, **config_kwargs)
