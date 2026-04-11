import dataclasses
import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def _infer_single_asset_id(assets_dir: pathlib.Path) -> str | None:
    if not assets_dir.exists():
        return None

    norm_stats_files = sorted(assets_dir.rglob("norm_stats.json"))
    if len(norm_stats_files) == 1:
        return norm_stats_files[0].parent.relative_to(assets_dir).as_posix()

    asset_dirs = sorted(path.name for path in assets_dir.iterdir() if path.is_dir())
    if len(asset_dirs) == 1:
        return asset_dirs[0]
    return None


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    asset_id_override: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".
        asset_id_override: Optional asset id override. Useful when loading checkpoints whose bundled
            normalization stats are stored under an asset id that differs from the static training config.

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if asset_id_override is not None:
        data_config = dataclasses.replace(data_config, asset_id=asset_id_override)

    if norm_stats is None:
        checkpoint_assets_dir = checkpoint_dir / "assets"

        # Prefer the copy saved inside the checkpoint so inference uses the exact training-time statistics.
        # Fall back to the config-provided assets when evaluating checkpoints that do not bundle norm stats
        # (for example a converted base PyTorch checkpoint).
        asset_id = data_config.asset_id
        if asset_id is None:
            asset_id = _infer_single_asset_id(checkpoint_assets_dir)
            if asset_id is not None:
                logging.info("Inferred checkpoint asset id %s from %s", asset_id, checkpoint_assets_dir)

        if asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")

        try:
            norm_stats = _checkpoints.load_norm_stats(checkpoint_assets_dir, asset_id)
        except FileNotFoundError:
            inferred_asset_id = _infer_single_asset_id(checkpoint_assets_dir)
            if inferred_asset_id is not None and inferred_asset_id != asset_id:
                logging.info(
                    "Norm stats not found for asset id %s in checkpoint %s, retrying with inferred asset id %s",
                    asset_id,
                    checkpoint_dir,
                    inferred_asset_id,
                )
                norm_stats = _checkpoints.load_norm_stats(checkpoint_assets_dir, inferred_asset_id)
                asset_id = inferred_asset_id
            elif data_config.norm_stats is None:
                raise
            else:
                logging.info(
                    "Norm stats not found in checkpoint %s, falling back to config assets for %s",
                    checkpoint_dir,
                    asset_id,
                )
                norm_stats = data_config.norm_stats

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
