"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.ur3_policy as ur3_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Base directory containing local LeRobot datasets.
    # The dataset is expected at <lerobot_root>/<repo_id>.
    lerobot_root: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Optional base directory for local LeRobot datasets.
    lerobot_root: str | None = None
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            lerobot_root=self.lerobot_root,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )

@dataclasses.dataclass(frozen=True)
class LeRobotUR3DataConfig(DataConfigFactory):
    """Config for UR3 datasets converted to LeRobot format."""

    default_prompt: str | None = None
    use_delta_joint_actions: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/base_image": "base_image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[ur3_policy.UR3Inputs(model_type=model_config.model_type)],
            outputs=[ur3_policy.UR3Outputs()],
        )

        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Fine-tuning UR3 configs.
    #
    TrainConfig(
        name="pi05_ur3_pvi",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="dinov2",
            pvi_aux_encoder_name="facebook/dinov2-base",
            pvi_injector_init_std=0.0,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        # Match the effective number of sampled training windows rather than only raw episode count.
        # UR3 here has ~20.2k usable action-horizon windows (61 episodes, mean length ~340, horizon=10).
        # With batch_size=32, 1.6k steps yields ~51.2k sampled windows, i.e. ~2.5 dataset passes.
        # This is a small extension over the previous 1.2k-step run while keeping the rest of the
        # optimization setup fixed. Override --exp-name, --model.pvi-aux-encoder-type, and
        # --model.pvi-aux-encoder-name per experiment.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=160,
            peak_lr=1.5e-5,
            decay_steps=1_600,
            decay_lr=1.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=1_600,
        batch_size=32,
        log_interval=20,
        save_interval=200,
        keep_period=1_000,
        overwrite=False,
        resume=False,
        wandb_enabled=True,
        exp_name="pi05_ur3_pvi_dinov2_pnp_1600",
    ),
    TrainConfig(
        name="pi05_ur3_pvi_dinov2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="dinov2",
            pvi_aux_encoder_name="facebook/dinov2-base",
            pvi_injector_init_std=0.0,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=160,
            peak_lr=1.5e-5,
            decay_steps=1_600,
            decay_lr=1.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=1_600,
        batch_size=32,
        log_interval=20,
        save_interval=200,
        keep_period=1_000,
        overwrite=False,
        resume=False,
        wandb_enabled=True,
        exp_name="pi05_ur3_pvi_dinov2_pnp_1600",
    ),
    TrainConfig(
        name="pi05_ur3_pvi_siglip",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="siglip",
            pvi_aux_encoder_name="google/siglip-base-patch16-224",
            pvi_injector_init_std=0.0,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=160,
            peak_lr=1.5e-5,
            decay_steps=1_600,
            decay_lr=1.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=1_600,
        batch_size=32,
        log_interval=20,
        save_interval=200,
        keep_period=1_000,
        overwrite=False,
        resume=False,
        wandb_enabled=True,
        exp_name="pi05_ur3_pvi_siglip_pnp_1600",
    ),
    TrainConfig(
        name="pi05_ur3_pvi_siglip_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="siglip",
            pvi_aux_encoder_name="google/siglip-base-patch16-224",
            pvi_injector_init_std=0.0,
            pytorch_compile_mode=None,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=30_000,
        batch_size=1,
        log_interval=100,
        save_interval=1000,
        keep_period=5000,
        overwrite=False,
        resume=False,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi05_ur3_pvi_hpr",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="hpr",
            pvi_aux_encoder_name="hpr_checkpoints/hpr_fullfinetune_base_lang_trace_negative_mod.ckpt",
            pvi_injector_init_std=0.0,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=160,
            peak_lr=1.5e-5,
            decay_steps=1_600,
            decay_lr=1.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=1_600,
        batch_size=32,
        log_interval=20,
        save_interval=200,
        keep_period=1_000,
        overwrite=False,
        resume=False,
        wandb_enabled=True,
        exp_name="pi05_ur3_pvi_hpr_pnp_1600",
    ),
    TrainConfig(
        name="pi05_ur3_pvi_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="dinov2",
            pvi_aux_encoder_name="facebook/dinov2-base",
            pvi_injector_init_std=0.0,
            pytorch_compile_mode=None,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=30_000,
        batch_size=1,
        log_interval=100,
        save_interval=1000,
        keep_period=5000,
        overwrite=False,
        resume=False,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi05_ur3_pvi_dinov2_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="dinov2",
            pvi_aux_encoder_name="facebook/dinov2-base",
            pvi_injector_init_std=0.0,
            pytorch_compile_mode=None,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=30_000,
        batch_size=1,
        log_interval=100,
        save_interval=1000,
        keep_period=5000,
        overwrite=False,
        resume=False,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi05_ur3_pvi_hpr_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="hpr",
            pvi_aux_encoder_name="hpr_checkpoints/hpr_fullfinetune_base_lang_trace_negative_mod.ckpt",
            pvi_injector_init_std=0.0,
            pytorch_compile_mode=None,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=30_000,
        batch_size=1,
        log_interval=100,
        save_interval=1000,
        keep_period=5000,
        overwrite=False,
        resume=False,
        wandb_enabled=False,
    ),
    #
    # UR3 experiments: original 30 Hz dataset + 50-step horizon.
    #
    TrainConfig(
        name="pi05_ur3_pvi_dinov2_h50",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="dinov2",
            pvi_aux_encoder_name="facebook/dinov2-base",
            pvi_injector_init_std=0.0,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset_unfold_towel",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1.5e-5,
            decay_steps=10_000,
            decay_lr=1.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=10_000,
        batch_size=32,
        log_interval=50,
        save_interval=1_000,
        keep_period=1_000,
        overwrite=False,
        resume=False,
        wandb_enabled=True,
        exp_name="pi05_ur3_pvi_dinov2_h50_unfold_towel",
    ),
    TrainConfig(
        name="pi05_ur3_pvi_dinov2_h50_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="dinov2",
            pvi_aux_encoder_name="facebook/dinov2-base",
            pvi_injector_init_std=0.0,
            pytorch_compile_mode=None,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset_unfold_towel",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=30_000,
        batch_size=1,
        log_interval=100,
        save_interval=1000,
        keep_period=5000,
        overwrite=False,
        resume=False,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi05_ur3_pvi_hpr_h50",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="hpr",
            pvi_aux_encoder_name="hpr_checkpoints/hpr_new_checkpoint.ckpt",
            pvi_injector_init_std=0.0,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset_unfold_towel",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1.5e-5,
            decay_steps=10_000,
            decay_lr=1.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=10_000,
        batch_size=32,
        log_interval=50,
        save_interval=1000,
        keep_period=1_000,
        overwrite=False,
        resume=False,
        wandb_enabled=True,
        exp_name="pi05_ur3_pvi_hpr_h50_unfold_towel",
    ),
    TrainConfig(
        name="pi05_ur3_pvi_hpr_h50_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="hpr",
            pvi_aux_encoder_name="hpr_checkpoints/hpr_fullfinetune_base_lang_trace_negative_mod.ckpt",
            pvi_injector_init_std=0.0,
            pytorch_compile_mode=None,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset_unfold_towel",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=30_000,
        batch_size=1,
        log_interval=100,
        save_interval=1000,
        keep_period=5000,
        overwrite=False,
        resume=False,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi05_ur3_pvi_siglip_h50",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="siglip",
            pvi_aux_encoder_name="google/siglip-base-patch16-224",
            pvi_injector_init_std=0.0,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset_unfold_towel",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1.5e-5,
            decay_steps=10_000,
            decay_lr=1.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=10_000,
        batch_size=32,
        log_interval=50,
        save_interval=1000,
        keep_period=1_000,
        overwrite=False,
        resume=False,
        wandb_enabled=True,
        exp_name="pi05_ur3_pvi_siglip_h50_unfold_towel",
    ),
    TrainConfig(
        name="pi05_ur3_pvi_siglip_h50_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="siglip",
            pvi_aux_encoder_name="google/siglip-base-patch16-224",
            pvi_injector_init_std=0.0,
            pytorch_compile_mode=None,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset_unfold_towel",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=30_000,
        batch_size=1,
        log_interval=100,
        save_interval=1000,
        keep_period=5000,
        overwrite=False,
        resume=False,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi05_ur3_pvi_clip_h50",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="clip",
            pvi_aux_encoder_name="openai/clip-vit-base-patch32",
            pvi_injector_init_std=0.0,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset_unfold_towel",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1.5e-5,
            decay_steps=10_000,
            decay_lr=1.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=10_000,
        batch_size=32,
        log_interval=50,
        save_interval=1000,
        keep_period=1_000,
        overwrite=False,
        resume=False,
        wandb_enabled=True,
        exp_name="pi05_ur3_pvi_clip_h50_unfold_towel",
    ),
    TrainConfig(
        name="pi05_ur3_pvi_clip_h50_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="clip",
            pvi_aux_encoder_name="openai/clip-vit-base-patch32",
            pvi_injector_init_std=0.0,
            pytorch_compile_mode=None,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset_unfold_towel",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=30_000,
        batch_size=1,
        log_interval=100,
        save_interval=1000,
        keep_period=5000,
        overwrite=False,
        resume=False,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi05_ur3_pvi_r3m_h50",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="r3m",
            pvi_aux_encoder_name="resnet34",
            pvi_injector_init_std=0.0,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset_unfold_towel",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1.5e-5,
            decay_steps=10_000,
            decay_lr=1.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=10_000,
        batch_size=32,
        log_interval=50,
        save_interval=1000,
        keep_period=1_000,
        overwrite=False,
        resume=False,
        wandb_enabled=True,
        exp_name="pi05_ur3_pvi_r3m_h50_unfold_towel",
    ),
    TrainConfig(
        name="pi05_ur3_pvi_r3m_h50_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="r3m",
            pvi_aux_encoder_name="resnet34",
            pvi_injector_init_std=0.0,
            pytorch_compile_mode=None,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset_unfold_towel",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=30_000,
        batch_size=1,
        log_interval=100,
        save_interval=1000,
        keep_period=5000,
        overwrite=False,
        resume=False,
        wandb_enabled=False,
    ),
    #
    # UR3 experiments: 15 Hz dataset (frame stride 2 from 30 Hz raw data) + 50-step horizon.
    # These stay separate from the existing 30 Hz / 10-step configs so old checkpoints remain compatible.
    #
    TrainConfig(
        name="pi05_ur3_pvi_15hz_h50",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="dinov2",
            pvi_aux_encoder_name="facebook/dinov2-base",
            pvi_injector_init_std=0.0,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset_15hz",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=160,
            peak_lr=1.5e-5,
            decay_steps=1_600,
            decay_lr=1.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=1_600,
        batch_size=32,
        log_interval=20,
        save_interval=200,
        keep_period=1_000,
        overwrite=False,
        resume=False,
        wandb_enabled=True,
        exp_name="pi05_ur3_pvi_dinov2_15hz_h50",
    ),
    TrainConfig(
        name="pi05_ur3_pvi_dinov2_15hz_h50",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="dinov2",
            pvi_aux_encoder_name="facebook/dinov2-base",
            pvi_injector_init_std=0.0,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset_15hz",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=160,
            peak_lr=1.5e-5,
            decay_steps=1_600,
            decay_lr=1.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=1_600,
        batch_size=32,
        log_interval=20,
        save_interval=200,
        keep_period=1_000,
        overwrite=False,
        resume=False,
        wandb_enabled=True,
        exp_name="pi05_ur3_pvi_dinov2_15hz_h50",
    ),
    TrainConfig(
        name="pi05_ur3_pvi_dinov2_15hz_h50_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="dinov2",
            pvi_aux_encoder_name="facebook/dinov2-base",
            pvi_injector_init_std=0.0,
            pytorch_compile_mode=None,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset_15hz",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=30_000,
        batch_size=1,
        log_interval=100,
        save_interval=1000,
        keep_period=5000,
        overwrite=False,
        resume=False,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi05_ur3_pvi_siglip_15hz_h50",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="siglip",
            pvi_aux_encoder_name="google/siglip-base-patch16-224",
            pvi_injector_init_std=0.0,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset_15hz",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=160,
            peak_lr=1.5e-5,
            decay_steps=1_600,
            decay_lr=1.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=1_600,
        batch_size=32,
        log_interval=20,
        save_interval=200,
        keep_period=1_000,
        overwrite=False,
        resume=False,
        wandb_enabled=True,
        exp_name="pi05_ur3_pvi_siglip_15hz_h50",
    ),
    TrainConfig(
        name="pi05_ur3_pvi_siglip_15hz_h50_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="siglip",
            pvi_aux_encoder_name="google/siglip-base-patch16-224",
            pvi_injector_init_std=0.0,
            pytorch_compile_mode=None,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset_15hz",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=30_000,
        batch_size=1,
        log_interval=100,
        save_interval=1000,
        keep_period=5000,
        overwrite=False,
        resume=False,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi05_ur3_pvi_hpr_15hz_h50",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="hpr",
            pvi_aux_encoder_name="hpr_checkpoints/hpr_fullfinetune_base_lang_trace_negative_mod.ckpt",
            pvi_injector_init_std=0.0,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset_15hz",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=160,
            peak_lr=1.5e-5,
            decay_steps=1_600,
            decay_lr=1.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=1_600,
        batch_size=32,
        log_interval=20,
        save_interval=200,
        keep_period=1_000,
        overwrite=False,
        resume=False,
        wandb_enabled=True,
        exp_name="pi05_ur3_pvi_hpr_15hz_h50",
    ),
    TrainConfig(
        name="pi05_ur3_pvi_hpr_15hz_h50_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            use_pvi=True,
            pvi_aux_encoder_type="hpr",
            pvi_aux_encoder_name="hpr_checkpoints/hpr_fullfinetune_base_lang_trace_negative_mod.ckpt",
            pvi_injector_init_std=0.0,
            pytorch_compile_mode=None,
        ),
        data=LeRobotUR3DataConfig(
            repo_id="ur3_dataset_15hz",
            lerobot_root="./datasets",
            assets=AssetsConfig(assets_dir="./assets/pi05_ur3_pvi"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./checkpoints/pytorch/pi05_base",
        num_train_steps=30_000,
        batch_size=1,
        log_interval=100,
        save_interval=1000,
        keep_period=5000,
        overwrite=False,
        resume=False,
        wandb_enabled=False,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
