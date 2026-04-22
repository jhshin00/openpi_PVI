"""
PyTorch training entrypoint for the pi0.5 encoder-replacement path.

This is intentionally separate from scripts/train_pytorch_PVI.py. It loads the same base PI0/PI05
PyTorch checkpoints, but only allows configs whose model has use_encoder_replace=True.
"""

import dataclasses
import gc
import importlib
import logging
import os
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.training.config as _config
import openpi.training.data_loader as _data

try:
    _shared_train = importlib.import_module("train_pytorch_PVI")
except ModuleNotFoundError:
    _shared_train = importlib.import_module("scripts.train_pytorch_PVI")


GRAD_NORM_LOG_EXACT_TARGETS = {
    "grad_norm/image_token_adapter.projector.weight": "image_token_adapter.projector.weight",
    "grad_norm/action_out_proj.weight": "action_out_proj.weight",
    "grad_norm/time_mlp_out.weight": "time_mlp_out.weight",
    "grad_norm/action_expert.layer0.q_proj.weight": (
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight"
    ),
}

GRAD_NORM_LOG_DYNAMIC_TARGETS = {
    "grad_norm/vision_lora.layer0.q_proj.lora_b": (
        ("replacement_encoder.",),
        (".q_proj.lora_b", ".query.lora_b", ".lora_b"),
    ),
    "grad_norm/vlm_lora.layer0.q_proj.lora_b": (
        (
            "paligemma_with_expert.paligemma.language_model.",
            "paligemma_with_expert.paligemma.model.language_model.",
        ),
        (".q_proj.lora_b", ".lora_b"),
    ),
    "grad_norm/action_expert_lora.layer0.q_proj.lora_b": (
        ("paligemma_with_expert.gemma_expert.model.",),
        (".q_proj.lora_b", ".lora_b"),
    ),
}


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True) -> None:
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
        return

    run_name = config.exp_name
    if config.name and config.exp_name and config.exp_name != config.name:
        run_name = f"{config.name}__{config.exp_name}"

    tags = [config.name]
    encoder_type = getattr(config.model, "encoder_replace_encoder_type", None)
    if encoder_type:
        tags.append("encoder_replace")
        tags.append(encoder_type)
        tags.append(getattr(config.model, "encoder_replace_variant", "v1"))

    wandb.init(
        name=run_name,
        group=config.exp_name,
        tags=tags,
        config=dataclasses.asdict(config),
        project=config.project_name,
    )
    (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def build_datasets(config: _config.TrainConfig):
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def init_logging() -> None:
    _shared_train.init_logging()


def get_model_state_dict(model):
    return _shared_train.get_model_state_dict(model)


def get_model_parameters(model):
    return _shared_train.get_model_parameters(model)


def get_named_model_parameters(model):
    return _shared_train.get_named_model_parameters(model)


def _parameter_scope(name: str, *, is_lora: bool) -> str:
    clean_name = name.removeprefix("module.")
    if clean_name.startswith("image_token_adapter."):
        return "image_token_adapter"
    if clean_name.startswith("replacement_encoder."):
        return "replacement_encoder_lora" if is_lora else "replacement_encoder_non_lora"
    if clean_name.startswith("paligemma_with_expert.paligemma."):
        return "vlm_lora" if is_lora else "vlm_non_lora"
    if clean_name.startswith("paligemma_with_expert.gemma_expert."):
        return "action_expert_lora" if is_lora else "action_expert_non_lora"
    if clean_name.startswith(("action_", "state_proj.", "time_mlp_")):
        return "action_side_other_lora" if is_lora else "action_side_other_non_lora"
    return "other_lora" if is_lora else "other_non_lora"


def log_encoder_replace_parameter_summary(model) -> None:
    model_to_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    variant = getattr(model_to_inspect, "encoder_replace_variant", "unknown")

    total_params = 0
    trainable_params = 0
    total_lora_params = 0
    trainable_lora_params = 0
    trainable_by_scope: dict[str, int] = {}
    trainable_lora_examples: list[str] = []

    for name, param in get_named_model_parameters(model):
        count = param.numel()
        is_lora = ".lora_" in name or name.endswith(("lora_a", "lora_b"))
        total_params += count
        if is_lora:
            total_lora_params += count
        if not param.requires_grad:
            continue

        trainable_params += count
        if is_lora:
            trainable_lora_params += count
            if len(trainable_lora_examples) < 8:
                trainable_lora_examples.append(name.removeprefix("module."))
        scope = _parameter_scope(name, is_lora=is_lora)
        trainable_by_scope[scope] = trainable_by_scope.get(scope, 0) + count

    trainable_pct = 100.0 * trainable_params / total_params if total_params else 0.0
    lora_pct = 100.0 * trainable_lora_params / trainable_params if trainable_params else 0.0
    logging.info(
        "Encoder-replacement parameter summary: variant=%s trainable=%s / total=%s (%.4f%%), "
        "trainable_lora=%s / total_lora=%s (%.4f%% of trainable)",
        variant,
        f"{trainable_params:,}",
        f"{total_params:,}",
        trainable_pct,
        f"{trainable_lora_params:,}",
        f"{total_lora_params:,}",
        lora_pct,
    )

    logging.info("Encoder-replacement trainable parameters by policy scope:")
    for scope, count in sorted(trainable_by_scope.items(), key=lambda item: (-item[1], item[0])):
        scope_pct = 100.0 * count / trainable_params if trainable_params else 0.0
        logging.info("  %s: %s (%.2f%%)", scope, f"{count:,}", scope_pct)

    adapter_params = trainable_by_scope.get("image_token_adapter", 0)
    vision_lora_params = trainable_by_scope.get("replacement_encoder_lora", 0)
    vision_non_lora_params = trainable_by_scope.get("replacement_encoder_non_lora", 0)
    vlm_lora_params = trainable_by_scope.get("vlm_lora", 0)
    vlm_non_lora_params = trainable_by_scope.get("vlm_non_lora", 0)
    action_lora_params = trainable_by_scope.get("action_expert_lora", 0)
    action_non_lora_params = trainable_by_scope.get("action_expert_non_lora", 0)

    logging.info(
        "Encoder-replacement trainable policy check: adapter=%s, vision_lora=%s, vision_non_lora=%s, "
        "vlm_lora=%s, vlm_non_lora=%s, action_lora=%s, action_non_lora=%s",
        f"{adapter_params:,}",
        f"{vision_lora_params:,}",
        f"{vision_non_lora_params:,}",
        f"{vlm_lora_params:,}",
        f"{vlm_non_lora_params:,}",
        f"{action_lora_params:,}",
        f"{action_non_lora_params:,}",
    )
    if trainable_lora_examples:
        logging.info("Trainable LoRA parameter examples: %s", ", ".join(trainable_lora_examples))

    if vision_non_lora_params:
        logging.warning("Replacement encoder has non-LoRA trainable parameters: %s", f"{vision_non_lora_params:,}")
    if vlm_non_lora_params:
        logging.warning("VLM backbone has non-LoRA trainable parameters: %s", f"{vlm_non_lora_params:,}")
    if variant == "v1" and trainable_lora_params:
        logging.warning("variant v1 unexpectedly has trainable LoRA parameters: %s", f"{trainable_lora_params:,}")
    if variant == "v2" and (vision_lora_params == 0 or vlm_lora_params == 0 or action_non_lora_params == 0):
        logging.warning("variant v2 expected vision_lora, vlm_lora, and full action expert trainable parameters")
    if variant == "v3" and (vision_lora_params == 0 or vlm_lora_params == 0 or action_lora_params == 0):
        logging.warning("variant v3 expected vision_lora, vlm_lora, and action expert LoRA trainable parameters")
    if variant == "v3" and action_non_lora_params:
        logging.warning("variant v3 action expert has non-LoRA trainable parameters: %s", f"{action_non_lora_params:,}")


def _find_first_trainable_parameter(
    named_parameters: dict[str, torch.nn.Parameter],
    *,
    prefixes: tuple[str, ...],
    suffixes: tuple[str, ...],
) -> tuple[str, torch.nn.Parameter] | None:
    for name, param in named_parameters.items():
        clean_name = name.removeprefix("module.")
        if not param.requires_grad:
            continue
        if not any(clean_name.startswith(prefix) for prefix in prefixes):
            continue
        if not any(clean_name.endswith(suffix) for suffix in suffixes):
            continue
        return clean_name, param
    return None


def resolve_grad_norm_log_parameters(model) -> dict[str, torch.nn.Parameter]:
    named_parameters = dict(get_named_model_parameters(model))
    resolved = {}
    missing = []
    frozen = []

    for log_key, param_name in GRAD_NORM_LOG_EXACT_TARGETS.items():
        param = named_parameters.get(param_name)
        if param is None:
            missing.append(param_name)
            continue
        if not param.requires_grad:
            frozen.append(param_name)
            continue
        resolved[log_key] = param

    resolved_names = {}
    for log_key, (prefixes, suffixes) in GRAD_NORM_LOG_DYNAMIC_TARGETS.items():
        match = _find_first_trainable_parameter(named_parameters, prefixes=prefixes, suffixes=suffixes)
        if match is None:
            missing.append(f"{log_key} prefixes={prefixes} suffixes={suffixes}")
            continue
        param_name, param = match
        resolved[log_key] = param
        resolved_names[log_key] = param_name

    if resolved:
        logging.info("Gradient norm logging enabled for: %s", ", ".join(resolved))
    if resolved_names:
        logging.info(
            "Gradient norm dynamic parameter mapping: %s",
            ", ".join(f"{key} -> {value}" for key, value in sorted(resolved_names.items())),
        )
    if missing:
        logging.info("Gradient norm logging skipped for missing parameters: %s", ", ".join(missing))
    if frozen:
        logging.info("Gradient norm logging skipped for frozen parameters: %s", ", ".join(frozen))
    return resolved


def collect_grad_norm_stats(parameters_to_log: dict[str, torch.nn.Parameter]) -> dict[str, float]:
    return _shared_train.collect_grad_norm_stats(parameters_to_log)


def pop_model_debug_metrics(model) -> dict[str, float]:
    return _shared_train.pop_model_debug_metrics(model)


def build_model_config(config: _config.TrainConfig) -> openpi.models.pi0_config.Pi0Config:
    if isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    else:
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
            use_encoder_replace=getattr(config.model, "use_encoder_replace", False),
            encoder_replace_encoder_type=getattr(config.model, "encoder_replace_encoder_type", "dinov2"),
            encoder_replace_encoder_name=getattr(
                config.model,
                "encoder_replace_encoder_name",
                "facebook/dinov2-base",
            ),
            encoder_replace_variant=getattr(config.model, "encoder_replace_variant", "v1"),
            encoder_replace_lora_rank=getattr(config.model, "encoder_replace_lora_rank", 16),
            encoder_replace_lora_alpha=getattr(config.model, "encoder_replace_lora_alpha", 16.0),
            encoder_replace_action_lora_rank=getattr(config.model, "encoder_replace_action_lora_rank", 32),
            encoder_replace_action_lora_alpha=getattr(config.model, "encoder_replace_action_lora_alpha", 32.0),
        )

    if not getattr(model_cfg, "use_encoder_replace", False):
        raise ValueError("scripts/train_pytorch_encoder_replace.py requires model.use_encoder_replace=True")
    return model_cfg


def log_sample_images(config: _config.TrainConfig) -> None:
    sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
    observation, actions = next(iter(sample_data_loader))
    sample_batch = observation.to_dict()
    sample_batch["actions"] = actions

    images_to_log = []
    batch_size = next(iter(sample_batch["image"].values())).shape[0]
    for i in range(min(5, batch_size)):
        img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
        images_to_log.append(wandb.Image(img_concatenated.cpu().numpy()))

    wandb.log({"camera_views": images_to_log}, step=0)

    del sample_batch, observation, actions, images_to_log, img_concatenated
    del sample_data_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("Cleared sample batch and data loader from memory")


def load_base_weights_for_encoder_replace(config, model, model_cfg) -> None:
    if config.pytorch_weight_path is None:
        return

    logging.info("Loading base weights from: %s", config.pytorch_weight_path)
    model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
    model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    missing, unexpected = safetensors.torch.load_model(model_to_load, model_path, strict=False)

    if not hasattr(model_to_load, "is_expected_base_missing_key"):
        raise TypeError("Encoder-replacement model must define is_expected_base_missing_key")

    unexpected_missing = [key for key in missing if not model_to_load.is_expected_base_missing_key(key)]
    if unexpected_missing:
        raise ValueError(
            f"Unexpected missing keys while loading encoder-replacement base weights: {unexpected_missing}"
        )
    if unexpected:
        logging.warning("Unexpected keys while loading base weights: %s", unexpected)
    if missing:
        logging.info("Expected encoder-replacement missing keys from base checkpoint: %d", len(missing))

    logging.info(
        "Loaded base checkpoint for encoder replacement: path=%s encoder_type=%s",
        config.pytorch_weight_path,
        getattr(model_cfg, "encoder_replace_encoder_type", "dinov2"),
    )
    logging.info(
        "Encoder-replacement checkpoint policy: variant=%s lora_rank=%s lora_alpha=%s action_lora_rank=%s "
        "action_lora_alpha=%s",
        getattr(model_cfg, "encoder_replace_variant", "v1"),
        getattr(model_cfg, "encoder_replace_lora_rank", 16),
        getattr(model_cfg, "encoder_replace_lora_alpha", 16.0),
        getattr(model_cfg, "encoder_replace_action_lora_rank", 32),
        getattr(model_cfg, "encoder_replace_action_lora_alpha", 32.0),
    )


def maybe_run_shape_check(model, observation, *, is_main: bool, global_step: int, wandb_enabled: bool) -> None:
    if not is_main or global_step != 0:
        return

    model_to_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if not hasattr(model_to_inspect, "check_image_token_shapes"):
        return

    was_training = model_to_inspect.training
    model_to_inspect.eval()
    try:
        metrics = model_to_inspect.check_image_token_shapes(observation)
    finally:
        model_to_inspect.train(was_training)

    logging.info("Encoder-replacement shape check: %s", ", ".join(f"{k}={v}" for k, v in metrics.items()))
    if wandb_enabled:
        wandb.log(metrics, step=global_step)


def train_loop(config: _config.TrainConfig) -> None:
    use_ddp, local_rank, device = _shared_train.setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    _shared_train.set_seed(config.seed, local_rank)

    resuming = False
    if config.resume:
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            latest_step = _shared_train.get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is None:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
            resuming = True
            logging.info(
                "Resuming from experiment checkpoint directory: %s at step %s", exp_checkpoint_dir, latest_step
            )
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists() and is_main:
        shutil.rmtree(config.checkpoint_dir)
        logging.info("Overwriting checkpoint directory: %s", config.checkpoint_dir)

    if not resuming and is_main:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Created experiment checkpoint directory: %s", config.checkpoint_dir)
    elif is_main:
        logging.info("Using existing experiment checkpoint directory: %s", config.checkpoint_dir)

    if use_ddp:
        dist.barrier()

    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        "Using batch size per GPU: %s (total batch size across %s GPUs: %s)",
        effective_batch_size,
        world_size,
        config.batch_size,
    )

    loader, data_config = build_datasets(config)

    if is_main and config.wandb_enabled and not resuming:
        log_sample_images(config)

    model_cfg = build_model_config(config)
    model = openpi.models_pytorch.pi0_pytorch.create_model(model_cfg).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    if is_main and torch.cuda.is_available():
        _shared_train.log_memory_usage(device, 0, "after_model_creation")

    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=world_size >= 8,
        )

    load_base_weights_for_encoder_replace(config, model, model_cfg)

    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    trainable_params = [param for param in get_model_parameters(model) if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found for encoder-replacement training")

    optim = torch.optim.AdamW(
        trainable_params,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    global_step = 0
    if resuming:
        global_step = _shared_train.load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info("Resumed training from step %s", global_step)

    def lr_schedule(step: int):
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []

    if is_main:
        logging.info(
            "Running on: %s | world_size=%s",
            platform.node(),
            torch.distributed.get_world_size() if use_ddp else 1,
        )
        logging.info(
            "Training config: batch_size=%s, effective_batch_size=%s, num_train_steps=%s",
            config.batch_size,
            effective_batch_size,
            config.num_train_steps,
        )
        logging.info("Memory optimizations: gradient_checkpointing=%s", enable_gradient_checkpointing)
        logging.info(
            "LR schedule: warmup=%s, peak_lr=%.2e, decay_steps=%s, end_lr=%.2e",
            warmup_steps,
            peak_lr,
            decay_steps,
            end_lr,
        )
        logging.info(
            "Optimizer: %s, weight_decay=%s, clip_norm=%s",
            type(config.optimizer).__name__,
            config.optimizer.weight_decay,
            config.optimizer.clip_gradient_norm,
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info("Training precision: %s", model_cfg.dtype)
        _shared_train.log_trainable_parameter_summary(model)
        log_encoder_replace_parameter_summary(model)

    grad_norm_log_parameters = resolve_grad_norm_log_parameters(model) if is_main else {}
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            if global_step >= config.num_train_steps:
                break

            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            for param_group in optim.param_groups:
                param_group["lr"] = lr_schedule(global_step)

            maybe_run_shape_check(
                model,
                observation,
                is_main=is_main,
                global_step=global_step,
                wandb_enabled=config.wandb_enabled,
            )

            losses = model(observation, actions)
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss detected at step {global_step}: {loss.detach().cpu().item()}"
                )
            debug_metric_stats = pop_model_debug_metrics(model) if is_main else {}

            loss.backward()

            bad_grads = _shared_train.find_nonfinite_parameter_names(get_named_model_parameters(model), check_grad=True)
            if bad_grads:
                raise FloatingPointError(
                    f"Non-finite gradients detected at step {global_step} in parameters: {bad_grads}"
                )
            grad_norm_stats = collect_grad_norm_stats(grad_norm_log_parameters) if is_main else {}

            if global_step < 5 and is_main and torch.cuda.is_available():
                _shared_train.log_memory_usage(device, global_step, "after_backward")

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)
            optim.step()

            bad_params = _shared_train.find_nonfinite_parameter_names(
                get_named_model_parameters(model),
                check_grad=False,
            )
            if bad_params:
                raise FloatingPointError(
                    f"Non-finite parameters detected immediately after optimizer step {global_step}: {bad_params}"
                )
            optim.zero_grad(set_to_none=True)

            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        **grad_norm_stats,
                        **debug_metric_stats,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                grad_vals = [info["grad_norm"] for info in infos if info.get("grad_norm") is not None]
                avg_grad_norm = sum(grad_vals) / len(grad_vals) if grad_vals else None

                avg_monitored_grad_norms = {}
                for log_key in grad_norm_log_parameters:
                    vals = [info[log_key] for info in infos if info.get(log_key) is not None]
                    if vals:
                        avg_monitored_grad_norms[log_key] = sum(vals) / len(vals)

                avg_debug_metrics = {}
                debug_metric_keys = sorted(
                    {key for info in infos for key in info if key.startswith("encoder_replace/")}
                )
                for log_key in debug_metric_keys:
                    vals = [info[log_key] for info in infos if info.get(log_key) is not None]
                    if vals:
                        avg_debug_metrics[log_key] = sum(vals) / len(vals)

                if avg_grad_norm is not None:
                    logging.info(
                        "step=%s loss=%.4f lr=%.2e grad_norm=%.2f time=%.1fs",
                        global_step,
                        avg_loss,
                        avg_lr,
                        avg_grad_norm,
                        elapsed,
                    )
                else:
                    logging.info("step=%s loss=%.4f lr=%.2e time=%.1fs", global_step, avg_loss, avg_lr, elapsed)

                if config.wandb_enabled and infos:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    log_payload.update(avg_monitored_grad_norms)
                    log_payload.update(avg_debug_metrics)
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []

            global_step += 1
            _shared_train.save_checkpoint(model, optim, global_step, config, is_main, data_config)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    if pbar is not None:
        pbar.close()

    if is_main and config.wandb_enabled:
        wandb.finish()

    _shared_train.cleanup_ddp()


def main() -> None:
    _shared_train.init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
