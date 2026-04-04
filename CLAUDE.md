# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of **openpi** (Physical Intelligence) containing the official π₀, π₀-FAST, and π₀.₅ vision-language-action (VLA) models for robotics. This fork adds **PVI (Policy Value Iteration)** - a dual-branch architecture that injects DINO visual features into the frozen main model via layer-wise residual control.

Key models:
- **π₀**: Flow-based VLA (JAX)
- **π₀-FAST**: Autoregressive VLA with FAST tokenizer (JAX)
- **π₀.₅**: Upgraded π₀ with knowledge insulation (JAX + PyTorch)
- **PVI**: PyTorch-only extension that adds trainable DINO-conditioned branch

## Installation & Setup

```bash
# Clone with submodules
git clone --recurse-submodules <repo>
git submodule update --init --recursive

# Install dependencies (requires uv)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# For PyTorch: Apply transformers patches (REQUIRED)
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

## Common Commands

### Linting & Formatting
```bash
uv run ruff check .         # Lint
uv run ruff format .        # Format
uv run ruff check --fix .   # Lint with auto-fix
pre-commit run --all-files  # Run all pre-commit hooks
```

### Testing
```bash
uv run pytest                           # Run all tests
uv run pytest src/openpi/models/        # Run tests in specific directory
uv run pytest path/to/test_file.py      # Run single test file
uv run pytest -k "test_name"            # Run tests matching pattern
uv run pytest -m "not manual"           # Skip manual tests
```

### Training (JAX)
```bash
# Compute normalization stats first
uv run scripts/compute_norm_stats.py --config-name <config_name>

# Train
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<name>
```

### Training (PyTorch)
```bash
# Single GPU
uv run scripts/train_pytorch.py <config_name> --exp_name <name>

# Multi-GPU (single node)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<gpus> scripts/train_pytorch.py <config_name> --exp_name <name>

# PVI training
uv run scripts/train_pytorch_PVI.py <config_name> --exp_name <name>
```

### JAX to PyTorch Conversion
```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir <jax_checkpoint> \
    --config_name <config_name> \
    --output_path <output_path>
```

### Inference Server
```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=<config_name> \
    --policy.dir=<checkpoint_dir>
```

## Architecture

### Directory Structure
- `src/openpi/models/` - JAX model implementations (π₀, π₀-FAST, π₀.₅)
- `src/openpi/models_pytorch/` - PyTorch implementations + PVI
- `src/openpi/policies/` - Robot-specific input/output transforms (ALOHA, DROID, LIBERO)
- `src/openpi/training/` - Training configs, data loaders, optimizers
- `src/openpi/transforms.py` - Data preprocessing transforms
- `scripts/` - Training, inference, and utility scripts
- `examples/` - Robot-specific examples (ALOHA, DROID, LIBERO, UR5)

### PVI Architecture (PyTorch only)
PVI extends `PI0Pytorch` with a dual-branch suffix computation:
1. **Main branch** (frozen): Original VLM prefix → action expert
2. **Copy branch** (trainable): DINO features → copied expert layers → zero-init injectors

Key files:
- [pi0_pvi_pytorch.py](src/openpi/models_pytorch/pi0_pvi_pytorch.py) - PVI model
- [pvi_modules.py](src/openpi/models_pytorch/pvi_modules.py) - DINO encoder, ZeroInitLinear
- [train_pytorch_PVI.py](scripts/train_pytorch_PVI.py) - PVI training script

### Config System
Training configs are defined in [src/openpi/training/config.py](src/openpi/training/config.py). Use `get_config("<name>")` to retrieve configs programmatically. Key PVI configs:
- `pi0_libero_pvi` / `pi05_libero_pvi` - PVI training
- `pi0_libero_pvi_infer` / `pi05_libero_pvi_infer` - PVI inference

### Data Pipeline
1. LeRobot dataset → `repack_transforms` (format adaptation)
2. → `data_transforms` (robot-specific preprocessing)
3. → normalization (quantile or z-score)
4. → `model_transforms` (tokenization, image resize)

## Environment Variables
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` - Allow JAX to use 90% GPU memory
- `HF_LEROBOT_HOME=<path>` - Local LeRobot datasets directory
- `OPENPI_DATA_HOME=<path>` - Override checkpoint cache location
- `CUDA_VISIBLE_DEVICES=<ids>` - GPU selection