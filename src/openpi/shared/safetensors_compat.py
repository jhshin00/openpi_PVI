import logging
import os
from typing import Union

import safetensors
import safetensors.torch
import torch

logger = logging.getLogger("openpi")

_FALLBACK_ERROR_MARKERS = (
    "no suitable name to keep for saving",
    "torch_shared_tensors",
)


def _should_fallback_to_plain_state_dict(exc: RuntimeError) -> bool:
    message = str(exc)
    return any(marker in message for marker in _FALLBACK_ERROR_MARKERS)


def _alias_metadata(filename: Union[str, os.PathLike]) -> dict[str, str]:
    with safetensors.safe_open(filename, framework="pt") as handle:
        metadata = handle.metadata() or {}
    return {str(alias): str(canonical) for alias, canonical in metadata.items()}


def _filter_alias_backed_missing_keys(
    missing_keys: list[str],
    *,
    state_dict_keys: set[str],
    alias_metadata: dict[str, str],
) -> tuple[list[str], list[tuple[str, str]]]:
    remaining_missing: list[str] = []
    ignored_aliases: list[tuple[str, str]] = []
    for key in missing_keys:
        canonical = alias_metadata.get(key)
        if canonical is not None and canonical in state_dict_keys:
            ignored_aliases.append((key, canonical))
        else:
            remaining_missing.append(key)
    return remaining_missing, ignored_aliases


def _format_incompatible_keys_error(
    model: torch.nn.Module,
    *,
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> RuntimeError:
    error = f"Error(s) in loading state_dict for {model.__class__.__name__}:"
    if missing_keys:
        missing = ", ".join(f'"{key}"' for key in sorted(missing_keys))
        error += f"\n    Missing key(s) in state_dict: {missing}"
    if unexpected_keys:
        unexpected = ", ".join(f'"{key}"' for key in sorted(unexpected_keys))
        error += f"\n    Unexpected key(s) in state_dict: {unexpected}"
    return RuntimeError(error)


def load_model_with_fallback(
    model: torch.nn.Module,
    filename: Union[str, os.PathLike],
    *,
    strict: bool = True,
    device: Union[str, int] = "cpu",
) -> tuple[list[str], list[str]]:
    """Load a safetensors checkpoint and fall back to plain state-dict loading for storage-alias edge cases."""
    try:
        missing, unexpected = safetensors.torch.load_model(model, filename, strict=strict, device=device)
        return list(missing), list(unexpected)
    except RuntimeError as exc:
        if not _should_fallback_to_plain_state_dict(exc):
            raise

        logger.warning(
            "safetensors.load_model failed shared-storage inspection for %s; "
            "falling back to load_file + load_state_dict. original_error=%s",
            filename,
            exc,
        )

        state_dict = safetensors.torch.load_file(filename, device=device)
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)

        alias_metadata = _alias_metadata(filename)
        missing_keys, ignored_aliases = _filter_alias_backed_missing_keys(
            missing_keys,
            state_dict_keys=set(state_dict.keys()),
            alias_metadata=alias_metadata,
        )
        if ignored_aliases:
            logger.info(
                "Ignoring %d alias-backed missing key(s) during safetensors fallback load.",
                len(ignored_aliases),
            )

        if strict and (missing_keys or unexpected_keys):
            raise _format_incompatible_keys_error(
                model,
                missing_keys=missing_keys,
                unexpected_keys=unexpected_keys,
            ) from exc

        return missing_keys, unexpected_keys
