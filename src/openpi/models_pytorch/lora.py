"""Small PyTorch LoRA helpers for encoder-replacement fine-tuning."""

from collections.abc import Iterable
import math

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


class LoRALinear(nn.Module):
    """Drop-in `nn.Linear` replacement that preserves base checkpoint key names.

    The wrapped module exposes `weight` and `bias` directly, so a checkpoint with
    keys like `layers.0.self_attn.q_proj.weight` still loads after replacing
    `q_proj` with `LoRALinear`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        bias: bool,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {alpha}")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        self.lora_a = nn.Parameter(torch.empty(rank, in_features, dtype=dtype, device=device))
        self.lora_b = nn.Parameter(torch.empty(out_features, rank, dtype=dtype, device=device))
        self.reset_parameters()

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, rank: int, alpha: float) -> "LoRALinear":
        module = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        module.weight.data.copy_(linear.weight.data)
        module.weight.requires_grad = False
        if linear.bias is not None:
            module.bias.data.copy_(linear.bias.data)
            module.bias.requires_grad = False
        return module

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)  # noqa: SLF001
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        nn.init.normal_(self.lora_a, mean=0.0, std=0.01)
        nn.init.normal_(self.lora_b, mean=0.0, std=0.01)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
        return base + lora


def _matches_target_suffix(module_name: str, target_suffixes: set[str]) -> bool:
    return any(module_name == suffix or module_name.endswith(f".{suffix}") for suffix in target_suffixes)


def apply_lora_to_linear_modules(
    root: nn.Module,
    *,
    target_suffixes: Iterable[str],
    rank: int,
    alpha: float,
) -> list[str]:
    """Replace matching `nn.Linear` children under `root` with `LoRALinear`."""

    target_suffix_set = set(target_suffixes)
    replaced: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear) and _matches_target_suffix(full_name, target_suffix_set):
                setattr(parent, child_name, LoRALinear.from_linear(child, rank=rank, alpha=alpha))
                replaced.append(full_name)
                continue
            visit(child, full_name)

    visit(root, "")
    return replaced


def lora_parameter_names(module: nn.Module) -> list[str]:
    return [name for name, _ in module.named_parameters() if ".lora_" in name or name.startswith("lora_")]
