import logging
import math
from typing import ClassVar

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.lora import apply_lora_to_linear_modules
from openpi.models_pytorch.lora import lora_parameter_names
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pvi_modules import CLIPAuxEncoder
from openpi.models_pytorch.pvi_modules import DinoAuxEncoder
from openpi.models_pytorch.pvi_modules import HPRAuxEncoder
from openpi.models_pytorch.pvi_modules import R3MAuxEncoder
from openpi.models_pytorch.pvi_modules import SigLIPAuxEncoder

logger = logging.getLogger("openpi")


class VisionTokenAdapter(nn.Module):
    """Map frozen replacement-encoder tokens into the PaliGemma image-prefix token space."""

    def __init__(self, input_dim: int, output_dim: int, target_tokens_per_view: int):
        super().__init__()
        target_grid_size = math.isqrt(target_tokens_per_view)
        if target_grid_size * target_grid_size != target_tokens_per_view:
            raise ValueError(f"target_tokens_per_view must be a square grid, got {target_tokens_per_view}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.target_tokens_per_view = target_tokens_per_view
        self.target_grid_size = target_grid_size
        self.input_norm = nn.LayerNorm(input_dim)
        self.projector = nn.Linear(input_dim, output_dim)

    @staticmethod
    def _square_grid_size(token_count: int) -> int | None:
        if token_count <= 0:
            return None
        grid_size = math.isqrt(token_count)
        if grid_size * grid_size != token_count:
            return None
        return grid_size

    def reset_parameters(self, *, reference_std: float | None = None) -> None:
        self.input_norm.reset_parameters()
        if reference_std is None or reference_std <= 0:
            nn.init.xavier_uniform_(self.projector.weight)
        else:
            nn.init.normal_(self.projector.weight, mean=0.0, std=reference_std)
        if self.projector.bias is not None:
            nn.init.zeros_(self.projector.bias)

    def tokens_to_view_grid(self, aux_features: torch.Tensor, patches_per_view: int) -> tuple[torch.Tensor, int, bool]:
        if aux_features.ndim != 3:
            raise ValueError(f"Expected replacement encoder features with shape [B, T, D], got {aux_features.shape}")
        if patches_per_view <= 0:
            raise ValueError(f"patches_per_view must be positive, got {patches_per_view}")

        batch_size, total_tokens, hidden_size = aux_features.shape
        if hidden_size != self.input_dim:
            raise ValueError(f"Expected replacement hidden size {self.input_dim}, got {hidden_size}")
        if total_tokens % patches_per_view != 0:
            raise ValueError(
                f"Total replacement tokens ({total_tokens}) must be divisible by patches_per_view ({patches_per_view})"
            )

        num_views = total_tokens // patches_per_view
        view_tokens = aux_features.reshape(batch_size, num_views, patches_per_view, hidden_size)
        source_grid_size = self._square_grid_size(patches_per_view)
        dropped_cls = False

        if source_grid_size is None:
            source_grid_size = self._square_grid_size(patches_per_view - 1)
            if source_grid_size is None:
                raise ValueError(
                    "Replacement encoder token count per view must be square, or square after dropping one CLS token; "
                    f"got {patches_per_view}"
                )
            view_tokens = view_tokens[:, :, 1:, :]
            dropped_cls = True

        return view_tokens, source_grid_size, dropped_cls

    def forward(self, aux_features: torch.Tensor, patches_per_view: int) -> torch.Tensor:
        view_tokens, source_grid_size, _ = self.tokens_to_view_grid(aux_features, patches_per_view)
        batch_size, num_views, _, hidden_size = view_tokens.shape

        if source_grid_size != self.target_grid_size:
            view_tokens = view_tokens.reshape(batch_size * num_views, source_grid_size, source_grid_size, hidden_size)
            view_tokens = view_tokens.permute(0, 3, 1, 2)
            view_tokens = F.interpolate(
                view_tokens,
                size=(self.target_grid_size, self.target_grid_size),
                mode="bilinear",
                align_corners=False,
            )
            view_tokens = view_tokens.permute(0, 2, 3, 1).reshape(
                batch_size,
                num_views,
                self.target_tokens_per_view,
                hidden_size,
            )

        view_tokens = view_tokens.reshape(batch_size, num_views * self.target_tokens_per_view, hidden_size)
        view_tokens = view_tokens.to(dtype=self.projector.weight.dtype)
        return self.projector(self.input_norm(view_tokens))


class PI0EncoderReplace(PI0Pytorch):
    EXPECTED_BASE_MISSING_PREFIXES = (
        "replacement_encoder.",
        "image_token_adapter.",
    )
    LORA_VARIANTS: ClassVar[set[str]] = {"v2", "v3"}
    GEMMA_LORA_TARGET_SUFFIXES: ClassVar[tuple[str, ...]] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    VISION_LORA_TARGET_SUFFIXES: ClassVar[tuple[str, ...]] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "query",
        "key",
        "value",
        "dense",
        "fc1",
        "fc2",
        "weights_in",
        "weights_out",
    )

    def __init__(self, config):
        super().__init__(config)
        self.replacement_encoder = self._create_replacement_encoder(config)
        self.encoder_replace_variant = getattr(config, "encoder_replace_variant", "v1")

        target_tokens_per_view = self._target_image_tokens_per_view()
        target_hidden_size = self._target_image_hidden_size()
        self.image_token_adapter = VisionTokenAdapter(
            self.replacement_encoder.hidden_size,
            target_hidden_size,
            target_tokens_per_view,
        )
        reference_std = (
            self.paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight.detach().float().std().item()
        )
        self.image_token_adapter.reset_parameters(reference_std=reference_std)
        self._apply_lora_for_variant(config)
        self._freeze_pretrained_components()

        logger.info(
            "Initialized PI0EncoderReplace: encoder_type=%s variant=%s hidden=%s target_tokens_per_view=%s "
            "target_hidden=%s",
            getattr(config, "encoder_replace_encoder_type", "dinov2"),
            self.encoder_replace_variant,
            self.replacement_encoder.hidden_size,
            target_tokens_per_view,
            target_hidden_size,
        )

    @staticmethod
    def _normalize_encoder_type(encoder_type: str) -> str:
        if encoder_type == "dino":
            return "dinov2"
        return encoder_type

    @classmethod
    def _create_replacement_encoder(cls, config):
        encoder_type = cls._normalize_encoder_type(getattr(config, "encoder_replace_encoder_type", "dinov2"))
        encoder_name = getattr(config, "encoder_replace_encoder_name", "facebook/dinov2-base")

        if encoder_type == "dinov2":
            return DinoAuxEncoder(encoder_name)
        if encoder_type == "siglip":
            return SigLIPAuxEncoder(encoder_name)
        if encoder_type == "hpr":
            return HPRAuxEncoder(encoder_name)
        if encoder_type == "clip":
            return CLIPAuxEncoder(encoder_name)
        if encoder_type == "r3m":
            return R3MAuxEncoder(encoder_name)
        raise ValueError(
            f"Unknown encoder_replace_encoder_type: {encoder_type}. "
            "Must be one of: 'dinov2', 'dino', 'siglip', 'hpr', 'clip', 'r3m'"
        )

    def _target_image_tokens_per_view(self) -> int:
        vision_embeddings = self.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings
        target_tokens_per_view = getattr(vision_embeddings, "num_positions", None)
        if target_tokens_per_view is None:
            image_size = self.paligemma_with_expert.paligemma.config.vision_config.image_size
            patch_size = self.paligemma_with_expert.paligemma.config.vision_config.patch_size
            target_tokens_per_view = (image_size // patch_size) ** 2

        target_grid_size = math.isqrt(target_tokens_per_view)
        if target_grid_size * target_grid_size != target_tokens_per_view:
            raise ValueError(f"PaliGemma image token count must be square, got {target_tokens_per_view}")
        return target_tokens_per_view

    def _target_image_hidden_size(self) -> int:
        return self.paligemma_with_expert.paligemma.model.multi_modal_projector.linear.out_features

    @staticmethod
    def _set_lora_requires_grad(module: nn.Module, *, requires_grad: bool) -> None:
        for name, param in module.named_parameters():
            if ".lora_" in name or name.startswith("lora_"):
                param.requires_grad = requires_grad

    def _apply_lora_for_variant(self, config) -> None:
        if self.encoder_replace_variant not in self.LORA_VARIANTS:
            self._encoder_replace_lora_modules: dict[str, list[str]] = {}
            return

        encoder_type = self._normalize_encoder_type(getattr(config, "encoder_replace_encoder_type", "dinov2"))
        if encoder_type == "r3m":
            raise ValueError("encoder_replace_variant v2/v3 does not support r3m; R3M needs Conv2d LoRA support")

        rank = getattr(config, "encoder_replace_lora_rank", 16)
        alpha = getattr(config, "encoder_replace_lora_alpha", 16.0)
        action_rank = getattr(config, "encoder_replace_action_lora_rank", 32)
        action_alpha = getattr(config, "encoder_replace_action_lora_alpha", 32.0)
        vision_lora_modules = apply_lora_to_linear_modules(
            self.replacement_encoder,
            target_suffixes=self.VISION_LORA_TARGET_SUFFIXES,
            rank=rank,
            alpha=alpha,
        )
        vlm_lora_modules = apply_lora_to_linear_modules(
            self.paligemma_with_expert.paligemma.language_model,
            target_suffixes=self.GEMMA_LORA_TARGET_SUFFIXES,
            rank=rank,
            alpha=alpha,
        )
        action_lora_modules: list[str] = []
        if self.encoder_replace_variant == "v3":
            action_lora_modules = apply_lora_to_linear_modules(
                self.paligemma_with_expert.gemma_expert.model,
                target_suffixes=self.GEMMA_LORA_TARGET_SUFFIXES,
                rank=action_rank,
                alpha=action_alpha,
            )

        if not vision_lora_modules:
            raise ValueError(f"No vision LoRA target modules were found for encoder type {encoder_type!r}")
        if not vlm_lora_modules:
            raise ValueError("No VLM/LLM LoRA target modules were found under paligemma.language_model")
        if self.encoder_replace_variant == "v3" and not action_lora_modules:
            raise ValueError("No action expert LoRA target modules were found under gemma_expert.model")

        if hasattr(self.replacement_encoder, "set_trainable_encoder"):
            self.replacement_encoder.set_trainable_encoder(enabled=True)

        self._encoder_replace_lora_modules = {
            "vision": vision_lora_modules,
            "vlm": vlm_lora_modules,
            "action_expert": action_lora_modules,
        }
        logger.info(
            "Applied encoder-replacement LoRA: variant=%s rank=%s alpha=%s action_rank=%s action_alpha=%s "
            "vision_modules=%s vlm_modules=%s action_expert_modules=%s",
            self.encoder_replace_variant,
            rank,
            alpha,
            action_rank,
            action_alpha,
            len(vision_lora_modules),
            len(vlm_lora_modules),
            len(action_lora_modules),
        )

    def _freeze_pretrained_components(self) -> None:
        for param in self.paligemma_with_expert.paligemma.parameters():
            param.requires_grad = False
        for param in self.paligemma_with_expert.gemma_expert.parameters():
            param.requires_grad = False
        for param in self.replacement_encoder.parameters():
            param.requires_grad = False

        if self.encoder_replace_variant in self.LORA_VARIANTS:
            self._set_lora_requires_grad(self.replacement_encoder, requires_grad=True)
            self._set_lora_requires_grad(
                self.paligemma_with_expert.paligemma.language_model,
                requires_grad=True,
            )

        if self.encoder_replace_variant == "v3":
            self._set_lora_requires_grad(
                self.paligemma_with_expert.gemma_expert.model,
                requires_grad=True,
            )
        else:
            for param in self.paligemma_with_expert.gemma_expert.model.parameters():
                param.requires_grad = True

        for param in self.image_token_adapter.parameters():
            param.requires_grad = True

        self.paligemma_with_expert.paligemma.eval()
        self.replacement_encoder.eval()
        self._validate_trainable_policy()

    def _validate_trainable_policy(self) -> None:
        if self.encoder_replace_variant == "v1":
            return

        vision_lora = lora_parameter_names(self.replacement_encoder)
        vlm_lora = lora_parameter_names(self.paligemma_with_expert.paligemma.language_model)
        if not vision_lora:
            raise ValueError("LoRA variant requires replacement encoder LoRA parameters")
        if not vlm_lora:
            raise ValueError("LoRA variant requires VLM/LLM LoRA parameters")
        if self.encoder_replace_variant == "v3" and not lora_parameter_names(
            self.paligemma_with_expert.gemma_expert.model
        ):
            raise ValueError("variant v3 requires action expert LoRA parameters")

    def train(self, mode: bool = True):  # noqa: FBT001, FBT002
        super().train(mode)
        self.paligemma_with_expert.paligemma.eval()
        self.replacement_encoder.eval()
        return self

    def gradient_checkpointing_enable(self):
        super().gradient_checkpointing_enable()
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False

    @classmethod
    def is_expected_base_missing_key(cls, key: str) -> bool:
        return key.startswith(cls.EXPECTED_BASE_MISSING_PREFIXES) or ".lora_" in key

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embs = []
        pad_masks = []
        att_masks = []

        aux_features, patches_per_view = self.replacement_encoder(images)
        img_emb = self.image_token_adapter(aux_features, patches_per_view)
        bsize, num_img_embs = img_emb.shape[:2]
        expected_img_embs = len(images) * self.image_token_adapter.target_tokens_per_view
        if num_img_embs != expected_img_embs:
            raise ValueError(f"Expected {expected_img_embs} replacement image tokens, got {num_img_embs}")

        embs.append(img_emb)
        pad_masks.append(
            torch.cat(
                [
                    img_mask[:, None].expand(bsize, self.image_token_adapter.target_tokens_per_view)
                    for img_mask in img_masks
                ],
                dim=1,
            )
        )
        att_masks += [0] * num_img_embs

        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    @torch.no_grad()
    def check_image_token_shapes(self, observation) -> dict[str, int | bool]:
        images, img_masks, lang_tokens, lang_masks, _ = self._preprocess_observation(observation, train=False)
        del img_masks, lang_tokens, lang_masks

        aux_features, patches_per_view = self.replacement_encoder(images)
        _, source_grid_size, dropped_cls = self.image_token_adapter.tokens_to_view_grid(aux_features, patches_per_view)
        image_embeddings = self.image_token_adapter(aux_features, patches_per_view)
        return {
            "encoder_replace/num_views": len(images),
            "encoder_replace/source_patches_per_view": patches_per_view,
            "encoder_replace/source_grid_size": source_grid_size,
            "encoder_replace/dropped_cls_token": dropped_cls,
            "encoder_replace/target_tokens_per_view": self.image_token_adapter.target_tokens_per_view,
            "encoder_replace/target_hidden_size": self.image_token_adapter.output_dim,
            "encoder_replace/image_prefix_tokens": image_embeddings.shape[1],
        }
