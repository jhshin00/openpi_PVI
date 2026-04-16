import copy
import logging

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers.models.gemma import modeling_gemma

from openpi.models_pytorch.pi0_pytorch import get_prefix_weights
from openpi.models_pytorch.pi0_pytorch import get_rtc_guidance_weight
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import PrefixAttentionSchedule
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.pvi_modules import DinoAuxEncoder
from openpi.models_pytorch.pvi_modules import SigLIPAuxEncoder
from openpi.models_pytorch.pvi_modules import HPRAuxEncoder
from openpi.models_pytorch.pvi_modules import CLIPAuxEncoder
from openpi.models_pytorch.pvi_modules import R3MAuxEncoder
from openpi.models_pytorch.pvi_modules import ZeroInitLinear

logger = logging.getLogger("openpi")


class ResidualAuxAdaptor(nn.Module):
    """Lightweight layer-wise adaptor that lets the shared DINO tokens evolve across copy layers."""

    def __init__(self, hidden_size: int, bottleneck_size: int = 256, residual_scale: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, bottleneck_size)
        self.fc2 = nn.Linear(bottleneck_size, hidden_size)
        self.residual_scale = residual_scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.fc1.reset_parameters()
        # Start close to identity so the copy branch stays stable while still receiving gradients immediately.
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_scale * self.fc2(F.silu(self.fc1(self.norm(x))))


class PI0PVI(PI0Pytorch):
    DEFAULT_INJECTOR_INIT_STD = 1e-4
    EXPECTED_BASE_MISSING_PREFIXES = (
        "aux_encoder.",
        "aux_input_norm.",
        "aux_projector.",
        "aux_adaptors.",
        "copy_expert.",
        "copy_conditioners.",
        "injectors.",
    )

    def __init__(self, config):
        super().__init__(config)
        copy_branch_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        self.injector_init_std = getattr(config, "pvi_injector_init_std", self.DEFAULT_INJECTOR_INIT_STD)
        self.aux_encoder = self._create_aux_encoder(config)
        # Normalize frozen auxiliary features before projection to keep the long-prefix backward path stable.
        self.aux_input_norm = nn.LayerNorm(self.aux_encoder.hidden_size, elementwise_affine=False)
        self.aux_projector = ZeroInitLinear(
            self.aux_encoder.hidden_size,
            self.paligemma_with_expert.paligemma.config.text_config.hidden_size,
        )
        hidden_size = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
        # Match the trainable PVI copy branch to the configured training precision to reduce memory pressure.
        self.copy_expert = copy.deepcopy(self.paligemma_with_expert.gemma_expert.model).to(dtype=copy_branch_dtype)
        self._restore_float32_norm_precision(self.copy_expert)
        self.copy_conditioners = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "input_layernorm": copy.deepcopy(
                            self.paligemma_with_expert.paligemma.language_model.layers[layer_idx].input_layernorm
                        ).to(dtype=copy_branch_dtype),
                        "k_proj": self._make_fresh_linear_like(
                            self.paligemma_with_expert.paligemma.language_model.layers[layer_idx].self_attn.k_proj,
                            copy_branch_dtype,
                        ),
                        "v_proj": self._make_fresh_linear_like(
                            self.paligemma_with_expert.paligemma.language_model.layers[layer_idx].self_attn.v_proj,
                            copy_branch_dtype,
                        ),
                    }
                )
                for layer_idx in range(self.copy_expert.config.num_hidden_layers)
            ]
        )
        self._restore_float32_norm_precision(self.copy_conditioners)
        self.aux_adaptors = nn.ModuleList(
            [
                ResidualAuxAdaptor(hidden_size).to(dtype=copy_branch_dtype)
                for _ in range(self.copy_expert.config.num_hidden_layers)
            ]
        )
        self.injectors = nn.ModuleList(
            [
                ZeroInitLinear(self.copy_expert.config.hidden_size, self.copy_expert.config.hidden_size)
                for _ in range(self.copy_expert.config.num_hidden_layers)
            ]
        )
        for injector in self.injectors:
            self._reset_injector_parameters(injector)
        self._freeze_pretrained_components()

    @staticmethod
    def _restore_float32_norm_precision(module: nn.Module) -> None:
        # Match the base model precision policy: Gemma RMSNorm weights and adaRMS modulators stay in float32.
        for submodule in module.modules():
            if isinstance(submodule, modeling_gemma.GemmaRMSNorm):
                submodule.to(dtype=torch.float32)

    @staticmethod
    def _create_aux_encoder(config):
        """Auxiliary encoder를 config에 따라 생성"""
        encoder_type = getattr(config, "pvi_aux_encoder_type", "dinov2")
        encoder_name = config.pvi_aux_encoder_name

        if encoder_type == "dinov2":
            return DinoAuxEncoder(encoder_name)
        elif encoder_type == "siglip":
            return SigLIPAuxEncoder(encoder_name)
        elif encoder_type == "hpr":
            return HPRAuxEncoder(encoder_name)
        elif encoder_type == "clip":
            return CLIPAuxEncoder(encoder_name)
        elif encoder_type == "r3m":
            return R3MAuxEncoder(encoder_name)
        else:
            raise ValueError(
                f"Unknown pvi_aux_encoder_type: {encoder_type}. "
                f"Must be one of: 'dinov2', 'siglip', 'hpr', 'clip', 'r3m'"
            )

    @staticmethod
    def _tensor_rms(tensor: torch.Tensor) -> float:
        return float(tensor.detach().float().pow(2).mean().sqrt().cpu())

    @staticmethod
    def _initialize_fresh_linear_like(proj: nn.Linear, ref_proj: nn.Linear) -> None:
        ref_std = ref_proj.weight.detach().float().std().item()
        if ref_std > 0:
            nn.init.normal_(proj.weight, mean=0.0, std=ref_std)
        else:
            nn.init.xavier_uniform_(proj.weight)
        if proj.bias is not None:
            nn.init.zeros_(proj.bias)

    @classmethod
    def _make_fresh_linear_like(cls, ref_proj: nn.Linear, dtype: torch.dtype) -> nn.Linear:
        proj = nn.Linear(
            ref_proj.in_features,
            ref_proj.out_features,
            bias=ref_proj.bias is not None,
        )
        cls._initialize_fresh_linear_like(proj, ref_proj)
        return proj.to(dtype=dtype)

    def _reset_injector_parameters(self, injector: nn.Linear) -> None:
        if self.injector_init_std == 0:
            injector.reset_parameters()
            return
        # A tiny non-zero init lets gradients reach the fresh aux-conditioning stack immediately.
        nn.init.normal_(injector.weight, mean=0.0, std=self.injector_init_std)
        if injector.bias is not None:
            nn.init.zeros_(injector.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.paligemma_with_expert.eval()
        self.aux_encoder.eval()
        return self

    def gradient_checkpointing_enable(self):
        super().gradient_checkpointing_enable()
        # The frozen VLM prefix path does not benefit much from checkpointing in PVI, and recomputation there
        # makes the aux-prefix backward path less numerically stable. Keep checkpointing only on the trainable copy expert.
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        if hasattr(self.copy_expert, "gradient_checkpointing"):
            self.copy_expert.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        super().gradient_checkpointing_disable()
        if hasattr(self.copy_expert, "gradient_checkpointing"):
            self.copy_expert.gradient_checkpointing = False

    @classmethod
    def is_expected_base_missing_key(cls, key: str) -> bool:
        return key.startswith(cls.EXPECTED_BASE_MISSING_PREFIXES)

    def initialize_pvi_from_main_expert(self):
        self.copy_expert.load_state_dict(self.paligemma_with_expert.gemma_expert.model.state_dict())
        for layer_idx, conditioner in enumerate(self.copy_conditioners):
            prefix_layer = self.paligemma_with_expert.paligemma.language_model.layers[layer_idx]
            conditioner["input_layernorm"].load_state_dict(prefix_layer.input_layernorm.state_dict())
            self._initialize_fresh_linear_like(conditioner["k_proj"], prefix_layer.self_attn.k_proj)
            self._initialize_fresh_linear_like(conditioner["v_proj"], prefix_layer.self_attn.v_proj)
        for adaptor in self.aux_adaptors:
            adaptor.reset_parameters()
        self.aux_projector.reset_parameters()
        for injector in self.injectors:
            self._reset_injector_parameters(injector)
        logger.info(
            "Initialized PVI copy branch from the pretrained main expert (injector_init_std=%s)",
            self.injector_init_std,
        )

    def _freeze_pretrained_components(self):
        for param in self.paligemma_with_expert.parameters():
            param.requires_grad = False
        for param in self.aux_encoder.parameters():
            param.requires_grad = False
        self.paligemma_with_expert.eval()
        self.aux_encoder.eval()

    def _embed_auxiliary_prefix(
        self, images: list[torch.Tensor], img_masks: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        aux_features, patches_per_view = self.aux_encoder(images)
        aux_features = self.aux_input_norm(aux_features.to(dtype=self.aux_projector.weight.dtype))
        aux_embeddings = self.aux_projector(aux_features)
        aux_pad_masks = torch.cat(
            [img_mask[:, None].expand(aux_embeddings.shape[0], patches_per_view) for img_mask in img_masks],
            dim=1,
        )
        aux_att_masks = torch.zeros_like(aux_pad_masks, dtype=torch.bool)
        return aux_embeddings, aux_pad_masks, aux_att_masks

    @staticmethod
    def _get_conditioning_modules(condition_provider: nn.Module) -> tuple[nn.Module, nn.Module, nn.Module]:
        if isinstance(condition_provider, nn.ModuleDict):
            return condition_provider["input_layernorm"], condition_provider["k_proj"], condition_provider["v_proj"]
        return condition_provider.input_layernorm, condition_provider.self_attn.k_proj, condition_provider.self_attn.v_proj

    def _compute_condition_key_value_states(
        self,
        condition_tokens: torch.Tensor,
        condition_provider: nn.Module,
        head_dim: int,
        condition_position_ids: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_layernorm, k_proj, v_proj = self._get_conditioning_modules(condition_provider)
        condition_dtype = k_proj.weight.dtype
        if condition_tokens.dtype != condition_dtype:
            condition_tokens = condition_tokens.to(dtype=condition_dtype)
        condition_hidden_states, _ = input_layernorm(condition_tokens, cond=None)
        condition_shape = condition_hidden_states.shape[:-1]
        condition_key_states = k_proj(condition_hidden_states).view(*condition_shape, -1, head_dim).transpose(1, 2)
        condition_value_states = v_proj(condition_hidden_states).view(*condition_shape, -1, head_dim).transpose(1, 2)
        if condition_key_states.dtype != target_dtype:
            condition_key_states = condition_key_states.to(dtype=target_dtype)
            condition_value_states = condition_value_states.to(dtype=target_dtype)
        condition_key_states = self._apply_rotary(condition_key_states, condition_position_ids)
        return condition_key_states, condition_value_states

    def _compute_prefix_hidden_states(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        *,
        requires_grad: bool,
    ) -> tuple[torch.Tensor, ...]:
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Use a boolean 4D mask here so HF Gemma can keep using memory-efficient SDPA/Flash attention.
        # The additive float mask path forced eager attention and blew up memory on long PVI prefixes.
        prefix_att_2d_masks_4d = prefix_att_2d_masks[:, None, :, :]
        language_model = self.paligemma_with_expert.paligemma.language_model

        def forward_prefix():
            outputs = language_model.forward(
                inputs_embeds=prefix_embs,
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            return outputs.hidden_states

        if requires_grad:
            return forward_prefix()
        with torch.no_grad():
            return forward_prefix()

    def _prepare_suffix_attention(
        self,
        prefix_pad_masks: torch.Tensor,
        suffix_pad_masks: torch.Tensor,
        suffix_att_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        suffix_position_ids = torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1
        return self._prepare_attention_masks_4d(full_att_2d_masks), prefix_position_ids, suffix_position_ids

    def _apply_rotary(self, states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        dummy_tensor = torch.zeros(
            states.shape[0],
            states.shape[2],
            states.shape[3],
            device=states.device,
            dtype=states.dtype,
        )
        cos, sin = self.paligemma_with_expert.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return (states * cos) + (modeling_gemma.rotate_half(states) * sin)

    def _suffix_layer_forward(
        self,
        condition_tokens: torch.Tensor,
        condition_provider: nn.Module,
        suffix_hidden_states: torch.Tensor,
        suffix_layer: nn.Module,
        attention_mask: torch.Tensor,
        condition_position_ids: torch.Tensor,
        suffix_position_ids: torch.Tensor,
        adarms_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = suffix_hidden_states
        normed_hidden_states, gate = suffix_layer.input_layernorm(suffix_hidden_states, cond=adarms_cond)

        suffix_shape = normed_hidden_states.shape[:-1]
        head_dim = suffix_layer.self_attn.head_dim
        num_heads = suffix_layer.self_attn.q_proj.weight.shape[0] // head_dim

        suffix_query_states = suffix_layer.self_attn.q_proj(normed_hidden_states).view(*suffix_shape, -1, head_dim).transpose(1, 2)
        suffix_key_states = suffix_layer.self_attn.k_proj(normed_hidden_states).view(*suffix_shape, -1, head_dim).transpose(1, 2)
        suffix_value_states = suffix_layer.self_attn.v_proj(normed_hidden_states).view(*suffix_shape, -1, head_dim).transpose(1, 2)

        condition_key_states, condition_value_states = self._compute_condition_key_value_states(
            condition_tokens,
            condition_provider,
            head_dim,
            condition_position_ids,
            suffix_key_states.dtype,
        )
        suffix_query_states = self._apply_rotary(suffix_query_states, suffix_position_ids)
        suffix_key_states = self._apply_rotary(suffix_key_states, suffix_position_ids)

        key_states = torch.cat([condition_key_states, suffix_key_states], dim=2)
        value_states = torch.cat([condition_value_states, suffix_value_states], dim=2)
        attn_output, _ = modeling_gemma.eager_attention_forward(
            suffix_layer.self_attn,
            suffix_query_states,
            key_states,
            value_states,
            attention_mask,
            suffix_layer.self_attn.scaling,
        )
        attn_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[1], num_heads * head_dim)
        if attn_output.dtype != suffix_layer.self_attn.o_proj.weight.dtype:
            attn_output = attn_output.to(dtype=suffix_layer.self_attn.o_proj.weight.dtype)

        out_emb = suffix_layer.self_attn.o_proj(attn_output)
        out_emb = modeling_gemma._gated_residual(residual, out_emb, gate)  # noqa: SLF001
        residual = out_emb
        out_emb, gate = suffix_layer.post_attention_layernorm(out_emb, cond=adarms_cond)
        if out_emb.dtype != suffix_layer.mlp.up_proj.weight.dtype:
            out_emb = out_emb.to(dtype=suffix_layer.mlp.up_proj.weight.dtype)
        out_emb = suffix_layer.mlp(out_emb)
        return modeling_gemma._gated_residual(residual, out_emb, gate)  # noqa: SLF001

    def _run_pvi_action_expert(
        self,
        main_prefix_hidden_states: tuple[torch.Tensor, ...],
        aux_condition_tokens: torch.Tensor,
        main_attention_mask: torch.Tensor,
        main_prefix_position_ids: torch.Tensor,
        main_suffix_position_ids: torch.Tensor,
        aux_attention_mask: torch.Tensor,
        aux_prefix_position_ids: torch.Tensor,
        aux_suffix_position_ids: torch.Tensor,
        suffix_embs: torch.Tensor,
        adarms_cond: torch.Tensor | None,
        *,
        collect_debug_metrics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        main_expert = self.paligemma_with_expert.gemma_expert.model
        prefix_layers = self.paligemma_with_expert.paligemma.language_model.layers
        main_dtype = main_expert.layers[0].self_attn.q_proj.weight.dtype
        copy_dtype = self.copy_expert.layers[0].self_attn.q_proj.weight.dtype
        main_suffix_hidden_states = suffix_embs.to(dtype=main_dtype) if suffix_embs.dtype != main_dtype else suffix_embs
        copy_suffix_hidden_states = suffix_embs.to(dtype=copy_dtype) if suffix_embs.dtype != copy_dtype else suffix_embs
        current_aux_tokens = aux_condition_tokens.to(dtype=copy_dtype) if aux_condition_tokens.dtype != copy_dtype else aux_condition_tokens
        debug_metrics: dict[str, float] = {}
        control_rms_values: list[float] = []
        control_ratio_values: list[float] = []
        last_layer_idx = len(self.injectors) - 1

        for layer_idx, injector in enumerate(self.injectors):
            main_suffix_hidden_states = self._suffix_layer_forward(
                main_prefix_hidden_states[layer_idx],
                prefix_layers[layer_idx],
                main_suffix_hidden_states,
                main_expert.layers[layer_idx],
                main_attention_mask,
                main_prefix_position_ids,
                main_suffix_position_ids,
                adarms_cond,
            )
            current_aux_tokens = self.aux_adaptors[layer_idx](current_aux_tokens)
            if collect_debug_metrics:
                aux_rms = self._tensor_rms(current_aux_tokens)
                if layer_idx == 0:
                    debug_metrics["pvi/aux_tokens_layer0_rms"] = aux_rms
                if layer_idx == last_layer_idx:
                    debug_metrics["pvi/aux_tokens_last_rms"] = aux_rms
            copy_suffix_hidden_states = self._suffix_layer_forward(
                current_aux_tokens,
                self.copy_conditioners[layer_idx],
                copy_suffix_hidden_states,
                self.copy_expert.layers[layer_idx],
                aux_attention_mask,
                aux_prefix_position_ids,
                aux_suffix_position_ids,
                adarms_cond,
            )
            main_rms = self._tensor_rms(main_suffix_hidden_states) if collect_debug_metrics else None
            control = injector(copy_suffix_hidden_states.to(dtype=injector.weight.dtype))
            if control.dtype != main_suffix_hidden_states.dtype:
                control = control.to(dtype=main_suffix_hidden_states.dtype)
            if collect_debug_metrics:
                control_rms = self._tensor_rms(control)
                control_ratio = control_rms / max(main_rms, 1e-8)
                control_rms_values.append(control_rms)
                control_ratio_values.append(control_ratio)
                if layer_idx == 0:
                    debug_metrics["pvi/control_layer0_rms"] = control_rms
                    debug_metrics["pvi/control_layer0_to_main_ratio"] = control_ratio
                if layer_idx == last_layer_idx:
                    debug_metrics["pvi/control_last_rms"] = control_rms
                    debug_metrics["pvi/control_last_to_main_ratio"] = control_ratio
            main_suffix_hidden_states = main_suffix_hidden_states + control

        if collect_debug_metrics and control_rms_values:
            debug_metrics["pvi/control_mean_rms"] = sum(control_rms_values) / len(control_rms_values)
            debug_metrics["pvi/control_mean_to_main_ratio"] = sum(control_ratio_values) / len(control_ratio_values)
            debug_metrics["pvi/copy_suffix_final_rms"] = self._tensor_rms(copy_suffix_hidden_states)
        main_suffix_hidden_states, _ = main_expert.norm(main_suffix_hidden_states, cond=adarms_cond)
        if collect_debug_metrics:
            debug_metrics["pvi/main_suffix_post_norm_rms"] = self._tensor_rms(main_suffix_hidden_states)
        return main_suffix_hidden_states, debug_metrics

    def _run_main_action_expert(
        self,
        main_prefix_hidden_states: tuple[torch.Tensor, ...],
        main_attention_mask: torch.Tensor,
        main_prefix_position_ids: torch.Tensor,
        main_suffix_position_ids: torch.Tensor,
        suffix_embs: torch.Tensor,
        adarms_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        main_expert = self.paligemma_with_expert.gemma_expert.model
        prefix_layers = self.paligemma_with_expert.paligemma.language_model.layers
        main_dtype = main_expert.layers[0].self_attn.q_proj.weight.dtype
        main_suffix_hidden_states = suffix_embs.to(dtype=main_dtype) if suffix_embs.dtype != main_dtype else suffix_embs

        for layer_idx in range(len(main_expert.layers)):
            main_suffix_hidden_states = self._suffix_layer_forward(
                main_prefix_hidden_states[layer_idx],
                prefix_layers[layer_idx],
                main_suffix_hidden_states,
                main_expert.layers[layer_idx],
                main_attention_mask,
                main_prefix_position_ids,
                main_suffix_position_ids,
                adarms_cond,
            )

        main_suffix_hidden_states, _ = main_expert.norm(main_suffix_hidden_states, cond=adarms_cond)
        return main_suffix_hidden_states

    def _compute_pvi_suffix_output(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        suffix_embs: torch.Tensor,
        suffix_pad_masks: torch.Tensor,
        suffix_att_masks: torch.Tensor,
        adarms_cond: torch.Tensor | None,
        *,
        collect_debug_metrics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        debug_metrics: dict[str, float] = {}
        main_prefix_hidden_states = self._compute_prefix_hidden_states(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            requires_grad=False,
        )
        aux_condition_tokens, aux_prefix_pad_masks, aux_prefix_att_masks = self._embed_auxiliary_prefix(images, img_masks)
        if collect_debug_metrics:
            debug_metrics["pvi/aux_embeddings_rms"] = self._tensor_rms(aux_condition_tokens)

        main_attention_mask, main_prefix_position_ids, main_suffix_position_ids = self._prepare_suffix_attention(
            prefix_pad_masks,
            suffix_pad_masks,
            suffix_att_masks,
        )
        aux_attention_mask, aux_prefix_position_ids, aux_suffix_position_ids = self._prepare_suffix_attention(
            aux_prefix_pad_masks,
            suffix_pad_masks,
            suffix_att_masks,
        )

        suffix_out, layer_debug_metrics = self._run_pvi_action_expert(
            main_prefix_hidden_states,
            aux_condition_tokens,
            main_attention_mask,
            main_prefix_position_ids,
            main_suffix_position_ids,
            aux_attention_mask,
            aux_prefix_position_ids,
            aux_suffix_position_ids,
            suffix_embs,
            adarms_cond,
            collect_debug_metrics=collect_debug_metrics,
        )
        debug_metrics.update(layer_debug_metrics)
        return suffix_out, debug_metrics

    def _pvi_denoise_step(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        main_prefix_hidden_states: tuple[torch.Tensor, ...],
        aux_condition_tokens: torch.Tensor,
        aux_prefix_pad_masks: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)
        suffix_embs = suffix_embs.to(dtype=self.copy_expert.layers[0].self_attn.q_proj.weight.dtype)

        main_attention_mask, main_prefix_position_ids, main_suffix_position_ids = self._prepare_suffix_attention(
            prefix_pad_masks,
            suffix_pad_masks,
            suffix_att_masks,
        )
        aux_attention_mask, aux_prefix_position_ids, aux_suffix_position_ids = self._prepare_suffix_attention(
            aux_prefix_pad_masks,
            suffix_pad_masks,
            suffix_att_masks,
        )

        suffix_out, _ = self._run_pvi_action_expert(
            main_prefix_hidden_states,
            aux_condition_tokens,
            main_attention_mask,
            main_prefix_position_ids,
            main_suffix_position_ids,
            aux_attention_mask,
            aux_prefix_position_ids,
            aux_suffix_position_ids,
            suffix_embs,
            adarms_cond,
            collect_debug_metrics=False,
        )
        return self.action_out_proj(suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32))

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        prefix_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        suffix_dtype = self.copy_expert.layers[0].self_attn.q_proj.weight.dtype
        prefix_embs = prefix_embs.to(dtype=prefix_dtype)
        suffix_embs = suffix_embs.to(dtype=suffix_dtype)

        suffix_out, debug_metrics = self._compute_pvi_suffix_output(
            images,
            img_masks,
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            suffix_embs,
            suffix_pad_masks,
            suffix_att_masks,
            adarms_cond,
            collect_debug_metrics=self.training,
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        if self.training:
            target_rms = self._tensor_rms(u_t)
            pred_rms = self._tensor_rms(v_t)
            debug_metrics["pvi/target_velocity_rms"] = target_rms
            debug_metrics["pvi/pred_velocity_rms"] = pred_rms
            debug_metrics["pvi/pred_to_target_rms_ratio"] = pred_rms / max(target_rms, 1e-8)
            self._set_debug_metrics(debug_metrics)
        else:
            self._set_debug_metrics({})
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def check_main_path_equivalence(
        self,
        observation,
        actions: torch.Tensor,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        *,
        max_diff_tol: float = 5e-3,
        mean_diff_tol: float = 5e-4,
    ) -> dict[str, float]:
        was_training = self.training
        self.eval()
        try:
            images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

            if noise is None:
                noise = self.sample_noise(actions.shape, actions.device)
            if time is None:
                time = self.sample_time(actions.shape[0], actions.device)

            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

            main_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            base_prefix_embs = prefix_embs.to(dtype=main_dtype) if prefix_embs.dtype != main_dtype else prefix_embs
            base_suffix_embs = suffix_embs.to(dtype=main_dtype) if suffix_embs.dtype != main_dtype else suffix_embs

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
            (_, base_suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[base_prefix_embs, base_suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            base_suffix_out = base_suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)
            base_v_t = self.action_out_proj(base_suffix_out)
            base_loss = F.mse_loss(u_t, base_v_t, reduction="none")

            main_prefix_hidden_states = self._compute_prefix_hidden_states(
                base_prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                requires_grad=False,
            )
            main_attention_mask, main_prefix_position_ids, main_suffix_position_ids = self._prepare_suffix_attention(
                prefix_pad_masks,
                suffix_pad_masks,
                suffix_att_masks,
            )
            manual_suffix_out = self._run_main_action_expert(
                main_prefix_hidden_states,
                main_attention_mask,
                main_prefix_position_ids,
                main_suffix_position_ids,
                suffix_embs,
                adarms_cond,
            )
            manual_suffix_out = manual_suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)
            manual_v_t = self.action_out_proj(manual_suffix_out)
            manual_loss = F.mse_loss(u_t, manual_v_t, reduction="none")

            pred_diff = torch.abs(base_v_t - manual_v_t)
            loss_diff = torch.abs(base_loss - manual_loss)
            metrics = {
                "pvi_equiv/base_loss_mean": float(base_loss.mean().cpu()),
                "pvi_equiv/manual_loss_mean": float(manual_loss.mean().cpu()),
                "pvi_equiv/pred_max_diff": float(pred_diff.max().cpu()),
                "pvi_equiv/pred_mean_diff": float(pred_diff.mean().cpu()),
                "pvi_equiv/loss_max_diff": float(loss_diff.max().cpu()),
                "pvi_equiv/loss_mean_diff": float(loss_diff.mean().cpu()),
            }
            metrics["pvi_equiv/passed"] = float(
                metrics["pvi_equiv/loss_max_diff"] <= max_diff_tol and metrics["pvi_equiv/loss_mean_diff"] <= mean_diff_tol
            )
            return metrics
        finally:
            self.train(was_training)

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        bsize = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise((bsize, self.config.action_horizon, self.config.action_dim), device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        prefix_embs = prefix_embs.to(dtype=prefix_dtype)

        aux_condition_tokens, aux_prefix_pad_masks, aux_prefix_att_masks = self._embed_auxiliary_prefix(images, img_masks)
        main_prefix_hidden_states = self._compute_prefix_hidden_states(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            requires_grad=False,
        )

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, expanded_time)
            suffix_embs = suffix_embs.to(dtype=self.copy_expert.layers[0].self_attn.q_proj.weight.dtype)

            main_attention_mask, main_prefix_position_ids, main_suffix_position_ids = self._prepare_suffix_attention(
                prefix_pad_masks,
                suffix_pad_masks,
                suffix_att_masks,
            )
            aux_attention_mask, aux_prefix_position_ids, aux_suffix_position_ids = self._prepare_suffix_attention(
                aux_prefix_pad_masks,
                suffix_pad_masks,
                suffix_att_masks,
            )

            suffix_out, _ = self._run_pvi_action_expert(
                main_prefix_hidden_states,
                aux_condition_tokens,
                main_attention_mask,
                main_prefix_position_ids,
                main_suffix_position_ids,
                aux_attention_mask,
                aux_prefix_position_ids,
                aux_suffix_position_ids,
                suffix_embs,
                adarms_cond,
                collect_debug_metrics=False,
            )
            v_t = self.action_out_proj(suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32))
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def realtime_action(
        self,
        device,
        observation,
        prev_action_chunk: torch.Tensor,
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: PrefixAttentionSchedule = "exp",
        max_guidance_weight: float = 5.0,
        noise: torch.Tensor | None = None,
        num_steps: int = 10,
    ) -> torch.Tensor:
        bsize = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise((bsize, self.config.action_horizon, self.config.action_dim), device)

        prev_action_chunk = prev_action_chunk.to(device=device, dtype=torch.float32)
        if prev_action_chunk.shape != (bsize, self.config.action_horizon, self.config.action_dim):
            raise ValueError(
                f"Expected prev_action_chunk with shape {(bsize, self.config.action_horizon, self.config.action_dim)}, "
                f"got {tuple(prev_action_chunk.shape)}"
            )

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        prefix_embs = prefix_embs.to(dtype=prefix_dtype)

        with torch.no_grad():
            aux_condition_tokens, aux_prefix_pad_masks, _ = self._embed_auxiliary_prefix(images, img_masks)
            main_prefix_hidden_states = self._compute_prefix_hidden_states(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                requires_grad=False,
            )

        prefix_weights = get_prefix_weights(
            inference_delay,
            prefix_attention_horizon,
            self.config.action_horizon,
            prefix_attention_schedule,
            device=device,
        )[None, :, None]

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise.to(dtype=torch.float32, device=device)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            with torch.enable_grad():
                x_t_var = x_t.detach().clone().requires_grad_(True)
                v_t = self._pvi_denoise_step(
                    state,
                    prefix_pad_masks,
                    main_prefix_hidden_states,
                    aux_condition_tokens,
                    aux_prefix_pad_masks,
                    x_t_var,
                    expanded_time,
                )
                action_estimate = x_t_var - expanded_time[:, None, None] * v_t
                error = (prev_action_chunk - action_estimate) * prefix_weights
                pinv_correction = torch.autograd.grad(
                    action_estimate,
                    x_t_var,
                    grad_outputs=error,
                    retain_graph=False,
                    create_graph=False,
                )[0]

            guidance_weight = get_rtc_guidance_weight(expanded_time, max_guidance_weight)[:, None, None]
            corrected_velocity = v_t.detach() + guidance_weight * pinv_correction.detach()
            x_t = x_t + dt * corrected_velocity
            time += dt

        return x_t.detach()
