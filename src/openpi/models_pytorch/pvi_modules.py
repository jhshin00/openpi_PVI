import os
from pathlib import Path

from filelock import FileLock
import torch
from torch import nn
from transformers import AutoModel, SiglipVisionModel, AutoImageProcessor, CLIPModel

_DINO_MEAN = (0.485, 0.456, 0.406)
_DINO_STD = (0.229, 0.224, 0.225)


class ZeroInitLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


def ensure_channels_first(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected a 4D image tensor, got {image.shape}")
    if image.shape[1] == 3:
        return image
    if image.shape[-1] == 3:
        return image.permute(0, 3, 1, 2)
    raise ValueError(f"Unable to infer channel dimension for image tensor with shape {image.shape}")


def normalize_image_to_unit_interval(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize an image tensor to the [0, 1] range if it is not already in that range.
    """
    image_min = float(image.amin())
    image_max = float(image.amax())

    if image_min < -1.01 or image_max > 255.01:
        raise ValueError(f"Unsupported image range for DINO preprocessing: min={image_min}, max={image_max}")

    if image_min < -0.01:
        return (image + 1.0) / 2.0
    if image_max > 1.01:
        return image / 255.0
    return image


def normalize_image_to_255(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize an image tensor to the [0, 255] range for R3M.
    """
    image_min = float(image.amin())
    image_max = float(image.amax())

    if image_min < -1.01 or image_max > 255.01:
        raise ValueError(f"Unsupported image range for R3M preprocessing: min={image_min}, max={image_max}")

    if image_min < -0.01:
        # [-1, 1] -> [0, 255]
        return ((image + 1.0) / 2.0) * 255.0
    if image_max > 1.01:
        # Already [0, 255]
        return image
    # [0, 1] -> [0, 255]
    return image * 255.0


class DinoAuxEncoder(nn.Module):
    def __init__(self, model_name: str = "facebook/dinov2-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, images: list[torch.Tensor]) -> tuple[torch.Tensor, int]:
        processed = []
        for image in images:
            image = ensure_channels_first(image).to(dtype=torch.float32)
            if image.shape[-2:] != (224, 224):
                image = torch.nn.functional.interpolate(
                    image,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
            # OpenPI batches can arrive as either [-1, 1], [0, 1], or uint8-like [0, 255] floats.
            # DINO expects [0, 1] before ImageNet normalization.
            image = normalize_image_to_unit_interval(image)
            mean = image.new_tensor(_DINO_MEAN).view(1, 3, 1, 1)
            std = image.new_tensor(_DINO_STD).view(1, 3, 1, 1)
            processed.append((image - mean) / std)

        batch_size = processed[0].shape[0]
        num_views = len(processed)
        pixel_values = torch.stack(processed, dim=1).reshape(batch_size * num_views, 3, 224, 224)
        outputs = self.encoder(pixel_values=pixel_values)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        patches_per_view = patch_tokens.shape[1]
        patch_tokens = patch_tokens.reshape(batch_size, num_views * patches_per_view, self.hidden_size)
        return patch_tokens, patches_per_view


class SigLIPAuxEncoder(nn.Module):
    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        """
        model_name: google/siglip-base-patch16-224
        """
        super().__init__()
        self.encoder = SiglipVisionModel.from_pretrained(model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        self._SigLIP_MEAN = self.image_processor.image_mean
        self._SigLIP_STD = self.image_processor.image_std

        self.hidden_size = self.encoder.config.hidden_size
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, images: list[torch.Tensor]) -> tuple[torch.Tensor, int]:
        processed = []
        for image in images:
            image = ensure_channels_first(image).to(dtype=torch.float32)
            if image.shape[-2:] != (224, 224):
                image = torch.nn.functional.interpolate(
                    image,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
            # OpenPI batches can arrive as either [-1, 1], [0, 1], or uint8-like [0, 255] floats.
            # SigLIP expects [0, 1] before normalization.
            image = normalize_image_to_unit_interval(image)
            mean = image.new_tensor(self._SigLIP_MEAN).view(1, 3, 1, 1)
            std = image.new_tensor(self._SigLIP_STD).view(1, 3, 1, 1)
            processed.append((image - mean) / std)

        batch_size = processed[0].shape[0]
        num_views = len(processed)
        pixel_values = torch.stack(processed, dim=1).reshape(batch_size * num_views, 3, 224, 224)
        outputs = self.encoder(pixel_values=pixel_values)
        patch_tokens = outputs.last_hidden_state
        patches_per_view = patch_tokens.shape[1]
        patch_tokens = patch_tokens.reshape(batch_size, num_views * patches_per_view, self.hidden_size)

        return patch_tokens, patches_per_view


class CLIPAuxEncoder(nn.Module):
    _CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    _CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.encoder = CLIPModel.from_pretrained(model_name).vision_model
        self.hidden_size = self.encoder.config.hidden_size
        for param in self.encoder.parameters():
            param.requires_grad = False
        
    @torch.no_grad()
    def forward(self, images: list[torch.Tensor]) -> tuple[torch.Tensor, int]:
        processed = []
        for image in images:
            image = ensure_channels_first(image).to(dtype=torch.float32)
            if image.shape[-2:] != (224, 224):
                image = torch.nn.functional.interpolate(
                    image,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
            # OpenPI batches can arrive as either [-1, 1], [0, 1], or uint8-like [0, 255] floats.
            # CLIP expects [0, 1] before normalization.
            image = normalize_image_to_unit_interval(image)
            mean = image.new_tensor(self._CLIP_MEAN).view(1, 3, 1, 1)
            std = image.new_tensor(self._CLIP_STD).view(1, 3, 1, 1)
            processed.append((image - mean) / std)

        batch_size = processed[0].shape[0]
        num_views = len(processed)
        pixel_values = torch.stack(processed, dim=1).reshape(batch_size * num_views, 3, 224, 224)
        outputs = self.encoder(pixel_values=pixel_values)
        patch_tokens = outputs.last_hidden_state
        patches_per_view = patch_tokens.shape[1]
        patch_tokens = patch_tokens.reshape(batch_size, num_views * patches_per_view, self.hidden_size)

        return patch_tokens, patches_per_view


class R3MAuxEncoder(nn.Module):
    def __init__(self, model_name: str = "resnet34"):
        super().__init__()
        try:
            import r3m
        except ImportError as exc:
            missing_module = getattr(exc, "name", None)
            if missing_module == "hydra":
                raise ImportError(
                    "R3M auxiliary encoder requires `hydra-core` in addition to `r3m`. "
                    "Install `hydra-core` or choose a different pvi_aux_encoder_type."
                ) from exc
            raise ImportError(
                "R3M auxiliary encoder requires the optional `r3m` package and its runtime dependencies. "
                "Install them or choose a different pvi_aux_encoder_type."
            ) from exc

        model_dir_names = {
            "resnet18": "r3m_18",
            "resnet34": "r3m_34",
            "resnet50": "r3m_50",
        }
        model_dir_name = model_dir_names.get(model_name)
        if model_dir_name is None:
            raise ValueError(f"Unsupported R3M model_name: {model_name}")

        # R3M lazily downloads weights into ~/.r3m on first use. Serialize that work so
        # multi-process torchrun jobs do not race while creating the same files.
        model_dir = Path.home() / ".r3m" / model_dir_name
        model_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(model_dir / ".download.lock")):
            self.encoder = r3m.load_r3m(modelid=model_name).module

        if model_name == "resnet34":
            self.hidden_size = 512
        elif model_name == "resnet50":
            self.hidden_size = 2048
        elif model_name == "resnet18":
            self.hidden_size = 512
        else:
            raise ValueError(f"Unsupported R3M model_name: {model_name}")
        
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature map from ResNet before global pooling."""
        convnet = self.encoder.convnet
        x = convnet.conv1(x)
        x = convnet.bn1(x)
        x = convnet.relu(x)
        x = convnet.maxpool(x)

        x = convnet.layer1(x)
        x = convnet.layer2(x)
        x = convnet.layer3(x)
        x = convnet.layer4(x)  # (B, C, 7, 7) for 224x224 input
        return x

    @torch.no_grad()
    def forward(self, images: list[torch.Tensor]) -> tuple[torch.Tensor, int]:
        processed = []
        for image in images:
            image = ensure_channels_first(image).to(dtype=torch.float32)
            if image.shape[-2:] != (224, 224):
                image = torch.nn.functional.interpolate(
                    image,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
            # OpenPI batches can arrive as either [-1, 1], [0, 1], or uint8-like [0, 255] floats.
            # R3M expects image input to be [0, 255]
            image = normalize_image_to_255(image)
            processed.append(image)

        batch_size = processed[0].shape[0]
        num_views = len(processed)
        pixel_values = torch.stack(processed, dim=1).reshape(batch_size * num_views, 3, 224, 224)

        # Get feature map before global pooling: (B * num_views, C, 7, 7)
        feature_map = self._forward_features(pixel_values)
        _, C, H, W = feature_map.shape

        # Reshape to patch-like format: (B * num_views, H*W, C)
        patch_tokens = feature_map.flatten(2).transpose(1, 2)  # (B * num_views, 49, C)
        patches_per_view = H * W  # 49 for 224x224 input

        # Reshape to (B, num_views * patches_per_view, hidden_size)
        patch_tokens = patch_tokens.reshape(batch_size, num_views * patches_per_view, self.hidden_size)

        return patch_tokens, patches_per_view


class HPRAuxEncoder(nn.Module):
    _HPR_MEAN = (0.485, 0.456, 0.406)
    _HPR_STD = (0.229, 0.224, 0.225) 

    def __init__(self, hpr_ckpt_path: str = "hpr_checkpoints/hpr_fullfinetune_base_lang_trace_negative_mod.ckpt"):
        super().__init__()

        # Load HPR pretrained rgb_encoder
        if not os.path.exists(hpr_ckpt_path):
            raise FileNotFoundError(f"HPR checkpoint not found at {hpr_ckpt_path}")
        
        print(f"Loading HPR checkpoint from {hpr_ckpt_path}...")
        ckpt = torch.load(hpr_ckpt_path, map_location="cpu")
        hparams = ckpt["hyper_parameters"]
        state_dict = ckpt["state_dict"]

        # Build rgb_encoder from hyperparameters
        self.encoder = self.build_rgb_encoder_from_hparams(hparams)

        # Load HPR pretrained weights into encoder
        rgb_sd = {
            k.replace("rgb_encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("rgb_encoder.")
        }
        incompatible = self.encoder.load_state_dict(rgb_sd, strict=False)
        if incompatible.missing_keys:
            print(get_missing_parameters_message(incompatible.missing_keys))
        if incompatible.unexpected_keys:
            print(get_unexpected_parameters_message(incompatible.unexpected_keys))

        self.hidden_size = self.encoder.embed_dim
        for param in self.encoder.parameters():
            param.requires_grad = False
        
    @torch.no_grad()
    def forward(self, images: list[torch.Tensor]) -> tuple[torch.Tensor, int]:
        processed = []
        for image in images:
            image = ensure_channels_first(image).to(dtype=torch.float32)
            if image.shape[-2:] != (224, 224):
                image = torch.nn.functional.interpolate(
                    image,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
            # OpenPI batches can arrive as either [-1, 1], [0, 1], or uint8-like [0, 255] floats.
            # HPR expects [0, 1] before Normalization.
            image = normalize_image_to_unit_interval(image)
            mean = image.new_tensor(self._HPR_MEAN).view(1, 3, 1, 1)
            std = image.new_tensor(self._HPR_STD).view(1, 3, 1, 1)
            processed.append((image - mean) / std)
        
        batch_size = processed[0].shape[0]
        num_views = len(processed)
        pixel_values = torch.stack(processed, dim=1).reshape(batch_size * num_views, 3, 224, 224)
        outputs = self.encoder(pixel_values, is_training=True)
        cls_token = outputs["x_norm_clstoken"]
        patch_tokens = outputs["x_norm_patchtokens"]
        patches_per_view = patch_tokens.shape[1]
        patch_tokens = patch_tokens.reshape(batch_size, num_views * patches_per_view, self.hidden_size)

        return patch_tokens, patches_per_view

    def build_rgb_encoder_from_hparams(self, hparams):
        """Build RGB encoder from hyperparameters."""
        use_pretrained_dinov2 = hparams.get("use_pretrained_dinov2", False)
        dinov2_freeze = hparams.get("dinov2_freeze", False)
        dinov2_num_adapter_blocks = hparams.get("dinov2_num_adapter_blocks", 4)
        size = hparams.get("size")
        use_hf_dinov2 = hparams.get("use_hf_dinov2", False)

        if use_pretrained_dinov2 and dinov2_freeze:
            if use_hf_dinov2:
                # HuggingFace DINOv2 with adapter
                enc = HFFrozenDinoV2WithAdapter(
                    size=size,
                    num_adapter_blocks=dinov2_num_adapter_blocks,
                )
            else:
                raise NotImplementedError("HPR pretrained DINOv2 with freezing is only implemented for HuggingFace DINOv2. Set use_hf_dinov2=True.")
        elif use_pretrained_dinov2:
            if use_hf_dinov2:
                # HuggingFace DINOv2 without adapter
                enc = HFDinoV2FullFinetune(size=size)
            else:
                raise NotImplementedError("HPR pretrained DINOv2 without freezing is only implemented for HuggingFace DINOv2. Set use_hf_dinov2=True.")
        else:
            raise NotImplementedError("HPR pretrained RGB encoder is only implemented for DINOv2. Set use_pretrained_dinov2=True.")
        
        return enc
    

class HFFrozenDinoV2WithAdapter(nn.Module):
    """
    Hugging Face DINOv2 (frozen) + HF Dinov2Layer as trainable adapter blocks.
    Fully PEFT/LoRA compatible.

    Architecture:
        Image -> [Frozen HF DINOv2] -> last_hidden_state -> [Trainable Dinov2Layer * K] -> [LayerNorm] -> output
    """

    def __init__(
        self,
        size: str = 'base',
        num_adapter_blocks: int = 4,
    ):
        super().__init__()

        from transformers import Dinov2Model
        from transformers.models.dinov2.configuration_dinov2 import Dinov2Config
        from transformers.models.dinov2.modeling_dinov2 import Dinov2Layer

        # HF model map
        hf_model_map = {
            'small': 'facebook/dinov2-small',
            'base': 'facebook/dinov2-base',
            'large': 'facebook/dinov2-large',
        }

        # Load HF DINOv2
        self.backbone = Dinov2Model.from_pretrained(hf_model_map[size])
        config = self.backbone.config
        self.embed_dim = config.hidden_size

        print(f"Loaded HF DINOv2: {hf_model_map[size]}")
        print(f"  hidden_size: {config.hidden_size}, num_heads: {config.num_attention_heads}")

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Trainable adapter blocks using HF Dinov2Layer
        adapter_config = Dinov2Config(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            mlp_ratio=config.mlp_ratio,
            hidden_act=config.hidden_act,
            layerscale_value=config.layerscale_value,
            drop_path_rate=0.0,
        )

        self.adapter_blocks = nn.ModuleList([
            Dinov2Layer(adapter_config)
            for _ in range(num_adapter_blocks)
        ])
        self.adapter_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        print(f"  num_adapter_blocks: {num_adapter_blocks}")

    def train(self, mode=True):
        """Override train to keep backbone frozen."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x, is_training=False):
        """
        Forward pass matching DINOv2 interface.

        Args:
            x: Input images (B, 3, H, W)
            is_training: If True, return dict; else return cls_token only

        Returns:
            If is_training=True: dict with x_norm_clstoken, x_norm_patchtokens
            If is_training=False: cls_token (B, embed_dim)
        """
        # HF forward (expects pixel_values)
        with torch.no_grad():
            outputs = self.backbone(pixel_values=x)

        # Get last hidden state: (B, 1+num_patches, embed_dim)
        hidden_states = outputs.last_hidden_state

        # Pass through trainable adapter blocks (HF Dinov2Layer)
        for layer in self.adapter_blocks:
            layer_outputs = layer(hidden_states)
            hidden_states = layer_outputs[0]  # Dinov2Layer returns tuple

        tokens = self.adapter_norm(hidden_states)

        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]

        if is_training:
            return {
                "x_norm_clstoken": cls_token,
                "x_norm_patchtokens": patch_tokens,
            }
        return cls_token


class HFDinoV2FullFinetune(nn.Module):
    """
    Hugging Face DINOv2 for full fine-tuning.
    PEFT/LoRA compatible - can apply get_peft_model() to self.backbone.

    Architecture:
        Image -> [HF DINOv2 (trainable)] -> last_hidden_state -> output
    """

    def __init__(self, size: str = 'base'):
        super().__init__()

        from transformers import Dinov2Model

        hf_model_map = {
            'small': 'facebook/dinov2-small',
            'base': 'facebook/dinov2-base',
            'large': 'facebook/dinov2-large',
        }

        self.backbone = Dinov2Model.from_pretrained(hf_model_map[size])
        self.embed_dim = self.backbone.config.hidden_size

        print(f"Loaded HF DINOv2 (full finetune): {hf_model_map[size]}")
        print(f"  hidden_size: {self.embed_dim}")

    def forward(self, x, is_training=False):
        """
        Forward pass matching DINOv2 interface.

        Args:
            x: Input images (B, 3, H, W)
            is_training: If True, return dict; else return cls_token only

        Returns:
            If is_training=True: dict with x_norm_clstoken, x_norm_patchtokens
            If is_training=False: cls_token (B, embed_dim)
        """
        outputs = self.backbone(pixel_values=x)

        cls_token = outputs.last_hidden_state[:, 0]
        patch_tokens = outputs.last_hidden_state[:, 1:]

        if is_training:
            return {
                "x_norm_clstoken": cls_token,
                "x_norm_patchtokens": patch_tokens,
            }
        return cls_token
    

from typing import Any, Dict, List
from termcolor import colored
from collections import defaultdict
    
def get_missing_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in 
    the model but not found in a checkpoint.
    Args:
        keys (list[str]): List of keys that were not found in the checkpoint.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = colored(
        "[WARNING] Some model parameters or buffers are not found in the checkpoint:\n",
        color="yellow",
    )
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "blue") for k, v in groups.items()
    )
    return msg


def get_unexpected_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in 
    the checkpoint but not found in the model.
    Args:
        keys (list[str]): List of keys that were not found in the model.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "magenta") for k, v in groups.items()
    )
    return msg

def _group_checkpoint_keys(keys: List[str]) -> Dict[str, List[str]]:
    """
    Group keys based on common prefixes. A prefix is the string up to the final
    "." in the key.
    Args:
        keys (list[str]): list of parameter names, i.e. keys in the model
            checkpoint dict.
    Returns:
        dict[list]: keys with common prefixes are grouped into lists.
    """
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1 :]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups


def _group_to_str(group: List[str]) -> str:
    """
    Format a group of parameter name suffixes into a loggable string.
    Args:
        group (list[str]): list of parameter name suffixes.
    Returns:
        str: formated string.
    """
    if len(group) == 0:
        return ""

    if len(group) == 1:
        return "." + group[0]

    return ".{" + ", ".join(group) + "}"
