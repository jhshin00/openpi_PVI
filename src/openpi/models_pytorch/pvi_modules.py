import torch
from torch import nn
from transformers import AutoModel

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
    image_min = float(image.amin())
    image_max = float(image.amax())

    if image_min < -1.01 or image_max > 255.01:
        raise ValueError(f"Unsupported image range for DINO preprocessing: min={image_min}, max={image_max}")

    if image_min < -0.01:
        return (image + 1.0) / 2.0
    if image_max > 1.01:
        return image / 255.0
    return image


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
