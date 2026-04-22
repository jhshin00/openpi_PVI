# Encoder-Replacement Path

This document summarizes the PyTorch encoder-replacement path for pi0.5.

The encoder-replacement path is separate from PVI. It does not add a PVI copy branch, injectors, or PVI conditioners. Instead, it replaces the original pi0.5 image encoder output path with a replacement auxiliary encoder plus an adapter that maps the replacement encoder tokens into the token space expected by the original PaliGemma vision prefix.

## Main Files

- `src/openpi/models_pytorch/pi0_encoder_replace_pytorch.py`
- `src/openpi/models_pytorch/lora.py`
- `src/openpi/models_pytorch/pvi_modules.py`
- `scripts/train_pytorch_encoder_replace.py`
- `examples/robotwin/workflow/finetune_encoder_replace.py`
- `examples/robotwin/finetune_all_tasks_encoder_replace.sh`

## High-Level Flow

The baseline `PI0Pytorch` prefix path is roughly:

```text
image
-> PaliGemma/SigLIP vision tower
-> image prefix tokens

language tokens
-> language embeddings

concat(image prefix tokens, language embeddings)
-> PaliGemma/Gemma VLM backbone
```

The encoder-replacement path changes only the image-token source:

```text
image
-> replacement encoder, e.g. HPR
-> VisionTokenAdapter
-> PaliGemma-compatible image prefix tokens

language tokens
-> language embeddings

concat(image prefix tokens, language embeddings)
-> original PaliGemma/Gemma VLM backbone
```

The language path and action diffusion / action expert path remain based on the original pi0.5 model.

## Replacement Encoder

`PI0EncoderReplace` subclasses `PI0Pytorch`.

```python
class PI0EncoderReplace(PI0Pytorch):
    ...
```

The replacement encoder is selected from config fields:

```text
encoder_replace_encoder_type = hpr | dinov2 | siglip | clip | r3m
encoder_replace_encoder_name = checkpoint path or HF model name
encoder_replace_variant = v1 | v2 | v3
```

The implementation reuses auxiliary encoder wrappers from `src/openpi/models_pytorch/pvi_modules.py`:

- `DinoAuxEncoder`
- `SigLIPAuxEncoder`
- `CLIPAuxEncoder`
- `R3MAuxEncoder`
- `HPRAuxEncoder`

For the current RoboTwin lift pot HPR setup:

```text
encoder_replace_encoder_type = hpr
encoder_replace_encoder_name = hpr_checkpoints/hpr_fullfinetune_base_lang_trace_negative_mod.ckpt
```

The HPR encoder produces patch tokens. In the current HPR run:

```text
num_views = 3
source_patches_per_view = 256
source_grid_size = 16
replacement hidden size = 768
```

So for three RoboTwin Aloha camera views:

```text
3 views * 256 tokens/view = 768 image tokens
```

## VisionTokenAdapter

The replacement encoder output is not assumed to already match the original PaliGemma image-prefix token space.

For HPR:

```text
replacement tokens: [B, 768, 768]
target tokens:      [B, 768, 2048]
```

`VisionTokenAdapter` maps replacement tokens into the PaliGemma-compatible token space.

The default adapter is a LLaVA-style 2-layer MLP:

```text
LayerNorm(input_dim)
Linear(input_dim -> target_hidden_size)
GELU
Linear(target_hidden_size -> target_hidden_size)
```

For HPR:

```text
LayerNorm(768)
Linear(768 -> 2048)
GELU
Linear(2048 -> 2048)
```

If the replacement encoder token grid does not match the target PaliGemma image token grid, `VisionTokenAdapter` reshapes tokens into a 2D grid and uses bilinear interpolation to match the target grid. For the current HPR setup:

```text
source grid = 16 x 16
target grid = 16 x 16
```

so no interpolation is needed.

## Prefix Embedding Override

`PI0EncoderReplace.embed_prefix(...)` overrides the baseline image embedding path.

The baseline path uses:

```python
self.paligemma_with_expert.embed_image(img)
```

The encoder-replacement path uses:

```python
aux_features, patches_per_view = self.replacement_encoder(images)
img_emb = self.image_token_adapter(aux_features, patches_per_view)
```

Language embedding is kept from the original model:

```python
lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
```

Then the model concatenates image prefix embeddings and language embeddings as before:

```text
concat(image embeddings, language embeddings)
```

The VLM backbone therefore still receives PaliGemma-shaped image prefix tokens, but those tokens are produced by:

```text
replacement encoder + VisionTokenAdapter
```

instead of the original PaliGemma/SigLIP vision tower.

## Freeze And Trainable Policy

Common frozen components:

```text
replacement encoder base weights
PaliGemma/VLM backbone base weights
```

Common trainable components:

```text
VisionTokenAdapter
action-side projection / time modules
```

For pi0.5, action-side modules include:

```text
action_in_proj
action_out_proj
time_mlp_in
time_mlp_out
```

### Variant 1

```text
replacement encoder: frozen
VLM/LLM: frozen
VisionTokenAdapter: trainable
action expert: full trainable
action-side modules: trainable
LoRA: none
```

Current HPR v1 trainable breakdown:

```text
action_expert_non_lora      427,932,672
action_side_other_non_lora    2,165,792
image_token_adapter           5,772,800
---------------------------------------
trainable total             435,871,264
```

### Variant 2

```text
replacement encoder base: frozen
replacement encoder LoRA: trainable
VLM/LLM base: frozen
VLM/LLM LoRA: trainable
VisionTokenAdapter: trainable
action expert: full trainable
action-side modules: trainable
```

Current HPR v2 trainable breakdown with JAX-style Gemma LoRA:

```text
action_expert_non_lora      427,932,672
vlm_lora                     27,869,184
replacement_encoder_lora      2,654,208
action_side_other_non_lora    2,165,792
image_token_adapter           5,772,800
---------------------------------------
trainable total             466,394,656
```

### Variant 3

```text
replacement encoder base: frozen
replacement encoder LoRA: trainable
VLM/LLM base: frozen
VLM/LLM LoRA: trainable
VisionTokenAdapter: trainable
action expert base: frozen
action expert LoRA: trainable
action-side modules: trainable
```

Current HPR v3 trainable breakdown with JAX-style Gemma LoRA:

```text
vlm_lora                     27,869,184
action_expert_lora            22,118,400
replacement_encoder_lora       2,654,208
action_side_other_non_lora     2,165,792
image_token_adapter            5,772,800
----------------------------------------
trainable total              60,580,384
```

## LoRA Implementation

LoRA is implemented in `src/openpi/models_pytorch/lora.py`.

There are two LoRA wrappers:

- `LoRALinear`
- `HeadwiseLoRALinear`

### Replacement Image Encoder LoRA

Replacement image encoder LoRA uses standard `LoRALinear`.

For HPR, the underlying encoder is HF DINOv2-base. LoRA is applied to:

```text
query
key
value
dense
fc1
fc2
```

For HPR / DINOv2-base:

```text
12 layers * 6 modules = 72 LoRA modules
replacement_encoder_lora = 2,654,208 params
```

### Gemma VLM / Action Expert LoRA

Gemma LoRA is implemented to match the official OpenPI JAX Gemma LoRA parameterization.

For Gemma attention:

```text
q_proj / o_proj:
  HeadwiseLoRALinear

k_proj / v_proj / gate_proj / up_proj / down_proj:
  LoRALinear
```

`HeadwiseLoRALinear` is used because official OpenPI JAX Gemma keeps the attention head axis in the LoRA parameters.

For Gemma 2B VLM:

```text
width = 2048
depth = 18
mlp_dim = 16384
num_heads = 8
num_kv_heads = 1
head_dim = 256
rank = 16
```

JAX-style VLM LoRA:

```text
vlm_lora = 27,869,184 params
```

For Gemma 300M action expert:

```text
width = 1024
depth = 18
mlp_dim = 4096
num_heads = 8
num_kv_heads = 1
head_dim = 256
rank = 32
```

JAX-style action expert LoRA:

```text
action_expert_lora = 22,118,400 params
```

## Parameter Count Summary

Current HPR encoder-replacement variants:

```text
v1 total params      = 3,709,110,800
v1 trainable params  =   435,871,264
v1 trainable LoRA    =             0

v2 total params      = 3,739,634,192
v2 trainable params  =   466,394,656
v2 trainable LoRA    =    30,523,392

v3 total params      = 3,761,752,592
v3 trainable params  =    60,580,384
v3 trainable LoRA    =    52,641,792
```

LoRA breakdown:

```text
v2:
  replacement_encoder_lora =  2,654,208
  vlm_lora                 = 27,869,184
  action_expert_lora       = 0
  -------------------------------------
  total LoRA               = 30,523,392

v3:
  replacement_encoder_lora =  2,654,208
  vlm_lora                 = 27,869,184
  action_expert_lora       = 22,118,400
  -------------------------------------
  total LoRA               = 52,641,792
```

## Checkpoint Loading

The encoder-replacement model starts from the pi0.5 base checkpoint:

```text
./checkpoints/pytorch/pi05_base/model.safetensors
```

The encoder-replacement model has modules that do not exist in the base checkpoint:

```text
replacement_encoder.*
image_token_adapter.*
*.lora_a
*.lora_b
```

`scripts/train_pytorch_encoder_replace.py` loads base weights with `strict=False` and treats these missing keys as expected.

Expected missing keys:

```text
v1:
  replacement_encoder.*
  image_token_adapter.*

v2:
  replacement_encoder.*
  image_token_adapter.*
  replacement encoder LoRA keys
  VLM/LLM LoRA keys

v3:
  replacement_encoder.*
  image_token_adapter.*
  replacement encoder LoRA keys
  VLM/LLM LoRA keys
  action expert LoRA keys
```

Important compatibility note:

```text
The v2/v3 Gemma LoRA tensor shapes changed after switching to JAX-style head-wise LoRA.
Old v2/v3 checkpoints created with flattened Linear LoRA are not resume-compatible.
Use overwrite or a new exp_name for JAX-style v2/v3 runs.
```

v1 is unaffected by the LoRA shape change.

## Difference From PVI

PVI and encoder replacement are different architecture paths.

PVI:

```text
base VLA remains
auxiliary encoder branch is added
copy expert / injectors / conditioners inject auxiliary visual information
```

Encoder replacement:

```text
original image encoder output path is replaced
replacement encoder + adapter produce PaliGemma-compatible image prefix tokens
no copy branch
no injector stack
no PVI conditioners
```

In short:

```text
PVI injects auxiliary visual information into the existing backbone.
Encoder replacement replaces the image-token source itself.
```

## Expected Logs

For HPR v1:

```text
Initialized PI0EncoderReplace: encoder_type=hpr variant=v1
trainable_lora=0
vision_non_lora=0
vlm_non_lora=0
action_non_lora=427,932,672
```

For HPR v2:

```text
Applied encoder-replacement LoRA: variant=v2
vision_modules=72
vlm_modules=126
action_expert_modules=0
replacement_encoder_lora=2,654,208
vlm_lora=27,869,184
action_expert_non_lora=427,932,672
```

For HPR v3:

```text
Applied encoder-replacement LoRA: variant=v3
vision_modules=72
vlm_modules=126
action_expert_modules=126
replacement_encoder_lora=2,654,208
vlm_lora=27,869,184
action_expert_lora=22,118,400
action_expert_non_lora=0
```

Common shape check:

```text
encoder_replace/num_views=3
encoder_replace/source_patches_per_view=256
encoder_replace/source_grid_size=16
encoder_replace/dropped_cls_token=False
encoder_replace/target_tokens_per_view=256
encoder_replace/target_hidden_size=2048
encoder_replace/adapter_type_mlp2=True
encoder_replace/image_prefix_tokens=768
```

These values indicate that HPR tokens are being mapped into the expected PaliGemma image-prefix token shape.
