# PVI PyTorch Implementation Notes

This document explains the current PyTorch PVI implementation centered on
[`src/openpi/models_pytorch/pi0_pvi_pytorch.py`](src/openpi/models_pytorch/pi0_pvi_pytorch.py).
It complements [PVI_readme.md](PVI_readme.md), which focuses on setup and command recipes.

## File Map

- `src/openpi/models_pytorch/pi0_pvi_pytorch.py`
  PVI model definition. Inherits from `PI0Pytorch` and replaces only the suffix expert execution path.
- `src/openpi/models_pytorch/pvi_modules.py`
  Auxiliary DINO encoder and zero-init layers used by PVI.
- `src/openpi/models_pytorch/pi0_pytorch.py`
  Baseline PI0 / PI0.5 PyTorch implementation that provides preprocessing, prefix embedding, suffix embedding,
  noise/time sampling, and the final action head.
- `scripts/train_pytorch_PVI.py`
  Training entrypoint that preserves PVI config fields and loads base checkpoints with PVI-aware initialization.

## High-Level Idea

The current implementation keeps the original PI0 / PI0.5 prefix pipeline intact and replaces the baseline
"single action-expert suffix pass" with a dual-path suffix computation:

1. A frozen main branch conditions on the original VLM prefix.
2. A trainable copy branch conditions on DINO-derived auxiliary visual tokens.
3. At every expert layer, the copy branch produces a residual control signal that is injected into the main branch.

The final action prediction still comes from a single suffix output passed through the inherited action head.

## Trainable vs Frozen Parts

### Frozen

- `paligemma_with_expert`
  The pretrained VLM prefix path and the pretrained main action expert are frozen.
- `aux_encoder`
  The DINO auxiliary encoder is frozen.

### Trainable

- `aux_input_norm`
- `aux_projector`
- `copy_expert`
- `copy_conditioners`
- `injectors`
- inherited PI0 heads and adapters:
  `action_in_proj`, `action_out_proj`, and either:
  `state_proj` + `action_time_mlp_*` for PI0,
  or `time_mlp_*` for PI0.5

PVI here is not "only train the new branch". The inherited action heads and suffix-side adapters remain trainable.

## Support Modules

### `ZeroInitLinear`

- Linear layer with zero-initialized weight and bias.
- Used for the auxiliary projector and per-layer injectors.
- At initialization, the auxiliary branch does not perturb the frozen main branch output.

### `DinoAuxEncoder`

- Loads `facebook/dinov2-giant` through Hugging Face `AutoModel`.
- Freezes all parameters.
- Converts images to channels-first `float32`.
- Resizes to `224 x 224` if needed.
- Converts image range to `[0, 1]` and applies ImageNet normalization.
- Removes the CLS token and returns patch tokens.

Returned shape:

```text
aux_features: [B, V * P, d_dino]
patches_per_view: P
```

## `PI0PVI` Structure

`PI0PVI` extends `PI0Pytorch`. It reuses:

- `_preprocess_observation`
- `sample_noise`
- `sample_time`
- `embed_prefix`
- `embed_suffix`
- `_prepare_attention_masks_4d`
- final `action_out_proj`

What changes is the suffix computation after embedding.

### Constructor

`__init__` first builds the baseline PI0 model and then adds:

- `aux_encoder`
  Frozen DINO feature extractor.
- `aux_input_norm`
  Trainable normalization before projection.
- `aux_projector`
  Zero-init projection from DINO hidden size to the PaliGemma text hidden size.
- `copy_expert`
  Deep copy of the pretrained main action expert.
- `copy_conditioners`
  Per-layer conditioning modules with `input_layernorm`, `k_proj`, and `v_proj`.
- `injectors`
  One zero-init linear layer per expert layer.

### Precision Policy

The code uses `config.dtype` for both the main pretrained branch and the PVI copy branch. If training runs in
`bfloat16`, both `copy_expert` and `copy_conditioners` are created in `bfloat16`.

### Train / Eval Behavior

`train()` calls `super().train(mode)` and then forces:

- `paligemma_with_expert.eval()`
- `aux_encoder.eval()`

That keeps the frozen backbone modules in eval mode even during training.

### Gradient Checkpointing Behavior

`gradient_checkpointing_enable()`:

1. calls the baseline method
2. turns checkpointing back off for the PaliGemma language model, vision tower, and frozen main expert
3. leaves checkpointing enabled only on `copy_expert` if supported

Because the PVI suffix path manually executes layers instead of calling one model forward, checkpointing savings are
weaker than in baseline `PI0Pytorch`.

## Initialization From a Base Checkpoint

When `scripts/train_pytorch_PVI.py` loads a base PyTorch checkpoint:

1. it calls `create_model(model_cfg)` so `use_pvi=True` returns `PI0PVI`
2. it loads the base checkpoint with `strict=False`
3. it checks that missing keys are only PVI-specific keys
4. it calls `initialize_pvi_from_main_expert()`

That initialization:

- copies `main_expert` weights into `copy_expert`
- copies each frozen prefix layer's `input_layernorm`, `k_proj`, and `v_proj` into the matching `copy_conditioner`
- resets `aux_projector` to zeros
- resets every injector to zeros

So the copy branch starts from the pretrained prior, while the new fusion interfaces start from zero.

## Core Data Structures

### Prefix Path

From inherited `embed_prefix(...)`:

```text
prefix_embs:      [B, prefix_len, d_prefix]
prefix_pad_masks: [B, prefix_len]
prefix_att_masks: [B, prefix_len]
```

The prefix contains image tokens from the frozen vision tower plus language token embeddings.

### Suffix Path

From inherited `embed_suffix(...)`:

```text
suffix_embs:      [B, suffix_len, d_suffix]
suffix_pad_masks: [B, suffix_len]
suffix_att_masks: [B, suffix_len]
adarms_cond:      None or [B, d_suffix]
```

For PI0, the suffix starts with a state token followed by action tokens. For PI0.5, the suffix contains only action
tokens.

### Auxiliary Path

From `_embed_auxiliary_prefix(...)`:

```text
aux_condition_tokens: [B, aux_prefix_len, d_prefix]
aux_pad_masks:        [B, aux_prefix_len]
aux_att_masks:        [B, aux_prefix_len]
```

The auxiliary tokens are projected into the prefix hidden size, not the action-expert hidden size, because they are
consumed through copied prefix-style `k_proj` and `v_proj` modules.

## Training Forward

High-level flow:

1. preprocess observation into images, language, masks, and state
2. sample flow-matching noise and time if not provided
3. build noisy actions `x_t = t * noise + (1 - t) * actions`
4. compute target velocity `u_t = noise - actions`
5. build the original image+language prefix with `embed_prefix(...)`
6. build the state/action/timestep suffix with `embed_suffix(...)`
7. run `_compute_pvi_suffix_output(...)` instead of the baseline single expert forward
8. slice the last `action_horizon` suffix positions
9. project with `action_out_proj`
10. compute the flow-matching MSE

## Dual-Path Suffix Computation

`_compute_pvi_suffix_output(...)` assembles the inputs for the custom dual-path suffix stack.

### Main Prefix Hidden States

`_compute_prefix_hidden_states(...)` runs only the frozen PaliGemma language model on the prefix with
`output_hidden_states=True`, so each suffix layer can condition on the matching prefix layer hidden state.

### Auxiliary Conditioning Tokens

`_embed_auxiliary_prefix(...)`:

1. extracts DINO patch tokens
2. applies `aux_input_norm`
3. projects with `aux_projector`
4. expands image masks from per-view masks to per-patch masks
5. creates a non-causal auxiliary attention mask

These tokens are not fed through a second transformer stack. They remain raw projected conditioning tokens.

### Condition K/V Construction

`_compute_condition_key_value_states(...)`:

1. takes `input_layernorm`, `k_proj`, and `v_proj`
2. casts conditioning tokens to the conditioner dtype if needed
3. normalizes tokens
4. projects into keys and values
5. reshapes to `[B, n_kv_heads, seq, head_dim]`
6. casts to the target dtype if needed
7. applies rotary embeddings to the keys

For the main branch, conditioning tokens come from frozen prefix hidden states. For the copy branch, conditioning
tokens come from projected DINO features.

### Per-Layer Update

Conceptually, `_run_pvi_action_expert(...)` does:

```python
main = suffix_embs cast to main dtype
copy = suffix_embs cast to copy dtype

for each layer l:
    main = suffix_layer_forward(
        condition_tokens=main_prefix_hidden_states[l],
        condition_provider=frozen_prefix_layer[l],
        suffix_hidden_states=main,
        suffix_layer=frozen_main_expert_layer[l],
    )

    copy = suffix_layer_forward(
        condition_tokens=aux_condition_tokens,
        condition_provider=copy_conditioners[l],
        suffix_hidden_states=copy,
        suffix_layer=copy_expert_layer[l],
    )

    main = main + injector_l(copy)

main = final_norm(main)
return main
```

### `_suffix_layer_forward(...)`

This method manually executes one decoder layer over suffix tokens only:

1. apply layer input norm to suffix states
2. build suffix query, key, and value states
3. build conditioning key and value states
4. apply rotary embeddings to suffix queries and keys
5. concatenate conditioning K/V with suffix K/V
6. run `modeling_gemma.eager_attention_forward(...)`
7. run output projection, residual connection, post-attention norm, and MLP

Only suffix tokens are updated. Prefix and auxiliary tokens remain static conditioning memory.

## Inference Path

Baseline `PI0Pytorch` inference uses prefix KV cache from a standard model forward. PVI does not.

`PI0PVI.sample_actions(...)` instead precomputes:

- `main_prefix_hidden_states`
- `aux_condition_tokens`
- their masks and position metadata

Then, inside the Euler denoising loop, it repeatedly:

1. rebuilds the suffix embedding for current `x_t`
2. rebuilds suffix attention masks
3. runs `_run_pvi_action_expert(...)`
4. projects to `v_t`
5. updates `x_t = x_t + dt * v_t`

## Key Methods

- `train`
  Keeps frozen backbone modules in eval mode.
- `gradient_checkpointing_enable` / `disable`
  Toggles checkpoint flags and disables them again on frozen main modules.
- `initialize_pvi_from_main_expert`
  Bootstraps the PVI copy branch from pretrained main weights.
- `_embed_auxiliary_prefix`
  Produces projected DINO patch-token conditioning sequence.
- `_compute_prefix_hidden_states`
  Extracts frozen main prefix hidden states for every layer.
- `_compute_condition_key_value_states`
  Converts conditioning tokens into per-head K/V tensors.
- `_prepare_suffix_attention`
  Builds additive 4D masks and position ids for suffix attention.
- `_suffix_layer_forward`
  Runs one suffix-only decoder layer with external conditioning memory.
- `_run_pvi_action_expert`
  Core dual-path layer loop with layer-wise injection.
- `_compute_pvi_suffix_output`
  Assembles prefix hidden states, auxiliary tokens, masks, and runs the dual-path expert stack.
- `forward`
  Training path with flow-matching loss.
- `sample_actions`
  Iterative denoising path for inference.

## Relationship to Baseline `PI0Pytorch`

Baseline training forward:

```text
embed_prefix + embed_suffix
-> concatenate prefix and suffix
-> single paligemma_with_expert.forward(...)
-> action_out_proj
```

PVI training forward:

```text
embed_prefix + embed_suffix + DINO auxiliary prefix
-> frozen prefix hidden-state extraction
-> custom dual-path expert loop
-> action_out_proj
```

The inherited preprocessing and suffix embedding logic stay the same. The main change is how suffix tokens are
processed.

## Why This Implementation Is Memory Heavy

The current structure has several memory costs:

1. two suffix branches are active at once
2. per-layer conditioning K/V tensors are materialized for both branches
3. the custom suffix path uses `modeling_gemma.eager_attention_forward(...)`, which materializes attention weights
4. prefix hidden states for every layer are stored in `main_prefix_hidden_states`

PVI is therefore substantially heavier than baseline `PI0Pytorch`, especially for large batch sizes or long prefixes.

## Training Script Differences

Compared to the baseline `train_pytorch.py`, `scripts/train_pytorch_PVI.py` adds three important behaviors:

### 1. Preserve PVI Config Fields

When rebuilding `Pi0Config`, it carries through:

- `use_pvi`
- `pvi_aux_encoder_name`

Without this, the factory would silently create the baseline model.

### 2. Use the Model Factory

It calls:

```python
openpi.models_pytorch.pi0_pytorch.create_model(model_cfg)
```

So `use_pvi=True` returns `PI0PVI`.

### 3. Load Base Checkpoints Non-Strictly

For PVI:

- base weights are loaded with `strict=False`
- missing keys are validated against `EXPECTED_BASE_MISSING_PREFIXES`
- `initialize_pvi_from_main_expert()` is called

This is the bridge that makes "start from baseline checkpoint, then attach PVI branch" work.

## Mental Model Summary

If you only keep one picture in mind, it should be this:

- baseline PI0 learns one suffix decoder conditioned on the original VLM prefix
- current PVI keeps that decoder frozen as the main path
- adds a second trainable suffix decoder conditioned on DINO patch tokens
- converts that second decoder's hidden states into layer-wise residual control
- injects the control into the frozen main path at every layer
- still predicts actions with the same final action head
