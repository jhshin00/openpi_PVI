# AGENTS.md

Guidance for Codex (and any other coding agent) working in this repository.

This is a fork of [openpi](https://github.com/Physical-Intelligence/openpi) that adds PyTorch
PVI (Plug-in Visual Injection) work on top of `π₀` / `π₀.₅` and wires
it into RoboTwin. The repo also contains ongoing work for a second PyTorch path that replaces the
base image encoder rather than using the PVI copy-branch mechanism.

If there is any conflict between this file and `CLAUDE.md` or `PVI_readme.md`, prefer this file
for workflow and editing policy, and prefer `PVI_readme.md` for PVI architecture details.

---

## 0. Branch / Scope Awareness

This repository has multiple active branches. Before editing, confirm which branch the user wants.
Do not assume all branches have identical code.

Known active branches include:

- `main`
- `ur3`
- `robotwin`
- `openpi-libero-plus`

Default working assumptions unless the user says otherwise:

- If the task mentions RoboTwin, prefer the `robotwin` branch.
- If the task mentions LIBERO-plus, prefer the `openpi-libero-plus` branch.
- If the task is about the current PVI implementation, inspect the PyTorch code path first.
- Do not merge branch-specific assumptions across branches without checking.

---

## 1. Primary Focus Areas

Unless the user explicitly asks otherwise, focus on the PyTorch path rather than the JAX path.

Primary code areas:

- `src/openpi/models_pytorch/`
  - `pi0_pytorch.py` — baseline PyTorch `PI0Pytorch`; this is the base model class and factory
    entry point.
  - `pi0_pvi_pytorch.py` — current PVI implementation (`PI0PVI`).
  - `pvi_modules.py` — frozen auxiliary vision encoders and helper modules used by PVI.
  - `gemma_pytorch.py`, `preprocessing_pytorch.py`
  - `transformers_replace/` — patches that must be copied into the active `transformers` install.
- `scripts/`
  - `train_pytorch.py` — baseline PyTorch trainer.
  - `train_pytorch_PVI.py` — PVI trainer.
  - `compute_norm_stats.py`
  - `serve_policy.py`
- `src/openpi/training/config.py`
  - named configs used by CLI training and inference.
- `examples/robotwin/`
  - training/eval pipeline for RoboTwin.
- `third_party/robotwin/`
  - RoboTwin submodule.

Treat the following as reference documents that should stay consistent when relevant:

- `README.md`
- `CLAUDE.md`
- `PVI_readme.md`
- `examples/robotwin/README.md`

---

## 2. Current Supported Architecture Paths

There are now **two distinct PyTorch architecture directions** in this repo.

### 2.1 Existing PVI path (already implemented)

This is the current working path described in `PVI_readme.md`.

High-level behavior:

- Start from `π₀.₅`
- Freeze the base VLA backbone
- Use a frozen auxiliary vision encoder
- Train PVI-specific modules such as:
  - copy action expert
  - copy conditioners
  - injection modules
  - selected action-head-side trainable pieces

Relevant files:

- `src/openpi/models_pytorch/pi0_pvi_pytorch.py`
- `src/openpi/models_pytorch/pvi_modules.py`
- `scripts/train_pytorch_PVI.py`

### 2.2 Encoder-replacement path (new work; implement in parallel)

This is a **new path** and must be implemented without breaking the existing PVI code.

Goal:

1. Replace the default `π₀.₅` image encoder path with one of the auxiliary encoders already used in
   PVI:
   - HPR
   - DINOv2
   - CLIP
   - SigLIP
   - R3M
2. Add a projection / adapter module that maps the new encoder output into the token space expected
   by the original PaliGemma vision pathway.
3. Freeze the new image encoder and freeze the VLM backbone.
4. Keep only the projection/adaptation layers and the action expert / action-side trainable pieces
   unfrozen for domain-data fine-tuning.
5. Keep this implementation separate from PVI.

Required new files for this path:

- `src/openpi/models_pytorch/pi0_encoder_replace_pytorch.py`
- `scripts/train_pytorch_encoder_replace.py`

Preferred config naming pattern:

- `pi05_robotwin_encoder_replace_from_base`
- `pi05_robotwin_encoder_replace_{dino,siglip,clip,r3m,hpr}`

Do **not** repurpose `pi0_pvi_pytorch.py` for this new method.

---

## 3. Non-Negotiable Editing Rules

When implementing the encoder-replacement path:

- Do **not** break or silently change the existing PVI path.
- Do **not** rename or repurpose current PVI config names.
- Do **not** overload `PI0PVI` with encoder-replacement logic.
- Do **not** delete or rewrite Korean comments in existing files unless explicitly asked.
- Do **not** introduce new dependencies unless the user explicitly approves.
- Do **not** commit or push unless the user explicitly asks.
- Do **not** edit `uv.lock`, dependency pins, or unrelated JAX files unless required and requested.

Preferred strategy:

- add parallel code paths
- reuse existing helper modules where appropriate
- keep model factory / config dispatch explicit and easy to audit

---

## 4. Design Requirements for Encoder Replacement

This section is critical for new architecture work.

### 4.1 Reuse existing auxiliary encoders

The encoder-replacement implementation should reuse encoder wrappers from:

- `src/openpi/models_pytorch/pvi_modules.py`

Use the same supported auxiliary encoder families already wired for PVI.

### 4.2 Preserve the original backbone expectations

The replacement encoder output must be projected into the token space expected by the original
PaliGemma vision pathway.

This means the implementation must account for at least:

- hidden dimension
- token count / sequence length
- token ordering assumptions
- any CLS-token or patch-token handling differences
- compatibility with downstream prefix processing

Do not assume that “same hidden dimension” alone is sufficient.

### 4.3 Freeze policy

For the encoder-replacement path, the intended training policy is:

Frozen:
- replacement image encoder
- VLM backbone

Trainable:
- projection / adapter module(s)
- action expert
- action-side trainable heads required for the policy path

If implementation details force a slightly different trainable set, state it clearly in the code
change summary before editing.

### 4.4 Separation from PVI

Encoder replacement is **not** PVI.

Do not add:
- copy branch logic
- injector stacks
- PVI-specific conditioners
- PVI-specific init assumptions

unless the user explicitly asks for a hybrid design.

---

## 5. Expected File-Level Responsibilities

### `src/openpi/models_pytorch/pi0_pytorch.py`

Treat this as the baseline model / factory anchor.

It may need **minimal** changes to dispatch to the new encoder-replacement model, but avoid heavy
logic growth here. Keep changes localized and obvious.

Typical acceptable changes:
- import the new model
- extend `create_model(config)` or equivalent dispatch
- add small, explicit branching based on config flags

### `src/openpi/models_pytorch/pi0_encoder_replace_pytorch.py`

This should hold the new model implementation.

Preferred responsibilities:
- new model class for encoder replacement
- replacement encoder hookup
- token projection / adaptation into original vision token space
- freezing / unfreezing policy
- any model-specific init or sanity checks

### `src/openpi/models_pytorch/pvi_modules.py`

This remains the source for reusable auxiliary vision encoders and helper modules.

Acceptable changes:
- add small reusable helper utilities that are broadly useful to both PVI and encoder replacement

Avoid:
- mixing encoder-replacement-specific control flow deeply into PVI-only code

### `scripts/train_pytorch_encoder_replace.py`

This should mirror the useful parts of `train_pytorch_PVI.py` but remain separate.

Preferred responsibilities:
- load the encoder-replacement model path
- preserve any encoder-replacement config fields across resume/rebuild
- handle base checkpoint loading
- keep missing/unexpected-key checks explicit
- report trainable parameter groups clearly

### `src/openpi/training/config.py`

Add new named configs for encoder replacement here.

Rules:
- names must be unique
- keep naming consistent with existing patterns
- do not modify unrelated configs unless necessary

---

## 6. Environment Rules

Two environments are used. Do not mix them.

### 6.1 Train / data environment (repo root, `uv`)

Used for:
- data processing
- LeRobot conversion
- norm stats
- PyTorch training
- checkpoint conversion
- lint / format / tests

Setup:

```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

cp -r ./src/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python3.11/site-packages/transformers/
```

### 6.2 RoboTwin eval environment (`examples/robotwin/.venv`)

Used for:
- RoboTwin simulator eval

Setup:

```bash
bash examples/robotwin/bootstrap_eval_env.sh --download-assets
source examples/robotwin/.venv/bin/activate
```

Rules:
- never run RoboTwin eval from the repo-root train env
- never run training from the RoboTwin eval env

### 6.3 Important environment variables

Commonly relevant:

- `HF_LEROBOT_HOME=/data/jhshin/openpi/datasets`
- `CUDA_VISIBLE_DEVICES=<ids>`
- `HF_HOME`
- `HF_DATASETS_CACHE`
- `XDG_CACHE_HOME`
- `OPENPI_DATA_HOME=<path>` (optional)

---

## 7. Workflow Priorities

### 7.1 For existing PVI work

Use:
- `PVI_readme.md`
- `pi0_pvi_pytorch.py`
- `train_pytorch_PVI.py`

### 7.2 For encoder-replacement work

Use this order:

1. inspect `pi0_pytorch.py`
2. inspect `pvi_modules.py`
3. inspect `train_pytorch_PVI.py`
4. implement the parallel model in `pi0_encoder_replace_pytorch.py`
5. implement the parallel trainer in `train_pytorch_encoder_replace.py`
6. add minimal new config entries
7. only then integrate with `examples/robotwin`

### 7.3 RoboTwin integration timing

For encoder-replacement work, do not start by editing `examples/robotwin/`.

First finish:
- model class
- trainer
- config wiring
- checkpoint loading
- forward-shape sanity checks
- trainable parameter verification

Only after that should RoboTwin workflow wrappers be updated.

---

## 8. Common Failure Modes

Things that often break silently:

- forgetting to copy `transformers_replace/*` into the active environment
- losing custom config fields when rebuilding configs on resume
- accidentally dispatching back to baseline `PI0Pytorch`
- mixing up the train env and RoboTwin eval env
- assuming auxiliary encoder token shape already matches the original PaliGemma vision token shape
- loading a checkpoint with the wrong strictness or without checking missing/unexpected keys
- modifying PVI code for encoder replacement instead of adding a parallel path

For encoder replacement specifically, always verify:

- projected hidden dimension matches the original expected dimension
- projected token count is valid for downstream processing
- positional / token ordering assumptions remain consistent
- only intended modules are trainable

---

## 9. Verification / Done Criteria

A change is not “done” just because it imports.

For model-path changes, the minimum completion bar is:

1. `ruff` passes on edited files
2. imports succeed
3. model factory dispatch selects the intended class
4. checkpoint loading behavior is explicit and checked
5. trainable vs frozen parameters match the intended policy
6. one forward-pass shape sanity check is possible without ambiguity
7. config names are unique and discoverable
8. existing PVI behavior is not broken by the new path

Useful commands:

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
uv run pytest -k "pytorch or pvi"
```

If tests are too heavy to run, at minimum provide:
- edited files
- dispatch path
- trainable module summary
- expected checkpoint compatibility notes
- known unverified parts

---

## 10. Code Style

- Python target: 3.11
- follow the existing `ruff` config
- match surrounding file style
- prefer explicit imports consistent with neighboring files
- prefer editing existing files over creating extra helper files unless there is a clear benefit
- keep comments for non-obvious intent only
- avoid broad refactors when the user asked for a targeted architecture change

---

## 11. What To Tell The User Before Large Edits

Before making architecture changes to any of these files:

- `pi0_pytorch.py`
- `pi0_pvi_pytorch.py`
- `pi0_encoder_replace_pytorch.py`
- `pvi_modules.py`
- `train_pytorch_PVI.py`
- `train_pytorch_encoder_replace.py`
- `src/openpi/training/config.py`

state briefly:

- which files will change
- whether the existing PVI path is untouched
- whether the new path is parallel or invasive
- any checkpoint-compatibility implications

---

## 12. When In Doubt

1. Re-read `PVI_readme.md` for what the current PVI path already does.
2. Re-read `pi0_pytorch.py` before adding new model-dispatch logic.
3. Keep encoder replacement separate from PVI unless the user explicitly wants a hybrid.
4. Prefer the smallest change that preserves clarity and branch-specific behavior.
