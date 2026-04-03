import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def make_ur3_example() -> dict:
    return {
        # "joints": np.random.rand(7),
        # "gripper": np.random.rand(1),
        "obsercation/state": np.random.rand(8),
        "observation/base_rgb": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/wrist_rgb": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class UR3Inputs(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # pi0면 mask padding / pi0-fast면 사용 X
        mask_padding = self.model_type == _model.ModelType.PI0

        # state = np.concatenate([data["joints", data["gripper"]]])
        # state = transforms.pad_to_dim(state, self.action_dim)
        state = transforms.pad_to_dim(data["obsercation/state"], self.action_dim)

        base_image = _parse_image(data["observation/base_image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        if "actions" in data:
            # action dimension : 7 dof
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR3Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 7 action dimensions -> return 7 dims
        return {"actions": np.asarray(data["actions"][:, :7])}
