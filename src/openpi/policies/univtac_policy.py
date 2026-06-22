import dataclasses
from typing import Literal

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


def _concat_tactile_pair(left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
    """Pack both tactile views into one image slot when using pi0's 3-view layout."""
    left_image = _parse_image(left_image)
    right_image = _parse_image(right_image)
    if left_image.shape[0] != right_image.shape[0]:
        height = min(left_image.shape[0], right_image.shape[0])
        left_image = left_image[:height]
        right_image = right_image[:height]
    return np.concatenate([left_image, right_image], axis=1)


@dataclasses.dataclass(frozen=True)
class UnivTacInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    use_tactile: bool = True
    use_wrist: bool = False
    image_layout: Literal["three_slot", "multi_image"] = "three_slot"

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        if self.image_layout == "multi_image":
            if self.use_wrist:
                images["wrist_0_rgb"] = _parse_image(data["observation/wrist_image"])
                image_masks["wrist_0_rgb"] = np.True_
            if self.use_tactile:
                images["left_tactile_0_rgb"] = _parse_image(data["observation/left_tactile_image"])
                images["right_tactile_0_rgb"] = _parse_image(data["observation/right_tactile_image"])
                image_masks["left_tactile_0_rgb"] = np.True_
                image_masks["right_tactile_0_rgb"] = np.True_
        else:
            if self.use_wrist:
                left_image = _parse_image(data["observation/wrist_image"])
            elif self.use_tactile:
                left_image = _parse_image(data["observation/left_tactile_image"])
            else:
                left_image = np.zeros_like(base_image)

            if self.use_tactile:
                if self.use_wrist:
                    right_image = _concat_tactile_pair(
                        data["observation/left_tactile_image"], data["observation/right_tactile_image"]
                    )
                else:
                    right_image = _parse_image(data["observation/right_tactile_image"])
            else:
                right_image = np.zeros_like(base_image)

            images["left_wrist_0_rgb"] = left_image
            images["right_wrist_0_rgb"] = right_image
            image_masks["left_wrist_0_rgb"] = np.True_ if (self.use_wrist or self.use_tactile) else np.False_
            image_masks["right_wrist_0_rgb"] = np.True_ if self.use_tactile else np.False_

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": images,
            "image_mask": image_masks,
        }
        if inputs["state"].shape[-1] != 8:
            raise ValueError(f"UniVTAC OpenPI expects 8D joint state, got {inputs['state'].shape}")

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
            if inputs["actions"].shape[-1] != 8:
                raise ValueError(f"UniVTAC OpenPI expects 8D qpos actions, got {inputs['actions'].shape}")
        if "prompt" in data:
            inputs["prompt"] = data["prompt"].decode("utf-8") if isinstance(data["prompt"], bytes) else data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class UnivTacOutputs(transforms.DataTransformFn):
    action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
