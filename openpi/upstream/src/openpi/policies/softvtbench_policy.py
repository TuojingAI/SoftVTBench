import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(255 * image, 0, 255).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SoftVTBenchInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    tactile_prefix_dim: int = 396
    tactile_suffix_dim: int = 396
    tactile_suffix_history: int = 2
    action_dim: int = 7

    def _extract_tactile_prefix(self, data: dict) -> np.ndarray:
        marker = data.get("observation/tactile_marker_motion", data.get("tactile_marker_motion"))
        if marker is None:
            return np.zeros((self.tactile_prefix_dim,), dtype=np.float32)
        marker = np.asarray(marker, dtype=np.float32)
        # Dataset stores (1 + H, 2*M, 2): row 0 is initial reference, rows 1: are current history.
        prefix = marker[-1].reshape(-1) if marker.ndim >= 3 else marker.reshape(-1)
        if prefix.shape[-1] < self.tactile_prefix_dim:
            prefix = np.pad(prefix, (0, self.tactile_prefix_dim - prefix.shape[-1]))
        return prefix[-self.tactile_prefix_dim :].astype(np.float32)

    def _extract_tactile_suffix(self, data: dict) -> np.ndarray:
        marker = data.get("observation/tactile_marker_motion", data.get("tactile_marker_motion"))
        if marker is None:
            return np.zeros((self.tactile_suffix_history, self.tactile_suffix_dim), dtype=np.float32)
        marker = np.asarray(marker, dtype=np.float32)
        # Drop the initial reference row and encode only current marker history.
        if marker.ndim >= 4:
            suffix = marker[1:].reshape(marker.shape[0] - 1, -1)
        elif marker.ndim == 3:
            suffix = marker[1:].reshape(marker.shape[0] - 1, -1)
        elif marker.ndim == 2:
            suffix = marker.reshape(1, -1)
        else:
            suffix = marker.reshape(1, -1)
        if suffix.shape[0] == 0:
            suffix = np.zeros((1, self.tactile_suffix_dim), dtype=np.float32)
        if suffix.shape[-1] < self.tactile_suffix_dim:
            suffix = np.pad(suffix, ((0, 0), (0, self.tactile_suffix_dim - suffix.shape[-1])))
        suffix = suffix[:, -self.tactile_suffix_dim :]
        if suffix.shape[0] < self.tactile_suffix_history:
            pad = np.repeat(suffix[:1], self.tactile_suffix_history - suffix.shape[0], axis=0)
            suffix = np.concatenate([pad, suffix], axis=0)
        return suffix[-self.tactile_suffix_history :].astype(np.float32)

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data.get("observation/image", data.get("image")))
        wrist_image = _parse_image(data.get("observation/wrist_image", data.get("wrist_image")))
        tactile_image = _parse_image(data.get("observation/tactile_image", data.get("tactile_image")))
        state = np.asarray(data.get("observation/state", data.get("state")), dtype=np.float32)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": tactile_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "tactile_prefix": self._extract_tactile_prefix(data),
            "tactile_suffix": self._extract_tactile_suffix(data),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)[..., : self.action_dim]

        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class SoftVTBenchOutputPlaceholders(transforms.DataTransformFn):
    tactile_prefix_dim: int = 396
    tactile_suffix_dim: int = 396
    tactile_suffix_history: int = 2

    def __call__(self, data: dict) -> dict:
        data.setdefault("tactile_prefix", np.zeros((self.tactile_prefix_dim,), dtype=np.float32))
        data.setdefault(
            "tactile_suffix",
            np.zeros((self.tactile_suffix_history, self.tactile_suffix_dim), dtype=np.float32),
        )
        return data


@dataclasses.dataclass(frozen=True)
class SoftVTBenchOutputs(transforms.DataTransformFn):
    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
