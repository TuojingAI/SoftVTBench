# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import glob
import os
import random
import torch
from typing import TYPE_CHECKING


import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import AssetBase
#from isaaclab.utils.datasets import HDF5DatasetFileHandler

# from isaaclab_tasks.manager_based.manipulation.stack.mdp.franka_stack_events import (
#     sample_object_poses,
#     sample_random_color,
# )

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv







def randomize_domelight_color_intensity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    intensity_range: tuple[float, float] | None = None,
    color_variation: float = 0.15,
    base_color: tuple[float, float, float] = (0.75, 0.75, 0.75),
    default_intensity: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
    textures: list[str] = [],
    default_texture: str = "",
):
    """Randomize scene light intensity/color; set texture only for DomeLight."""

    asset: AssetBase = env.scene[asset_cfg.name]
    prim = asset.prims[0]

    intensity_attr = prim.GetAttribute("inputs:intensity")
    color_attr = prim.GetAttribute("inputs:color")
    is_dome_light = prim.GetTypeName() == "DomeLight"
    texture_file_attr = prim.GetAttribute("inputs:texture:file") if is_dome_light else None

    if intensity_attr is not None:
        if intensity_range is None:
            if default_intensity is not None:
                intensity_attr.Set(float(default_intensity))
        else:
            sampled_intensity = random.uniform(intensity_range[0], intensity_range[1])
            intensity_attr.Set(float(sampled_intensity))
    color_attr.Set(tuple(base_color))
    # if color_attr is not None:
    #     if color_variation is None or color_variation <= 0.0:
    #         color_attr.Set(tuple(base_color))
    #     else:
    #         color_attr.Set(sample_random_color(base=base_color, variation=color_variation))

    if texture_file_attr and texture_file_attr.IsValid():
        if textures:
            new_texture = random.sample(textures, 1)[0]
            texture_file_attr.Set(new_texture)
        else:
            texture_file_attr.Set(default_texture)