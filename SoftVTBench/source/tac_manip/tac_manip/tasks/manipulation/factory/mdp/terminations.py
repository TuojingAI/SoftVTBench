# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_a_is_into_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object_a"),
    object_b_cfg: SceneEntityCfg = SceneEntityCfg("object_b"),
    xy_threshold: float = 0.03,  # xy_distance_threshold
    height_threshold: float = 0.04,  # height_distance_threshold
    height_diff: float = 0.0,  # expected height_diff
) -> torch.Tensor:
    """Check if an object a is put into another object b by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    object_a: RigidObject = env.scene[object_a_cfg.name]
    object_b: RigidObject = env.scene[object_b_cfg.name]

    pos_diff = object_a.data.root_pos_w - object_b.data.root_pos_w
    # print("object_a.data.root_pos_w: ", object_a.data.root_pos_w, "object_b.data.root_pos_w: ", object_b.data.root_pos_w)
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    # print("pos_diff: ", pos_diff, "xy_dist: ", xy_dist, "height_dist: ", height_dist)

    success = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

    gripper_joint_ids, _ = robot.find_joints(env.unwrapped.cfg.gripper_joint_names)
    success = torch.logical_and(
        success,
        torch.abs(
            torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]]) - env.unwrapped.cfg.gripper_open_val.to(env.device)
        )
        < env.unwrapped.cfg.gripper_threshold,
    )
    success = torch.logical_and(
        success,
        torch.abs(
            torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]]) - env.unwrapped.cfg.gripper_open_val.to(env.device)
        )
        < env.unwrapped.cfg.gripper_threshold,
    )
    # print("position with gripper object_a_is_into_b: ", success)
    return success


def reset_timely(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    # 50 timestep to reset
    return env.episode_length_buf * 12 >= env.max_episode_length
