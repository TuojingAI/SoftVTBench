# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_xy_threshold: float = 0.06,
    height_threshold: float = 0.10,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot end_effector."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pos_diff = object_pos - end_effector_pos

    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    grasped = torch.logical_and(xy_dist < diff_xy_threshold, height_dist < height_threshold)

    gripper_joint_ids, _ = robot.find_joints(env.unwrapped.cfg.gripper_joint_names)
    grasped = torch.logical_and(
        grasped,
        torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]] - env.unwrapped.cfg.gripper_open_val.to(env.device))
        > env.unwrapped.cfg.gripper_threshold,
    )
    grasped = torch.logical_and(
        grasped,
        torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]] - env.unwrapped.cfg.gripper_open_val.to(env.device))
        > env.unwrapped.cfg.gripper_threshold,
    )
    # print("object_pos: ", object_pos, "end_effector_pos: ", end_effector_pos, "xy_dist: ", xy_dist, "height_dist: ",
    #       height_dist, "grasped: ", grasped)

    return grasped
