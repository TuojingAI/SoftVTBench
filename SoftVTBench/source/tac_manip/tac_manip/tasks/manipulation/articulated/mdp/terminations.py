# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from tac_manip.utils.decorators import subtask_termination

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@subtask_termination
def drawer_is_open(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    drawer_cfg: SceneEntityCfg,
    drawer_open_threshold: float = 0.1,
) -> torch.Tensor:
    """Check if drawer is open and gripper is open."""

    robot: Articulation = env.scene[robot_cfg.name]
    drawer: Articulation = env.scene[drawer_cfg.name]

    # check if the drawer is open
    drawer_joint_ids, _ = drawer.find_joints(drawer_cfg.joint_names)
    drawer_pos = drawer.data.joint_pos[:, drawer_joint_ids[0]]
    opened = drawer_pos > drawer_open_threshold

    # check if the gripper is open
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        opened = torch.logical_and(suction_cup_is_open, opened)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            opened = torch.logical_and(
                opened,
                torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]]) - env.cfg.gripper_open_val)
                < env.cfg.gripper_threshold,
            )
            opened = torch.logical_and(
                opened,
                torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]]) - env.cfg.gripper_open_val)
                < env.cfg.gripper_threshold,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return opened


@subtask_termination
def drawer_is_closed(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    drawer_cfg: SceneEntityCfg,
    drawer_close_threshold: float = 0.02,
) -> torch.Tensor:
    """Check if the drawer is closed and gripper is open."""

    robot: Articulation = env.scene[robot_cfg.name]
    drawer: Articulation = env.scene[drawer_cfg.name]

    # check if the drawer is closed
    drawer_joint_ids, _ = drawer.find_joints(drawer_cfg.joint_names)
    drawer_pos = drawer.data.joint_pos[:, drawer_joint_ids[0]]
    closed = drawer_pos < drawer_close_threshold

    # check if the gripper is open
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        closed = torch.logical_and(suction_cup_is_open, closed)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            closed = torch.logical_and(
                closed,
                torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]]) - env.cfg.gripper_open_val)
                < env.cfg.gripper_threshold,
            )
            closed = torch.logical_and(
                closed,
                torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]]) - env.cfg.gripper_open_val)
                < env.cfg.gripper_threshold,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")
    return closed


@subtask_termination
def obj_is_into_drawer_and_drawer_is_closed(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    drawer_cfg: SceneEntityCfg = SceneEntityCfg(
        "cabinet", joint_names=["drawer_bottom_joint"], body_names=["drawer_bottom"]
    ),
    drawer_close_threshold: float = 0.02,  # threshold for drawer to be closed
    xy_threshold: float = 0.03,  # xy_distance_threshold for object to be into the drawer
    height_threshold: float = 0.04,  # height_distance_threshold for object to be into the drawer
    height_diff: float = 0.0,  # expected height_diff for object to be into the drawer
    force_threshold: float = 1.0,  # force_threshold for object to be into the drawer
) -> torch.Tensor:
    """Check if the object is put into the drawer and the drawer is closed and gripper is open."""

    robot: Articulation = env.scene[robot_cfg.name]
    drawer: Articulation = env.scene[drawer_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # check if the object is into the drawer
    drawer_rigid_body_id, _ = drawer.find_bodies(drawer_cfg.body_names)
    drawer_rigid_body_pos = drawer.data.body_pos_w[:, drawer_rigid_body_id[0]]

    pos_diff = object.data.root_pos_w - drawer_rigid_body_pos
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)

    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    obj_is_into_drawer = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

    if "contact_object" in env.scene.sensors and env.scene["contact_object"] is not None:
        contact_force_object = env.scene["contact_object"].data.force_matrix_w  # shape:(N, 1, 1, 3)
        contact_force_norm = torch.linalg.vector_norm(contact_force_object.squeeze(2).squeeze(1), dim=1)
        obj_is_into_drawer = torch.logical_and(
            obj_is_into_drawer.clone().detach(), contact_force_norm > force_threshold
        )

    # check if the drawer is closed
    drawer_joint_ids, _ = drawer.find_joints(drawer_cfg.joint_names)
    drawer_pos = drawer.data.joint_pos[:, drawer_joint_ids[0]]
    closed = drawer_pos < drawer_close_threshold

    # check if the gripper is open
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        closed = torch.logical_and(suction_cup_is_open, closed)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            closed = torch.logical_and(
                closed,
                torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]]) - env.cfg.gripper_open_val)
                < env.cfg.gripper_threshold,
            )
            closed = torch.logical_and(
                closed,
                torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]]) - env.cfg.gripper_open_val)
                < env.cfg.gripper_threshold,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    success = torch.logical_and(obj_is_into_drawer, closed)
    return success
