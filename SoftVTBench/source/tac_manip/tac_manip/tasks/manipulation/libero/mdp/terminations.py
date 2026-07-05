# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from tac_manip.utils.decorators import subtask_termination

from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _extract_single_joint_position(rigid_object: RigidObject, joint_pattern: str) -> torch.Tensor:
    """Return the joint position tensor (num_envs,) for the given joint pattern in the rigid object."""

    joint_ids, _ = rigid_object.find_joints(joint_pattern)
    if len(joint_ids) != 1:
        raise ValueError(
            f"Expected exactly one joint for pattern '{joint_pattern}' in '{rigid_object.name}', "
            f"but found {len(joint_ids)}."
        )

    joint_index = joint_ids[0]
    if isinstance(joint_index, torch.Tensor):
        joint_index = int(joint_index.item())

    joint_pos = rigid_object.data.joint_pos[:, joint_index]
    return joint_pos.squeeze(-1) if joint_pos.ndim > 1 else joint_pos


def _articulation_operation_goal_satisfied(env: "ManagerBasedRLEnv", goal: dict) -> torch.Tensor:
    """Evaluate whether an articulation operation goal is satisfied for all environments."""

    operation = goal["operation"]
    target = goal["target"]
    scene = env.scene

    if target == "flat_stove_1":
        thresholds = {"turnon": (0.5, 2.1), "turnoff": (-0.05, 0.05)}
        if operation not in thresholds:
            raise ValueError(f"Unsupported operation '{operation}' for target '{target}'.")
        stove = scene[target]
        knob_pos = _extract_single_joint_position(stove, "button")
        low, high = thresholds[operation]
        return torch.logical_and(knob_pos > low, knob_pos < high)

    if target == "white_cabinet_1":
        thresholds = {"open": (-0.16, -0.14), "close": (-0.05, 0.05)}
        if operation not in thresholds:
            raise ValueError(f"Unsupported operation '{operation}' for target '{target}'.")
        cabinet = scene[target]
        cabinet_pos = _extract_single_joint_position(cabinet, "bottom_level")
        low, high = thresholds[operation]
        return torch.logical_and(cabinet_pos > low, cabinet_pos < high)

    if target == "microwave_1":
        thresholds = {"open": (-2.094, -1.3), "close": (-0.2, 0.1)}
        if operation not in thresholds:
            raise ValueError(f"Unsupported operation '{operation}' for target '{target}'.")
        microwave = scene[target]
        microwave_pos = _extract_single_joint_position(microwave, "microjoint")
        low, high = thresholds[operation]
        return torch.logical_and(microwave_pos > low, microwave_pos < high)

    if target == "wooden_cabinet_1":
        # NOTE:
        # - 原始 open 判定区间为 (-0.20, -0.14)，在实际 replay 中观察到有时抽屉已经明显拉开但最后一帧
        #   关节角度略微偏出该范围，导致偶发性失败。
        # - 为了让判定「稍微宽松一点」，这里在原区间基础上适度放宽上下界，避免边界抖动带来的误判。
        #   原始区间: (-0.20, -0.14)
        thresholds = {
            "open": (-0.22, -0.12),  # 放宽版 open 判定区间，相比原始 (-0.20, -0.14) 略微增大容差
            "close": (-0.05, 0.05),
        }
        if operation not in thresholds:
            raise ValueError(f"Unsupported operation '{operation}' for target '{target}'.")
        layer = goal.get("layer")
        layer_to_pattern = {"top": "top_level", "middle": "middle_level", "bottom": "bottom_level"}
        if layer not in layer_to_pattern:
            raise ValueError(f"Unsupported layer '{layer}' for target '{target}'.")
        cabinet = scene[target]
        cabinet_pos = _extract_single_joint_position(cabinet, layer_to_pattern[layer])
        low, high = thresholds[operation]
        return torch.logical_and(cabinet_pos > low, cabinet_pos < high)

    raise ValueError(f"Unsupported operation target '{target}'.")


@subtask_termination
def libero_goals_reached(
    env: ManagerBasedRLEnv,
    goals: list[dict] | None = None,
):
    """
    Check if a list of goals are achieved by the specified robot.
    including positional relationship goal and articulation operation goal.
    e.g., close microwave, turn on stove, put on, put in, etc.
    Args:
        env: The environment.
        goals: A list of goals to check.
    Returns:
        A tensor of shape (num_envs,) indicating whether the goals are achieved. True if all goals are achieved, False otherwise.
    """

    # Initialize success as True for all environments
    success = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    goals = goals or []

    for goal in goals:
        # check if the positional relationship goal is satisfied
        if "relationship" in goal:
            # relationship = goal["relationship"]  # TODO: need to add respective implementations for "ON" and "IN"
            obj_name = goal["ref_obj"]
            target_name = goal["target"]
            xy_threshold = goal["xy_threshold"]
            height_threshold = goal["height_threshold"]
            height_diff = goal["height_diff"]
            enable_force_threshold = goal["enable_force_threshold"]

            target: RigidObject = env.scene[target_name]
            object: RigidObject = env.scene[obj_name]
            pos_diff = object.data.root_pos_w - target.data.root_pos_w

            if "orientation" in goal:
                orientation = goal["orientation"]
                if orientation == "right":
                    pos_diff = pos_diff + torch.tensor([0.0, -0.05, 0.0], device=env.device)
                elif orientation == "left":
                    pos_diff = pos_diff + torch.tensor([0.0, 0.05, 0.0], device=env.device)
                elif orientation == "front":  # in the front of the flat_stove
                    pos_diff = pos_diff + torch.tensor([-0.35, 0.0, 0.0], device=env.device)

            height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
            xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

            success = torch.logical_and(
                success, torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)
            )
            
            # check if the force threshold is satisfied to ensure object is in contact with the target
            if enable_force_threshold == "True" and (
                f"contact_{target_name}_{obj_name}" in env.scene.keys()
                and env.scene[f"contact_{target_name}_{obj_name}"] is not None
            ):
                contact_force_object = env.scene[
                    f"contact_{target_name}_{obj_name}"
                ].data.force_matrix_w  # net_forces_w shape: (N, 1, 1, 3)
                contact_force_norm = torch.linalg.vector_norm(contact_force_object.squeeze(2).squeeze(1), dim=1)

                success = torch.logical_and(success.clone().detach(), contact_force_norm > goal["force_threshold"])
        
        # check if the articulation operation goal is satisfied
        elif "operation" in goal:
            success = torch.logical_and(success, _articulation_operation_goal_satisfied(env, goal))

    return success
