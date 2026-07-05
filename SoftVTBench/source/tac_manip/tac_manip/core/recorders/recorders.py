# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorder implementations for recording robot actions and states."""

from __future__ import annotations

import torch
from isaaclab.managers.recorder_manager import RecorderTerm


class PostStepAbsEEFPoseBinaryGripperActionsRecorder(RecorderTerm):
    """Recorder term that records the absolute eef_pose actions of the environment at the end of each step."""

    def record_post_step(self):
        # FIXME: record current eef_pose as action_target may bring a little bit of delay
        # this is the eef_pose in world frame
        eef_pose = self._env.obs_buf["policy"]["eef_pose"]

        if "Suction" in self._env.unwrapped.cfg.env_name:
            gripper_bool_action = torch.tensor(
                [not cup.is_closed() for cup in self._env._suction_cup], dtype=torch.bool
            ).view(-1, 1)
            gripper_value_action = torch.where(gripper_bool_action, torch.tensor(1.0), torch.tensor(-1.0)).to(
                self._env.device
            )
            final_action = torch.cat([eef_pose, gripper_value_action], dim=-1)
            return "actions", final_action
        else:
            gripper_value_action = self._env.action_manager.action[:, -1:]
            final_action = torch.cat([eef_pose, gripper_value_action], dim=-1)
            return "actions", final_action


class PostStepAbsEEFPoseAbsGripperActionsRecorder(RecorderTerm):
    """Recorder term that records the absolute eef_pose actions of the environment at the end of each step."""

    def record_post_step(self):
        # FIXME: record current eef_pose as action_target may bring a little bit of delay
        # this is the eef_pose in world frame
        eef_pose = self._env.obs_buf["policy"]["eef_pose"]
        gripper_pos = self._env.obs_buf["policy"]["gripper_pos"]

        if "Suction" in self._env.unwrapped.cfg.env_name:
            gripper_bool_action = torch.tensor(
                [not cup.is_closed() for cup in self._env._suction_cup], dtype=torch.bool
            ).view(-1, 1)
            gripper_value_action = torch.where(gripper_bool_action, torch.tensor(1.0), torch.tensor(-1.0)).to(
                self._env.device
            )
            final_action = torch.cat([eef_pose, gripper_value_action], dim=-1)
            return "actions", final_action
        else:
            absolute_gripper_pos = gripper_pos[:, 0].view(-1, 1)
            final_action = torch.cat([eef_pose, absolute_gripper_pos], dim=-1)
            return "actions", final_action


def quat2axisangle_torch(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to axis-angle representation (ax, ay, az).

    Args:
        quat: Quaternion tensor of shape [N, 4] with order (w, x, y, z)

    Returns:
        Axis-angle tensor of shape [N, 3]
    """
    # Clip quaternion w component to valid range
    q0 = torch.clamp(quat[:, 0:1], -1.0, 1.0)

    # Compute denominator
    den = torch.sqrt(torch.clamp(1.0 - q0 * q0, min=1e-8))

    # Handle near-zero rotation case
    mask = den < 1e-6
    axis_angle = torch.zeros_like(quat[:, 1:4])

    # Compute axis-angle for non-zero rotations
    angle = 2.0 * torch.acos(q0)
    axis_angle[~mask.squeeze(-1)] = (quat[~mask.squeeze(-1), 1:4] * angle[~mask]) / den[~mask]

    return axis_angle


class PostStepAbsEEFPoseAxisAngleBinaryGripperActionsRecorder(PostStepAbsEEFPoseBinaryGripperActionsRecorder):
    """Recorder term that records absolute eef_pose with axis-angle rotation representation.

    Inherits from PostStepAbsEEFPoseBinaryGripperActionsRecorder and converts quaternion to axis-angle.
    Actions: absolute_eef_pose (6D: x, y, z, ax, ay, az) + binary_gripper_action (1D)
    """

    def record_post_step(self):
        """Record eef pose actions with axis-angle rotation representation."""
        # Get eef_pose from parent class logic
        eef_pose = self._env.obs_buf['policy']['eef_pose']

        # Extract position and quaternion
        pos = eef_pose[:, :3]  # [N, 3]
        quat = eef_pose[:, 3:7]  # [N, 4] - (w, x, y, z)

        # Convert quaternion to axis-angle
        axis_angle = quat2axisangle_torch(quat)  # [N, 3]

        # Combine position and axis-angle: [N, 6]
        eef_pose_axis_angle = torch.cat([pos, axis_angle], dim=-1)

        # Use parent class logic for gripper handling
        if 'Suction' in self._env.unwrapped.cfg.env_name:
            gripper_bool_action = torch.tensor(
                [not cup.is_closed() for cup in self._env._suction_cup], dtype=torch.bool
            ).view(-1, 1)
            gripper_value_action = torch.where(gripper_bool_action, torch.tensor(1.0), torch.tensor(-1.0)).to(
                self._env.device
            )
            final_action = torch.cat([eef_pose_axis_angle, gripper_value_action], dim=-1)
            return 'actions', final_action
        else:
            gripper_value_action = self._env.action_manager.action[:, -1:]
            final_action = torch.cat([eef_pose_axis_angle, gripper_value_action], dim=-1)
            return 'actions', final_action


class PostStepAbsEEFPoseAxisAngleAbsGripperActionsRecorder(PostStepAbsEEFPoseAbsGripperActionsRecorder):
    """Recorder term that records absolute eef_pose with axis-angle rotation representation.

    Inherits from PostStepAbsEEFPoseAbsGripperActionsRecorder and converts quaternion to axis-angle.
    Actions: absolute_eef_pose (6D: x, y, z, ax, ay, az) + absolute_gripper_action (1D)
    """

    def record_post_step(self):
        """Record eef pose actions with axis-angle rotation representation."""
        # Get eef_pose and gripper_pos from parent class logic
        eef_pose = self._env.obs_buf['policy']['eef_pose']
        gripper_pos = self._env.obs_buf['policy']['gripper_pos']

        # Extract position and quaternion
        pos = eef_pose[:, :3]  # [N, 3]
        quat = eef_pose[:, 3:7]  # [N, 4] - (w, x, y, z)

        # Convert quaternion to axis-angle
        axis_angle = quat2axisangle_torch(quat)  # [N, 3]

        # Combine position and axis-angle: [N, 6]
        eef_pose_axis_angle = torch.cat([pos, axis_angle], dim=-1)

        # Use parent class logic for gripper handling
        if 'Suction' in self._env.unwrapped.cfg.env_name:
            gripper_bool_action = torch.tensor(
                [not cup.is_closed() for cup in self._env._suction_cup], dtype=torch.bool
            ).view(-1, 1)
            gripper_value_action = torch.where(gripper_bool_action, torch.tensor(1.0), torch.tensor(-1.0)).to(
                self._env.device
            )
            final_action = torch.cat([eef_pose_axis_angle, gripper_value_action], dim=-1)
            return 'actions', final_action
        else:
            absolute_gripper_pos = gripper_pos[:, 0].view(-1, 1)
            final_action = torch.cat([eef_pose_axis_angle, absolute_gripper_pos], dim=-1)
            return 'actions', final_action


class PostStepAbsEEFPoseAxisAngleAbsGripperWithForceActionsRecorder(
    PostStepAbsEEFPoseAxisAngleAbsGripperActionsRecorder
):
    """Recorder term that records absolute eef_pose (axis-angle) + abs gripper + forces.

    Inherits from PostStepAbsEEFPoseAxisAngleAbsGripperActionsRecorder.
    Actions: eef_pose(6D) + abs_gripper(1D) + left_force(3D) + right_force(3D) = 13D
    Requires 'gripper_net_force' in policy observations with shape (N, history_length, 2, 3).
    Extracts the most recent force (index 0) for both fingers.
    """

    def record_post_step(self):
        """Record eef pose actions with axis-angle rotation and forces."""
        # Get base action (7D) from parent: pos(3) + axis_angle(3) + gripper(1)
        key, base_action = super().record_post_step()

        # Get forces from policy observations
        obs = self._env.obs_buf['policy']

        # Extract forces from gripper_net_force observation
        # Shape: (N, history_length, 2, 3) where history_length=4, 2=fingers, 3=force components
        if 'gripper_net_force' in obs:
            gripper_net_force = obs['gripper_net_force']  # (N, 4, 2, 3)
            
            # Take the most recent time step (index 0) for both fingers
            # gripper_net_force[:, 0, :, :] -> (N, 2, 3)
            current_forces = gripper_net_force[:, 0, :, :]  # Most recent timestep
            
            # Extract left and right finger forces
            left_force = current_forces[:, 0, :]  # (N, 3) - left finger
            right_force = current_forces[:, 1, :]  # (N, 3) - right finger
        else:
            # Fallback: zero forces if not available
            left_force = torch.zeros((base_action.shape[0], 3), device=base_action.device, dtype=base_action.dtype)
            right_force = torch.zeros((base_action.shape[0], 3), device=base_action.device, dtype=base_action.dtype)

        final_action = torch.cat([base_action, left_force, right_force], dim=-1)
        return key, final_action


#
# NOTE:
# The binary-gripper + force recorder has been removed/deprecated.
# If you need forces recorded into actions, use the 7dpf recorder
# (axis-angle + abs gripper + Force(6)).
#


class PostStepAbsEEFPoseAxisAngleGripperCommandActionsRecorder(PostStepAbsEEFPoseAxisAngleAbsGripperActionsRecorder):
    """Companion term for dual-gripper recording (7dp2 mode).

    Same absolute eef pose in axis-angle form, but the last dim is the raw
    gripper COMMAND from the action manager (the 7d2 convention). Exported
    under the separate key 'actions_7d2' so the main 'actions' dataset stays
    7dp and existing consumers are unaffected.
    """

    def record_post_step(self):
        eef_pose = self._env.obs_buf['policy']['eef_pose']
        pos = eef_pose[:, :3]
        quat = eef_pose[:, 3:7]
        axis_angle = quat2axisangle_torch(quat)
        eef_pose_axis_angle = torch.cat([pos, axis_angle], dim=-1)

        if 'Suction' in self._env.unwrapped.cfg.env_name:
            gripper_bool_action = torch.tensor(
                [not cup.is_closed() for cup in self._env._suction_cup], dtype=torch.bool
            ).view(-1, 1)
            gripper_value_action = torch.where(gripper_bool_action, torch.tensor(1.0), torch.tensor(-1.0)).to(
                self._env.device
            )
        else:
            gripper_value_action = self._env.action_manager.action[:, -1:]
        final_action = torch.cat([eef_pose_axis_angle, gripper_value_action], dim=-1)
        return 'actions_7d2', final_action
