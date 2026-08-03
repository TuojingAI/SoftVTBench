# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def contact_force_in_gripper_frame(
    env: ManagerBasedRLEnv,
    contact_sensor_name: str = "contact_gripper",
    history_length: int = 1,
    left_gripper_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_gripper_frame"),
    right_gripper_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_gripper_frame"),
) -> torch.Tensor:
    """Convert contact forces from world coordinate system to gripper local coordinate system."""

    left_gripper_frame: FrameTransformer = env.scene[left_gripper_frame_cfg.name]
    right_gripper_frame: FrameTransformer = env.scene[right_gripper_frame_cfg.name]

    if contact_sensor_name in env.scene.keys() and env.scene[contact_sensor_name] is not None:
        # contact_force = env.scene[contact_sensor_name].data.net_forces_w  # shape:(N, 2, 3) for two fingers
        contact_force_history = env.scene[
            contact_sensor_name
        ].data.net_forces_w_history  # shape: (N, H_sensor, 2, 3) for two fingers
        # Clamp to at most `history_length`, but also handle the case where the sensor
        # has a smaller history buffer than requested.
        contact_force_history = contact_force_history[:, :history_length, :, :]
        # Use the actual available history length from the tensor to avoid shape mismatch.
        H = contact_force_history.shape[1]
        left_contact_w = contact_force_history[:, :, 0, :]  # shape: (N, H, 3)
        right_contact_w = contact_force_history[:, :, 1, :]  # shape: (N, H, 3)

        # Get gripper quaternion orientation in world coordinate system
        # target_quat_w shape is (N, M, 4), where M is the number of target frames
        # Assume we only have one target frame (gripper), so take index 0
        left_gripper_quat_w = left_gripper_frame.data.target_quat_w[:, 0, :]  # shape: (N, 4)
        right_gripper_quat_w = right_gripper_frame.data.target_quat_w[:, 0, :]  # shape: (N, 4)

        # Method 1: Use quaternion inverse transformation (recommended method)
        # Calculate rotation from world coordinate system to gripper local coordinate system
        left_gripper_quat_inv = math_utils.quat_inv(left_gripper_quat_w)  # shape: (N, 4)
        right_gripper_quat_inv = math_utils.quat_inv(right_gripper_quat_w)  # shape: (N, 4)

        # Convert force vectors from world coordinate system to gripper local coordinate system
        # Need to expand quaternions to the same dimensions as force history
        left_quat_expanded = left_gripper_quat_inv[:, None, :].expand(
            -1, H, -1
        )  # shape: (N, H, 4)
        right_quat_expanded = right_gripper_quat_inv[:, None, :].expand(
            -1, H, -1
        )  # shape: (N, H, 4)

        # Apply rotation transformation
        left_contact_local = math_utils.quat_apply(
            left_quat_expanded.reshape(-1, 4),  # reshape to (N*H, 4)
            left_contact_w.reshape(-1, 3),  # reshape to (N*H, 3)
        ).reshape(
            -1, H, 3
        )  # reshape back to (N, H, 3)

        right_contact_local = math_utils.quat_apply(
            right_quat_expanded.reshape(-1, 4),  # reshape to (N*H, 4)
            right_contact_w.reshape(-1, 3),  # reshape to (N*H, 3)
        ).reshape(
            -1, H, 3
        )  # reshape back to (N, H, 3)

        # Combine left and right gripper local coordinate forces
        contact_force_local = torch.stack(
            [left_contact_local, right_contact_local], dim=2
        )  # shape: (N, H, 2, 3)

        return contact_force_local

    else:
        # If there is no contact sensor, return zero values
        return torch.zeros((env.num_envs, history_length, 2, 3), device=env.device)


def _pose_in_robot_base_frame(
    env: ManagerBasedRLEnv,
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
) -> torch.Tensor:
    """Convert a world-frame pose to the robot root/base frame used by IsaacLab IK."""

    robot = env.scene["robot"]
    pos_b, quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, pos_w, quat_w
    )
    return torch.cat((pos_b, quat_b), dim=1)


def object_poses_in_base_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Object root pose as xyz + quaternion in the robot root/base frame."""

    obj = env.scene[object_cfg.name]
    return _pose_in_robot_base_frame(env, obj.data.root_pos_w, obj.data.root_quat_w)


def ee_frame_pose_in_base_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector target frame pose in the robot root/base frame."""

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    pos_w = ee_frame.data.target_pos_w[:, 0, :]
    quat_w = ee_frame.data.target_quat_w[:, 0, :]
    return _pose_in_robot_base_frame(env, pos_w, quat_w)


def franka_arm_joint_pos(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return 7D Franka arm joint positions for each env.

    NOTE:
        We assume the first 7 DoFs of ``robot.data.joint_pos`` correspond to the arm joints
        (panda_joint1 ~ panda_joint7). This matches the FRANKA_PANDA_LIBERO_HIGH_PD_CFG setup
        used in Libero env configs.
    """
    robot = env.scene["robot"]
    # joint_pos: (N, dof), where the first 7 entries are arm joints
    return robot.data.joint_pos[:, :7]


def gripper_marker_motion_data(
    env: ManagerBasedRLEnv,
    sensor_name_list: list[str] = ["gsmini_left", "gsmini_right"],
) -> torch.Tensor:
    """Get the marker motion field of the gripper.
    Output shape: (N, S, T=2, M, 2)
      - N: number of environments
      - S=2: number of sensors (left, right fingers)
      - T=2: (init, current) two frames
      - M: number of markers
      - 2: (x, y)
    """
    device = env.device
    num_envs = env.num_envs

    tensors = []
    for name in sensor_name_list:
        if name in env.scene.sensors and env.scene[name] is not None:
            sensor = env.scene[name]
            data = sensor.data.output.get("marker_motion")
            if data is not None:
                # Expected shape (N, 2, M, 2)
                tensors.append(data)
            else:
                # Fallback: try to infer M from cfg
                mm_cfg = getattr(sensor.cfg, "marker_motion_sim_cfg", None)
                marker_params = getattr(mm_cfg, "marker_params", None) if mm_cfg is not None else None
                m = getattr(marker_params, "num_markers", 0) if marker_params is not None else 0
                tensors.append(torch.zeros((num_envs, 2, m, 2), device=device))
        else:
            raise ValueError(f"Sensor {name} not found in scene")

    m0 = tensors[0].shape[2]
    for i, t in enumerate(tensors[1:], start=1):
        assert t.shape[2] == m0, (
            "marker count mismatch between sensors; "
            f"sensor {sensor_name_list[0]} has {m0}, sensor {sensor_name_list[i]} has {t.shape[2]}"
        )

    # Stack on sensor dimension S
    out = torch.stack(tensors, dim=1)  # (N, S=2, T=2, M, 2)
    return out
