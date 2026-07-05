# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
import random
from dataclasses import dataclass, field

import cv2
import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "10.19.222.31"  # "10.19.222.31"    # "0.0.0.0"
    port: int = 8000
    target_image_size: tuple[int, int, int] = (224, 224, 3)
    replan_steps: int = 10
    num_steps_wait: int = 20  # Number of steps to wait for objects to stabilize i n sim
    task_horizon: int = 100  # Each inference, will execute 5 action plans, max horizon: 120*5
    num_success_steps: int = 8  # continuous success steps to consider the policy as successful
    num_total_experiments: int = 100  # total number of experiments to do policy evaluation
    env_name: str = "Isaac-Stack-Cube-Galbot-Left-Arm-Joint-Position-Image-Based-v0"
    record_camera_output_path: str = None
    record_images: bool = False
    record_videos: bool = False
    num_envs: int = 1
    seed: int = 10
    device: str = "cuda"
    overwite_state_pos: bool = True
    debug_mode: bool = False
    task_description: str = (
        "Pick up the red cube and put it onto the blue cube.Then pick up the green cube and place it onto the red cube."
    )

    # For stack cube tasks with Galbot left arm
    # "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4",
    # "left_arm_joint5", "left_arm_joint6", "left_arm_joint7", "left_gripper_left_joint"
    training_joints_index: list = field(default_factory=lambda: [5, 8, 10, 12, 14, 16, 18, 20])
    gripper_state_index: int = 20
    # True, will use RmpFlow for eef control; False, will use joint space.
    eef_mode: bool = False


def resize_frames_with_padding(
    frames: torch.Tensor, target_image_size: tuple, bgr_conversion: bool = False, pad_img: bool = True
):
    """Process batch of frames with padding and resizing vectorized
    Args:
        frames: np.ndarray of shape [N, 256, 160, 3]
        target_image_size: target size
        bgr_conversion: whether to convert BGR to RGB
        pad_img: whether to resize images
    """
    # convert to cpu
    frames = frames.cpu().numpy()
    if bgr_conversion:
        frames = np.array([cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames])

    if pad_img:
        top_padding = (frames.shape[2] - frames.shape[1]) // 2
        bottom_padding = top_padding

        # Add padding to all frames at once
        frames = np.pad(
            frames,
            pad_width=((0, 0), (top_padding, bottom_padding), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    # Resize all frames at once
    if frames.shape[1:] != target_image_size:
        frames = np.stack([cv2.resize(f, target_image_size[:2]) for f in frames])

    return frames


def _create_sim_environment(args):
    """
    Creates a simulation environment based on the given arguments.
    Args:
        args (Args): The arguments for the simulation environment.
    Returns:
        gym.Env: The created simulation environment.
    """
    if args.eef_mode:
        env_name = "Isaac-Stack-Cube-Galbot-Left-Arm-RmpFlow-Image-Based-v0"
    else:
        env_name = "Isaac-Stack-Cube-Galbot-Left-Arm-Joint-Position-Image-Based-v0"
    env_cfg = parse_env_cfg(env_name, device=args.device, num_envs=args.num_envs)

    if args.eef_mode:
        # set init idle action, task space
        env_cfg.idle_action = torch.tensor([0.5287, 0.3977, 1.0008, -0.0450, 0.7060, -0.0463, -0.7052, 0.035])
    else:
        # set init idle action, joint space
        env_cfg.idle_action = torch.tensor([-0.5480, -0.6551, 3.14 - 0.7330, 1.3641, -0.4416, 0.1168, 1.2308, 0.035])

    env_cfg.env_name = env_name
    if args.record_images or args.record_videos:
        env_cfg.scene.record_cam = None
        if args.record_camera_output_path is not None:
            # Ensure directory exists
            os.makedirs(args.record_camera_output_path, exist_ok=True)

    # Disable all recorders and terminations
    env_cfg.recorders = {}
    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        print(
            "No success termination term was found in the environment."
            " Will not be able to mark policy evaluation result as successful."
        )
    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None

    # create environment from loaded config
    env = gym.make(args.env_name, cfg=env_cfg)
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env.unwrapped.seed(args.seed)
    return env, env_cfg, success_term


def axisangle2quat(axisangle):
    """
    Converts axis-angle format to quaternion.

    Args:
        axisangle (np.array): (ax, ay, az) axis-angle exponential coordinates

    Returns:
        np.array: (w, x, y, z) quaternion
    """
    angle = np.linalg.norm(axisangle)
    if math.isclose(angle, 0.0):
        # Zero rotation quaternion
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axisangle / angle
    half_angle = angle / 2.0
    w = math.cos(half_angle)
    xyz = axis * math.sin(half_angle)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def quat2axisangle(quat):
    """
    Copied and modified from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    Args:
        quat (np.array): (w,x,y,z) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[0] > 1.0:
        quat[0] = 1.0
    elif quat[0] < -1.0:
        quat[0] = -1.0

    den = np.sqrt(1.0 - quat[0] * quat[0])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[1:] * 2.0 * math.acos(quat[0])) / den


def _quat2axisangle(quats):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    Robosuite: XYZW
    Isaac Lab: WXYZ, https://docs.isaacsim.omniverse.nvidia.com/4.5.0/reference_material/reference_conventions.html#quaternions
    Modified the code slightly based on the sequence of quat.
    """
    # clip quaternion
    results = []
    for quat in quats:
        q0 = quat[0]
        # clip q0
        if q0 > 1.0:
            q0 = 1.0
        elif q0 < -1.0:
            q0 = -1.0

        den = np.sqrt(1.0 - q0 * q0)
        if math.isclose(den, 0.0):
            results.append(np.zeros(3))
        else:
            axis_angle = (quat[1:] * 2.0 * math.acos(q0)) / den
            results.append(axis_angle)
    return np.array(results)
