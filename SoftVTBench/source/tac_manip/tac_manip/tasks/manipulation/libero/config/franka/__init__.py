# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

##
# Register Gym environments.
##


##
# Replay demo from Libero Dataset
# use JointPositionController, with camera
# NOTE: This is the preferred/maintained replay env. The non-camera replay env
# has been removed.
##
gym.register(
    id="Isaac-Libero-Franka-Replay-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_libero_env_cfg:JointPositionLiberoCameraEnvCfg",
    },
    disable_env_checker=True,
)


##
# Replay demo from Libero Dataset
# use DiffIKController
##
gym.register(
    id="Isaac-Libero-Franka-IK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_libero_env_cfg:IKLiberoCameraEnvCfg",
    },
    disable_env_checker=True,
)

##
# Replay demo from Libero Dataset
# use OperationalSpaceController
##
gym.register(
    id="Isaac-Libero-Franka-OscPose-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_libero_env_cfg:OscPoseLiberoCameraEnvCfg",
    },
    disable_env_checker=True,
)

##
# Replay demo from Libero Dataset
# use JointPositionController, with camera, with tactile sensor
##
gym.register(
    id="Isaac-Libero-Franka-Replay-Camera-ContactForce-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_tactile_libero_env_cfg:JointPositionContactForceLiberoCameraEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Libero-Franka-Replay-Camera-Tactile-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_tactile_libero_env_cfg:JointPositionTactileLiberoCameraEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Libero-Franka-Replay-Camera-Tactile-DualForce-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_tactile_libero_env_cfg:JointPositionTactileDualForceLiberoCameraEnvCfg",
    },
    disable_env_checker=True,
)

##
# Record demo with Libero tasks
# use Relative IK Controller, with camera, with tactile sensor
##
gym.register(
    id="Isaac-Libero-Franka-IK-Camera-Tactile-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_tactile_libero_env_cfg:IKTactileLiberoCameraEnvCfg",
    },
    disable_env_checker=True,
)

##
# Record / replay demo with Libero tasks
# use Relative IK Controller (pure position), with camera, with contact force sensor
##
gym.register(
    id="Isaac-Libero-Franka-IK-Camera-ContactForce-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_tactile_libero_env_cfg:IKContactForceLiberoCameraEnvCfg",
    },
    disable_env_checker=True,
)

##
# Hybrid force–position OSC controller with finger-level forces.
##
gym.register(
    id="Isaac-Libero-Franka-Hybrid-ContactForce-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_tactile_libero_env_cfg:ForcePositionLiberoCameraEnvCfg",
    },
    disable_env_checker=True,
)

##
# Hybrid force–position OSC controller + GelSight tactile sensors.
##
gym.register(
    id="Isaac-Libero-Franka-Hybrid-Tactile-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_tactile_libero_env_cfg:ForcePositionTactileLiberoCameraEnvCfg",
    },
    disable_env_checker=True,
)

##
# Hybrid force–position OSC controller with finger-level forces + binary gripper.
##
#
# NOTE:
# The Hybrid binary-gripper force-position environment has been removed/deprecated.
# If you need a 13D hybrid action space, use:
# - Isaac-Libero-Franka-Hybrid-ContactForce-v0 (standard fingers + contact forces)
# - Isaac-Libero-Franka-Hybrid-Tactile-v0 (GelSight + gelpad forces)
#
