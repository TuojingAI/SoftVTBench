# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Open-Drawer-Franka-IK-IL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:FrankaOpenDrawerEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Put-Into-And-Close-Drawer-Franka-IK-IL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:FrankaPutIntoAndCloseDrawerEnvCfg",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control with Tactile Sensor
##

gym.register(
    id="Isaac-Open-Drawer-Franka-IK-Tactile-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tactile_ik_rel_env_cfg:FrankaOpenDrawerTactileEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Put-Into-And-Close-Drawer-Franka-IK-Tactile-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tactile_ik_rel_env_cfg:FrankaPutIntoAndCloseDrawerTactileEnvCfg",
    },
    disable_env_checker=True,
)
