# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

from . import factory_ik_env_cfg, factory_ik_tactile_env_cfg

##
# Register Gym environments.
##

##
# Inverse Kinematics
##
gym.register(
    id="Isaac-Factory-NutThread-IK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": factory_ik_env_cfg.FrankaFactoryNutThreadEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Factory-PegInsert-IK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": factory_ik_env_cfg.FrankaFactoryPegInsertEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Factory-GearMesh-IK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": factory_ik_env_cfg.FrankaFactoryGearMeshEnvCfg,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Tactile Sensor
##
gym.register(
    id="Isaac-Factory-NutThread-IK-Tactile-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": factory_ik_tactile_env_cfg.FrankaFactoryNutThreadTactileEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Factory-PegInsert-IK-Tactile-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": factory_ik_tactile_env_cfg.FrankaFactoryPegInsertTactileEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Factory-GearMesh-IK-Tactile-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": factory_ik_tactile_env_cfg.FrankaFactoryGearMeshTactileEnvCfg,
    },
    disable_env_checker=True,
)
