# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_HIGH_PD_WITH_GSMINI_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control and Gelsight tactile sensor
* :obj:`FRANKA_PANDA_LIBERO_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control for LIBERO task

Reference: https://github.com/frankaemika/franka_ros
"""

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from tac_manip.assets import ASSETS_DATA_DIR

##
# Configuration
##

FRANKA_PANDA_LIBERO_HIGH_PD_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.spawn.activate_contact_sensors = True
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.init_state.joint_pos = {
    'panda_joint1': 0.0,
    'panda_joint2': -0.569,
    'panda_joint3': 0.0,
    'panda_joint4': -2.810,
    'panda_joint5': 0.0,
    'panda_joint6': 3.037,
    'panda_joint7': 0.741,
    'panda_finger_joint.*': 0.04,
}
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.init_state.pos = (-0.51, 0.0, 0.42)  # for libero_living_room_tabletop_manipulation task
# higher stiffness and damping for replay success rate
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.actuators['panda_shoulder'].stiffness = 8000.0
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.actuators['panda_shoulder'].damping = 800.0
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.actuators['panda_forearm'].stiffness = 8000.0
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.actuators['panda_forearm'].damping = 800.0
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.actuators['panda_hand'].stiffness = 2000.0    # high 200 
FRANKA_PANDA_LIBERO_HIGH_PD_CFG.actuators['panda_hand'].damping = 100.0       # high 10
"""Configuration of Franka Emika Panda robot with stiffer PD control for LIBERO task."""

FRANKA_PANDA_LIBERO_HIGH_PD_WITH_GSMINI_CFG = FRANKA_PANDA_LIBERO_HIGH_PD_CFG.copy()
# Gripper (panda_hand) PD gains:
# - You can edit these two values to adjust the "gripper force feel" during data collection.
# - Suggested presets:
#   * soft : stiffness ~ 800, damping ~ 30
#   * strong: stiffness > 2000 (e.g., 2500-4000), adjust damping accordingly
FRANKA_PANDA_LIBERO_HIGH_PD_WITH_GSMINI_CFG.actuators['panda_hand'].stiffness = 2000.0  #500
FRANKA_PANDA_LIBERO_HIGH_PD_WITH_GSMINI_CFG.actuators['panda_hand'].damping = 100.0     #25
FRANKA_PANDA_LIBERO_HIGH_PD_WITH_GSMINI_CFG.spawn.usd_path = (
    f'{ASSETS_DATA_DIR}/Robots/Franka_gsmini/physx_rigid_gelpads.usd'
)

"""Configuration of Franka Emika Panda robot with stiffer PD control for factory assembly task."""
FRANKA_PANDA_FACTORY_CFG = FRANKA_PANDA_LIBERO_HIGH_PD_CFG.copy()
FRANKA_PANDA_FACTORY_CFG.actuators['panda_shoulder'].stiffness = 150.0
FRANKA_PANDA_FACTORY_CFG.actuators['panda_shoulder'].damping = 30.0
FRANKA_PANDA_FACTORY_CFG.actuators['panda_forearm'].stiffness = 150.0
FRANKA_PANDA_FACTORY_CFG.actuators['panda_forearm'].damping = 30.0
FRANKA_PANDA_FACTORY_CFG.actuators['panda_hand'].stiffness = 150.0
FRANKA_PANDA_FACTORY_CFG.actuators['panda_hand'].damping = 30.0
FRANKA_PANDA_FACTORY_CFG.init_state.pos = (0.0, 0.0, 0.0)  # for factory assembly task

"""Configuration of Franka Emika Panda robot with with Gelsight tactile sensor for LIBERO task."""

FRANKA_PANDA_HIGH_PD_WITH_GSMINI_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
FRANKA_PANDA_HIGH_PD_WITH_GSMINI_CFG.spawn.usd_path = f'{ASSETS_DATA_DIR}/Robots/Franka_gsmini/physx_rigid_gelpads.usd'

"""Configuration of Franka Emika Panda robot with with Gelsight tactile sensor for articulated task."""
