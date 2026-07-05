# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from tac_manip.compat.devices import DevicesCfg
from tac_manip.compat.teleop import Se3KeyboardCfg, Se3SpaceMouseCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (
    FRAME_MARKER_SMALL_CFG,
)
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from tac_manip.assets import ASSETS_DATA_DIR
from tac_manip.tasks.manipulation.articulated import mdp

from tac_manip.tasks.manipulation.articulated.open_drawer_env_cfg import (  # isort: skip
    OpenDrawerEnvCfg,
)
from tac_manip.tasks.manipulation.articulated.put_into_and_close_drawer_env_cfg import (  # isort: skip
    PutIntoAndCloseDrawerEnvCfg,
)


##
# Pre-defined configs
##
from tac_manip.assets import UR10e_UMIXENSE_CFG  # isort: skip


@configclass
class UR10EOpenDrawerEnvCfg(OpenDrawerEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set UR10e as robot
        self.scene.robot = UR10e_UMIXENSE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (-0.4, -0.25, 0.0)

        # Reset the ur10e arm pose to close to the drawer handle
        self.events.init_franka_arm_pose = EventTerm(
            func=franka_stack_events.set_default_joint_pose,
            mode="reset",
            params={
                "default_pose": [0.0, -1.5708, 1.5708, -3.14159, -1.5708, -1.5708, -0.03, -0.03],
                # "default_pose": [0.0, -1.5708, 1.5708, -1.5708, -1.5708, -1.5708, -0.03, -0.03],
            },
        )
        # disable physics material events
        self.events.robot_physics_material = None
        self.events.cabinet_physics_material = None

        use_relative_mode_env = os.getenv("USE_RELATIVE_MODE", "False")
        self.use_relative_mode = use_relative_mode_env.lower() in ["true", "1", "t"]

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*_joint"],
            body_name="ee_base_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=self.use_relative_mode, ik_method="dls"
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.179]),
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint[1,2]"],
            open_command_expr={"joint[1,2]": -0.03},
            close_command_expr={"joint[1,2]": 0.025},
        )

        self.gripper_joint_names = ["joint[1,2]"]
        self.gripper_open_val = -0.03
        self.gripper_threshold = 0.01

        # Listens to the required transforms
        # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
        # the other frames are the fingers
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/ee_base_link",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.179),
                    ),
                ),
            ],
        )
        # teleop devices
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )


@configclass
class UR10EPutIntoAndCloseDrawerEnvCfg(PutIntoAndCloseDrawerEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Set UR10e as robot
        self.scene.robot = UR10e_UMIXENSE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (-0.5, -0.2, 0.0)

        # Reset the ur10e arm pose
        self.events.init_franka_arm_pose = EventTerm(
            func=franka_stack_events.set_default_joint_pose,
            mode="reset",
            params={
                "default_pose": [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 1.5708, -0.03, -0.03],
            },
        )
        # disable physics material events
        self.events.robot_physics_material = None
        self.events.cabinet_physics_material = None

        # add object to be grasped and put into the drawer
        # Rigid body properties of object_a and object_b
        object_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.35, 0.71)),
            spawn=UsdFileCfg(
                usd_path=f"{ASSETS_DATA_DIR}/Objects/Apple.usd",
                scale=(0.5, 0.5, 0.5),
                rigid_props=object_properties,
                mass_props=MassPropertiesCfg(mass=0.05),
            ),
        )

        use_relative_mode_env = os.getenv("USE_RELATIVE_MODE", "False")
        self.use_relative_mode = use_relative_mode_env.lower() in ["true", "1", "t"]

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*_joint"],
            body_name="ee_base_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=self.use_relative_mode, ik_method="dls"
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.179]),
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint[1,2]"],
            open_command_expr={"joint[1,2]": -0.03},
            close_command_expr={"joint[1,2]": 0.025},
        )

        self.gripper_joint_names = ["joint[1,2]"]
        self.gripper_open_val = -0.03
        self.gripper_threshold = 0.01

        # Listens to the required transforms
        # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
        # the other frames are the fingers
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/ee_base_link",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.179),
                    ),
                ),
            ],
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )
