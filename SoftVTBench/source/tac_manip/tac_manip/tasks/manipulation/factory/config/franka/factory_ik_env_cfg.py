# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os

import isaaclab.sim as sim_utils
import torch
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from tac_manip.tasks.manipulation.factory import mdp
from tac_manip.tasks.manipulation.factory.factory_env_cfg import (
    FactoryEnvCfg,
    FactoryGearMeshSceneCfg,
    FactoryNutThreadSceneCfg,
    FactoryPegInsertSceneCfg,
)
from tac_manip.tasks.manipulation.factory.mdp import factory_events

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from tac_manip.assets import FRANKA_PANDA_FACTORY_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for events."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class GearMeshEventCfg(EventCfg):
    """Configuration for events."""

    randomize_gear_positions = EventTerm(
        func=factory_events.randomize_object_serials_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.3, 0.6), "y": (-0.25, 0.25), "z": (0.0, 0.0), "yaw": (-1.0, 1.0)},
            "min_separation": 0.18,
            "asset_cfgs": [SceneEntityCfg("fixed_asset"), SceneEntityCfg("held_asset")],
            "relative_asset_cfgs": [SceneEntityCfg("small_gear"), SceneEntityCfg("large_gear")],
        },
    )


@configclass
class NutThreadEventCfg(EventCfg):
    """Configuration for events."""

    randomize_fixed_asset_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.3, 0.6), "y": (-0.15, 0.15), "z": (0.0, 0.0), "yaw": (-1.0, 1.0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("fixed_asset")],
        },
    )

    randomize_held_asset_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.3, 0.6), "y": (-0.15, 0.15), "z": (0.0, 0.0), "yaw": (-1.0, 1.0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("held_asset")],
        },
    )


@configclass
class PegInsertEventCfg(EventCfg):
    """Configuration for events."""

    randomize_fixed_asset_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.3, 0.6), "y": (-0.15, 0.15), "z": (0.0, 0.0), "yaw": (-1.0, 1.0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("fixed_asset")],
        },
    )

    randomize_held_asset_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.3, 0.6), "y": (-0.15, 0.15), "z": (0.0, 0.0), "yaw": (-1.0, 1.0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("held_asset")],
        },
    )


@configclass
class FrankaFactoryEnvCfg(FactoryEnvCfg):
    def __post_init__(
        self,
    ):
        # post init of parent
        super().__post_init__()

        # Set events
        if self.task_name == "peg_insert":
            self.scene = FactoryPegInsertSceneCfg(num_envs=128, env_spacing=2.0, replicate_physics=False)
            self.events = PegInsertEventCfg()
        elif self.task_name == "gear_mesh":
            self.scene = FactoryGearMeshSceneCfg(num_envs=128, env_spacing=2.0, replicate_physics=False)
            self.events = GearMeshEventCfg()
        elif self.task_name == "nut_thread":
            self.scene = FactoryNutThreadSceneCfg(num_envs=128, env_spacing=2.0, replicate_physics=False)
            self.events = NutThreadEventCfg()
        else:
            raise RuntimeError(f'We only support factory task in ["peg_insert", "gear_mesh", "nut_thread"]')

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_FACTORY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Read use_relative_mode value from environment variable
        use_relative_mode_env = os.getenv("USE_RELATIVE_MODE", "false")
        self.use_relative_mode = use_relative_mode_env.lower() in ["true", "1", "t"]

        # Set actions for the specific robot type (franka)
        # For teleoperation...
        # if use_relative_mode=True:
        # scale can be < 1.0, and input action is (N, 6) delta_eef_pos, delta_eef_rpy
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=self.use_relative_mode, ik_method="dls"
            ),
            scale=0.5 if self.use_relative_mode else 1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        self.gripper_joint_names = ["panda_finger.*"]
        self.gripper_open_val = torch.tensor([0.04])
        self.gripper_threshold = 0.01

        # Listens to the required transforms
        self.marker_cfg = FRAME_MARKER_CFG.copy()
        self.marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        self.marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,
            visualizer_cfg=self.marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )


@configclass
class FrankaFactoryPegInsertEnvCfg(FrankaFactoryEnvCfg):
    task_name = "peg_insert"


@configclass
class FrankaFactoryGearMeshEnvCfg(FrankaFactoryEnvCfg):
    task_name = "gear_mesh"


@configclass
class FrankaFactoryNutThreadEnvCfg(FrankaFactoryEnvCfg):
    task_name = "nut_thread"
