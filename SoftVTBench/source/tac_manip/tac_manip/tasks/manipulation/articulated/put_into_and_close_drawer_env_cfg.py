# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (
    FRAME_MARKER_SMALL_CFG,
)
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from tac_manip.assets import ASSETS_DATA_DIR

from . import mdp
from .open_drawer_env_cfg import OpenDrawerCabinetSceneCfg


##
# Scene definition
##
@configclass
class CloseDrawerCabinetSceneCfg(OpenDrawerCabinetSceneCfg):
    """Configuration for the cabinet scene with a robot and a cabinet.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """

    def __post_init__(self):

        # update sektion_cabinet_instanceable.usd with the bottom_drawer's collider approximation as convex_decomposition
        self.cabinet.spawn.usd_path = f"{ASSETS_DATA_DIR}/Articulated/cabinet_collider.usd"
        self.cabinet.spawn.activate_contact_sensors = False

        # open the bottom drawer
        self.cabinet.init_state.joint_pos["drawer_bottom_joint"] = 0.25

        # Frame definitions for the cabinet.
        self.cabinet_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Cabinet/cabinet/sektion",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/CabinetFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Cabinet/cabinet/drawer_handle_bottom",
                    name="drawer_handle_bottom",
                    offset=OffsetCfg(
                        pos=(0.305, 0.0, 0.01),
                        rot=(0.5, 0.5, -0.5, -0.5),  # align with end-effector frame
                    ),
                ),
            ],
        )
        # Add a table
        self.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.7, 0.0), rot=(0.7071, 0.0, 0.0, 0.7071)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ASSETS_DATA_DIR}/Objects/table.usd",
                scale=(0.5, 0.3, 0.7),
            ),
        )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cabinet_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_bottom_joint"])},
        )
        cabinet_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_bottom_joint"])},
        )
        rel_ee_drawer_distance = ObsTerm(func=mdp.rel_ee_drawer_distance)
        # add obs for mimic task
        eef_pose = ObsTerm(func=mdp.ee_frame_pose_in_base_frame)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        drawer_pose = ObsTerm(func=mdp.object_poses_in_base_frame, params={"object_cfg": SceneEntityCfg("cabinet")})

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp = ObsTerm(
            func=mdp.object_grasped_w_force,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("object"),
                "diff_threshold": 0.05,
                "force_threshold": 1.0,  # add contact_sensor to check grasping force
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    cabinet_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cabinet", body_names=["drawer_handle_.*", "door_.*_nob_link"]),
            "static_friction_range": (1.0, 1.25),
            "dynamic_friction_range": (1.25, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Reset the franka arm pose to close to the drawer handle
    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [1.57, -1.309, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # reset object init poses
    randomize_object_poses = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.2), "y": (0.4, 0.5), "z": (0.615, 0.615), "yaw": (-1.57, 1.57)},
            "asset_cfgs": [SceneEntityCfg("object")],
        },
    )
    # reset cabinet init poses
    randomize_cabinet_poses = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.9, 1.0), "y": (0.05, 0.15), "z": (0.4, 0.4), "yaw": (3.14 - 0.1, 3.14 + 0.1)},
            "asset_cfgs": [SceneEntityCfg("cabinet")],
        },
    )
    # reset drawer init poses
    randomize_drawer_bottom_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.05,
            "asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_bottom_joint"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(
        func=mdp.obj_is_into_drawer_and_drawer_is_closed,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "drawer_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_bottom_joint"], body_names=["drawer_bottom"]),
            "object_cfg": SceneEntityCfg("object"),
            "drawer_close_threshold": 0.05,
            "xy_threshold": 0.35,
            "height_threshold": 0.04,
            "height_diff": 0.0,
        },
    )


##
# Environment configuration
##


@configclass
class PutIntoAndCloseDrawerEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cabinet environment."""

    # Scene settings
    scene: CloseDrawerCabinetSceneCfg = CloseDrawerCabinetSceneCfg(num_envs=4096, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 3
        self.episode_length_s = 8.0

        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625

        # Set viewer camera pose
        self.viewer.lookat = (0.2, 0.0, 0.5)
        self.viewer.eye = (0.0, 1.0, 1.0)
