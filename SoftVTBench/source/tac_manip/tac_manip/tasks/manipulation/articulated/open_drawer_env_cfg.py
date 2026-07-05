# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (
    CabinetSceneCfg,
)
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from . import mdp


##
# Scene definition
##
@configclass
class OpenDrawerCabinetSceneCfg(CabinetSceneCfg):
    """Configuration for the cabinet scene with a robot and a cabinet.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """

    def __post_init__(self):

        # set stiffness, damping, and friction to 0 for the drawers and doors for passive control
        self.cabinet.actuators["drawers"].stiffness = 0.0
        self.cabinet.actuators["drawers"].damping = 0.0
        self.cabinet.actuators["drawers"].friction = 0.005
        self.cabinet.actuators["doors"].stiffness = 0.0
        self.cabinet.actuators["doors"].damping = 0.0
        self.cabinet.actuators["doors"].friction = 0.01


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
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )
        cabinet_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
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

        drawer_is_open = ObsTerm(
            func=mdp.drawer_is_open,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "drawer_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
                "drawer_open_threshold": 0.125,
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
            "default_pose": [0.0, -1.309, 0.0, -2.793, 0.0, 3.037, -0.830, 0.04, 0.04],
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
    # reset cabinet init poses
    randomize_cabinet_poses = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.8, 0.9), "y": (-0.05, 0.05), "z": (0.6, 0.6), "yaw": (3.14 - 0.1, 3.14 + 0.1)},
            "asset_cfgs": [SceneEntityCfg("cabinet")],
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.drawer_is_open,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "drawer_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
            "drawer_open_threshold": 0.15,
        },
    )


##
# Environment configuration
##


@configclass
class OpenDrawerEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cabinet environment."""

    # Scene settings
    scene: OpenDrawerCabinetSceneCfg = OpenDrawerCabinetSceneCfg(num_envs=4096, env_spacing=2.0)
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

        self.scene.cabinet_frame.debug_vis = False
