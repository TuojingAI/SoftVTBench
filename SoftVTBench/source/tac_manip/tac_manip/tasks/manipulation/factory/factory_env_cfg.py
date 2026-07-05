# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from tac_manip.compat.devices import DevicesCfg
from tac_manip.compat.teleop import (
    GripperRetargeterCfg,
    OpenXRDevice,
    OpenXRDeviceCfg,
    Se3AbsRetargeterCfg,
    Se3KeyboardCfg,
    Se3RelRetargeterCfg,
    Se3SpaceMouseCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim import PhysxCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from tac_manip.tasks.manipulation.factory import mdp
from tac_manip.tasks.manipulation.factory.factory_tasks_cfg import *

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"

RIGID_BODY_PROPERTY_A = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    max_depenetration_velocity=5.0,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=3666.0,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=192,
    solver_velocity_iteration_count=1,
    max_contact_impulse=1e32,
)

RIGID_BODY_PROPERTY_B = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    max_depenetration_velocity=5.0,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=3666.0,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=32,
    solver_velocity_iteration_count=32,
    max_contact_impulse=1e32,
)

ASSET_CFGS = {
    "gear_mesh": {
        "fixed_asset": GearBase(),
        "held_asset": MediumGear(),
    },
    "nut_thread": {
        "fixed_asset": BoltM16(),
        "held_asset": NutM16(),
    },
    "peg_insert": {
        "fixed_asset": Hole8mm(),
        "held_asset": Peg8mm(),
    },
}


##
# Scene definition
##
@configclass
class FactorySceneCfg(InteractiveSceneCfg):
    """Configuration for the factory assembly environment."""

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.55, 0.0, 0.0], rot=[0.70711, 0.0, 0.0, 0.70711]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -0.4]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )


class FactoryGearMeshSceneCfg(FactorySceneCfg):
    small_gear = ArticulationCfg(
        prim_path="/World/envs/env_.*/SmallGearAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/factory_gear_small.usd",
            activate_contact_sensors=True,
            rigid_props=RIGID_BODY_PROPERTY_B,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            scale=(2.0, 2.0, 2.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    large_gear = ArticulationCfg(
        prim_path="/World/envs/env_.*/LargeGearAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/factory_gear_large.usd",
            activate_contact_sensors=True,
            rigid_props=RIGID_BODY_PROPERTY_B,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            scale=(2.0, 2.0, 2.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    fixed_asset = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ASSET_CFGS["gear_mesh"]["fixed_asset"].usd_path,
            activate_contact_sensors=True,
            rigid_props=RIGID_BODY_PROPERTY_B,
            mass_props=sim_utils.MassPropertiesCfg(mass=ASSET_CFGS["gear_mesh"]["fixed_asset"].mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            scale=(2.0, 2.0, 2.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    held_asset = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ASSET_CFGS["gear_mesh"]["held_asset"].usd_path,
            activate_contact_sensors=True,
            rigid_props=RIGID_BODY_PROPERTY_B,
            mass_props=sim_utils.MassPropertiesCfg(mass=ASSET_CFGS["gear_mesh"]["held_asset"].mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            scale=(2.0, 2.0, 2.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


class FactoryNutThreadSceneCfg(FactorySceneCfg):
    fixed_asset = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ASSET_CFGS["nut_thread"]["fixed_asset"].usd_path,
            activate_contact_sensors=True,
            rigid_props=RIGID_BODY_PROPERTY_A,
            mass_props=sim_utils.MassPropertiesCfg(mass=ASSET_CFGS["nut_thread"]["fixed_asset"].mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            scale=(2.0, 2.0, 2.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    held_asset = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ASSET_CFGS["nut_thread"]["held_asset"].usd_path,
            activate_contact_sensors=True,
            rigid_props=RIGID_BODY_PROPERTY_B,
            mass_props=sim_utils.MassPropertiesCfg(mass=ASSET_CFGS["nut_thread"]["held_asset"].mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            scale=(2.0, 2.0, 2.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


class FactoryPegInsertSceneCfg(FactorySceneCfg):
    # Add the specific Object for this env
    fixed_asset = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ASSET_CFGS["peg_insert"]["fixed_asset"].usd_path,
            activate_contact_sensors=True,
            rigid_props=RIGID_BODY_PROPERTY_A,
            mass_props=sim_utils.MassPropertiesCfg(mass=ASSET_CFGS["peg_insert"]["fixed_asset"].mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.05, rest_offset=0.0),
            scale=(3.0, 3.0, 3.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    held_asset = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ASSET_CFGS["peg_insert"]["held_asset"].usd_path,
            activate_contact_sensors=True,
            rigid_props=RIGID_BODY_PROPERTY_B,
            mass_props=sim_utils.MassPropertiesCfg(mass=ASSET_CFGS["peg_insert"]["held_asset"].mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.05, rest_offset=0.0),
            scale=(3.0, 3.0, 3.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        fixed_asset_pose = ObsTerm(
            func=mdp.object_poses_in_base_frame, params={"object_cfg": SceneEntityCfg("fixed_asset")}
        )
        held_asset_pose = ObsTerm(
            func=mdp.object_poses_in_base_frame, params={"object_cfg": SceneEntityCfg("held_asset")}
        )
        eef_pose = ObsTerm(func=mdp.ee_frame_pose_in_base_frame)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("held_asset"),
                "diff_xy_threshold": 0.06,
                "height_threshold": 0.06,  # Since the peg is long, the threshold is relative big
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("held_asset")},
    )

    success = DoneTerm(
        func=mdp.object_a_is_into_b,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_a_cfg": SceneEntityCfg("fixed_asset"),
            "object_b_cfg": SceneEntityCfg("held_asset"),
            "xy_threshold": 0.01,
            "height_threshold": 0.02,
        },
    )


@configclass
class FactoryEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene = None
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    rewards = None
    events = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = 2
        self.sim.physx = PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        )
        self.sim.physics_material = RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        )

        if self.task_name == "peg_insert":
            self.observations.subtask_terms.grasp = ObsTerm(
                func=mdp.object_grasped,
                params={
                    "robot_cfg": SceneEntityCfg("robot"),
                    "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                    "object_cfg": SceneEntityCfg("held_asset"),
                    # Since the peg is long and thin, the XY threshold is small, and the height threshold is comparatively large.
                    "diff_xy_threshold": 0.02,
                    "height_threshold": 0.15,
                },
            )

        self.teleop_devices = DevicesCfg(
            devices={
                # Added by weihuaz, not sure which is easy for teleoperation, absolute or relative?
                # "handtracking": OpenXRDeviceCfg(
                #     retargeters=[
                #         Se3AbsRetargeterCfg(
                #             bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                #             zero_out_xy_rotation=True,
                #             use_wrist_rotation=False,
                #             use_wrist_position=True,
                #             sim_device=self.sim.device,
                #         ),
                #         GripperRetargeterCfg(
                #             bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                #         ),
                #     ],
                #     sim_device=self.sim.device,
                #     xr_cfg=self.xr,
                # ),
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )
