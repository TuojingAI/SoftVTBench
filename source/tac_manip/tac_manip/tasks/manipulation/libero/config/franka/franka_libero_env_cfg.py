# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import json
import os
import random
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from tac_manip.compat.devices import DevicesCfg
from tac_manip.compat.teleop import (
    GripperRetargeterCfg,
    OpenXRDevice,
    OpenXRDeviceCfg,
    Se3KeyboardCfg,
    Se3RelRetargeterCfg,
    Se3SpaceMouseCfg,
)
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    OperationalSpaceControllerActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import DeformableBodyPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from tac_manip.tasks.manipulation.libero import mdp
from tac_manip.core.sensors import GripperContactSensorCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from tac_manip.assets import FRANKA_PANDA_LIBERO_HIGH_PD_CFG  # isort: skip


def _franka_default_joint_pose_with_optional_jitter() -> list[float]:
    explicit = os.getenv("SOFTVTBENCH_ROBOT_INIT_JOINT_POS", "").strip()
    if explicit:
        values = [float(x) for x in explicit.replace(",", " ").split()]
        if len(values) == 7:
            values = values + [0.04, 0.04]
        if len(values) != 9:
            raise ValueError(
                "SOFTVTBENCH_ROBOT_INIT_JOINT_POS expects 7 arm joints or 9 arm+finger joints, "
                f"got {len(values)}: {explicit}"
            )
        return values

    default_pose = [
        -0.019882839432642387,
        -0.18734066496238144,
        0.0076694004538321505,
        -2.4034025985475256,
        0.004964681607500244,
        2.2453365042123963,
        0.7948478983158621,
        0.04,
        0.04,
    ]
    raw_jitter = os.getenv("SOFTVTBENCH_ROBOT_JOINT_JITTER_RAD", "").strip()
    jitter = float(raw_jitter) if raw_jitter else 0.0
    if jitter <= 0.0:
        return default_pose
    seed = int(os.getenv("SOFTVTBENCH_ROBOT_INIT_SEED", "0") or "0")
    rng = random.Random(seed)
    # Keep fingers fully open. Only perturb arm joints so initial grasp state is consistent.
    for idx in range(7):
        default_pose[idx] += rng.uniform(-jitter, jitter)
    return default_pose


@configclass
class LiberoTaskConfig:
    """Configuration for Libero task parameters."""

    # Config-file and USD asset directories; the defaults can be overridden by
    # environment variables. They point at the LIBERO configs and USD assets
    # shipped in this repository.
    config_dir: str = os.getenv(
        "LIBERO_CONFIG_DIR",
        os.path.abspath("benchmarks/datasets/libero/config"),
    )
    assets_dir: str = os.getenv(
        "LIBERO_ASSETS_DATA_DIR",
        os.path.abspath("benchmarks/datasets/libero/USD"),
    )

    def __post_init__(self):
        # Load task info
        # Env var names: TASK_SUITE / TASK_ID
        self.task_suite = os.getenv("TASK_SUITE", "libero_10")
        self.task_id = os.getenv("TASK_ID", "0")
        task_suite_info_file = os.path.join(self.config_dir, f"{self.task_suite}.json")
        self.group_size = int(os.getenv("GROUP_SIZE", "1"))
        with open(task_suite_info_file) as f:
            task_suite_info = json.load(f)
            self.task_info = task_suite_info["tasks"][int(self.task_id)]
            self.workspace_name = self.task_info["workspace_name"]
            self.fixtures = self.task_info["fixtures"]  # static objects/colliders: table, floor, etc.
            self.objects = self.task_info["objects"]  # dynamic objects: cream cheese, basket, etc.
            self.regions = self.task_info["regions"]  # regions: initial position of objects
            self.obj_of_interest = self.task_info["obj_of_interest"]  # objects of interest: objects to grasp.
            self.targets = self.task_info["targets"]  # targets: objects to place onto.
            self.goals = self.task_info["goals"]  # goals: trajectory success conditions.
            self.robot_base_pos = self.task_info["robot_base_pos"]  # robot base position: [x, y, z]
            self.robot_base_ori = self.task_info["robot_base_ori"]  # robot base orientation: [w, x, y, z]

            # add targets for tactile sensor
            self.tactile_targets = self.task_info.get("tactile_targets", self.obj_of_interest)


@configclass
class EventCfgFrankaPanda:
    """Configuration for events."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": _franka_default_joint_pose_with_optional_jitter(),
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Create individual event terms for each object
    def __post_init__(self):
        libero_config = LiberoTaskConfig()
                # self.randomize_light = EventTerm(
        #     func=mdp.randomize_domelight_color_intensity,
        #     mode="reset",
        #     params={
        #         "intensity_range": (500, 700),
        #         "color_variation": 0.4,
        #         "base_color": (0.75, 0.75, 0.75),
        #         "default_intensity": 600.0,
        #         "asset_cfg": SceneEntityCfg("light"),
        #         "textures": [
        #             # f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
        #             # f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
        #             # f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
        #             # f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
        #             # f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
        #             # f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
        #             # f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
        #             # f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
        #             # f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/small_empty_house_4k.hdr",
        #             # f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/surgery_4k.hdr",
        #             # f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
        #         ],
        #         "default_texture": f"{ASSETS_DATA_DIR}/Scenes/1.jpeg",
        #     },
        # )
        # Create event terms for each object
        for obj in libero_config.objects.items():
            obj_name = obj[0]
            init_region_name = obj[1]["initial_region"]

            # Create a unique name for each object's event
            event_name = f"init_{obj_name}_pose"

            # Create the event term
            event_term = EventTerm(
                func=franka_stack_events.randomize_object_pose,
                mode="reset",
                params={
                    "pose_range": libero_config.regions[init_region_name]["pose_range"],
                    "asset_cfgs": [SceneEntityCfg(obj_name)],
                },
            )

            # Add the event term to the class
            setattr(self, event_name, event_term)


##
# Scene definition
##


@configclass
class KitchenTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the Kitchen Table scene with a robot and objects.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )

    def __post_init__(self):
        libero_config = LiberoTaskConfig()
        # add table
        self.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.845)
            ),  # make sure the knob is not underneath the table, otherwise the knob will be stuck
            spawn=UsdFileCfg(
                usd_path=f"{libero_config.assets_dir}/kitchen_table/kitchen_table.usd", scale=(0.02, 0.02, 0.02)
            ),
        )


@configclass
class LivingRoomTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the Living Room Table scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )

    def __post_init__(self):
        libero_config = LiberoTaskConfig()
        # add table
        self.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.14 * 1.5, -0.005), rot=(0.707, 0.0, 0.0, 0.707)),
            # table size: [1.2, 0.8, 0.3]
            spawn=UsdFileCfg(
                usd_path=f"{libero_config.assets_dir}/living_room_table/living_room_table.usd", scale=(1.5, 1.5, 1.5)
            ),
        )


@configclass
class FloorSceneCfg(InteractiveSceneCfg):
    """Configuration for the Floor scene with a robot and objects.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )

    def __post_init__(self):
        libero_config = LiberoTaskConfig()
        # add floor
        self.floor = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Floor",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.025)),
            # floor size: [6.0, 6.0, 0.025]
            spawn=UsdFileCfg(usd_path=f"{libero_config.assets_dir}/floor/floor.usd", scale=(1.0, 1.0, 1.0)),
        )


@configclass
class StudyTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the Study Room Table scene with a robot and objects.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )

    def __post_init__(self):
        libero_config = LiberoTaskConfig()
        # add table
        self.study_table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/study_table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.2, 0, 0.867 - 0.85)),
            spawn=UsdFileCfg(usd_path=f"{libero_config.assets_dir}/study_table/study_table.usd", scale=(1.0, 1.0, 1.0)),
        )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        # Task-space pose of end-effector (legacy state used by older pipelines)
        eef_pose = ObsTerm(func=mdp.ee_frame_pose_in_base_frame)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        # Joint-space arm state (7D arm joints), used for pi0-style OpenPI state
        arm_joint_pos = ObsTerm(func=mdp.franka_arm_joint_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        pass

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

            libero_config = LiberoTaskConfig()
            for i in range(len(libero_config.obj_of_interest)):
                grasp_i = ObsTerm(
                    func=mdp.object_grasped_w_force,
                    params={
                        "robot_cfg": SceneEntityCfg("robot"),
                        "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                        "object_cfg": SceneEntityCfg(libero_config.obj_of_interest[i]),
                        "diff_threshold": 0.25,  # larger threshold for bowl/cup-like object
                    },
                )
                setattr(self, f"grasp_{i + 1}", grasp_i)

    # observation groups
    def __post_init__(self):
        self.policy = ObservationsCfg.PolicyCfg()
        self.subtask_terms = ObservationsCfg.SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    def __post_init__(self):
        libero_config = LiberoTaskConfig()
        for i in range(len(libero_config.obj_of_interest)):
            setattr(
                self,
                f"object_{i + 1}_dropped",
                DoneTerm(
                    func=mdp.root_height_below_minimum,
                    params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg(libero_config.obj_of_interest[i])},
                ),
            )

        self.success = DoneTerm(
            func=mdp.libero_goals_reached,
            params={
                "goals": libero_config.goals,
            },
        )


@configclass
class LiberoEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: InteractiveSceneCfg = MISSING

    actions: ActionsCfg = ActionsCfg()

    # Unused managers
    commands = None
    rewards = None
    events = None
    curriculum = None


    def __post_init__(self):
        """Post initialization."""
        self.libero_config = LiberoTaskConfig()
        # Basic settings
        self.observations = ObservationsCfg()
        # MDP settings
        self.terminations = TerminationsCfg()

        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Set viewer camera pose
        self.viewer.eye = [1.0, 0.0, 2.0]
        self.viewer.lookat = [0.0, 0.0, 1.0]


@configclass
class JointPositionLiberoEnvCfg(LiberoEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Set events
        self.events = EventCfgFrankaPanda()

        # setup the scene workspace
        if self.libero_config.workspace_name == "living_room_table":
            self.scene = LivingRoomTableSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=False)
        elif self.libero_config.workspace_name == "floor":
            self.scene = FloorSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=False)
        elif self.libero_config.workspace_name == "study_table":
            self.scene = StudyTableSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=False)
        elif self.libero_config.workspace_name == "kitchen_table" or self.libero_config.workspace_name == "table":
            self.scene = KitchenTableSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=False)

        # Set Franka as robot, ** USE HIGH PD CONTROL FOR REPLAY **
        self.scene.robot = FRANKA_PANDA_LIBERO_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if os.getenv("SOFTVTBENCH_ROBOT_HIGH_FRICTION", "0").strip().lower() in ("1", "true", "yes", "on"):
            self.scene.robot.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=float(os.getenv("SOFTVTBENCH_ROBOT_STATIC_FRICTION", "5.0")),
                dynamic_friction=float(os.getenv("SOFTVTBENCH_ROBOT_DYNAMIC_FRICTION", "4.0")),
                restitution=0.0,
                friction_combine_mode=os.getenv("SOFTVTBENCH_ROBOT_FRICTION_COMBINE_MODE", "max"),
                restitution_combine_mode="min",
            )
        self.scene.robot.init_state.pos = self.libero_config.robot_base_pos
        self.scene.robot.init_state.rot = self.libero_config.robot_base_ori

        # Set joint_actions for the specific robot type (franka)
        # input action is (N, 7) for joint_pos
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=False,  # If True: use default_joint_positinos as offset
        )

        # define gripper actions for Franka Panda
        # -- binary gripper action with manual threshold 0.5: replay [-1, 1], gr00t nx evaluation [0, 1] --
        # Soft grasp experiments need a continuous finger target width. Keep
        # the default binary action for existing replay/collection scripts and
        # enable absolute position only when explicitly requested.
        gripper_action_mode = os.getenv("SOFTVTBENCH_GRIPPER_ACTION_MODE", "binary").strip().lower()
        if gripper_action_mode in ("abs", "absolute", "position", "joint_position"):
            self.actions.gripper_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["panda_finger.*"],
                scale=1.0,
                use_default_offset=False,
            )
        else:
            self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
                asset_name="robot",
                joint_names=["panda_finger.*"],
                open_command_expr={"panda_finger_.*": 0.04},
                close_command_expr={"panda_finger_.*": 0.0},
            )

        self.gripper_joint_names = ["panda_finger.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.01

        # Rigid body properties of all objects
        object_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # add all objects
        for obj in self.libero_config.objects.items():
            obj_name = obj[0]
            obj_type = obj[1]["type"]
            if obj_type == "flat_stove":  # add flat_stove as an articulation
                self.scene.flat_stove_1 = ArticulationCfg(
                    prim_path="{ENV_REGEX_NS}/flat_stove_1",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=f"{self.libero_config.assets_dir}/{obj_type}/{obj_type}.usd",
                        activate_contact_sensors=True,
                    ),
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=(0.8, 0.0, 0.0),
                        rot=(0.0, 0.0, 0.0, 1.0),
                        joint_pos={"button": 0.0},  # joint for the flat stove knob
                    ),
                    actuators={
                        "knob_actuator": ImplicitActuatorCfg(
                            joint_names_expr=["button"],
                            effort_limit_sim=10.0,  # increase the passive effort limit for easy rotation
                            velocity_limit_sim=2.0,
                            stiffness=0.0,
                            damping=0.0,
                            friction=0.5,  # default friction is from USD: 0.8, reduce to enable easy rotation but with small resistance
                            # make sure the knob is not underneath the table, otherwise the knob will be stuck
                        ),
                    },
                )

                # Frame definitions for the knob button of the flat stove.
                self.knob_frame = FrameTransformerCfg(
                    prim_path="{ENV_REGEX_NS}/flat_stove_1/flat_stove/burnerplate",
                    debug_vis=False,
                    visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/KnobFrameTransformer"),
                    target_frames=[
                        FrameTransformerCfg.FrameCfg(
                            prim_path="{ENV_REGEX_NS}/flat_stove_1/flat_stove/knob/knob",
                            name="knob",
                        ),
                    ],
                )
                # Frame definitions for the knob button of the flat stove.
                self.burnerplate_frame = FrameTransformerCfg(
                    prim_path="{ENV_REGEX_NS}/flat_stove_1/flat_stove/burnerplate",
                    debug_vis=False,
                    visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/BurnerplateFrameTransformer"),
                    target_frames=[
                        FrameTransformerCfg.FrameCfg(
                            prim_path="{ENV_REGEX_NS}/flat_stove_1/flat_stove/burnerplate",
                            name="burnerplate",
                            offset=OffsetCfg(
                                pos=(0.273 * 0.55, 0.0, 0.032 * 0.55),
                            ),
                        ),
                    ],
                )
            elif obj_type == "microwave":
                self.scene.microwave_1 = ArticulationCfg(
                    prim_path="{ENV_REGEX_NS}/microwave_1",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=f"{self.libero_config.assets_dir}/{obj_type}/{obj_type}.usd",
                        activate_contact_sensors=True,
                    ),
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=(0.0, 0.36, 0.0),
                        rot=(1.0, 0.0, 0.0, 0.0),
                        joint_pos={"microjoint": -1.79},  # joint for the microwave door
                    ),
                    actuators={
                        "door_actuator": ImplicitActuatorCfg(
                            joint_names_expr=["microjoint"],
                            effort_limit_sim=0.01,
                            velocity_limit_sim=20.0,
                            stiffness=0.0,
                            damping=0.0,
                            friction=0.2,  # default friction is from USD: 0.0
                        ),
                    },
                )
            elif obj_type == "white_cabinet":
                self.scene.white_cabinet_1 = ArticulationCfg(
                    prim_path="{ENV_REGEX_NS}/white_cabinet_1",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=f"{self.libero_config.assets_dir}/{obj_type}/{obj_type}.usd",
                        activate_contact_sensors=True,
                    ),
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=(0.0, 0.36, 0.0),
                        rot=(1.0, 0.0, 0.0, 0.0),
                        joint_pos={
                            "top_level": 0.0,  # joint for the top drawer
                            "middle_level": 0.0,  # joint for the middle drawer
                            "bottom_level": -0.1523,  # joint for the bottom drawer
                        },
                    ),
                    actuators={
                        "drawer_bottom_actuator": ImplicitActuatorCfg(
                            joint_names_expr=["bottom_level"],
                            effort_limit_sim=0.0,
                            velocity_limit_sim=20.0,
                            stiffness=0.0,
                            damping=0.0,
                            friction=0.0,  # default friction is from USD: 0.0
                        ),
                        "drawer_top_middle_actuator": ImplicitActuatorCfg(
                            joint_names_expr=["top_level", "middle_level"],
                            effort_limit_sim=10.0,
                            velocity_limit_sim=20.0,
                            stiffness=1000.0,
                            damping=0.0,
                        ),
                    },
                )
            elif obj_type == "wooden_cabinet":
                self.scene.wooden_cabinet_1 = ArticulationCfg(
                    prim_path="{ENV_REGEX_NS}/wooden_cabinet_1",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=f"{self.libero_config.assets_dir}/{obj_type}/{obj_type}.usd",
                        activate_contact_sensors=True,
                    ),
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=(0.0, 0.36, 0.0),
                        rot=(1.0, 0.0, 0.0, 0.0),
                        joint_pos={
                            "top_level": 0.0,  # joint for the top drawer
                            "middle_level": 0.0,  # joint for the middle drawer
                            "bottom_level": 0.0,  # joint for the bottom drawer
                        },
                    ),
                    actuators={
                        "drawer_actuator": ImplicitActuatorCfg(
                            joint_names_expr=["top_level", "middle_level", "bottom_level"],
                            effort_limit_sim=10.0,  # Drawer opens easily, minimal resistance, if 0.0, drawer barely moves
                            velocity_limit_sim=2.0,
                            stiffness=0.0,
                            damping=0.1,
                            # Original friction coefficient: 0.01 (almost frictionless, so a
                            # light collision could push the drawer back in).
                            # Joint friction is raised here so the drawer does not retract on
                            # incidental contact. Raise it further (e.g. 0.1, 0.2) if the
                            # drawer still feels too slippery.
                            friction=0.1,
                        ),
                    },
                )
            else:
                setattr(
                    self.scene,
                    obj_name,
                    RigidObjectCfg(
                        prim_path="{ENV_REGEX_NS}" + f"/{obj_name}",
                        init_state=RigidObjectCfg.InitialStateCfg(),
                        spawn=UsdFileCfg(
                            usd_path=f"{self.libero_config.assets_dir}/{obj_type}/{obj_type}.usd",
                            activate_contact_sensors=(
                                obj_name in self.libero_config.targets
                            ),  # need to activate contact sensor for target object
                            scale=obj[1]["scale"],
                            rigid_props=object_properties,
                        ),
                    ),
                )

        def _float_tuple_from_raw(raw: object, default: tuple[float, ...], name: str) -> tuple[float, ...]:
            if raw is None or raw == "":
                return default
            if isinstance(raw, str):
                values = tuple(float(x) for x in raw.replace(",", " ").split())
            else:
                values = tuple(float(x) for x in raw)
            if len(values) != len(default):
                raise ValueError(f"{name} expects {len(default)} floats, got {len(values)}: {raw}")
            return values

        def _float_tuple_from_env(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
            return _float_tuple_from_raw(os.getenv(name, "").strip(), default, name)

        def _bool_from_env(name: str, default: bool) -> bool:
            raw = os.getenv(name, "").strip().lower()
            if not raw:
                return default
            return raw in ("1", "true", "t", "yes", "y", "on")

        def _float_from_env(name: str, default: float | None) -> float | None:
            raw = os.getenv(name, "").strip()
            return default if not raw else float(raw)

        def _int_from_env(name: str, default: int) -> int:
            raw = os.getenv(name, "").strip()
            return default if not raw else int(raw)

        def _add_extra_asset(extra: dict[str, object], index: int = 0):
            extra_asset_usd = str(extra.get("usd_path", "")).strip()
            if not extra_asset_usd:
                return
            extra_asset_name = str(extra.get("name", f"soft_asset_{index}")).strip() or f"soft_asset_{index}"
            extra_is_deformable = bool(extra.get("deformable", False))
            extra_is_rigid = bool(extra.get("rigid", False)) and not extra_is_deformable
            extra_asset_cfg_cls = DeformableObjectCfg if extra_is_deformable else (RigidObjectCfg if extra_is_rigid else AssetBaseCfg)
            prefix = str(extra.get("env_prefix", "SOFTVTBENCH_EXTRA")).strip() or "SOFTVTBENCH_EXTRA"

            spawn_kwargs = {
                "usd_path": extra_asset_usd,
                "scale": _float_tuple_from_raw(extra.get("scale"), (1.0, 1.0, 1.0), f"{extra_asset_name}.scale"),
            }
            if extra_is_deformable:
                spawn_kwargs["deformable_props"] = DeformableBodyPropertiesCfg(
                    deformable_enabled=True,
                    self_collision=bool(extra.get("self_collision", _bool_from_env(f"{prefix}_SELF_COLLISION", False))),
                    settling_threshold=float(extra.get("settling_threshold", _float_from_env(f"{prefix}_SETTLING_THRESHOLD", 0.002))),
                    sleep_damping=float(extra.get("sleep_damping", _float_from_env(f"{prefix}_SLEEP_DAMPING", 8.0))),
                    sleep_threshold=float(extra.get("sleep_threshold", _float_from_env(f"{prefix}_SLEEP_THRESHOLD", 0.003))),
                    solver_position_iteration_count=int(extra.get("solver_iters", _int_from_env(f"{prefix}_SOLVER_ITERS", 64))),
                    vertex_velocity_damping=float(extra.get("vertex_damping", _float_from_env(f"{prefix}_VERTEX_DAMPING", 2.5))),
                    simulation_hexahedral_resolution=int(extra.get("hex_res", _int_from_env(f"{prefix}_HEX_RES", 6))),
                    collision_simplification=bool(
                        extra.get("collision_simplification", _bool_from_env(f"{prefix}_COLLISION_SIMPLIFICATION", True))
                    ),
                    collision_simplification_remeshing=bool(
                        extra.get(
                            "collision_simplification_remeshing",
                            _bool_from_env(f"{prefix}_COLLISION_SIMPLIFICATION_REMESHING", True),
                        )
                    ),
                    contact_offset=float(extra.get("contact_offset", _float_from_env(f"{prefix}_CONTACT_OFFSET", 0.003))),
                    rest_offset=float(extra.get("rest_offset", _float_from_env(f"{prefix}_REST_OFFSET", 0.001))),
                    max_depenetration_velocity=float(
                        extra.get("max_depenetration_velocity", _float_from_env(f"{prefix}_MAX_DEPENETRATION_VELOCITY", 0.05))
                    ),
                )
            elif extra_is_rigid:
                spawn_kwargs["activate_contact_sensors"] = bool(extra.get("activate_contact_sensors", False))
                spawn_kwargs["rigid_props"] = RigidBodyPropertiesCfg(
                    solver_position_iteration_count=int(extra.get("solver_iters", _int_from_env(f"{prefix}_RIGID_SOLVER_ITERS", 16))),
                    solver_velocity_iteration_count=int(extra.get("solver_velocity_iters", _int_from_env(f"{prefix}_RIGID_SOLVER_VELOCITY_ITERS", 1))),
                    max_angular_velocity=float(extra.get("max_angular_velocity", _float_from_env(f"{prefix}_MAX_ANGULAR_VELOCITY", 1000.0))),
                    max_linear_velocity=float(extra.get("max_linear_velocity", _float_from_env(f"{prefix}_MAX_LINEAR_VELOCITY", 1000.0))),
                    max_depenetration_velocity=float(
                        extra.get("max_depenetration_velocity", _float_from_env(f"{prefix}_RIGID_MAX_DEPENETRATION_VELOCITY", 2.0))
                    ),
                    disable_gravity=bool(extra.get("disable_gravity", False)),
                )

            setattr(
                self.scene,
                extra_asset_name,
                extra_asset_cfg_cls(
                    prim_path="{ENV_REGEX_NS}" + f"/{extra_asset_name}",
                    init_state=extra_asset_cfg_cls.InitialStateCfg(
                        pos=_float_tuple_from_raw(extra.get("pos"), (0.0, 0.0, 0.05), f"{extra_asset_name}.pos"),
                        rot=_float_tuple_from_raw(extra.get("rot"), (1.0, 0.0, 0.0, 0.0), f"{extra_asset_name}.rot"),
                    ),
                    spawn=UsdFileCfg(**spawn_kwargs),
                ),
            )

        extra_asset_usd = os.getenv("SOFTVTBENCH_EXTRA_ASSET_USD", "").strip()
        if extra_asset_usd:
            _add_extra_asset(
                {
                    "name": os.getenv("SOFTVTBENCH_EXTRA_ASSET_NAME", "soft_asset").strip() or "soft_asset",
                    "usd_path": extra_asset_usd,
                    "deformable": os.getenv("SOFTVTBENCH_EXTRA_ASSET_DEFORMABLE") == "1",
                    "rigid": os.getenv("SOFTVTBENCH_EXTRA_ASSET_RIGID") == "1",
                    "pos": _float_tuple_from_env("SOFTVTBENCH_EXTRA_ASSET_POS", (0.0, 0.0, 0.05)),
                    "rot": _float_tuple_from_env("SOFTVTBENCH_EXTRA_ASSET_ROT", (1.0, 0.0, 0.0, 0.0)),
                    "scale": _float_tuple_from_env("SOFTVTBENCH_EXTRA_ASSET_SCALE", (1.0, 1.0, 1.0)),
                }
            )

        extra_assets_json = os.getenv("SOFTVTBENCH_EXTRA_ASSETS_JSON", "").strip()
        if extra_assets_json:
            for i, extra in enumerate(json.loads(extra_assets_json)):
                _add_extra_asset(extra, i)

        # add contact force sensor for grasped checking
        skip_deformable_contact_sensors = _bool_from_env("SOFTVTBENCH_SKIP_DEFORMABLE_CONTACT_SENSORS", False)
        extra_deformable_names: set[str] = set()
        if skip_deformable_contact_sensors:
            extra_asset_name = os.getenv("SOFTVTBENCH_EXTRA_ASSET_NAME", "").strip()
            if extra_asset_name and os.getenv("SOFTVTBENCH_EXTRA_ASSET_DEFORMABLE") == "1":
                extra_deformable_names.add(extra_asset_name)
            extra_assets_json = os.getenv("SOFTVTBENCH_EXTRA_ASSETS_JSON", "").strip()
            if extra_assets_json:
                for extra in json.loads(extra_assets_json):
                    if bool(extra.get("deformable", False)):
                        extra_name = str(extra.get("name", "")).strip()
                        if extra_name:
                            extra_deformable_names.add(extra_name)
        for obj in self.libero_config.obj_of_interest:
            if obj in extra_deformable_names:
                continue
            setattr(
                self.scene,
                f"contact_grasp_{obj}",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_.*finger",
                    update_period=0.0,
                    history_length=6,
                    debug_vis=False,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/" + f"{obj}"],
                ),
            )

        # multiple rigid bodies exist for articulations, only check contact with a specific rigid body
        rigid_body_prim_name = {
            "flat_stove_1": "flat_stove_1/flat_stove/burnerplate",
            "microwave_1": "microwave_1/microwave/microwave/Xform",
            "white_cabinet_1": "white_cabinet_1/wooden_cabinet_base_col/wooden_cabinet_bottom/drawer_bottom",
            "wooden_cabinet_1": [
                "wooden_cabinet_1/wooden_cabinet_base_col/wooden_cabinet_top/drawer_top",  # top drawer rigid_body, check if obj is in the top drawer
                "wooden_cabinet_1/wooden_cabinet_base_col/wooden_cabinet_base_col",
            ],  # base rigid_body, check if obj is onto the cabinet
        }

        # add contact force sensor for object contact checking with targets
        for item in self.libero_config.goals:
            if "relationship" in item:
                target_name = item["target"]
                obj_name = item["ref_obj"]
                if target_name == "wooden_cabinet_1":
                    if item["relationship"] == "in":
                        target_prim_path = rigid_body_prim_name[target_name][0]
                    elif item["relationship"] == "on":
                        target_prim_path = rigid_body_prim_name[target_name][1]
                    else:
                        raise ValueError(f"Invalid relationship for wooden cabinet: {item['relationship']}")
                else:
                    target_prim_path = rigid_body_prim_name.get(target_name, target_name)

                setattr(
                    self.scene,
                    f"contact_{target_name}_{obj_name}",
                    ContactSensorCfg(
                        prim_path="{ENV_REGEX_NS}/"
                        + target_prim_path,  # this prim must be the prim with RigidBody setup
                        update_period=0.0,
                        history_length=6,
                        debug_vis=False,
                        filter_prim_paths_expr=["{ENV_REGEX_NS}/" + f"{obj_name}"],
                    ),
                )

        # Listens to the required transforms
        self.marker_cfg = FRAME_MARKER_CFG.copy()
        self.marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=self.marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )
        # add frame transformer for gripper frame
        self.scene.left_gripper_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=self.marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="left_gripper",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.045], rot=[0.5, 0.5, 0.5, -0.5]),
                ),
            ],
        )
        self.scene.right_gripper_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=self.marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="right_gripper",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.045], rot=[0.5, -0.5, 0.5, 0.5]),
                ),
            ],
        )

        # Set the simulation parameters
        self.sim.dt = 1 / 60
        self.sim.render_interval = 3

        self.decimation = 3
        self.episode_length_s = 16.0 if self.libero_config.task_suite == "libero_10" else 8.0

        # Set settings for camera rendering
        self.rerender_on_reset = True

        # teleop devices
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.2,
                    sim_device=self.sim.device,
                ),
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
            }
        )


@configclass
class JointPositionLiberoCameraEnvCfg(JointPositionLiberoEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set eye_in_hand camera
        self.scene.eye_in_hand_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/eye_in_hand_camera",
            update_period=0.05,
            height=512,
            width=512,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=9.77, focus_distance=400.0, horizontal_aperture=15.0, clipping_range=(0.001, 1.0e5)
            ),  # fovy = 75
            offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(0.0, 0.707108, 0.707108, 0.0), convention="opengl"),
        )

        # Set agentview camera
        if self.libero_config.workspace_name == "living_room_table":
            self.scene.agentview_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/agentview_camera",
                update_period=0.05,
                height=512,
                width=512,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0, focus_distance=400.0, horizontal_aperture=15.0, clipping_range=(0.1, 1.9)
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.6065773716836134, 0.0, 0.96),
                    rot=(
                        0.6182166934013367,
                        0.3432307541370392,
                        0.3432314395904541,
                        0.6182177066802979,
                    ),
                    convention="opengl",
                ),
            )
        elif self.libero_config.workspace_name == "kitchen_table" or self.libero_config.workspace_name == "table":
            self.scene.agentview_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/agentview_camera",
                update_period=0.05,
                height=512,
                width=512,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0, focus_distance=400.0, horizontal_aperture=15.0, clipping_range=(0.1, 1.9)
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.6586131746834771, 0.0, 1.6103500240372423),
                    rot=(
                        0.6380177736282349,
                        0.3048497438430786,
                        0.30484986305236816,
                        0.6380177736282349,
                    ),
                    convention="opengl",
                ),
            )
        elif self.libero_config.workspace_name == "floor":
            self.scene.agentview_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/agentview_camera",
                update_period=0.05,
                height=512,
                width=512,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0, focus_distance=400.0, horizontal_aperture=15.0, clipping_range=(0.1, 1.9)
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.8965773716836134, 5.216182733499864e-07, 0.65),
                    rot=(
                        0.6182166934013367,
                        0.3432307541370392,
                        0.3432314395904541,
                        0.6182177066802979,
                    ),
                    convention="opengl",
                ),
            )
        elif self.libero_config.workspace_name == "study_table":
            self.scene.agentview_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/agentview_camera",
                update_period=0.05,
                height=512,
                width=512,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0, focus_distance=400.0, horizontal_aperture=15.0, clipping_range=(0.1, 1.9)
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.4586131746834771, 0.0, 1.6103500240372423),
                    rot=(
                        0.6380177736282349,
                        0.3048497438430786,
                        0.30484986305236816,
                        0.6380177736282349,
                    ),
                    convention="opengl",
                ),
            )
        else:
            raise ValueError(
                f"Workspace type {self.libero_config.workspace_name} not supported for agentviewcamera setup."
            )

        # Set settings for camera rendering
        self.rerender_on_reset = True
        # set assistant cameras for easy teleoperation
        self.teleop_camera_positions = [(1.0, 0, 1.2), (-0.2, -1.0, 1.2), (-0.25, 0, 2.3)]
        self.teleop_camera_rotations = [(90, 0, 90), (90, 0, 0), (0, 0, 90)]


@configclass
class IKLiberoCameraEnvCfg(JointPositionLiberoCameraEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # read use_relative_mode from environment variable
        use_relative_mode_env = os.getenv("USE_RELATIVE_MODE", "False")
        self.use_relative_mode = use_relative_mode_env.lower() in ["true", "1", "t"]

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", 
                use_relative_mode=self.use_relative_mode, 
                ik_method="dls",
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1034]),
        )

        # define gripper actions for Franka Panda if in teleoperation mode
        gripper_action_mode = os.getenv("SOFTVTBENCH_GRIPPER_ACTION_MODE", "binary").strip().lower()
        if gripper_action_mode in ("abs", "absolute", "position", "joint_position"):
            self.actions.gripper_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["panda_finger.*"],
                scale=1.0,
                use_default_offset=False,
            )
        elif self.use_relative_mode:
            self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
                asset_name="robot",
                joint_names=["panda_finger.*"],
                open_command_expr={"panda_finger_.*": 0.04},
                close_command_expr={"panda_finger_.*": 0.0},
            )



# -- this is OSC_POSE controller tuned from robosuite --
@configclass
class OscPoseLiberoCameraEnvCfg(JointPositionLiberoCameraEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # disable PD control for the arm, OSC use torque control
        self.scene.robot.actuators["panda_shoulder"].stiffness = 0.0
        self.scene.robot.actuators["panda_shoulder"].damping = 0.0
        self.scene.robot.actuators["panda_forearm"].stiffness = 0.0
        self.scene.robot.actuators["panda_forearm"].damping = 0.0

        self.osc_type = os.getenv("LIBERO_OSC_TYPE", "pose_rel")

        # Set actions for the specific robot type (franka) using OSC_POSE controller
        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=[self.osc_type],
                impedance_mode="fixed",
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=False,
                motion_stiffness_task=(  # more stiffness for Lab teleoperation
                    150.0*8,
                    150.0*8,
                    150.0*8,
                    150.0*8,
                    150.0*8,
                    150.0*8,
                ),  # Fixed impedance values from robosuite
                motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                nullspace_control="position",
                nullspace_stiffness=10.0,  # joint_kp=10 from robosuite
            ),
            nullspace_joint_pos_target="default",
            position_scale=(
                1.0 if self.osc_type == "pose_abs" else 1.0 / 20.0
            ),  # scaling values from robosuite: input [-1, 1] -> output [-0.05, 0.05]
            orientation_scale=(
                1.0 if self.osc_type == "pose_abs" else 1.0 / 2.0
            ),  # scaling values from robosuite: input [-1, 1] -> output [-0.5, 0.5]
            body_offset=OperationalSpaceControllerActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1034]),
        )

        # -- note: if replay original libero dataset: -1 is open, 1 is close, else comment out -- #
        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["panda_finger.*"],
        #     open_command_expr={"panda_finger_.*": 0.0},
        #     close_command_expr={"panda_finger_.*": 0.04},
        # )

        # teleop devices
        self.teleop_devices.devices['keyboard'].pos_sensitivity = 1.0
        self.teleop_devices.devices['keyboard'].rot_sensitivity = 0.5
        self.teleop_devices.devices['spacemouse'].pos_sensitivity = 1.0
        self.teleop_devices.devices['spacemouse'].rot_sensitivity = 0.5
