# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import os

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from tac_manip.core.sensors import GripperContactSensorCfg
from tac_manip.tasks.manipulation.libero import mdp

from .franka_libero_env_cfg import IKLiberoCameraEnvCfg, JointPositionLiberoCameraEnvCfg

from tac_manip.assets import (  # isort: skip
    FRANKA_PANDA_LIBERO_HIGH_PD_WITH_GSMINI_CFG,
    GELSIGHT_MINI_TAXIM_FOTS_CFG,
)


##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

# Force history length (H) used for **inference/testing** in Hybrid-style environments.
# - This is mainly for Hybrid/ForcePosition controllers and debugging/metrics that may benefit from
#   a short force history window (e.g., smoothing, temporal features).
# - Recording datasets (softvtbench / softvtbench_force) should NOT rely on this window; they always record
#   only the current force frame via `RECORD_FORCE_HISTORY_LENGTH = 1`.
HYBRID_FORCE_HISTORY_LENGTH = 8

# Recording-time force history length:
# - For both softvtbench_force (ContactForce replay env) and softvtbench (Tactile replay env),
#   we only record the *current* frame to keep datasets clean and lightweight.
# - Any force-history window (e.g., H=8) should be constructed in offline conversion scripts.
RECORD_FORCE_HISTORY_LENGTH = 1


##
# New observations with tactile sensor
##
@configclass
class MarkerMotionPolicyCfg(ObsGroup):
    """Observations for policy group with state values."""

    actions = ObsTerm(func=mdp.last_action)
    eef_pose = ObsTerm(func=mdp.ee_frame_pose_in_base_frame)
    gripper_pos = ObsTerm(func=mdp.gripper_pos)
    gripper_marker_motion = ObsTerm(
        func=mdp.gripper_marker_motion_data,
        params={
            "sensor_name_list": ["gsmini_left", "gsmini_right"],
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class TactileForceAndMotionPolicyCfg(ObsGroup):
    """Policy observations for tactile Franka with GelSight + gelpad force.

    - EEF pose + gripper position + joint-space arm state
    - Tactile marker motion field from both GelSight minis
    - Net 3D force on each gelpad (left/right), expressed in gripper frames
    """

    actions = ObsTerm(func=mdp.last_action)
    eef_pose = ObsTerm(func=mdp.ee_frame_pose_in_base_frame)
    gripper_pos = ObsTerm(func=mdp.gripper_pos)
    # Joint-space arm state for pi0-style obs (7 arm joints)
    arm_joint_pos = ObsTerm(func=mdp.franka_arm_joint_pos)

    # GelSight Marker Motion (optical flow / displacement field)
    gripper_marker_motion = ObsTerm(
        func=mdp.gripper_marker_motion_data,
        params={
            "sensor_name_list": ["gsmini_left", "gsmini_right"],
        },
    )

    # 6D net force on Gelpad (left/right, each 3D) – recording uses current frame only (H=1)
    gripper_net_force = ObsTerm(
        func=mdp.contact_force_in_gripper_frame,
        params={
            "contact_sensor_name": "contact_gripper",
            "left_gripper_frame_cfg": SceneEntityCfg("left_gripper_frame"),
            "right_gripper_frame_cfg": SceneEntityCfg("right_gripper_frame"),
            "history_length": RECORD_FORCE_HISTORY_LENGTH,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class TactileDualForceAndMotionPolicyCfg(TactileForceAndMotionPolicyCfg):
    """Tactile observations plus an extra structure-force diagnostic channel.

    `gripper_net_force` remains the GelSight gelpad force used by the existing
    13D recorder. `gripper_structure_net_force` records contacts on the rigid
    finger bodies, which is useful for drawer/handle tasks where the gripper
    structure touches the object without indenting the gelpad.
    """

    gripper_structure_net_force = ObsTerm(
        func=mdp.contact_force_in_gripper_frame,
        params={
            "contact_sensor_name": "contact_gripper_structure",
            "left_gripper_frame_cfg": SceneEntityCfg("left_gripper_frame"),
            "right_gripper_frame_cfg": SceneEntityCfg("right_gripper_frame"),
            "history_length": RECORD_FORCE_HISTORY_LENGTH,
        },
    )


##
# New Tasks with tactile sensor
##


@configclass
class JointPositionContactForceLiberoCameraEnvCfg(JointPositionLiberoCameraEnvCfg):
    """
    Add a GripperContactSensor attached to the gripper fingers with various debug visualization settings
    - the sensor does not need a filter_prim_paths_expr, since it is designed to report forces for fingers no matter what they are contacting
    - debug_vis should be set to True to plot any visualization, default is to plot contact spheres
    - visualize_net_force_arrows and visualize_triaxial_forces should be set to True to plot net force arrows and triaxial forces
    - left_finger_offset and right_finger_offset are used to define the frame transformation for the sensor
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Recording-time policy observations:
        # - Keep the same keys as HybridForcePolicyCfg
        # - But record only the current force frame (H=1) for dataset cleanliness.
        self.observations.policy = ContactForceRecordingPolicyCfg()

        # add contact force sensor for gripper fingers (record current frame only)
        self.scene.contact_gripper = GripperContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_.*finger",  # this prim_path must be the leaf rigid body actually in contact with the object
            update_period=0.0,
            history_length=RECORD_FORCE_HISTORY_LENGTH,
            debug_vis=False,  # if True, default is to plot contact spheres
            # visualize_net_force_arrows=True,  # if True, plot net force arrows
            visualize_triaxial_forces=True,  # if True, plot triaxial forces
            max_force=5.0,
            max_force_arrow_length=0.5,
            left_finger_offset=OffsetCfg(
                pos=(0.0, 0.0, 0.045), rot=(0.5, 0.5, 0.5, -0.5)
            ),  # offset to make +Z towards grasp direction and at finger tip
            right_finger_offset=OffsetCfg(
                pos=(0.0, 0.0, 0.045), rot=(0.5, -0.5, 0.5, 0.5)
            ),  # offset to make +Z towards grasp direction and at finger tip
        )


@configclass
class IKContactForceLiberoCameraEnvCfg(IKLiberoCameraEnvCfg):
    """IK control (pure position) + finger contact forces (standard Franka), with camera."""

    def __post_init__(self):
        # post init of parent (IK + camera base env)
        super().__post_init__()

        # Recording-time policy observations:
        # - Keep the same keys as HybridForcePolicyCfg
        # - But record only the current force frame (H=1) for dataset cleanliness.
        self.observations.policy = ContactForceRecordingPolicyCfg()

        # add contact force sensor for gripper fingers (record current frame only)
        contact_prim_expr = os.environ.get(
            "SOFTVTBENCH_CONTACT_PRIM_EXPR",
            "{ENV_REGEX_NS}/Robot/panda_.*finger",
        )
        self.scene.contact_gripper = GripperContactSensorCfg(
            prim_path=contact_prim_expr,
            update_period=0.0,
            history_length=RECORD_FORCE_HISTORY_LENGTH,
            debug_vis=False,
            visualize_triaxial_forces=True,
            max_force=5.0,
            max_force_arrow_length=0.5,
            left_finger_offset=OffsetCfg(pos=(0.0, 0.0, 0.045), rot=(0.5, 0.5, 0.5, -0.5)),
            right_finger_offset=OffsetCfg(pos=(0.0, 0.0, 0.045), rot=(0.5, -0.5, 0.5, 0.5)),
        )


@configclass
class HybridForcePolicyCfg(ObsGroup):
    """Policy observations for HybridForce controllers (pose + gripper + net finger forces)."""

    actions = ObsTerm(func=mdp.last_action)
    eef_pose = ObsTerm(func=mdp.ee_frame_pose_in_base_frame)
    gripper_pos = ObsTerm(func=mdp.gripper_pos)
    # Joint-space arm state for pi0-style obs (7 arm joints)
    arm_joint_pos = ObsTerm(func=mdp.franka_arm_joint_pos)
    # Fingertip force observation from the contact_gripper sensor, shaped (N, H, 2, 3).
    gripper_net_force = ObsTerm(
        func=mdp.contact_force_in_gripper_frame,
        params={
            "contact_sensor_name": "contact_gripper",
            "left_gripper_frame_cfg": SceneEntityCfg("left_gripper_frame"),
            "right_gripper_frame_cfg": SceneEntityCfg("right_gripper_frame"),
            "history_length": HYBRID_FORCE_HISTORY_LENGTH,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class ContactForceRecordingPolicyCfg(ObsGroup):
    """Policy observations for *recording* ContactForce replay envs.

    Same keys as `HybridForcePolicyCfg`, but with `gripper_net_force` using current frame only (H=1).
    Any desired force-history window should be constructed offline in conversion scripts.
    """

    actions = ObsTerm(func=mdp.last_action)
    eef_pose = ObsTerm(func=mdp.ee_frame_pose_in_base_frame)
    gripper_pos = ObsTerm(func=mdp.gripper_pos)
    arm_joint_pos = ObsTerm(func=mdp.franka_arm_joint_pos)
    gripper_net_force = ObsTerm(
        func=mdp.contact_force_in_gripper_frame,
        params={
            "contact_sensor_name": "contact_gripper",
            "left_gripper_frame_cfg": SceneEntityCfg("left_gripper_frame"),
            "right_gripper_frame_cfg": SceneEntityCfg("right_gripper_frame"),
            "history_length": RECORD_FORCE_HISTORY_LENGTH,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class ForcePositionLiberoCameraEnvCfg(JointPositionLiberoCameraEnvCfg):
    """Franka Libero camera environment with hybrid fingertip force-position control (DiffIK)."""

    def __post_init__(self):
        # Initialise scene, robot, sensors and cameras via the parent class first.
        super().__post_init__()

        # HybridForce policy observations: the existing pose + gripper terms plus
        # gripper_net_force, so obs["policy"]["gripper_net_force"] is available both in
        # the recorded HDF5 and during OpenPI inference.
        self.observations.policy = HybridForcePolicyCfg()

        # The custom ForcePositionAction is the only arm action; it wraps DiffIK plus
        # fingertip force control.
        # Notes:
        # - The action is 13-D: 6-D absolute EEF pose (axis-angle) + 1-D gripper +
        #   3-D local force for each of the two fingers.
        # - ForcePositionAction already drives the gripper, so no separate
        #   gripper_action is configured here.
        self.actions.arm_action = mdp.ForcePositionActionCfg(
            asset_name="robot",
            ik_cfg=DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=False,
                    ik_method="dls",
                ),
                scale=1.0,
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
            ee_frame_name="ee_frame",
            left_gripper_frame_name="left_gripper_frame",
            right_gripper_frame_name="right_gripper_frame",
            contact_sensor_name="contact_gripper",
            history_length=1,
            pos_kp=(-0.0001, -0.0001, -0.0001), #-0.0001
            squeeze_kp=0.0002,
        )

        # Tell the ActionManager explicitly which implementation class to use.
        self.actions.arm_action.class_type = mdp.ForcePositionAction

        # Disable the legacy binary gripper action so it cannot fight ForcePositionAction.
        self.actions.gripper_action = None

        # Add the gripper force sensor that ForcePositionAction consumes.
        self.scene.contact_gripper = GripperContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_.*finger",  # the two end-effector fingers
            update_period=0.0,
            history_length=HYBRID_FORCE_HISTORY_LENGTH,
            debug_vis=False,
            visualize_triaxial_forces=True,
            max_force=5.0,
            max_force_arrow_length=0.5,
            left_finger_offset=OffsetCfg(
                pos=(0.0, 0.0, 0.045), rot=(0.5, 0.5, 0.5, -0.5)
            ),
            right_finger_offset=OffsetCfg(
                pos=(0.0, 0.0, 0.045), rot=(0.5, -0.5, 0.5, 0.5)
            ),
        )


@configclass
class JointPositionTactileLiberoCameraEnvCfg(JointPositionLiberoCameraEnvCfg):
    """
    Tactile Franka Libero environment with GelSight mini sensors and gelpad forces.

    - Uses Franka with mounted GelSight Mini sensors (left/right)
    - Provides:
      * RGB / depth from cameras (agentview, eye_in_hand)
      * Tactile images / videos from GelSight sensors (handled in replay scripts)
      * Marker motion field from GelSight sensors
      * Net forces on gelpads via GripperContactSensor attached to gelpad prims
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Use combined tactile + force + marker motion policy observations
        # Notes:
        # - eef_pose / gripper_pos / arm_joint_pos match the original JointPositionLibero.
        # - gripper_marker_motion: marker motion field from the left/right GelSight Mini.
        # - gripper_net_force: gelpad force from the contact_gripper sensor
        #   (H=1, 2 fingers x 3D).
        self.observations.policy = TactileForceAndMotionPolicyCfg()

        # replace robot with franka with gelsight mini sensor
        self.scene.robot = FRANKA_PANDA_LIBERO_HIGH_PD_WITH_GSMINI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = self.libero_config.robot_base_pos
        self.scene.robot.init_state.rot = self.libero_config.robot_base_ori

        # add gelsight mini sensor with rgb and marker motion field output
        self.tactile_marker_0_cfg = FRAME_MARKER_CFG.copy()
        self.tactile_marker_0_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        self.tactile_marker_0_cfg.prim_path = "/Visuals/FrameTransformer_gelpad_left"
        self.tactile_marker_1_cfg = self.tactile_marker_0_cfg.copy()
        self.tactile_marker_1_cfg.prim_path = "/Visuals/FrameTransformer_gelpad_right"

        self.scene.gsmini_left = GELSIGHT_MINI_TAXIM_FOTS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/gelsight_mini_case_left",  # prim to attach camera sensor
            debug_vis=False,  # visualizer for tactile sensor output
        )

        # Keep marker-motion frame transformer consistent with ForcePositionTactileLiberoCameraEnvCfg
        # so that the recorded obs structure is aligned across tactile vs. contact-sensor envs.
        self.scene.gsmini_left.marker_motion_sim_cfg.frame_transformer_cfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gelpad_left",  # track the gelpad frame
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/" + name)
                for name in self.libero_config.tactile_targets
            ],
            debug_vis=False,
            visualizer_cfg=self.tactile_marker_0_cfg,
        )

        self.scene.gsmini_right = GELSIGHT_MINI_TAXIM_FOTS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/gelsight_mini_case_right",
            debug_vis=False,
        )
        self.scene.gsmini_right.marker_motion_sim_cfg.frame_transformer_cfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gelpad_right",
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/" + name)
                for name in self.libero_config.tactile_targets
            ],
            debug_vis=False,
            visualizer_cfg=self.tactile_marker_1_cfg,
        )

        # IMPORTANT:
        # The 13D (7dpf) recorder extracts Force(6) from obs["policy"]["gripper_net_force"].
        # `gripper_net_force` is computed from the `contact_gripper` sensor. Without this sensor
        # the observation silently becomes zeros (see mdp.contact_force_in_gripper_frame).
        #
        # Diagnostic override: read contact forces directly from the gelpad rigid bodies.
        # The previous finger-level sensor only reported the initial squeeze/impact force
        # and then dropped to zero while the bowl was still being transported.
        self.scene.contact_gripper = GripperContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gelpad_.*",
            update_period=0.0,
            history_length=RECORD_FORCE_HISTORY_LENGTH,
            debug_vis=False,
            visualize_triaxial_forces=True,
            max_force=5,
            max_force_arrow_length=0.5,
            left_finger_offset=OffsetCfg(
                pos=(0.0, 0.0, 0.045), rot=(0.5, 0.5, 0.5, -0.5)
            ),
            right_finger_offset=OffsetCfg(
                pos=(0.0, 0.0, 0.045), rot=(0.5, -0.5, 0.5, 0.5)
            ),
        )


@configclass
class JointPositionTactileDualForceLiberoCameraEnvCfg(JointPositionTactileLiberoCameraEnvCfg):
    """Tactile replay environment with both gelpad and rigid-finger force sensors."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy = TactileDualForceAndMotionPolicyCfg()

        self.scene.contact_gripper_structure = GripperContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_.*finger",
            update_period=0.0,
            history_length=RECORD_FORCE_HISTORY_LENGTH,
            debug_vis=False,
            visualize_triaxial_forces=False,
            max_force=120,
            max_force_arrow_length=0.5,
            left_finger_offset=OffsetCfg(
                pos=(0.0, 0.0, 0.045), rot=(0.5, 0.5, 0.5, -0.5)
            ),
            right_finger_offset=OffsetCfg(
                pos=(0.0, 0.0, 0.045), rot=(0.5, -0.5, 0.5, 0.5)
            ),
        )


@configclass
class ForcePositionTactileLiberoCameraEnvCfg(JointPositionTactileLiberoCameraEnvCfg):
    """
    Franka Libero camera environment combining hybrid force-position control with
    GelSight tactile sensing.

    Builds on `JointPositionTactileLiberoCameraEnvCfg` (which mounts the GelSight Mini
    and the gelpad force sensors) and reuses the 13-D ForcePositionAction control logic
    from `ForcePositionLiberoCameraEnvCfg`:
    - 6-D absolute EEF pose (axis-angle)
    - 1-D gripper
    - 3-D local force for each of the two fingers
    """

    def __post_init__(self):
        # Initialise the tactile scene, robot, sensors and cameras first.
        super().__post_init__()

        # Keep the tactile environment's policy observations by default (marker_motion
        # plus gelpad force) so recording and replay both expose tactile and force
        # information. To align strictly with OpenPI's HybridForce obs instead, use:
        #   self.observations.policy = HybridForcePolicyCfg()

        # Use the same 13-D ForcePositionAction arm action as ForcePositionLiberoCameraEnvCfg.
        self.actions.arm_action = mdp.ForcePositionActionCfg(
            asset_name="robot",
            ik_cfg=DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=False,
                    ik_method="dls",
                ),
                scale=1.0,
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
            ee_frame_name="ee_frame",
            left_gripper_frame_name="left_gripper_frame",
            right_gripper_frame_name="right_gripper_frame",
            contact_sensor_name="contact_gripper",
            history_length=1,
            pos_kp=(-0.0001, -0.0001, -0.0001),
            squeeze_kp=0.0008,
            squeeze_deadzone=0.25,
            # pos_kp=(0.0, 0.0, 0.0),
            # squeeze_kp=0.0
        )

        # Pin the action implementation class and disable the legacy gripper_action.
        self.actions.arm_action.class_type = mdp.ForcePositionAction
        self.actions.gripper_action = None
        self.scene.gsmini_left.marker_motion_sim_cfg.frame_transformer_cfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gelpad_left",  # prim to gelpad, to track the transformation of indenter to gelpad
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/" + name)
                for name in self.libero_config.tactile_targets
            ],
            debug_vis=False,  # visualizer for frame transformer
            visualizer_cfg=self.tactile_marker_0_cfg,
        )
        self.scene.gsmini_right.marker_motion_sim_cfg.frame_transformer_cfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gelpad_right",  # prim to gelpad, to track the transformation of indenter to gelpad
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/" + name)
                for name in self.libero_config.tactile_targets
            ],
            debug_vis=False,  # visualizer for frame transformer
            visualizer_cfg=self.tactile_marker_1_cfg,
        )

        # Add contact force sensor for gripper fingers.
        # NOTE: PhysX GPU attributes FixedJoint child body (gelpad) contact forces to the
        # parent articulation link (finger). Bind to finger to get correct force readings.
        contact_prim_expr = os.environ.get(
            "SOFTVTBENCH_CONTACT_PRIM_EXPR",
            "{ENV_REGEX_NS}/Robot/panda_.*finger",
        )
        self.scene.contact_gripper = GripperContactSensorCfg(
            prim_path=contact_prim_expr,
            update_period=0.0,
            history_length=RECORD_FORCE_HISTORY_LENGTH,
            debug_vis=False,
            visualize_triaxial_forces=True,
            max_force=30,
            max_force_arrow_length=0.5,
            left_finger_offset=OffsetCfg(
                pos=(0.0, 0.0, 0.045), rot=(0.5, 0.5, 0.5, -0.5)
            ),
            right_finger_offset=OffsetCfg(
                pos=(0.0, 0.0, 0.045), rot=(0.5, -0.5, 0.5, 0.5)
            ),
        )




@configclass
class IKTactileLiberoCameraEnvCfg(IKLiberoCameraEnvCfg):
    """
    Add gelsight mini sensor with rgb and marker motion field output
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # IMPORTANT:
        # This env is intended for teleop/manual data collection (e.g. scripts/tools/record_demos.py)
        # using IK control, while keeping the *same* tactile-related observation keys as the replay
        # tactile env:
        #   - obs/policy/gripper_marker_motion
        #   - obs/policy/gripper_net_force  (gelpad contact forces, 2 fingers × 3D)
        # so downstream datasets / external collectors can reuse the same parsing logic.
        self.observations.policy = TactileForceAndMotionPolicyCfg()

        # replace robot with franka with gelsight mini sensor
        self.scene.robot = FRANKA_PANDA_LIBERO_HIGH_PD_WITH_GSMINI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = self.libero_config.robot_base_pos
        self.scene.robot.init_state.rot = self.libero_config.robot_base_ori

        # add gelsight mini sensor with rgb and marker motion field output
        self.tactile_marker_0_cfg = FRAME_MARKER_CFG.copy()
        self.tactile_marker_0_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        self.tactile_marker_0_cfg.prim_path = "/Visuals/FrameTransformer_gelpad_left"
        self.tactile_marker_1_cfg = self.tactile_marker_0_cfg.copy()
        self.tactile_marker_1_cfg.prim_path = "/Visuals/FrameTransformer_gelpad_right"

        self.scene.gsmini_left = GELSIGHT_MINI_TAXIM_FOTS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/gelsight_mini_case_left",  # prim to attach camera sensor
            debug_vis=False,  # visualizer for tactile sensor output
        )
        self.scene.gsmini_left.marker_motion_sim_cfg.frame_transformer_cfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gelpad_left",  # prim to gelpad, to track the transformation of indenter to gelpad
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/" + name)
                for name in self.libero_config.tactile_targets
            ],  # FIXME: you need to know in advance which object is the contact target
            debug_vis=False,  # visualizer for frame transformer
            visualizer_cfg=self.tactile_marker_0_cfg,
        )

        self.scene.gsmini_right = GELSIGHT_MINI_TAXIM_FOTS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/gelsight_mini_case_right",  # prim to attach camera sensor
            debug_vis=False,  # visualizer for tactile sensor output
        )
        self.scene.gsmini_right.marker_motion_sim_cfg.frame_transformer_cfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gelpad_right",  # prim to gelpad, to track the transformation of indenter to gelpad
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/" + name)
                for name in self.libero_config.tactile_targets
            ],
            debug_vis=False,  # visualizer for frame transformer
            visualizer_cfg=self.tactile_marker_1_cfg,
        )
        
        # Match the replay tactile env: record contact forces on the two GelSight gelpads.
        # The 7dpf recorder stores obs["policy"]["gripper_net_force"] as Force(6).
        contact_prim_expr = os.environ.get(
            "SOFTVTBENCH_CONTACT_PRIM_EXPR",
            "{ENV_REGEX_NS}/Robot/gelpad_.*",
        )
        self.scene.contact_gripper = GripperContactSensorCfg(
            prim_path=contact_prim_expr,
            update_period=0.0,
            history_length=RECORD_FORCE_HISTORY_LENGTH,
            debug_vis=False,
            visualize_triaxial_forces=True,
            max_force=5,
            max_force_arrow_length=0.5,
            left_finger_offset=OffsetCfg(pos=(0.0, 0.0, 0.045), rot=(0.5, 0.5, 0.5, -0.5)),
            right_finger_offset=OffsetCfg(pos=(0.0, 0.0, 0.045), rot=(0.5, -0.5, 0.5, 0.5)),
        )
        
        # IMPORTANT: Override ee_frame offset for tactile sensor robot
        # Tactile sensor robot has different physical structure
        # Use standard Franka offset (0.107m) for tactile environment
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=self.marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.107],  # Standard Franka offset for tactile robot
                    ),
                ),
            ],
        )
