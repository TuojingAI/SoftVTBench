# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs.mdp.recorders import (
    InitialStateRecorderCfg,
    PostStepStatesRecorderCfg,
    PreStepFlatPolicyObservationsRecorderCfg,
)
from isaaclab.managers.recorder_manager import (
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.utils import configclass

from . import recorders

##
# State recorders.
##


@configclass
class PostStepAbsEEFPoseBinaryGripperActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the step absolute eef pose + binary gripper actions recorder term."""

    class_type: type[RecorderTerm] = recorders.PostStepAbsEEFPoseBinaryGripperActionsRecorder


@configclass
class PostStepAbsEEFPoseAbsGripperActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the step absolute eef pose + absolute gripper actions recorder term."""

    class_type: type[RecorderTerm] = recorders.PostStepAbsEEFPoseAbsGripperActionsRecorder


@configclass
class PostStepAbsEEFPoseAxisAngleBinaryGripperActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the step absolute eef pose (axis-angle) + binary gripper actions recorder term."""

    class_type: type[RecorderTerm] = recorders.PostStepAbsEEFPoseAxisAngleBinaryGripperActionsRecorder


@configclass
class PostStepAbsEEFPoseAxisAngleAbsGripperActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the step absolute eef pose (axis-angle) + absolute gripper actions recorder term."""

    class_type: type[RecorderTerm] = recorders.PostStepAbsEEFPoseAxisAngleAbsGripperActionsRecorder


@configclass
class PostStepAbsEEFPoseAxisAngleAbsGripperWithForceActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the step absolute eef pose (axis-angle) + absolute gripper actions + forces recorder term."""

    class_type: type[RecorderTerm] = recorders.PostStepAbsEEFPoseAxisAngleAbsGripperWithForceActionsRecorder


##
# Recorder manager configurations for Mimic workflow.
# Actions: absolute_eef_pose: (N, 7) + binary_gripper_action: (N, 1)
##


@configclass
class AbsEEFPoseBinaryGripperActionStateRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configurations for recording actions and states."""

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_post_step_abs_eef_pose_binary_gripper_actions = PostStepAbsEEFPoseBinaryGripperActionsRecorderCfg()
    record_pre_step_flat_policy_observations = PreStepFlatPolicyObservationsRecorderCfg()


##
# Recorder manager configurations for Mimic workflow.
# Actions: absolute_eef_pose: (N, 7) + absolute_gripper_action: (N, 1)
##
@configclass
class AbsEEFPoseAbsGripperActionStateRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configurations for recording actions and states."""

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_post_step_abs_eef_pose_abs_gripper_actions = PostStepAbsEEFPoseAbsGripperActionsRecorderCfg()
    record_pre_step_flat_policy_observations = PreStepFlatPolicyObservationsRecorderCfg()


@configclass
class AbsEEFPoseAxisAngleBinaryGripperActionStateRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configurations for recording actions and states with axis-angle rotation.

    Actions: absolute_eef_pose (6D: x, y, z, ax, ay, az) + binary_gripper_action (1D)
    """

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_post_step_abs_eef_pose_axis_angle_binary_gripper_actions = (
        PostStepAbsEEFPoseAxisAngleBinaryGripperActionsRecorderCfg()
    )
    record_pre_step_flat_policy_observations = PreStepFlatPolicyObservationsRecorderCfg()


@configclass
class AbsEEFPoseAxisAngleAbsGripperActionStateRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configurations for recording actions and states with axis-angle rotation.

    Actions: absolute_eef_pose (6D: x, y, z, ax, ay, az) + absolute_gripper_action (1D)
    """

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_post_step_abs_eef_pose_axis_angle_abs_gripper_actions = (
        PostStepAbsEEFPoseAxisAngleAbsGripperActionsRecorderCfg()
    )
    record_pre_step_flat_policy_observations = PreStepFlatPolicyObservationsRecorderCfg()


@configclass
class AbsEEFPoseAxisAngleAbsGripperWithForceActionStateRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configurations for recording actions and states with axis-angle rotation and forces.

    Actions: absolute_eef_pose (6D) + absolute_gripper_action (1D) + left_force (3D) + right_force (3D) = 13D
    """

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_post_step_abs_eef_pose_axis_angle_abs_gripper_with_force_actions = (
        PostStepAbsEEFPoseAxisAngleAbsGripperWithForceActionsRecorderCfg()
    )
    record_pre_step_flat_policy_observations = PreStepFlatPolicyObservationsRecorderCfg()


##
# NOTE:
# The binary-gripper + force recorder has been removed/deprecated.
# If you need forces recorded into actions, use the 7dpf recorder
# (axis-angle + abs gripper + Force(6)).
##


@configclass
class PostStepAbsEEFPoseAxisAngleGripperCommandActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the companion axis-angle + gripper-command (7d2) actions recorder term."""

    class_type: type[RecorderTerm] = recorders.PostStepAbsEEFPoseAxisAngleGripperCommandActionsRecorder


@configclass
class AbsEEFPoseAxisAngleDualGripperActionStateRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configuration that captures BOTH gripper encodings in one replay pass.

    Datasets written per demo:
      actions      -- absolute_eef_pose (6D axis-angle) + absolute gripper position (7dp)
      actions_7d2  -- absolute_eef_pose (6D axis-angle) + raw gripper command      (7d2)
    """

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_post_step_abs_eef_pose_axis_angle_abs_gripper_actions = (
        PostStepAbsEEFPoseAxisAngleAbsGripperActionsRecorderCfg()
    )
    record_post_step_abs_eef_pose_axis_angle_gripper_command_actions = (
        PostStepAbsEEFPoseAxisAngleGripperCommandActionsRecorderCfg()
    )
    record_pre_step_flat_policy_observations = PreStepFlatPolicyObservationsRecorderCfg()
