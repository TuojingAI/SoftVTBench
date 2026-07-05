# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from tac_manip.tasks.manipulation.articulated import mdp

from .ik_rel_env_cfg import UR10EOpenDrawerEnvCfg, UR10EPutIntoAndCloseDrawerEnvCfg

##
# Pre-defined configs
##

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from tac_manip.assets import (  # isort: skip
    UMI_XENSE_TAXIM_FOTS_CFG,
)


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


##
# New Tasks with tactile sensor
##


@configclass
class UR10EOpenDrawerTactileEnvCfg(UR10EOpenDrawerEnvCfg):
    """
    Add gelsight mini sensor with rgb and marker motion field output
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # add marker motion fields to observations
        self.observations.policy = MarkerMotionPolicyCfg()

        # add gelsight mini sensor with rgb and marker motion field output
        self.tactile_marker_0_cfg = FRAME_MARKER_CFG.copy()
        self.tactile_marker_0_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        self.tactile_marker_0_cfg.prim_path = "/Visuals/FrameTransformer_gelpad_left"
        self.tactile_marker_1_cfg = self.tactile_marker_0_cfg.copy()
        self.tactile_marker_1_cfg.prim_path = "/Visuals/FrameTransformer_gelpad_right"

        self.scene.gsmini_left = UMI_XENSE_TAXIM_FOTS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/camera1_base",  # prim to attach camera sensor
            debug_vis=True,  # visualizer for tactile sensor output
        )
        self.scene.gsmini_left.marker_motion_sim_cfg.frame_transformer_cfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/pad1_base",  # prim to gelpad, to track the transformation of indenter to gelpad
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
                    name="drawer_handle_top",
                    offset=OffsetCfg(
                        pos=(0.305, 0.0, 0.01),
                        rot=(0.5, 0.5, -0.5, -0.5),  # align with end-effector frame
                    ),
                ),
            ],
            debug_vis=False,  # visualizer for frame transformer
            visualizer_cfg=self.tactile_marker_0_cfg,
        )

        self.scene.gsmini_right = UMI_XENSE_TAXIM_FOTS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/camera2_base",  # prim to attach camera sensor
            debug_vis=True,  # visualizer for tactile sensor output
        )
        self.scene.gsmini_right.marker_motion_sim_cfg.frame_transformer_cfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/pad2_base",  # prim to gelpad, to track the transformation of indenter to gelpad
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
                    name="drawer_handle_top",
                    offset=OffsetCfg(
                        pos=(0.305, 0.0, 0.01),
                        rot=(0.5, 0.5, -0.5, -0.5),  # align with end-effector frame
                    ),
                ),
            ],
            debug_vis=False,  # visualizer for frame transformer
            visualizer_cfg=self.tactile_marker_1_cfg,
        )


@configclass
class UR10EPutIntoAndCloseDrawerTactileEnvCfg(UR10EPutIntoAndCloseDrawerEnvCfg):
    """
    Add gelsight mini sensor with rgb and marker motion field output
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # add marker motion fields to observations
        self.observations.policy = MarkerMotionPolicyCfg()

        # add gelsight mini sensor with rgb and marker motion field output
        self.tactile_marker_0_cfg = FRAME_MARKER_CFG.copy()
        self.tactile_marker_0_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        self.tactile_marker_0_cfg.prim_path = "/Visuals/FrameTransformer_gelpad_left"
        self.tactile_marker_1_cfg = self.tactile_marker_0_cfg.copy()
        self.tactile_marker_1_cfg.prim_path = "/Visuals/FrameTransformer_gelpad_right"

        self.scene.gsmini_left = UMI_XENSE_TAXIM_FOTS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/camera1_base",  # prim to attach camera sensor
            debug_vis=True,  # visualizer for tactile sensor output
        )
        self.scene.gsmini_left.marker_motion_sim_cfg.frame_transformer_cfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/pad1_base",  # prim to gelpad, to track the transformation of indenter to gelpad
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Object/Apple_3", name="apple"),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Cabinet/cabinet/drawer_handle_bottom",
                    name="drawer_handle_bottom",
                    offset=OffsetCfg(
                        pos=(0.305, 0.0, 0.01),
                        rot=(0.5, 0.5, -0.5, -0.5),  # align with end-effector frame
                    ),
                ),
            ],
            debug_vis=False,  # visualizer for frame transformer
            visualizer_cfg=self.tactile_marker_0_cfg,
        )

        self.scene.gsmini_right = UMI_XENSE_TAXIM_FOTS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/camera2_base",  # prim to attach camera sensor
            debug_vis=True,  # visualizer for tactile sensor output
        )
        self.scene.gsmini_right.marker_motion_sim_cfg.frame_transformer_cfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/pad2_base",  # prim to gelpad, to track the transformation of indenter to gelpad
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Object/Apple_3", name="apple"),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Cabinet/cabinet/drawer_handle_bottom",
                    name="drawer_handle_bottom",
                    offset=OffsetCfg(
                        pos=(0.305, 0.0, 0.01),
                        rot=(0.5, 0.5, -0.5, -0.5),  # align with end-effector frame
                    ),
                ),
            ],
            debug_vis=False,  # visualizer for frame transformer
            visualizer_cfg=self.tactile_marker_1_cfg,
        )
