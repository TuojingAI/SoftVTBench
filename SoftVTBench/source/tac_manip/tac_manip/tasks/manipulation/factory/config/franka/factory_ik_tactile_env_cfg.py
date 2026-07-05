from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from tac_manip.tasks.manipulation.factory import mdp
from tac_manip.tasks.manipulation.factory.config.franka.factory_ik_env_cfg import (
    FrankaFactoryEnvCfg,
)

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from tac_manip.assets import (  # isort: skip
    FRANKA_PANDA_HIGH_PD_WITH_GSMINI_CFG,
    GELSIGHT_MINI_TAXIM_FOTS_CFG,
)


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
class FrankaFactoryTactileEnvCfg(FrankaFactoryEnvCfg):
    def __post_init__(
        self,
    ):
        # post init of parent
        super().__post_init__()
        # add marker motion fields to observations
        self.observations.policy = MarkerMotionPolicyCfg()

        # replace robot with franka with gelsight mini sensor
        self.scene.robot = FRANKA_PANDA_HIGH_PD_WITH_GSMINI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # add gelsight mini sensor with rgb and marker motion field output
        self.tactile_marker_0_cfg = FRAME_MARKER_CFG.copy()
        self.tactile_marker_0_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        self.tactile_marker_0_cfg.prim_path = "/Visuals/FrameTransformer_gelpad_left"
        self.tactile_marker_1_cfg = self.tactile_marker_0_cfg.copy()
        self.tactile_marker_1_cfg.prim_path = "/Visuals/FrameTransformer_gelpad_right"

        if self.task_name == "peg_insert":
            contact_prim_path = "{ENV_REGEX_NS}/HeldAsset/forge_round_peg_8mm"
        elif self.task_name == "gear_mesh":
            contact_prim_path = "{ENV_REGEX_NS}/HeldAsset/factory_gear_medium"
        elif self.task_name == "nut_thread":
            contact_prim_path = "{ENV_REGEX_NS}/HeldAsset/factory_nut_loose"
        else:
            raise RuntimeError(f'We only support factory task in ["peg_insert", "gear_mesh", "nut_thread"]')

        self.scene.gsmini_left = GELSIGHT_MINI_TAXIM_FOTS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/gelsight_mini_case_left",  # prim to attach camera sensor
            debug_vis=True,  # visualizer for tactile sensor output
        )
        self.scene.gsmini_left.marker_motion_sim_cfg.frame_transformer_cfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gelpad_left",  # prim to gelpad, to track the transformation of indenter to gelpad
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=contact_prim_path,
                    name="held_asset",
                ),
            ],
            debug_vis=False,  # visualizer for frame transformer
            visualizer_cfg=self.tactile_marker_0_cfg,
        )

        self.scene.gsmini_right = GELSIGHT_MINI_TAXIM_FOTS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot/gelsight_mini_case_right",  # prim to attach camera sensor
            debug_vis=True,  # visualizer for tactile sensor output
        )
        self.scene.gsmini_right.marker_motion_sim_cfg.frame_transformer_cfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gelpad_right",  # prim to gelpad, to track the transformation of indenter to gelpad
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=contact_prim_path,
                    name="held_asset",
                ),
            ],
            debug_vis=False,  # visualizer for frame transformer
            visualizer_cfg=self.tactile_marker_1_cfg,
        )


@configclass
class FrankaFactoryPegInsertTactileEnvCfg(FrankaFactoryTactileEnvCfg):
    task_name = "peg_insert"


@configclass
class FrankaFactoryGearMeshTactileEnvCfg(FrankaFactoryTactileEnvCfg):
    task_name = "gear_mesh"


@configclass
class FrankaFactoryNutThreadTactileEnvCfg(FrankaFactoryTactileEnvCfg):
    task_name = "nut_thread"
