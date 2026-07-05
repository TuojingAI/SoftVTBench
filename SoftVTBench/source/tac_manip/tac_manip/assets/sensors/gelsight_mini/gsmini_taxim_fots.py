from tac_manip.assets import ASSETS_DATA_DIR
from tac_manip.core.sensors.tacex import GelSightSensorCfg
from tac_manip.core.sensors.tacex.simulation_approaches.fots import (
    FOTSMarkerSimulatorCfg,
)
from tac_manip.core.sensors.tacex.simulation_approaches.gpu_taxim import (
    TaximSimulatorCfg,
)

from .gsmini_cfg import GelSightMiniCfg

"""Configuration for simulating the Gelsight Mini via GPU-Taxim and FOTS."""

GELSIGHT_MINI_TAXIM_FOTS_CFG = GelSightMiniCfg()
GELSIGHT_MINI_TAXIM_FOTS_CFG = GELSIGHT_MINI_TAXIM_FOTS_CFG.replace(
    sensor_camera_cfg=GelSightSensorCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=0,
        resolution=(160, 120),  # (32, 24),
        data_types=["depth"],
        clipping_range=(0.024, 0.029),
    ),
    update_period=0.01,
    data_types=["tactile_rgb", "marker_motion", "markers_rgb"],
    optical_sim_cfg=TaximSimulatorCfg(
        calib_folder_path=f"{ASSETS_DATA_DIR}/Sensors/GelSight_Mini/calibs/640x480",
        gelpad_height=GELSIGHT_MINI_TAXIM_FOTS_CFG.gelpad_dimensions.height,
        gelpad_to_camera_min_distance=0.024,
        with_shadow=False,
        tactile_img_res=(320, 240),
        device="cuda",
    ),
    marker_motion_sim_cfg=FOTSMarkerSimulatorCfg(
        lamb=[0.00125, 0.00021, 0.00038],
        pyramid_kernel_size=[51, 21, 11, 5],  # [11, 11, 11, 11, 11, 5],
        kernel_size=5,
        marker_params=FOTSMarkerSimulatorCfg.MarkerParams(
            num_markers_col=11,
            num_markers_row=9,
            x0=15,
            y0=26,
            dx=26,
            dy=29,
        ),
        tactile_img_res=(320, 240),
        device="cuda",
        frame_transformer_cfg=None,
        backend="numpy",  # select from "tensor" or "numpy"
    ),
    compute_indentation_depth_class="optical_sim",
    device="cuda",  # use gpu per default
    debug_vis=True,  # for visualizing the tactile marker motion field
)

"""Configuration for simulating the UMI gripper with XENSE tactile sensor via GPU-Taxim and FOTS."""

UMI_XENSE_TAXIM_FOTS_CFG = GELSIGHT_MINI_TAXIM_FOTS_CFG.copy()
# real gelpad height is 2mm, a little bit larger to track the upper interface deformation
UMI_XENSE_TAXIM_FOTS_CFG.sensor_camera_cfg.clipping_range = (0.0095, 0.012)
UMI_XENSE_TAXIM_FOTS_CFG.optical_sim_cfg.gelpad_to_camera_min_distance = 0.0095
# a little bit smaller than real height
UMI_XENSE_TAXIM_FOTS_CFG.gelpad_dimensions.height = 1.95 / 1000
UMI_XENSE_TAXIM_FOTS_CFG.optical_sim_cfg.gelpad_height = 1.95 / 1000
