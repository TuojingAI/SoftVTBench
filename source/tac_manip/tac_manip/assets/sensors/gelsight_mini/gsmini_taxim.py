from tac_manip.assets import ASSETS_DATA_DIR
from tac_manip.core.sensors.tacex import GelSightSensorCfg
from tac_manip.core.sensors.tacex.simulation_approaches.gpu_taxim import (
    TaximSimulatorCfg,
)

from .gsmini_cfg import GelSightMiniCfg

"""Configuration for simulating the Gelsight Mini via GPU-Taxim."""

GELSIGHT_MINI_TAXIM_CFG = GelSightMiniCfg()
GELSIGHT_MINI_TAXIM_CFG = GELSIGHT_MINI_TAXIM_CFG.replace(
    sensor_camera_cfg=GelSightSensorCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=0,
        resolution=(320, 240),
        data_types=["depth"],
        clipping_range=(0.024, 0.029),
    ),
    update_period=0.01,
    data_types=["tactile_rgb"],
    optical_sim_cfg=TaximSimulatorCfg(
        calib_folder_path=f"{ASSETS_DATA_DIR}/Sensors/GelSight_Mini/calibs/640x480",
        gelpad_height=GELSIGHT_MINI_TAXIM_CFG.gelpad_dimensions.height,
        gelpad_to_camera_min_distance=0.024,
        with_shadow=False,
        tactile_img_res=(320, 240),
        device="cuda",
    ),
    marker_motion_sim_cfg=None,
    compute_indentation_depth_class="optical_sim",
    device="cuda",  # use gpu per default
    debug_vis=True,  # for visualizing the tactile image
)
