from dataclasses import MISSING
from typing import Literal

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from ..gelsight_simulator_cfg import GelSightSimulatorCfg
from .fots_marker_sim import FOTSMarkerSimulator as FOTSMarkerNumpySimulator
from .fots_marker_sim_tensor import FOTSMarkerSimulator as FOTSMarkerTensorSimulator

"""Configuration for a tactile Marker Motion simulation with FOTS."""


@configclass
class FOTSMarkerSimulatorCfg(GelSightSimulatorCfg):
    # Default to tensor version
    simulation_approach_class: type = FOTSMarkerNumpySimulator

    # Add: backend selection
    backend: Literal["tensor", "numpy"] = "numpy"

    calib_folder_path: str = ""
    device: str = None

    with_shadow: bool = False

    tactile_img_res: tuple = (240, 320)
    """Resolution of the Tactile Image.

    Can be different from the Sensor Camera.
    If this is the case, then height map from camera is going to be up/down sampled.
    """

    lamb: list[float] = []
    """Parameters for exponential functions used by FOTS for marker simulation"""

    # experimental params
    ball_radius = 4.70 / 2  # mm
    mm_to_pixel = 19.58  # units = pix/mm

    # optical simulation params
    pyramid_kernel_size: list[int] = []
    kernel_size: int = 0

    @configclass
    class MarkerParams:
        """Dimensions here are in mm (we assume that the world units are meters)"""

        num_markers_col: int = 11
        num_markers_row: int = 9
        num_markers: int = 99
        x0: float = 15.0
        y0: float = 26.0
        dx: float = 26.0
        dy: float = 29.0

    marker_params: MarkerParams = MarkerParams()

    init_marker_pos: tuple = ([[]], [[]])
    """Initial Marker positions.

    Tuple (xx_init pos, yy_init pos):
    - xx_init = initial position of each marker along the "height" of the tactile img (top-down)
        -> for each marker the initial x pos. Shape: (num_markers_row, num_marker_column)
    - yy_init = initial position of each marker along the "width" of the tactile img (left-right)
        -> for each marker the initial y pos. Shape: (num_markers_row, num_marker_column)
    """

    frame_transformer_cfg: FrameTransformerCfg = MISSING

    """
    Sensor may have multiple in-contact targets, and we need to select the closest one for marker motion simulation.
    Below configs are for frame transformer target selection (in marker_motion_simulation)
    """
    target_select_mode: Literal["auto_distance", "manual"] = "auto_distance"
    distance_metric: Literal["3d", "abs_z"] = "3d"  # 3D Euclidean distance or only |z|
    switch_margin: float = 0.005  # m, switch margin, to prevent jittering
    switch_hysteresis_steps: int = 3  # new target needs to be stable for N steps before switching

    # backend selection
    def __post_init__(self):
        if self.backend == "tensor":
            self.simulation_approach_class = FOTSMarkerTensorSimulator
        elif self.backend == "numpy":
            self.simulation_approach_class = FOTSMarkerNumpySimulator
        else:
            raise ValueError(f"Unknown backend {self.backend} for FOTSMarkerSimulatorCfg.")
