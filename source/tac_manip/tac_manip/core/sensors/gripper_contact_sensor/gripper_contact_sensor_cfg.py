# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors.contact_sensor import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .gripper_contact_sensor import GripperContactSensor

##
# Contact force arrow markers configuration
##

NET_CONTACT_FORCE_ARROW_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "left_finger_force": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(1.0, 0.5, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
        ),
        "right_finger_force": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(1.0, 0.5, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
        ),
    }
)
"""Configuration for contact force arrow markers."""


TRIAXIAL_CONTACT_FORCE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "x_force": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(1.0, 0.5, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "y_force": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(1.0, 0.5, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        "z_force": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(1.0, 0.5, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    }
)
"""Configuration for triaxial contact force component markers."""


@configclass
class GripperContactSensorCfg(ContactSensorCfg):
    """Configuration for the contact sensor."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.class_type = GripperContactSensor

    @configclass
    class OffsetCfg:
        """The offset pose configuration for contact sensor coordinate frame transformation."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""

        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    visualize_net_force_arrows: bool = False
    """Whether to visualize net contact forces as arrows instead of spheres. Defaults to False."""

    visualize_triaxial_forces: bool = False
    """Whether to visualize triaxial contact force components (X, Y, Z) as separate arrows. Defaults to False."""

    max_force: float = 50.0
    """Maximum force threshold for normalization. Forces above this value will be clipped for visualization. The arrow length will be normalized to this maximum force."""

    max_force_arrow_length: float = 1.0
    """Maximum force arrows length when visualize_force_arrows is True. The arrow length will be normalized_magnitude * max_force_arrow_length."""

    left_finger_offset: OffsetCfg = OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0))
    """The offset configuration for the left finger coordinate frame transformation.
    """

    right_finger_offset: OffsetCfg = OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0))
    """The offset configuration for the right finger coordinate frame transformation.
    """

    net_force_visualizer_cfg: VisualizationMarkersCfg = NET_CONTACT_FORCE_ARROW_MARKER_CFG.replace(
        prim_path="/Visuals/ContactForceSensor"
    )
    """The configuration object for force arrow visualization markers. Defaults to CONTACT_FORCE_ARROW_MARKER_CFG.

    .. note::
        This attribute is only used when debug visualization is enabled and visualize_force_arrows is True.
    """

    triaxial_force_visualizer_cfg: VisualizationMarkersCfg = TRIAXIAL_CONTACT_FORCE_MARKER_CFG.replace(
        prim_path="/Visuals/TriaxialContactForceSensor"
    )
    """The configuration object for triaxial force component visualization markers.

    .. note::
        This attribute is only used when debug visualization is enabled and visualize_triaxial_forces is True.
    """
    vis_force_threshold: float = 1e-6
    """ Minimum force threshold (in Newtons) to determine whether contact exists.
    Contact forces below this value are considered zero.
    This threshold affects visualization visibility."""

    vis_offset_distance: float = 0.1
    """Offset distance (in meters) for visualized arrows/markers along the local -Z direction from the contact point.
    Increasing this value helps avoid overlap between arrows and geometry, but moves the arrow base further from the actual contact point."""

    arrow_thickness: float = 0.02
    """Thickness of the visualized arrows (in meters), corresponding to scaling in the Y/Z directions.
    Only affects rendering appearance, does not change the force value or normalized length."""
