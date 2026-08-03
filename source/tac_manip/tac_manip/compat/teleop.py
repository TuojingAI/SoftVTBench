from dataclasses import field
from typing import Any

from isaaclab.devices.openxr.openxr_device import OpenXRDevice
from isaaclab.utils import configclass

try:
    from isaaclab.devices.keyboard import Se3KeyboardCfg  # noqa: F401
except ImportError:
    @configclass
    class Se3KeyboardCfg:
        pos_sensitivity: float = 0.4
        rot_sensitivity: float = 0.8
        sim_device: str | None = None

try:
    from isaaclab.devices.spacemouse import Se3SpaceMouseCfg  # noqa: F401
except ImportError:
    @configclass
    class Se3SpaceMouseCfg:
        pos_sensitivity: float = 0.4
        rot_sensitivity: float = 0.8
        sim_device: str | None = None

try:
    from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg  # noqa: F401
except ImportError:
    @configclass
    class OpenXRDeviceCfg:
        retargeters: list[Any] = field(default_factory=list)
        sim_device: str | None = None
        xr_cfg: Any | None = None

try:
    from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg  # noqa: F401
except ImportError:
    @configclass
    class GripperRetargeterCfg:
        bound_hand: Any = None
        sim_device: str | None = None

try:
    from isaaclab.devices.openxr.retargeters.manipulator.se3_abs_retargeter import Se3AbsRetargeterCfg  # noqa: F401
except ImportError:
    @configclass
    class Se3AbsRetargeterCfg:
        bound_hand: Any = None
        zero_out_xy_rotation: bool = False
        use_wrist_rotation: bool = False
        use_wrist_position: bool = False
        sim_device: str | None = None

try:
    from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import Se3RelRetargeterCfg  # noqa: F401
except ImportError:
    @configclass
    class Se3RelRetargeterCfg:
        bound_hand: Any = None
        zero_out_xy_rotation: bool = False
        use_wrist_rotation: bool = False
        use_wrist_position: bool = True
        delta_pos_scale_factor: float = 10.0
        delta_rot_scale_factor: float = 10.0
        alpha_pos: float = 0.5
        alpha_rot: float = 0.5
        enable_visualization: bool = False
        sim_device: str | None = None
