try:
    from isaaclab.devices.device_base import DevicesCfg  # noqa: F401
except ImportError:
    from dataclasses import field

    from isaaclab.utils import configclass

    @configclass
    class DevicesCfg:
        devices: dict = field(default_factory=dict)
