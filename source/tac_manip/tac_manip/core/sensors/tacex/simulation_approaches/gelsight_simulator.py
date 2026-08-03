"""Base class for optical simulation approaches.

This class defines an interface for simulating the optical response of GelSight tactile sensors.
Each simulation method class should inherit from this class and implement the abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..gelsight_sensor import GelSightSensor
    from ..simulation_approaches.gelsight_simulator_cfg import GelSightSimulatorCfg


class GelSightSimulator(ABC):
    """Base class for implementing an optical simulation approach."""

    def __init__(self, sensor: GelSightSensor, cfg: GelSightSimulatorCfg):
        self.cfg = cfg
        self.sensor = sensor

        if self.cfg.device is None:
            # use same device as simulation
            self._device = self.sensor.device
        else:
            self._device = self.cfg.device

        # self._num_envs = self.sensor._num_envs

        # self._initialize_impl()

    @abstractmethod
    def _initialize_impl(self):
        raise NotImplementedError

    # @abstractmethod
    def optical_simulation(self):
        """Simulates the optical output of a Tactile sensor."""
        raise NotImplementedError

    # @abstractmethod
    def marker_motion_simulation(self):
        """Simulates the marker motion of a Tactile sensor."""
        raise NotImplementedError

    # @abstractmethod make it optional, in case another method is used for computing indentation depth
    def compute_indentation_depth(self):
        """Computes how deep the indenter is pressed into the gelpad"""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    """
    Debug visualization
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.

        This function calls the visualization objects and sets the data to visualize into them.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    """
    Internal simulation callbacks.
    """
