from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import omni.usd
import torch
import torchvision.transforms.functional as F

from ...gelsight_sensor import GelSightSensor
from ..gelsight_simulator import GelSightSimulator
from .sim import Taxim

if TYPE_CHECKING:
    from .taxim_sim_cfg import TaximSimulatorCfg


class TaximSimulator(GelSightSimulator):
    """Wraps around the Taxim simulation for the optical simulation of GelSight sensors
    inside Isaac Sim.

    """

    cfg: TaximSimulatorCfg

    def __init__(self, sensor: GelSightSensor, cfg: TaximSimulatorCfg):
        self.sensor = sensor

        super().__init__(sensor=sensor, cfg=cfg)

    def _initialize_impl(self):
        calib_folder = Path(self.cfg.calib_folder_path)

        if self.cfg.device is None:
            # use same device as simulation
            self._device = self.sensor.device
        else:
            self._device = self.cfg.device

        self._num_envs = self.sensor._num_envs
        # todo make size adaptable? I mean with env_ids. This way we would always simulate everything
        self._indentation_depth = torch.zeros((self.sensor._num_envs), device=self.sensor._device)
        """Indentation depth, i.e. how deep the object is pressed into the gelpad.
        Values are in mm.

        Indentation depth is equal to the maximum pressing depth of the object in the gelpad.
        It is used for shifting the height map for the Taxim simulation.
        """
        self.tactile_rgb_img = torch.zeros(
            (self.sensor._num_envs, self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0], 3),
            device=self._device,
        )

        # Force torch backend by default. Taxim(..., backend="auto") may pick JAX if installed,
        # but this simulator downstream expects torch tensors (torchvision resize, movedim, etc.).
        self._taxim: Taxim = Taxim(
            calib_folder=calib_folder,
            device=self._device,
            backend=getattr(self.cfg, "backend", "torch"),
        )
        if getattr(self._taxim, "backend_name", "torch") != "torch":
            raise ValueError(
                f"TaximSimulator only supports the torch backend, but got backend='{self._taxim.backend_name}'. "
                "Set TaximSimulatorCfg.backend='torch' (or 'auto' with torch installed)."
            )
        # update Taxim settings via settings from cfg class
        # print(self._taxim.width)
        # self._taxim.width = self.cfg.tactile_img_res[0]
        # self._taxim.height = self.cfg.tactile_img_res[1]

        # -- note Taxim sim uses (channels, height, width) format

        # tactile rgb image without indentation
        self.background_img = self._taxim.background_img
        #  up/downscale height map if different than tactile img res
        if self.background_img.shape != (3, self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]):
            self.background_img = F.resize(
                self.background_img, (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0])
            )
        # last dim should be channels for isaac
        self.background_img = self.background_img.movedim(0, 2)

        # use background as initial tactile_rgb_img
        self.tactile_rgb_img[:] = self.background_img

        # if camera resolution is different than the tactile RGB res, scale img
        self.img_res = self.cfg.tactile_img_res

    def optical_simulation(self):
        """Returns simulation output of Taxim optical simulation.

        Images have the shape (num_envs, height, width, channels) and values in range [0,255].
        """
        height_map = self.sensor._data.output["height_map"]

        # up/downscale height map if camera res different than tactile img res
        if (height_map.shape[1], height_map.shape[2]) != (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]):
            height_map = F.resize(height_map, (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]))

        if self._device == "cpu":
            height_map = height_map.cpu()

        # only simulate in case of indentation
        # self.tactile_rgb_img[self._indentation_depth <= 0][:] = self.background_img
        # if height_map[self._indentation_depth > 0].shape[0] > 0:
        #     self.tactile_rgb_img[self._indentation_depth > 0] = self._taxim.render_direct(
        #         height_map[self._indentation_depth > 0],
        #         with_shadow=self.cfg.with_shadow,
        #         press_depth=self._indentation_depth[self._indentation_depth > 0],
        #         orig_hm_fmt=False,
        #     ).movedim(1, 3) #*255).type(torch.uint8)

        tactile_rgb_img = self._taxim.render_direct(
            height_map[:],
            with_shadow=self.cfg.with_shadow,
            press_depth=self._indentation_depth,
            orig_hm_fmt=False,
        ).movedim(1, 3)
        self.tactile_rgb_img[:] = (tactile_rgb_img * 255.0).type(torch.uint8)

        return self.tactile_rgb_img

    def compute_indentation_depth(self):
        height_map = self.sensor._data.output["height_map"] / 1000  # convert height map from mm to meter
        min_distance_obj = height_map.amin((1, 2))
        # smallest distance between object and sensor case
        dist_obj_sensor_case = min_distance_obj - self.cfg.gelpad_to_camera_min_distance

        # print("dist_obj_sensor_case", dist_obj_sensor_case)
        # if (dist_obj_sensor_case < 0):  # object is "inside the sensor", cause the object is closer to the camera than the edge of the sensor
        #     # print("Object is inside the sensor!!! Gelpad would be broken!!!")
        #     dist_obj_sensor_case = 0
        dist_obj_sensor_case = torch.where(dist_obj_sensor_case < 0, 0, dist_obj_sensor_case)

        self._indentation_depth[:] = torch.where(
            dist_obj_sensor_case <= self.cfg.gelpad_height, (self.cfg.gelpad_height - dist_obj_sensor_case) * 1000, 0
        )

        return self._indentation_depth

    def reset(self):
        self._indentation_depth = torch.zeros((self._num_envs), device=self._device)
        self.tactile_rgb_img[:] = self.background_img

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Creates an USD attribute for the sensor asset, which can visualize the tactile image.

        Select the GelSight sensor case whose output you want to see in the Isaac Sim GUI,
        i.e. the `gelsight_mini_case` Xform (not the mesh!).
        Scroll down in the properties panel to "Raw Usd Properties" and click "Extra Properties".
        There is an attribute called "show_tactile_image".
        Toggle it on to show the sensor output in the GUI.

        If only optical simulation is used, then only an optical img is displayed.
        If only the marker simulatios is used, then only an image displaying the marker positions is displayed.
        If both, optical and marker simulation, are used, then the images are overlaid.
        """
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "_debug_windows"):
                # dict of windows that show the simulated tactile images, if the attribute of the sensor asset is turned on
                self._debug_windows = {}
                self._debug_img_providers = {}
                # todo check if we can make implementation more efficient than dict of dicts
                if "tactile_rgb" in self.sensor.cfg.data_types:
                    self._debug_windows = {}
                    self._debug_img_providers = {}
        else:
            pass

    def _debug_vis_callback(self, event):
        if self.sensor._prim_view is None:
            return

        # Update the GUI windows
        for i, prim in enumerate(self.sensor.prim_view.prims):
            if "tactile_rgb" in self.sensor.cfg.data_types:
                show_img = prim.GetAttribute("debug_tactile_rgb").Get()
                if show_img:
                    if str(i) not in self._debug_windows:
                        # create a window
                        window = omni.ui.Window(
                            self.sensor._prim_view.prim_paths[i] + "/taxim_rgb",
                            height=self.cfg.tactile_img_res[1],
                            width=self.cfg.tactile_img_res[0],
                        )
                        self._debug_windows[str(i)] = window
                        # create image provider
                        self._debug_img_providers[str(i)] = (
                            omni.ui.ByteImageProvider()
                        )  # default format omni.ui.TextureFormat.RGBA8_UNORM

                    frame = self.sensor.data.output["tactile_rgb"][i].cpu().numpy()

                    # update image of the window
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                    height, width, channels = frame.shape

                    with self._debug_windows[str(i)].frame:
                        # self._img_providers[str(i)].set_data_array(frame, [width, height, channels]) #method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        self._debug_img_providers[str(i)].set_bytes_data(
                            frame.flatten().data, [width, height]
                        )  # method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        omni.ui.ImageWithProvider(
                            self._debug_img_providers[str(i)]
                        )  # , fill_policy=omni.ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT -> fill_policy by default: specifying the width and height of the item causes the image to be scaled to that size
                elif str(i) in self._debug_windows:
                    # remove window/img_provider from dictionary and destroy them
                    self._debug_windows.pop(str(i)).destroy()
                    self._debug_img_providers.pop(str(i)).destroy()
