from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import omni
import torch
import torchvision.transforms.functional as F
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import euler_xyz_from_quat

from ...gelsight_sensor import GelSightSensor
from ..gelsight_simulator import GelSightSimulator
from ..gpu_taxim import TaximSimulator
from ..gpu_taxim.sim import TaximTorch
from .sim import MarkerMotion

if TYPE_CHECKING:
    from .fots_marker_sim_cfg import FOTSMarkerSimulatorCfg


class FOTSMarkerSimulator(GelSightSimulator):
    """Wraps around the FOTS simulation for marker simulation of GelSight sensors inside Isaac Sim.

    The class uses an instance of the gpu_taxim simulator for generating the deformed height map.
    """

    cfg: FOTSMarkerSimulatorCfg

    def __init__(self, sensor: GelSightSensor, cfg: FOTSMarkerSimulatorCfg):
        self.sensor: GelSightSensor = sensor

        super().__init__(sensor=sensor, cfg=cfg)

        # resolve regex which should be dealt with in interactive scene
        self.transformer_cfg = self.cfg.frame_transformer_cfg
        if hasattr(self.transformer_cfg, "prim_path"):
            self.transformer_cfg.prim_path = self.transformer_cfg.prim_path.format(ENV_REGEX_NS="/World/envs/env_.*")
        if hasattr(self.transformer_cfg, "target_frames"):
            updated_target_frames = []
            for target_frame in self.transformer_cfg.target_frames:
                target_frame.prim_path = target_frame.prim_path.format(ENV_REGEX_NS="/World/envs/env_.*")
                updated_target_frames.append(target_frame)
            self.transformer_cfg.target_frames = updated_target_frames

        # use IsaacLab FrameTransformer for keeping track of relative position/rotation
        self.frame_transformer: FrameTransformer = FrameTransformer(self.transformer_cfg)

    def _initialize_impl(self):
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

        # use Taxim for gpu based operations
        if (self.sensor.optical_simulator is not None) and (type(self.sensor.optical_simulator) is TaximSimulator):
            self._taxim: TaximTorch = self.sensor.optical_simulator._taxim
        else:
            raise RuntimeError(
                "Currently FOTS simulation approach has to be used in combination with GPU-Taxim as optical-simulator."
            )

        # tactile rgb image without indentation
        bg_img = self._taxim.background_img.movedim(0, 2).cpu().numpy()
        self.marker_motion_sim = MarkerMotion(
            frame0_blur=bg_img,
            mm2pix=self.cfg.mm_to_pixel,
            num_markers_col=self.cfg.marker_params.num_markers_col,  # 20, #11
            num_markers_row=self.cfg.marker_params.num_markers_row,  # 15, #9
            tactile_img_width=self.cfg.tactile_img_res[0],  # default 320 (W)
            tactile_img_height=self.cfg.tactile_img_res[1],  # default 240 (H)
            lamb=(self.cfg.lamb if self.cfg.lamb else [0.00125, 0.00021, 0.00038]),
            x0=self.cfg.marker_params.x0,
            y0=self.cfg.marker_params.y0,
        )

        self.init_marker_pos = np.stack(
            (self.marker_motion_sim.init_marker_x_pos, self.marker_motion_sim.init_marker_y_pos), axis=-1
        )
        self.init_marker_pos = self.init_marker_pos.reshape(-1, 2)
        # if camera resolution is different than the tactile RGB res, scale img
        self.img_res = self.cfg.tactile_img_res

        # create buffers
        self.marker_data = torch.zeros(
            (self.sensor._num_envs, 2, self.cfg.marker_params.num_markers, 2), device=self._device
        )
        """Marker flow data. Shape is [num_envs, 2, num_markers, 2]

        dim=1: [initial, current] marker positions
        dim=3: [x,y] values of the markers in the image of the sensor.
        """
        # set initial marker pos
        self.marker_data[:, 0] = torch.tensor(self.init_marker_pos, device=self._device)

        self.sensor._data.output["traj"] = []
        for _ in range(self.sensor._num_envs):
            self.sensor._data.output["traj"].append([])
        self.theta = torch.zeros((self.sensor._num_envs), device=self._device)

        # need to initialize manually
        self.frame_transformer._initialize_impl()
        self.frame_transformer._is_initialized = True
        print("Frame transformer for FOTS: ", self.frame_transformer)

        # for tactile_targets selection
        self._active_target_idx = torch.zeros((self._num_envs,), dtype=torch.long, device=self._device)
        self._stable_steps = torch.zeros((self._num_envs,), dtype=torch.long, device=self._device)

    def marker_motion_simulation(self):
        self._indentation_depth = self.sensor._indentation_depth
        height_map = self.sensor._data.output[
            "height_map"
        ]  # height map has shape (height, width) cause row-column format

        # up/downscale height map if camera res different than tactile img res
        if (height_map.shape[1], height_map.shape[2]) != (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]):
            height_map = F.resize(height_map, (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]))

        if self._device == "cpu":
            height_map = height_map.cpu()
            self._indentation_depth = self.sensor._indentation_depth.cpu()

        height_map_shifted = self._taxim._TaximTorch__get_shifted_height_map(self._indentation_depth, height_map)
        deformed_gel, contact_mask = self._taxim._TaximTorch__compute_gel_pad_deformation(height_map_shifted)
        deformed_gel = deformed_gel.max() - deformed_gel

        for env_id in range(deformed_gel.shape[0]):
            # fix the bug of frame transformer not updating the data
            self.frame_transformer.update(dt=0.001)

            # for tactile_targets selection
            pos_bt = self.frame_transformer.data.target_pos_source  # [B(num_envs), T(num_targets), 3]
            if self.cfg.target_select_mode == "auto_distance":
                if self.cfg.distance_metric == "abs_z":
                    dist_bt = pos_bt[..., 2].abs()  # [B, T]
                else:
                    dist_bt = torch.linalg.norm(pos_bt, dim=-1)  # [B, T]
                idx_b = torch.argmin(dist_bt, dim=-1)  # [B]
                # hysteresis process: only switch to new target if it is much closer, or stable for N steps
                cur = self._active_target_idx
                better = (
                    dist_bt[torch.arange(dist_bt.size(0), device=self._device), idx_b] + self.cfg.switch_margin
                    < dist_bt[torch.arange(dist_bt.size(0), device=self._device), cur]
                )
                stable = self._stable_steps >= self.cfg.switch_hysteresis_steps
                to_switch = better | stable
                self._active_target_idx = torch.where(to_switch, idx_b, cur)
                self._stable_steps = torch.where(
                    idx_b == cur, self._stable_steps + 1, torch.zeros_like(self._stable_steps)
                )
            else:
                # manual mode: _active_target_idx is set by external API
                pass

            # use the selected target_idx to get the relative orientation
            B = pos_bt.shape[0]
            rel_orient = self.frame_transformer.data.target_quat_source[
                torch.arange(B, device=self._device), self._active_target_idx
            ]  # [B, 4]
            roll, pitch, yaw = euler_xyz_from_quat(rel_orient)
            yaw_b = yaw[:, 0] if yaw.ndim >= 2 else yaw

            if self._indentation_depth[env_id].item() > 0.0:
                # compute contact center based on contact_mask
                contact_points = torch.argwhere(contact_mask[env_id])
                mean = torch.mean(contact_points.float(), dim=0).cpu().numpy()
                # print("should be pix ", mean[1], mean[0])
                # rows = height = y values
                mean[0] = (mean[0] - self.marker_motion_sim.tactile_img_height / 2) / self.marker_motion_sim.mm2pix
                # columns = width = x values
                mean[1] = (mean[1] - self.marker_motion_sim.tactile_img_width / 2) / self.marker_motion_sim.mm2pix

                theta = yaw_b[env_id].cpu().numpy()

                # traj takes [x,y,theta] values
                self.sensor._data.output["traj"][env_id].append([mean[1], mean[0], theta])

                # todo vectorize with pytorch
                marker_x_pos, marker_y_pos = self.marker_motion_sim.marker_sim(
                    deformed_gel[env_id].cpu().numpy(),
                    contact_mask[env_id].cpu().numpy(),
                    self.sensor._data.output["traj"][env_id],
                )
            else:
                self.sensor._data.output["traj"][env_id] = []
                marker_x_pos = self.marker_motion_sim.init_marker_x_pos
                marker_y_pos = self.marker_motion_sim.init_marker_y_pos

            marker_pos = np.stack((marker_x_pos, marker_y_pos), axis=-1).reshape(-1, 2)
            self.marker_data[env_id, 1] = torch.tensor(marker_pos, device=self._device)

        return self.marker_data

    def marker_motion_rendering(self):
        if self.sensor._prim_view is None:
            return None

        # Initialize tensor for rendered frames for all environments
        # Assuming tactile image resolution from config
        img_height, img_width = self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]
        rendered_frames = torch.zeros(
            (self._num_envs, img_height, img_width, 3), device=self._device, dtype=torch.uint8
        )

        # Process all environments like in marker_motion_simulation
        for env_id in range(self._num_envs):
            if "marker_motion" in self.sensor.cfg.data_types:
                marker_flow_i = self.sensor._data.output["marker_motion"][env_id]
                frame = self._create_marker_img(marker_flow_i)

                # Compose marker-flow mask over tactile_rgb using tensor ops (avoid CPU/Numpy conversions)
                if "tactile_rgb" in self.sensor.cfg.data_types:
                    if (
                        self.sensor.cfg.optical_sim_cfg.tactile_img_res
                        == self.sensor.cfg.marker_motion_sim_cfg.tactile_img_res
                    ):
                        tactile_rgb = self.sensor._data.output["tactile_rgb"][env_id]
                        frame_tensor = torch.from_numpy(frame.astype(np.float64)).to(self._device) / 255.0
                        frame_expanded = frame_tensor.unsqueeze(-1).expand(-1, -1, 3)
                        combined_frame = (tactile_rgb * frame_expanded).clamp(0, 255).to(torch.uint8)
                        rendered_frames[env_id] = combined_frame
                    else:
                        # Different resolutions: up/down sampling required (not implemented here)
                        pass
                else:
                    frame_tensor = torch.from_numpy(frame).to(self._device)
                    if frame_tensor.dim() == 2:
                        frame_tensor = frame_tensor.unsqueeze(-1).expand(-1, -1, 3)
                    rendered_frames[env_id] = frame_tensor.to(torch.uint8)

        return rendered_frames

    def reset(self):
        self._indentation_depth = torch.zeros((self._num_envs), device=self._device)
        self.init_marker_pos = (self.marker_motion_sim.init_marker_x_pos, self.marker_motion_sim.init_marker_y_pos)

        # clean trajectory data
        for env_id in range(self.sensor._num_envs):
            self.sensor._data.output["traj"][env_id] = []

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Creates an USD attribute for the sensor asset, which can visualize the tactile image.

        Select the GelSight sensor case whose output you want to see in the Isaac Sim GUI,
        i.e. the `gelsight_mini_case` Xform (not the mesh!).
        Scroll down in the properties panel to "Raw Usd Properties" and click "Extra Properties".marker_motion_rendering
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
                if "marker_motion" in self.sensor.cfg.data_types:
                    self._debug_windows = {}
                    self._debug_img_providers = {}
        else:
            pass

    def _debug_vis_callback(self, event):
        if self.sensor._prim_view is None:
            return

        # Update the GUI windows_prim_view
        for i, prim in enumerate(self.sensor._prim_view.prims):
            if "marker_motion" in self.sensor.cfg.data_types:
                show_img = prim.GetAttribute("debug_marker_motion").Get()
                if show_img:
                    if str(i) not in self._debug_windows:
                        # create a window
                        window = omni.ui.Window(
                            self.sensor._prim_view.prim_paths[i] + "/fots_marker",
                            height=self.cfg.tactile_img_res[1],
                            width=self.cfg.tactile_img_res[0],
                        )
                        self._debug_windows[str(i)] = window
                        # create image provider
                        self._debug_img_providers[str(i)] = (
                            omni.ui.ByteImageProvider()
                        )  # default format omni.ui.TextureFormat.RGBA8_UNORM

                    frame = self.sensor._data.output["markers_rgb"][i].cpu().numpy()

                    # update image of the window

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

                    height, width, channels = frame.shape

                    with self._debug_windows[str(i)].frame:
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

    def _create_marker_img(self, marker_data):
        """Visualization of marker flow, like in the original FOTS simulation.

        Marker data needs to have the shape [2, num_markers, 2]
        - dim=0: init and current markers
        - dim=2: x and y values of the marker position

        Args:
            marker_data: marker flow data with shape [2, num_markers, 2]
        """
        # for visualization -> white background with black dots
        color = (0, 0, 0)
        arrow_scale = 1

        frame = np.ones((self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0])).astype(np.uint8)

        # marker data has shape [2, num_markers, 2], where first dim = init and current marker position
        init_marker_pos = marker_data[0].cpu().numpy()
        current_marker_pos = marker_data[1].cpu().numpy()

        num_markers = marker_data.shape[1]

        for marker_index in range(num_markers):
            init_x_pos = int(init_marker_pos[marker_index][0])
            init_y_pos = int(init_marker_pos[marker_index][1])

            x_pos = int(current_marker_pos[marker_index][0])
            y_pos = int(current_marker_pos[marker_index][1])

            if (x_pos >= frame.shape[1]) or (x_pos < 0) or (y_pos >= frame.shape[0]) or (y_pos < 0):
                continue
            # cv2.circle(frame,(init_y_pos,init_x_pos), 6, (255,255,255), 1, lineType=8)

            pt1 = (init_x_pos, init_y_pos)
            pt2 = (x_pos + arrow_scale * int(x_pos - init_x_pos), y_pos + arrow_scale * int(y_pos - init_y_pos))
            cv2.arrowedLine(frame, pt1, pt2, color, 2, tipLength=0.2)

        frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # (240, 320)

        return frame

    def set_active_target_idx(self, env_ids, idxs):
        # switch to manual mode and set the active target_idx
        self.cfg.target_select_mode = "manual"
        if isinstance(env_ids, int):
            env_ids = [env_ids]
            idxs = [idxs]
        for e, i in zip(env_ids, idxs):
            self._active_target_idx[e] = int(i)
            self._stable_steps[e] = 0
