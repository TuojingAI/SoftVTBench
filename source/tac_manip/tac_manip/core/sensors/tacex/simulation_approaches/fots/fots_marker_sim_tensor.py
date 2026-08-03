from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import omni
import torch
import torchvision.transforms.functional as F
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import euler_xyz_from_quat

from ...gelsight_sensor import GelSightSensor
from ..gelsight_simulator import GelSightSimulator
from ..gpu_taxim import TaximSimulator
from ..gpu_taxim.sim import TaximTorch
from .sim import MarkerMotionTensor as MarkerMotion

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
        bg_img = self._taxim.background_img.movedim(0, 2)
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
        # Trajectory buffers for batched processing
        self._traj_first = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self._traj_last = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self._traj_has_first = torch.zeros((self._num_envs,), device=self._device, dtype=torch.bool)

        self.init_marker_pos = torch.stack(
            (
                self.marker_motion_sim.init_marker_x_pos,
                self.marker_motion_sim.init_marker_y_pos,
            ),
            dim=-1,
        ).reshape(-1, 2)

        # Pre-cache pixel grid once (H, W)
        W = self.cfg.tactile_img_res[0]
        H = self.cfg.tactile_img_res[1]
        x = torch.arange(0, W, device=self._device, dtype=torch.float32)
        y = torch.arange(0, H, device=self._device, dtype=torch.float32)
        self._grid_x, self._grid_y = torch.meshgrid(x, y, indexing="xy")

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
        self.marker_data[:, 0] = self.init_marker_pos

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
        height_map = self.sensor._data.output["height_map"]  # [B, H, W]

        if (height_map.shape[1], height_map.shape[2]) != (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]):
            height_map = F.resize(height_map, (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]))

        if self._device == "cpu":
            height_map = height_map.cpu()
            self._indentation_depth = self.sensor._indentation_depth.cpu()

        height_map_shifted = self._taxim._TaximTorch__get_shifted_height_map(self._indentation_depth, height_map)
        deformed_gel, contact_mask = self._taxim._TaximTorch__compute_gel_pad_deformation(height_map_shifted)
        deformed_gel = deformed_gel.max() - deformed_gel  # [B, H, W]

        B, H, W = deformed_gel.shape
        # Update frame transformer (batched)
        self.frame_transformer.update(dt=0.001)

        # Tactile Targets Selection: auto or manual, and with hysteresis
        pos_bt = self.frame_transformer.data.target_pos_source  # [B(num_envs), T(num_targets), 3]
        if self.cfg.target_select_mode == "auto_distance":
            if self.cfg.distance_metric == "abs_z":
                dist_bt = pos_bt[..., 2].abs()  # [B, T]
            else:
                dist_bt = torch.linalg.norm(pos_bt, dim=-1)  # [B, T]
            idx_b = torch.argmin(dist_bt, dim=-1)  # [B]
            # hysteresis: new target needs to be much closer, or stable for N steps before switching
            cur = self._active_target_idx
            better = (
                dist_bt[torch.arange(dist_bt.size(0), device=self._device), idx_b] + self.cfg.switch_margin
                < dist_bt[torch.arange(dist_bt.size(0), device=self._device), cur]
            )
            stable = self._stable_steps >= self.cfg.switch_hysteresis_steps
            to_switch = better | stable
            self._active_target_idx = torch.where(to_switch, idx_b, cur)
            self._stable_steps = torch.where(idx_b == cur, self._stable_steps + 1, torch.zeros_like(self._stable_steps))
        else:
            # manual mode: _active_target_idx is set by external API
            pass

        # use the selected target_idx to get the relative orientation
        B = pos_bt.shape[0]
        rel_orient = self.frame_transformer.data.target_quat_source[
            torch.arange(B, device=self._device), self._active_target_idx
        ]  # [B, 4]
        roll, pitch, yaw = euler_xyz_from_quat(rel_orient)
        yaw_b = yaw[:, 0] if yaw.ndim >= 2 else yaw  # [B]

        # Compute contact center in pixels
        m = contact_mask.to(torch.float32)
        m_sum = m.reshape(B, -1).sum(dim=1)  # [B]
        # Active when indentation > 0 and contact exists
        active = (self._indentation_depth > 0.0) & (m_sum > 0)

        mean_x_pix = (m * self._grid_x).reshape(B, -1).sum(dim=1) / torch.clamp_min(m_sum, 1e-6)
        mean_y_pix = (m * self._grid_y).reshape(B, -1).sum(dim=1) / torch.clamp_min(m_sum, 1e-6)

        # Convert center to mm based on mm2pix (cfg stores (W, H); tensors/images are indexed as (H, W))
        mm2pix = float(self.marker_motion_sim.mm2pix)
        center_x_mm = (mean_x_pix - (W / 2.0)) / mm2pix
        center_y_mm = (mean_y_pix - (H / 2.0)) / mm2pix
        traj_curr = torch.stack([center_x_mm, center_y_mm, yaw_b.to(torch.float32)], dim=1)  # [B, 3]

        # Maintain per-env first/last trajectory samples
        new_first_mask = active & (~self._traj_has_first)
        self._traj_first[new_first_mask] = traj_curr[new_first_mask]
        self._traj_last[active] = traj_curr[active]

        # Reset when inactive
        inactive = (~active) & self._traj_has_first
        self._traj_has_first[new_first_mask] = True
        self._traj_has_first[inactive] = False
        self._traj_first[inactive] = 0.0
        self._traj_last[inactive] = 0.0

        # Batch update marker positions
        marker_x_pos, marker_y_pos = self.marker_motion_sim.marker_sim_batched(
            deformed_gel.to(self._device).to(torch.float32),
            contact_mask.to(self._device),
            self._traj_first,
            self._traj_last,
            self._traj_has_first,
        )  # [B, R, C]

        marker_pos = torch.stack([marker_x_pos, marker_y_pos], dim=-1).reshape(B, -1, 2)  # [B, M, 2]
        self.marker_data[:, 1] = marker_pos.to(self._device).to(torch.float32)

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

                # Compose marker-flow mask over tactile_rgb using pure tensor ops
                if "tactile_rgb" in self.sensor.cfg.data_types:
                    if (
                        self.sensor.cfg.optical_sim_cfg.tactile_img_res
                        == self.sensor.cfg.marker_motion_sim_cfg.tactile_img_res
                    ):
                        tactile_rgb = self.sensor._data.output["tactile_rgb"][env_id]
                        frame_tensor = frame / 255.0
                        frame_expanded = frame_tensor.unsqueeze(-1).expand(-1, -1, 3)
                        combined_frame = (tactile_rgb * frame_expanded).clamp(0, 255).to(torch.uint8)
                        rendered_frames[env_id] = combined_frame
                    else:
                        # Different resolutions: up/down sampling would be required here
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
        self._traj_first.zero_()
        self._traj_last.zero_()
        self._traj_has_first[:] = False

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

    def _create_marker_img(
        self,
        marker_data,
        thickness: float = 3.0,
        tip_length: float = 0.2,
        head_len_min: float = 3.0,
        head_len_max: float = 15.0,
        min_arrow_len_px: float = 6.0,
        head_angle_deg: float = 25.0,
    ):
        """Visualization of marker flow, like in the original FOTS simulation.
        Use tensor operations instead of NumPy and OpenCV to avoid CPU/Numpy conversions.

        Marker data needs to have the shape [2, num_markers, 2]
        - dim=0: init and current markers
        - dim=2: x and y values of the marker position

        Args:
            marker_data: marker flow data with shape [2, num_markers, 2]
            thickness: thickness of the arrow
            tip_length: length of the arrow tip
            head_len_min: minimum length of the arrow head
            head_len_max: maximum length of the arrow head
            min_arrow_len_px: minimum length of the arrow in pixels
            head_angle_deg: angle of the arrow head

        Returns:
            Image with the markers visualized as arrows.
        """
        # white background with black arrows (0)
        device = marker_data.device if torch.is_tensor(marker_data) else self._device
        if not torch.is_tensor(marker_data):
            marker_data = torch.as_tensor(marker_data, device=device, dtype=torch.float32)
        else:
            marker_data = marker_data.to(device=device, dtype=torch.float32)

        H, W = self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]

        # marker data: [2, N, 2] -> (x, y)
        starts = marker_data[0]  # [N, 2]
        ends = marker_data[1]  # [N, 2]
        N = starts.shape[0]
        if N == 0:
            return torch.ones((H, W), device=device, dtype=torch.float32) * 255.0

        # main direction
        vec = ends - starts  # [N, 2]
        lens = torch.linalg.norm(vec, dim=1).clamp_min(1e-6)  # [N]
        u = vec / lens.unsqueeze(-1)  # unit vec [N, 2]
        # perpendicular
        perp = torch.stack([-u[:, 1], u[:, 0]], dim=1)  # [N, 2]

        alpha = math.radians(head_angle_deg)
        ca_t = torch.tensor(math.cos(alpha), device=device, dtype=torch.float32)
        sa_t = torch.tensor(math.sin(alpha), device=device, dtype=torch.float32)

        # Two arrow branches (backwards V-shape)
        head_dir1 = -ca_t * u + sa_t * perp  # [N, 2]
        head_dir2 = -ca_t * u - sa_t * perp  # [N, 2]

        # OpenCV equivalent: head_len = tip_length * |line|, and clamp
        head_len_vec = (lens * tip_length).clamp(min=head_len_min, max=head_len_max)  # [N]
        # Short segments suppress arrow heads
        draw_head_mask = lens >= min_arrow_len_px  # [N]
        head_end1 = ends + head_len_vec.unsqueeze(-1) * head_dir1
        head_end2 = ends + head_len_vec.unsqueeze(-1) * head_dir2

        # distance-field rasterization for a batch of segments
        def rasterize(seg_starts, seg_ends):
            if seg_starts.shape[0] == 0:
                return torch.zeros((0, H, W), dtype=torch.bool, device=device)
            sx = seg_starts[:, 0].view(-1, 1, 1)
            sy = seg_starts[:, 1].view(-1, 1, 1)
            ex = seg_ends[:, 0].view(-1, 1, 1)
            ey = seg_ends[:, 1].view(-1, 1, 1)
            vx = ex - sx
            vy = ey - sy
            len2 = (vx * vx + vy * vy).clamp_min(1e-6)

            X = self._grid_x.view(1, H, W)
            Y = self._grid_y.view(1, H, W)
            t = ((X - sx) * vx + (Y - sy) * vy) / len2
            t = t.clamp(0.0, 1.0)
            qx = sx + t * vx
            qy = sy + t * vy
            d2 = (X - qx) * (X - qx) + (Y - qy) * (Y - qy)
            r2 = (thickness * 0.5) ** 2
            return d2 <= r2  # [N, H, W] bool

        # Main line
        mask_main = rasterize(starts, ends)  # [N, H, W]

        # Arrows (only draw lines that meet threshold)
        idx = torch.nonzero(draw_head_mask, as_tuple=False).squeeze(-1)
        if idx.numel() > 0:
            mask_head = rasterize(ends[idx], head_end1[idx]) | rasterize(ends[idx], head_end2[idx])  # [N', H, W]
            heads_any = mask_head.any(dim=0)
        else:
            heads_any = torch.zeros((H, W), dtype=torch.bool, device=device)

        mask = mask_main.any(dim=0) | heads_any  # [H, W]

        # compose to image: white(255) background, black(0) strokes
        frame = (1 - mask.to(torch.uint8)) * 255
        return frame.to(torch.float32)

    def set_active_target_idx(self, env_ids, idxs):
        # switch to manual mode and set the active target_idx
        self.cfg.target_select_mode = "manual"
        if isinstance(env_ids, int):
            env_ids = [env_ids]
            idxs = [idxs]
        for e, i in zip(env_ids, idxs):
            self._active_target_idx[e] = int(i)
            self._stable_steps[e] = 0
