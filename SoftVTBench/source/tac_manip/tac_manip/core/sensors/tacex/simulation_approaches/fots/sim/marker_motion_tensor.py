# Modified version of the source code from:
# @article{zhao2024fots,
#   title={FOTS: A Fast Optical Tactile Simulator for Sim2Real Learning of Tactile-motor Robot Manipulation Skills},
#   author={Zhao, Yongqiang and Qian, Kun and Duan, Boyi and Luo, Shan},
#   journal={IEEE Robotics and Automation Letters},
#   year={2024}
# }
# Ref.: https://github.com/Rancho-zhao/FOTS/tree/main
#
# Main changes:
# - rewrote computations to be more akin to the equations in the paper
# - added delta s_max and delta theta_max to clip the marker motion
# - #todo vectorize
# - #todo multi FrameTransformer rotation extraction (currently, can only use FOTS for single object)

import math

import cv2
import numpy as np
import torch


class MarkerMotionTensor:
    def __init__(
        self,
        frame0_blur,
        lamb,
        mm2pix=15.7729,  # FOTS default is 19.58, we use 1/0.0634 = 15.7729 (see https://github.com/gelsightinc/gsrobotics/tree/main?tab=readme-ov-file#1-what-is-the-mm-to-pixel-conversion-value)
        num_markers_col=9,
        num_markers_row=11,
        tactile_img_width=320,
        tactile_img_height=240,
        x0=0.0,
        y0=0.0,
        is_flow=True,
    ):
        self.frame0_blur = frame0_blur

        self.lamb = lamb

        self.mm2pix = mm2pix
        self.num_markers_col = num_markers_col
        self.num_markers_row = num_markers_row
        self.tactile_img_width = tactile_img_width
        self.tactile_img_height = tactile_img_height

        self.contact = []
        self.moving = False
        self.rotation = False

        self.mkr_rng = 0.5

        self.device = getattr(self, "device", "cuda")

        # Cache lambda on device (float32)
        self.lamb_t = torch.as_tensor(self.lamb, device=self.device, dtype=torch.float32)
        self.lamb0, self.lamb1, self.lamb2 = self.lamb_t[0], self.lamb_t[1], self.lamb_t[2]

        # x column-wise [0..W-1], y row-wise [0..H-1]
        self.x = torch.arange(0, self.tactile_img_width, device=self.device, dtype=torch.float32)
        self.y = torch.arange(0, self.tactile_img_height, device=self.device, dtype=torch.float32)
        self.xx, self.yy = torch.meshgrid(self.x, self.y, indexing="xy")  # [H, W]

        # Compute marker indices
        marker_x_idx = torch.linspace(x0, self.tactile_img_width - x0, self.num_markers_col, device=self.device).to(
            torch.int64
        )
        marker_y_idx = torch.linspace(y0, self.tactile_img_height - y0, self.num_markers_row, device=self.device).to(
            torch.int64
        )

        mx, my = torch.meshgrid(marker_x_idx, marker_y_idx, indexing="xy")
        self.marker_x_idx = mx.reshape(-1).to(torch.int64)
        self.marker_y_idx = my.reshape(-1).to(torch.int64)

        self.init_marker_x_pos = (
            self.xx[self.marker_y_idx, self.marker_x_idx]
            .reshape(self.num_markers_row, self.num_markers_col)
            .to(torch.float32)
        )
        self.init_marker_y_pos = (
            self.yy[self.marker_y_idx, self.marker_x_idx]
            .reshape(self.num_markers_row, self.num_markers_col)
            .to(torch.float32)
        )

        self.marker_x_pos = self.init_marker_x_pos.clone()
        self.marker_y_pos = self.init_marker_y_pos.clone()

    def _shear(self, center_x, center_y, lamb, shear_x, shear_y, xx, yy, shear_max=10.0):
        # center_x, center_y: [B], shear_x, shear_y: [B]
        # xx, yy: [B, R, C]
        sx = torch.clamp(shear_x, -shear_max, shear_max).view(-1, 1, 1)
        sy = torch.clamp(shear_y, -shear_max, shear_max).view(-1, 1, 1)
        cx = center_x.view(-1, 1, 1)
        cy = center_y.view(-1, 1, 1)

        dx_ = xx - cx
        dy_ = yy - cy
        g = torch.exp(-lamb * (dx_ * dx_ + dy_ * dy_))
        dx = sx * g
        dy = sy * g
        return dx, dy

    def _twist(self, center_x, center_y, lamb, theta, xx, yy, theta_max_deg=60.0):
        # theta: [B] (rad), clamp
        theta = torch.clamp(
            theta,
            -theta_max_deg / 180.0 * math.pi,
            theta_max_deg / 180.0 * math.pi,
        ).view(-1, 1, 1)
        cx = center_x.view(-1, 1, 1)
        cy = center_y.view(-1, 1, 1)

        off_x = xx - cx
        off_y = yy - cy
        g = torch.exp(-lamb * (off_x * off_x + off_y * off_y))

        rotx = off_x * torch.cos(theta - 1.0) - off_y * torch.sin(theta)
        roty = off_x * torch.sin(theta) + off_y * torch.cos(theta - 1.0)

        dx = rotx * g
        dy = roty * g
        return dx, dy

    def _dilate_masked(self, lamb, marker_x_pos, marker_y_pos, in_contact, contact_height):
        # marker_x_pos, marker_y_pos: [B, R, C] float
        # in_contact, contact_height: [B, R, C]

        B, R, C = marker_x_pos.shape
        M = R * C
        out_x = torch.zeros_like(marker_x_pos)
        out_y = torch.zeros_like(marker_y_pos)

        x_flat = marker_x_pos.reshape(B, M)
        y_flat = marker_y_pos.reshape(B, M)
        h_flat = contact_height.reshape(B, M)
        w_flat = in_contact.reshape(B, M)

        for b in range(B):
            idx = torch.nonzero(w_flat[b] > 0, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            cx = x_flat[b, idx]  # [K]
            cy = y_flat[b, idx]  # [K]
            hh = h_flat[b, idx]  # [K]

            tx = x_flat[b].unsqueeze(1) - cx.unsqueeze(0)  # [M,K]
            ty = y_flat[b].unsqueeze(1) - cy.unsqueeze(0)  # [M,K]
            g = torch.exp(-lamb * (tx * tx + ty * ty))
            dx = (tx * g * hh.unsqueeze(0)).sum(dim=1).reshape(R, C)
            dy = (ty * g * hh.unsqueeze(0)).sum(dim=1).reshape(R, C)
            out_x[b], out_y[b] = dx, dy
        return out_x, out_y

    def _generate(self, xx, yy):
        img = np.zeros_like(self.frame0_blur.detach().cpu().numpy())

        for i in range(self.num_markers_col):
            for j in range(self.num_markers_row):
                ini_r = int(self.init_marker_y_pos[j, i])
                ini_c = int(self.init_marker_x_pos[j, i])
                r = int(yy[j, i])
                c = int(xx[j, i])
                if r >= self.tactile_img_height or r < 0 or c >= self.tactile_img_width or c < 0:
                    continue

                k = 5

                pt1 = (ini_c, ini_r)
                pt2 = (c + k * (c - ini_c), r + k * (r - ini_r))
                color = (0, 255, 0)
                cv2.arrowedLine(img, pt1, pt2, color, 2, tipLength=0.2)

        img = img[: self.tactile_img_height, : self.tactile_img_width]
        return img

    def _motion_callback(self, marker_x_pos, marker_y_pos, depth_map, contact_mask, traj_first, traj_last, has_traj):
        """Update marker positions based on the trajectory and depth map.

        Args:  (All inputs are torch tensors unless noted.)
            marker_x_pos, marker_y_pos: [B, R, C] float
            depth_map, contact_mask:    [B, H, W] (depth_map is shifted to cm in-place here)
            traj_first, traj_last:      [B, 3] -> [x_mm, y_mm, theta]
            has_traj:                   [B] bool

        Returns:
            new_x_pos, new_y_pos: [B, R, C] float
        """
        B, R, C = marker_x_pos.shape
        H = self.tactile_img_height
        W = self.tactile_img_width

        # Normalize height map per batch (min-shift) and convert mm->cm (/10)
        depth_map = depth_map - depth_map.amin(dim=(1, 2), keepdim=True)
        depth_map = depth_map / 10.0

        # Determine which markers are in contact (sample contact_mask / depth at marker locations)
        yi = torch.clamp(marker_y_pos.to(torch.long), 0, H - 1)
        xi = torch.clamp(marker_x_pos.to(torch.long), 0, W - 1)
        valid = (marker_y_pos >= 0) & (marker_y_pos < H) & (marker_x_pos >= 0) & (marker_x_pos < W)

        lin = yi * W + xi  # [B, R, C]
        depth_flat = depth_map.reshape(B, -1)
        mask_flat = contact_mask.reshape(B, -1)

        h_at_marker = torch.gather(depth_flat, 1, lin.reshape(B, -1)).reshape(B, R, C)
        m_at_marker = torch.gather(mask_flat, 1, lin.reshape(B, -1)).reshape(B, R, C) > 0.5
        in_contact = m_at_marker & valid

        if not in_contact.any():
            return self.init_marker_x_pos.expand(B, -1, -1), self.init_marker_y_pos.expand(B, -1, -1)

        # Normal load (dilation term)
        x_dd, y_dd = self._dilate_masked(self.lamb0, marker_x_pos, marker_y_pos, in_contact, h_at_marker)

        new_x = marker_x_pos + x_dd
        new_y = marker_y_pos + y_dd

        # Shear/Twist only apply when a valid trajectory exists (len >= 2)
        active_idx = torch.nonzero(has_traj, as_tuple=False).squeeze(-1)
        if active_idx.numel() > 0:
            mm2pix = float(self.mm2pix)

            cx_pix = traj_first[active_idx, 0] * mm2pix + (W / 2.0)
            cy_pix = traj_first[active_idx, 1] * mm2pix + (H / 2.0)
            lcx_pix = traj_last[active_idx, 0] * mm2pix + (W / 2.0)
            lcy_pix = traj_last[active_idx, 1] * mm2pix + (H / 2.0)
            sx_pix = (traj_last[active_idx, 0] - traj_first[active_idx, 0]) * mm2pix
            sy_pix = (traj_last[active_idx, 1] - traj_first[active_idx, 1]) * mm2pix
            dtheta = traj_last[active_idx, 2] - traj_first[active_idx, 2]

            # Use truncation to match reference implementation
            cx_i = torch.trunc(cx_pix).to(torch.float32)
            cy_i = torch.trunc(cy_pix).to(torch.float32)
            lcx_i = torch.trunc(lcx_pix).to(torch.float32)
            lcy_i = torch.trunc(lcy_pix).to(torch.float32)
            sx_i = torch.trunc(sx_pix).to(torch.float32)
            sy_i = torch.trunc(sy_pix).to(torch.float32)

            # calculate shear/twist for subset of tensors
            mx_sub = marker_x_pos[active_idx]  # [K,R,C]
            my_sub = marker_y_pos[active_idx]  # [K,R,C]

            dx_s, dy_s = self._shear(
                cx_i,
                cy_i,
                self.lamb1,
                sx_i,
                sy_i,
                mx_sub,
                my_sub,
                shear_max=10.0,
            )
            dx_t, dy_t = self._twist(
                lcx_i,
                lcy_i,
                self.lamb2,
                dtheta,
                mx_sub,
                my_sub,
                theta_max_deg=60.0,
            )

            # write back to corresponding envs
            new_x[active_idx] = new_x[active_idx] + dx_s + dx_t
            new_y[active_idx] = new_y[active_idx] + dy_s + dy_t

        return new_x, new_y

    def marker_sim(self, depth_map, contact_mask, traj):
        # Compatible with old interface: treat as batch=1
        if isinstance(depth_map, np.ndarray):
            depth_map = torch.from_numpy(depth_map).to(self.device).to(torch.float32)
        if isinstance(contact_mask, np.ndarray):
            contact_mask = torch.from_numpy(contact_mask).to(self.device)
        # traj is a Python list: [[x,y,theta], ...]
        if len(traj) == 0:
            has_traj = torch.tensor([False], device=self.device)
            traj_first = torch.zeros((1, 3), device=self.device)
            traj_last = torch.zeros((1, 3), device=self.device)
        else:
            has_traj = torch.tensor([len(traj) >= 2], device=self.device)
            traj_first = torch.tensor(traj[0], device=self.device, dtype=torch.float32).view(1, 3)
            traj_last = torch.tensor(traj[-1], device=self.device, dtype=torch.float32).view(1, 3)

        x0 = self.init_marker_x_pos.unsqueeze(0)
        y0 = self.init_marker_y_pos.unsqueeze(0)
        new_x, new_y = self._motion_callback(
            x0, y0, depth_map.unsqueeze(0), contact_mask.unsqueeze(0), traj_first, traj_last, has_traj
        )
        return new_x[0], new_y[0]

    def marker_sim_batched(self, depth_map, contact_mask, traj_first, traj_last, has_traj):
        # depth_map, contact_mask: [B, H, W]
        # traj_first, traj_last: [B, 3], has_traj: [B]
        B = depth_map.shape[0]
        x0 = self.init_marker_x_pos.unsqueeze(0).expand(B, -1, -1)
        y0 = self.init_marker_y_pos.unsqueeze(0).expand(B, -1, -1)
        new_x, new_y = self._motion_callback(x0, y0, depth_map, contact_mask, traj_first, traj_last, has_traj)
        return new_x, new_y
