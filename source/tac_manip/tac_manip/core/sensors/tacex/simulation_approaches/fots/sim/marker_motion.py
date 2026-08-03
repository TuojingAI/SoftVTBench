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
# - Rewrote computations to align with equations from the paper
# - Added clipping for delta s and delta theta
# - TODO: multi FrameTransformer rotation extraction (currently only supports a single object)

import math

import cv2
import numpy as np


class MarkerMotion:
    def __init__(
        self,
        frame0_blur,
        lamb,
        mm2pix=15.7729,  # FOTS default is 19.58, we use 1/0.0634 = 15.7729 (see https://github.com/gelsightinc/gsrobotics/tree/main?tab=readme-ov-file#1-what-is-the-mm-to-pixel-conversion-value)
        num_markers_col=9,
        num_markers_row=11,
        tactile_img_width=320,
        tactile_img_height=240,
        x0=0,
        y0=0,
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

        self.x = np.arange(0, self.tactile_img_width, 1)  # x is column-wise defined
        self.y = np.arange(0, self.tactile_img_height, 1)  # y is row-wise defined
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        # self.xx_init, self.yy_init = np.meshgrid(self.x, self.y)

        # compute marker indices based on number of markers per column/row and image shape
        marker_x_idx = np.linspace(x0, self.tactile_img_width - x0, self.num_markers_col, dtype=int)
        marker_y_idx = np.linspace(y0, self.tactile_img_height - y0, self.num_markers_row, dtype=int)
        # marker_x_idx = np.arange(int(self.tactile_img_width/self.num_markers_col), self.tactile_img_width - int(self.tactile_img_width/self.num_markers_col), int(self.tactile_img_width/self.num_markers_col))
        # marker_y_idx = np.arange(int(self.tactile_img_height/self.num_markers_row), self.tactile_img_height - int(self.tactile_img_height/self.num_markers_row), int(self.tactile_img_height/self.num_markers_row))

        marker_x_idx, marker_y_idx = np.meshgrid(marker_x_idx, marker_y_idx)
        self.marker_x_idx = (marker_x_idx.reshape([1, -1])[0]).astype(np.int16)
        self.marker_y_idx = (marker_y_idx.reshape([1, -1])[0]).astype(np.int16)

        self.init_marker_x_pos = self.xx[self.marker_y_idx, self.marker_x_idx].reshape(
            [self.num_markers_row, self.num_markers_col]
        )
        self.init_marker_y_pos = self.yy[self.marker_y_idx, self.marker_x_idx].reshape(
            [self.num_markers_row, self.num_markers_col]
        )

        self.marker_x_pos = self.init_marker_x_pos
        self.marker_y_pos = self.init_marker_y_pos

    def _shear(self, center_x, center_y, lamb, shear_x, shear_y, xx, yy, shear_max=10):
        # TODO: add force and torque effect
        g = np.exp(-lamb * ((xx - center_x) ** 2 + (yy - center_y) ** 2))

        # use max delta s, like in the paper eq (12)
        shear_x = np.clip(shear_x, -shear_max, shear_max)
        shear_y = np.clip(shear_y, -shear_max, shear_max)

        dx, dy = shear_x * g, shear_y * g

        return dx, dy

    def _twist(self, center_x, center_y, lamb, theta, xx, yy, theta_max_deg=60):
        theta = np.clip(theta, -theta_max_deg / 180.0 * math.pi, theta_max_deg / 180.0 * math.pi)

        offset_x = xx - center_x
        offset_y = yy - center_y

        g = np.exp(-lamb * (offset_x**2 + offset_y**2))

        rotx = offset_x * np.cos(theta - 1) - offset_y * np.sin(theta)
        roty = offset_x * np.sin(theta) + offset_y * np.cos(theta - 1)

        # rotx = offset_x * np.cos(theta) - offset_y * np.sin(theta)
        # roty = offset_x * np.sin(theta) + offset_y * np.cos(theta)

        # rotx = np.clip(rotx, -theta_max_deg / 180.0 * math.pi, theta_max_deg / 180.0 * math.pi)
        # roty = np.clip(roty, -theta_max_deg / 180.0 * math.pi, theta_max_deg / 180.0 * math.pi)

        dx = rotx * g
        dy = roty * g

        return dx, dy

    def _dilate(self, lamb, xx, yy):
        dx, dy = 0.0, 0.0
        # todo vectorize
        for i in range(len(self.contact)):
            g = np.exp(-lamb * ((xx - self.contact[i][1]) ** 2 + (yy - self.contact[i][0]) ** 2))

            dx += self.contact[i][2] * (xx - self.contact[i][1]) * g
            dy += self.contact[i][2] * (yy - self.contact[i][0]) * g

        return dx, dy

    def _generate(self, xx, yy):
        img = np.zeros_like(self.frame0_blur.copy())

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

    def _motion_callback(self, marker_x_pos, marker_y_pos, depth_map, contact_mask, traj):
        """Update marker positions based on the trajectory and depth map.

        Args:
            marker_x_pos, marker_y_pos: [R, C] float
            depth_map, contact_mask:    [H, W] (depth_map is shifted to cm in-place here)
            traj:                       [3] -> [x_mm, y_mm, theta]

        Returns:
            new_x_pos, new_y_pos: [R, C] float
        """
        # normalize height of depth map
        depth_map = depth_map - depth_map.min()
        # Convert from m to cm, cause this cm is used in fots?
        # See https://github.com/Rancho-zhao/FOTS/blob/FOTS-mujoco/marker_motion.py#L107
        depth_map /= 10

        # Contact masks are dense images, while physical markers are sparse.
        # The original implementation only reacted if a marker pixel landed exactly
        # inside the contact mask, which makes small contacts disappear. Use the
        # contact patch center as the primary source, then keep the exact marker
        # hits below as extra local sources.
        contact_pixels = np.argwhere(contact_mask > 0)
        if contact_pixels.size > 0:
            ys = contact_pixels[:, 0]
            xs = contact_pixels[:, 1]
            weights = np.maximum(depth_map[ys, xs], 1e-6)
            center_y = float(np.average(ys, weights=weights))
            center_x = float(np.average(xs, weights=weights))
            contact_height = float(np.max(weights))
            self.contact.append([center_y, center_x, contact_height])

        # extract which markers are in contact
        for i in range(self.num_markers_col):
            for j in range(self.num_markers_row):
                y_pos = int(marker_y_pos[j, i])
                x_pos = int(marker_x_pos[j, i])
                if (
                    (y_pos >= self.tactile_img_height)
                    or (y_pos < 0)
                    or (x_pos >= self.tactile_img_width)
                    or (x_pos < 0)
                ):
                    continue
                if contact_mask[y_pos, x_pos] == 1.0:
                    tactile_img_height = depth_map[y_pos, x_pos]

                    self.contact.append([y_pos, x_pos, tactile_img_height])

        if not self.contact:
            marker_x_pos, marker_y_pos = self.init_marker_x_pos, self.init_marker_y_pos
            return marker_x_pos, marker_y_pos

        # compute marker motion under normal load
        x_dd, y_dd = self._dilate(self.lamb[0], marker_x_pos, marker_y_pos)
        new_x_pos = marker_x_pos + x_dd  # *0.0
        new_y_pos = marker_y_pos + y_dd  # *0.0

        if len(traj) >= 2:
            # under shear load
            # print("traj diff x,y ", (traj[-1][0] - traj[0][0]), (traj[-1][1] - traj[0][1]))
            x_ds, y_ds = self._shear(
                int(traj[0][0] * self.mm2pix + self.tactile_img_width / 2),
                int(traj[0][1] * self.mm2pix + self.tactile_img_height / 2),
                self.lamb[1],
                int((traj[-1][0] - traj[0][0]) * self.mm2pix),
                int((traj[-1][1] - traj[0][1]) * self.mm2pix),
                marker_x_pos,
                marker_y_pos,
            )
            new_x_pos += x_ds
            new_y_pos += y_ds

            # # under twist load
            # print("theta diff ", np.rad2deg(traj[-1][2]-traj[0][2]))
            # theta = max(min(traj[-1][2]-traj[0][2], 50 / 180.0 * math.pi), -50 / 180.0 * math.pi) # -50 = max angle, traj rotation is in rad!
            theta = traj[-1][2] - traj[0][2]
            # print("theta is ", np.rad2deg(theta))

            x_dt, y_dt = self._twist(
                int(traj[-1][0] * self.mm2pix + self.tactile_img_width / 2),
                int(traj[-1][1] * self.mm2pix + self.tactile_img_height / 2),
                self.lamb[2],
                theta,
                marker_x_pos,
                marker_y_pos,
            )
            new_x_pos += x_dt
            new_y_pos += y_dt

        return new_x_pos, new_y_pos

    def marker_sim(self, depth_map, contact_mask, traj):
        # update marker positions
        new_marker_x_pos, new_marker_y_pos = self._motion_callback(
            self.init_marker_x_pos, self.init_marker_y_pos, depth_map, contact_mask, traj
        )

        self.contact = []

        return new_marker_x_pos, new_marker_y_pos
