# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ignore optional memory usage warning globally
# pyright: reportOptionalSubscript=false

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from .gripper_contact_sensor_cfg import GripperContactSensorCfg

from isaaclab.sensors import ContactSensor


class GripperContactSensor(ContactSensor):
    """A contact sensor with enhanced visualization for gripper applications.

    Extends the base ContactSensor with:
    - Net force arrow visualization
    - Triaxial force component visualization
    - Gripper-specific offset handling as visualization utilities
    """

    cfg: GripperContactSensorCfg
    """The configuration parameters."""

    def __init__(self, cfg: GripperContactSensorCfg):
        """Initializes the gripper contact sensor object.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)

    """
    Add varied visualization methods:
    - visualize the net contact forces
    - visualize the triaxial forces
    - visualize the contact spheres
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if self.cfg.visualize_net_force_arrows:
                # create net force arrow markers
                if not hasattr(self, "force_arrow_visualizer"):
                    self.force_arrow_visualizer = VisualizationMarkers(self.cfg.net_force_visualizer_cfg)
                self.force_arrow_visualizer.set_visibility(True)
            elif self.cfg.visualize_triaxial_forces:
                # create triaxial force component markers
                if not hasattr(self, "triaxial_force_visualizer"):
                    self.triaxial_force_visualizer = VisualizationMarkers(self.cfg.triaxial_force_visualizer_cfg)
                self.triaxial_force_visualizer.set_visibility(True)
            else:
                # create markers if necessary for the first tome
                if not hasattr(self, "contact_visualizer"):
                    self.contact_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
                # set their visibility to true
                self.contact_visualizer.set_visibility(True)

        else:
            # hide all visualizers
            if hasattr(self, "contact_visualizer"):
                self.contact_visualizer.set_visibility(False)
            if hasattr(self, "force_arrow_visualizer"):
                self.force_arrow_visualizer.set_visibility(False)
            if hasattr(self, "triaxial_force_visualizer"):
                self.triaxial_force_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Debug visualization callback function."""
        if self.cfg.visualize_net_force_arrows:
            # visualize net force arrows
            self._visualize_force_arrows()
        elif self.cfg.visualize_triaxial_forces:
            # visualize triaxial forces
            self._visualize_triaxial_forces()
        else:
            # visualize contact spheres
            self._visualize_contact_spheres()

    def _visualize_contact_spheres(self):
        """Visualize contact spheres for Contact or Not checking."""
        # safely return if view becomes invalid
        # note: this invalidity happens because of isaac sim view callbacks
        if self.body_physx_view is None:
            return
        # marker indices
        # 0: contact, 1: no contact
        net_contact_force_w = torch.norm(self._data.net_forces_w, dim=-1)
        marker_indices = torch.where(net_contact_force_w > self.cfg.force_threshold, 0, 1)
        # check if prim is visualized
        if self.cfg.track_pose:
            frame_origins: torch.Tensor = self._data.pos_w
        else:
            pose = self.body_physx_view.get_transforms()
            frame_origins = pose.view(-1, self._num_bodies, 7)[:, :, :3]
        # visualize
        self.contact_visualizer.visualize(frame_origins.view(-1, 3), marker_indices=marker_indices.view(-1))

    """
    Net Forces and Triaxial Forces visualization utilities.
    """

    def _validate_visualizer(self, visualizer_name: str) -> bool:
        """Validates if a visualizer attribute exists and is not None."""
        if not hasattr(self, visualizer_name):
            return False
        if self.__getattribute__(visualizer_name) is None:
            return False
        return True

    def _compute_rotation_quaternions(
        self, target_directions: torch.Tensor, source_axis: torch.Tensor = None
    ) -> torch.Tensor:
        """Batch-compute quaternions (w, x, y, z) that rotate the source axis onto the target directions.

        Args:
            target_directions (torch.Tensor): (N, 3) or (3,), target direction vectors.
            source_axis (torch.Tensor, optional): (N, 3) or (3,), source axis for rotation. Defaults to +X if None.

        Returns:
            torch.Tensor: (N, 4) or (4,), quaternions in (w, x, y, z) format.
        """
        # Handle single-vector input
        single_input = target_directions.ndim == 1
        if single_input:
            target_directions = target_directions.unsqueeze(0)

        device = target_directions.device
        num = target_directions.shape[0]

        # Prepare source axis
        if source_axis is None:
            source = torch.tensor([1.0, 0.0, 0.0], device=device).expand(num, -1)
        else:
            source = source_axis.to(device)
            if source.ndim == 1:
                source = source.unsqueeze(0).expand(num, -1)

        # Normalize
        source = math_utils.normalize(source)
        target = math_utils.normalize(target_directions)

        # Dot and cross products
        dot = (source * target).sum(-1).clamp(-1.0, 1.0)  # (N,)
        cross = torch.cross(source, target, dim=-1)  # (N,3)
        cross_norm = torch.linalg.vector_norm(cross, dim=-1)  # (N,)

        # Case masks
        aligned = (1.0 - dot).abs() < 1e-6
        opposite = (1.0 + dot).abs() < 1e-6
        general = ~(aligned | opposite)

        # Result container (identity quaternion by default)
        quats = torch.zeros(num, 4, device=device)
        quats[:, 0] = 1.0

        # General case: axis = normalize(cross), angle = atan2(||cross||, dot)
        if general.any():
            axis_g = math_utils.normalize(cross[general])
            angle_g = torch.atan2(cross_norm[general], dot[general])  # [0, pi]
            quats[general] = math_utils.quat_from_angle_axis(angle_g, axis_g)

        # 180° case: choose an axis orthogonal to source; angle = pi
        if opposite.any():
            ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(num, -1)
            ey = torch.tensor([0.0, 1.0, 0.0], device=device).expand(num, -1)
            # Avoid collinearity with source: use y-axis if nearly collinear with x-axis, else x-axis
            use_ex = (source[:, 0].abs() < 0.9).unsqueeze(-1).expand(-1, 3)
            u = torch.where(use_ex, ex, ey)
            axis_o = math_utils.normalize(torch.cross(source[opposite], u[opposite], dim=-1))
            angle_o = torch.full((axis_o.shape[0],), math.pi, device=device)
            quats[opposite] = math_utils.quat_from_angle_axis(angle_o, axis_o)

        # 0° case is already the identity quaternion; nothing to do
        return quats.squeeze(0) if single_input else quats

    def _world_force_to_local(self, world_force: torch.Tensor, local_quat: torch.Tensor) -> torch.Tensor:
        """
        Transform forces from world coordinates to local coordinates.

        Args:
            world_force (torch.Tensor): (N, 3), force vectors in world frame.
            local_quat (torch.Tensor): (N, 4), local frame quaternion (w, x, y, z).

        Returns:
            torch.Tensor: (N, 3), force vectors expressed in the local frame.
        """
        # Quaternion conjugate is the inverse rotation
        quat_conj = math_utils.quat_conjugate(local_quat)
        # Apply inverse rotation to transform world-frame vectors into local-frame
        local_force = math_utils.quat_apply(quat_conj, world_force)
        return local_force

    def _visualize_force_arrows(self):
        """Visualize resultant force arrows for fingers."""

        if not self._validate_visualizer("force_arrow_visualizer"):
            return

        finger_data = self._get_finger_data()
        if finger_data is None or not finger_data["has_force"].any():
            self.force_arrow_visualizer.set_visibility(False)
            return

        # Apply offsets: finger frame is transformed from body origin to +Z grasp direction at fingertip
        positions, orientations = self._apply_finger_offsets(finger_data["positions"], finger_data["orientations"])

        # Compute positions for visualization
        vis_positions = self._compute_visualization_positions(positions, orientations)

        flat_positions = vis_positions.view(-1, 3)  # (N*2, 3)
        flat_forces = finger_data["forces"].view(-1, 3)  # (N*2, 3)
        flat_magnitudes = finger_data["magnitudes"].view(-1)  # (N*2,)

        # Render arrows
        self._render_force_arrows(flat_positions, flat_forces, flat_magnitudes, self.force_arrow_visualizer)

    def _visualize_triaxial_forces(self):
        """Visualize tri-axial force components for each finger."""

        if not self._validate_visualizer("triaxial_force_visualizer"):
            return

        finger_data = self._get_finger_data()
        if finger_data is None or not finger_data["has_force"].any():
            self.triaxial_force_visualizer.set_visibility(False)
            return

        if finger_data["forces"].shape[1] < 2:
            raise ValueError("Tri-axial force visualization requires at least two fingers.")

        # Apply offsets and convert forces into local coordinates
        positions, orientations = self._apply_finger_offsets(finger_data["positions"], finger_data["orientations"])

        local_forces = self._world_force_to_local(finger_data["forces"], orientations)

        # Compute positions for visualization
        vis_positions = self._compute_visualization_positions(positions, orientations)

        flat_positions = vis_positions.view(-1, 3)  # (N, 2, 3) -> (N*2, 3)
        flat_forces = local_forces.view(-1, 3)  # (N, 2, 3) -> (N*2, 3)
        flat_orientations = orientations.view(-1, 4)  # (N, 2, 4) -> (N*2, 4)

        # Render tri-axial arrows
        self._render_triaxial_arrows(flat_positions, flat_forces, flat_orientations)

    def _get_finger_data(self):
        """Collect force, position, and orientation data for fingers.

        Returns:
            dict: A dictionary with keys:
                - 'forces' (torch.Tensor): (N, F, 3), net forces per finger in world frame.
                - 'positions' (torch.Tensor): (N, F, 3), finger positions in world frame.
                - 'orientations' (torch.Tensor): (N, F, 4), finger quaternions (w, x, y, z) in world frame.
                - 'magnitudes' (torch.Tensor): (N, F), L2 norms of forces.
                - 'has_force' (torch.Tensor): (N, F), boolean mask where force > threshold.
                where F = min(actual_fingers, 2).
        """
        contact_force = self._data.net_forces_w
        if contact_force is None:
            return None

        # Handle single/two-finger cases
        num_fingers = min(contact_force.shape[1], 2)
        finger_forces = contact_force[:, :num_fingers, :]

        # Get positions and orientations
        if self.cfg.track_pose:
            positions = self._data.pos_w[:, :num_fingers, :]
            orientations = self._data.quat_w[:, :num_fingers, :]
        else:
            pose = self.body_physx_view.get_transforms()
            frame_data = pose.view(-1, self._num_bodies, 7)
            positions = frame_data[:, :num_fingers, :3]
            orientations = math_utils.convert_quat(frame_data[:, :num_fingers, 3:], to="wxyz")

        # Compute force magnitudes and mask
        force_magnitudes = torch.linalg.vector_norm(finger_forces, dim=-1)
        has_force = force_magnitudes > self.cfg.vis_force_threshold

        return {
            "forces": finger_forces,
            "positions": positions,
            "orientations": orientations,
            "magnitudes": force_magnitudes,
            "has_force": has_force,
        }

    def _apply_finger_offsets(self, positions, orientations):
        """Apply per-finger pose offsets.

        Args:
            positions (torch.Tensor): (N, F, 3), finger positions in world frame.
            orientations (torch.Tensor): (N, F, 4), finger quaternions (w, x, y, z).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - positions (torch.Tensor): (N, F, 3), offset positions in world frame.
                - orientations (torch.Tensor): (N, F, 4), orientations after applying rotational offsets.
        """
        offset_quats = torch.stack(
            [
                torch.tensor(self.cfg.left_finger_offset.rot, device=self._device),
                torch.tensor(self.cfg.right_finger_offset.rot, device=self._device),
            ]
        )
        offset_positions = torch.stack(
            [
                torch.tensor(self.cfg.left_finger_offset.pos, device=self._device),
                torch.tensor(self.cfg.right_finger_offset.pos, device=self._device),
            ]
        )

        # Apply rotational offsets in world frame
        transformed_quats = math_utils.quat_mul(orientations, offset_quats.unsqueeze(0).expand(self._num_envs, -1, -1))

        # Apply translational offsets in world frame
        offset_pos_world = math_utils.quat_apply(orientations, offset_positions)
        transformed_positions = positions + offset_pos_world

        return transformed_positions, transformed_quats

    def _compute_visualization_positions(self, positions, orientations):
        """Compute arrow base positions used for visualization.

        Args:
            positions (torch.Tensor): (N, F, 3), finger positions in world frame.
            orientations (torch.Tensor): (N, F, 4), finger quaternions (w, x, y, z).

        Returns:
            torch.Tensor: (N, F, 3), positions shifted along the finger frame -Z axis by the configured distance.
        """
        # Offset along local -Z direction
        vis_offset_distance = self.cfg.vis_offset_distance

        # Create -Z direction vectors for all fingers
        z_direction = torch.tensor([0.0, 0.0, -1.0], device=self._device)
        z_directions = z_direction.expand(positions.shape[0], positions.shape[1], 3)

        # Rotate local -Z into world frame
        offset_axis_world = math_utils.quat_apply(orientations, z_directions)

        # Shift positions by the offset
        vis_positions = positions + offset_axis_world * vis_offset_distance

        return vis_positions

    def _render_force_arrows(self, positions, directions, magnitudes, visualizer, marker_indices=None):
        """Render force arrows (generic utility).

        Args:
            positions (torch.Tensor): (N, 3), arrow base positions in world frame.
            directions (torch.Tensor): (N, 3), arrow direction vectors; will be normalized.
            magnitudes (torch.Tensor): (N,), force magnitudes to determine arrow lengths.
            visualizer: Visualization handler for arrows (must support set_visibility and visualize).
            marker_indices (Optional[torch.Tensor]): (N,), per-arrow indices to map to marker instances.

        Returns:
            None
        """
        if len(positions) == 0:
            visualizer.set_visibility(False)
            return

        # Read configuration for rendering
        arrow_length = self.cfg.max_force_arrow_length
        arrow_thickness = self.cfg.arrow_thickness
        max_force = self.cfg.max_force

        # Normalize direction vectors and compute quaternions to rotate +X onto direction
        directions_normalized = torch.nn.functional.normalize(directions, dim=-1)
        arrow_orientations = self._compute_rotation_quaternions(
            directions_normalized, source_axis=torch.tensor([1.0, 0.0, 0.0], device=self._device)  # x-axis vector
        )

        # Normalize and clamp magnitudes
        normalized_magnitudes = (magnitudes / max_force).clamp(0.0, 1.0)

        # Compute scales
        arrow_scales = torch.stack(
            [
                normalized_magnitudes * arrow_length,  # x-scale (length)
                torch.full_like(magnitudes, arrow_thickness),  # y-scale (thickness)
                torch.full_like(magnitudes, arrow_thickness),  # z-scale (thickness)
            ],
            dim=-1,
        )

        # Default marker indices if not provided
        if marker_indices is None:
            marker_indices = torch.arange(len(positions), device=self._device)

        # Render
        visualizer.set_visibility(True)
        visualizer.visualize(
            translations=positions, orientations=arrow_orientations, scales=arrow_scales, marker_indices=marker_indices
        )

    def _render_triaxial_arrows(self, positions, forces, orientations):
        """Render tri-axial component arrows per point using the generic force arrow renderer.

        Args:
            positions (torch.Tensor): (N, 3), base positions in world frame.
            forces (torch.Tensor): (N, 3), local-frame forces (Fx, Fy, Fz).
            orientations (torch.Tensor): (N, 4), local-to-world quaternions (w, x, y, z).

        Returns:
            None
        """
        if forces.shape[0] == 0:
            self.triaxial_force_visualizer.set_visibility(False)
            return

        # For each data point, render 3 axes
        n_points = forces.shape[0]

        # Local axes unit vectors: X, Y, Z
        local_axes = torch.eye(3, device=self._device)  # [[1,0,0], [0,1,0], [0,0,1]]

        # Expand to all points and flatten
        arrow_positions = positions.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)  # (N*3, 3)

        # Transform local axes into world frame for each orientation
        orientations_expanded = orientations.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 4)  # (N*3, 4)
        local_axes_expanded = local_axes.unsqueeze(0).expand(n_points, -1, -1).reshape(-1, 3)  # (N*3, 3)
        world_directions = math_utils.quat_apply(orientations_expanded, local_axes_expanded)  # (N*3, 3)

        # Use component magnitudes and signs to encode local force direction:
        # - magnitude: abs(Fx), abs(Fy), abs(Fz)
        # - direction: +axis for positive, -axis for negative
        forces_flat = forces.view(-1, 3)  # (N, 3)
        force_magnitudes = torch.abs(forces_flat).reshape(-1)  # (N*3,)
        axis_signs = torch.sign(forces_flat).reshape(-1)  # (N*3,)
        # Avoid zero-length directions for near-zero forces to keep normalization stable
        axis_signs = torch.where(
            force_magnitudes > 0.0,
            axis_signs,
            torch.ones_like(axis_signs),
        )
        world_directions = world_directions * axis_signs.unsqueeze(-1)

        # Axis indices 0/1/2 for X/Y/Z
        axis_indices = torch.arange(3, device=self._device).unsqueeze(0).expand(n_points, -1).reshape(-1)

        # Render using the generic function
        self._render_force_arrows(
            positions=arrow_positions,
            directions=world_directions,
            magnitudes=force_magnitudes,
            visualizer=self.triaxial_force_visualizer,
            marker_indices=axis_indices,
        )
