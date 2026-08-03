from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
)

from .observations import contact_force_in_gripper_frame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation
    from isaaclab.sensors import FrameTransformer


@configclass
class ForcePositionActionCfg(ActionTermCfg):
    """Configuration for the hybrid force-position action (13-D command: 6-D EEF pose
    in axis-angle + 1-D gripper + 3-D target force per finger).

    The action term does not expose IK details to the policy. Instead it:

    - wraps an Isaac Lab `DifferentialInverseKinematicsAction` internally (arm joints only);
    - combines the 3-D target force of each finger with the measured contact force to:
      - synthesise an external EEF wrench used as the OSC `wrench_abs` input;
      - compute the squeeze-force error and update the gripper aperture incrementally:
        `d_cmd = d_curr + squeeze_kp * (f_sq_curr - f_sq_target)`.

    The 13-D action layout matches the recorder
    `AbsEEFPoseAxisAngleAbsGripperWithForceActionStateRecorder`:

    - 0:6   -> absolute EEF pose in the base frame, (x, y, z, ax, ay, az) with (ax, ay, az)
               an axis-angle rotation
    - 6:7   -> absolute gripper aperture (not used directly here; the gripper is driven by
               the incremental force loop)
    - 7:10  -> left-finger target force (finger-local frame, fx, fy, fz)
    - 10:13 -> right-finger target force (finger-local frame, fx, fy, fz)

    Internally (x, y, z, ax, ay, az) is converted to (x, y, z, qw, qx, qy, qz) and fed to
    DiffIK as an absolute pose command; the target and measured fingertip forces form the
    hybrid position command:

        P_pos_hybrid = P_pos_target + K_pos * (F_target^b - F_measured^b)
    """

    # Nested IK action config, populated by the env cfg (asset_name/joint_names/body_name/offset).
    ik_cfg: DifferentialInverseKinematicsActionCfg = MISSING

    # Sensor and frame names; these must already exist in the env cfg.
    ee_frame_name: str = "ee_frame"
    left_gripper_frame_name: str = "left_gripper_frame"
    right_gripper_frame_name: str = "right_gripper_frame"
    contact_sensor_name: str = "contact_gripper"
    history_length: int = 1

    # Gains.
    # Position blending gain: a scalar shares one K across xyz, or pass (kx, ky, kz).
    pos_kp: float | tuple[float, float, float] = 0.0
    squeeze_kp: float = 0.001  # Force-error gain on the aperture. Since squeeze is defined with a
    # factor of 2, the control law scales the error by 0.5 to keep the effective magnitude.
    squeeze_deadzone: float = 0.1  # Squeeze-error dead zone, evaluated on the position correction
    # |delta_d| against |squeeze_kp| * squeeze_deadzone (kept for backwards compatibility).

    # Filtering of the measured squeeze force. Applies only to the gripper squeeze loop; it
    # does not affect observations/recording and does not filter the per-finger 3-D forces.
    # EMA: s_filt = alpha * s_curr + (1-alpha) * s_filt_prev
    # - alpha=1.0: no filtering (original behaviour)
    # - smaller alpha: stronger filtering, slower response
    meas_force_filter_alpha: float = 0.2

    # Squeeze feed-forward compensation (anti-slip: grip harder as the predicted squeeze
    # force grows). The target squeeze force is raised to
    #   f_sq_target_eff = f_sq_target + squeeze_ff_k_load_z * f_sq_target
    # which is equivalent to (1 + squeeze_ff_k_load_z) * f_sq_target.
    # A coefficient of 0 leaves the behaviour unchanged.
    squeeze_ff_k_load_z: float = 0.6
    squeeze_ff_contact_threshold: float = 1.0  # When > 0, feed-forward engages only once
    # f_sq_meas_raw >= threshold.

    # ActionTerm type used by the manager. A non-MISSING default is provided here and
    # overridden at the end of this module.
    class_type: type[ActionTerm] = ActionTerm


class ForcePositionAction(ActionTerm):
    """OSC-based hybrid force-position action term.

    Input action: (N, 13)
        - 0:6   -> absolute EEF pose (x, y, z, ax, ay, az) in the base frame, rotation as
                   axis-angle
        - 6:7   -> absolute gripper value (dimension is kept for alignment; not driven directly)
        - 7:10  -> left-finger target force (finger-local frame, fx, fy, fz)
        - 10:13 -> right-finger target force (finger-local frame, fx, fy, fz)

    Behaviour:
    - From the per-finger target forces it:
        - synthesises an external EEF wrench in the base frame for the internal OSC
          (`wrench_abs`);
        - updates the gripper aperture incrementally from the measured squeeze force:
          `d_cmd = d_curr + squeeze_kp * (f_sq_curr - f_sq_target)`.
    - The EEF pose is currently passed straight through as the OSC `pose_abs` target; an
      outer force-feedback loop engages when cfg.eef_kp > 0.
    """

    cfg: ForcePositionActionCfg

    def __init__(self, cfg: ForcePositionActionCfg, env: ManagerBasedRLEnv) -> None:
        # Initialise the base class, which resolves asset_name -> robot articulation.
        super().__init__(cfg, env)

        self._env: ManagerBasedRLEnv = env
        self._device = env.device

        # Robot.
        self._robot: Articulation = self._asset

        # Internal DiffIK action term (arm joints only).
        self._ik_term = DifferentialInverseKinematicsAction(cfg.ik_cfg, env)

        # Frames and sensors.
        self._ee_frame: FrameTransformer = env.scene[cfg.ee_frame_name]
        self._left_frame: FrameTransformer = env.scene[cfg.left_gripper_frame_name]
        self._right_frame: FrameTransformer = env.scene[cfg.right_gripper_frame_name]
        # InteractiveScene does not implement dict.get; mirror the check used in observations.
        self._contact_sensor = (
            env.scene[cfg.contact_sensor_name]
            if cfg.contact_sensor_name in env.scene.keys()
            and env.scene[cfg.contact_sensor_name] is not None
            else None
        )

        # Resolve the gripper joints (parallel gripper, two fingers).
        if not hasattr(env.cfg, "gripper_joint_names"):
            raise RuntimeError(
                "[ForcePositionAction] env.cfg is missing gripper_joint_names, so the "
                "aperture cannot be updated from the force error."
            )
        self._gripper_joint_ids, self._gripper_joint_names = self._robot.find_joints(
            env.cfg.gripper_joint_names
        )
        if len(self._gripper_joint_ids) != 2:
            raise RuntimeError(
                f"[ForcePositionAction] expected 2 finger joints for a parallel gripper, "
                f"resolved {len(self._gripper_joint_ids)}."
            )

        # raw / processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self._device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # Cache the split target quantities.
        self._eef_pos_cmd = torch.zeros(self.num_envs, 3, device=self._device)
        self._eef_aa_cmd = torch.zeros(self.num_envs, 3, device=self._device)
        self._gripper_abs_cmd = torch.zeros(self.num_envs, 1, device=self._device)
        self._fL_target_local = torch.zeros(self.num_envs, 3, device=self._device)
        self._fR_target_local = torch.zeros(self.num_envs, 3, device=self._device)

        # EMA filter state for the measured squeeze force (scalar).
        self._f_sq_meas_ema = torch.zeros(self.num_envs, device=self._device)
        self._f_sq_meas_ema_initialized = False

        # Debug cache; visualisation only, never affects control.
        self._debug: dict[str, torch.Tensor] = {}
        self._last_d_cmd = torch.zeros(self.num_envs, device=self._device)

    # --------------------------------------------------------------------- #
    # Properties
    # --------------------------------------------------------------------- #

    @property
    def action_dim(self) -> int:
        # 6 (eef pose: pos+axis-angle) + 1 (gripper) + 3 (left force) + 3 (right force)
        return 13

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        # Plain cache; the control logic lives in apply_actions.
        return self._processed_actions

    @property
    def debug_info(self) -> dict:
        """Debug information for the current step (env 0), for visualisation."""
        if not self._debug:
            return {}
        out: dict[str, object] = {}
        for k, v in self._debug.items():
            if isinstance(v, torch.Tensor):
                # Take env 0 only and move to CPU/numpy so matplotlib can consume it.
                out[k] = v[0].detach().cpu().numpy()
            else:
                out[k] = v
        return out

    @property
    def last_d_cmd(self) -> torch.Tensor:
        """Most recently computed gripper aperture target d_cmd (one scalar per env)."""
        return self._last_d_cmd

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _split_squeeze_and_applied_from_lr_local(
        fL_local: torch.Tensor,
        fR_local: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the per-finger local 3-D forces into a squeeze scalar and an applied-force
        3-D vector (still in the local frame).

        Conventions (shared with the LIBERO hybrid force-position and tactile environments):
        - The left and right finger-local frames satisfy:
          - x axis: points along world "up" for both fingers;
          - y axis: points along world "forward" for both fingers;
          - z axis: points along the gripper closing direction (+z agrees on both fingers).
        - The sensor measures the contact force applied *by the environment on the fingertip*.

        Definitions:
        - Squeeze scalar:
              f_sq = 2 * min(|fL_z|, |fR_z|)
          Physically, the common magnitude of the paired opposing/aligned force along the
          squeeze direction.
        - Applied force: the local-frame 3-D vector F_app = (Fx, Fy, Fz), where:
          - Fx, Fy are the per-finger components summed directly, signs preserved:
                Fx = fL_x + fR_x
                Fy = fL_y + fR_y
          - Fz removes the paired squeeze contribution along z, keeping only the residual
            load the two fingers do not cancel:
                a = fL_z, b = fR_z
                common = min(|a|, |b|)
                Fz = a + b - common * (sign(a) + sign(b))
          Consequently:
          - For pure opposing squeeze (a ~= -b), sign(a)+sign(b) ~= 0 and Fz ~= a+b ~= 0, so
            the squeeze appears only in f_sq;
          - For aligned loading (a and b share a sign), the common part is removed and only
            the difference remains, i.e. the net excess load.
        """
        # 3-D force components in the finger-local frame.
        fL_x, fL_y, fL_z = fL_local[:, 0], fL_local[:, 1], fL_local[:, 2]
        fR_x, fR_y, fR_z = fR_local[:, 0], fR_local[:, 1], fR_local[:, 2]

        # Squeeze: the smaller |z| component of the two fingers, doubled to represent the pair.
        abs_fL_z = torch.abs(fL_z)
        abs_fR_z = torch.abs(fR_z)
        squeeze = 2.0 * torch.minimum(abs_fL_z, abs_fR_z)  # (N,)

        # Applied force x/y: direct sum.
        Fx = fL_x + fR_x
        Fy = fL_y + fR_y

        # Applied force z: subtract the paired squeeze part, keep the uncancelled residual.
        signL = torch.sign(fL_z)
        signR = torch.sign(fR_z)
        common = torch.minimum(abs_fL_z, abs_fR_z)
        Fz = fL_z + fR_z - common * (signL + signR)

        F_app_local = torch.stack([Fx, Fy, Fz], dim=-1)  # (N, 3)
        return squeeze, F_app_local

    # --------------------------------------------------------------------- #
    # Core logic
    # --------------------------------------------------------------------- #

    def process_actions(self, actions: torch.Tensor):
        """Cache the raw action and split it into EEF pose (axis-angle), gripper and
        per-finger target forces."""
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions

        # 0:3 -> position, 3:6 -> axis-angle, 6:7 -> gripper, 7:10 and 10:13 -> left/right
        # finger target force.
        self._eef_pos_cmd[:] = actions[:, 0:3]
        self._eef_aa_cmd[:] = actions[:, 3:6]
        self._gripper_abs_cmd[:] = actions[:, 6:7]
        self._fL_target_local[:] = actions[:, 7:10]
        self._fR_target_local[:] = actions[:, 10:13]

    def apply_actions(self):
        """Called every simulation step: update the gripper aperture and run the internal DiffIK."""
        # ------------------------------
        # 1) Per-finger measured force (local frame).
        # ------------------------------
        if self._contact_sensor is not None:
            # Reuse the observation helper to rotate world-frame force into the finger frame.
            force_hist_local = contact_force_in_gripper_frame(
                self._env,
                contact_sensor_name=self.cfg.contact_sensor_name,
                history_length=self.cfg.history_length,
            )  # (N, H, 2, 3)
            # Take the most recent frame.
            force_curr_local = force_hist_local[:, -1, :, :]  # (N, 2, 3)
            fL_meas_local_raw = force_curr_local[:, 0, :]  # (N, 3)
            fR_meas_local_raw = force_curr_local[:, 1, :]  # (N, 3)
        else:
            fL_meas_local_raw = torch.zeros_like(self._fL_target_local)
            fR_meas_local_raw = torch.zeros_like(self._fR_target_local)

        # NOTE: the per-finger 3-D forces are not filtered; only the squeeze scalar is.
        fL_meas_local = fL_meas_local_raw
        fR_meas_local = fR_meas_local_raw

        # ------------------------------
        # 2) Compute squeeze and applied force in the finger-local frame.
        # ------------------------------
        f_sq_meas_raw, F_app_meas_local = self._split_squeeze_and_applied_from_lr_local(fL_meas_local, fR_meas_local)
        f_sq_target, F_app_target_local = self._split_squeeze_and_applied_from_lr_local(
            self._fL_target_local, self._fR_target_local
        )

        # Optional: EMA on the measured squeeze scalar only, to suppress the high-frequency
        # chatter introduced by min() switching between fingers.
        alpha = float(getattr(self.cfg, "meas_force_filter_alpha", 1.0))
        if 0.0 < alpha < 1.0:
            if not self._f_sq_meas_ema_initialized:
                self._f_sq_meas_ema[:] = f_sq_meas_raw
                self._f_sq_meas_ema_initialized = True
            else:
                self._f_sq_meas_ema.mul_(1.0 - alpha).add_(f_sq_meas_raw, alpha=alpha)
            f_sq_meas = self._f_sq_meas_ema
        else:
            f_sq_meas = f_sq_meas_raw

        # Squeeze feed-forward compensation (anti-slip: grip harder under load), driven by
        # the net z-axis applied load.
        f_sq_target_eff = f_sq_target
        if self.cfg.squeeze_ff_k_load_z != 0.0:
            if self.cfg.squeeze_ff_contact_threshold > 0.0:
                enable_ff = f_sq_meas_raw >= float(self.cfg.squeeze_ff_contact_threshold)
            else:
                enable_ff = torch.ones_like(f_sq_meas_raw, dtype=torch.bool)
            ff = float(self.cfg.squeeze_ff_k_load_z) * torch.abs(f_sq_target)
            f_sq_target_eff = torch.where(enable_ff, f_sq_target + ff, f_sq_target)

        # ------------------------------
        # 3) Applied force: finger-local -> world -> base, for the outer position loop.
        # ------------------------------
        # Use the left-finger frame as the representative grasp frame; the cfg already aligns
        # the two finger axes.
        left_quat_w = self._left_frame.data.target_quat_w[:, 0, :]  # (N, 4)
        F_app_target_w = math_utils.quat_apply(left_quat_w, F_app_target_local)
        F_app_meas_w = math_utils.quat_apply(left_quat_w, F_app_meas_local)

        # World -> base.
        root_quat_w = self._robot.data.root_quat_w  # (N,4)
        F_app_pred_b = math_utils.quat_apply_inverse(root_quat_w, F_app_target_w)
        F_app_meas_b = math_utils.quat_apply_inverse(root_quat_w, F_app_meas_w)

        # ------------------------------
        # 4) Squeeze force -> update the aperture from the predicted aperture plus force error.
        # ------------------------------
        # The recorded absolute gripper value is treated as the predicted aperture d_pred.
        # d_pred is needed for the debug fields even when squeeze_kp == 0, so it is computed
        # unconditionally to avoid an UnboundLocalError.
        d_pred = self._gripper_abs_cmd.squeeze(-1)  # (N,)

        if self.cfg.squeeze_kp != 0.0:
            # (predicted - measured) keeps the sign of the gain intuitive to tune.
            # squeeze is now 2*min(|fL_z|,|fR_z|), which doubles the error, so scale by 0.5
            # to preserve the effective control magnitude.
            delta_f_sq = 0.5 * (f_sq_target_eff - f_sq_meas)  # (N,)  = f_pred - f_actual

            # Dead zone on the position correction: compute delta_d = squeeze_kp * delta_f
            # first, then decide from |delta_d| whether to apply the correction.
            if self.cfg.squeeze_deadzone > 0.0:
                delta_d = self.cfg.squeeze_kp * delta_f_sq
                dz = abs(float(self.cfg.squeeze_kp)) * float(self.cfg.squeeze_deadzone)
                use_correction = torch.abs(delta_d) >= dz
                d_cmd = d_pred - delta_d
                d_cmd = torch.where(use_correction, d_cmd, d_pred)
            else:
                # Without a dead zone, apply the continuous incremental update directly.
                d_cmd = d_pred - self.cfg.squeeze_kp * delta_f_sq

            # Simple saturation against the env.cfg open/close range when available.
            d_min = torch.zeros_like(d_cmd)
            d_max = torch.full_like(d_cmd, getattr(self._env.cfg, "gripper_open_val", 0.04))
            d_cmd = torch.clamp(d_cmd, d_min, d_max)

            # Command both finger joints; a parallel gripper drives them identically.
            d_cmd_two = torch.stack([d_cmd, d_cmd], dim=-1)  # (N,2)
            self._robot.set_joint_position_target(d_cmd_two, joint_ids=self._gripper_joint_ids)

            # Record the latest d_cmd for debug visualisation.
            self._last_d_cmd = d_cmd.detach().clone()
        else:
            # Squeeze correction disabled: send the predicted absolute gripper value straight
            # through (pure position-controlled gripper).
            d_min = torch.zeros_like(d_pred)
            d_max = torch.full_like(d_pred, getattr(self._env.cfg, "gripper_open_val", 0.04))
            d_cmd = torch.clamp(d_pred, d_min, d_max)

            d_cmd_two = torch.stack([d_cmd, d_cmd], dim=-1)  # (N,2)
            self._robot.set_joint_position_target(d_cmd_two, joint_ids=self._gripper_joint_ids)

            # Record the latest d_cmd for debug visualisation.
            self._last_d_cmd = d_cmd.detach().clone()

        # ------------------------------
        # 5) Build and dispatch the internal DiffIK action (absolute pose).
        # ------------------------------
        # Outer position loop: P_pos_hybrid = P_pos_target + K_pos * (F_app_target^b - F_app_measured^b)
        if isinstance(self.cfg.pos_kp, tuple):
            k_vec = torch.tensor(self.cfg.pos_kp, device=self._device).view(1, 3)
        else:
            k_vec = torch.full((1, 3), float(self.cfg.pos_kp), device=self._device)
        K_pos = k_vec.expand(self.num_envs, -1)  # (N,3)
        F_err_b = F_app_pred_b - F_app_meas_b    # (N,3) = F_target - F_measured
        pos_hybrid = self._eef_pos_cmd + K_pos * F_err_b

        # Orientation passes through unchanged: P_axis_hybrid = P_axis_target.
        aa = self._eef_aa_cmd  # (N,3)
        angle = torch.linalg.vector_norm(aa, dim=-1, keepdim=True)  # (N,1)
        eps = 1e-6
        safe_axis = torch.zeros_like(aa)
        safe_axis[:, 0] = 1.0
        axis = torch.where(angle > eps, aa / angle, safe_axis)
        quat = math_utils.quat_from_angle_axis(angle.squeeze(-1), axis)  # (N,4)

        eef_pose_quat = torch.cat([pos_hybrid, quat], dim=-1)  # (N,7)

        ik_action_dim = self._ik_term.action_dim
        ik_actions = torch.zeros(self.num_envs, ik_action_dim, device=self._device)
        ik_actions[:, 0:7] = eef_pose_quat

        # Hand off to the internal DiffIK term, which reads the current EEF pose/velocity and
        # Jacobian itself.
        self._ik_term.process_actions(ik_actions)
        self._ik_term.apply_actions()

        # Applied-force magnitude in the base frame, for debug visualisation and statistics.
        F_app_norm_pred = torch.linalg.vector_norm(F_app_pred_b, dim=-1)  # (N,)
        F_app_norm_meas = torch.linalg.vector_norm(F_app_meas_b, dim=-1)  # (N,)

        # ------------------------------------------------------------------
        # 6) Refresh the debug cache (only env 0 is visualised).
        # ------------------------------------------------------------------
        # Measured aperture: read the current joint positions and average the two fingers (m).
        # NOTE: debug/visualisation only, never fed back into control.
        try:
            d_meas_two = self._robot.data.joint_pos[:, self._gripper_joint_ids]  # (N,2)
            d_meas = d_meas_two.mean(dim=-1)  # (N,)
        except Exception:
            d_meas = self._last_d_cmd.detach().clone()

        self._debug = {
            "fL_pred_local": self._fL_target_local.detach().clone(),
            "fR_pred_local": self._fR_target_local.detach().clone(),
            # meas: what the controller actually used (possibly EMA-filtered); raw is kept
            # alongside for comparison.
            "fL_meas_local": fL_meas_local.detach().clone(),
            "fR_meas_local": fR_meas_local.detach().clone(),
            "fL_meas_local_raw": fL_meas_local_raw.detach().clone(),
            "fR_meas_local_raw": fR_meas_local_raw.detach().clone(),
            # Applied-force vector and magnitude in the base frame; the legacy F_ext_* field
            # names are kept for compatibility.
            "F_app_pred_b": F_app_pred_b.detach().clone(),
            "F_app_meas_b": F_app_meas_b.detach().clone(),
            "F_app_norm_pred": F_app_norm_pred.detach().clone(),
            "F_app_norm_meas": F_app_norm_meas.detach().clone(),
            "F_ext_pred_b": F_app_pred_b.detach().clone(),
            "F_ext_meas_b": F_app_meas_b.detach().clone(),
            "f_sq_pred": f_sq_target.detach().clone(),
            "f_sq_meas": f_sq_meas.detach().clone(),
            "f_sq_meas_raw": f_sq_meas_raw.detach().clone(),
            "f_sq_pred_eff": f_sq_target_eff.detach().clone(),
            "d_pred": d_pred.detach().clone(),
            # Field name kept for force_position_debug_viz.py compatibility: d_actual is now
            # the measured joint position.
            "d_actual": d_meas.detach().clone(),
            # Also expose the aperture target the controller dispatched, for comparison.
            "d_cmd": self._last_d_cmd.detach().clone(),
            "eef_pos_pred": self._eef_pos_cmd.detach().clone(),
            # Outer-loop position offset after blending (visualisation only).
            "eef_pos_delta": (pos_hybrid - self._eef_pos_cmd).detach().clone(),
        }


# Point the cfg's class_type back at this action term so the manager can instantiate it.
ForcePositionActionCfg.class_type = ForcePositionAction
