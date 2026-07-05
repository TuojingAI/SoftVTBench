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
    """混合力–位控制 Action 的配置（指令 13 维：6D EEF 位姿(轴角) + 1D 夹爪 + 左右指各 3D 目标力）。

    该 ActionTerm 本身不直接暴露给策略使用 IK 的细节，而是：

    - 内部包装一个 IsaacLab 自带的 `DifferentialInverseKinematicsAction`（只控制机械臂关节）；
    - 使用左右手指的 3D 目标力 + 传感器测得的当前力：
      - 合成对外的 EEF wrench，作为 OSC 的 `wrench_abs` 输入；
      - 计算挤压内力误差，用增量式公式更新夹爪开合度：
        `d_cmd = d_curr + squeeze_kp * (f_sq_curr - f_sq_target)`。

    对齐数据录制类 `AbsEEFPoseAxisAngleAbsGripperWithForceActionStateRecorder` 的约定，
    输入动作的 13 维含义为：

    - 0:6  ->  EEF 绝对位姿，基座坐标系下的 (x, y, z, ax, ay, az)，其中 (ax, ay, az) 为轴角
    - 6:7  ->  绝对 gripper 开合度（当前实现中不直接使用，夹爪由力闭环增量式控制）
    - 7:10 ->  左指目标力（指头局部坐标系，fx, fy, fz）
    - 10:13->  右指目标力（指头局部坐标系，fx, fy, fz）

    内部会把 (x, y, z, ax, ay, az) 转成 (x, y, z, qw, qx, qy, qz) 喂给 DiffIK 的绝对位姿命令，
    并根据左右指目标力与实际指尖力构造成混合位姿：

        P_pos_hybrid = P_pos_target + K_pos * (F_target^b - F_measured^b)
    """

    # 嵌套的 IK action 配置，由 env cfg 填好（asset_name/joint_names/body_name/offset 等）
    ik_cfg: DifferentialInverseKinematicsActionCfg = MISSING

    # 传感器和坐标系名称（需在 env cfg 里已经创建）
    ee_frame_name: str = "ee_frame"
    left_gripper_frame_name: str = "left_gripper_frame"
    right_gripper_frame_name: str = "right_gripper_frame"
    contact_sensor_name: str = "contact_gripper"
    history_length: int = 1

    # 增益
    # 位置混合增益：可以是标量，表示 xyz 共用一个 K，也可以是 (kx, ky, kz)
    pos_kp: float | tuple[float, float, float] = 0.0
    squeeze_kp: float = 0.001  # 夹爪开合度的力误差增益（squeeze 定义×2 后，在控制律中对误差做 0.5 缩放以保持等效幅度）
    squeeze_deadzone: float = 0.1  # 挤压力误差死区（等效）：按位置修正量 |Δd| 判断，阈值取 |squeeze_kp|*squeeze_deadzone（兼容旧配置）

    # 实测挤压力滤波（仅用于夹爪 squeeze 闭环，不影响 obs/数据录制，也不滤波左右指 3D 力）
    # EMA: s_filt = alpha * s_curr + (1-alpha) * s_filt_prev
    # - alpha=1.0: 不滤波（默认保持原行为）
    # - alpha 越小：滤波越强（响应越慢）
    meas_force_filter_alpha: float = 0.2

    # squeeze 前馈补偿（用于防滑/随“预测挤压力”增压）：把目标挤压力提升为
    #   f_sq_target_eff = f_sq_target + squeeze_ff_k_load_z * f_sq_target
    # 等价于：f_sq_target_eff = (1 + squeeze_ff_k_load_z) * f_sq_target
    # 默认系数为 0，不改变现有行为。
    squeeze_ff_k_load_z: float = 0.9
    squeeze_ff_contact_threshold: float = 1.0  # >0 时，仅当 f_sq_meas_raw >= threshold 才启用前馈

    # 由 manager 使用的 ActionTerm 类型（提供一个非 MISSING 的默认值，后续在模块末尾覆盖）
    class_type: type[ActionTerm] = ActionTerm


class ForcePositionAction(ActionTerm):
    """基于 OSC 的力–位混合控制 ActionTerm。

    输入 action: (N, 13)
        - 0:6  ->  EEF 绝对位姿 (x, y, z, ax, ay, az) 基座坐标系下，旋转为轴角
        - 6:7  ->  绝对 gripper 值（当前实现中仅对齐维度，不直接控制）
        - 7:10 ->  左指目标力（指头局部坐标系，fx, fy, fz）
        - 10:13->  右指目标力（指头局部坐标系，fx, fy, fz）

    作用：
    - 利用左右指目标力：
        - 合成对外 EEF wrench（在 base frame 下），喂给内部的 OSC (`wrench_abs`);
        - 通过当前测得的挤压力，增量式更新夹爪开合度：
          `d_cmd = d_curr + squeeze_kp * (f_sq_curr - f_sq_target)`。
    - EEF 位姿部分目前直接作为 OSC 的 `pose_abs` 目标（可在 cfg.eef_kp > 0 时加力反馈外环）。
    """

    cfg: ForcePositionActionCfg

    def __init__(self, cfg: ForcePositionActionCfg, env: ManagerBasedRLEnv) -> None:
        # 初始化基类（解析 asset_name -> robot articulation）
        super().__init__(cfg, env)

        self._env: ManagerBasedRLEnv = env
        self._device = env.device

        # 机器人
        self._robot: Articulation = self._asset

        # 内部 DiffIK ActionTerm（只控制机械臂关节）
        self._ik_term = DifferentialInverseKinematicsAction(cfg.ik_cfg, env)

        # 帧 & 传感器
        self._ee_frame: FrameTransformer = env.scene[cfg.ee_frame_name]
        self._left_frame: FrameTransformer = env.scene[cfg.left_gripper_frame_name]
        self._right_frame: FrameTransformer = env.scene[cfg.right_gripper_frame_name]
        # InteractiveScene 不实现 dict.get，用与 observations 相同的检查逻辑
        self._contact_sensor = (
            env.scene[cfg.contact_sensor_name]
            if cfg.contact_sensor_name in env.scene.keys()
            and env.scene[cfg.contact_sensor_name] is not None
            else None
        )

        # 解析夹爪关节（平行夹爪，两个 finger）
        if not hasattr(env.cfg, "gripper_joint_names"):
            raise RuntimeError(
                "[ForcePositionAction] env.cfg 中缺少 gripper_joint_names，"
                "无法根据力误差更新夹爪开合度。"
            )
        self._gripper_joint_ids, self._gripper_joint_names = self._robot.find_joints(
            env.cfg.gripper_joint_names
        )
        if len(self._gripper_joint_ids) != 2:
            raise RuntimeError(
                f"[ForcePositionAction] 期望平行夹爪有 2 个 finger 关节，实际解析到 {len(self._gripper_joint_ids)} 个。"
            )

        # raw / processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self._device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # 缓存拆分后的目标量
        self._eef_pos_cmd = torch.zeros(self.num_envs, 3, device=self._device)
        self._eef_aa_cmd = torch.zeros(self.num_envs, 3, device=self._device)
        self._gripper_abs_cmd = torch.zeros(self.num_envs, 1, device=self._device)
        self._fL_target_local = torch.zeros(self.num_envs, 3, device=self._device)
        self._fR_target_local = torch.zeros(self.num_envs, 3, device=self._device)

        # 实测挤压力 EMA 滤波状态（标量）
        self._f_sq_meas_ema = torch.zeros(self.num_envs, device=self._device)
        self._f_sq_meas_ema_initialized = False

        # 调试信息缓存（仅用于可视化，不影响控制逻辑）
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
        # 这里只是简单缓存，实际控制逻辑在 apply_actions 中完成
        return self._processed_actions

    @property
    def debug_info(self) -> dict:
        """返回当前 step 的调试信息（env0），用于可视化."""
        if not self._debug:
            return {}
        out: dict[str, object] = {}
        for k, v in self._debug.items():
            if isinstance(v, torch.Tensor):
                # 只取 env 0，并搬到 CPU/numpy，方便 matplotlib 使用
                out[k] = v[0].detach().cpu().numpy()
            else:
                out[k] = v
        return out

    @property
    def last_d_cmd(self) -> torch.Tensor:
        """最近一次计算得到的夹爪目标开合度 d_cmd（每 env 一个标量）."""
        return self._last_d_cmd

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _split_squeeze_and_applied_from_lr_local(
        fL_local: torch.Tensor,
        fR_local: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """基于左右指局部系 3D 力，计算挤压力标量和「加持力」3D 向量（局部系）.

        约定（与 LIBERO Hybrid（force-position）/ 触觉环境保持一致）：
        - 左右指局部坐标系满足：
          - x 轴：均指向世界系的「上」方向；
          - y 轴：均指向世界系的「前」方向；
          - z 轴：均指向夹爪闭合方向（两指 +z 一致）。
        - 传感器测得的是「环境对指尖」的接触力。

        定义：
        - 挤压力标量（squeeze）：
              f_sq = 2 * min(|fL_z|, |fR_z|)
          物理含义：两指在挤压方向上成对出现的那一部分对向/同向力的公共模长。
        - 加持力（applied force）：在局部系下的 3D 向量 F_app = (Fx, Fy, Fz)，其中：
          - Fx, Fy：左右指对应分量直接相加（保持正负号）：
                Fx = fL_x + fR_x
                Fy = fL_y + fR_y
          - Fz：在 z 轴上减去成对出现的「挤压力」部分，仅保留不被两指相互抵消的剩余载荷：
                a = fL_z, b = fR_z
                common = min(|a|, |b|)
                Fz = a + b - common * (sign(a) + sign(b))
          这样：
          - 若为纯对向挤压（a ≈ -b），则 sign(a)+sign(b)≈0，Fz≈a+b≈0，仅通过 f_sq 体现挤压力；
          - 若为同向加载（a, b 同号），则去掉公共部分，仅保留差值，体现「多出来」的净载荷。
        """
        # 3D force components（finger 局部坐标系）
        fL_x, fL_y, fL_z = fL_local[:, 0], fL_local[:, 1], fL_local[:, 2]
        fR_x, fR_y, fR_z = fR_local[:, 0], fR_local[:, 1], fR_local[:, 2]

        # 挤压力：左右指 z 轴分量绝对值的较小者（乘 2 视为两指合计）
        abs_fL_z = torch.abs(fL_z)
        abs_fR_z = torch.abs(fR_z)
        squeeze = 2.0 * torch.minimum(abs_fL_z, abs_fR_z)  # (N,)

        # 加持力 x/y：直接相加
        Fx = fL_x + fR_x
        Fy = fL_y + fR_y

        # 加持力 z：减掉「成对出现」的挤压力部分，只保留不被抵消的剩余
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
        """缓存原始 action，并拆分为 EEF 位姿(轴角)、夹爪和 finger 目标力。"""
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions

        # 0:3 -> 位置, 3:6 -> 轴角, 6:7 -> gripper, 7:10/10:13 -> 左/右指目标力
        self._eef_pos_cmd[:] = actions[:, 0:3]
        self._eef_aa_cmd[:] = actions[:, 3:6]
        self._gripper_abs_cmd[:] = actions[:, 6:7]
        self._fL_target_local[:] = actions[:, 7:10]
        self._fR_target_local[:] = actions[:, 10:13]

    def apply_actions(self):
        """在每个仿真步被调用：更新夹爪开合 + 调用内部 DiffIK。"""
        # ------------------------------
        # 1) finger 级当前力观测（局部系）
        # ------------------------------
        if self._contact_sensor is not None:
            # 使用已有的观测函数把世界系力转为 finger 局部系
            force_hist_local = contact_force_in_gripper_frame(
                self._env,
                contact_sensor_name=self.cfg.contact_sensor_name,
                history_length=self.cfg.history_length,
            )  # (N, H, 2, 3)
            # 取最近一帧
            force_curr_local = force_hist_local[:, -1, :, :]  # (N, 2, 3)
            fL_meas_local_raw = force_curr_local[:, 0, :]  # (N, 3)
            fR_meas_local_raw = force_curr_local[:, 1, :]  # (N, 3)
        else:
            fL_meas_local_raw = torch.zeros_like(self._fL_target_local)
            fR_meas_local_raw = torch.zeros_like(self._fR_target_local)

        # NOTE: 不对左右指 3D 力做滤波；只在后续对 squeeze 标量做滤波
        fL_meas_local = fL_meas_local_raw
        fR_meas_local = fR_meas_local_raw

        # ------------------------------
        # 2) 基于 finger 局部系计算挤压力与「加持力」
        # ------------------------------
        f_sq_meas_raw, F_app_meas_local = self._split_squeeze_and_applied_from_lr_local(fL_meas_local, fR_meas_local)
        f_sq_target, F_app_target_local = self._split_squeeze_and_applied_from_lr_local(
            self._fL_target_local, self._fR_target_local
        )

        # 可选：只对“实测挤压力标量”做 EMA，抑制 min() 切换导致的高频锯齿
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

        # squeeze 目标前馈补偿（用于防滑/随负载增压）：基于 applied force 的 z 轴净载荷
        f_sq_target_eff = f_sq_target
        if self.cfg.squeeze_ff_k_load_z != 0.0:
            if self.cfg.squeeze_ff_contact_threshold > 0.0:
                enable_ff = f_sq_meas_raw >= float(self.cfg.squeeze_ff_contact_threshold)
            else:
                enable_ff = torch.ones_like(f_sq_meas_raw, dtype=torch.bool)
            ff = float(self.cfg.squeeze_ff_k_load_z) * torch.abs(f_sq_target)
            f_sq_target_eff = torch.where(enable_ff, f_sq_target + ff, f_sq_target)

        # ------------------------------
        # 3) 「加持力」从 finger 局部系 -> 世界 -> base（用于位置外环混合）
        # ------------------------------
        # 使用左指局部系作为代表性抓取坐标系（两指局部轴向已在 cfg 中对齐）
        left_quat_w = self._left_frame.data.target_quat_w[:, 0, :]  # (N, 4)
        F_app_target_w = math_utils.quat_apply(left_quat_w, F_app_target_local)
        F_app_meas_w = math_utils.quat_apply(left_quat_w, F_app_meas_local)

        # 世界 -> base
        root_quat_w = self._robot.data.root_quat_w  # (N,4)
        F_app_pred_b = math_utils.quat_apply_inverse(root_quat_w, F_app_target_w)
        F_app_meas_b = math_utils.quat_apply_inverse(root_quat_w, F_app_meas_w)

        # ------------------------------
        # 4) finger 挤压力 -> 基于“预测开合度 + 力误差”更新夹爪开合度
        # ------------------------------
        # 录制时的 abs gripper 视为“预测开合度”：d_pred
        # 注意：即使 squeeze_kp == 0，我们也需要 d_pred 用于 debug 字段，避免 UnboundLocalError。
        d_pred = self._gripper_abs_cmd.squeeze(-1)  # (N,)

        if self.cfg.squeeze_kp != 0.0:
            # 注意这里使用 (预测值 - 测量值)，方便直观调节增益方向
            # squeeze 定义已变为 2*min(|fL_z|,|fR_z|)，误差会整体×2；这里乘 0.5 以保持等效控制幅度
            delta_f_sq = 0.5 * (f_sq_target_eff - f_sq_meas)  # (N,)  = f_pred - f_actual

            # 死区逻辑（位置调整死区）：先算位置修正量 Δd = squeeze_kp * Δf，再基于 |Δd| 判断是否启用修正。
            if self.cfg.squeeze_deadzone > 0.0:
                delta_d = self.cfg.squeeze_kp * delta_f_sq
                dz = abs(float(self.cfg.squeeze_kp)) * float(self.cfg.squeeze_deadzone)
                use_correction = torch.abs(delta_d) >= dz
                d_cmd = d_pred - delta_d
                d_cmd = torch.where(use_correction, d_cmd, d_pred)
            else:
                # 无死区时，直接使用连续增量式
                d_cmd = d_pred - self.cfg.squeeze_kp * delta_f_sq

            # 简单饱和：使用 env.cfg 的 open/close 范围（如果有）
            d_min = torch.zeros_like(d_cmd)
            d_max = torch.full_like(d_cmd, getattr(self._env.cfg, "gripper_open_val", 0.04))
            d_cmd = torch.clamp(d_cmd, d_min, d_max)

            # 设定两个 finger 关节的目标（平行夹爪，两个 finger 相同）
            d_cmd_two = torch.stack([d_cmd, d_cmd], dim=-1)  # (N,2)
            self._robot.set_joint_position_target(d_cmd_two, joint_ids=self._gripper_joint_ids)

            # 记录最近一次 d_cmd，供调试可视化使用
            self._last_d_cmd = d_cmd.detach().clone()
        else:
            # squeeze 修正关闭时：直接把预测的 abs gripper 当作目标下发（纯位置式夹爪）
            d_min = torch.zeros_like(d_pred)
            d_max = torch.full_like(d_pred, getattr(self._env.cfg, "gripper_open_val", 0.04))
            d_cmd = torch.clamp(d_pred, d_min, d_max)

            d_cmd_two = torch.stack([d_cmd, d_cmd], dim=-1)  # (N,2)
            self._robot.set_joint_position_target(d_cmd_two, joint_ids=self._gripper_joint_ids)

            # 记录最近一次 d_cmd，供调试可视化使用
            self._last_d_cmd = d_cmd.detach().clone()

        # ------------------------------
        # 5) 构造内部 DiffIK 的动作并调用（绝对位姿）
        # ------------------------------
        # 位置外环混合：P_pos_hybrid = P_pos_target + K_pos * (F_app_target^b - F_app_measured^b)
        if isinstance(self.cfg.pos_kp, tuple):
            k_vec = torch.tensor(self.cfg.pos_kp, device=self._device).view(1, 3)
        else:
            k_vec = torch.full((1, 3), float(self.cfg.pos_kp), device=self._device)
        K_pos = k_vec.expand(self.num_envs, -1)  # (N,3)
        F_err_b = F_app_pred_b - F_app_meas_b    # (N,3) = F_target - F_measured
        pos_hybrid = self._eef_pos_cmd + K_pos * F_err_b

        # 姿态部分保持不变：P_axis_hybrid = P_axis_target
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

        # 下发到内部 DiffIK term：它自己会读取当前 EEF pose/vel、Jacobian 等
        self._ik_term.process_actions(ik_actions)
        self._ik_term.apply_actions()

        # 「加持力」模长（在 base 系下），用于调试可视化 / 统计
        F_app_norm_pred = torch.linalg.vector_norm(F_app_pred_b, dim=-1)  # (N,)
        F_app_norm_meas = torch.linalg.vector_norm(F_app_meas_b, dim=-1)  # (N,)

        # ------------------------------------------------------------------
        # 6) 更新调试信息缓存（仅 env0 会被可视化使用）
        # ------------------------------------------------------------------
        # 夹爪“实际”开合度：读取当前关节位置（两指取平均，单位：m）
        # NOTE: 这不会影响控制，只用于 debug/可视化。
        try:
            d_meas_two = self._robot.data.joint_pos[:, self._gripper_joint_ids]  # (N,2)
            d_meas = d_meas_two.mean(dim=-1)  # (N,)
        except Exception:
            d_meas = self._last_d_cmd.detach().clone()

        self._debug = {
            "fL_pred_local": self._fL_target_local.detach().clone(),
            "fR_pred_local": self._fR_target_local.detach().clone(),
            # meas: 默认是控制器实际使用的（可能 EMA 后）；raw 额外保留一份方便对比
            "fL_meas_local": fL_meas_local.detach().clone(),
            "fR_meas_local": fR_meas_local.detach().clone(),
            "fL_meas_local_raw": fL_meas_local_raw.detach().clone(),
            "fR_meas_local_raw": fR_meas_local_raw.detach().clone(),
            # 「加持力」向量及其模长（在 base 系下）；同时兼容旧字段名 F_ext_*
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
            # 保持字段名兼容 force_position_debug_viz.py：d_actual 现在是“实测关节位置”
            "d_actual": d_meas.detach().clone(),
            # 额外提供控制器下发的目标开合度，便于对比
            "d_cmd": self._last_d_cmd.detach().clone(),
            "eef_pos_pred": self._eef_pos_cmd.detach().clone(),
            # 记录外环混合后的增量（仅用于可视化）
            "eef_pos_delta": (pos_hybrid - self._eef_pos_cmd).detach().clone(),
        }


# 把 cfg 的 class_type 指回本 ActionTerm，供 manager 创建
ForcePositionActionCfg.class_type = ForcePositionAction
