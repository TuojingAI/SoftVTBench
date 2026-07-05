from __future__ import annotations

"""
通用的力学指标（metrics）工具，用于基于 13 维混合力–位动作向量
统计夹爪挤压力（squeeze force）和加持力（applied force，旧称 external force）。

设计目标：
- **统一数据格式**：对齐 `AbsEEFPoseAxisAngleAbsGripperWithForceActionStateRecorder` /
  `ForcePositionAction` 的 13D 动作约定：

    - 0:3  -> EEF 位置 (x, y, z)
    - 3:6  -> EEF 朝向轴角 (ax, ay, az)
    - 6:7  -> 夹爪标量（绝对值或二值，取决于控制器）
    - 7:10 -> 左指局部坐标系力 (fx, fy, fz)
    - 10:13-> 右指局部坐标系力 (fx, fy, fz)

- **统一挤压力与外力定义**：参考 `ForcePositionAction`：

    - 挤压力标量：`f_sq = 2 * min(|fL_z|, |fR_z|)`
      （左右两指在 *各自局部坐标系* 下 z 轴分量绝对值的较小者，乘 2 视为两指合计）
    - 对外合力：`F_ext = fL + fR`
      （本工具默认在 finger 局部系相加；若需要严格的 base frame，
       请在上层先做坐标变换再调用 `from_lr_forces` 系列接口）

- **统一 contact 过滤逻辑**：按「力非零即视为接触」的规则，仅在
  `||fL|| + ||fR|| > eps` 的时间步上统计挤压力 / 外力。

典型使用场景：
- SoftVTBench / SoftVTBench-force / pi0 风格 replayed_demos HDF5 中，取出动作轨迹 (T, 13)
  后离线统计一条轨迹或一组轨迹的挤压力和外力分布；
- OpenPI inference hybrid 模式下，收集一条 episode 的 13D 动作或指尖力，
  调用本模块获得与数据集完全一致定义的 squeeze / external metrics。
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class _ArrayLike(Protocol):
    """兼容 numpy / torch / list 的最小 array-like 协议."""

    def __array__(self, dtype=None):  # pragma: no cover - duck typing only
        ...


@dataclass
class ContactForceSeries:
    """逐时间步的力学量（不做任何聚合），便于后续可视化或自定义统计."""

    # 每个时间步的挤压力标量 f_sq = 2 * min(|fL_z|, |fR_z|)，shape: (T,)
    squeeze: np.ndarray

    # 每个时间步的加持力向量（兼容字段名 external），shape: (T, 3)
    external: np.ndarray

    # 每个时间步的加持力模长 ||F_app||_2（兼容字段名 external_norm），shape: (T,)
    external_norm: np.ndarray

    # 接触 mask：True 表示该时间步存在接触（力非零），shape: (T,)
    contact_mask: np.ndarray


@dataclass
class ContactForceMetrics:
    """基于接触时间步（contact_mask == True）的聚合统计量."""

    # 基本计数
    num_steps: int
    num_contact_steps: int
    contact_ratio: float  # num_contact_steps / num_steps

    # 挤压力标量统计（仅在接触时间步上）
    squeeze_mean: float
    squeeze_max: float
    squeeze_p95: float
    squeeze_sum: float

    # 加持力模长统计（仅在接触时间步上）
    external_norm_mean: float
    external_norm_max: float
    external_norm_p95: float
    external_norm_sum: float


# -----------------------------------------------------------------------------
# 低层：从 13D 动作 / 左右指 3D 力中拆解挤压力与外力
# -----------------------------------------------------------------------------


def _to_numpy(x: _ArrayLike) -> np.ndarray:
    """将各种 array-like（包括 torch.Tensor）转换为 numpy.ndarray."""

    if isinstance(x, np.ndarray):
        return x
    # 尝试走 __array__ 协议（torch.Tensor / list / tuple 等都能覆盖）
    return np.asarray(x)


def split_lr_forces_from_13d(actions_13d: _ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """从 13D 混合力–位动作中拆出左/右指力（局部坐标系），shape: (T, 3).

    Args:
        actions_13d: 形状为 (T, 13) 的动作数组（或可转换为该形状的 array-like）。

    Returns:
        fL_local, fR_local: 均为 (T, 3) 的 numpy 数组。
    """
    arr = _to_numpy(actions_13d)
    if arr.ndim != 2 or arr.shape[1] != 13:
        raise ValueError(f"expect actions with shape (T, 13), got {arr.shape}")

    forces = arr[:, 7:]  # (T, 6)
    fL = forces[:, 0:3]
    fR = forces[:, 3:6]
    return fL, fR


def compute_contact_force_series_from_lr_forces(
    fL: _ArrayLike,
    fR: _ArrayLike,
    contact_eps: float = 1e-6,
) -> ContactForceSeries:
    """基于左右指 3D 力（局部或统一坐标系）计算逐时间步的挤压力与加持力.

    该函数假定输入满足 LIBERO Hybrid（force-position）环境中对 finger 局部系的约定（x 向上、
    y 向前、z 为夹爪闭合方向），从而使用与控制器一致的物理定义：

    - 挤压力：`f_sq = 2 * min(|fL_z|, |fR_z|)`；
    - 加持力向量（字段名 external）：

          Fx = fL_x + fR_x
          Fy = fL_y + fR_y
          a, b = fL_z, fR_z
          common = min(|a|, |b|)
          Fz = a + b - common * (sign(a) + sign(b))
          F_app = (Fx, Fy, Fz)

    Args:
        fL: 左指力，shape: (T, 3) 或可转换为该形状的 array-like。
        fR: 右指力，shape: (T, 3) 或可转换为该形状的 array-like。
        contact_eps: 接触阈值，当 `||fL|| + ||fR|| > contact_eps` 时视为存在接触。

    Returns:
        ContactForceSeries: 包含 squeeze / external / external_norm / contact_mask。
    """
    fL_np = _to_numpy(fL)
    fR_np = _to_numpy(fR)

    if fL_np.shape != fR_np.shape:
        raise ValueError(f"fL and fR must have same shape, got {fL_np.shape} vs {fR_np.shape}")
    if fL_np.ndim != 2 or fL_np.shape[1] != 3:
        raise ValueError(f"expect fL/fR with shape (T, 3), got {fL_np.shape}")

    # 分量
    fL_x, fL_y, fL_z = fL_np[:, 0], fL_np[:, 1], fL_np[:, 2]
    fR_x, fR_y, fR_z = fR_np[:, 0], fR_np[:, 1], fR_np[:, 2]

    # 挤压力标量：左右指 z 轴分量绝对值的较小者（乘 2 视为两指合计）
    abs_fL_z = np.abs(fL_z)
    abs_fR_z = np.abs(fR_z)
    squeeze = 2.0 * np.minimum(abs_fL_z, abs_fR_z)  # (T,)

    # 加持力 x/y：直接相加
    Fx = fL_x + fR_x
    Fy = fL_y + fR_y

    # 加持力 z：减掉「成对出现」的挤压力部分，只保留不被抵消的剩余
    signL = np.sign(fL_z)
    signR = np.sign(fR_z)
    common = np.minimum(abs_fL_z, abs_fR_z)
    Fz = fL_z + fR_z - common * (signL + signR)

    external = np.stack([Fx, Fy, Fz], axis=-1)  # (T, 3)，语义为 applied force
    external_norm = np.linalg.norm(external, axis=-1)  # (T,)

    # 接触 mask：任一指头的力非零即视为接触
    norm_L = np.linalg.norm(fL_np, axis=-1)
    norm_R = np.linalg.norm(fR_np, axis=-1)
    contact_mask = (norm_L + norm_R) > float(contact_eps)

    return ContactForceSeries(
        squeeze=squeeze,
        external=external,
        external_norm=external_norm,
        contact_mask=contact_mask,
    )


def compute_contact_force_series_from_13d(
    actions_13d: _ArrayLike,
    contact_eps: float = 1e-6,
) -> ContactForceSeries:
    """从 13D 混合力–位动作向量直接计算逐时间步力学量.

    这是最常用接口，适用于：
    - replayed_demos HDF5 中直接存的 13D 动作；
    - OpenPI hybrid 控制客户端侧缓存的 13D 动作序列；
    - 任意与 `ForcePositionAction` 对齐的数据。
    """
    fL, fR = split_lr_forces_from_13d(actions_13d)
    return compute_contact_force_series_from_lr_forces(fL, fR, contact_eps=contact_eps)


# -----------------------------------------------------------------------------
# 聚合：在「有接触」的时间步上统计挤压力 / 外力分布
# -----------------------------------------------------------------------------


def summarize_contact_force(series: ContactForceSeries) -> ContactForceMetrics:
    """在接触时间步上聚合计算挤压力与对外合力指标.

    约定：
    - 仅在 `series.contact_mask == True` 的时间步上做统计；
    - 若整条轨迹从未接触（全 False），则返回的统计量为 0，同时 contact_ratio 为 0。
    """
    squeeze = np.asarray(series.squeeze)
    ext_norm = np.asarray(series.external_norm)
    contact_mask = np.asarray(series.contact_mask).astype(bool)

    if squeeze.ndim != 1 or ext_norm.ndim != 1 or squeeze.shape != ext_norm.shape:
        raise ValueError("squeeze and external_norm must be 1D arrays with same shape.")

    num_steps = squeeze.shape[0]
    if num_steps == 0:
        return ContactForceMetrics(
            num_steps=0,
            num_contact_steps=0,
            contact_ratio=0.0,
            squeeze_mean=0.0,
            squeeze_max=0.0,
            squeeze_p95=0.0,
            squeeze_sum=0.0,
            external_norm_mean=0.0,
            external_norm_max=0.0,
            external_norm_p95=0.0,
            external_norm_sum=0.0,
        )

    contact_mask = contact_mask[:num_steps]
    num_contact = int(contact_mask.sum())

    if num_contact == 0:
        # 无接触：返回 0，但保留总步数信息
        return ContactForceMetrics(
            num_steps=num_steps,
            num_contact_steps=0,
            contact_ratio=0.0,
            squeeze_mean=0.0,
            squeeze_max=0.0,
            squeeze_p95=0.0,
            squeeze_sum=0.0,
            external_norm_mean=0.0,
            external_norm_max=0.0,
            external_norm_p95=0.0,
            external_norm_sum=0.0,
        )

    squeeze_c = squeeze[contact_mask]
    ext_norm_c = ext_norm[contact_mask]

    def _percentile(x: np.ndarray, q: float) -> float:
        return float(np.percentile(x, q)) if x.size > 0 else 0.0

    return ContactForceMetrics(
        num_steps=num_steps,
        num_contact_steps=num_contact,
        contact_ratio=float(num_contact) / float(num_steps),
        squeeze_mean=float(squeeze_c.mean()),
        # 「最大挤压力」改为：在一次 demo 中，挤压力非零的帧里取最大的 Top5% 帧的平均值
        squeeze_max=compute_topk_mean(squeeze_c, frac=0.05),
        squeeze_p95=_percentile(squeeze_c, 95.0),
        squeeze_sum=float(squeeze_c.sum()),
        external_norm_mean=float(ext_norm_c.mean()),
        # 「最大加持力」同样改为 Top5% 帧的平均值
        external_norm_max=compute_topk_mean(ext_norm_c, frac=0.05),
        external_norm_p95=_percentile(ext_norm_c, 95.0),
        external_norm_sum=float(ext_norm_c.sum()),
    )


def compute_topk_mean(values: _ArrayLike, frac: float = 0.05) -> float:
    """对任意 array-like 计算「最大 Top-K(=ceil(frac*N))」的平均值（仅统计 >0 的样本）."""

    arr = _to_numpy(values)
    if arr.size == 0:
        return 0.0

    # 只在「力非零」的样本上统计
    arr_nz = arr[arr > 0]
    if arr_nz.size == 0:
        return 0.0

    k = int(np.ceil(float(frac) * float(arr_nz.size)))
    k = max(k, 1)
    arr_sorted = np.sort(arr_nz)[::-1]
    topk = arr_sorted[:k]
    return float(topk.mean())


# -----------------------------------------------------------------------------
# 便捷高层接口：一行得到 metrics（面向离线评估脚本 / OpenPI client 使用）
# -----------------------------------------------------------------------------


def compute_contact_force_metrics_from_13d(
    actions_13d: _ArrayLike,
    contact_eps: float = 1e-6,
) -> ContactForceMetrics:
    """从 13D 动作序列直接得到挤压力 / 外力的接触统计指标."""
    series = compute_contact_force_series_from_13d(actions_13d, contact_eps=contact_eps)
    return summarize_contact_force(series)


def compute_contact_force_metrics_from_lr_forces(
    fL: _ArrayLike,
    fR: _ArrayLike,
    contact_eps: float = 1e-6,
) -> ContactForceMetrics:
    """从左右指 3D 力序列直接得到挤压力 / 外力的接触统计指标."""
    series = compute_contact_force_series_from_lr_forces(fL, fR, contact_eps=contact_eps)
    return summarize_contact_force(series)


__all__ = [
    "ContactForceSeries",
    "ContactForceMetrics",
    "split_lr_forces_from_13d",
    "compute_contact_force_series_from_lr_forces",
    "compute_contact_force_series_from_13d",
    "summarize_contact_force",
    "compute_contact_force_metrics_from_13d",
    "compute_contact_force_metrics_from_lr_forces",
    "compute_topk_mean",
]


