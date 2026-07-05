"""Canonical SoftVTBench v1 deformable-body metric."""
from __future__ import annotations

import numpy as np

METRIC_ID = "fem_rms_rigid_aligned_bbox_pct_v1"


def _as_frames(points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    if value.ndim == 2:
        value = value[None, ...]
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"nodal positions must have shape (T,N,3) or (N,3), got {value.shape}")
    if value.shape[1] < 3:
        raise ValueError("at least three FEM nodes are required")
    if not np.isfinite(value).all():
        raise ValueError("nodal positions contain non-finite values")
    return value


def deformation_series(reference: np.ndarray, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return rigid-aligned RMS/max deformation as bbox-diagonal percentages.

    Translation is removed by centering every frame. Rotation is removed with a
    proper Kabsch alignment from the current centered nodes to the reference
    centered nodes. Reflection is explicitly disallowed.
    """

    ref = _as_frames(reference)[0]
    cur = _as_frames(frames)
    if cur.shape[1:] != ref.shape:
        raise ValueError(f"reference/frame shape mismatch: {ref.shape} vs {cur.shape[1:]}")

    ref_centered = ref - ref.mean(axis=0, keepdims=True)
    bbox_diag = float(np.linalg.norm(ref.max(axis=0) - ref.min(axis=0)))
    if not np.isfinite(bbox_diag) or bbox_diag <= 1.0e-12:
        raise ValueError(f"invalid reference bbox diagonal: {bbox_diag}")

    rms = np.empty(cur.shape[0], dtype=np.float64)
    max_v = np.empty(cur.shape[0], dtype=np.float64)
    for index, frame in enumerate(cur):
        frame_centered = frame - frame.mean(axis=0, keepdims=True)
        covariance = frame_centered.T @ ref_centered
        u, _, vh = np.linalg.svd(covariance, full_matrices=False)
        correction = np.eye(3)
        correction[-1, -1] = np.sign(np.linalg.det(u @ vh)) or 1.0
        rotation = u @ correction @ vh
        aligned = frame_centered @ rotation
        displacement = np.linalg.norm(aligned - ref_centered, axis=-1)
        rms[index] = 100.0 * np.sqrt(np.mean(displacement * displacement)) / bbox_diag
        max_v[index] = 100.0 * displacement.max() / bbox_diag
    return rms.astype(np.float32), max_v.astype(np.float32)
