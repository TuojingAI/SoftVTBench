"""Quaternion (wxyz) <-> axis-angle (rotvec) -- the core of train/inference consistency.

Ground truth on the training side: the training pipeline's
raw_softvt_hdf5_to_act_dp.py::quat_wxyz_to_axis_angle. It performs branch
continuity "offline, statefully, over the whole trajectory":
  1) normalize + w>=0 principal-value canonicalization (flip sign when w<0)
  2) RotX(pi)-like detection (median |xyz| over the segment), then flip frames with q[1]<0 to +X
  3) per-frame temporal continuity: if dot(q[i-1], q[i]) < 0: q[i] *= -1
  4) to rotvec: angle = 2*atan2(|v|, w); out = v/|v| * angle

Evaluation is step-wise closed-loop and never sees the whole trajectory.
QuatBranchTracker replays 1) and 3) step by step using the "previously emitted
quaternion"; the RotX alignment of 2) is applied at the first frame after reset (rx ~= pi
always holds for this robot, so rot_x_pi_like defaults to True). The sealed trajectory
in ``tests/test_rotation.py`` locks this branch convention against accidental drift.

On the action side, axis_angle -> quat has no branch ambiguity (any rotvec maps to a
unique quat) and needs no state.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-10


def _canonical_wpos(q: np.ndarray) -> np.ndarray:
    """Normalize + w>=0 principal value. q: (4,) wxyz."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-8:
        raise ValueError("zero-norm quaternion")
    q = q / n
    if q[0] < 0:
        q = -q
    return q


def quat_wxyz_to_axis_angle(q_wxyz: np.ndarray) -> np.ndarray:
    """One canonicalized quaternion -> rotvec(3), consistent with training step 4."""
    v = q_wxyz[1:]
    vn = np.linalg.norm(v)
    if vn > _EPS:
        angle = 2.0 * np.arctan2(vn, q_wxyz[0])
        return (v / vn * angle).astype(np.float32)
    # near-zero rotation: training side uses the small-angle approximation 2*v
    return (2.0 * v).astype(np.float32)


def axis_angle_to_quat_wxyz(aa: np.ndarray) -> np.ndarray:
    """rotvec(3) -> quaternion wxyz(4). Stateless, no branch ambiguity; used for action decoding.

    Exactly matches the `recovered` back-computation inside the training-side quat_wxyz_to_axis_angle.
    """
    aa = np.asarray(aa, dtype=np.float64)
    angle = np.linalg.norm(aa)
    q = np.zeros(4, dtype=np.float64)
    q[0] = np.cos(0.5 * angle)
    if angle > _EPS:
        q[1:] = aa / angle * np.sin(0.5 * angle)
    return q.astype(np.float32)


class QuatBranchTracker:
    """Stateful tracker that replays the training-side whole-segment branch continuity step by step (online).

    Must be reset() at the start of every episode. rot_x_pi_like corresponds to the
    training-side RotX(pi) alignment; rx ~= pi always holds for this Franka data, so it
    defaults to True. Adjust if the robot / data distribution changes.
    """

    def __init__(self, rot_x_pi_like: bool = True):
        self.rot_x_pi_like = rot_x_pi_like
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        self._prev = None

    def to_axis_angle(self, q_wxyz: np.ndarray) -> np.ndarray:
        """Feed the current frame's quaternion (wxyz); returns a branch-continuous rotvec(3)."""
        q = _canonical_wpos(q_wxyz)          # training step 1
        if self._prev is None:
            # first frame: apply the RotX(pi) +X alignment (the effect of training step 2 on frame 0)
            if self.rot_x_pi_like and q[1] < 0:
                q = -q
        else:
            # training step 2 applies to later frames too (flip to +X when q[1]<0), then step 3 temporal continuity
            if self.rot_x_pi_like and q[1] < 0:
                q = -q
            if float(np.dot(self._prev, q)) < 0:  # training step 3
                q = -q
        self._prev = q
        return quat_wxyz_to_axis_angle(q)
