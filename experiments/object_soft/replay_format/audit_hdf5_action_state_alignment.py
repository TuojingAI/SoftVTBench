#!/usr/bin/env python3
"""Audit object-soft HDF5 action/state rotation alignment.

This is intentionally read-only. It fails if any HDF5 is missing the fields
that make OpenPI-style delta = action - state safe.
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import h5py
import numpy as np


def canonical_axis_angle_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4).copy()
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    quat /= norm
    if quat[0] < 0.0:
        quat *= -1.0
    vec = quat[1:4]
    q0 = float(np.clip(quat[0], -1.0, 1.0))
    den = math.sqrt(max(1.0 - q0 * q0, 1e-12))
    if den < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return vec * ((2.0 * math.acos(q0)) / den)


def iter_files(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            out.extend(Path(p) for p in matches)
        else:
            out.append(Path(pattern))
    return sorted(set(out))


def audit_file(path: Path, max_bad_delta: float) -> tuple[bool, str]:
    with h5py.File(path, "r") as f:
        if "data/demo_0" not in f:
            return False, "missing data/demo_0"
        demo = f["data/demo_0"]
        required = ["actions", "obs/eef_pose", "obs/eef_axis_angle", "obs/actions"]
        missing = [key for key in required if key not in demo]
        if missing:
            return False, f"missing {missing}"

        actions = demo["actions"][:]
        eef_pose = demo["obs/eef_pose"][:]
        eef_axis = demo["obs/eef_axis_angle"][:]
        obs_actions = demo["obs/actions"][:]
        if actions.ndim != 2 or actions.shape[1] != 13:
            return False, f"actions shape {actions.shape}, expected (T,13)"
        if eef_pose.ndim != 2 or eef_pose.shape[1] != 7:
            return False, f"obs/eef_pose shape {eef_pose.shape}, expected (T,7)"
        if eef_axis.ndim != 2 or eef_axis.shape[1] != 3:
            return False, f"obs/eef_axis_angle shape {eef_axis.shape}, expected (T,3)"
        if obs_actions.ndim != 2 or obs_actions.shape[1] != 8:
            return False, f"obs/actions shape {obs_actions.shape}, expected (T,8)"

        n = min(actions.shape[0], eef_pose.shape[0], eef_axis.shape[0])
        if n == 0:
            return False, "zero frames"
        if np.any(eef_pose[:n, 3] < -1e-6):
            return False, "obs/eef_pose has negative quaternion w"
        if np.any(obs_actions[:n, 3] < -1e-6):
            return False, "obs/actions has negative quaternion w"

        from_quat = np.stack([canonical_axis_angle_from_quat_wxyz(q) for q in eef_pose[:n, 3:7]], axis=0)
        action_state_delta = np.linalg.norm(actions[:n, 3:6] - eef_axis[:n], axis=1)
        quat_state_delta = np.linalg.norm(from_quat - eef_axis[:n], axis=1)
        bad_action = int(np.sum(action_state_delta > max_bad_delta))
        bad_quat = int(np.sum(quat_state_delta > max_bad_delta))
        if bad_action or bad_quat:
            return (
                False,
                "rotation delta failed "
                f"bad_action={bad_action} max_action={float(np.max(action_state_delta)):.6g} "
                f"bad_quat={bad_quat} max_quat={float(np.max(quat_state_delta)):.6g}",
            )

        if "soft_extras" in demo:
            attrs = demo["soft_extras"].attrs
            attr_bad = int(attrs.get("axis_angle_action_state_delta_bad_frames_gt_pi", 0))
            if attr_bad:
                return False, f"soft_extras attr reports bad_frames_gt_pi={attr_bad}"

        return (
            True,
            f"ok frames={n} max_action_delta={float(np.max(action_state_delta)):.6g} "
            f"max_quat_delta={float(np.max(quat_state_delta)):.6g}",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="HDF5 files or glob patterns")
    parser.add_argument("--max-bad-delta", type=float, default=1e-4)
    args = parser.parse_args()

    files = iter_files(args.paths)
    ok_count = 0
    fail_count = 0
    for path in files:
        ok, msg = audit_file(path, args.max_bad_delta)
        print(("OK  " if ok else "FAIL") + f" {path} :: {msg}")
        ok_count += int(ok)
        fail_count += int(not ok)
    print(f"SUMMARY ok={ok_count} fail={fail_count} total={len(files)}")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
