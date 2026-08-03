"""Build training-consistent proprio / marker features from env obs -- shared by all three policies.

EnvObs is the canonical dict the evaluation rollout assembles from the Isaac env each step
(keys map 1:1 to training-data fields):
  eef_pose            (7,)  xyz + quat(wxyz), base frame,  == training obs/eef_pose
  gripper_pos         (>=1,)                         == training obs/gripper_pos
  agentview_rgb       (H,W,3) uint8 RGB              -> cam_high / image
  eye_in_hand_rgb     (H,W,3) uint8 RGB              -> cam_wrist / wrist_image
  # VT only:
  tactile_left_rgb    (H,W,3) uint8 RGB              == gsmini_left_markers_rgb
  tactile_right_rgb   (H,W,3) uint8 RGB              == gsmini_right_markers_rgb
  gripper_marker_motion (2,2,99,2)                   == one frame of training obs/gripper_marker_motion

The proprio axis-angle must use the stateful QuatBranchTracker (reset per episode), see rotation.py.
"""
from __future__ import annotations

import numpy as np

from softvtbench.evaluation.preprocessing.rotation import QuatBranchTracker

MARKER_SHAPE = (2, 2, 99, 2)      # [left/right finger, 2-frame history, 99 points, dx/dy]
MARKER_DIM = 792                  # 2*2*99*2, training-side raw_softvt_hdf5_to_act_dp_tactile.py


def build_proprio(env_obs: dict, tracker: QuatBranchTracker) -> np.ndarray:
    """7D proprio = xyz(3) + axis_angle(3, branch-continuous) + gripper_pos[0](1).

    == training make_state_action: state = concat(pose[:,:3], aa, gripper[:,:1]).
    """
    pose = np.asarray(env_obs["eef_pose"], dtype=np.float32).reshape(-1)
    if pose.shape[0] != 7:
        raise ValueError(f"eef_pose must be 7D xyz+quat(wxyz), got {pose.shape}")
    gripper = np.asarray(env_obs["gripper_pos"], dtype=np.float32).reshape(-1)
    aa = tracker.to_axis_angle(pose[3:])          # wxyz -> rotvec, stateful
    return np.concatenate([pose[:3], aa, gripper[:1]]).astype(np.float32)


def build_marker(env_obs: dict) -> np.ndarray:
    """Flatten one marker frame -> (792,). == the reshape in training make_marker_motion."""
    marker = np.asarray(env_obs["gripper_marker_motion"], dtype=np.float32)
    if marker.shape != MARKER_SHAPE:
        raise ValueError(f"gripper_marker_motion must be {MARKER_SHAPE}, got {marker.shape}")
    return marker.reshape(MARKER_DIM).astype(np.float32)
