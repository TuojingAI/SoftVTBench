# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import hashlib
import json
import os
import sys
import re
from datetime import datetime
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import h5py  # optional expert-action replay
import numpy as np
import torch
import tyro
from isaaclab.app import AppLauncher
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaacsim import SimulationApp
from openpi_client import websocket_client_policy as _websocket_client_policy

# Utilize the common utility functions from gr00t for OpenPI inference
from benchmarks.common.closedloop_policy_inference import (
    ClosedLoopArguments,
    ClosedLoopPolicyInference,
)
from benchmarks.common.metrics import (
    compute_contact_force_metrics_from_13d,
    compute_contact_force_metrics_from_lr_forces,
    compute_contact_force_series_from_lr_forces,
    compute_topk_mean,
)


TARGET_IMAGE_HW = (224, 224)
SOFTVTBENCH_DEFORMATION_METRIC_ID = "fem_rms_rigid_aligned_bbox_pct_v1"

_SAFE_DIR_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _sanitize_dirname(name: str) -> str:
    """Sanitize a string to be safe as a single path component."""
    s = (name or "").strip()
    if not s:
        return "none"
    s = s.replace(" ", "_")
    s = _SAFE_DIR_RE.sub("_", s)
    s = s.strip("._-")
    return s or "none"


def _to_uint8_rgb(img) -> np.ndarray:
    """Convert an image tensor/ndarray to uint8 RGB numpy array."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.dtype in (np.float32, np.float64):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value not in ("0", "false", "no", "off", "disable", "disabled")


def _arr_preview(arr, *, rows: int = 3, cols: int = 13) -> list:
    x = np.asarray(arr)
    if x.ndim == 0:
        return [float(x)]
    if x.ndim == 1:
        return np.round(x[:cols].astype(np.float64), 6).tolist()
    return np.round(x[:rows, : min(cols, x.shape[1])].astype(np.float64), 6).tolist()


def _delta_summary(action_chunk, state_7: np.ndarray) -> dict:
    action = np.asarray(action_chunk, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] < 7:
        return {}
    state = np.asarray(state_7, dtype=np.float32).reshape(-1)
    cols = min(7, state.shape[0], action.shape[1])
    delta = action[:, :cols] - state[:cols][None, :]
    out = {
        "delta_first": _arr_preview(delta[0], cols=cols),
    }
    if cols >= 3:
        xyz_norm = np.linalg.norm(delta[:, :3], axis=1)
        out["xyz_delta_norm_first"] = float(np.round(xyz_norm[0], 6))
        out["xyz_delta_norm_max"] = float(np.round(np.max(xyz_norm), 6))
    if cols >= 6:
        rot_norm = np.linalg.norm(delta[:, 3:6], axis=1)
        out["rot_delta_norm_first"] = float(np.round(rot_norm[0], 6))
        out["rot_delta_norm_max"] = float(np.round(np.max(rot_norm), 6))
        out["rot_delta_first"] = _arr_preview(delta[0, 3:6], cols=3)
    if cols >= 7:
        out["gripper_delta_first"] = float(np.round(delta[0, 6], 6))
        out["gripper_minmax"] = [
            float(np.round(np.min(action[:, 6]), 6)),
            float(np.round(np.max(action[:, 6]), 6)),
        ]
    return out


def _write_policy_action_debug(
    *,
    debug_path,
    enabled: bool,
    stdout: bool,
    record: dict,
) -> None:
    if not enabled:
        return
    try:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if stdout:
            print("[PolicyActionDebug] " + line, flush=True)
        if debug_path:
            path = Path(debug_path)
            path.mkdir(parents=True, exist_ok=True)
            with (path / "policy_action_debug.jsonl").open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:
        print(f"[PolicyActionDebug] failed to write debug record: {exc}", flush=True)


def _pad_history_front(items: list[np.ndarray], target_len: int) -> list[np.ndarray]:
    """Pad a history list by repeating the earliest item (front-padding)."""
    if target_len <= 0:
        return []
    if len(items) == 0:
        raise ValueError('Cannot pad empty history.')
    if len(items) >= target_len:
        return items[-target_len:]
    pad_n = target_len - len(items)
    return [items[0]] * pad_n + items


def _build_tactile_mosaic(
    left_hist: list[np.ndarray],
    right_hist: list[np.ndarray],
    *,
    out_hw: tuple[int, int] = TARGET_IMAGE_HW,
) -> np.ndarray:
    """Build the 4x4 tactile mosaic.

    Two layouts exist in the training converters:
    - "cols" (default, legacy): left finger 4x2 in columns 0..1, right in columns
      2..3 — matches convert_all_libero_to_softvtbench.py (rigid pipeline).
    - "rows": cell = sensor_offset*8 + k, (row, col) = divmod(cell, 4): left
      history fills rows 0..1 row-major old->new, right fills rows 2..3 —
      matches convert_softvtbench_tactile_data_to_lerobot.py (soft pipeline).
    Select with SOFTVTBENCH_TACTILE_MOSAIC_LAYOUT=rows|cols to match the converter
    the policy was trained with.
    """
    H_out, W_out = out_hw
    cell_h, cell_w = H_out // 4, W_out // 4
    canvas = np.zeros((H_out, W_out, 3), dtype=np.uint8)

    layout = os.environ.get("SOFTVTBENCH_TACTILE_MOSAIC_LAYOUT", "cols").strip().lower()
    if layout == "rows":
        for sensor_offset, hist in enumerate((left_hist, right_hist)):
            for k in range(8):
                cell = sensor_offset * 8 + k
                r, c = divmod(cell, 4)
                y0, y1 = r * cell_h, (r + 1) * cell_h
                x0, x1 = c * cell_w, (c + 1) * cell_w
                canvas[y0:y1, x0:x1] = cv2.resize(hist[k], (cell_w, cell_h))
        return canvas

    # Legacy "cols" layout:
    # - Left finger: 4x2 grid in columns 0..1
    # - Right finger: 4x2 grid in columns 2..3
    for k in range(8):
        r = k // 2  # 0..3
        c = k % 2  # 0..1
        y0, y1 = r * cell_h, (r + 1) * cell_h

        # left
        x0, x1 = c * cell_w, (c + 1) * cell_w
        canvas[y0:y1, x0:x1] = cv2.resize(left_hist[k], (cell_w, cell_h))

        # right
        x0, x1 = (c + 2) * cell_w, (c + 3) * cell_w
        canvas[y0:y1, x0:x1] = cv2.resize(right_hist[k], (cell_w, cell_h))

    return canvas


class _OnlineTactileBuffer:
    """Maintain online tactile/force/marker histories to match SoftVTBench dataset fields."""

    def __init__(
        self,
        *,
        tactile_sensors: tuple[str, str],
        tactile_output_type: str,
        tactile_history_len: int = 8,
        force_history_len: int = 8,
        marker_history_len: int = 8,
    ) -> None:
        if tactile_history_len != 8:
            raise ValueError('tactile_history_len must be 8 to match the 4x4 mosaic layout.')
        self.tactile_sensors = tactile_sensors
        self.tactile_output_type = tactile_output_type
        self.force_history_len = force_history_len
        self.marker_history_len = marker_history_len
        self.reset()

    def reset(self) -> None:
        self._left_frames: deque[np.ndarray] = deque(maxlen=8)
        self._right_frames: deque[np.ndarray] = deque(maxlen=8)
        self._force_hist: deque[np.ndarray] = deque(maxlen=self.force_history_len)
        self._marker_hist: deque[np.ndarray] = deque(maxlen=self.marker_history_len)
        self._marker_init: np.ndarray | None = None

    def update_tactile_frames(self, env, env_id: int = 0) -> None:
        left_name, right_name = self.tactile_sensors
        left_img = env.unwrapped.scene.sensors[left_name].data.output[self.tactile_output_type][env_id]
        right_img = env.unwrapped.scene.sensors[right_name].data.output[self.tactile_output_type][env_id]
        self._left_frames.append(_to_uint8_rgb(left_img))
        self._right_frames.append(_to_uint8_rgb(right_img))

    def update_force(self, obs: dict) -> None:
        policy_obs = obs.get('policy', {}) if isinstance(obs, dict) else {}
        if not isinstance(policy_obs, dict) or 'gripper_net_force' not in policy_obs:
            return
        gnf = policy_obs['gripper_net_force']
        if isinstance(gnf, torch.Tensor):
            gnf = gnf.detach().cpu().numpy()
        gnf = np.asarray(gnf)
        gnf0 = np.squeeze(gnf, axis=0)
        if gnf0.ndim == 2:
            # (2,3)
            inst = gnf0.reshape(6).astype(np.float32)
        else:
            # (H,2,3): take current step at index 0
            inst = gnf0[0].reshape(6).astype(np.float32)
        self._force_hist.append(inst)

    def update_marker_motion(self, obs: dict) -> None:
        policy_obs = obs.get('policy', {}) if isinstance(obs, dict) else {}
        if not isinstance(policy_obs, dict) or 'gripper_marker_motion' not in policy_obs:
            return
        gmm = policy_obs['gripper_marker_motion']
        if isinstance(gmm, torch.Tensor):
            gmm = gmm.detach().cpu().numpy()
        gmm = np.asarray(gmm)
        gmm0 = np.squeeze(gmm, axis=0)
        if gmm0.ndim != 4:
            return
        # (2,2,M,2): sensor, (init/current), marker, xy
        init_pos = gmm0[:, 0, :, :].reshape(-1, 2).astype(np.float32)  # (2*M,2)
        curr_pos = gmm0[:, 1, :, :].reshape(-1, 2).astype(np.float32)  # (2*M,2)
        if self._marker_init is None:
            self._marker_init = init_pos
        self._marker_hist.append(curr_pos)

    def get_tactile_image(self) -> np.ndarray | None:
        if len(self._left_frames) == 0 or len(self._right_frames) == 0:
            return None
        left_hist = _pad_history_front(list(self._left_frames), 8)
        right_hist = _pad_history_front(list(self._right_frames), 8)
        return _build_tactile_mosaic(left_hist, right_hist, out_hw=TARGET_IMAGE_HW)

    def get_force_history(self) -> np.ndarray | None:
        if len(self._force_hist) == 0:
            return None
        hist = _pad_history_front([x.astype(np.float32) for x in self._force_hist], self.force_history_len)
        return np.stack(hist, axis=0).astype(np.float32)  # (H,6)

    def get_marker_motion(self) -> np.ndarray | None:
        if self._marker_init is None or len(self._marker_hist) == 0:
            return None
        hist = _pad_history_front([x.astype(np.float32) for x in self._marker_hist], self.marker_history_len)
        out = np.zeros((1 + self.marker_history_len, self._marker_init.shape[0], 2), dtype=np.float32)
        out[0] = self._marker_init
        out[1:] = np.stack(hist, axis=0)
        return out


class _TactileGripperController:
    """Small online gripper wrapper driven only by tactile marker displacement."""

    def __init__(self, *, open_finger: float) -> None:
        self.open_finger = float(open_finger)
        self.contact_thr = float(os.environ.get("SOFTVTBENCH_TACTILE_CTRL_CONTACT_THR", "8.0"))
        self.single_contact_thr = float(os.environ.get("SOFTVTBENCH_TACTILE_CTRL_SINGLE_CONTACT_THR", "5.0"))
        self.safety_thr = float(os.environ.get("SOFTVTBENCH_TACTILE_CTRL_SAFETY_THR", "30.0"))
        self.slip_drop_thr = float(os.environ.get("SOFTVTBENCH_TACTILE_CTRL_SLIP_DROP_THR", "4.0"))
        self.close_step = float(os.environ.get("SOFTVTBENCH_TACTILE_CTRL_CLOSE_STEP", "0.0010"))
        self.open_step = float(os.environ.get("SOFTVTBENCH_TACTILE_CTRL_OPEN_STEP", "0.0010"))
        self.min_finger = float(os.environ.get("SOFTVTBENCH_TACTILE_CTRL_MIN_FINGER", "0.0060"))
        self.release_open_finger = float(os.environ.get("SOFTVTBENCH_TACTILE_CTRL_RELEASE_OPEN_FINGER", "0.0320"))
        self.release_min_frame = int(os.environ.get("SOFTVTBENCH_TACTILE_CTRL_RELEASE_MIN_FRAME", "80"))
        self.topk_side = int(os.environ.get("SOFTVTBENCH_TACTILE_CTRL_TOPK_SIDE", "10"))
        self.topk_global = int(os.environ.get("SOFTVTBENCH_TACTILE_CTRL_TOPK_GLOBAL", "20"))
        self.pre_contact_mode = os.environ.get("SOFTVTBENCH_TACTILE_CTRL_PRE_CONTACT_MODE", "v2").strip().lower()
        self.safety_action = os.environ.get("SOFTVTBENCH_TACTILE_CTRL_SAFETY_ACTION", "open").strip().lower()
        self.release_require_contact_drop = os.environ.get(
            "SOFTVTBENCH_TACTILE_CTRL_RELEASE_REQUIRE_CONTACT_DROP", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.release_contact_drop_ratio = float(os.environ.get("SOFTVTBENCH_TACTILE_CTRL_RELEASE_CONTACT_DROP_RATIO", "0.55"))
        self.max_pre_contact_close = float(os.environ.get("SOFTVTBENCH_TACTILE_CTRL_MAX_PRE_CONTACT_CLOSE", "0.0200"))
        self.max_safety_open_finger = float(os.environ.get("SOFTVTBENCH_TACTILE_CTRL_MAX_SAFETY_OPEN_FINGER", "0.0240"))
        self.reset()

    def reset(self) -> None:
        self.contact_seen = False
        self.releasing = False
        self.hold_finger: float | None = None
        self.search_finger: float | None = None
        self.prev_min_contact_score: float | None = None
        self.peak_min_contact_score: float = 0.0
        self.marker_baseline: np.ndarray | None = None
        self.last_diag: dict[str, object] = {
            "active": True,
            "reason": "reset",
            "contact_seen": False,
        }

    @staticmethod
    def _topk_mean(values: np.ndarray, k: int) -> float:
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        if flat.size == 0:
            return 0.0
        kk = int(max(1, min(k, flat.size)))
        return float(np.partition(flat, flat.size - kk)[-kk:].mean())

    def _marker_scores(self, obs: dict) -> tuple[dict[str, object], bool]:
        policy_obs = obs.get("policy", {}) if isinstance(obs, dict) else {}
        gmm = policy_obs.get("gripper_marker_motion", None) if isinstance(policy_obs, dict) else None
        if gmm is None:
            return {"active": True, "reason": "missing_gripper_marker_motion"}, False
        if isinstance(gmm, torch.Tensor):
            gmm = gmm.detach().cpu().numpy()
        gmm = np.asarray(gmm, dtype=np.float32)
        try:
            gmm0 = np.squeeze(gmm, axis=0)
        except Exception:
            gmm0 = np.squeeze(gmm)
        if gmm0.ndim != 4 or gmm0.shape[0] < 2 or gmm0.shape[1] < 2 or gmm0.shape[-1] != 2:
            return {"active": True, "reason": f"bad_marker_shape:{tuple(gmm0.shape)}"}, False

        # gmm0: (sensor=2, init/current=2, marker, xy).
        # Online obs may refresh its internal "init" slot, so keep an
        # episode-level baseline here, matching the converter/buffer semantics.
        curr = gmm0[:, 1, :, :].astype(np.float32)
        if self.marker_baseline is None or tuple(self.marker_baseline.shape) != tuple(curr.shape):
            self.marker_baseline = curr.copy()
        delta = curr - self.marker_baseline
        mag = np.linalg.norm(delta, axis=-1)
        side_scores = [
            self._topk_mean(mag[side], self.topk_side)
            for side in range(min(2, mag.shape[0]))
        ]
        while len(side_scores) < 2:
            side_scores.append(0.0)
        global_score = self._topk_mean(mag, self.topk_global)
        min_contact_score = float(min(side_scores[0], side_scores[1]))
        contact = bool(side_scores[0] >= self.contact_thr and side_scores[1] >= self.contact_thr)
        single_contact = bool(max(side_scores[0], side_scores[1]) >= self.single_contact_thr)
        diag = {
            "active": True,
            "reason": "ok",
            "left_score": float(side_scores[0]),
            "right_score": float(side_scores[1]),
            "min_contact_score": min_contact_score,
            "global_score": float(global_score),
            "contact": contact,
            "single_contact": single_contact,
            "safety_over": bool(global_score >= self.safety_thr),
            "contact_seen": bool(self.contact_seen),
            "release_allowed": False,
        }
        return diag, True

    def adjust_chunk(
        self,
        *,
        policy_finger: np.ndarray,
        obs: dict,
        frame_count: int,
    ) -> tuple[np.ndarray, dict[str, object]]:
        policy_finger = np.asarray(policy_finger, dtype=np.float32)
        out = np.clip(policy_finger.copy(), 0.0, self.open_finger)
        diag, ok = self._marker_scores(obs)
        if not ok:
            diag.update(
                {
                    "contact_seen": bool(self.contact_seen),
                    "hold_finger": self.hold_finger,
                    "policy_first": float(out[0]) if out.size else None,
                    "exec_first": float(out[0]) if out.size else None,
                    "action": "follow_policy",
                }
            )
            self.last_diag = diag
            return out, diag

        policy_first = float(out[0]) if out.size else self.open_finger
        contact = bool(diag.get("contact", False))
        single_contact = bool(diag.get("single_contact", False))
        safety_over = bool(diag.get("safety_over", False))
        min_score = float(diag.get("min_contact_score", 0.0))
        prev_min = self.prev_min_contact_score
        contact_drop = bool(prev_min is not None and self.contact_seen and (prev_min - min_score) >= self.slip_drop_thr)
        release_allowed = bool(frame_count >= self.release_min_frame)
        wants_release = bool(policy_first >= self.release_open_finger)

        action = "follow_policy"
        target = policy_first
        if contact and not self.contact_seen:
            self.contact_seen = True
            self.hold_finger = float(np.clip(policy_first, self.min_finger, self.open_finger))

        if self.contact_seen:
            self.peak_min_contact_score = max(self.peak_min_contact_score, min_score)
            if self.hold_finger is None:
                self.hold_finger = float(np.clip(policy_first, self.min_finger, self.open_finger))

            if self.release_require_contact_drop and self.peak_min_contact_score > 0:
                release_allowed = release_allowed and (
                    min_score <= self.release_contact_drop_ratio * self.peak_min_contact_score
                )

            if wants_release and release_allowed:
                self.releasing = True
                target = policy_first
                action = "release"
            elif wants_release and not release_allowed:
                target = self.hold_finger
                action = "delay_release"
            elif safety_over:
                if self.safety_action in ("hold", "freeze"):
                    target = self.hold_finger
                    action = "hold_for_safety"
                else:
                    target = min(
                        self.open_finger,
                        self.max_safety_open_finger,
                        max(self.hold_finger, policy_first) + self.open_step,
                    )
                    action = "open_for_safety"
                self.hold_finger = target
            elif contact_drop:
                target = max(self.min_finger, min(self.hold_finger, policy_first) - self.close_step)
                self.hold_finger = target
                action = "close_for_slip"
            else:
                # Once both sides have touched, avoid extra policy-driven squeezing.
                target = max(self.hold_finger, policy_first)
                self.hold_finger = target
                action = "hold_contact"
        elif single_contact:
            # Nudge closed very gently to search for balanced two-sided contact.
            if self.pre_contact_mode in ("passive", "follow", "policy"):
                action = "single_contact_follow_policy"
                target = policy_first
            elif self.search_finger is None:
                self.search_finger = float(np.clip(policy_first, self.min_finger, self.open_finger))
                target = max(self.min_finger, min(self.search_finger, policy_first) - self.close_step)
                self.search_finger = target
                action = "close_for_second_side"
            else:
                lower_bound = max(self.min_finger, min(self.max_pre_contact_close, self.open_finger))
                target = max(lower_bound, min(self.search_finger, policy_first) - self.close_step)
                self.search_finger = target
                action = "close_for_second_side"
        else:
            self.search_finger = None

        target = float(np.clip(target, self.min_finger, self.open_finger))
        if out.size:
            if action.startswith("close"):
                out[:] = np.minimum(out, target)
            elif action in ("hold_contact", "delay_release", "open_for_safety"):
                out[:] = target
            elif action == "release":
                out[:] = np.clip(policy_finger, 0.0, self.open_finger)

        self.prev_min_contact_score = min_score
        diag.update(
            {
                "contact_seen": bool(self.contact_seen),
                "release_allowed": release_allowed,
                "wants_release": wants_release,
                "contact_drop": contact_drop,
                "peak_min_contact_score": float(self.peak_min_contact_score),
                "prev_min_contact_score": float(prev_min) if prev_min is not None else None,
                "hold_finger": float(self.hold_finger) if self.hold_finger is not None else None,
                "search_finger": float(self.search_finger) if self.search_finger is not None else None,
                "policy_first": policy_first,
                "exec_first": float(out[0]) if out.size else None,
                "delta_first": float(out[0] - policy_first) if out.size else None,
                "action": action,
            }
        )
        self.last_diag = diag
        return out.astype(np.float32), diag


@dataclass
class OpenpiClientArguments(ClosedLoopArguments):

    record_images: bool = False
    record_videos: bool = False
    num_envs: int = 1
    background_env_usd_path: str | None = None
    record_camera_output_path: str | None = None

    # Server connection parameters
    server_host: str = "127.0.1.1"
    server_port: int = 8000
    target_image_size: tuple[int, int, int] = (224, 224, 3)

    # Simulator specific parameters
    # Default to headless to avoid X11/GLX BadMatch crashes on servers or misconfigured displays.
    # If you want a GUI window, pass: --no-headless
    headless: bool = True
    seed: int = 11
    # debug_mode:
    #   0: 关闭所有额外调试，仅打印基础统计信息
    #   1: 在 0 的基础上额外保存动作 (action_XXXX.npy)
    #   2: 在 1 的基础上额外保存相机帧到 debug_path
    #   3: 在 2 的基础上额外 dump 关节状态 / 图像序列
    #   4: 在 0 的基础上开启 Hybrid 力–位混合可视化（不依赖 1-3 的其它 dump）
    #   5: 不实时画图；逐帧记录挤压力（预测/实测）到 benchmarks/softvtbench/gripper_force/<task_id>/
    #   6: 逐帧保存：
    #        - 双相机 RGB（第三人称 agentview + 腕部 eye_in_hand）
    #        - 左右触觉 markers_rgb（gsmini_left/right）
    #        - 夹持/外力的预测量与实测量（含 3D 向量与 squeeze/ap 派生指标）
    #      输出目录：<debug_path>/capture_mode6/<suite>/task_<id>/<adverb_tag>/<timestamp>/exp_XXX/...
    debug_mode: int = 0
    # Default to a repo-local folder for full debug records (images + tactile + forces).
    # You can override via CLI: --debug_path /abs/path/to/dir
    debug_path: str = str(project_root / "full_records")

    camera_names: tuple[str, ...] = ("agentview_cam", "eye_in_hand_cam")
    tactile_sensor_names: tuple[str, str] = ("gsmini_left", "gsmini_right")
    tactile_output_type: str = "tactile_rgb"  # or "markers_rgb"
    tactile_history_len: int = 8
    force_history_len: int = 8
    marker_history_len: int = 8
    num_steps_wait: int = 5  # Number of steps to wait for objects to stabilize i n sim
    replan_steps: int = 10  # For each action, will execute replan_steps times
    max_inference_steps: int = 30  # max number of inference steps to run
    num_success_steps: int = 8  # continuous success steps to consider the policy as successful
    num_total_experiments: int = 50  # total number of experiments to do policy evaluation

    # Control mode parameters
    # Supported modes:
    #   - "diffik": Task-space control via Differential IK
    #   - "osc":    Task-space control via OSC
    #   - "hybrid":  Hybrid force–position control (ContactForce)
    #   - "tactile": Hybrid force–position + tactile observations (GelSight)
    #   - "binary": IK + tactile observations (GelSight), but execute 7D actions with **binary gripper**
    #
    # OpenPI server always returns a 32D action vector (padded), but:
    #   - For "diffik": we use the first 7D
    #       (x, y, z, rx, ry, rz, gripper) - axis-angle + gripper
    #       and convert it to 8D quaternion before sending to the env:
    #       (x, y, z, qw, qx, qy, qz, gripper)
    #   - For "osc": we use the first 7D directly:
    #       (x, y, z, rx, ry, rz, gripper)
    #   - For "hybrid"/"tactile": we use the first 13D **directly** as the Hybrid action:
    #       (x, y, z, rx, ry, rz, gripper, fL(3), fR(3))  -- no zero padding on the client side
    control_mode: str = "diffik"
    task: str = ""  # Will be auto-set based on control_mode if not provided

    # Ablation (short flag): tactile obs/model branch, but execute absolute 7D task-space actions.
    # - Env still expects 13D in tactile mode: pad force dims with zeros
    # - Disable pos_kp/squeeze_kp corrections at runtime
    abs7d: bool = False

    # Task setup parameters
    task_suite: str = "libero_goal"
    task_id: int = 1
    task_config_path: Path = Path(__file__).parent.parent.resolve() / "datasets" / "libero" / "config"
    language_instruction: str = ""

    # Optional: prompt adverb augmentation (SoftVTBench-style).
    # Keep CLI flags compatible with scripts/tools/run_task_evaluations.py:
    #   --prompt-adverb, --prompt-adverbs, --prompt-seed
    prompt_adverb: str = ""
    prompt_adverbs: tuple[str, ...] = ()
    prompt_seed: int = 0

    # HDF5 dataset parameters for initial state loading（目录内需含 {task_suite}_task{id}_*_demo.hdf5）
    # 未指定时唯一来源：环境变量 HDF5_TRAJ_SOURCE_DIR（与 set_replay_env.sh / task_configs 一致）
    hdf5_folder: Optional[Path] = None

    # Debug/sanity mode: bypass OpenPI server and replay actions from a root-format HDF5
    # containing a top-level `actions` dataset (e.g. SoftVTBench success package).
    replay_action_hdf5: Optional[Path] = None


# Parse arguments first to get task_suite and task_id
args = tyro.cli(OpenpiClientArguments)


def _choose_adverb(seed: int, key: str, adverbs: tuple[str, ...]) -> str:
    """Deterministically choose one adverb from a list (match convert_all_libero_to_softvtbench.py)."""
    if not adverbs:
        return ""
    digest = hashlib.blake2b(f"{int(seed)}:{key}".encode("utf-8"), digest_size=8).digest()
    idx = int.from_bytes(digest, "big") % len(adverbs)
    return (adverbs[idx] or "").strip()


def _rewrite_instruction(instruction: str, adverb: str, seed: int, key: str) -> str:
    """Rewrite instruction with an adverb in a more natural English style (deterministic).

    Strategy (deterministic per key):
    - randomly choose between:
      * prefix:  "{adverb} {instruction}"   (e.g. "gently open the drawer")
      * suffix:  "{instruction} {adverb}"   (e.g. "open the drawer gently")
    """
    instruction = (instruction or "").strip()
    adverb = (adverb or "").strip()
    if not adverb:
        return instruction
    if not instruction:
        return adverb

    lower = instruction.lower()
    if lower.startswith(f"{adverb} "):
        return instruction
    if lower.endswith(f" {adverb}"):
        return instruction

    style = _choose_adverb(int(seed), f"{key}:style", ("prefix", "suffix"))
    if style == "suffix":
        return f"{instruction} {adverb}"
    return f"{adverb} {instruction}"


def _add_bytes_key_aliases(d: dict, keys: tuple[str, ...]) -> None:
    """Add bytes-key aliases for servers that decode msgpack map keys as bytes."""
    for k in keys:
        if k in d and isinstance(k, str):
            d[k.encode("utf-8")] = d[k]


def _debug_soft_goal_state(env, goals: list[dict] | None = None) -> dict:
    """Return lightweight goal diagnostics for soft-object placement tasks."""
    out: dict[str, object] = {}
    goals = goals or []
    for goal in goals:
        if "relationship" not in goal:
            continue
        obj_name = goal.get("ref_obj")
        target_name = goal.get("target")
        try:
            obj = env.scene[obj_name]
            target = env.scene[target_name]
            pos_diff = obj.data.root_pos_w - target.data.root_pos_w
            xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
            height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
            out.update(
                {
                    "goal_ref_obj": str(obj_name),
                    "goal_target": str(target_name),
                    "goal_obj_pos": obj.data.root_pos_w[0].detach().cpu().numpy().astype(float).tolist(),
                    "goal_target_pos": target.data.root_pos_w[0].detach().cpu().numpy().astype(float).tolist(),
                    "goal_pos_diff": pos_diff[0].detach().cpu().numpy().astype(float).tolist(),
                    "goal_xy_dist": float(xy_dist[0].detach().cpu()),
                    "goal_height_dist": float(height_dist[0].detach().cpu()),
                    "goal_xy_threshold": float(goal.get("xy_threshold", float("nan"))),
                    "goal_height_threshold": float(goal.get("height_threshold", float("nan"))),
                    "goal_height_diff": float(goal.get("height_diff", 0.0)),
                }
            )
        except Exception as exc:
            out["goal_debug_error"] = str(exc)
        break
    return out


def _deformable_names_from_env_and_goals(goals: list[dict] | None = None) -> list[str]:
    """Return likely deformable object names for debug FEM logging."""
    names: list[str] = []
    for key in ("SOFTVTBENCH_EXTRA_ASSET_NAME", "SOFTVTBENCH_EXTRA_ASSET_NAME_SOFT"):
        name = os.environ.get(key, "").strip()
        if name and name not in names:
            names.append(name)
    for goal in goals or []:
        name = str(goal.get("ref_obj", "")).strip()
        if name and name not in names:
            names.append(name)
    return names


def _get_scene_nodal_pos_w(env, names: list[str]) -> tuple[str | None, torch.Tensor | None]:
    """Fetch the first available deformable nodal positions from the scene."""
    scene = getattr(env, "scene", env)
    for name in names:
        try:
            obj = scene[name]
            nodes = getattr(getattr(obj, "data", None), "nodal_pos_w", None)
            if nodes is None:
                continue
            if isinstance(nodes, torch.Tensor):
                return name, nodes[0].detach().float().clone()
            return name, torch.as_tensor(nodes[0], dtype=torch.float32).clone()
        except Exception:
            continue
    return None, None


def _debug_fem_deformation_state(
    env,
    reference_nodes: dict[str, torch.Tensor] | None,
    goals: list[dict] | None = None,
) -> dict:
    """Return FEM deformation metrics in the same units as saved SoftVTBench soft extras.

    The metric removes translation and rotation with a proper Kabsch alignment,
    then normalizes nodal displacement by the reference bbox diagonal and
    expresses it as a percentage.
    """
    out: dict[str, object] = {}
    if not reference_nodes:
        return {"fem_deformation_known": False, "fem_deformation_reason": "missing_reference"}

    names = _deformable_names_from_env_and_goals(goals)
    names.extend([name for name in reference_nodes.keys() if name not in names])
    asset_name, current = _get_scene_nodal_pos_w(env, names)
    if asset_name is None or current is None:
        return {"fem_deformation_known": False, "fem_deformation_reason": "missing_runtime_nodes"}

    reference = reference_nodes.get(asset_name)
    if reference is None:
        return {
            "fem_deformation_known": False,
            "fem_deformation_asset": asset_name,
            "fem_deformation_reason": "missing_matching_reference",
        }
    if tuple(reference.shape) != tuple(current.shape):
        return {
            "fem_deformation_known": False,
            "fem_deformation_asset": asset_name,
            "fem_deformation_reason": f"shape_mismatch:{tuple(reference.shape)}!={tuple(current.shape)}",
        }

    try:
        reference = reference.to(device=current.device, dtype=current.dtype)
        ref_centered = reference - reference.mean(dim=0, keepdim=True)
        cur_centered = current - current.mean(dim=0, keepdim=True)
        covariance = cur_centered.transpose(0, 1) @ ref_centered
        u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
        correction = torch.eye(3, device=current.device, dtype=current.dtype)
        correction[-1, -1] = torch.where(
            torch.linalg.det(u @ vh) < 0,
            current.new_tensor(-1.0),
            current.new_tensor(1.0),
        )
        rotation = u @ correction @ vh
        aligned = cur_centered @ rotation
        disp = torch.linalg.vector_norm(aligned - ref_centered, dim=-1)
        ref_bbox = reference.max(dim=0).values - reference.min(dim=0).values
        cur_bbox = current.max(dim=0).values - current.min(dim=0).values
        ref_diag = torch.linalg.vector_norm(ref_bbox).clamp_min(1.0e-8)
        rms_pct = 100.0 * torch.sqrt(torch.mean(disp * disp)) / ref_diag
        max_pct = 100.0 * torch.max(disp) / ref_diag
        bbox_delta = torch.linalg.vector_norm(cur_bbox - ref_bbox)
        out.update(
            {
                "fem_deformation_known": True,
                "fem_deformation_metric_id": SOFTVTBENCH_DEFORMATION_METRIC_ID,
                "fem_deformation_asset": asset_name,
                "fem_deformation_nodes": int(current.shape[0]),
                "fem_deformation_rms": float(rms_pct.detach().cpu()),
                "fem_deformation_max": float(max_pct.detach().cpu()),
                "fem_bbox_diag": float(ref_diag.detach().cpu()),
                "fem_bbox_delta": float(bbox_delta.detach().cpu()),
                "fem_bbox_dims": cur_bbox.detach().cpu().numpy().astype(float).tolist(),
            }
        )
    except Exception as exc:
        out.update(
            {
                "fem_deformation_known": False,
                "fem_deformation_asset": asset_name,
                "fem_deformation_reason": str(exc),
            }
        )
    return out


# Set USE_RELATIVE_MODE environment variable for DiffIK controller
# For OpenPI inference with absolute pose control, we always use absolute mode (False)
if "USE_RELATIVE_MODE" not in os.environ:
    os.environ["USE_RELATIVE_MODE"] = "False"
    print("Set USE_RELATIVE_MODE=False for absolute pose control (OpenPI default)")

# Map control mode to corresponding environment if task not explicitly set
if not args.task:
    control_mode_to_env = {
        "diffik": "Isaac-Libero-Franka-IK-v0",  # Differential IK control
        "osc": "Isaac-Libero-Franka-OscPose-v0",  # OSC control
        # 兼容模式：
        # - hybrid  -> 纯 Hybrid-ContactForce 环境（无 GelSight），保持与旧版一致
        # - tactile -> Hybrid-Tactile 环境（推荐，用于触觉+力评估）
        "hybrid": "Isaac-Libero-Franka-Hybrid-ContactForce-v0",
        "tactile": "Isaac-Libero-Franka-Hybrid-Tactile-v0",
        # binary -> IK + tactile env (non-hybrid). Action execution: 8D pose + **binary** gripper.
        "binary": "Isaac-Libero-Franka-IK-Camera-Tactile-v0",
    }
    if args.control_mode not in control_mode_to_env:
        raise ValueError(f"Invalid control mode: {args.control_mode}. Supported modes: {list(control_mode_to_env.keys())}")
    args.task = control_mode_to_env[args.control_mode]
    print(f"Using task environment: {args.task} for control mode: {args.control_mode}")
else:
    print(f"Using explicitly specified task environment: {args.task}")

# HDF5 目录：仅认 HDF5_TRAJ_SOURCE_DIR；可选 CLI --hdf5_folder 覆盖并写回该环境变量。
if args.hdf5_folder is None:
    traj = (os.environ.get("HDF5_TRAJ_SOURCE_DIR") or "").strip()
    if not traj:
        raise ValueError(
            "Missing HDF5 folder for OpenPI inference.\n"
            "  export HDF5_TRAJ_SOURCE_DIR=/path/to/assembled_hdf5\n"
            "  # 或: source scripts/tools/set_replay_env.sh inference\n"
            "Or pass: --hdf5-folder /path/to/assembled_hdf5"
        )
    args.hdf5_folder = Path(traj)
    print(f"Using HDF5 folder from HDF5_TRAJ_SOURCE_DIR: {args.hdf5_folder}")
else:
    os.environ["HDF5_TRAJ_SOURCE_DIR"] = str(args.hdf5_folder)
    print(f"Using HDF5 folder from command line (--hdf5.folder): {args.hdf5_folder}")

# Launch the simulator FIRST before importing tac_manip modules
app_experience = os.environ.get("SOFTVTBENCH_APP_EXPERIENCE", "").strip()
if app_experience:
    print(f"Using SOFTVTBENCH_APP_EXPERIENCE={app_experience}")
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True, num_envs=1, experience=app_experience)
else:
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True, num_envs=1)
simulation_app = app_launcher.app

# add configs for dataset generation for various task_suite and task_id,
# supported task_suites: [xhumanoid, libero, etc.]
# NOTE: Import tac_manip modules AFTER AppLauncher is initialized
if args.task_suite is not None:
    from tac_manip.utils.task_configs import setup_task_objects

    setup_task_objects(args.task_suite, args.task_id)

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import import_packages

from benchmarks.openpi.env import (
    axisangle2quat,
    quat2axisangle,
    resize_frames_with_padding,
)

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp", "pick_place"]
# Import all configs in this package
import_packages("isaaclab_tasks", _BLACKLIST_PKGS)


def get_episode_map(names):
    """Get a mapping of episode indices to their names.

    Args:
        names: List or dict of episode names

    Returns:
        dict: Mapping of episode indices to their names (e.g., {0: 'episode_0', 2: 'episode_2', 5: 'episode_5'})
    """
    import re

    def extract_episode_index(name):
        """Extract the episode index from the name."""
        match = re.search(r"(\d+)", name)
        if match:
            return int(match.group(1))
        return 0

    # Create a mapping of episode index to episode name
    episode_map = {}
    for name in names:
        idx = extract_episode_index(name)
        episode_map[idx] = name

    return episode_map


def find_hdf5_file(hdf5_folder: Path, task_suite: str, task_id: int) -> Path | None:
    """Find the HDF5 file for the given task_suite and task_id.

    Args:
        hdf5_folder: Path to the folder containing HDF5 files
        task_suite: Task suite name (e.g., "libero_10", "xhumanoid")
        task_id: Task ID number

    Returns:
        Path to the HDF5 file if found, None otherwise
    """
    if not hdf5_folder.exists():
        print(f"HDF5 folder does not exist: {hdf5_folder}")
        return None

    # Create pattern to match the HDF5 file
    pattern = f"{task_suite}_task{task_id}_*_demo.hdf5"

    # Find matching files
    matching_files = list(hdf5_folder.glob(pattern))

    if matching_files:
        hdf5_file = matching_files[0]
        print(f"Found HDF5 file: {hdf5_file}")
        return hdf5_file
    else:
        print(f"No HDF5 file found matching pattern: {pattern}")
        print(f"Searched in: {hdf5_folder}")
        # List available files for debugging
        available_files = list(hdf5_folder.glob("*.hdf5"))
        if available_files:
            print("Available HDF5 files:")
            for file in available_files:
                print(f"  - {file.name}")
        return None


def run_closed_loop_policy(  # noqa: C901
    args: OpenpiClientArguments,
    simulation_app: SimulationApp,
    env: gym.Env,
    env_cfg: ManagerBasedRLEnvCfg,
    success_term: Callable[[gym.Env], bool] | None,
):
    """Run the closed loop policy evaluation."""
    tactile_buf = _OnlineTactileBuffer(
        tactile_sensors=args.tactile_sensor_names,
        tactile_output_type=args.tactile_output_type,
        tactile_history_len=args.tactile_history_len,
        force_history_len=args.force_history_len,
        marker_history_len=args.marker_history_len,
    )
    env_action_dim = int(np.prod(env.action_space.shape))
    policy_action_debug = _env_flag("SOFTVTBENCH_PRINT_POLICY_ACTIONS", False)
    policy_action_debug_stdout = _env_flag("SOFTVTBENCH_PRINT_POLICY_ACTIONS_STDOUT", True)
    try:
        policy_action_debug_every = max(1, int(os.environ.get("SOFTVTBENCH_PRINT_POLICY_ACTIONS_EVERY", "1")))
    except Exception:
        policy_action_debug_every = 1
    try:
        policy_action_debug_rows = max(1, int(os.environ.get("SOFTVTBENCH_PRINT_POLICY_ACTIONS_ROWS", "3")))
    except Exception:
        policy_action_debug_rows = 3
    if policy_action_debug:
        print(
            "[PolicyActionDebug] enabled "
            f"every={policy_action_debug_every} rows={policy_action_debug_rows} "
            f"stdout={policy_action_debug_stdout}",
            flush=True,
        )
    gripper_controller = os.environ.get("SOFTVTBENCH_EVAL_GRIPPER_CONTROLLER", "").strip().lower()
    if gripper_controller in ("0", "false", "none", "off", "disable", "disabled"):
        gripper_controller = ""
    gripper_open_finger = float(os.environ.get("SOFTVTBENCH_EVAL_GRIPPER_OPEN_FINGER", "0.04"))
    gripper_close_norm_default = float(os.environ.get("SOFTVTBENCH_EVAL_GRIPPER_CLOSE_NORM", "0.49"))
    gripper_close_after_frame = int(os.environ.get("SOFTVTBENCH_EVAL_GRIPPER_CLOSE_AFTER_FRAME", "80"))
    gripper_open_after_frame = int(os.environ.get("SOFTVTBENCH_EVAL_GRIPPER_OPEN_AFTER_FRAME", "360"))
    gripper_close_finger_env = os.environ.get("SOFTVTBENCH_EVAL_GRIPPER_CLOSE_FINGER", "").strip()
    gripper_close_finger_override = float(gripper_close_finger_env) if gripper_close_finger_env else None
    tactile_gripper_ctrl = (
        _TactileGripperController(open_finger=gripper_open_finger)
        if gripper_controller in ("tactile_abs", "tactile", "marker_abs", "marker")
        else None
    )

    def _controller_norm_for_frame(frame: int) -> float:
        if gripper_controller in ("fixed_abs", "fixed"):
            return gripper_close_norm_default
        if gripper_controller in ("stage_abs", "stage", "script_abs"):
            if frame < gripper_close_after_frame:
                return 1.0
            if frame >= gripper_open_after_frame:
                return 1.0
            return gripper_close_norm_default
        return float("nan")

    def _controller_finger_from_norm(norm_open: float) -> float:
        if gripper_close_finger_override is not None and norm_open < 0.999:
            return float(np.clip(gripper_close_finger_override, 0.0, gripper_open_finger))
        return float(np.clip(norm_open * gripper_open_finger, 0.0, gripper_open_finger))

    if gripper_controller:
        print(
            "[EvalGripperController] "
            f"mode={gripper_controller} env_action_dim={env_action_dim} "
            f"close_norm={gripper_close_norm_default:.4f} open_finger={gripper_open_finger:.4f} "
            f"close_finger_override={gripper_close_finger_override} "
            f"close_after_frame={gripper_close_after_frame} open_after_frame={gripper_open_after_frame}"
        )
        if tactile_gripper_ctrl is not None:
            print(
                "[EvalGripperController] tactile params "
                f"contact_thr={tactile_gripper_ctrl.contact_thr:.3f} "
                f"safety_thr={tactile_gripper_ctrl.safety_thr:.3f} "
                f"close_step={tactile_gripper_ctrl.close_step:.5f} "
                f"open_step={tactile_gripper_ctrl.open_step:.5f} "
                f"min_finger={tactile_gripper_ctrl.min_finger:.5f} "
                f"release_min_frame={tactile_gripper_ctrl.release_min_frame} "
                f"pre_contact_mode={tactile_gripper_ctrl.pre_contact_mode} "
                f"safety_action={tactile_gripper_ctrl.safety_action} "
                f"release_require_contact_drop={tactile_gripper_ctrl.release_require_contact_drop}"
            )

    # debug_mode=1/2/3 才使用 debug_path 做本地 dump
    if args.debug_mode in (1, 2, 3):
        os.makedirs(args.debug_path, exist_ok=True)

    # debug_mode=5: 逐帧挤压力记录目录
    force_dump_dir: Path | None = None
    if args.debug_mode == 5:
        # 统一副词：推荐用 --prompt-adverb firmly/gently；若使用 --prompt-adverbs，则标记为 mixed
        adverb_tag = "mixed" if args.prompt_adverbs else _sanitize_dirname(args.prompt_adverb)
        force_dump_dir = (
            project_root
            / "benchmarks"
            / "softvtbench"
            / "gripper_force"
            / _sanitize_dirname(str(args.task_suite))
            / f"task_{int(args.task_id)}"
            / adverb_tag
        )
        force_dump_dir.mkdir(parents=True, exist_ok=True)

    # debug_mode=6: 保存相机+触觉 markers_rgb + 预测/实测夹持力（逐帧）
    capture_mode6_root: Path | None = None
    if args.debug_mode == 6:
        # 统一副词标签，便于不同 prompt 版本的对照
        adverb_tag = "mixed" if args.prompt_adverbs else _sanitize_dirname(args.prompt_adverb)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        capture_mode6_root = (
            Path(args.debug_path)
            / "capture_mode6"
            / _sanitize_dirname(str(args.task_suite))
            / f"task_{int(args.task_id)}"
            / adverb_tag
            / ts
        )
        capture_mode6_root.mkdir(parents=True, exist_ok=True)
        # 写一份 run 级别 meta，方便回溯配置
        try:
            meta = {
                "task_suite": args.task_suite,
                "task_id": int(args.task_id),
                "task": args.task,
                "control_mode": args.control_mode,
                "camera_names": list(args.camera_names),
                "tactile_sensor_names": list(args.tactile_sensor_names),
                "tactile_output_type": args.tactile_output_type,
                "debug_mode": int(args.debug_mode),
                "debug_path": str(args.debug_path),
                "prompt_adverb": (args.prompt_adverb or "").strip(),
                "prompt_adverbs": list(args.prompt_adverbs) if args.prompt_adverbs else [],
                "prompt_seed": int(args.prompt_seed),
                "timestamp": ts,
            }
            with open(capture_mode6_root / "run_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[DebugMode 6] Failed to write run_meta.json: {e}")

    # Hybrid 力–位混合在线可视化（仅在 debug_mode == 4 时启用）
    force_viz = None
    if args.debug_mode == 4 and "Isaac-Libero-Franka-Hybrid-" in args.task and args.num_envs == 1:
        try:
            # 复用 scripts/tools 中的调试可视化工具
            # 注意：此处从工程根目录下的 scripts.tools.common 导入，而不是相对路径的 common。
            from scripts.tools.common.force_position_debug_viz import ForcePositionDebugVisualizer

            force_viz = ForcePositionDebugVisualizer()
            print("[DebugMode 4] Enabled Hybrid force-position debug visualizer.")
        except Exception as e:
            print(f"[DebugMode 4] Failed to initialize ForcePositionDebugVisualizer: {e}")
            force_viz = None
    elif args.debug_mode == 4:
        print(
            "[DebugMode 4] Force-position visualization is only available for Hybrid environments "
            "with num_envs == 1. Skipping visualizer initialization."
        )

    successful_experiments = 0
    attempted_experiments = 0
    consecutive_failures = 0
    max_consecutive_failures = int(os.environ.get("SOFTVTBENCH_MAX_CONSECUTIVE_FAILURES", "0") or "0")

    # 统计：跨所有成功 experiment 的夹爪挤压力均值（仅在 Hybrid 环境中有效）
    succ_squeeze_pred_sum = 0.0
    succ_squeeze_meas_sum = 0.0
    succ_squeeze_count = 0

    # 统计：跨所有成功 experiment 的挤压力 / 加持力 metrics（仅在 Hybrid + 13D action 时有效）
    succ_metrics_count = 0
    succ_squeeze_max_sum = 0.0
    succ_app_max_sum = 0.0
    succ_app_mean_sum = 0.0

    # 统计：跨所有成功 experiment 的「实测」Top5% 最大挤压力 / 加持力及平均加持力
    succ_squeeze_max_meas_sum = 0.0
    succ_squeeze_max_meas_count = 0
    succ_ap_mean_meas_sum = 0.0
    succ_ap_mean_meas_count = 0
    succ_ap_max_meas_sum = 0.0
    succ_ap_max_meas_count = 0

    # Find HDF5 file based on task_suite and task_id
    hdf5_file = find_hdf5_file(args.hdf5_folder, args.task_suite, args.task_id)

    # Load dataset and episode information if HDF5 file is found
    episode_indices_to_use = []
    episode_map = {}
    dataset_file_handler = None

    if hdf5_file and hdf5_file.exists():
        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(str(hdf5_file))
        episode_count = dataset_file_handler.get_num_episodes()
        episode_map = get_episode_map(dataset_file_handler.get_episode_names())
        # Use actual episode indices from episode_map instead of assuming they're consecutive
        episode_indices_to_use = sorted(episode_map.keys())
        print(f"Loaded {episode_count} initial_states of episodes from dataset: {hdf5_file}")
        print(f"Available episode indices: {episode_indices_to_use}")
    else:
        print(
            f"No valid HDF5 file found for {args.task_suite}_task{args.task_id}, will use default reset for all"
            " experiments"
        )

    # Read language instruction from task_suite_config as a fallback.
    # If the user provided --language-instruction, do NOT override it.
    task_config_path = args.task_config_path / f"{args.task_suite}.json"
    if not task_config_path.exists():
        raise FileNotFoundError(f"Task config file not found: {task_config_path}")
    with open(task_config_path) as f:
        task_suite_config = json.load(f)

    cli_instruction = (args.language_instruction or "").strip()
    if cli_instruction:
        print(f"\nUsing language instruction (from CLI): {cli_instruction}")
        args.language_instruction = cli_instruction
    else:
        for task in task_suite_config["tasks"]:
            task_id = task["task_id"]
            if task_id == args.task_id:
                args.language_instruction = task["language_instruction"]
                print(f"\nUsing language instruction (from task config): {args.language_instruction}")
                break

    replay_actions: np.ndarray | None = None
    if args.replay_action_hdf5 is not None:
        replay_path = Path(args.replay_action_hdf5)
        with h5py.File(replay_path, "r") as f:
            replay_actions = np.asarray(f["actions"], dtype=np.float32)
        print(f"[ReplayActions] Loaded {replay_actions.shape} actions from {replay_path}")
        client = None
    else:
        client = _websocket_client_policy.WebsocketClientPolicy(args.server_host, args.server_port)

    # Debug-only ablation: replace selected action dimensions by an HDF5 expert replay.
    # This isolates whether failures come from xyz trajectory vs gripper/force/orientation.
    override_actions: np.ndarray | None = None
    override_dims: list[int] = []
    override_path_raw = os.environ.get("SOFTVTBENCH_ACTION_OVERRIDE_HDF5", "").strip()
    override_dims_raw = os.environ.get("SOFTVTBENCH_ACTION_OVERRIDE_DIMS", "").strip()
    if override_path_raw and override_dims_raw:
        if override_path_raw.lower() in ("selected", "auto"):
            override_path_raw = os.environ.get("SOFTVTBENCH_SELECTED_HDF5_SOFT", "").strip()
        override_path = Path(override_path_raw)
        override_dims = [int(x) for x in override_dims_raw.replace(",", " ").split()]
        with h5py.File(override_path, "r") as f:
            override_actions = np.asarray(f["actions"], dtype=np.float32)
        print(f"[ActionOverrideDims] Loaded {override_actions.shape} from {override_path}; dims={override_dims}")

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        for exp_idx in range(args.num_total_experiments):
            print(f"\n[{exp_idx + 1}/{args.num_total_experiments}] Starting experiment...", end=" ", flush=True)
            success_step_count = 0
            experiment_success = False
            total_steps_taken = 0

            # 当前 experiment 的挤压力统计（均值）
            exp_fsq_pred_sum = 0.0
            exp_fsq_meas_sum = 0.0
            exp_fsq_count = 0

            # 当前 experiment 的逐帧挤压力 / 加持力记录（用于 Top5% 统计）
            exp_fsq_pred_values: list[float] = []
            exp_fsq_meas_values: list[float] = []
            exp_ap_pred_values: list[float] = []
            exp_ap_meas_values: list[float] = []
            # binary 模式：额外缓存逐步左右指 3D 力序列（用于严格复用 metrics.py 的统计定义）
            exp_fL_meas_values: list[np.ndarray] = []
            exp_fR_meas_values: list[np.ndarray] = []

            # 当前 experiment 的 Hybrid 13D 动作缓存（仅在 control_mode == "hybrid" 时使用）
            exp_actions_13d: list[torch.Tensor] = []

            # debug_mode=6: per-experiment capture directories + force log (JSONL)
            mode6_exp_dir: Path | None = None
            mode6_cam_dir: Path | None = None
            mode6_tac_dir: Path | None = None
            mode6_force_fh = None
            if capture_mode6_root is not None:
                try:
                    mode6_exp_dir = capture_mode6_root / f"exp_{exp_idx:03d}"
                    mode6_cam_dir = mode6_exp_dir / "camera_rgb"
                    mode6_tac_dir = mode6_exp_dir / "tactile_markers_rgb"
                    mode6_cam_dir.mkdir(parents=True, exist_ok=True)
                    mode6_tac_dir.mkdir(parents=True, exist_ok=True)
                    mode6_force_fh = open(mode6_exp_dir / "forces.jsonl", "w", encoding="utf-8")
                except Exception as e:
                    print(f"[DebugMode 6] Failed to init exp dir/log for exp_{exp_idx:03d}: {e}")
                    mode6_exp_dir = mode6_cam_dir = mode6_tac_dir = None
                    mode6_force_fh = None

            # 每个 experiment 开始时重置力–位可视化
            if force_viz is not None:
                try:
                    force_viz.reset()
                except Exception:
                    pass

            # reset environment with initial state from HDF5 if available
            if episode_indices_to_use:
                # Use episode index from the list (cycling through all episodes)
                episode_index = episode_indices_to_use[exp_idx % len(episode_indices_to_use)]
                episode_data = dataset_file_handler.load_episode(episode_map[episode_index], env.unwrapped.device)

                if "initial_state" in episode_data.data:
                    # reset environment
                    obs, info = env.reset()
                    # Set initial state for the environment
                    initial_state = episode_data.get_initial_state()
                    # print("---- initial_state: ", initial_state)
                    obs, info = env.reset_to(
                        initial_state, torch.arange(args.num_envs, device=env.unwrapped.device), is_relative=True
                    )

                else:
                    # Fallback to default reset if no initial state available
                    obs, info = env.reset()
            else:
                # Fallback to default reset if no dataset file specified or doesn't exist
                obs, info = env.reset()

            # Optionally re-apply the collection-time scene visuals (floor texture
            # overlay + dim reference DomeLight) so eval renders match training data.
            # Gated: only active when both env vars are set (spatial-soft eval).
            if os.environ.get("SOFTVTBENCH_APPLY_COLLECTION_SCENE_VISUALS", "0") == "1" and not globals().get(
                "_SCENE_VISUALS_APPLIED", False
            ):
                try:
                    import importlib.util as _ilu

                    _mod_path = os.environ.get("SOFTVTBENCH_SCENE_VISUALS_MODULE", "").strip()
                    if _mod_path:
                        _spec = _ilu.spec_from_file_location("softvtbench_collection_scene_visuals", _mod_path)
                        _mod = _ilu.module_from_spec(_spec)
                        _spec.loader.exec_module(_mod)
                        _mod.apply_collection_scene_visuals()
                        globals()["_SCENE_VISUALS_APPLIED"] = True
                    else:
                        print("[scene-visuals] SOFTVTBENCH_SCENE_VISUALS_MODULE not set; skipping", flush=True)
                except Exception as _exc:
                    print(f"[scene-visuals] failed to apply: {_exc!r}", flush=True)

            fem_debug_goals = getattr(success_term, "params", {}).get("goals") if success_term is not None else None
            fem_reference_nodes: dict[str, torch.Tensor] = {}
            for fem_name in _deformable_names_from_env_and_goals(fem_debug_goals):
                asset_name, nodes = _get_scene_nodal_pos_w(env.unwrapped, [fem_name])
                if asset_name is not None and nodes is not None:
                    fem_reference_nodes[asset_name] = nodes

            # Reset online histories per experiment to match dataset windowing.
            tactile_buf.reset()
            if tactile_gripper_ctrl is not None:
                tactile_gripper_ctrl.reset()

            frame_count = 0
            gripper_hysteresis_active = False
            delayed_force_active = False
            delayed_force_ever_active = False
            tactile_force_input_active = False
            tactile_force_input_ever_active = False
            terminated = torch.tensor([False])  # Initialize to handle case where inner loop doesn't execute
            truncated = torch.tensor([False])

            # Build prompt once per experiment (SoftVTBench-style adverb augmentation).
            base_instruction = (args.language_instruction or "").strip()
            exp_adv = ""
            if args.prompt_adverbs:
                # Deterministic per experiment; include task identifiers for stability.
                key = f"{args.task_suite}:{args.task_id}:{exp_idx}"
                exp_adv = _choose_adverb(int(args.prompt_seed), key, tuple(args.prompt_adverbs))
            else:
                exp_adv = (args.prompt_adverb or "").strip()
            exp_prompt = _rewrite_instruction(
                base_instruction, exp_adv, seed=int(args.prompt_seed), key=f"{args.task_suite}:{args.task_id}:{exp_idx}"
            )

            # debug_mode=6: write per-experiment meta once prompt is decided
            if mode6_exp_dir is not None:
                try:
                    meta = {
                        "task_suite": args.task_suite,
                        "task_id": int(args.task_id),
                        "exp_idx": int(exp_idx),
                        "prompt_adverb": (args.prompt_adverb or "").strip(),
                        "prompt_adverbs": list(args.prompt_adverbs) if args.prompt_adverbs else [],
                        "adverb_used": exp_adv,
                        "prompt": exp_prompt,
                        "camera_names": list(args.camera_names),
                        "tactile_sensor_names": list(args.tactile_sensor_names),
                        "tactile_output_type_saved": "markers_rgb",
                    }
                    with open(mode6_exp_dir / "exp_meta.json", "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            for action_idx in range(args.max_inference_steps):
                # Get camera images from live cameras
                rgbs = []
                for cam_name in list(args.camera_names):
                    cam_id = cam_name.split("_")[0]
                    cam = env.unwrapped.scene[cam_name]
                    rgb = cam.data.output["rgb"]
                    rgb = resize_frames_with_padding(rgb, args.target_image_size, bgr_conversion=False, pad_img=True)
                    rgbs.append(rgb)

                    # 仅在 debug_mode=2/3 时保存相机帧到本地
                    if args.debug_mode in (2, 3):
                        rgb_np = (rgb * 255).astype(np.uint8) if rgb.dtype == np.float32 else rgb.copy()
                        cv2.imwrite(
                            str(f"{args.debug_path}/frame_{frame_count:04d}_{cam_id}.png"),
                            cv2.cvtColor(rgb_np[0], cv2.COLOR_RGB2BGR),
                        )

                # Run model inference to get predicted actions (for comparison or execution)
                inference_actions = None

                # pi0-style **task-space** observation for OpenPI:
                #   - base_state: [x, y, z, ax, ay, az, gripper_abs] -> 7D
                #   - hybrid   : base_state plus separate H×6 finger forces (sent via 'observation/gripper_force')
                #
                # Get current EEF pose from policy observations: (x, y, z, qw, qx, qy, qz)
                eef_pose = obs["policy"]["eef_pose"].cpu().numpy()
                eef_pose = np.squeeze(eef_pose, axis=0)  # (7,)
                pos = eef_pose[:3]                       # (3,)
                quat = eef_pose[3:7].copy()                # (4,) (w,x,y,z)
                # Match the object-soft training conversion: recorded HDF5 quaternions use q0 >= 0.
                # Live Isaac can return the equivalent -q branch, which would flip the rotvec near pi.
                if quat[0] < 0.0:
                    quat *= -1.0

                # Convert quaternion to axis-angle (ax, ay, az)
                axis_angle = quat2axisangle(quat.copy())  # (3,)

                # Gripper scalar: use first component of gripper_pos observation (abs position)
                gripper_pos = obs["policy"]["gripper_pos"].cpu().numpy()
                gripper_pos = np.squeeze(gripper_pos, axis=0)
                if gripper_pos.ndim == 1:
                    gripper_scalar = np.array([gripper_pos[0]], dtype=np.float32)
                else:
                    gripper_scalar = np.array([gripper_pos[0]], dtype=np.float32)

                # Base 7D state: [x, y, z, ax, ay, az, gripper_abs]
                task_state_7 = np.concatenate((pos, axis_angle, gripper_scalar), axis=0).astype(np.float32)

                # For Hybrid force–position control, compute finger force history (left/right, 3D each)
                tactile_buf.update_force(obs)

                # SoftVTBench ablation: the 20260618 training set feeds proxy tactile forces
                # (e.g. [0,+5.9,0, 0,-5.9,0]) to the policy, while live soft eval can expose
                # near-zero measured force because deformable contact sensors are skipped. This
                # optional override tests that train/eval input-force distribution mismatch.
                tactile_force_input_override = None
                tactile_force_input_active_now = False
                tactile_force_input_env = os.environ.get("SOFTVTBENCH_TACTILE_FORCE_INPUT_OVERRIDE_6D", "").strip()
                if tactile_force_input_env:
                    try:
                        vals = np.asarray([float(x) for x in tactile_force_input_env.replace(",", " " ).split()], dtype=np.float32)
                        if vals.shape == (6,):
                            gate_env = os.environ.get("SOFTVTBENCH_TACTILE_FORCE_INPUT_AFTER_STATE", "").strip()
                            gate_ok = True
                            gate_desc = "always"
                            if gate_env:
                                parts = [float(x) for x in gate_env.replace(",", " " ).split()]
                                g_thr = parts[0] if len(parts) >= 1 else 0.035
                                min_frame = int(parts[1]) if len(parts) >= 2 else 0
                                z_max = parts[2] if len(parts) >= 3 else None
                                gate_ok = int(frame_count) >= min_frame and abs(float(task_state_7[6])) <= float(g_thr)
                                if z_max is not None:
                                    gate_ok = gate_ok and float(task_state_7[2]) <= float(z_max)
                                gate_desc = f"state_g<= {g_thr}, frame>= {min_frame}, state_z_max={z_max}"
                            if gate_ok or bool(tactile_force_input_active):
                                tactile_force_input_active = True
                                tactile_force_input_ever_active = True
                                if len(tactile_buf._force_hist) > 0:
                                    tactile_buf._force_hist.pop()
                                tactile_buf._force_hist.append(vals.astype(np.float32))
                                tactile_force_input_override = vals.astype(np.float32)
                                tactile_force_input_active_now = True
                            if action_idx == 0 and frame_count == 0:
                                print(f"[TactileInputAblation] override policy tactile force input {vals.tolist()} after {gate_desc}")
                        else:
                            print(f"[TactileInputAblation] expected 6 values, got {vals.shape}: {tactile_force_input_env!r}")
                    except Exception as exc:
                        print(f"[TactileInputAblation] invalid SOFTVTBENCH_TACTILE_FORCE_INPUT_OVERRIDE_6D={tactile_force_input_env!r}: {exc}")

                _send_tactile_obs = args.control_mode in ("tactile", "binary") and (
                    not args.abs7d
                    or os.environ.get("SOFTVTBENCH_TACTILE_OBS_WITH_ABS7D", "0") == "1"
                )
                if _send_tactile_obs:
                    # Tactile modalities (SoftVTBench-style): tactile_image + tactile_gripper_force + tactile_marker_motion.
                    # In abs7d pure-vision ablations, do not send tactile observations to the policy server.
                    # SOFTVTBENCH_TACTILE_OBS_WITH_ABS7D=1 opts in to sending tactile obs for abs7d tactile policies.
                    try:
                        tactile_buf.update_tactile_frames(env, env_id=0)
                    except Exception:
                        pass
                    tactile_buf.update_marker_motion(obs)

                # All modes: state is pure task-space 7D; forces are sent separately for hybrid
                eef_pose_states = task_state_7

                image = _to_uint8_rgb(np.squeeze(rgbs[0], axis=0))
                wrist_image = _to_uint8_rgb(np.squeeze(rgbs[1], axis=0))

                # Print modified instruction once so you can verify what is sent to the server.
                if action_idx == 0 and (exp_idx == 0 or args.debug_mode > 0):
                    if exp_adv:
                        print(f"[Prompt] {exp_prompt}   (adverb='{exp_adv}')")
                    else:
                        print(f"[Prompt] {exp_prompt}")

                element = {
                    # Top-level keys (image / state) for OpenPI transforms
                    "image": image,
                    "wrist_image": wrist_image,
                    "state": eef_pose_states,
                    # Nested "observation/*" keys to keep SoftVTBench-style compatibility
                    "observation/image": image,
                    "observation/wrist_image": wrist_image,
                    "observation/state": eef_pose_states,
                    "prompt": exp_prompt,
                }
                if args.control_mode == "hybrid":
                    gf = tactile_buf.get_force_history()
                    if gf is not None:
                        # Duplicate both top-level and nested keys
                        element["gripper_force"] = gf
                        element["observation/gripper_force"] = gf
                elif _send_tactile_obs:
                    tac_img = tactile_buf.get_tactile_image()
                    tac_force = tactile_buf.get_force_history()
                    tac_mm = tactile_buf.get_marker_motion()
                    if tac_img is not None:
                        # OpenPI's Libero tactile policy expects `tactile_image` at top-level
                        element["tactile_image"] = tac_img
                        element["observation/tactile_image"] = tac_img
                    if tac_force is not None:
                        element["tactile_gripper_force"] = tac_force
                        element["observation/tactile_gripper_force"] = tac_force
                    if tac_mm is not None:
                        element["tactile_marker_motion"] = tac_mm
                        element["observation/tactile_marker_motion"] = tac_mm

                # Legacy compatibility: only add bytes-key aliases for servers that require them.
                # Newer OpenPI/JAX servers expect homogeneous str keys in pytrees.
                if os.environ.get("OPENPI_ADD_BYTES_KEY_ALIASES", "0") == "1":
                    _add_bytes_key_aliases(
                    element,
                    (
                        "image",
                        "wrist_image",
                        "state",
                        "prompt",
                        "gripper_force",
                        "tactile_image",
                        "tactile_gripper_force",
                        "tactile_marker_motion",
                        "observation/image",
                        "observation/wrist_image",
                        "observation/state",
                        "observation/gripper_force",
                        "observation/tactile_image",
                        "observation/tactile_gripper_force",
                        "observation/tactile_marker_motion",
                    ),
                )

                # Get action predictions from OpenPI
                # OpenPI outputs 32D (padded). We slice out the **effective** dims:
                #   - diffik/osc: first 7D   (x, y, z, rx, ry, rz, gripper)
                #   - hybrid/tactile: first 13D (x, y, z, rx, ry, rz, gripper, fL(3), fR(3))
                # Debug/eval stabilization: optionally override commanded axis-angle with a fixed
                # nominal orientation to isolate axis-angle discontinuity from translation/gripper behavior.
                aa_override_env = os.environ.get("SOFTVTBENCH_ACTION_AXIS_ANGLE_OVERRIDE", "").strip()
                delta_override_env = os.environ.get("SOFTVTBENCH_OUTPUT_DELTA_ACTION_DIMS", "").strip()
                if replay_actions is not None:
                    start = int(frame_count)
                    end = min(start + int(args.replan_steps), int(replay_actions.shape[0]))
                    if start >= int(replay_actions.shape[0]):
                        action_chunk = replay_actions[-1:, :]
                    else:
                        action_chunk = replay_actions[start:end, :]
                    if len(action_chunk) < args.replan_steps:
                        pad = np.repeat(action_chunk[-1:, :], args.replan_steps - len(action_chunk), axis=0)
                        action_chunk = np.concatenate([action_chunk, pad], axis=0)
                else:
                    action_chunk = client.infer(element)["actions"]
                raw_policy_action_chunk = np.asarray(action_chunk, dtype=np.float32).copy()
                policy_action_source = "replay_hdf5" if replay_actions is not None else "openpi_server"
                if delta_override_env:
                    try:
                        delta_dims = [int(x) for x in delta_override_env.replace(",", " ").split()]
                        if delta_dims:
                            action_chunk = np.array(action_chunk, copy=True)
                            max_dim = min(action_chunk.shape[1], task_state_7.shape[0])
                            for dim in delta_dims:
                                if 0 <= dim < max_dim:
                                    action_chunk[:, dim] += task_state_7[dim]
                            if action_idx == 0 and frame_count == 0:
                                print(f"[ActionDeltaOutput] Converted policy delta dims to absolute dims={delta_dims}")
                    except Exception as exc:
                        print(f"[ActionDeltaOutput] invalid SOFTVTBENCH_OUTPUT_DELTA_ACTION_DIMS={delta_override_env!r}: {exc}")
                if aa_override_env:
                    try:
                        aa_override = np.asarray([float(x) for x in aa_override_env.replace(",", " ").split()], dtype=np.float32)
                        if aa_override.shape == (3,) and action_chunk.shape[1] >= 6:
                            action_chunk = np.array(action_chunk, copy=True)
                            action_chunk[:, 3:6] = aa_override[None, :]
                    except Exception as exc:
                        print(f"[ActionOverride] invalid SOFTVTBENCH_ACTION_AXIS_ANGLE_OVERRIDE={aa_override_env!r}: {exc}")
                xyz_offset_env = os.environ.get("SOFTVTBENCH_ACTION_XYZ_OFFSET", "").strip()
                if xyz_offset_env:
                    try:
                        xyz_offset = np.asarray([float(x) for x in xyz_offset_env.replace(",", " ").split()], dtype=np.float32)
                        if xyz_offset.shape == (3,) and action_chunk.shape[1] >= 3:
                            action_chunk = np.array(action_chunk, copy=True)
                            action_chunk[:, :3] += xyz_offset[None, :]
                    except Exception as exc:
                        print(f"[ActionOffset] invalid SOFTVTBENCH_ACTION_XYZ_OFFSET={xyz_offset_env!r}: {exc}")
                if override_actions is not None and override_dims:
                    start = int(frame_count)
                    target_len = int(action_chunk.shape[0])
                    end = min(start + target_len, int(override_actions.shape[0]))
                    if start >= int(override_actions.shape[0]):
                        override_chunk = override_actions[-1:, :]
                    else:
                        override_chunk = override_actions[start:end, :]
                    if len(override_chunk) < target_len:
                        pad = np.repeat(override_chunk[-1:, :], target_len - len(override_chunk), axis=0)
                        override_chunk = np.concatenate([override_chunk, pad], axis=0)
                    action_chunk = np.array(action_chunk, copy=True)
                    max_dim = min(action_chunk.shape[1], override_chunk.shape[1])
                    for dim in override_dims:
                        if 0 <= dim < max_dim:
                            action_chunk[:, dim] = override_chunk[:target_len, dim]

                # Debug/ablation knobs for SoftVTBench force/gripper issues.
                # Force dims follow 13D action: [7:10]=left xyz, [10:13]=right xyz.
                force_y_to_z_env = os.environ.get("SOFTVTBENCH_FORCE_Y_TO_Z", "").strip().lower()
                if force_y_to_z_env and action_chunk.shape[1] >= 13:
                    action_chunk = np.array(action_chunk, copy=True)
                    if force_y_to_z_env in ("neg_abs", "negative", "oldsign"):
                        action_chunk[:, 9] = -np.abs(action_chunk[:, 8])
                        action_chunk[:, 12] = -np.abs(action_chunk[:, 11])
                    else:
                        action_chunk[:, 9] = action_chunk[:, 8]
                        action_chunk[:, 12] = action_chunk[:, 11]
                    action_chunk[:, 8] = 0.0
                    action_chunk[:, 11] = 0.0
                    if action_idx == 0 and frame_count == 0:
                        print(f"[ForceAblation] SOFTVTBENCH_FORCE_Y_TO_Z={force_y_to_z_env}: moved force y components into z components")

                force_override_env = os.environ.get("SOFTVTBENCH_FORCE_OVERRIDE_6D", "").strip()
                if force_override_env and action_chunk.shape[1] >= 13:
                    try:
                        vals = np.asarray([float(x) for x in force_override_env.replace(",", " ").split()], dtype=np.float32)
                        if vals.shape == (6,):
                            delayed_force_env = os.environ.get("SOFTVTBENCH_FORCE_OVERRIDE_AFTER_GRIPPER", "").strip()
                            if delayed_force_env:
                                parts = [float(x) for x in delayed_force_env.replace(",", " ").split()]
                                close_thr = parts[0] if len(parts) >= 1 else 0.020
                                min_frame = int(parts[1]) if len(parts) >= 2 else 0
                                state_g_thr = parts[2] if len(parts) >= 3 else None
                                state_z_max = parts[3] if len(parts) >= 4 else None
                                g_pred = np.asarray(action_chunk[:, 6], dtype=np.float32)
                                local_active = bool(delayed_force_active)
                                mask = np.zeros(action_chunk.shape[0], dtype=bool)
                                lookahead_env = os.environ.get("SOFTVTBENCH_FORCE_OVERRIDE_LOOKAHEAD_STEPS", "").strip()
                                try:
                                    lookahead_steps = int(lookahead_env) if lookahead_env else int(args.replan_steps)
                                except Exception:
                                    lookahead_steps = int(args.replan_steps)
                                lookahead_steps = max(1, min(int(lookahead_steps), int(action_chunk.shape[0])))
                                state_gate_ok = True
                                try:
                                    if state_g_thr is not None:
                                        state_gate_ok = state_gate_ok and float(task_state_7[6]) <= float(state_g_thr)
                                    if state_z_max is not None:
                                        state_gate_ok = state_gate_ok and float(task_state_7[2]) <= float(state_z_max)
                                except Exception:
                                    state_gate_ok = False
                                # Only allow future steps that will actually be executed before the next replan to
                                # activate the delayed override; otherwise a 50-step horizon can turn force on early.
                                # Optional gates use current measured state: parts=[action_g_thr, min_frame, state_g_thr, state_z_max].
                                for gi in range(lookahead_steps):
                                    raw_g = g_pred[gi]
                                    global_frame = int(frame_count) + gi
                                    if (not local_active) and state_gate_ok and global_frame >= min_frame and float(raw_g) <= close_thr:
                                        local_active = True
                                    mask[gi] = local_active
                                if local_active and lookahead_steps < int(action_chunk.shape[0]):
                                    mask[lookahead_steps:] = True
                                delayed_force_active = local_active
                                delayed_force_ever_active = bool(delayed_force_ever_active or mask.any())
                                if mask.any():
                                    action_chunk = np.array(action_chunk, copy=True)
                                    action_chunk[mask, 7:13] = vals[None, :]
                                if action_idx == 0 and frame_count == 0:
                                    print(
                                        f"[ForceAblation] delayed force override after action_gripper<= {close_thr}, "
                                        f"frame>= {min_frame}, state_g_thr={state_g_thr}, state_z_max={state_z_max}; vals={vals.tolist()}"
                                    )
                            else:
                                action_chunk = np.array(action_chunk, copy=True)
                                action_chunk[:, 7:13] = vals[None, :]
                                if action_idx == 0 and frame_count == 0:
                                    print(f"[ForceAblation] SOFTVTBENCH_FORCE_OVERRIDE_6D={vals.tolist()}")
                        else:
                            print(f"[ForceAblation] expected 6 values in SOFTVTBENCH_FORCE_OVERRIDE_6D, got {vals.shape}: {force_override_env!r}")
                    except Exception as exc:
                        print(f"[ForceAblation] invalid SOFTVTBENCH_FORCE_OVERRIDE_6D={force_override_env!r}: {exc}")

                gripper_override_env = os.environ.get("SOFTVTBENCH_GRIPPER_OVERRIDE", "").strip()
                if gripper_override_env and action_chunk.shape[1] >= 7:
                    try:
                        g = float(gripper_override_env)
                        action_chunk = np.array(action_chunk, copy=True)
                        action_chunk[:, 6] = g
                        if action_idx == 0 and frame_count == 0:
                            print(f"[GripperAblation] SOFTVTBENCH_GRIPPER_OVERRIDE={g}")
                    except Exception as exc:
                        print(f"[GripperAblation] invalid SOFTVTBENCH_GRIPPER_OVERRIDE={gripper_override_env!r}: {exc}")

                gripper_clamp_env = os.environ.get("SOFTVTBENCH_GRIPPER_CLAMP_MAX", "").strip()
                if gripper_clamp_env and action_chunk.shape[1] >= 7:
                    try:
                        gmax = float(gripper_clamp_env)
                        action_chunk = np.array(action_chunk, copy=True)
                        action_chunk[:, 6] = np.minimum(action_chunk[:, 6], gmax)
                        if action_idx == 0 and frame_count == 0:
                            print(f"[GripperAblation] SOFTVTBENCH_GRIPPER_CLAMP_MAX={gmax}")
                    except Exception as exc:
                        print(f"[GripperAblation] invalid SOFTVTBENCH_GRIPPER_CLAMP_MAX={gripper_clamp_env!r}: {exc}")

                gripper_hysteresis_env = os.environ.get("SOFTVTBENCH_GRIPPER_HYSTERESIS", "").strip()
                if gripper_hysteresis_env and action_chunk.shape[1] >= 7:
                    try:
                        vals = [float(x) for x in gripper_hysteresis_env.replace(",", " " ).split()]
                        close_thr = vals[0] if len(vals) >= 1 else 0.020
                        open_thr = vals[1] if len(vals) >= 2 else 0.033
                        close_value = vals[2] if len(vals) >= 3 else 0.006
                        g_pred = np.asarray(action_chunk[:, 6], dtype=np.float32)
                        action_chunk = np.array(action_chunk, copy=True)
                        adjusted_g = action_chunk[:, 6].copy()
                        active = bool(gripper_hysteresis_active)
                        for gi, raw_g in enumerate(g_pred):
                            if active and float(raw_g) >= open_thr:
                                active = False
                            if (not active) and float(raw_g) <= close_thr:
                                active = True
                            if active:
                                adjusted_g[gi] = min(float(adjusted_g[gi]), close_value)
                        gripper_hysteresis_active = active
                        action_chunk[:, 6] = adjusted_g
                        if action_idx == 0 and frame_count == 0:
                            print(
                                f"[GripperAblation] SOFTVTBENCH_GRIPPER_HYSTERESIS close_thr={close_thr} "
                                f"open_thr={open_thr} close_value={close_value}"
                            )
                    except Exception as exc:
                        print(f"[GripperAblation] invalid SOFTVTBENCH_GRIPPER_HYSTERESIS={gripper_hysteresis_env!r}: {exc}")
                final_policy_action_chunk = np.asarray(action_chunk, dtype=np.float32).copy()
                assert len(action_chunk) >= args.replan_steps, (
                    f"We want to replan every {args.replan_steps} steps, but policy only predicts"
                    f" {len(action_chunk)} steps."
                )

                if args.control_mode in ("hybrid", "tactile"):
                    # Hybrid force–position + binary gripper control:
                    #   [x, y, z, rx, ry, rz, gripper, fL(3), fR(3)]  -> 13D
                    n = action_chunk.shape[0]
                    d = action_chunk.shape[1]
                    if args.control_mode == "tactile" and args.abs7d:
                        if d < 7:
                            raise ValueError(
                                f"abs7d expects at least 7D actions from OpenPI, "
                                f"but got shape {action_chunk.shape}."
                            )
                        # Force ablation: ignore force outputs (even if present) and pad zeros to 13D.
                        zeros6 = np.zeros((n, 6), dtype=np.float32)
                        hybrid_actions = np.concatenate([action_chunk[:, :7].astype(np.float32), zeros6], axis=1)
                    else:
                        if d < 13:
                            raise ValueError(
                                f"Hybrid control_mode expects at least 13D actions from OpenPI, "
                                f"but got shape {action_chunk.shape}."
                            )
                        hybrid_actions = action_chunk[:, :13].astype(np.float32)  # (N, 13)
                    inference_actions = torch.from_numpy(hybrid_actions).float()
                    inference_actions = inference_actions[: args.replan_steps, :]
                elif args.control_mode == "osc":
                    # OSC env action shape is 7D:
                    #   Input from OpenPI: (x, y, z, rx, ry, rz, gripper)
                    #   Output to env:     (x, y, z, rx, ry, rz, gripper)
                    if action_chunk.shape[1] < 7:
                        raise ValueError(
                            f"osc control_mode expects at least 7D actions from OpenPI, "
                            f"but got shape {action_chunk.shape}."
                        )
                    inference_actions = torch.from_numpy(action_chunk[:, :7].astype(np.float32)).float()
                    inference_actions = inference_actions[: args.replan_steps, :]
                elif args.control_mode == "binary":
                    # IK + tactile (non-hybrid) with **binary** gripper:
                    #   Input from OpenPI: (x, y, z, rx, ry, rz, gripper) - 7D axis-angle
                    #   Output to env:     (x, y, z, qw, qx, qy, qz, gripper_binary) - 8D quaternion
                    #   If SOFTVTBENCH_GRIPPER_ACTION_MODE=abs and action_dim=9:
                    #     (x, y, z, qw, qx, qy, qz, left_finger, right_finger)
                    if action_chunk.shape[1] < 7:
                        raise ValueError(
                            f"binary control_mode expects at least 7D actions from OpenPI, "
                            f"but got shape {action_chunk.shape}."
                        )
                    action_chunk_7d = action_chunk[:, :7].astype(np.float32)
                    binary_gripper_raw = action_chunk_7d[:, 6].astype(np.float32).copy()
                    controller_gripper_norm = None
                    controller_gripper_finger = None
                    controller_kind = ""

                    eef_pose_quat = np.array([axisangle2quat(act[3:6]) for act in action_chunk_7d], dtype=np.float32)
                    if gripper_controller and env_action_dim == 9:
                        if gripper_controller in ("policy_abs", "policy", "model_abs", "model"):
                            controller_gripper_norm = np.full(
                                action_chunk_7d.shape[0],
                                np.nan,
                                dtype=np.float32,
                            )
                            controller_gripper_finger = np.clip(
                                action_chunk_7d[:, 6].astype(np.float32),
                                0.0,
                                gripper_open_finger,
                            )
                            controller_kind = "policy_abs_joint_position"
                            tactile_ctrl_diag = None
                        elif tactile_gripper_ctrl is not None:
                            controller_gripper_norm = np.full(
                                action_chunk_7d.shape[0],
                                np.nan,
                                dtype=np.float32,
                            )
                            policy_gripper_finger = np.clip(
                                action_chunk_7d[:, 6].astype(np.float32),
                                0.0,
                                gripper_open_finger,
                            )
                            controller_gripper_finger, tactile_ctrl_diag = tactile_gripper_ctrl.adjust_chunk(
                                policy_finger=policy_gripper_finger,
                                obs=obs,
                                frame_count=int(frame_count),
                            )
                            controller_kind = "tactile_marker_abs_joint_position"
                        else:
                            frames = np.arange(action_chunk_7d.shape[0], dtype=np.int32) + int(frame_count)
                            controller_gripper_norm = np.asarray(
                                [_controller_norm_for_frame(int(fr)) for fr in frames],
                                dtype=np.float32,
                            )
                            controller_gripper_finger = np.asarray(
                                [_controller_finger_from_norm(float(g)) for g in controller_gripper_norm],
                                dtype=np.float32,
                            )
                            controller_kind = "abs_joint_position"
                            tactile_ctrl_diag = None
                        eef_pose_with_gripper = np.concatenate(
                            (
                                action_chunk_7d[:, :3],
                                eef_pose_quat,
                                controller_gripper_finger.reshape(-1, 1),
                                controller_gripper_finger.reshape(-1, 1),
                            ),
                            axis=1,
                        )  # (N, 9)
                        gripper_threshold = float("nan")
                        gripper_threshold_kind = controller_kind
                        if action_idx == 0 and frame_count == 0:
                            print(
                                f"[EvalGripperController] using {controller_kind} "
                                f"norm_first={controller_gripper_norm[: args.replan_steps].tolist()} "
                                f"finger_first={controller_gripper_finger[: args.replan_steps].tolist()}"
                            )
                    else:
                        # Binarize gripper:
                        # - IsaacLab BinaryJointAction uses positive=open, negative=close.
                        # - Some OpenPI policies output Franka finger targets in meters
                        #   (roughly 0.04=open, 0.005=closed), not probabilities.
                        g = action_chunk_7d[:, 6]
                        g_min = float(np.nanmin(g))
                        g_max = float(np.nanmax(g))
                        threshold_env = os.environ.get("SOFTVTBENCH_BINARY_GRIPPER_THRESHOLD", "").strip()
                        physical_threshold_env = os.environ.get("SOFTVTBENCH_BINARY_GRIPPER_PHYSICAL_THRESHOLD", "0.02").strip()
                        if threshold_env:
                            gripper_threshold = float(threshold_env)
                            gripper_threshold_kind = "override"
                            g_bin = np.where(g >= gripper_threshold, 1.0, -1.0).astype(np.float32)
                        elif g_min >= -1e-4 and g_max <= 0.08:
                            gripper_threshold = float(physical_threshold_env)
                            gripper_threshold_kind = "physical_finger_target"
                            g_bin = np.where(g >= gripper_threshold, 1.0, -1.0).astype(np.float32)
                        elif np.all(g >= 0.0) and np.all(g <= 1.0):
                            gripper_threshold = 0.5
                            gripper_threshold_kind = "unit_interval"
                            g_bin = np.where(g >= gripper_threshold, 1.0, -1.0).astype(np.float32)
                        else:
                            gripper_threshold = 0.0
                            gripper_threshold_kind = "signed"
                            g_bin = np.where(g >= 0.0, 1.0, -1.0).astype(np.float32)
                        if gripper_controller and env_action_dim != 9 and action_idx == 0 and frame_count == 0:
                            print(
                                "[EvalGripperController] requested but env_action_dim is not 9; "
                                "falling back to binary gripper."
                            )
                        if action_idx == 0 and frame_count == 0:
                            unique_bin = sorted(float(x) for x in np.unique(g_bin))
                            print(
                                "[BinaryGripper] "
                                f"kind={gripper_threshold_kind} threshold={gripper_threshold:.6f} "
                                f"raw_min={g_min:.6f} raw_max={g_max:.6f} bin_values={unique_bin} "
                                "(positive=open, negative=close)"
                            )
                        eef_pose_with_gripper = np.concatenate(
                            (action_chunk_7d[:, :3], eef_pose_quat, g_bin.reshape(-1, 1)), axis=1
                        )  # (N, 8)
                    inference_actions = torch.from_numpy(eef_pose_with_gripper).float()
                    inference_actions = inference_actions[: args.replan_steps, :]
                else:
                    # DiffIK task-space control:
                    #   Input from OpenPI: (x, y, z, rx, ry, rz, gripper) - 7D axis-angle
                    #   Output to env:     (x, y, z, qw, qx, qy, qz, gripper) - 8D quaternion
                    if action_chunk.shape[1] < 7:
                        raise ValueError(
                            f"diffik control_mode expects at least 7D actions from OpenPI, "
                            f"but got shape {action_chunk.shape}."
                        )
                    action_chunk_7d = action_chunk[:, :7]
                    eef_pose_quat = np.array([axisangle2quat(act[3:6]) for act in action_chunk_7d])
                    eef_pose_with_gripper = np.concatenate(
                        (action_chunk_7d[:, :3], eef_pose_quat, action_chunk_7d[:, 6:7]), axis=1
                    )  # (N, 8)
                    inference_actions = torch.from_numpy(eef_pose_with_gripper).float()
                    inference_actions = inference_actions[: args.replan_steps, :]

                # Execute inference actions
                action = inference_actions
                if policy_action_debug and (action_idx % policy_action_debug_every == 0):
                    executed_action = action.detach().cpu().numpy()
                    raw_shape = list(raw_policy_action_chunk.shape)
                    final_shape = list(final_policy_action_chunk.shape)
                    exec_shape = list(executed_action.shape)
                    record = {
                        "exp_idx": int(exp_idx),
                        "action_idx": int(action_idx),
                        "frame_count": int(frame_count),
                        "source": policy_action_source,
                        "task_suite": str(args.task_suite),
                        "task_id": int(args.task_id),
                        "control_mode": str(args.control_mode),
                        "abs7d": bool(args.abs7d),
                        "env_action_dim": int(env_action_dim),
                        "replan_steps": int(args.replan_steps),
                        "state_7": _arr_preview(task_state_7, cols=7),
                        "state_xyz": _arr_preview(task_state_7[:3], cols=3),
                        "state_axis_angle": _arr_preview(task_state_7[3:6], cols=3),
                        "state_gripper": float(np.round(float(task_state_7[6]), 6)),
                        "raw_policy_shape": raw_shape,
                        "raw_policy_first_rows": _arr_preview(
                            raw_policy_action_chunk,
                            rows=policy_action_debug_rows,
                            cols=min(13, raw_policy_action_chunk.shape[1] if raw_policy_action_chunk.ndim == 2 else 13),
                        ),
                        "raw_policy_delta": _delta_summary(raw_policy_action_chunk, task_state_7),
                        "final_policy_shape": final_shape,
                        "final_policy_first_rows": _arr_preview(
                            final_policy_action_chunk,
                            rows=policy_action_debug_rows,
                            cols=min(13, final_policy_action_chunk.shape[1] if final_policy_action_chunk.ndim == 2 else 13),
                        ),
                        "final_policy_delta": _delta_summary(final_policy_action_chunk, task_state_7),
                        "exec_action_shape": exec_shape,
                        "exec_action_first_rows": _arr_preview(
                            executed_action,
                            rows=policy_action_debug_rows,
                            cols=min(13, executed_action.shape[1] if executed_action.ndim == 2 else 13),
                        ),
                    }
                    try:
                        if args.control_mode == "binary" and "controller_kind" in locals():
                            record["controller_kind"] = str(controller_kind)
                            if controller_gripper_finger is not None:
                                record["controller_gripper_finger_first_rows"] = _arr_preview(
                                    controller_gripper_finger,
                                    rows=policy_action_debug_rows,
                                    cols=policy_action_debug_rows,
                                )
                            if controller_gripper_norm is not None:
                                record["controller_gripper_norm_first_rows"] = _arr_preview(
                                    controller_gripper_norm,
                                    rows=policy_action_debug_rows,
                                    cols=policy_action_debug_rows,
                                )
                    except Exception:
                        pass
                    _write_policy_action_debug(
                        debug_path=args.debug_path,
                        enabled=True,
                        stdout=policy_action_debug_stdout,
                        record=record,
                    )

                # 仅在 debug_mode 1/2/3 时保存动作
                if args.debug_mode in (1, 2, 3):
                    np.save(str(f"{args.debug_path}/action_{frame_count:04d}.npy"), action.cpu().numpy())

                # Execute actions step by step
                # NOTE: We limit to the actual number of actions we have (might be less than replan_steps)
                num_actions_to_execute = min(action.shape[0], args.replan_steps)
                for i in range(num_actions_to_execute):
                    obs, reward, terminated, truncated, info = env.step(action[i].reshape([1, -1]))

                    # 若为 Hybrid 控制模式，则缓存 13D 动作以便后续计算 metrics
                    if args.control_mode in ("hybrid", "tactile"):
                        try:
                            if action[i].shape[-1] == 13:
                                exp_actions_13d.append(action[i].detach().cpu())
                        except Exception:
                            pass

                    # 从 ForcePositionAction.debug_info 统计当前 step 的挤压力（若可用）
                    try:
                        term = env.action_manager.get_term("arm_action")
                        debug = getattr(term, "debug_info", None)
                    except Exception:
                        debug = None
                    if debug:
                        try:
                            f_sq_pred = float(debug.get("f_sq_pred", 0.0))
                            f_sq_meas = float(debug.get("f_sq_meas", 0.0))
                            exp_fsq_pred_sum += f_sq_pred
                            exp_fsq_meas_sum += f_sq_meas
                            exp_fsq_count += 1
                            exp_fsq_pred_values.append(f_sq_pred)
                            exp_fsq_meas_values.append(f_sq_meas)

                            # 加持力模长（在 base frame 下），用于 Ap 相关统计
                            ap_pred = debug.get("F_app_norm_pred", None)
                            ap_meas = debug.get("F_app_norm_meas", None)
                            try:
                                if ap_pred is not None:
                                    exp_ap_pred_values.append(float(ap_pred))
                                if ap_meas is not None:
                                    exp_ap_meas_values.append(float(ap_meas))
                            except Exception:
                                pass
                        except Exception:
                            pass
                    elif args.control_mode == "binary":
                        # Binary (IK+tactile) env doesn't use ForcePositionAction, so `debug_info` may be absent.
                        # For compatibility with existing evaluation parsers:
                        # - We still track squeeze_pred/squeeze_meas, but set pred to 0.0 (sentinel; should be ignored).
                        # - We additionally track applied force magnitude (ap_meas) from `gripper_net_force`,
                        #   using the EXACT same definition as ForcePositionAction / benchmarks.common.metrics.
                        try:
                            gnf = obs["policy"]["gripper_net_force"]  # (N, H=1, 2, 3) typically
                            # pick env0, current frame 0: (2,3)
                            f_lr = gnf[0, 0].detach().cpu().numpy().astype(np.float32)
                            f_left = f_lr[0]
                            f_right = f_lr[1]
                            # Reuse the canonical hybrid metric definition:
                            # - squeeze: 2*min(|fL_z|,|fR_z|)
                            # - applied force vector: Fx=fLx+fRx, Fy=fLy+fRy,
                            #   Fz=a+b-common*(sign(a)+sign(b)), then ap = ||F_app||_2
                            series = compute_contact_force_series_from_lr_forces(
                                fL=np.asarray([f_left], dtype=np.float32),
                                fR=np.asarray([f_right], dtype=np.float32),
                            )
                            squeeze_meas = float(series.squeeze[0])
                            ap_meas = float(series.external_norm[0])

                            # pred placeholders (0.0) for log/regex compatibility
                            squeeze_pred = 0.0
                            ap_pred = 0.0

                            exp_fsq_pred_sum += squeeze_pred
                            exp_fsq_meas_sum += squeeze_meas
                            exp_fsq_count += 1
                            exp_fsq_pred_values.append(squeeze_pred)
                            exp_fsq_meas_values.append(squeeze_meas)

                            exp_ap_pred_values.append(ap_pred)
                            exp_ap_meas_values.append(ap_meas)

                            # Keep raw 3D forces for strict per-episode metrics aggregation.
                            exp_fL_meas_values.append(np.asarray(f_left, dtype=np.float32))
                            exp_fR_meas_values.append(np.asarray(f_right, dtype=np.float32))
                        except Exception:
                            # If force obs is missing, skip silently (do not break main loop).
                            pass

                        # debug_mode=4: 在线更新 Hybrid 力–位混合可视化
                        if force_viz is not None and args.debug_mode == 4:
                            try:
                                force_viz.update(debug)
                            except Exception:
                                # 可视化失败不应中断主流程
                                pass

                    total_steps_taken += 1

                    if terminated[0] or truncated[0]:
                        experiment_success = False
                        break

                    if success_term is not None:
                        if bool(success_term.func(env, **success_term.params)[0]):
                            success_step_count += 1
                            if success_step_count >= args.num_success_steps:
                                experiment_success = True
                                break
                        else:
                            success_step_count = 0

                    # debug_mode=6: dump camera RGB + tactile markers_rgb + (pred/meas) gripper position + (pred/meas) squeeze
                    if mode6_force_fh is not None and mode6_cam_dir is not None and mode6_tac_dir is not None:
                        try:
                            # --- Images (post-step) ---
                            for cam_name in list(args.camera_names):
                                cam_id = cam_name.split("_")[0]
                                cam = env.unwrapped.scene[cam_name]
                                rgb = cam.data.output["rgb"][0]
                                rgb_u8 = _to_uint8_rgb(rgb)
                                cv2.imwrite(
                                    str(mode6_cam_dir / f"frame_{frame_count:04d}_{cam_id}.png"),
                                    cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR),
                                )

                            # tactile markers_rgb (left/right)
                            for tac_name in list(args.tactile_sensor_names):
                                try:
                                    sensor = env.unwrapped.scene.sensors[tac_name]
                                    outputs = sensor.data.output
                                    if "markers_rgb" not in outputs:
                                        continue
                                    tac_img = outputs["markers_rgb"][0]
                                    tac_u8 = _to_uint8_rgb(tac_img)
                                    cv2.imwrite(
                                        str(mode6_tac_dir / f"frame_{frame_count:04d}_{tac_name}_markers_rgb.png"),
                                        cv2.cvtColor(tac_u8, cv2.COLOR_RGB2BGR),
                                    )
                                except Exception:
                                    # tactile sensor may not exist in non-tactile envs
                                    continue

                            # --- Scalars (pred/meas) ---
                            # gripper_cmd: from executed action (1D)
                            # gripper_meas: from obs["policy"]["gripper_pos"] (2,) -> mean (1D)
                            gripper_cmd = None
                            gripper_meas = None

                            # squeeze_pred: prefer ForcePositionAction.debug_info if available, else derive from 13D action forces
                            # squeeze_meas: prefer ForcePositionAction.debug_info if available, else derive from obs["policy"]["gripper_net_force"]
                            squeeze_pred = None
                            squeeze_meas = None

                            # commanded action from executed action (shape depends on control_mode)
                            a_np = None
                            try:
                                a_np = action[i].detach().cpu().numpy().astype(np.float32)
                                # - hybrid/tactile: 13D => gripper at index 6
                                # - diffik/osc/binary: 8D => gripper at index 7
                                if a_np.shape[-1] >= 13:
                                    gripper_cmd = float(a_np[6])
                                elif a_np.shape[-1] >= 8:
                                    gripper_cmd = float(a_np[7])
                                elif a_np.shape[-1] >= 7:
                                    gripper_cmd = float(a_np[6])
                            except Exception:
                                pass

                            # measured gripper position and measured squeeze from policy obs
                            try:
                                policy_obs = obs.get("policy", {}) if isinstance(obs, dict) else {}
                                gp = policy_obs.get("gripper_pos", None)
                                gripper_meas_lr = None
                                gripper_meas_width = None
                                if gp is not None:
                                    gp0 = gp[0].detach().cpu().numpy().astype(np.float32).reshape(-1)
                                    if gp0.size > 0:
                                        gripper_meas_lr = gp0.tolist()
                                        gripper_meas = float(abs(gp0[0]))
                                        if gp0.size >= 2:
                                            gripper_meas_width = float(abs(gp0[0]) + abs(gp0[1]))
                                        else:
                                            gripper_meas_width = float(abs(gp0[0]))
                                gnf = policy_obs.get("gripper_net_force", None)
                                if gnf is not None:
                                    f_lr = gnf[0, 0].detach().cpu().numpy().astype(np.float32)  # (2,3)
                                    meas_series = compute_contact_force_series_from_lr_forces(
                                        fL=np.asarray([f_lr[0].copy()], dtype=np.float32),
                                        fR=np.asarray([f_lr[1].copy()], dtype=np.float32),
                                    )
                                    squeeze_meas = float(meas_series.squeeze[0])
                            except Exception:
                                pass

                            # Prefer debug_info squeeze values when available (matches existing reporting semantics)
                            try:
                                if debug:
                                    # These are scalars per-step
                                    squeeze_pred = float(debug.get("f_sq_pred", squeeze_pred or 0.0))
                                    squeeze_meas = float(debug.get("f_sq_meas", squeeze_meas or 0.0))
                            except Exception:
                                pass

                            # If no debug squeeze_pred but action is 13D, derive squeeze_pred from predicted forces
                            if squeeze_pred is None:
                                try:
                                    a_np = action[i].detach().cpu().numpy().astype(np.float32)
                                    if a_np.shape[-1] >= 13:
                                        fL_pred = a_np[7:10].copy()
                                        fR_pred = a_np[10:13].copy()
                                        pred_series = compute_contact_force_series_from_lr_forces(
                                            fL=np.asarray([fL_pred], dtype=np.float32),
                                            fR=np.asarray([fR_pred], dtype=np.float32),
                                        )
                                        squeeze_pred = float(pred_series.squeeze[0])
                                except Exception:
                                    pass

                            payload = {
                                "task_suite": args.task_suite,
                                "task_id": int(args.task_id),
                                "exp_idx": int(exp_idx),
                                "action_idx": int(action_idx),
                                "replan_i": int(i),
                                "frame": int(frame_count),
                                "success_step_count": int(success_step_count),
                                "success_now": bool(success_term.func(env, **success_term.params)[0]) if success_term is not None else None,
                                "gripper_cmd": gripper_cmd,
                                "gripper_cmd_lr": a_np[7:9].tolist() if a_np is not None and a_np.shape[-1] >= 9 else None,
                                "model_gripper_raw": float(binary_gripper_raw[i]) if args.control_mode == "binary" and 'binary_gripper_raw' in locals() and i < len(binary_gripper_raw) else None,
                                "model_gripper_threshold": float(gripper_threshold)
                                if args.control_mode == "binary"
                                and 'gripper_threshold' in locals()
                                and np.isfinite(gripper_threshold)
                                else None,
                                "model_gripper_threshold_kind": gripper_threshold_kind if args.control_mode == "binary" and 'gripper_threshold_kind' in locals() else None,
                                "controller_gripper_mode": gripper_controller if 'gripper_controller' in locals() and gripper_controller else None,
                                "controller_gripper_kind": controller_kind if args.control_mode == "binary" and 'controller_kind' in locals() and controller_kind else None,
                                "controller_gripper_norm": float(controller_gripper_norm[i])
                                if args.control_mode == "binary"
                                and 'controller_gripper_norm' in locals()
                                and controller_gripper_norm is not None
                                and i < len(controller_gripper_norm)
                                and np.isfinite(controller_gripper_norm[i])
                                else None,
                                "controller_gripper_finger": float(controller_gripper_finger[i]) if args.control_mode == "binary" and 'controller_gripper_finger' in locals() and controller_gripper_finger is not None and i < len(controller_gripper_finger) else None,
                                "tactile_ctrl_action": tactile_ctrl_diag.get("action") if 'tactile_ctrl_diag' in locals() and isinstance(tactile_ctrl_diag, dict) else None,
                                "tactile_ctrl_contact": tactile_ctrl_diag.get("contact") if 'tactile_ctrl_diag' in locals() and isinstance(tactile_ctrl_diag, dict) else None,
                                "tactile_ctrl_single_contact": tactile_ctrl_diag.get("single_contact") if 'tactile_ctrl_diag' in locals() and isinstance(tactile_ctrl_diag, dict) else None,
                                "tactile_ctrl_contact_seen": tactile_ctrl_diag.get("contact_seen") if 'tactile_ctrl_diag' in locals() and isinstance(tactile_ctrl_diag, dict) else None,
                                "tactile_ctrl_release_allowed": tactile_ctrl_diag.get("release_allowed") if 'tactile_ctrl_diag' in locals() and isinstance(tactile_ctrl_diag, dict) else None,
                                "tactile_ctrl_wants_release": tactile_ctrl_diag.get("wants_release") if 'tactile_ctrl_diag' in locals() and isinstance(tactile_ctrl_diag, dict) else None,
                                "tactile_ctrl_left_score": tactile_ctrl_diag.get("left_score") if 'tactile_ctrl_diag' in locals() and isinstance(tactile_ctrl_diag, dict) else None,
                                "tactile_ctrl_right_score": tactile_ctrl_diag.get("right_score") if 'tactile_ctrl_diag' in locals() and isinstance(tactile_ctrl_diag, dict) else None,
                                "tactile_ctrl_global_score": tactile_ctrl_diag.get("global_score") if 'tactile_ctrl_diag' in locals() and isinstance(tactile_ctrl_diag, dict) else None,
                                "tactile_ctrl_safety_over": tactile_ctrl_diag.get("safety_over") if 'tactile_ctrl_diag' in locals() and isinstance(tactile_ctrl_diag, dict) else None,
                                "tactile_ctrl_contact_drop": tactile_ctrl_diag.get("contact_drop") if 'tactile_ctrl_diag' in locals() and isinstance(tactile_ctrl_diag, dict) else None,
                                "tactile_ctrl_hold_finger": tactile_ctrl_diag.get("hold_finger") if 'tactile_ctrl_diag' in locals() and isinstance(tactile_ctrl_diag, dict) else None,
                                "tactile_ctrl_search_finger": tactile_ctrl_diag.get("search_finger") if 'tactile_ctrl_diag' in locals() and isinstance(tactile_ctrl_diag, dict) else None,
                                "tactile_ctrl_policy_first": tactile_ctrl_diag.get("policy_first") if 'tactile_ctrl_diag' in locals() and isinstance(tactile_ctrl_diag, dict) else None,
                                "tactile_ctrl_exec_first": tactile_ctrl_diag.get("exec_first") if 'tactile_ctrl_diag' in locals() and isinstance(tactile_ctrl_diag, dict) else None,
                                "tactile_ctrl_delta_first": tactile_ctrl_diag.get("delta_first") if 'tactile_ctrl_diag' in locals() and isinstance(tactile_ctrl_diag, dict) else None,
                                "gripper_meas": gripper_meas,
                                "gripper_meas_lr": gripper_meas_lr if 'gripper_meas_lr' in locals() else None,
                                "gripper_meas_width": gripper_meas_width if 'gripper_meas_width' in locals() else None,
                                "gripper_hysteresis_active": bool(gripper_hysteresis_active) if 'gripper_hysteresis_active' in locals() else False,
                                "delayed_force_active": bool(delayed_force_active) if 'delayed_force_active' in locals() else False,
                                "delayed_force_ever_active": bool(delayed_force_ever_active) if 'delayed_force_ever_active' in locals() else False,
                                "tactile_force_input_active": bool(tactile_force_input_active_now) if 'tactile_force_input_active_now' in locals() else False,
                                "tactile_force_input_ever_active": bool(tactile_force_input_ever_active) if 'tactile_force_input_ever_active' in locals() else False,
                                "tactile_force_input_override": tactile_force_input_override.tolist() if 'tactile_force_input_override' in locals() and tactile_force_input_override is not None else None,
                                "squeeze_pred": squeeze_pred,
                                "squeeze_meas": squeeze_meas,
                                "action": a_np.tolist() if a_np is not None else None,
                                "action_xyz": a_np[:3].tolist() if a_np is not None and a_np.shape[-1] >= 3 else None,
                                "action_axis_angle": a_np[3:6].tolist() if a_np is not None and a_np.shape[-1] >= 6 else None,
                                "state": task_state_7.tolist(),
                                "state_xyz": task_state_7[:3].tolist(),
                                "state_axis_angle": task_state_7[3:6].tolist(),
                            }
                            payload.update(_debug_fem_deformation_state(env.unwrapped, fem_reference_nodes, fem_debug_goals))
                            if args.debug_mode == 6 and (frame_count % 10 == 0 or success_step_count > 0):
                                try:
                                    payload.update(_debug_soft_goal_state(env.unwrapped, fem_debug_goals))
                                except Exception as exc:
                                    payload["goal_debug_error"] = str(exc)
                            mode6_force_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                            mode6_force_fh.flush()
                        except Exception:
                            # Never break evaluation due to debug capture
                            pass

                    # 仅在 debug_mode=3 时额外 dump 关节状态 / 图像序列
                    if args.debug_mode == 3:
                        # get joint states
                        cam = env.unwrapped.scene["agentview_cam"]
                        rgb = cam.data.output["rgb"][0]
                        # get joint states
                        robot = env.unwrapped.scene["robot"]
                        states = robot.data.joint_pos
                        states = states.cpu().numpy()

                        np.save(str(f"{args.debug_path}/state_{frame_count:04d}_{i:02d}.npy"), states)
                        # Convert to numpy if it's a tensor
                        if isinstance(rgb, torch.Tensor):
                            rgb = rgb.cpu().numpy()
                        # Ensure correct format for saving
                        if rgb.dtype == np.float32:
                            rgb = (rgb * 255).astype(np.uint8)
                        # Save RGB image
                        cv2.imwrite(
                            str(f"{args.debug_path}/frame_{frame_count:04d}_{i:02d}.png"),
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        )
                    frame_count += 1

                if experiment_success:
                    successful_experiments += 1
                    consecutive_failures = 0
                    current_sr = (successful_experiments / (exp_idx + 1)) * 100

                    # 累加当前成功 experiment 的平均挤压力
                    if exp_fsq_count > 0:
                        avg_pred = exp_fsq_pred_sum / exp_fsq_count
                        avg_meas = exp_fsq_meas_sum / exp_fsq_count
                        succ_squeeze_pred_sum += avg_pred
                        succ_squeeze_meas_sum += avg_meas
                        succ_squeeze_count += 1
                        print(
                            f"✓ Success | Current SR: {successful_experiments}/{exp_idx + 1} ({current_sr:.1f}%) "
                            f"| squeeze_pred={avg_pred:.4f}, squeeze_meas={avg_meas:.4f}"
                        )
                    else:
                        print(f"✓ Success | Current SR: {successful_experiments}/{exp_idx + 1} ({current_sr:.1f}%)")

                    # Binary mode: record measured squeeze/apply metrics (no predicted forces).
                    # We compute Top5% stats from per-step measured sequences to mirror hybrid reporting.
                    if args.control_mode == "binary":
                        # Strict: reuse metrics.py aggregation on the per-step LR force series.
                        try:
                            if exp_fL_meas_values and exp_fR_meas_values:
                                fL = np.stack(exp_fL_meas_values, axis=0)  # (T,3)
                                fR = np.stack(exp_fR_meas_values, axis=0)  # (T,3)
                                meas_metrics = compute_contact_force_metrics_from_lr_forces(fL, fR)

                                # Fill the "pred-style" metrics slots with measured metrics in binary mode
                                # (there is no force prediction in this control mode).
                                succ_metrics_count += 1
                                succ_squeeze_max_sum += float(meas_metrics.squeeze_max)
                                succ_app_max_sum += float(meas_metrics.external_norm_max)
                                succ_app_mean_sum += float(meas_metrics.external_norm_mean)

                                # Also populate measured aggregates (same definitions).
                                succ_squeeze_max_meas_sum += float(meas_metrics.squeeze_max)
                                succ_squeeze_max_meas_count += 1
                                succ_ap_mean_meas_sum += float(meas_metrics.external_norm_mean)
                                succ_ap_mean_meas_count += 1
                                succ_ap_max_meas_sum += float(meas_metrics.external_norm_max)
                                succ_ap_max_meas_count += 1
                        except Exception:
                            pass

                    # 若为 Hybrid 控制模式且缓存到了 13D 动作，则为该成功 experiment 计算一次力学 metrics
                    if args.control_mode in ("hybrid", "tactile") and exp_actions_13d:
                        try:
                            actions_13d = torch.stack(exp_actions_13d, dim=0).numpy()  # (T, 13)
                            metrics = compute_contact_force_metrics_from_13d(actions_13d)
                            succ_metrics_count += 1
                            # squeeze_max / external_norm_max 已在 metrics.py 中按 Top5% 帧均值定义
                            succ_squeeze_max_sum += metrics.squeeze_max
                            succ_app_max_sum += metrics.external_norm_max
                            succ_app_mean_sum += metrics.external_norm_mean

                            # 统计当前成功 experiment 的「实测」挤压力 / 加持力指标
                            # 1) 实测挤压力 Top5% 最大值（均值）
                            sq_max_meas_top5 = compute_topk_mean(exp_fsq_meas_values, frac=0.05)
                            if sq_max_meas_top5 is not None:
                                succ_squeeze_max_meas_sum += sq_max_meas_top5
                                succ_squeeze_max_meas_count += 1

                            # 2) 实测加持力平均值（直接在该 demo 内求均值）
                            if exp_ap_meas_values:
                                ap_mean_meas = float(np.mean(exp_ap_meas_values))
                                succ_ap_mean_meas_sum += ap_mean_meas
                                succ_ap_mean_meas_count += 1

                            # 3) 实测加持力 Top5% 最大值（均值）
                            ap_max_meas_top5 = compute_topk_mean(exp_ap_meas_values, frac=0.05)
                            if ap_max_meas_top5 is not None:
                                succ_ap_max_meas_sum += ap_max_meas_top5
                                succ_ap_max_meas_count += 1

                            print(
                                "    [Hybrid-Metrics] "
                                f"squeeze_max={metrics.squeeze_max:.4f}, "
                                f"squeeze_mean={metrics.squeeze_mean:.4f}, "
                                f"app_max={metrics.external_norm_max:.4f}, "
                                f"app_mean={metrics.external_norm_mean:.4f}"
                            )
                        except Exception:
                            # metrics 计算失败不影响主流程
                            pass

                    break

                # Check if we broke out of inner loop due to unexpected termination
                if i < args.replan_steps - 1 and (terminated[0] or truncated[0]):
                    current_sr = (successful_experiments / (exp_idx + 1)) * 100
                    print(f"✗ Failed (terminated) | Current SR: {successful_experiments}/{exp_idx + 1} ({current_sr:.1f}%)")
                    break

                if action_idx >= args.max_inference_steps - 1:
                    current_sr = (successful_experiments / (exp_idx + 1)) * 100
                    print(f"✗ Failed (max steps) | Current SR: {successful_experiments}/{exp_idx + 1} ({current_sr:.1f}%)")

            if not experiment_success:
                consecutive_failures += 1
            attempted_experiments = exp_idx + 1

            # debug_mode=5: 每个 experiment 结束后落盘一份逐帧挤压力序列
            if force_dump_dir is not None:
                try:
                    payload = {
                        "task_suite": args.task_suite,
                        "task_id": int(args.task_id),
                        "exp_idx": int(exp_idx),
                        "prompt_adverb": (args.prompt_adverb or "").strip(),
                        "prompt_adverbs": list(args.prompt_adverbs) if args.prompt_adverbs else [],
                        "adverb_used": exp_adv,
                        "prompt": exp_prompt,
                        "success": bool(experiment_success),
                        "terminated": bool(terminated[0]) if hasattr(terminated, "__len__") else bool(terminated),
                        "truncated": bool(truncated[0]) if hasattr(truncated, "__len__") else bool(truncated),
                        "num_frames": int(len(exp_fsq_pred_values)),
                        # 逐帧挤压力：与 env.step() 次数一一对应
                        "squeeze_pred": [float(x) for x in exp_fsq_pred_values],
                        "squeeze_meas": [float(x) for x in exp_fsq_meas_values],
                        # 逐帧加持力模长（ap_pred/ap_meas，与 ForcePositionAction / metrics.py 一致）：
                        # - hybrid/tactile: from ForcePositionAction.debug_info when available
                        # - binary: pred is always 0.0 (sentinel), meas derived from gripper_net_force
                        "ap_pred": [float(x) for x in exp_ap_pred_values],
                        "ap_meas": [float(x) for x in exp_ap_meas_values],
                    }
                    out_path = force_dump_dir / f"exp_{exp_idx:03d}.json"
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[DebugMode 5] Failed to dump gripper force for exp_{exp_idx:03d}: {e}")

            # debug_mode=6: close per-experiment file handle
            try:
                if mode6_force_fh is not None:
                    mode6_force_fh.close()
            except Exception:
                pass

            if max_consecutive_failures > 0 and consecutive_failures >= max_consecutive_failures:
                print(
                    f"[EarlyStop] Reached {consecutive_failures} consecutive failures; "
                    f"stopping task after {attempted_experiments}/{args.num_total_experiments} experiments."
                )
                break

    total_reported_experiments = attempted_experiments or args.num_total_experiments
    success_rate = (successful_experiments / total_reported_experiments) * 100
    print("\nEvaluation Results:")
    print(f"Total experiments: {total_reported_experiments}")
    print(f"Successful experiments: {successful_experiments}")
    print(f"Success rate: {success_rate:.2f}%")

    # 1) 挤压力平均值（预测 / 实测）——仅用于后续 metrics 行中输出
    task_avg_pred = None
    task_avg_meas = None
    if succ_squeeze_count > 0:
        task_avg_pred = succ_squeeze_pred_sum / succ_squeeze_count
        task_avg_meas = succ_squeeze_meas_sum / succ_squeeze_count

        # Keep backward-compatible line for scripts/tools/run_task_evaluations.py parser.
        if args.control_mode in ("hybrid", "tactile", "binary"):
            print(
                f"[Hybrid] Task avg squeeze_pred={task_avg_pred:.4f}, squeeze_meas={task_avg_meas:.4f} "
                f"over {succ_squeeze_count} successes"
            )

    # 2) Top5% 最大挤压力 / 最大加持力 + 平均加持力（预测 / 实测）——统一在 Hybrid-Metrics 一行输出
    if succ_metrics_count > 0:
        task_squeeze_max_mean = succ_squeeze_max_sum / succ_metrics_count
        task_app_max_mean = succ_app_max_sum / succ_metrics_count
        task_app_mean_mean = succ_app_mean_sum / succ_metrics_count

        # 实测挤压力 / 加持力（若有）
        task_squeeze_max_meas_mean = (
            succ_squeeze_max_meas_sum / succ_squeeze_max_meas_count
            if succ_squeeze_max_meas_count > 0
            else None
        )
        task_ap_mean_meas_mean = (
            succ_ap_mean_meas_sum / succ_ap_mean_meas_count if succ_ap_mean_meas_count > 0 else None
        )
        task_ap_max_meas_mean = (
            succ_ap_max_meas_sum / succ_ap_max_meas_count if succ_ap_max_meas_count > 0 else None
        )

        fragments: list[str] = []
        if task_avg_pred is not None:
            fragments.append(f"squeeze_avg_pred={task_avg_pred:.4f}")
        if task_avg_meas is not None:
            fragments.append(f"squeeze_avg_meas={task_avg_meas:.4f}")

        fragments.extend(
            [
                f"squeeze_max_mean={task_squeeze_max_mean:.4f}",
                f"app_max_mean={task_app_max_mean:.4f}",
                f"app_mean_mean={task_app_mean_mean:.4f}",
            ]
        )
        if task_squeeze_max_meas_mean is not None:
            fragments.append(f"squeeze_max_meas_mean={task_squeeze_max_meas_mean:.4f}")
        if task_ap_max_meas_mean is not None:
            fragments.append(f"ap_max_meas_mean={task_ap_max_meas_mean:.4f}")
        if task_ap_mean_meas_mean is not None:
            fragments.append(f"ap_mean_meas_mean={task_ap_mean_meas_mean:.4f}")

        print(
            "[Hybrid-Metrics] Task contact_metrics "
            + ", ".join(fragments)
            + f" over {succ_metrics_count} successes"
        )
    # 关闭 Hybrid 力–位可视化窗口
    if force_viz is not None:
        try:
            force_viz.close()
        except Exception:
            pass


if __name__ == "__main__":
    print("args", args)

    # Initialize the closed loop policy inference
    # Only support task space / hybrid control (diffik, osc, hybrid, tactile, binary)
    if args.control_mode in ["diffik", "osc", "hybrid", "tactile", "binary"]:
        inferencer = ClosedLoopPolicyInference(args)
    else:
        raise ValueError(
            f"Invalid control mode: {args.control_mode}. "
            f"Supported modes: ['diffik', 'osc', 'hybrid', 'tactile', 'binary']"
        )

    # Initialize client policy inference
    env, env_cfg, success_term = inferencer.create_sim_environment()

    # Ablation: tactile obs/model, but pure position actions (no force) and no corrections.
    if args.control_mode == "tactile" and args.abs7d:
        try:
            term = env.action_manager.get_term("arm_action")
            term.cfg.pos_kp = (0.0, 0.0, 0.0)
            term.cfg.squeeze_kp = 0.0
            print("[Ablation] abs7d enabled: pos_kp=(0,0,0), squeeze_kp=0, force dims zeroed.")
        except Exception as e:
            print(f"[Ablation] Failed to disable pos_kp/squeeze_kp on arm_action: {e}")

    # Run the closed loop policy
    run_closed_loop_policy(
        args=args, simulation_app=simulation_app, env=env, env_cfg=env_cfg, success_term=success_term
    )

    if os.environ.get("SOFTVTBENCH_SKIP_ISAAC_CLEANUP_ON_EXIT", "0") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    # Close environment and simulation app after replay is complete
    env.close()
    simulation_app.close()
