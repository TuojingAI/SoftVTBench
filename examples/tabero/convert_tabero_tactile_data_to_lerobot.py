#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Convert Isaac Lab HDF5 tactile datasets (Libero + Tabero) to LeRobot format
# for Tabero-style OpenPI / pi0 训练。
#
# 与 convert_all_libero_to_lerobot_openpi.py 的主要区别在于单帧输出格式：
#   - 仍使用 LeRobotDataset，但每帧额外包含：
#     * tactile_image: 由左右手指最近 H_tactile=8 个时间步的触觉图像拼接成 4×4 阵列
#       - 左侧 2×4 为左手指 8 帧
#       - 右侧 2×4 为右手指 8 帧
#       - 输出分辨率与 image/wrist_image 相同（默认 224×224），因此单个小图为 56×56
#     * tactile_gripper_force: 当前帧及过去 H_force=8 个时间步的左右指力历史，shape=(H_force, 6)
#       - 不足 8 帧时用最早一帧进行 padding
#     * tactile_marker_motion: 初始基准帧 + 最近 H_marker=8 个时间步的 marker motion
#       - 输出 shape=(1+H_marker, 2*M, 2)，M 为每个指上的 marker 数量（默认 99，可自动推断）
#   - actions:
#       * 仅支持 13D (x, y, z, ax, ay, az, gripper, fL(3), fR(3)) —— 与 7dpf 录制对齐
#

from __future__ import annotations

import sys
import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import os


def lerobot_add_frame(dataset, frame: dict, task: str) -> None:
    try:
        dataset.add_frame(frame, task=task)
    except TypeError:
        frame["task"] = task
        dataset.add_frame(frame)

import cv2
import h5py
import numpy as np
import shutil
import tyro
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    from lerobot.common.constants import HF_LEROBOT_HOME
except ImportError:
    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    except ImportError:
        hf_home = os.environ.get("HF_HOME")
        if hf_home is None:
            xdg_cache = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
            hf_home = str(Path(xdg_cache) / "huggingface")
        HF_LEROBOT_HOME = Path(hf_home) / "lerobot"
from tqdm import tqdm

TARGET_IMAGE_SIZE = (224, 224)


def _align_frames(frames: list[np.ndarray], target_len: int) -> list[np.ndarray]:
    """Return frames for an already length-checked stream."""
    if target_len <= 0:
        return []
    if len(frames) == 0:
        return []
    if len(frames) >= target_len:
        return frames[:target_len]
    return frames


@dataclass
class Config:
    """配置 Tabero 触觉数据到 LeRobot 数据集的转换参数."""

    # Task suites to convert
    task_suites: tuple[str, ...] = ("libero_10", "libero_spatial", "libero_goal", "libero_object")
    chunks_size: int = 1000
    fps: int = 20

    # 输入 / 输出路径（默认以你的 TacManip 根目录为基准）
    # NOTE: 这里使用绝对路径作为默认值，避免因脚本位置/工作目录变化导致路径被错误重写。
    data_root: Path = Path("/data/mingxinwang/datasets/Isaaclab_Libero")
    # 可选：同时输入两套数据（firm_force / gentle_force）并在转换时合并到同一个输出数据集中。
    # 这两套目录都应包含：
    # - replayed_demos/      (HDF5)
    # - video_datasets/      (videos/ + tactile_outputs/)
    strong_data_root: Path | None = None
    soft_data_root: Path | None = None
    # prompt 改写：仅修改语言提示（task 字段），不改 task_id / 文件名 / 其他字段
    strong_adverb: str = "firmly"
    soft_adverb: str = "gently"
    # 多副词版本（每条 trajectory 会确定性随机选一个；若为空则回退到 strong_adverb/soft_adverb 单值）
    strong_adverbs: tuple[str, ...] = ("firmly", "tightly") #, "forcefully"
    soft_adverbs: tuple[str, ...] = ("gently", "softly") #, "lightly", "delicately"
    prompt_seed: int = 0
    # replayed_demos HDF5 路径（若为空，则默认为 data_root / "replayed_demos"）
    hdf5_folder: Path = Path("")
    # 视频路径（若为空，则默认为 data_root / "video_datasets"）
    video_dir: Path = Path("")
    # LeRobot 输出目录
    # 固定输出到：.../benchmarks/datasets/tabero_pi0/<repo_name>
    output_dir: Path = HF_LEROBOT_HOME
    repo_name: str = "NathanWu7/tabero"
    # Optional task subset JSON. Defaults to examples/tabero/config/tabero_tasks.json.
    task_subset_path: Path = Path("")

    # NOTE:
    # Tabero tactile conversion requires FORCE in actions (13D).
    # If your HDF5 actions are 7D/8D, re-record with recorder_type=7dpf.

    # 力 / 力场 历史长度
    force_history_len: int = 8  # 每帧导出的 gripper_force: (H_force, 6)
    marker_history_len: int = 8  # 除初始帧外的 marker motion 历史帧数

    # 触觉传感器配置
    tactile_sensors: tuple[str, ...] = ("gsmini_left", "gsmini_right")
    # 与 replay_demos_with_camera.py 中生成的视频命名保持一致：
    #   demo_10_gsmini_left_tactile_rgb.mp4
    tactile_output_type: str = "tactile_rgb"

    # Marker 数量（若为 None，则从 HDF5 中自动推断）
    num_markers: int | None = None

    def __post_init__(self):
        # Normalize paths (do NOT prefix with repo_root).
        self.data_root = Path(self.data_root).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        if self.task_subset_path and str(self.task_subset_path) not in (".", ""):
            self.task_subset_path = Path(self.task_subset_path).expanduser().resolve()

        # Set default paths if not provided
        # 默认：按当前命名约定，合并 firm_force + gentle_force 两套数据源（如需单数据源，请显式传 --strong_data_root None --soft_data_root None）
        # 单数据源模式：从 data_root 推导 replayed_demos / video_datasets。
        if self.strong_data_root is None and self.soft_data_root is None:
            if not self.hdf5_folder or str(self.hdf5_folder) in (".", ""):
                self.hdf5_folder = self.data_root / "replayed_demos"
            if not self.video_dir or str(self.video_dir) in (".", ""):
                self.video_dir = self.data_root / "video_datasets"
        # 输出目录已固定为 repo 内的 benchmarks/datasets/tabero_pi0

        if not self.repo_name:
            self.repo_name = "NathanWu7/tabero"

        # 加载所有 task_suites 的任务配置（用于写入文本描述）
        self.task_configs = load_all_task_configs(self)

        # 状态维度始终为 7D: [x, y, z, ax, ay, az, gripper_abs]
        self.state_shape: int | None = 7

        # 动作维度：始终为 13D (pos+axis-angle+gripper+forces)
        self.action_shape: int | None = 13


def load_all_task_configs(config: Config) -> dict[int, str]:
    """Load task configurations from all task suite config files."""
    all_tasks: dict[int, str] = {}
    all_tasks[8888] = "valid"  # valid is 8888

    for suite_idx, task_suite in enumerate(config.task_suites):
        task_config_path = Path(__file__).parent / "config" / f"{task_suite}.json"
        print("task_config_path: ", task_config_path)

        if not task_config_path.exists():
            raise FileNotFoundError(f"Task config file not found: {task_config_path}")

        with open(task_config_path) as f:
            task_suite_config = json.load(f)

        # Map task IDs to global IDs
        for task in task_suite_config["tasks"]:
            original_task_id = task["task_id"]
            global_task_id = suite_idx * 10 + original_task_id  # 0-9, 10-19, ...
            language_instruction = task["language_instruction"]
            all_tasks[global_task_id] = language_instruction

    return all_tasks


def _load_tabero_task_subset_json(config: Config) -> dict[str, list[int]]:
    """Load Tabero task subset mapping from benchmarks/datasets/tabero/config/tabero_tasks.json.

    Returns:
        suite_name -> [task_ids]
    """
    path = (
        Path(config.task_subset_path)
        if config.task_subset_path and str(config.task_subset_path) not in (".", "")
        else Path(__file__).parent / "config" / "tabero_tasks.json"
    )
    with open(path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError('tabero_tasks.json must be a dict: suite_name -> [task_ids]')
    out: dict[str, list[int]] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, list):
            out[k] = [int(x) for x in v]
    return out


def get_all_task_hdf5_files(config: Config, subset_map: dict[str, list[int]]) -> dict[int, list[str]]:
    """Get HDF5 files organized by global task ID for all task suites."""
    hdf5_dir = Path(config.hdf5_folder)
    if not hdf5_dir.exists():
        raise FileNotFoundError(f"HDF5 directory not found: {hdf5_dir}")

    all_task_files: dict[int, list[str]] = {}

    for suite_idx, task_suite in enumerate(config.task_suites):
        allow = set(subset_map.get(task_suite, []))
        pattern = f"{task_suite}_task*_*demo.hdf5"

        for hdf5_file in hdf5_dir.glob(pattern):
            # Extract task ID from filename: libero_10_task0_... -> 0
            filename = hdf5_file.name
            task_id_str = filename.split("_task")[1].split("_")[0]
            original_task_id = int(task_id_str)
            if original_task_id not in allow:
                continue
            global_task_id = suite_idx * 10 + original_task_id

            if global_task_id not in all_task_files:
                all_task_files[global_task_id] = []
            all_task_files[global_task_id].append(str(hdf5_file))

    return all_task_files


def _choose_adverb(seed: int, key: str, adverbs: tuple[str, ...]) -> str:
    if not adverbs:
        return ""
    digest = hashlib.blake2b(f"{seed}:{key}".encode("utf-8"), digest_size=8).digest()
    idx = int.from_bytes(digest, "big") % len(adverbs)
    return (adverbs[idx] or "").strip()


def _rewrite_instruction(instruction: str, adverb: str, seed: int, key: str) -> str:
    """Rewrite instruction with an adverb in a more natural English style.

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

    style = _choose_adverb(seed, f"{key}:style", ("prefix", "suffix"))
    if style == "suffix":
        return f"{instruction} {adverb}"
    return f"{adverb} {instruction}"


def _iter_sources(config: Config) -> list[tuple[str, Path, Path, str]]:
    """Return [(label, hdf5_dir, video_root, adverb)]."""
    if config.strong_data_root is not None or config.soft_data_root is not None:
        if config.strong_data_root is None or config.soft_data_root is None:
            raise ValueError("Please set both strong_data_root and soft_data_root (or neither).")
        strong_root = Path(config.strong_data_root)
        soft_root = Path(config.soft_data_root)
        return [
            ("strong", strong_root / "replayed_demos", strong_root / "video_datasets", config.strong_adverb),
            ("soft", soft_root / "replayed_demos", soft_root / "video_datasets", config.soft_adverb),
        ]
    return [("default", Path(config.hdf5_folder), Path(config.video_dir), "")]


def get_all_task_hdf5_files_multi(config: Config, subset_map: dict[str, list[int]]) -> dict[int, list[tuple[str, str]]]:
    """Get HDF5 files organized by global task ID across sources.

    Returns:
        global_task_id -> list of (source_label, hdf5_file_path)
    """
    all_task_files: dict[int, list[tuple[str, str]]] = {}
    for source_label, hdf5_dir, _video_root, _adverb in _iter_sources(config):
        if not hdf5_dir.exists():
            raise FileNotFoundError(f"HDF5 directory not found: {hdf5_dir}")
        for suite_idx, task_suite in enumerate(config.task_suites):
            allow = set(subset_map.get(task_suite, []))
            pattern = f"{task_suite}_task*_*demo.hdf5"
            for hdf5_file in hdf5_dir.glob(pattern):
                filename = hdf5_file.name
                task_id_str = filename.split("_task")[1].split("_")[0]
                original_task_id = int(task_id_str)
                if original_task_id not in allow:
                    continue
                global_task_id = suite_idx * 10 + original_task_id
                all_task_files.setdefault(global_task_id, []).append((source_label, str(hdf5_file)))
    return all_task_files


def check_failed_videos(config: Config, suite_name: str, task_id: int, video_root: Path | None = None) -> list[str]:
    """Check video folder and find failed video IDs (same as OpenPI script)."""
    root = Path(video_root) if video_root is not None else Path(config.video_dir)
    video_dir = root / f"{suite_name}_task{task_id}" / "videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    # Get all mp4 files
    video_files = list(video_dir.glob("*.mp4"))

    # Find all failed video IDs
    failed_ids: list[str] = []
    for video_file in video_files:
        if "failed" in video_file.name:
            # Extract ID from filename, e.g., from "failed_123_ego_rgb.mp4" extract "123"
            traj_id = video_file.name.split("_")[1]
            if traj_id not in failed_ids:
                failed_ids.append(traj_id)

    print(f"\nFound {len(failed_ids)} failed trajectories:")
    for failed_id in sorted(failed_ids):
        print(failed_id, end=" ")

    return failed_ids


def load_videos_frames(video_path: Path) -> list[np.ndarray]:
    """Load all frames from an mp4 file and resize to TARGET_IMAGE_SIZE."""
    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB (cv2 reads in BGR format by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        re_frame = cv2.resize(frame_rgb, TARGET_IMAGE_SIZE)
        frames.append(re_frame)
    cap.release()
    return frames


def _quat2axisangle(quats: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz, Isaac convention) to axis-angle (robosuite 改写版)."""
    results: list[np.ndarray] = []
    for quat in quats:
        q0 = quat[0]
        # clip q0
        q0 = max(min(q0, 1.0), -1.0)

        den = np.sqrt(1.0 - q0 * q0)
        if math.isclose(den, 0.0):
            results.append(np.zeros(3))
        else:
            axis_angle = (quat[1:] * 2.0 * math.acos(q0)) / den
            results.append(axis_angle)
    return np.array(results)


def _build_sliding_window_with_pad(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    """构造时间维度的滑动窗口 [T, window, ...]，不足部分用最早帧 pad."""
    T = values.shape[0]
    out = np.zeros((T, window) + values.shape[1:], dtype=values.dtype)
    for t in range(T):
        # 对于当前 t，收集 t-window+1 .. t 的数据，越界部分 clamp 到 0
        for k in range(window):
            src_t = t - (window - 1 - k)
            if src_t < 0:
                src_t = 0
            out[t, k] = values[src_t]
    return out


def _infer_num_markers(config: Config, all_task_files: dict[int, list[str]]) -> int:
    """在所有 HDF5 里找到第一个包含 obs/gripper_marker_motion 的轨迹，推断 marker 个数."""
    for file_list in all_task_files.values():
        for sample_path_str in file_list:
            sample_path = Path(sample_path_str)
            with h5py.File(sample_path, "r") as f:
                data_group = f["data"]
                for traj_name in data_group.keys():
                    if not traj_name.startswith("demo_"):
                        continue
                    traj = data_group[traj_name]
                    if "obs" not in traj:
                        continue
                    traj_obs = traj["obs"]
                    if "gripper_marker_motion" in traj_obs:
                        gmm = np.array(traj_obs["gripper_marker_motion"])  # (T, 2, 2, M, 2)
                        _, _, _, m, _ = gmm.shape
                        return int(m)
    raise KeyError(
        "无法在提供的 HDF5 中找到 'obs/gripper_marker_motion' 字段，"
        "请确保使用 Tactile 环境并开启 gripper_marker_motion 观测后再进行转换。"
    )


def combine_traj_and_images_tabero(
    config: Config,
    trajectory_id: str,
    trajectory,
    suite_name: str,
    original_task_id: int,
    video_root: Path | None = None,
) -> tuple[bool, np.ndarray, np.ndarray, dict, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """将单条轨迹的数据（states/actions/视频）组合成 Tabero 风格输出."""
    is_valid = True
    images_dict: dict[str, list[np.ndarray]] = {}
    root = Path(video_root) if video_root is not None else Path(config.video_dir)

    # -- Base task-space state: EEF pose (xyz + quat) + 1 gripper scalar --
    #   - eef_pose    : (T, 7)  from env.obs['policy']['eef_pose']  = [x, y, z, qw, qx, qy, qz]
    #   - gripper_pos : (T, 1)  from env.obs['policy']['gripper_pos']，取第一个指头作为标量
    eef_pose = np.array(trajectory["obs"]["eef_pose"])  # (T, 7)
    pos = eef_pose[:, :3]  # (T, 3)
    quat = eef_pose[:, 3:7]  # (T, 4) (w,x,y,z)
    gripper_array = np.array(trajectory["obs"]["gripper_pos"])
    if gripper_array.ndim == 2:
        gripper_scalar = gripper_array[:, 0].reshape(-1, 1)  # (T,1)
    else:
        gripper_scalar = gripper_array.reshape(-1, 1)

    # Convert quaternion to axis-angle for all timesteps: (T,4) -> (T,3)
    eef_axisangle = _quat2axisangle(quat)  # (T, 3)

    # Final 7D task-space state: [x, y, z, ax, ay, az, gripper_abs]
    state_eef_7 = np.concatenate((pos, eef_axisangle, gripper_scalar), axis=1).astype(np.float32)  # (T, 7)

    # -- Actions: MUST be 13D (axis-angle+forces). No fallback to force-less actions. --
    action_array = np.array(trajectory["actions"])
    num_dims = action_array.shape[1]
    if num_dims != 13:
        print(
            f"[SKIP] Trajectory {trajectory_id}: actions 维度为 {num_dims}，Tabero 转换要求 13D (7dpf)。跳过该 trajectory。"
        )
        return (
            False,
            np.zeros((0, 13), dtype=np.float32),
            np.zeros((0, 7), dtype=np.float32),
            {},
            None,
            None,
            None,
        )

    actions = action_array.astype(np.float32)

    states = state_eef_7

    actions_len = actions.shape[0]
    if states.shape[0] != actions_len:
        raise ValueError(
            f"State/action length mismatch for {trajectory_id}: states={states.shape[0]}, actions={actions_len}."
        )

    # -- Load camera videos (agentview / wrist) --
    video_views = {
        "image": "agentview_rgb",
        "wrist_image": "eye_in_hand_rgb",
    }

    for video_key, view_suffix in video_views.items():
        try:
            original_video_path = (
                root / f"{suite_name}_task{original_task_id}" / "videos" / f"{trajectory_id}_{view_suffix}.mp4"
            )
            frames = load_videos_frames(original_video_path)
            if len(frames) != actions_len:
                raise ValueError(
                    f"Frame/action length mismatch for {trajectory_id}_{view_suffix}: "
                    f"video_frames={len(frames)}, actions={actions_len}. Refusing to silently realign."
                )
            frames = _align_frames(frames, actions_len)
            if len(frames) == 0:
                images_dict[video_key] = []
                is_valid = False
            else:
                images_dict[video_key] = frames
        except Exception as e:
            print(f"Error processing video {video_key} for trajectory {trajectory_id}: {e}")
            images_dict[video_key] = []
            is_valid = False

    # -- Load tactile videos for left/right sensors --
    tactile_images: np.ndarray | None = None
    try:
        tactile_dir = root / f"{suite_name}_task{original_task_id}" / "tactile_outputs"
        # 期望文件名：demo_i_{sensor}_{tactile_output_type}.mp4
        tactile_frames_per_sensor: dict[str, list[np.ndarray]] = {}
        for sensor in config.tactile_sensors:
            video_path = tactile_dir / f"{trajectory_id}_{sensor}_{config.tactile_output_type}.mp4"
            frames = load_videos_frames(video_path)
            if len(frames) != actions_len:
                raise ValueError(
                    f"Frame/action length mismatch for {trajectory_id}_{sensor}_{config.tactile_output_type}: "
                    f"video_frames={len(frames)}, actions={actions_len}. Refusing to silently realign."
                )
            frames = _align_frames(frames, actions_len)
            if len(frames) == 0:
                raise ValueError(
                    f"Tactile video for sensor {sensor} has 0 frames, cannot align to actions_len={actions_len}."
                )
            tactile_frames_per_sensor[sensor] = frames

        # Strict length checks above guarantee frame-index alignment.
        stream_lens = [
            len(images_dict.get("image", [])),
            len(images_dict.get("wrist_image", [])),
            len(tactile_frames_per_sensor[config.tactile_sensors[0]]),
            len(tactile_frames_per_sensor[config.tactile_sensors[1]]),
        ]
        if any(length != actions_len for length in stream_lens):
            raise ValueError(
                f"Frame/action length mismatch after loading {trajectory_id}: "
                f"stream_lens={stream_lens}, actions={actions_len}."
            )

        # 构造每帧的 4×4 拼接触觉图像
        H_out, W_out = TARGET_IMAGE_SIZE
        cell_h, cell_w = H_out // 4, W_out // 4
        T = actions_len
        tactile_images = np.zeros((T, H_out, W_out, 3), dtype=np.uint8)

        # 对每个时间步 t，构造最近 H_tactile=force_history_len 帧的 mosaic（假定与力历史对齐）
        H_tactile = config.force_history_len
        left_sensor = config.tactile_sensors[0]
        right_sensor = config.tactile_sensors[1] if len(config.tactile_sensors) > 1 else config.tactile_sensors[0]

        left_frames = tactile_frames_per_sensor[left_sensor]
        right_frames = tactile_frames_per_sensor[right_sensor]

        for t in range(T):
            # 收集 t-H_tactile+1 .. t 的索引，并在开头处用 t=0 padding（不做“尾部重复帧”兜底）
            indices = [t - (H_tactile - 1 - k) for k in range(H_tactile)]
            indices = [max(0, idx) for idx in indices]

            canvas = np.zeros((H_out, W_out, 3), dtype=np.uint8)

            # 新布局：
            #   - 左指：4×2 = 8 小图，按时间从上到下、从左到右，放在画面的「左半部分」两列
            #   - 右指：4×2 = 8 小图，同样按时间顺序，放在画面的「右半部分」两列
            #
            # 每一帧的索引 k ∈ [0,7]：
            #   行 r = k // 2 ∈ {0,1,2,3}
            #   列 c = k % 2  ∈ {0,1}

            # 左指：占据列 0、1
            for k, idx in enumerate(indices):
                r = k // 2  # 0..3
                c = k % 2  # 0 or 1
                y0, y1 = r * cell_h, (r + 1) * cell_h
                x0, x1 = c * cell_w, (c + 1) * cell_w

                img = cv2.resize(left_frames[idx], (cell_w, cell_h))
                canvas[y0:y1, x0:x1] = img

            # 右指：占据列 2、3（在空间上与左指左右分开）
            for k, idx in enumerate(indices):
                r = k // 2  # 0..3
                c = k % 2  # 0 or 1
                y0, y1 = r * cell_h, (r + 1) * cell_h
                x0, x1 = (c + 2) * cell_w, (c + 3) * cell_w

                img = cv2.resize(right_frames[idx], (cell_w, cell_h))
                canvas[y0:y1, x0:x1] = img

            tactile_images[t] = canvas

    except Exception as e:
        print(f"[SKIP] Failed to construct tactile mosaics for trajectory {trajectory_id}: {e}")
        return is_valid and False, np.zeros((0, 13), dtype=np.float32), np.zeros((0, 7), dtype=np.float32), {}, None, None, None

    # -- 构造 gripper_force: shape (T, H_force, 6) --
    tactile_gripper_force: np.ndarray | None = None
    try:
        gnf = np.array(trajectory["obs"]["gripper_net_force"])  # (T_full, H_sensor, 2, 3)
        if gnf.shape[0] != actions_len:
            raise ValueError(f"gripper_net_force length={gnf.shape[0]} but actions={actions_len}")
        T, H_sensor, _, _ = gnf.shape
        # 取当前帧（index=0）作为每个时间步的“即时力” (T,2,3) -> (T,6)
        inst_force = gnf[:, 0, :, :].reshape(T, 2 * 3).astype(np.float32)  # (T,6)
        tactile_gripper_force = _build_sliding_window_with_pad(inst_force, config.force_history_len)
    except Exception as e:
        print(f"[SKIP] Trajectory {trajectory_id}: cannot build tactile_gripper_force ({e})")
        return is_valid and False, np.zeros((0, 13), dtype=np.float32), np.zeros((0, 7), dtype=np.float32), {}, None, None, None

    # -- 构造 tactile_marker_motion: shape (T, 1+H_marker, 2*M, 2) --
    tactile_marker_motion: np.ndarray | None = None
    try:
        gmm = np.array(trajectory["obs"]["gripper_marker_motion"])  # (T_full, 2, 2, M, 2)
        if gmm.shape[0] != actions_len:
            raise ValueError(f"gripper_marker_motion length={gmm.shape[0]} but actions={actions_len}")
        T, S, T_mm, M, D = gmm.shape  # S=2(sensor), T_mm=2(init,current), D=2(x,y)
        assert S == 2 and T_mm == 2 and D == 2

        H_marker = config.marker_history_len
        # 拆出 init & current
        init_pos = gmm[:, :, 0, :, :]  # (T, 2, M, 2)
        curr_pos = gmm[:, :, 1, :, :]  # (T, 2, M, 2)

        # init 使用 t=0 的 init 作为全程统一基准
        init_concat = init_pos[0].reshape(2 * M, 2).astype(np.float32)  # (2*M,2)

        # 对 current 构造滑动窗口 (T, H_marker, 2*M, 2)
        curr_concat = curr_pos.reshape(T, 2 * M, 2).astype(np.float32)
        curr_hist = _build_sliding_window_with_pad(curr_concat, H_marker)  # (T,H_marker,2*M,2)

        # 拼成 (T, 1+H_marker, 2*M, 2)，第 0 维是 init，后面是历史 current
        tactile_marker_motion = np.zeros((T, 1 + H_marker, 2 * M, 2), dtype=np.float32)
        tactile_marker_motion[:, 0, :, :] = init_concat[None, :, :]
        tactile_marker_motion[:, 1:, :, :] = curr_hist
    except Exception as e:
        print(f"[SKIP] Trajectory {trajectory_id}: cannot build tactile_marker_motion ({e})")
        return is_valid and False, np.zeros((0, 13), dtype=np.float32), np.zeros((0, 7), dtype=np.float32), {}, None, None, None

    return is_valid, actions, states, images_dict, tactile_images, tactile_gripper_force, tactile_marker_motion


def print_dataset_info(config: Config, all_task_files: dict[int, list[str]]) -> None:
    """Print detailed dataset information."""
    print("\n" + "=" * 60)
    print("Tabero Tactile Dataset Conversion Info")
    print("=" * 60)
    print(f"Input Data Root: {config.data_root}")
    print(f"HDF5 Directory: {config.hdf5_folder}")
    print(f"Video Directory: {config.video_dir}")
    print(f"Output Directory: {config.output_dir / config.repo_name}")
    print(f"Task Suites: {', '.join(config.task_suites)}")
    print(f"FPS: {config.fps}")
    print(f"Chunk Size: {config.chunks_size}")

    # Task statistics
    total_files = sum(len(files) for files in all_task_files.values())
    print(f"\nTotal Tasks: {len(all_task_files)}")
    print(f"Total HDF5 Files: {total_files}")

    # Per-suite breakdown
    print("\nTask Suite Details:")
    for suite_idx, task_suite in enumerate(config.task_suites):
        suite_tasks = [
            (task_id, len(files))
            for task_id, files in all_task_files.items()
            if suite_idx * 10 <= task_id < (suite_idx + 1) * 10
        ]
        total_suite_files = sum(count for _, count in suite_tasks)
        print(f"  {task_suite}: {len(suite_tasks)} tasks, {total_suite_files} files")
        for task_id, file_count in suite_tasks:
            original_task_id = task_id % 10
            task_name = config.task_configs.get(task_id, "Unknown")
            print(f"    Task {task_id} (original ID {original_task_id}): {file_count} files - {task_name}")
    print("=" * 60 + "\n")


def main(config: Config):
    # Print configuration
    print("\n" + "=" * 50)
    print("LIBERO Tactile TO LEROBOT Tabero Conversion Configuration")
    print("=" * 50)
    for key, value in vars(config).items():
        if key != "task_configs":
            print(f"{key}: {value}")
    print("=" * 50)

    output_path = config.output_dir / config.repo_name
    print(f"\nLeRobot Output Path: {output_path}")
    if output_path.exists():
        print("Cleaning existing output directory...")
        shutil.rmtree(output_path)

    sources = _iter_sources(config)
    if len(sources) > 1:
        print("\n[Multi-source] Merging two datasets (strong/soft) into one output:")
        for label, hdf5_dir, video_root, adv in sources:
            print(f"  - {label}: hdf5_dir={hdf5_dir}, video_root={video_root}, adverb='{adv}'")

    # 唯一子集筛选逻辑：严格按 tabero_tasks.json
    subset_map = _load_tabero_task_subset_json(config)

    # Get all task-specific HDF5 files
    if len(sources) == 1 and sources[0][0] == "default":
        if not Path(config.hdf5_folder).exists():
            raise FileNotFoundError(f"HDF5 directory not found: {config.hdf5_folder}")
        all_task_files: dict[int, list] = get_all_task_hdf5_files(config, subset_map)
        multi_mode = False
    else:
        all_task_files = get_all_task_hdf5_files_multi(config, subset_map)
        multi_mode = True

    if not all_task_files:
        raise RuntimeError('未找到任何待转换任务：请检查 tabero_tasks.json 与 replayed_demos/*.hdf5 是否匹配。')

    # Print detailed dataset information
    # Print detailed dataset information (best-effort for multi-mode)
    if not multi_mode:
        print_dataset_info(config, all_task_files)  # type: ignore[arg-type]

    # 若用户未指定 num_markers，则从 HDF5 中自动推断
    if config.num_markers is None:
        # _infer_num_markers expects mapping to list[str]; in multi-mode use file paths only.
        if not multi_mode:
            config.num_markers = _infer_num_markers(config, all_task_files)  # type: ignore[arg-type]
        else:
            tmp: dict[int, list[str]] = {
                k: [p for (_lbl, p) in v] for k, v in all_task_files.items()  # type: ignore[union-attr]
            }
            config.num_markers = _infer_num_markers(config, tmp)
        print(f"Inferred num_markers from HDF5: {config.num_markers}")

    IMAGE_SETTING = {
        "dtype": "image",
        "shape": (TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 3),
        "names": ["height", "width", "channel"],
    }

    state_shape = config.state_shape
    action_shape = config.action_shape

    # 定义 LeRobot features
    features: Dict[str, Dict] = {
        "image": IMAGE_SETTING,
        "wrist_image": IMAGE_SETTING,
        "tactile_image": IMAGE_SETTING,
        "state": {
            "dtype": "float32",
            "shape": (state_shape,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (action_shape,),
            "names": ["actions"],
        },
    }

    # 导出 H_force×6 指力历史
    if config.force_history_len is not None and config.force_history_len > 0:
        features["tactile_gripper_force"] = {
            "dtype": "float32",
            "shape": (config.force_history_len, 6),
            "names": ["tactile_gripper_force"],
        }

    # 导出 (1+H_marker)×(2*M)×2 的 marker motion 历史
    if config.marker_history_len is not None and config.marker_history_len >= 0 and config.num_markers is not None:
        features["tactile_marker_motion"] = {
            "dtype": "float32",
            "shape": (1 + config.marker_history_len, 2 * config.num_markers, 2),
            "names": ["tactile_marker_motion"],
        }

    dataset = LeRobotDataset.create(
        repo_id=config.repo_name,
        root=output_path,
        robot_type="franka",
        fps=config.fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Process each global task (already filtered by tabero_tasks.json)
    for global_task_id in sorted(all_task_files.keys()):
        suite_idx = global_task_id // 10
        original_task_id = global_task_id % 10
        suite_name = config.task_suites[suite_idx]

        task_hdf5_files = all_task_files[global_task_id]
        print(
            f"\nProcessing Global Task {global_task_id} "
            f"({suite_name}_task{original_task_id}): "
            f"{config.task_configs[global_task_id]}"
        )

        # Pre-build source lookup for multi-mode
        source_video_root = {lbl: vr for (lbl, _hd, vr, _adv) in sources}
        source_adverb = {lbl: adv for (lbl, _hd, _vr, adv) in sources}

        for item in tqdm(task_hdf5_files, desc=f"Global Task {global_task_id}"):
            if multi_mode:
                source_label, hdf5_file_path = item
                video_root = source_video_root[source_label]
                adverb = source_adverb[source_label]
            else:
                source_label = "default"
                hdf5_file_path = item
                video_root = Path(config.video_dir)
                adverb = ""

            # NOTE: This processing must run for BOTH single-source and multi-source modes.
            with h5py.File(hdf5_file_path, "r") as hdf5_handler:
                hdf5_data = hdf5_handler["data"]
                trajectory_ids = sorted(
                    [k for k in hdf5_data.keys() if k.startswith("demo_")],
                    key=lambda x: int(x.split("_")[1]),
                )

                # Check for failed videos
                failed_ids = check_failed_videos(config, suite_name, original_task_id, video_root=Path(video_root))

                print(f"\nProcessing {len(trajectory_ids)} trajectories...")
                for trajectory_id in trajectory_ids:
                    # Skip failed trajectories
                    if trajectory_id in [f"demo_{failed_id}" for failed_id in failed_ids]:
                        print(f"Skipping failed trajectory: {trajectory_id}")
                        continue

                    trajectory = hdf5_data[trajectory_id]
                    (
                        is_valid,
                        actions,
                        states,
                        images_dict,
                        tactile_images,
                        tactile_gripper_force,
                        tactile_marker_motion,
                    ) = combine_traj_and_images_tabero(
                        config,
                        trajectory_id,
                        trajectory,
                        suite_name,
                        original_task_id,
                        video_root=Path(video_root),
                    )
                    if is_valid:
                        base_task = config.task_configs[global_task_id]
                        if source_label == "strong":
                            advs = config.strong_adverbs or (config.strong_adverb,)
                        elif source_label == "soft":
                            advs = config.soft_adverbs or (config.soft_adverb,)
                        else:
                            advs = (adverb,) if adverb else ()
                        chosen = _choose_adverb(
                            config.prompt_seed,
                            f"{source_label}:{suite_name}:{original_task_id}:{Path(hdf5_file_path).name}:{trajectory_id}",
                            advs,
                        )
                        task_description = _rewrite_instruction(
                            base_task,
                            chosen,
                            seed=config.prompt_seed,
                            key=f"{source_label}:{suite_name}:{original_task_id}:{Path(hdf5_file_path).name}:{trajectory_id}",
                        )
                        T = actions.shape[0]
                        for i in range(T):
                            frame = {
                                "image": images_dict["image"][i],
                                "wrist_image": images_dict["wrist_image"][i],
                                "tactile_image": tactile_images[i],
                                "state": np.array(states[i], dtype=np.float32),
                                "actions": np.array(actions[i], dtype=np.float32),
                                "tactile_gripper_force": np.array(tactile_gripper_force[i], dtype=np.float32),
                                "tactile_marker_motion": np.array(tactile_marker_motion[i], dtype=np.float32),
                            }
                            lerobot_add_frame(dataset, frame, task_description)
                        dataset.save_episode()



    print("\nDataset saved successfully")

    # Print final statistics
    print("\n" + "=" * 60)
    print("Conversion Completion Statistics")
    print("=" * 60)
    print(f"Dataset Output Path: {output_path}")
    print(f"Total Episodes: {dataset.num_episodes}")
    print(f"Total Frames: {len(dataset)}")
    print(f"Number of Tasks: {len(config.task_configs)}")
    print("=" * 60)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
