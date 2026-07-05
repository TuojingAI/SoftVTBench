#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Convert Isaac Lab replay datasets (7D/8D no-force actions) to LeRobot format,
# while merging two sources (firm_force + gentle_force) into a single dataset.
#
# This is designed for "softvtbench_binary" workflows:
# - Recorded with recorder_type=7d2 (axis-angle + position + binary gripper) => 7D actions
# - Output in a pi0/OpenPI-style LeRobot dataset: image, wrist_image, state(7D), actions(7D)
# - Merge firm/gentle folders and rewrite prompt with adverbs (firmly/gently)
#

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import cv2
import h5py
import numpy as np
import tyro
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# Ensure project root is in sys.path so `import benchmarks.*` works when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.common.lerobot_compat import lerobot_add_frame

TARGET_IMAGE_SIZE = (224, 224)


def _align_frames(frames: list[np.ndarray], target_len: int) -> list[np.ndarray]:
    """Align frames to target_len by truncation; do NOT pad tail."""
    if target_len <= 0 or len(frames) == 0:
        return []
    if len(frames) >= target_len:
        return frames[:target_len]
    return frames


def _random_keep_indices(total_len: int, target_len: int, seed: int) -> np.ndarray:
    """Randomly keep target_len indices from [0,total_len), sorted."""
    if target_len >= total_len:
        return np.arange(total_len, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(total_len, size=target_len, replace=False)
    return np.sort(idx.astype(np.int64))


def _quat2axisangle(quats: np.ndarray) -> np.ndarray:
    """Convert WXYZ quaternion array (T,4) to axis-angle (T,3)."""
    results: list[np.ndarray] = []
    for quat in quats:
        q0 = quat[0]
        q0 = max(min(float(q0), 1.0), -1.0)
        den = math.sqrt(max(1.0 - q0 * q0, 0.0))
        if math.isclose(den, 0.0):
            results.append(np.zeros(3, dtype=np.float32))
        else:
            axis_angle = (quat[1:].astype(np.float32) * (2.0 * math.acos(q0))) / float(den)
            results.append(axis_angle.astype(np.float32))
    return np.stack(results, axis=0).astype(np.float32)


def load_videos_frames(video_path: Path) -> list[np.ndarray]:
    """Load all frames from an mp4 file and resize to TARGET_IMAGE_SIZE (RGB)."""
    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame_rgb, TARGET_IMAGE_SIZE))
    cap.release()
    return frames


def _choose_adverb(seed: int, key: str, adverbs: tuple[str, ...]) -> str:
    if not adverbs:
        return ""
    digest = hashlib.blake2b(f"{seed}:{key}".encode("utf-8"), digest_size=8).digest()
    idx = int.from_bytes(digest, "big") % len(adverbs)
    return (adverbs[idx] or "").strip()


def _rewrite_instruction(instruction: str, adverb: str, seed: int, key: str) -> str:
    """Rewrite instruction with an adverb in a natural English style (deterministic per key)."""
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


def _load_softvtbench_task_subset_json() -> dict[str, list[int]]:
    """Load subset mapping from benchmarks/datasets/softvtbench/config/softvtbench_tasks.json."""
    path = _PROJECT_ROOT / "benchmarks" / "datasets" / "softvtbench" / "config" / "softvtbench_tasks.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("softvtbench_tasks.json must be a dict: suite_name -> [task_ids]")
    out: dict[str, list[int]] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, list):
            out[k] = [int(x) for x in v]
    return out


def _iter_files(root: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(root.glob(pattern))


@dataclass
class Config:
    """Convert & merge softvtbench_binary replay datasets to LeRobot (pi0/OpenPI style)."""

    # Suites to convert (usually only libero_object when using softvtbench_tasks.json)
    task_suites: tuple[str, ...] = ("libero_object",)
    chunks_size: int = 1000
    fps: int = 20

    # Input root containing firm_force/ and gentle_force/ (each has replayed_demos/ and video_datasets/)
    data_root: Path = Path("benchmarks/datasets/softvtbench_binary")
    strong_data_root: Path | None = None
    soft_data_root: Path | None = None

    # Prompt rewrite (strong=firm, soft=gentle)
    strong_adverb: str = "firmly"
    soft_adverb: str = "gently"
    strong_adverbs: tuple[str, ...] = ("firmly", "tightly")
    soft_adverbs: tuple[str, ...] = ("gently", "softly")
    prompt_seed: int = 0

    # LeRobot output
    output_dir: Path = Path("benchmarks/datasets/softvtbench_pi0_binary")
    repo_name: str = "softvtbench_binary"

    # NOTE: 7D state/action (no forces)
    state_shape: int = 7
    action_shape: int = 7

    def __post_init__(self):
        self.data_root = Path(self.data_root).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        if self.strong_data_root is None and self.soft_data_root is None:
            self.strong_data_root = self.data_root / "firm_force"
            self.soft_data_root = self.data_root / "gentle_force"
        if self.strong_data_root is None or self.soft_data_root is None:
            raise ValueError("Please set both strong_data_root and soft_data_root (or neither).")
        self.strong_data_root = Path(self.strong_data_root).expanduser().resolve()
        self.soft_data_root = Path(self.soft_data_root).expanduser().resolve()
        self.task_configs = load_all_task_configs(self.task_suites)


def load_all_task_configs(task_suites: tuple[str, ...]) -> dict[int, str]:
    """Load libero language instructions (global_task_id -> instruction)."""
    all_tasks: dict[int, str] = {8888: "valid"}
    config_dir = _PROJECT_ROOT / "benchmarks" / "datasets" / "libero" / "config"
    for suite_idx, task_suite in enumerate(task_suites):
        path = config_dir / f"{task_suite}.json"
        with open(path, "r", encoding="utf-8") as f:
            suite_cfg = json.load(f)
        for task in suite_cfg.get("tasks", []):
            original_task_id = int(task.get("task_id"))
            global_task_id = suite_idx * 10 + original_task_id
            all_tasks[global_task_id] = str(task.get("language_instruction", "")).strip()
    return all_tasks


def _collect_hdf5_files(
    *,
    hdf5_dir: Path,
    task_suites: tuple[str, ...],
    subset_map: dict[str, list[int]],
) -> dict[int, list[str]]:
    """Group HDF5 files by global_task_id (suite_idx*10 + task_id) with subset filter."""
    out: dict[int, list[str]] = {}
    if not hdf5_dir.exists():
        raise FileNotFoundError(f"HDF5 directory not found: {hdf5_dir}")
    for suite_idx, task_suite in enumerate(task_suites):
        allow = set(subset_map.get(task_suite, []))
        pattern = f"{task_suite}_task*_*demo.hdf5"
        for f in _iter_files(hdf5_dir, pattern):
            name = f.name
            task_id_str = name.split("_task")[1].split("_")[0]
            original_task_id = int(task_id_str)
            if allow and original_task_id not in allow:
                continue
            global_task_id = suite_idx * 10 + original_task_id
            out.setdefault(global_task_id, []).append(str(f))
    return out


def _collect_hdf5_files_multi(
    *,
    strong_root: Path,
    soft_root: Path,
    task_suites: tuple[str, ...],
    subset_map: dict[str, list[int]],
) -> dict[int, list[tuple[str, str]]]:
    """global_task_id -> list[(source_label, hdf5_path)]"""
    out: dict[int, list[tuple[str, str]]] = {}
    for label, root in (("strong", strong_root), ("soft", soft_root)):
        hdf5_dir = root / "replayed_demos"
        grouped = _collect_hdf5_files(hdf5_dir=hdf5_dir, task_suites=task_suites, subset_map=subset_map)
        for k, files in grouped.items():
            out.setdefault(k, []).extend([(label, p) for p in files])
    return out


def combine_traj_and_images_binary(
    *,
    config: Config,
    trajectory_id: str,
    trajectory: h5py.Group,
    suite_name: str,
    original_task_id: int,
    video_root: Path,
) -> tuple[bool, np.ndarray, np.ndarray, dict]:
    """Combine HDF5 trajectory with RGB videos; return (is_valid, actions, states, images_dict)."""
    is_valid = True
    images_dict: dict[str, list[np.ndarray]] = {}

    # --- states (7D): pos(3) + axis-angle(3) + gripper_abs(1) ---
    eef_pose = np.array(trajectory["obs"]["eef_pose"])  # (T,7) = pos(3) + quat(4)
    pos = eef_pose[:, :3]
    quat = eef_pose[:, 3:7]
    gripper_array = np.array(trajectory["obs"]["gripper_pos"])
    if gripper_array.ndim == 2:
        gripper_scalar = gripper_array[:, 0].reshape(-1, 1)
    else:
        gripper_scalar = gripper_array.reshape(-1, 1)
    eef_axisangle = _quat2axisangle(quat)
    states = np.concatenate((pos, eef_axisangle, gripper_scalar), axis=1).astype(np.float32)  # (T,7)

    # --- actions (7D or legacy 8D -> 7D) ---
    action_array = np.array(trajectory["actions"])
    if action_array.ndim != 2 or action_array.shape[1] not in (7, 8):
        return False, np.zeros((0, 7), dtype=np.float32), np.zeros((0, 7), dtype=np.float32), {}
    num_dims = int(action_array.shape[1])
    if num_dims == 8:
        # 8D: pos(3) + quat(4) + gripper(1)
        axis = _quat2axisangle(action_array[:, 3:7])
        actions = np.concatenate((action_array[:, :3], axis, action_array[:, 7:]), axis=1).astype(np.float32)
    else:
        actions = action_array.astype(np.float32)

    actions_len = actions.shape[0]

    # --- videos (agentview + wrist) ---
    video_views = {"image": "agentview_rgb", "wrist_image": "eye_in_hand_rgb"}
    for k, suffix in video_views.items():
        try:
            p = video_root / f"{suite_name}_task{original_task_id}" / "videos" / f"{trajectory_id}_{suffix}.mp4"
            frames = load_videos_frames(p)
            frames = _align_frames(frames, actions_len)
            if len(frames) == 0:
                images_dict[k] = []
                is_valid = False
            else:
                images_dict[k] = frames
        except Exception:
            images_dict[k] = []
            is_valid = False

    # Align by random dropping if any stream is shorter than actions
    if is_valid:
        lens = [len(images_dict.get("image", [])), len(images_dict.get("wrist_image", []))]
        target_len = min([x for x in lens if x > 0], default=0)
        if target_len > 0 and target_len < actions_len:
            seed = int.from_bytes(
                hashlib.blake2b(f"{config.prompt_seed}:{trajectory_id}".encode("utf-8"), digest_size=8).digest(),
                "big",
            )
            keep_idx = _random_keep_indices(actions_len, target_len, seed=seed)
            actions = actions[keep_idx]
            states = states[keep_idx]
            actions_len = target_len
            images_dict["image"] = images_dict["image"][:actions_len]
            images_dict["wrist_image"] = images_dict["wrist_image"][:actions_len]

    return is_valid, actions, states, images_dict


def main(config: Config) -> None:
    print("\n" + "=" * 60)
    print("SOFTVTBENCH_BINARY (7D) TO LEROBOT Conversion Configuration")
    print("=" * 60)
    for k, v in vars(config).items():
        if k != "task_configs":
            print(f"{k}: {v}")
    print("=" * 60)

    out_path = config.output_dir / config.repo_name
    print(f"\nLeRobot Output Path: {out_path}")
    if out_path.exists():
        print("Cleaning existing output directory...")
        shutil.rmtree(out_path)

    subset_map = _load_softvtbench_task_subset_json()
    all_task_files = _collect_hdf5_files_multi(
        strong_root=config.strong_data_root,
        soft_root=config.soft_data_root,
        task_suites=config.task_suites,
        subset_map=subset_map,
    )
    if not all_task_files:
        raise RuntimeError("未找到任何待转换任务：请检查 softvtbench_tasks.json 与 replayed_demos/*.hdf5 是否匹配。")

    IMAGE_SETTING = {
        "dtype": "image",
        "shape": (TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 3),
        "names": ["height", "width", "channel"],
    }
    features: Dict[str, Dict] = {
        "image": IMAGE_SETTING,
        "wrist_image": IMAGE_SETTING,
        "state": {"dtype": "float32", "shape": (config.state_shape,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (config.action_shape,), "names": ["actions"]},
    }

    dataset = LeRobotDataset.create(
        repo_id=config.repo_name,
        root=out_path,
        robot_type="franka",
        fps=config.fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Source lookup
    source_video_root = {
        "strong": Path(config.strong_data_root) / "video_datasets",
        "soft": Path(config.soft_data_root) / "video_datasets",
    }

    for global_task_id in sorted(all_task_files.keys()):
        suite_idx = global_task_id // 10
        original_task_id = global_task_id % 10
        suite_name = config.task_suites[suite_idx]

        items = all_task_files[global_task_id]
        print(f"\nProcessing {suite_name}_task{original_task_id}: {len(items)} files")
        for source_label, hdf5_path in tqdm(items, desc=f"{suite_name}_task{original_task_id}"):
            video_root = source_video_root[source_label]
            with h5py.File(hdf5_path, "r") as f:
                data = f["data"]
                traj_ids = sorted(
                    [k for k in data.keys() if k.startswith("demo_")],
                    key=lambda x: int(x.split("_")[1]),
                )
                for traj_id in traj_ids:
                    traj = data[traj_id]
                    ok, actions, states, images = combine_traj_and_images_binary(
                        config=config,
                        trajectory_id=traj_id,
                        trajectory=traj,
                        suite_name=suite_name,
                        original_task_id=original_task_id,
                        video_root=video_root,
                    )
                    if not ok:
                        continue

                    base_task = config.task_configs.get(global_task_id, "").strip()
                    if source_label == "strong":
                        advs = config.strong_adverbs or (config.strong_adverb,)
                    else:
                        advs = config.soft_adverbs or (config.soft_adverb,)
                    chosen = _choose_adverb(
                        config.prompt_seed,
                        f"{source_label}:{suite_name}:{original_task_id}:{Path(hdf5_path).name}:{traj_id}",
                        advs,
                    )
                    task_desc = _rewrite_instruction(
                        base_task,
                        chosen,
                        seed=config.prompt_seed,
                        key=f"{source_label}:{suite_name}:{original_task_id}:{Path(hdf5_path).name}:{traj_id}",
                    )

                    T = actions.shape[0]
                    for i in range(T):
                        frame = {
                            "image": images["image"][i],
                            "wrist_image": images["wrist_image"][i],
                            "state": np.array(states[i], dtype=np.float32),
                            "actions": np.array(actions[i], dtype=np.float32),
                        }
                        lerobot_add_frame(dataset, frame, task_desc)
                    dataset.save_episode()

    print("\nDataset saved successfully")
    print(f"Dataset Output Path: {out_path}")
    print(f"Total Episodes: {dataset.num_episodes}")
    print(f"Total Frames: {len(dataset)}")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
