#!/usr/bin/env python3
"""Convert ContactForce replay datasets (softvtbench_force) to LeRobot, building force-history offline."""
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Convert Isaac Lab HDF5 replay datasets (ContactForce replay env, i.e. softvtbench_force)
# to LeRobot format, while constructing a force-history window offline.
#
# Design choice (important):
# - Recording-time HDF5 should keep `obs/gripper_net_force` as current-frame only (H=1),
#   to keep the dataset clean/lightweight and avoid embedding an arbitrary history window.
# - This script constructs `observation/gripper_force` with shape (H_out, 6) per frame
#   by building a sliding window over instantaneous forces.
#
# References:
# - benchmarks/common/convert_all_libero_to_lerobot_openpi.py (video + states/actions conversion)
# - benchmarks/common/convert_all_libero_to_softvtbench.py (sliding-window + padding utilities)
#

from __future__ import annotations

import sys
import json
import hashlib
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

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

# Keep consistent with `convert_all_libero_to_softvtbench.py`:
# use per-frame images (NOT encoded video), to avoid codec logs and align dataset layout.
IMAGE_SETTING = {
    'dtype': 'image',
    'shape': (TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 3),
    'names': ['height', 'width', 'channel'],
}


def _align_frames(frames: list[np.ndarray], target_len: int) -> list[np.ndarray]:
    """Align video frames to target length.

    Policy (OpenPI-style + user-requested padding):
    - If len(frames) >= target_len: truncate to target_len
    - If 0 < len(frames) < target_len: keep as-is (caller may downsample state/action to match)
    - If len(frames) == 0: return empty list (caller should treat as invalid)
    """
    if target_len <= 0:
        return []
    if len(frames) == 0:
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
    results = []
    for quat in quats:
        q0 = quat[0]
        q0 = min(1.0, max(-1.0, q0))
        den = np.sqrt(1.0 - q0 * q0)
        if math.isclose(den, 0.0):
            results.append(np.zeros(3, dtype=np.float32))
        else:
            axis_angle = (quat[1:] * 2.0 * math.acos(q0)) / den
            results.append(axis_angle.astype(np.float32))
    return np.array(results, dtype=np.float32)


def _build_sliding_window_with_pad(x: np.ndarray, window: int) -> np.ndarray:
    """Build a causal sliding window with edge padding.

    Args:
        x: (T, D) float array
        window: H_out
    Returns:
        (T, H_out, D) where each timestep t contains [t-H+1 .. t] with padding by t=0.
    """
    if window <= 0:
        raise ValueError(f'window must be > 0, got {window}')
    T, D = x.shape
    out = np.zeros((T, window, D), dtype=x.dtype)
    for t in range(T):
        idxs = [t - (window - 1 - k) for k in range(window)]
        idxs = [0 if i < 0 else i for i in idxs]
        out[t] = x[idxs]
    return out


def load_all_task_configs(task_suites: tuple[str, ...]) -> dict[int, str]:
    """Load Libero task language instructions (global_task_id -> instruction)."""
    all_tasks: dict[int, str] = {}
    all_tasks[8888] = 'valid'
    config_dir = Path(__file__).parent.parent.resolve() / 'datasets' / 'libero' / 'config'
    for suite_idx, task_suite in enumerate(task_suites):
        task_config_path = config_dir / f'{task_suite}.json'
        with open(task_config_path) as f:
            task_suite_config = json.load(f)
        for task in task_suite_config['tasks']:
            original_task_id = task['task_id']
            global_task_id = suite_idx * 10 + original_task_id
            all_tasks[global_task_id] = task['language_instruction']
    return all_tasks


def get_all_task_hdf5_files(
    hdf5_dir: Path, task_suites: tuple[str, ...], subset_map: dict[str, list[int]] | None = None
) -> dict[int, list[str]]:
    """Group HDF5 files by global_task_id (suite_idx*10 + task_id)."""
    if not hdf5_dir.exists():
        raise FileNotFoundError(f'HDF5 directory not found: {hdf5_dir}')
    all_task_files: dict[int, list[str]] = {}
    for suite_idx, task_suite in enumerate(task_suites):
        allow = None
        if subset_map is not None and task_suite in subset_map:
            allow = set(subset_map.get(task_suite, []))
        pattern = f'{task_suite}_task*_*demo.hdf5'
        for hdf5_file in hdf5_dir.glob(pattern):
            filename = hdf5_file.name
            task_id_str = filename.split('_task')[1].split('_')[0]
            original_task_id = int(task_id_str)
            if allow is not None and original_task_id not in allow:
                continue
            global_task_id = suite_idx * 10 + original_task_id
            all_task_files.setdefault(global_task_id, []).append(str(hdf5_file))
    return all_task_files


def _load_softvtbench_task_subset_json() -> dict[str, list[int]]:
    """Load SoftVTBench task subset mapping from benchmarks/datasets/softvtbench/config/softvtbench_tasks.json."""
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / 'benchmarks' / 'datasets' / 'softvtbench' / 'config' / 'softvtbench_tasks.json'
    with open(path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError('softvtbench_tasks.json must be a dict: suite_name -> [task_ids]')
    out: dict[str, list[int]] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, list):
            out[k] = [int(x) for x in v]
    return out


def check_failed_videos(video_dir: Path) -> list[str]:
    """Return failed demo IDs from a `videos/` folder (filenames contain 'failed')."""
    if not video_dir.exists():
        return []
    failed_ids: list[str] = []
    for video_file in video_dir.glob('*.mp4'):
        if 'failed' in video_file.name:
            traj_id = video_file.name.split('_')[1]
            if traj_id not in failed_ids:
                failed_ids.append(traj_id)
    return failed_ids


def load_videos_frames(video_path: Path) -> list[np.ndarray]:
    """Load a video file and return list of resized RGB frames."""
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


def combine_traj_and_images_softvtbench_force(
    trajectory_id: str,
    trajectory: h5py.Group,
    suite_name: str,
    original_task_id: int,
    video_root: Path,
    force_history_len: int,
) -> tuple[bool, np.ndarray, np.ndarray, dict, np.ndarray | None]:
    """Combine HDF5 (states/actions/obs) with videos, and build force history offline."""
    is_valid = True

    # --- states (7D) ---
    eef_pose = np.array(trajectory['obs']['eef_pose'])  # (T, 7) = pos(3) + quat(4)
    pos = eef_pose[:, :3]
    quat = eef_pose[:, 3:7]
    gripper_array = np.array(trajectory['obs']['gripper_pos'])
    if gripper_array.ndim == 2:
        gripper_scalar = gripper_array[:, 0].reshape(-1, 1)
    else:
        gripper_scalar = gripper_array.reshape(-1, 1)
    eef_axisangle = _quat2axisangle(quat)
    states = np.concatenate((pos, eef_axisangle, gripper_scalar), axis=1).astype(np.float32)  # (T,7)

    # --- actions ---
    action_array = np.array(trajectory['actions'])
    num_dims = action_array.shape[1]
    actions: np.ndarray
    if num_dims != 13:
        print(
            f'[SKIP] Trajectory {trajectory_id}: actions dims={num_dims}, softvtbench_force conversion requires 13D (7dpf).'
        )
        return False, np.zeros((0, 13), dtype=np.float32), np.zeros((0, 7), dtype=np.float32), {}, None

    actions = action_array.astype(np.float32)

    # --- videos (agentview + wrist) ---
    T_full = actions.shape[0]
    videos_dir = video_root / f'{suite_name}_task{original_task_id}' / 'videos'
    # Naming follows scripts/tools/common/replay_utils.py:
    #   demo_{id}_{view}_{img_type}.mp4  -> for RGB: *_agentview_rgb.mp4, *_eye_in_hand_rgb.mp4
    view_suffix = 'agentview_rgb'
    wrist_suffix = 'eye_in_hand_rgb'
    video_path = videos_dir / f'{trajectory_id}_{view_suffix}.mp4'
    wrist_video_path = videos_dir / f'{trajectory_id}_{wrist_suffix}.mp4'

    image_frames = load_videos_frames(video_path)
    wrist_frames = load_videos_frames(wrist_video_path)
    image_frames = _align_frames(image_frames, T_full)
    wrist_frames = _align_frames(wrist_frames, T_full)
    if len(image_frames) == 0 or len(wrist_frames) == 0:
        return False, np.zeros((0, actions.shape[1]), dtype=np.float32), np.zeros((0, 7), dtype=np.float32), {}, None

    # If video streams are shorter than actions, randomly drop time-steps from actions/states/obs to match.
    T = min(len(image_frames), len(wrist_frames))
    keep_idx: np.ndarray | None = None
    if T < T_full:
        seed = int.from_bytes(hashlib.blake2b(f'{suite_name}:{original_task_id}:{trajectory_id}'.encode('utf-8'), digest_size=8).digest(), 'big')
        keep_idx = _random_keep_indices(T_full, T, seed=seed)
        actions = actions[keep_idx]
        states = states[keep_idx]
        image_frames = image_frames[:T]
        wrist_frames = wrist_frames[:T]

    images_dict = {
        'image': np.array(image_frames, dtype=np.uint8),
        'wrist_image': np.array(wrist_frames, dtype=np.uint8),
    }

    # --- force history: (T, H_out, 6) ---
    try:
        gnf = np.array(trajectory['obs']['gripper_net_force'])  # (T_full, H_sensor, 2, 3)
        if keep_idx is not None:
            gnf = gnf[keep_idx]
        inst_force = gnf[:, 0, :, :].reshape(T, 6).astype(np.float32)  # (T,6)
        gripper_force = _build_sliding_window_with_pad(inst_force, force_history_len)  # (T,H,6)
    except Exception:
        # Strict: no placeholder / no fallback.
        return False, np.zeros((0, actions.shape[1]), dtype=np.float32), np.zeros((0, 7), dtype=np.float32), {}, None

    return is_valid, actions, states, images_dict, gripper_force


@dataclass
class Config:
    """SoftVTBench-force conversion config (ContactForce replay env)."""

    task_suites: tuple[str, ...] = ('libero_10', 'libero_spatial', 'libero_goal', 'libero_object')
    chunks_size: int = 1000
    fps: int = 20
    # Optional: only convert tasks listed in benchmarks/datasets/softvtbench/config/softvtbench_tasks.json
    use_softvtbench_tasks: bool = False

    # Input root, containing:
    #   - replayed_demos/*.hdf5
    #   - video_datasets/<suite>_task<id>/videos/*.mp4
    # Paths are relative to the SoftVTBench simulator checkout by default.
    data_root: Path = Path('benchmarks/datasets/softvtbench_force')
    hdf5_folder: Path = Path('')
    video_dir: Path = Path('')

    # Optional: merge two datasets (strong/soft) into one output dataset.
    # Each root should contain:
    # - replayed_demos/*.hdf5
    # - video_datasets/<suite>_task<id>/videos/*.mp4
    strong_data_root: Path | None = None
    soft_data_root: Path | None = None
    # Prompt rewrite: only modify the `task` text; keep everything else unchanged.
    strong_adverb: str = 'firmly'
    soft_adverb: str = 'gently'
    # Multi-adverb version (deterministic "random" choice per trajectory).
    strong_adverbs: tuple[str, ...] = ('firmly', 'tightly')  # , 'forcefully'
    soft_adverbs: tuple[str, ...] = ('gently', 'softly')  # , 'lightly', 'delicately'
    prompt_seed: int = 0

    # Output LeRobot dataset directory
    # 固定输出到：.../benchmarks/datasets/softvtbench_force_pi0/<repo_name>
    output_dir: Path = Path('benchmarks/datasets/softvtbench_force_pi0')
    repo_name: str = ''

    # Output force history window length (H_out)
    force_history_len: int = 8

    # NOTE:
    # This conversion requires FORCE in actions (13D).
    # If your HDF5 actions are 7D, re-record with recorder_type=7dpf.

    def __post_init__(self):
        """Normalize paths and derive dataset dimensions."""
        # Normalize paths (do NOT prefix with repo_root).
        self.data_root = Path(self.data_root).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        if self.strong_data_root is not None:
            self.strong_data_root = Path(self.strong_data_root).expanduser().resolve()
        if self.soft_data_root is not None:
            self.soft_data_root = Path(self.soft_data_root).expanduser().resolve()

        # Single-source mode: derive from data_root.
        if self.strong_data_root is None and self.soft_data_root is None:
            if not self.hdf5_folder or str(self.hdf5_folder) in ('.', ''):
                self.hdf5_folder = self.data_root / 'replayed_demos'
            if not self.video_dir or str(self.video_dir) in ('.', ''):
                self.video_dir = self.data_root / 'video_datasets'
        if not self.repo_name:
            self.repo_name = 'softvtbench_force_all_libero_suites' if len(self.task_suites) > 1 else f'softvtbench_force_{self.task_suites[0]}'

        self.task_configs = load_all_task_configs(self.task_suites)
        self.state_shape = 7
        self.action_shape = 13


def _choose_adverb(seed: int, key: str, adverbs: tuple[str, ...]) -> str:
    if not adverbs:
        return ''
    digest = hashlib.blake2b(f'{seed}:{key}'.encode('utf-8'), digest_size=8).digest()
    idx = int.from_bytes(digest, 'big') % len(adverbs)
    return (adverbs[idx] or '').strip()


def _rewrite_instruction(instruction: str, adverb: str, seed: int, key: str) -> str:
    """Rewrite instruction with an adverb in a more natural English style (deterministic)."""
    instruction = (instruction or '').strip()
    adverb = (adverb or '').strip()
    if not adverb:
        return instruction
    if not instruction:
        return adverb

    lower = instruction.lower()
    if lower.startswith(f'{adverb} '):
        return instruction
    if lower.endswith(f' {adverb}'):
        return instruction

    style = _choose_adverb(seed, f'{key}:style', ('prefix', 'suffix'))
    if style == 'suffix':
        return f'{instruction} {adverb}'
    return f'{adverb} {instruction}'


def _iter_sources(config: Config) -> list[tuple[str, Path, Path, str]]:
    """Return [(label, hdf5_dir, video_root, adverb)]."""
    if config.strong_data_root is not None or config.soft_data_root is not None:
        if config.strong_data_root is None or config.soft_data_root is None:
            raise ValueError('Please set both strong_data_root and soft_data_root (or neither).')
        strong_root = Path(config.strong_data_root)
        soft_root = Path(config.soft_data_root)
        return [
            ('strong', strong_root / 'replayed_demos', strong_root / 'video_datasets', config.strong_adverb),
            ('soft', soft_root / 'replayed_demos', soft_root / 'video_datasets', config.soft_adverb),
        ]
    return [('default', Path(config.hdf5_folder), Path(config.video_dir), '')]


def get_all_task_hdf5_files_multi(
    config: Config, subset_map: dict[str, list[int]] | None = None
) -> dict[int, list[tuple[str, str]]]:
    """Get HDF5 files organized by global_task_id across sources.

    Returns:
        global_task_id -> list of (source_label, hdf5_file_path)
    """
    all_task_files: dict[int, list[tuple[str, str]]] = {}
    for source_label, hdf5_dir, _video_root, _adverb in _iter_sources(config):
        per_src = get_all_task_hdf5_files(hdf5_dir, config.task_suites, subset_map=subset_map)
        for k, v in per_src.items():
            all_task_files.setdefault(k, [])
            all_task_files[k].extend([(source_label, p) for p in v])
    return all_task_files


def main() -> None:
    """CLI entrypoint."""
    config = tyro.cli(Config)

    output_path = config.output_dir / config.repo_name
    if output_path.exists():
        shutil.rmtree(output_path)
    # IMPORTANT:
    # LeRobotDataset.create() will create `root` with exist_ok=False.
    # So we must NOT pre-create `output_path`, otherwise it will crash with FileExistsError.

    sources = _iter_sources(config)
    multi_mode = not (len(sources) == 1 and sources[0][0] == 'default')
    if multi_mode:
        print('\n[Multi-source] Merging two datasets (strong/soft) into one output:')
        for label, hdf5_dir, video_root, adv in sources:
            print(f"  - {label}: hdf5_dir={hdf5_dir}, video_root={video_root}, adverb='{adv}'")

    subset_map = _load_softvtbench_task_subset_json() if config.use_softvtbench_tasks else None
    if not multi_mode:
        all_task_files: dict[int, list] = get_all_task_hdf5_files(Path(config.hdf5_folder), config.task_suites, subset_map=subset_map)
    else:
        all_task_files = get_all_task_hdf5_files_multi(config, subset_map=subset_map)

    source_video_root = {lbl: vr for (lbl, _hd, vr, _adv) in sources}
    source_adverb = {lbl: adv for (lbl, _hd, _vr, adv) in sources}

    features = {
        'image': IMAGE_SETTING,
        'wrist_image': IMAGE_SETTING,
        'state': {'dtype': 'float32', 'shape': (config.state_shape,), 'names': ['state']},
        'actions': {'dtype': 'float32', 'shape': (config.action_shape,), 'names': ['actions']},
        'gripper_force': {'dtype': 'float32', 'shape': (config.force_history_len, 6), 'names': ['gripper_force']},
    }

    dataset = LeRobotDataset.create(
        repo_id=config.repo_name,
        root=output_path,
        robot_type='franka',
        fps=config.fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for global_task_id in sorted(all_task_files.keys()):
        suite_idx = global_task_id // 10
        original_task_id = global_task_id % 10
        suite_name = config.task_suites[suite_idx]

        task_hdf5_files = all_task_files[global_task_id]

        for item in tqdm(task_hdf5_files, desc=f'{suite_name}_task{original_task_id}'):
            if multi_mode:
                source_label, hdf5_file_path = item
                video_root = Path(source_video_root[source_label])
                adverb = source_adverb[source_label]
            else:
                source_label = 'default'
                hdf5_file_path = item
                video_root = Path(config.video_dir)
                adverb = ''

            videos_dir = video_root / f'{suite_name}_task{original_task_id}' / 'videos'
            failed_ids = check_failed_videos(videos_dir)

            with h5py.File(hdf5_file_path, 'r') as f:
                data_group = f['data']
                trajectory_ids = sorted(
                    [k for k in data_group.keys() if k.startswith('demo_')],
                    key=lambda x: int(x.split('_')[1]),
                )

                for trajectory_id in trajectory_ids:
                    if trajectory_id in [f'demo_{fid}' for fid in failed_ids]:
                        continue

                    trajectory = data_group[trajectory_id]
                    is_valid, actions, states, images_dict, gripper_force = combine_traj_and_images_softvtbench_force(
                        trajectory_id=trajectory_id,
                        trajectory=trajectory,
                        suite_name=suite_name,
                        original_task_id=original_task_id,
                        video_root=video_root,
                        force_history_len=config.force_history_len,
                    )
                    if not is_valid:
                        continue

                    base_task = config.task_configs.get(global_task_id, '')
                    if source_label == 'strong':
                        advs = config.strong_adverbs or (config.strong_adverb,)
                    elif source_label == 'soft':
                        advs = config.soft_adverbs or (config.soft_adverb,)
                    else:
                        advs = (adverb,) if adverb else ()
                    key = f'{source_label}:{suite_name}:{original_task_id}:{Path(hdf5_file_path).name}:{trajectory_id}'
                    chosen = _choose_adverb(config.prompt_seed, key, advs)
                    task_description = _rewrite_instruction(base_task, chosen, seed=config.prompt_seed, key=key)
                    for i in range(actions.shape[0]):
                        frame = {
                            'image': images_dict['image'][i],
                            'wrist_image': images_dict['wrist_image'][i],
                            'state': np.array(states[i], dtype=np.float32),
                            'actions': np.array(actions[i], dtype=np.float32),
                        }
                        # Strict: gripper_force must exist for a valid trajectory.
                        frame['gripper_force'] = np.array(gripper_force[i], dtype=np.float32)
                        lerobot_add_frame(dataset, frame, task_description)
                    dataset.save_episode()


if __name__ == '__main__':
    main()
