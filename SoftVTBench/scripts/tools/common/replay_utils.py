# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video processing utilities for demonstration scripts."""
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch


def _resolve_single_hdf5_from_dir(
    *,
    env_var: str,
    env_label: str,
    task_suite: str,
    task_id: int,
    pattern: str,
) -> str:
    """Resolve a single per-task HDF5 from a directory env var with strict, unambiguous matching."""

    dir_str = os.environ.get(env_var, '').strip()
    if not dir_str:
        raise ValueError(f'Missing env var: {env_var}')

    base_dir = Path(dir_str).expanduser().resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f'{env_label} does not exist: {base_dir}')

    matches: list[Path] = sorted(base_dir.glob(pattern))

    # de-dup while preserving order
    dedup: list[Path] = []
    seen: set[str] = set()
    for m in matches:
        s = str(m)
        if s not in seen:
            seen.add(s)
            dedup.append(m)

    if not dedup:
        raise FileNotFoundError(f'No HDF5 matched pattern {pattern} under {env_var}={base_dir}')
    if len(dedup) > 1:
        raise ValueError(
            f'Ambiguous HDF5 matches for {task_suite} task{int(task_id)} under {env_var}={base_dir}:\n'
            + '\n'.join([f'  - {p}' for p in dedup])
        )
    return str(dedup[0])


def axisangle2quat(axisangle: np.ndarray) -> np.ndarray:
    """
    Converts axis-angle format to quaternion.
    
    Args:
        axisangle: (ax, ay, az) axis-angle exponential coordinates
    
    Returns:
        (w, x, y, z) quaternion
    """
    angle = np.linalg.norm(axisangle)
    if math.isclose(angle, 0.0):
        # Zero rotation quaternion
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axisangle / angle
    half_angle = angle / 2.0
    w = math.cos(half_angle)
    xyz = axis * math.sin(half_angle)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def convert_action_axisangle_to_quat(action: torch.Tensor) -> torch.Tensor:
    """
    Convert 7D action (position + axis-angle + gripper) to 8D action (position + quaternion + gripper).
    
    Args:
        action: 7D action tensor (x, y, z, ax, ay, az, gripper)
    
    Returns:
        8D action tensor (x, y, z, qw, qx, qy, qz, gripper)
    """
    # Extract position, axis-angle, and gripper
    pos = action[:3]
    axisangle = action[3:6]
    gripper = action[6:7]
    
    # Convert axis-angle to quaternion
    quat = axisangle2quat(axisangle.cpu().numpy())
    quat_tensor = torch.tensor(quat, device=action.device, dtype=action.dtype)
    
    # Combine as 8D: (x, y, z, qw, qx, qy, qz, gripper)
    return torch.cat([pos, quat_tensor, gripper])


def resolve_hdf5_from_replayed_demos_dir(task_suite: str, task_id: int) -> str:
    """Resolve a per-task HDF5 file from REPLAYED_DEMOS_DIR by (task_suite, task_id).

    This is a strict resolver used for non-libero/custom suites, to support the same UX
    as Libero: pick a dataset by (task_suite, task_id), not by an explicit file path.

    Rules:
    - Requires env var: REPLAYED_DEMOS_DIR
    - Matches files under that directory using:
        1) {task_suite}_task{task_id}_*_demo.hdf5
    - If multiple matches exist, raise (avoid silently picking the wrong file).
    """
    tid = int(task_id)
    pattern = f'{task_suite}_task{tid}_*_demo.hdf5'
    return _resolve_single_hdf5_from_dir(
        env_var='REPLAYED_DEMOS_DIR',
        env_label='REPLAYED_DEMOS_DIR',
        task_suite=task_suite,
        task_id=tid,
        pattern=pattern,
    )


def resolve_hdf5_from_traj_source_dir(task_suite: str, task_id: int) -> str:
    """Resolve a per-task assembled_hdf5 file from HDF5_TRAJ_SOURCE_DIR by (task_suite, task_id).

    Rules:
    - Requires env var: HDF5_TRAJ_SOURCE_DIR
    - Matches files under that directory using:
        1) {task_suite}_task{task_id}_*_demo.hdf5
    - If multiple matches exist, raise (avoid silently picking the wrong file).
    """
    tid = int(task_id)
    pattern = f'{task_suite}_task{tid}_*_demo.hdf5'
    return _resolve_single_hdf5_from_dir(
        env_var='HDF5_TRAJ_SOURCE_DIR',
        env_label='HDF5_TRAJ_SOURCE_DIR',
        task_suite=task_suite,
        task_id=tid,
        pattern=pattern,
    )


def resolve_hdf5_from_recorded_demos_dir(task_suite: str, task_id: int) -> str:
    """Resolve a per-task recorded demo HDF5 from RECORDED_DEMOS_DIR by (task_suite, task_id).

    Rules:
    - Requires env var: RECORDED_DEMOS_DIR
    - Matches files under that directory using:
        1) {task_suite}_task{task_id}_*_recorded_demo.hdf5
    - If multiple matches exist, raise (avoid silently picking the wrong file).
    """
    tid = int(task_id)
    pattern = f'{task_suite}_task{tid}_*_recorded_demo.hdf5'
    return _resolve_single_hdf5_from_dir(
        env_var='RECORDED_DEMOS_DIR',
        env_label='RECORDED_DEMOS_DIR',
        task_suite=task_suite,
        task_id=tid,
        pattern=pattern,
    )


def resolve_input_hdf5(
    *,
    task: str | None,
    task_suite: str,
    task_id: int,
    prefer: str = 'auto',
) -> tuple[str, str]:
    """Resolve input dataset HDF5 for replay/recording in one place.

    prefer:
    - 'auto'     : choose by env type (task-space -> replayed, otherwise -> assembled)
    - 'replayed' : resolve from REPLAYED_DEMOS_PATH or REPLAYED_DEMOS_DIR
    - 'assembled': resolve from HDF5_TRAJ_SOURCE_DIR
    - 'recorded' : resolve from RECORDED_DEMOS_PATH or RECORDED_DEMOS_DIR
    """

    if prefer not in ('auto', 'replayed', 'assembled', 'recorded'):
        raise ValueError(f"Unsupported prefer='{prefer}'")

    if prefer == 'replayed':
        direct = os.environ.get('REPLAYED_DEMOS_PATH', '').strip()
        if direct:
            return direct, f'Using REPLAYED_DEMOS_PATH: {direct}'
        p = resolve_hdf5_from_replayed_demos_dir(task_suite, task_id)
        return p, f'Using REPLAYED_DEMOS_DIR: {p}'

    if prefer == 'assembled':
        p = resolve_hdf5_from_traj_source_dir(task_suite, task_id)
        return p, f'Using HDF5_TRAJ_SOURCE_DIR: {p}'

    if prefer == 'recorded':
        direct = os.environ.get('RECORDED_DEMOS_PATH', '').strip()
        if direct:
            return direct, f'Using RECORDED_DEMOS_PATH: {direct}'
        p = resolve_hdf5_from_recorded_demos_dir(task_suite, task_id)
        return p, f'Using RECORDED_DEMOS_DIR: {p}'

    # prefer == 'auto'
    is_task_space_env = bool(
        task
        and (
            'IK' in task
            or 'Osc' in task
            or 'Isaac-Libero-Franka-Hybrid-' in task
        )
    )
    is_replay_env = bool(task and 'Replay' in task)
    requires_task_space = bool(is_task_space_env and not is_replay_env)
    if requires_task_space:
        return resolve_input_hdf5(task=task, task_suite=task_suite, task_id=task_id, prefer='replayed')
    return resolve_input_hdf5(task=task, task_suite=task_suite, task_id=task_id, prefer='assembled')


def select_data_source_for_libero(
    task: str | None,
    task_suite: str | None = None,
) -> tuple[str | None, str]:
    """
    Select appropriate data source for Libero tasks based on environment type.
    
    Args:
        task: Task environment name (e.g., "Isaac-Libero-Franka-IK-v0")
        task_suite: Task suite name (for environment variable access)
        
    Returns:
        Tuple of (dataset_file_path, data_source_description)
    """
    if task_suite is None or not task_suite.startswith('libero'):
        return None, ''
    task_id_env = int(os.environ.get('TASK_ID', '0'))
    path, desc = resolve_input_hdf5(task=task, task_suite=task_suite, task_id=task_id_env, prefer='auto')
    return path, desc


def display_action_info(
    episode_data,
    task: str,
    dataset_file: str,
    device: torch.device,
) -> None:
    """
    Display action information based on environment type and data source.
    
    Args:
        episode_data: Loaded episode data
        task: Task environment name
        dataset_file: Path to dataset file
        device: Torch device
    """
    is_task_space_env = (
        "IK" in task
        or "Osc" in task
        or "Isaac-Libero-Franka-Hybrid-" in task
    )
    is_replayed_source = dataset_file and "replayed_demos" in str(dataset_file).lower()
    
    action_shape = episode_data.data['actions'].shape
    
    if is_task_space_env:
        if is_replayed_source:
            print(f"   → Using Task Space actions from replayed_demos ({action_shape[0]} steps, {action_shape[1]}D)")
        else:
            print(f"   ⚠️  WARNING: Task-space env expects task-space actions, but dataset does not look like replayed_demos.")
            print(f"   → Current data source: {dataset_file}")
            print(f"   → Fix: use replayed_demos (Task Space) and the matching env pairing.")
    else:
        source = "replayed_demos" if is_replayed_source else "assembled_hdf5"
        print(f"   → Using Joint Space actions from {source} ({action_shape[0]} steps, {action_shape[1]}D)")


def extract_eef_pose_from_robot_states(episode_data, device: torch.device) -> torch.Tensor | None:
    """Deprecated: task-space extraction during replay is intentionally disabled.

    Current policy: do not "fix" dataset/env mismatches in replay. Pick the correct dataset
    (e.g., replayed_demos for task-space envs) instead of extracting task-space actions from robot_states.
    """
    _ = (episode_data, device)
    return None


def compare_states(state_from_dataset, runtime_state, runtime_env_index) -> (bool, str):
    """Compare states from dataset and runtime.

    Args:
        state_from_dataset: State from dataset.
        runtime_state: State from runtime.
        runtime_env_index: Index of the environment in the runtime states to be compared.

    Returns:
        bool: True if states match, False otherwise.
        str: Log message if states don't match.
    """
    states_matched = True
    output_log = ""
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name][runtime_env_index]
                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name]
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(f"State shape of {state_name} for asset {asset_name} don't match")
                for i in range(len(dataset_asset_state)):
                    if abs(dataset_asset_state[i] - runtime_asset_state[i]) > 0.01:
                        states_matched = False
                        output_log += f'\tState ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] don\'t match\r\n'
                        output_log += f"\t  Dataset:\t{dataset_asset_state[i]}\r\n"
                        output_log += f"\t  Runtime: \t{runtime_asset_state[i]}\r\n"
    return states_matched, output_log


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


def create_video_from_images(input_pattern: str, output_video: str, framerate: float = 20.0):
    """Create a video from a sequence of images using ffmpeg.

    Args:
        input_pattern: Input image pattern (e.g. "frame_%04d_rgb.png")
        output_video: Output video path
        framerate: Video framerate (default: 20.0)
    """
    ffmpeg_exe = os.environ.get("FFMPEG_EXE") or shutil.which("ffmpeg")
    if not ffmpeg_exe or not Path(ffmpeg_exe).exists():
        try:
            import imageio_ffmpeg

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = "/usr/bin/ffmpeg"

    codec = os.environ.get("SPATIAL_FFMPEG_CODEC", "libx264")
    pix_fmt = os.environ.get("SPATIAL_FFMPEG_PIX_FMT", "yuv420p")
    crf = os.environ.get("SPATIAL_FFMPEG_CRF", "18")
    preset = os.environ.get("SPATIAL_FFMPEG_PRESET", "medium")

    cmd = [
        ffmpeg_exe,
        "-nostdin",
        "-y",
        "-framerate",
        str(framerate),
        "-i",
        input_pattern,
        "-c:v",
        codec,
        "-pix_fmt",
        pix_fmt,
        "-crf",
        str(crf),
        "-preset",
        preset,
        output_video,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error creating video {output_video}: {e}")
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg: sudo apt install ffmpeg")


def _base_path(task_suite: str | None, task_id: int | None, task: str | None, root_dir_prefix: str = "") -> str:
    """Compute base path for a task or suite."""
    if task_suite is not None:
        return f"{root_dir_prefix}{task_suite}_task{task_id}"
    else:
        return f"{root_dir_prefix}{task}"


def setup_video_directories(
    video: bool,
    task_suite: str | None = None,
    task_id: int | None = None,
    task: str | None = None,
    root_dir_prefix: str = "",
):
    """Setup video save directories.

    Args:
        video: Whether video recording is enabled.
        task_suite: Task suite name.
        task_id: Task ID.
        task: Task name.
        root_dir_prefix: Optional root prefix (e.g., "replayed_videos/").

    Returns:
        Video save directory path or None.
    """
    if video:
        base = _base_path(task_suite, task_id, task, root_dir_prefix)
        video_save_dir = f"{base}/videos"
        os.makedirs(video_save_dir, exist_ok=True)
        return video_save_dir
    return None


def setup_replay_output_directories(
    video: bool,
    tactile_sensor_list: list | None = None,
    task_suite: str | None = None,
    task_id: int | None = None,
    task: str | None = None,
    root_dir_prefix: str = "replayed_videos/",
):
    """Setup replay output directories for videos and tactile outputs.

    Returns:
        (video_save_dir, tactile_outputs_save_dir) or (None, None) if video=False
    """
    if not video:
        return None, None
    base = _base_path(task_suite, task_id, task, root_dir_prefix)
    video_save_dir = f"{base}/videos"
    os.makedirs(video_save_dir, exist_ok=True)

    tactile_outputs_save_dir = None
    if tactile_sensor_list:
        tactile_outputs_save_dir = f"{base}/tactile_outputs"
        os.makedirs(tactile_outputs_save_dir, exist_ok=True)

    return video_save_dir, tactile_outputs_save_dir


def _images_demo_dir(
    current_episode_index: int,
    task_suite: str | None,
    task_id: int | None,
    task: str | None,
    root_dir_prefix: str = "",
) -> str:
    base = _base_path(task_suite, task_id, task, root_dir_prefix)
    return f"{base}/videos/demo_{current_episode_index}"


def _images_failed_dir(
    current_episode_index: int,
    task_suite: str | None,
    task_id: int | None,
    task: str | None,
    root_dir_prefix: str = "",
) -> str:
    base = _base_path(task_suite, task_id, task, root_dir_prefix)
    return f"{base}/videos/failed_{current_episode_index}"


def _tactile_demo_dir(
    current_episode_index: int,
    task_suite: str | None,
    task_id: int | None,
    task: str | None,
    root_dir_prefix: str = "",
) -> str:
    base = _base_path(task_suite, task_id, task, root_dir_prefix)
    return f"{base}/tactile_outputs/demo_{current_episode_index}"


def _tactile_failed_dir(
    current_episode_index: int,
    task_suite: str | None,
    task_id: int | None,
    task: str | None,
    root_dir_prefix: str = "",
) -> str:
    base = _base_path(task_suite, task_id, task, root_dir_prefix)
    return f"{base}/tactile_outputs/failed_{current_episode_index}"


def _tactile_arrays_demo_dir(
    current_episode_index: int,
    task_suite: str | None,
    task_id: int | None,
    task: str | None,
    root_dir_prefix: str = "",
) -> str:
    base = _base_path(task_suite, task_id, task, root_dir_prefix)
    return f"{base}/tactile_arrays/demo_{current_episode_index}"


def _tactile_arrays_failed_dir(
    current_episode_index: int,
    task_suite: str | None,
    task_id: int | None,
    task: str | None,
    root_dir_prefix: str = "",
) -> str:
    base = _base_path(task_suite, task_id, task, root_dir_prefix)
    return f"{base}/tactile_arrays/failed_{current_episode_index}"


def _tactile_arrays_output_dir(
    task_suite: str | None,
    task_id: int | None,
    task: str | None,
    root_dir_prefix: str = "",
) -> str:
    base = _base_path(task_suite, task_id, task, root_dir_prefix)
    return f"{base}/tactile_arrays"


_TACTILE_ALL_OUTPUT_TYPES = ["markers_rgb", "tactile_rgb", "height_map", "marker_motion"]
_TACTILE_IMAGE_OUTPUT_TYPES = {"markers_rgb", "tactile_rgb", "height_map", "camera_depth"}
_TACTILE_ARRAY_OUTPUT_TYPES = {"tactile_rgb", "height_map", "marker_motion"}
_TACTILE_ALLOWED_OUTPUT_TYPES = set(_TACTILE_ALL_OUTPUT_TYPES) | {"camera_depth"}


def _normalize_tactile_output_types(tactile_output_type: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if tactile_output_type is None:
        return []
    if isinstance(tactile_output_type, str):
        output_types = [tactile_output_type]
    else:
        output_types = list(tactile_output_type)

    if "all" in output_types:
        output_types = list(_TACTILE_ALL_OUTPUT_TYPES)

    normalized: list[str] = []
    for output_type in output_types:
        if output_type not in _TACTILE_ALLOWED_OUTPUT_TYPES:
            allowed = sorted(_TACTILE_ALLOWED_OUTPUT_TYPES | {"all"})
            raise ValueError(f"Invalid tactile output type '{output_type}'. Allowed values: {allowed}")
        if output_type not in normalized:
            normalized.append(output_type)
    return normalized


def _rgb_to_bgr_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.float32)
        finite = np.isfinite(frame)
        if finite.any() and float(np.nanmax(frame)) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def _depth_to_bgr_uint8(frame: np.ndarray) -> np.ndarray:
    depth = np.squeeze(np.asarray(frame)).astype(np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        gray = np.zeros(depth.shape, dtype=np.uint8)
    else:
        valid = depth[finite]
        lo, hi = np.percentile(valid, [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        gray = np.clip((np.nan_to_num(depth, nan=lo, posinf=hi, neginf=lo) - lo) / (hi - lo) * 255.0, 0, 255)
        gray = gray.astype(np.uint8)
    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    return cv2.applyColorMap(gray, colormap)


def _tactile_output_to_bgr_image(output_type: str, frame: np.ndarray) -> np.ndarray:
    if output_type in ("markers_rgb", "tactile_rgb"):
        return _rgb_to_bgr_uint8(frame)
    if output_type in ("height_map", "camera_depth"):
        return _depth_to_bgr_uint8(frame)
    raise ValueError(f"'{output_type}' cannot be written as an image")


def _process_tactile_arrays(
    source_dir: str,
    current_episode_index: int,
    tactile_sensor_list: list,
    tactile_output_type: str | list[str],
    task_suite: str | None,
    task_id: int | None,
    task: str | None,
    root_dir_prefix: str,
    prefix: str = "demo",
):
    output_types = [t for t in _normalize_tactile_output_types(tactile_output_type) if t in _TACTILE_ARRAY_OUTPUT_TYPES]
    if not output_types or not os.path.exists(source_dir):
        return

    output_dir = _tactile_arrays_output_dir(task_suite, task_id, task, root_dir_prefix)
    os.makedirs(output_dir, exist_ok=True)

    for tac_sensor in tactile_sensor_list:
        for output_type in output_types:
            frames = sorted(Path(source_dir).glob(f"frame_*_{tac_sensor}_{output_type}.npy"))
            if not frames:
                continue
            data = np.stack([np.load(str(frame_path), allow_pickle=False) for frame_path in frames], axis=0)
            output_path = os.path.join(output_dir, f"{prefix}_{current_episode_index}_{tac_sensor}_{output_type}.npz")
            np.savez_compressed(
                output_path,
                data=data,
                sensor=np.array(tac_sensor),
                output_type=np.array(output_type),
                frame_count=np.array(data.shape[0], dtype=np.int32),
            )


def save_camera_images(
    env,
    env_id: int,
    current_episode_index: int | None,
    frame_index: int,
    video: bool,
    camera_view_list: list,
    save_depth: bool = False,
    task_suite: str | None = None,
    task_id: int | None = None,
    task: str | None = None,
    root_dir_prefix: str = "",
):
    """Save RGB and depth images for the current frame."""
    if not video or current_episode_index is None:
        return

    demo_save_dir = _images_demo_dir(current_episode_index, task_suite, task_id, task, root_dir_prefix)
    os.makedirs(demo_save_dir, exist_ok=True)

    for view in camera_view_list:
        rgb_cam = env.scene.sensors[f"{view}_cam"].data.output["rgb"].cpu().numpy()[env_id]
        rgb_path = os.path.join(demo_save_dir, f"frame_{frame_index:04d}_{view}_rgb.png")
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb_cam, cv2.COLOR_RGB2BGR))

        if save_depth:
            depth_cam = env.scene.sensors[f"{view}_cam"].data.output["distance_to_image_plane"].cpu().numpy()[env_id]
            depth_16bit = (depth_cam * 1000).astype(np.uint16)
            depth_path = os.path.join(demo_save_dir, f"frame_{frame_index:04d}_{view}_depth.png")
            cv2.imwrite(depth_path, depth_16bit)


def process_successful_demo_videos(
    current_episode_index: int | None,
    video_save_dir: str | None,
    video: bool,
    camera_view_list: list,
    save_depth: bool = False,
    task_suite: str | None = None,
    task_id: int | None = None,
    task: str | None = None,
    root_dir_prefix: str = "",
):
    """Process videos for successful demos."""
    if not video or current_episode_index is None:
        return

    demo_save_dir = _images_demo_dir(current_episode_index, task_suite, task_id, task, root_dir_prefix)

    # 将逐帧图片编码成视频
    num_video_files = 0
    for view in camera_view_list:
        for img_type in ["rgb", "depth"] if save_depth else ["rgb"]:
            input_pattern = os.path.join(demo_save_dir, f"frame_%04d_{view}_{img_type}.png")
            output_video = os.path.join(video_save_dir, f"demo_{current_episode_index}_{view}_{img_type}.mp4")
            create_video_from_images(input_pattern, output_video)
            num_video_files += 1

    # 打印最终视频存储位置（与 HDF5 日志风格一致，更关注“结果”而不是中间临时图片目录）
    print(
        f"🎥 Replay videos saved under: {video_save_dir} "
        f"(demo_{current_episode_index}, {num_video_files} file(s))."
    )

    # 清理临时图片目录（静默或仅在报错时提示）
    try:
        if os.path.exists(demo_save_dir):
            shutil.rmtree(demo_save_dir)
    except Exception as e:
        print(f"Error removing images folder: {e}")


def process_failed_demo_videos(
    current_episode_index: int | None,
    video_save_dir: str | None,
    video: bool,
    camera_view_list: list,
    save_depth: bool = False,
    task_suite: str | None = None,
    task_id: int | None = None,
    task: str | None = None,
    root_dir_prefix: str = "",
):
    """Process videos for failed demos."""
    if not video or current_episode_index is None:
        return

    demo_save_dir = _images_demo_dir(current_episode_index, task_suite, task_id, task, root_dir_prefix)
    failed_save_dir = _images_failed_dir(current_episode_index, task_suite, task_id, task, root_dir_prefix)

    if os.path.exists(demo_save_dir):
        # If failed_dir already exists, remove it first to avoid rename error
        if os.path.exists(failed_save_dir):
            try:
                shutil.rmtree(failed_save_dir)
            except Exception as e:
                print(f"Warning: Failed to remove existing failed_dir {failed_save_dir}: {e}")
        os.rename(demo_save_dir, failed_save_dir)
        print(f"## Renamed unsuccessful demo folder from {demo_save_dir} to {failed_save_dir}")

        for view in camera_view_list:
            for img_type in ["rgb", "depth"] if save_depth else ["rgb"]:
                input_pattern = os.path.join(failed_save_dir, f"frame_%04d_{view}_{img_type}.png")
                output_video = os.path.join(video_save_dir, f"failed_{current_episode_index}_{view}_{img_type}.mp4")
                create_video_from_images(input_pattern, output_video)

        try:
            shutil.rmtree(failed_save_dir)
            print(f"Successfully removed failed demo images folder: {failed_save_dir}")
        except Exception as e:
            print(f"Error removing failed demo images folder: {e}")
    else:
        print(f"## demo_save_dir for demo_{current_episode_index} does not exist...")


def save_tactile_images(
    env,
    env_id: int,
    current_episode_index: int | None,
    frame_index: int,
    video: bool,
    tactile_sensor_list: list,
    tactile_output_type: str = "markers_rgb",
    task_suite: str | None = None,
    task_id: int | None = None,
    task: str | None = None,
    root_dir_prefix: str = "",
):
    """Save tactile frame images and raw arrays for the current replay frame."""
    if not video or current_episode_index is None or not tactile_sensor_list:
        return

    output_types = _normalize_tactile_output_types(tactile_output_type)
    demo_save_dir = _tactile_demo_dir(current_episode_index, task_suite, task_id, task, root_dir_prefix)
    os.makedirs(demo_save_dir, exist_ok=True)
    arrays_save_dir = _tactile_arrays_demo_dir(current_episode_index, task_suite, task_id, task, root_dir_prefix)
    if any(output_type in _TACTILE_ARRAY_OUTPUT_TYPES for output_type in output_types):
        os.makedirs(arrays_save_dir, exist_ok=True)

    for tac_sensor in tactile_sensor_list:
        sensor = env.scene.sensors[tac_sensor]
        outputs = sensor.data.output
        for output_type in output_types:
            if output_type not in outputs:
                available = sorted([str(k) for k in outputs.keys()])
                raise KeyError(
                    f"tactile_output_type='{output_type}' not available for sensor '{tac_sensor}'. "
                    f"Available keys: {available}"
                )
            tactile_data = outputs[output_type].detach().cpu().numpy()[env_id]

            if output_type in _TACTILE_ARRAY_OUTPUT_TYPES:
                array_path = os.path.join(arrays_save_dir, f"frame_{frame_index:04d}_{tac_sensor}_{output_type}.npy")
                np.save(array_path, tactile_data)

            if output_type in _TACTILE_IMAGE_OUTPUT_TYPES:
                image_path = os.path.join(demo_save_dir, f"frame_{frame_index:04d}_{tac_sensor}_{output_type}.png")
                cv2.imwrite(image_path, _tactile_output_to_bgr_image(output_type, tactile_data))


def process_successful_tactile_videos(
    current_episode_index: int | None,
    tactile_outputs_save_dir: str | None,
    video: bool,
    tactile_sensor_list: list,
    tactile_output_type: str = "markers_rgb",
    task_suite: str | None = None,
    task_id: int | None = None,
    task: str | None = None,
    root_dir_prefix: str = "",
):
    """Encode tactile image sequences and pack raw tactile arrays for successful demos."""
    if not video or current_episode_index is None:
        return

    output_types = _normalize_tactile_output_types(tactile_output_type)
    tactile_demo_dir = _tactile_demo_dir(current_episode_index, task_suite, task_id, task, root_dir_prefix)
    tactile_arrays_dir = _tactile_arrays_demo_dir(current_episode_index, task_suite, task_id, task, root_dir_prefix)

    _process_tactile_arrays(
        tactile_arrays_dir,
        current_episode_index,
        tactile_sensor_list,
        output_types,
        task_suite,
        task_id,
        task,
        root_dir_prefix,
        prefix="demo",
    )

    if os.path.exists(tactile_demo_dir) and tactile_outputs_save_dir is not None:
        for tac_sensor in tactile_sensor_list:
            for output_type in output_types:
                if output_type not in _TACTILE_IMAGE_OUTPUT_TYPES:
                    continue
                first_frame = os.path.join(tactile_demo_dir, f"frame_0000_{tac_sensor}_{output_type}.png")
                if not os.path.exists(first_frame):
                    continue
                input_pattern = os.path.join(tactile_demo_dir, f"frame_%04d_{tac_sensor}_{output_type}.png")
                output_video = os.path.join(
                    tactile_outputs_save_dir, f"demo_{current_episode_index}_{tac_sensor}_{output_type}.mp4"
                )
                create_video_from_images(input_pattern, output_video)

    try:
        if os.path.exists(tactile_demo_dir):
            shutil.rmtree(tactile_demo_dir)
        if os.path.exists(tactile_arrays_dir):
            shutil.rmtree(tactile_arrays_dir)
    except Exception as e:
        print(f"Error removing tactile temporary folder: {e}")


def process_failed_tactile_videos(
    current_episode_index: int | None,
    tactile_outputs_save_dir: str | None,
    video: bool,
    tactile_sensor_list: list,
    tactile_output_type: str = "markers_rgb",
    task_suite: str | None = None,
    task_id: int | None = None,
    task: str | None = None,
    root_dir_prefix: str = "",
):
    """Encode tactile image sequences and pack raw tactile arrays for failed demos."""
    if not video or current_episode_index is None:
        return

    output_types = _normalize_tactile_output_types(tactile_output_type)
    tactile_demo_dir = _tactile_demo_dir(current_episode_index, task_suite, task_id, task, root_dir_prefix)
    tactile_failed_dir = _tactile_failed_dir(current_episode_index, task_suite, task_id, task, root_dir_prefix)
    tactile_arrays_demo_dir = _tactile_arrays_demo_dir(current_episode_index, task_suite, task_id, task, root_dir_prefix)
    tactile_arrays_failed_dir = _tactile_arrays_failed_dir(current_episode_index, task_suite, task_id, task, root_dir_prefix)

    if os.path.exists(tactile_demo_dir):
        # If failed_dir already exists, remove it first to avoid rename error
        if os.path.exists(tactile_failed_dir):
            try:
                shutil.rmtree(tactile_failed_dir)
            except Exception as e:
                print(f"Warning: Failed to remove existing failed_dir {tactile_failed_dir}: {e}")
        os.rename(tactile_demo_dir, tactile_failed_dir)
        print(f"## Renamed unsuccessful tactile folder from {tactile_demo_dir} to {tactile_failed_dir}")

        if tactile_outputs_save_dir is not None:
            for tac_sensor in tactile_sensor_list:
                for output_type in output_types:
                    if output_type not in _TACTILE_IMAGE_OUTPUT_TYPES:
                        continue
                    first_frame = os.path.join(tactile_failed_dir, f"frame_0000_{tac_sensor}_{output_type}.png")
                    if not os.path.exists(first_frame):
                        continue
                    input_pattern = os.path.join(tactile_failed_dir, f"frame_%04d_{tac_sensor}_{output_type}.png")
                    output_video = os.path.join(
                        tactile_outputs_save_dir, f"failed_{current_episode_index}_{tac_sensor}_{output_type}.mp4"
                    )
                    create_video_from_images(input_pattern, output_video)

        try:
            if os.path.exists(tactile_failed_dir):
                shutil.rmtree(tactile_failed_dir)
        except Exception as e:
            print(f"Error removing failed tactile images folder: {e}")
    else:
        print(f"## tactile_save_dir for demo_{current_episode_index} does not exist...")

    if os.path.exists(tactile_arrays_demo_dir):
        if os.path.exists(tactile_arrays_failed_dir):
            try:
                shutil.rmtree(tactile_arrays_failed_dir)
            except Exception as e:
                print(f"Warning: Failed to remove existing failed tactile arrays dir {tactile_arrays_failed_dir}: {e}")
        os.rename(tactile_arrays_demo_dir, tactile_arrays_failed_dir)
        _process_tactile_arrays(
            tactile_arrays_failed_dir,
            current_episode_index,
            tactile_sensor_list,
            output_types,
            task_suite,
            task_id,
            task,
            root_dir_prefix,
            prefix="failed",
        )
        try:
            if os.path.exists(tactile_arrays_failed_dir):
                shutil.rmtree(tactile_arrays_failed_dir)
        except Exception as e:
            print(f"Error removing failed tactile arrays folder: {e}")


def write_failure_jsonl(output_failure_record_file: str, key: str, failed_demo_ids: list[int]):
    """Append/merge failed demo IDs to a JSONL file with per-task key.

    - Merges with existing entries
    - Removes duplicates while keeping order
    """
    if not failed_demo_ids:
        return

    # Read existing
    existing: dict[str, list[int]] = {}
    if os.path.exists(output_failure_record_file):
        try:
            with open(output_failure_record_file) as f:
                for line in f:
                    data = json.loads(line.strip())
                    for k, v in data.items():
                        existing.setdefault(k, [])
                        existing[k].extend(v)
        except (json.JSONDecodeError, FileNotFoundError):
            existing = {}

    # Update current key
    current = existing.get(key, [])
    current.extend(failed_demo_ids)
    # Remove duplicates preserve order
    dedup = list(dict.fromkeys(current))
    existing[key] = dedup

    # Write back
    with open(output_failure_record_file, "w") as f:
        for k, v in existing.items():
            f.write(json.dumps({k: v}) + "\n")
    print(f"Failed demo IDs written to {output_failure_record_file}: {key}: {failed_demo_ids}")
