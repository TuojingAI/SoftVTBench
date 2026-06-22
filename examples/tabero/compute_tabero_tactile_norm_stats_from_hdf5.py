"""Compute Tabero tactile normalization stats directly from staged HDF5 demos."""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
from openpi import transforms


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    q = quat.copy()
    q[..., 0] = np.clip(q[..., 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(q[..., 0])
    sin_half = np.sqrt(np.maximum(1.0 - q[..., 0] ** 2, 0.0))
    axis = np.zeros_like(q[..., 1:])
    valid = sin_half > 1e-6
    axis[valid] = q[..., 1:][valid] / sin_half[valid, None]
    return (axis * angle[..., None]).astype(np.float32)


def _sliding_window_with_pad(values: np.ndarray, window: int) -> np.ndarray:
    idx = np.arange(values.shape[0])[:, None] - (window - 1 - np.arange(window))[None, :]
    idx = np.maximum(idx, 0)
    return values[idx]


def _state_from_demo(demo) -> np.ndarray:
    eef_pose = np.asarray(demo["obs"]["eef_pose"], dtype=np.float32)
    gripper = np.asarray(demo["obs"]["gripper_pos"], dtype=np.float32)
    gripper_scalar = gripper[:, 0:1] if gripper.ndim == 2 else gripper.reshape(-1, 1)
    return np.concatenate([eef_pose[:, :3], _quat2axisangle(eef_pose[:, 3:7]), gripper_scalar], axis=-1)


def _tactile_suffix_from_demo(demo, force_history_len: int) -> np.ndarray:
    force = np.asarray(demo["obs"]["gripper_net_force"], dtype=np.float32)
    inst_force = force[:, 0, :, :].reshape(force.shape[0], 6)
    return _sliding_window_with_pad(inst_force, force_history_len).astype(np.float32)


def _tactile_prefix_from_demo(demo, marker_history_len: int) -> np.ndarray:
    marker = np.asarray(demo["obs"]["gripper_marker_motion"], dtype=np.float32)
    init_pos = marker[:, :, 0, :, :]
    curr_pos = marker[:, :, 1, :, :]
    init_concat = init_pos[0].reshape(-1, 2)
    curr_concat = curr_pos.reshape(marker.shape[0], -1, 2)
    curr_hist = _sliding_window_with_pad(curr_concat, marker_history_len)
    tactile_marker_motion = np.zeros(
        (marker.shape[0], 1 + marker_history_len, curr_concat.shape[1], 2), dtype=np.float32
    )
    tactile_marker_motion[:, 0, :, :] = init_concat[None, :, :]
    tactile_marker_motion[:, 1:, :, :] = curr_hist
    return tactile_marker_motion[:, -1].reshape(marker.shape[0], -1).astype(np.float32)


def _action_chunks(actions: np.ndarray, horizon: int) -> np.ndarray:
    idx = np.arange(actions.shape[0])[:, None] + np.arange(horizon)[None, :]
    idx = np.minimum(idx, actions.shape[0] - 1)
    return actions[idx]


def main(
    data_root: pathlib.Path,
    output_dir: pathlib.Path,
    action_horizon: int = 50,
    batch_size: int = 256,
    force_history_len: int = 8,
    marker_history_len: int = 8,
    delta_actions: bool = True,
) -> None:
    hdf5_dir = data_root / "replayed_demos"
    files = sorted(hdf5_dir.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {hdf5_dir}")

    keys = ["state", "actions", "tactile_prefix", "tactile_suffix"]
    stats = {key: normalize.RunningStats() for key in keys}
    pending = {key: [] for key in keys}
    delta = transforms.DeltaActions(transforms.make_bool_mask(6)) if delta_actions else None

    def flush() -> None:
        for key, values in pending.items():
            if values:
                stats[key].update(np.concatenate(values, axis=0))
                values.clear()

    demos = 0
    frames = 0
    for path in tqdm.tqdm(files, desc="HDF5 files"):
        with h5py.File(path, "r") as f:
            for name, demo in f["data"].items():
                if not name.startswith("demo_"):
                    continue
                actions = np.asarray(demo["actions"], dtype=np.float32)
                if actions.shape[-1] != 13:
                    raise ValueError(f"{path}:{name} has action dim {actions.shape[-1]}, expected 13")
                state = _state_from_demo(demo)
                action_chunks = _action_chunks(actions, action_horizon)
                sample = {"state": state, "actions": action_chunks}
                if delta is not None:
                    sample = delta(sample)
                pending["state"].append(sample["state"])
                pending["actions"].append(sample["actions"])
                pending["tactile_prefix"].append(_tactile_prefix_from_demo(demo, marker_history_len))
                pending["tactile_suffix"].append(_tactile_suffix_from_demo(demo, force_history_len))
                demos += 1
                frames += actions.shape[0]
                if frames % batch_size < actions.shape[0]:
                    flush()

    flush()
    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    print(f"Processed demos={demos}, frames={frames}")
    print(f"Writing stats to: {output_dir}")
    normalize.save(output_dir, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
