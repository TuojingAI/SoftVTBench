#!/usr/bin/env python3
"""Verify Tabero HDF5 trajectory lengths match all external MP4 streams.

This checks the contract used by the Tabero converters: frame index i in each
video stream is paired with HDF5 index i for actions/states/tactile arrays.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import h5py


VIDEO_SUFFIXES = (
    ("agentview", "videos", "agentview_rgb"),
    ("wrist", "videos", "eye_in_hand_rgb"),
    ("gsmini_left", "tactile_outputs", "gsmini_left_tactile_rgb"),
    ("gsmini_right", "tactile_outputs", "gsmini_right_tactile_rgb"),
)


def _count_mp4_frames(path: Path, decode: bool = False) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return -1
    if not decode:
        count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
        return count
    count = 0
    while True:
        ok, _frame = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count


def _parse_suite_task(hdf5_path: Path) -> tuple[str, int]:
    stem = hdf5_path.name
    prefix, rest = stem.split("_task", 1)
    task_id = int(rest.split("_", 1)[0])
    return prefix, task_id


def verify(root: Path, decode: bool = False, max_mismatches: int = 20) -> int:
    hdf5_dir = root / "replayed_demos"
    video_root = root / "video_datasets"
    if not hdf5_dir.exists():
        raise FileNotFoundError(f"missing hdf5 dir: {hdf5_dir}")
    if not video_root.exists():
        raise FileNotFoundError(f"missing video dir: {video_root}")

    checked = 0
    mismatches: list[str] = []
    missing: list[str] = []
    summary = Counter()

    for hdf5_path in sorted(hdf5_dir.glob("*_demo.hdf5")):
        suite, task_id = _parse_suite_task(hdf5_path)
        task_video_dir = video_root / f"{suite}_task{task_id}"
        with h5py.File(hdf5_path, "r") as f:
            for demo_id in sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]) if x.startswith("demo_") else -1):
                if not demo_id.startswith("demo_"):
                    continue
                demo = f["data"][demo_id]
                lengths = {
                    "actions": int(demo["actions"].shape[0]),
                    "eef_pose": int(demo["obs/eef_pose"].shape[0]),
                    "gripper_pos": int(demo["obs/gripper_pos"].shape[0]),
                    "gripper_net_force": int(demo["obs/gripper_net_force"].shape[0]),
                    "gripper_marker_motion": int(demo["obs/gripper_marker_motion"].shape[0]),
                }
                for label, subdir, suffix in VIDEO_SUFFIXES:
                    mp4 = task_video_dir / subdir / f"{demo_id}_{suffix}.mp4"
                    if not mp4.exists():
                        missing.append(str(mp4))
                        lengths[label] = -1
                    else:
                        lengths[label] = _count_mp4_frames(mp4, decode=decode)
                checked += 1
                if len(set(lengths.values())) == 1:
                    summary["all_equal"] += 1
                else:
                    summary["mismatch"] += 1
                    if len(mismatches) < max_mismatches:
                        mismatches.append(f"{hdf5_path.name}:{demo_id} {lengths}")

    print(f"root: {root}")
    print(f"checked demos: {checked}")
    print(f"summary: {dict(summary)}")
    print(f"missing count: {len(missing)}")
    if missing[:max_mismatches]:
        print("missing examples:")
        for item in missing[:max_mismatches]:
            print(f"  {item}")
    if mismatches:
        print("mismatch examples:")
        for item in mismatches:
            print(f"  {item}")
    return 1 if missing or summary.get("mismatch", 0) else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Dataset root containing replayed_demos/ and video_datasets/.")
    parser.add_argument("--decode", action="store_true", help="Decode videos to count frames instead of using MP4 metadata.")
    args = parser.parse_args()
    raise SystemExit(verify(args.root.expanduser().resolve(), decode=args.decode))


if __name__ == "__main__":
    main()
