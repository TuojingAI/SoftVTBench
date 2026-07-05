#!/usr/bin/env python3
"""Build a portable canonical JSONL manifest from a downloaded suite.

The output contains only paths relative to the suite root. It never carries the
original collection machine's ``source_hdf5`` or other absolute paths.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import h5py


SOFT_SUITES = {"object-soft": "libero_object", "spatial-soft": "libero_spatial"}
RIGID_SUITES = {"object-rigid": "libero_object", "spatial-rigid": "libero_spatial"}


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value) if value is not None else ""


def _task_id(path: Path, task_suite: str) -> int:
    match = re.search(rf"{re.escape(task_suite)}_task(\d+)", str(path))
    if not match:
        raise ValueError(f"cannot infer task id from {path}")
    return int(match.group(1))


def _hdf5_files(root: Path, suite: str, task_suite: str) -> list[Path]:
    if suite in SOFT_SUITES:
        return sorted(root.glob(f"{task_suite}/{task_suite}_task*/replayed_demos/*.hdf5"))
    return sorted((root / "replayed_demos").glob("*.hdf5"))


def _video_root(root: Path, suite: str, task_suite: str, task_id: int) -> Path:
    task_name = f"{task_suite}_task{task_id}"
    if suite in SOFT_SUITES:
        return root / task_suite / task_name / "video_datasets" / task_name
    return root / "video_datasets" / task_name


def build(root: Path, suite: str) -> list[dict[str, Any]]:
    task_suite = (SOFT_SUITES | RIGID_SUITES)[suite]
    hdf5_files = _hdf5_files(root, suite, task_suite)
    if len(hdf5_files) != 10:
        raise ValueError(f"expected 10 task HDF5 files, found {len(hdf5_files)}")

    records: list[dict[str, Any]] = []
    for hdf5_path in hdf5_files:
        task_id = _task_id(hdf5_path, task_suite)
        video_root = _video_root(root, suite, task_suite, task_id)
        with h5py.File(hdf5_path, "r") as file:
            if "data" not in file:
                raise ValueError(f"missing /data: {hdf5_path}")
            demo_keys = sorted(
                file["data"],
                key=lambda name: int(name.rsplit("_", 1)[-1]) if name.rsplit("_", 1)[-1].isdigit() else name,
            )
            for demo_key in demo_keys:
                demo = file["data"][demo_key]
                videos = sorted(video_root.glob(f"**/{demo_key}_*.mp4"))
                actions = demo.get("actions")
                record = {
                    "schema_version": 1,
                    "suite": suite,
                    "task_suite": task_suite,
                    "task_id": task_id,
                    "episode": demo_key,
                    "demo_id": int(demo.attrs["demo_id"]) if "demo_id" in demo.attrs else None,
                    "hdf5_path": hdf5_path.relative_to(root).as_posix(),
                    "num_samples": int(actions.shape[0]) if actions is not None else None,
                    "language": _text(demo.attrs.get("language", "")),
                    "asset_name": _text(demo.attrs.get("asset_name", "")),
                    "success": bool(demo.attrs["success"]) if "success" in demo.attrs else None,
                    "video_files": [path.relative_to(root).as_posix() for path in videos],
                }
                records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="downloaded suite root")
    parser.add_argument("--suite", choices=sorted(SOFT_SUITES | RIGID_SUITES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.output}; pass --overwrite")
    records = build(root, args.suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {args.output}: {len(records)} episodes")


if __name__ == "__main__":
    main()
