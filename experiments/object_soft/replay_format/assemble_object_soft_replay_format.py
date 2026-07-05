#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common.deformation import METRIC_ID, deformation_series


STRIP_REPLAY_DATASET_PATHS = (
    "obs/eef_axis_angle",
    "soft_extras/eef_axis_angle",
)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:110] or "libero_object_soft"


def iter_source_hdf5(runs_root: Path):
    runs_dir = runs_root / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"missing runs directory: {runs_dir}")
    yield from sorted(runs_dir.glob("*/replayed_demos/*.hdf5"))


def read_demo_meta(hdf5_path: Path):
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data/demo_0"]
        task_suite = str(demo.attrs.get("task_suite", "libero_spatial"))
        task_id = int(demo.attrs.get("task_id"))
        demo_id = int(demo.attrs.get("demo_id", -1))
        language = str(demo.attrs.get("language", ""))
        success = str(demo.attrs.get("success", "True")).lower() in {"1", "true", "yes"}
        num_samples = int(demo.attrs.get("num_samples", demo.attrs.get("num_frames", demo["actions"].shape[0])))
    return {
        "task_suite": task_suite,
        "task_id": task_id,
        "demo_id": demo_id,
        "language": language,
        "success": success,
        "num_samples": num_samples,
    }


def next_demo_name(data_group):
    used = []
    for name in data_group.keys():
        if name.startswith("demo_"):
            try:
                used.append(int(name.split("_", 1)[1]))
            except ValueError:
                pass
    return f"demo_{max(used) + 1 if used else 0}"


def copy_demo(src_path: Path, dst_path: Path, source_meta: dict):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "a") as dst:
        data = dst.require_group("data")
        dst_demo_name = next_demo_name(data)
        src.copy(src["data/demo_0"], data, name=dst_demo_name)
        demo = data[dst_demo_name]
        for dataset_path in STRIP_REPLAY_DATASET_PATHS:
            if dataset_path in demo:
                del demo[dataset_path]
        # Portable provenance only: never embed the collection machine's
        # absolute path in a released HDF5 file.
        demo.attrs["source_hdf5"] = src_path.name
        demo.attrs["original_demo_id"] = int(source_meta["demo_id"])
        if "num_samples" not in demo.attrs and "actions" in demo:
            demo.attrs["num_samples"] = int(demo["actions"].shape[0])
        ensure_soft_summary_fields(demo)
    return dst_demo_name


def write_if_missing(group, name: str, data, attrs: dict | None = None):
    if data is None or name in group:
        return
    ds = group.create_dataset(name, data=np.asarray(data, dtype=np.float32), compression="gzip")
    if attrs:
        for key, value in attrs.items():
            ds.attrs[key] = value


def frame_count(demo):
    if "actions" in demo:
        return int(demo["actions"].shape[0])
    if "obs/actions" in demo:
        return int(demo["obs/actions"].shape[0])
    return int(demo.attrs.get("num_samples", demo.attrs.get("num_frames", 0)))


def gripper_width_from_demo(demo):
    if "obs/gripper_pos" not in demo:
        return None
    arr = np.asarray(demo["obs/gripper_pos"], dtype=np.float32)
    if arr.ndim < 2 or arr.shape[0] == 0:
        return None
    flat = arr.reshape(arr.shape[0], -1)
    if flat.shape[1] >= 2:
        return np.sum(np.abs(flat[:, :2]), axis=1).astype(np.float32)
    return np.abs(flat[:, 0]).astype(np.float32)


def close_norm_from_demo(demo):
    count = frame_count(demo)
    raw = None
    if "soft_extras/actions_raw" in demo:
        raw = np.asarray(demo["soft_extras/actions_raw"], dtype=np.float32)
    elif "obs/actions" in demo:
        raw = np.asarray(demo["obs/actions"], dtype=np.float32)
    if raw is not None and raw.ndim >= 2 and raw.shape[0] > 0:
        flat = raw.reshape(raw.shape[0], -1)
        if flat.shape[1] >= 1:
            return np.clip((flat[:, -1] + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
    meta = {}
    if "metadata_json" in demo.attrs:
        try:
            meta = json.loads(str(demo.attrs["metadata_json"]))
        except Exception:
            meta = {}
    value = float(meta.get("gripper_close_norm", np.nan))
    return np.full((count,), value, dtype=np.float32) if count else None


def deformable_group(demo):
    if "states/deformable_object" not in demo:
        return None
    group = demo["states/deformable_object"]
    asset_name = str(demo.attrs.get("asset_name", ""))
    if asset_name and asset_name in group:
        return group[asset_name]
    names = sorted(group.keys())
    return group[names[0]] if names else None


def fem_summary_from_demo(demo):
    group = deformable_group(demo)
    if group is None or "nodal_pos_w" not in group:
        return None, None
    pts = np.asarray(group["nodal_pos_w"], dtype=np.float32)
    if pts.ndim < 3 or pts.shape[0] == 0:
        return None, None
    return deformation_series(pts[0], pts)


def fem_bbox_dims_from_demo(demo):
    group = deformable_group(demo)
    if group is None or "bbox_min_w" not in group or "bbox_max_w" not in group:
        return None
    mn = np.asarray(group["bbox_min_w"], dtype=np.float32)
    mx = np.asarray(group["bbox_max_w"], dtype=np.float32)
    if mn.shape != mx.shape:
        return None
    return (mx - mn).reshape(mn.shape[0], -1).astype(np.float32)


def ensure_soft_summary_fields(demo):
    obs = demo.require_group("obs")
    soft = demo.require_group("soft_extras")
    summaries = (
        ("gripper_width", gripper_width_from_demo(demo), {"source": "sum_abs_obs_gripper_pos"}),
        ("gripper_close_norm", close_norm_from_demo(demo), {"source": "script_action_last_channel_or_metadata_json"}),
        ("fem_deformation_rms", fem_summary_from_demo(demo)[0], {"reference": "first_recorded_soft_nodal_pos_w", "metric_id": METRIC_ID, "unit": "percent"}),
        ("fem_deformation_max", fem_summary_from_demo(demo)[1], {"reference": "first_recorded_soft_nodal_pos_w", "metric_id": METRIC_ID, "unit": "percent"}),
        ("fem_bbox_dims", fem_bbox_dims_from_demo(demo), None),
    )
    for name, data, attrs in summaries:
        write_if_missing(obs, name, data, attrs)
        write_if_missing(soft, name, data, attrs)


def demo_index_from_name(demo_name: str) -> int:
    return int(demo_name.split("_", 1)[1])


def rename_video_for_aligned_demo(src_name: str, original_demo_id: int, aligned_demo_index: int) -> str:
    prefix = f"demo_{original_demo_id}_"
    if src_name.startswith(prefix):
        return f"demo_{aligned_demo_index}_{src_name[len(prefix):]}"
    return re.sub(r"demo_\\d+_", f"demo_{aligned_demo_index}_", src_name, count=1)


def copy_demo_videos(src_hdf5: Path, task_dir: Path, task_dataset_dir_name: str, demo_id: int, dst_demo_name: str):
    run_dir = src_hdf5.parent.parent
    src_base = run_dir / "video_datasets" / task_dataset_dir_name
    if not src_base.exists():
        return []

    copied = []
    aligned_demo_index = demo_index_from_name(dst_demo_name)
    for subdir in ("videos", "tactile_outputs"):
        src_dir = src_base / subdir
        if not src_dir.exists():
            continue
        dst_dir = task_dir / "video_datasets" / task_dataset_dir_name / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(src_dir.glob(f"demo_{demo_id}_*.mp4")):
            dst = dst_dir / rename_video_for_aligned_demo(src.name, demo_id, aligned_demo_index)
            if not dst.exists() or src.stat().st_size != dst.stat().st_size:
                shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True, type=Path, action="append")
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--suite-dir", default="libero_object")
    parser.add_argument("--task-dir-prefix", default="libero_object_task")
    parser.add_argument("--hdf5-file-prefix", default="libero_object_task")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-failures", action="store_true")
    args = parser.parse_args()

    if args.overwrite and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out_root / "manifest.jsonl"
    failure_path = args.out_root / "failure.jsonl"
    manifest_path.write_text("", encoding="utf-8")
    failure_path.write_text("", encoding="utf-8")

    counts = {}
    failure_count = 0
    duplicate_count = 0
    video_count = 0
    seen = set()
    duplicate_path = args.out_root / "duplicate.jsonl"
    duplicate_path.write_text("", encoding="utf-8")
    for runs_root in args.runs_root:
        for src_path in iter_source_hdf5(runs_root):
            meta = read_demo_meta(src_path)
            unique_key = (meta["task_id"], meta["demo_id"])
            if not meta["success"] and not args.include_failures:
                failure_count += 1
                with failure_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({**meta, "source_hdf5": src_path.name}, sort_keys=True) + "\n")
                continue
            if unique_key in seen:
                duplicate_count += 1
                with duplicate_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({**meta, "source_hdf5": src_path.name}, sort_keys=True) + "\n")
                continue
            seen.add(unique_key)

            task_id = meta["task_id"]
            task_dir = args.out_root / args.suite_dir / f"{args.task_dir_prefix}{task_id}"
            replayed_dir = task_dir / "replayed_demos"
            language_slug = slugify(meta["language"])
            dst_hdf5 = replayed_dir / f"{args.hdf5_file_prefix}{task_id}_{language_slug}_replayed_demo.hdf5"
            dst_demo = copy_demo(src_path, dst_hdf5, meta)
            task_dataset_dir_name = f"{args.task_dir_prefix}{task_id}"
            videos = copy_demo_videos(src_path, task_dir, task_dataset_dir_name, meta["demo_id"], dst_demo)

            counts[task_id] = counts.get(task_id, 0) + 1
            video_count += len(videos)
            row = {
                **meta,
                "schema_version": 1,
                "hdf5_path": dst_hdf5.relative_to(args.out_root).as_posix(),
                "episode": dst_demo,
                "aligned_hdf5": dst_hdf5.relative_to(args.out_root).as_posix(),
                "aligned_demo": dst_demo,
                "source_hdf5": src_path.name,
                "video_files": [Path(path).relative_to(args.out_root).as_posix() for path in videos],
            }
            with manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "runs_roots": [path.name for path in args.runs_root],
        "out_root": ".",
        "task_counts": {str(k): v for k, v in sorted(counts.items())},
        "total_success": sum(counts.values()),
        "total_videos": video_count,
        "skipped_failures": failure_count,
        "skipped_duplicates": duplicate_count,
    }
    (args.out_root / "assemble_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
