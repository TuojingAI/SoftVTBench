import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


def find_hdf5_file(assembled_dir: str, task_suite: str, task_id: int) -> Optional[str]:
    pattern = os.path.join(assembled_dir, f"{task_suite}_task{task_id}_*_demo.hdf5")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def list_demo_groups(h5f: h5py.File) -> List[h5py.Group]:
    # Expected structure: /data/demo_0, demo_1, ...
    if "data" in h5f:
        eps_grp = h5f["data"]
        names = sorted(
            [k for k in eps_grp.keys() if k.startswith("demo_")],
            key=lambda s: int("".join([c for c in s if c.isdigit()] or "0")),
        )
        return [eps_grp[name] for name in names]
    else:
        raise ValueError(f"HDF5 file missing 'data' group: {h5f.filename}")


def get_state_group(ep: h5py.Group, asset_type: str, obj_name: str) -> Optional[h5py.Group]:
    # Expected path: initial_state/{asset_type}/{obj_name}/root_pose
    if "initial_state" not in ep:
        return None
    st = ep["initial_state"]
    if asset_type not in st:
        return None
    at = st[asset_type]
    return at.get(obj_name)


def quat_to_rpy_wxyz(quat: np.ndarray) -> Tuple[float, float, float]:
    # Quaternion order: [w, x, y, z]
    w, x, y, z = quat
    # roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    # yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def unwrap_angles_relative(angles: np.ndarray, center: float) -> np.ndarray:
    # Map angles to a continuous domain centered at 'center', avoiding [-pi, pi] discontinuity
    wrapped = (angles - center + math.pi) % (2 * math.pi) - math.pi
    return wrapped + center


def circular_interval(
    angles: np.ndarray, use_percentile: bool = True, p_low: float = 1.0, p_high: float = 99.0
) -> Tuple[float, float]:
    # Compute circular statistics-based interval [min, max] without crossing discontinuity, in range [-pi, pi]
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    center = math.atan2(sin_mean, cos_mean)
    unwrapped = unwrap_angles_relative(angles, center)
    if use_percentile:
        lo = np.percentile(unwrapped, p_low)
        hi = np.percentile(unwrapped, p_high)
    else:
        lo = np.min(unwrapped)
        hi = np.max(unwrapped)

    # Normalize back to [-pi, pi]
    def norm(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    lo_n = norm(lo)
    hi_n = norm(hi)
    # Ensure lo <= hi (within same unwrapped domain)
    if lo_n > hi_n:
        # If reversed, likely near boundary; swap to align
        lo_n, hi_n = hi_n, lo_n
    return lo_n, hi_n


def collect_initial_pose_and_joint_pos_for_object(ep_groups: List[h5py.Group], obj_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collects initial pose and joint positions for an Rigid Object/Articulation from HDF5 files.
    Args:
        ep_groups: List of h5py.Group objects representing episodes, each containing initial state of the object
        obj_name: Name of the object to process
    Returns:
        pos_arr: (N, 3) array of x, y, z coordinates
        rpy_arr: (N, 3) array of roll, pitch, yaw angles
        joint_pos_arr: (N, J) array of joint positions
    """
    xs, ys, zs = [], [], []
    r_list, p_list, y_list = [], [], []
    joint_pos_list = []
    for ep in ep_groups:
        grp = get_state_group(ep, "rigid_object", obj_name)
        if grp is None:
            grp = get_state_group(ep, "articulation", obj_name)
            if grp is not None and "joint_position" in grp:
                joint_pos = np.array(grp["joint_position"])[0, :]
                joint_pos_list.append(joint_pos)
            else:
               raise ValueError(f"joint_position not found in Articulation {obj_name} for {ep.name}")
        if grp is not None and "root_pose" in grp:
            pose = np.array(grp["root_pose"])
            # Expected shape: (T, 7) or (7,)
            if pose.ndim == 2 and pose.shape[0] >= 1:
                p0 = pose[0, :3]
                q0 = pose[0, 3:]
            elif pose.ndim == 1 and pose.size == 7:
                p0 = pose[:3]
                q0 = pose[3:]
            else:
                continue
            xs.append(p0[0])
            ys.append(p0[1])
            zs.append(p0[2])
            r, p, yw = quat_to_rpy_wxyz(q0)
            r_list.append(r)
            p_list.append(p)
            y_list.append(yw)
        else:
            raise ValueError(f"root_pose not found in {obj_name} for {ep.name}")
    if not xs or not joint_pos_list:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 0))

    pos_arr = np.stack([xs, ys, zs], axis=1)
    rpy_arr = np.stack([r_list, p_list, y_list], axis=1)
    joint_pos_arr = np.array(joint_pos_list)
    return pos_arr, rpy_arr, joint_pos_arr


def decide_z_range(obs_z: np.ndarray, json_z_range: Tuple[float, float]) -> (Tuple[float, float], bool):
    # Returns (z_low, z_high), whether to replace
    if obs_z.size == 0:
        return json_z_range, False
    z_std = float(np.std(obs_z))
    z_mean = float(np.mean(obs_z))
    z_lo_obs = float(np.percentile(obs_z, 1.0))
    z_hi_obs = float(np.percentile(obs_z, 99.0))
    z_lo_json, z_hi_json = json_z_range
    # If JSON z is constant and close to data mean with low variance -> keep original
    if abs(z_lo_json - z_hi_json) < 1e-6 and z_std < 0.005 and abs(z_mean - z_lo_json) < 0.01:
        return (z_lo_json, z_hi_json), False
    # Otherwise, use observed distribution (with light outlier removal)
    return (z_lo_obs, z_hi_obs), True


def process_config_file(
    libero_json_path: str,
    assembled_dir: str,
    pad_xy: float,
    percentile: float,
    dry_run: bool,
) -> bool:
    with open(libero_json_path) as f:
        cfg = json.load(f)
    # deep copy for backup
    orig_cfg = json.loads(json.dumps(cfg))

    tasks = cfg.get("tasks", [])
    if not isinstance(tasks, list):
        print(f"[WARN] Invalid 'tasks' in {libero_json_path}, skip")
        return False

    changed = False
    p_low, p_high = percentile, 100.0 - percentile

    # 1) add tactile_targets (copy from obj_of_interest)
    for task in tasks:
        obj_of_interest = task.get("obj_of_interest", [])
        tactile_targets = task.get("tactile_targets", [])
        if tactile_targets == []:
            task["tactile_targets"] = list(obj_of_interest)
            changed = True

    # 2) update regions.pose_range
    for task in tasks:
        task_suite = libero_json_path.split("/")[-1].split(".")[0]
        task_id = int(task.get("task_id", 0))
        objects: Dict = task.get("objects", {})
        regions: Dict = task.get("regions", {})

        h5_path = find_hdf5_file(assembled_dir, task_suite, task_id)
        if not h5_path or not os.path.exists(h5_path):
            print(f"[WARN] HDF5 not found: {task_suite}_task{task_id} -> skip in {os.path.basename(libero_json_path)}")
            continue

        print(
            f"[INFO] ({os.path.basename(libero_json_path)}) Processing {task_suite}_task{task_id}:"
            f" {os.path.basename(h5_path)}"
        )
        with h5py.File(h5_path, "r") as h5f:
            ep_groups = list_demo_groups(h5f)
            if not ep_groups:
                print(f"[WARN] HDF5 has no episodes: {h5_path} -> skip")
                continue

            # For each object's initial_region, compute pose distribution and update corresponding region
            for obj_name, obj_desc in objects.items():
                region_name = obj_desc.get("initial_region")
                if not region_name or region_name not in regions:
                    continue

                pos_arr, rpy_arr, joint_pos_arr = collect_initial_pose_and_joint_pos_for_object(ep_groups, obj_name)
                if pos_arr.shape[0] == 0:
                    print(f"[WARN] {obj_name}: initial pose not found in HDF5, skip")
                    continue

                # XY: percentile range + padding
                x_lo = float(np.percentile(pos_arr[:, 0], p_low)) - pad_xy
                x_hi = float(np.percentile(pos_arr[:, 0], p_high)) + pad_xy
                y_lo = float(np.percentile(pos_arr[:, 1], p_low)) - pad_xy
                y_hi = float(np.percentile(pos_arr[:, 1], p_high)) + pad_xy

                # roll/pitch/yaw: circular statistics interval
                roll_lo, roll_hi = circular_interval(rpy_arr[:, 0], use_percentile=True, p_low=p_low, p_high=p_high)
                pitch_lo, pitch_hi = circular_interval(rpy_arr[:, 1], use_percentile=True, p_low=p_low, p_high=p_high)
                yaw_lo, yaw_hi = circular_interval(rpy_arr[:, 2], use_percentile=True, p_low=p_low, p_high=p_high)

                # Z: auto-decide whether to replace
                json_pose_range = regions[region_name].get("pose_range", {})
                json_z_range = tuple(json_pose_range.get("z", [float(pos_arr[0, 2]), float(pos_arr[0, 2])]))
                (z_lo, z_hi), z_replaced = decide_z_range(pos_arr[:, 2], json_z_range)

                # for articulated objects, update joint_pos_range
                if joint_pos_arr.shape[0] > 0:
                    new_joint_pos_range = {}
                    for i in range(joint_pos_arr.shape[1]):
                        joint_pos_lo, joint_pos_hi = circular_interval(joint_pos_arr[:, i], use_percentile=True, p_low=p_low, p_high=p_high)
                        new_joint_pos_range[i] = [joint_pos_lo, joint_pos_hi]
                else:
                    new_joint_pos_range = {}

                # Apply update
                new_pose_range = {
                    "x": [x_lo, x_hi],
                    "y": [y_lo, y_hi],
                    "z": [z_lo, z_hi],
                    "roll": [roll_lo, roll_hi],
                    "pitch": [pitch_lo, pitch_hi],
                    "yaw": [yaw_lo, yaw_hi],
                }
                if new_joint_pos_range:
                    new_pose_range["joint_pos_range"] = new_joint_pos_range

                regions[region_name]["pose_range"] = new_pose_range
                print(
                    f"  - {obj_name} -> {region_name}: "
                    f"x[{x_lo:.3f},{x_hi:.3f}] y[{y_lo:.3f},{y_hi:.3f}] "
                    f"z[{z_lo:.3f},{z_hi:.3f}]{'' if z_replaced else ' (keep)'} "
                    f"roll[{roll_lo:.3f},{roll_hi:.3f}] pitch[{pitch_lo:.3f},{pitch_hi:.3f}] "
                    f"yaw[{yaw_lo:.3f},{yaw_hi:.3f}]"
                )
                if new_joint_pos_range:
                    print(f"  - {obj_name} -> {region_name}: joint_pos[{', '.join([f'{k}:[{v[0]:.3f},{v[1]:.3f}]' for k, v in new_joint_pos_range.items()])}]")
                changed = True

    if changed and not dry_run:
        # backup to config_bk/ file with the same name
        cfg_dir = os.path.dirname(libero_json_path)
        bk_dir = os.path.join(cfg_dir, "config_bk")
        os.makedirs(bk_dir, exist_ok=True)
        bk_path = os.path.join(bk_dir, os.path.basename(libero_json_path))
        with open(bk_path, "w") as f:
            json.dump(orig_cfg, f, indent=2, ensure_ascii=False)

        # write back updated config
        with open(libero_json_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"[OK] Updated and written: {libero_json_path} (backup in: {bk_path})")
    elif changed:
        print(f"[DRY RUN] Changes detected but not written for: {libero_json_path}")
    else:
        print(f"[INFO] No updates detected for: {libero_json_path}")

    return changed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Update regions.pose_range (x,y,z,roll,pitch,yaw) in libero*.json using initial pose statistics from"
            " assembled_hdf5, and add tactile_targets from obj_of_interest."
        )
    )
    parser.add_argument(
        "--libero_json",
        type=str,
        default="./benchmarks/datasets/libero/config/libero.json",
        help="Path to a single libero.json (kept for backward compatibility).",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="./benchmarks/datasets/libero/config",
        help="Directory containing libero_*.json files, e.g. ./benchmarks/datasets/libero/config",
    )
    parser.add_argument(
        "--assembled_dir",
        type=str,
        default="./benchmarks/datasets/libero/assembled_hdf5",
        help="Path to assembled_hdf5 folder, e.g. ./benchmarks/datasets/libero/assembled_hdf5",
    )
    parser.add_argument("--pad_xy", type=float, default=0.005, help="XY range padding margin (meters) to encourage more diverse poses")
    parser.add_argument("--percentile", type=float, default=1.0, help="Percentile for min/max (symmetric: p and 100-p) to avoid outliers")
    parser.add_argument("--dry_run", action="store_true", help="Print changes only, do not write back")
    parser.add_argument(
        "--update_variants",
        action="store_true",
        help=(
            "Update all libero_*.json in --config_dir (libero_10.json, libero_90.json, libero_goal.json,"
            " libero_object.json, libero_spatial.json)."
        ),
    )
    args = parser.parse_args()

    if args.update_variants:
        cfg_dir = args.config_dir if args.config_dir else os.path.dirname(args.libero_json)
        pattern = os.path.join(cfg_dir, "libero_*.json")
        target_files = sorted(glob.glob(pattern))
        if not target_files:
            print(f"[ERROR] No files matched: {pattern}")
            return

        print(f"[INFO] Batch updating {len(target_files)} files in {cfg_dir}")
        total_changed = 0
        for path in target_files:
            # Only pick the 5 variants if user expects those explicitly
            base = os.path.basename(path)
            if base in {
                "libero_10.json",
                "libero_90.json",
                "libero_goal.json",
                "libero_object.json",
                "libero_spatial.json",
            }:
                if base == "libero_spatial.json":  # spatial config cannot enable randomization, since some objects are closely connected
                    pad_xy = 0.0
                else:
                    pad_xy = args.pad_xy
                changed = process_config_file(path, args.assembled_dir, pad_xy, args.percentile, args.dry_run)
                total_changed += int(changed)
        print(f"[INFO] Done. Files changed (detected): {total_changed}")
    else:
        if args.libero_json.endswith("libero_spatial.json"):  # spatial config cannot enable randomization, since some objects are closely connected
            pad_xy = 0.0
        else:
            pad_xy = args.pad_xy
        process_config_file(args.libero_json, args.assembled_dir, pad_xy, args.percentile, args.dry_run)


if __name__ == "__main__":
    main()
