#!/usr/bin/env python3
"""Trajectory smoothness QA for spatial soft pastry HDF5 data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class DemoSmoothness:
    task: str
    demo: str
    hdf5: str
    frames: int
    duration_steps: int
    pos_step_p95: float
    pos_step_max: float
    pos_acc_p95: float
    pos_acc_max: float
    pos_jerk_p95: float
    pos_jerk_max: float
    rot_step_p95: float
    rot_step_max: float
    rot_acc_p95: float
    rot_acc_max: float
    gripper_step_p95: float
    gripper_step_max: float
    gripper_acc_p95: float
    gripper_acc_max: float
    stationary_frac: float
    max_stationary_run: int
    jump_count: int
    gripper_jump_count: int
    score: float
    flags: str


def natural_key(text: str):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", text)]


def quat_wxyz_to_rotvec(q: np.ndarray) -> np.ndarray:
    # Local implementation to avoid scipy dependency.
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    # Choose the short branch for continuity with the stored action convention.
    q = np.where(q[..., :1] < 0, -q, q)
    w = np.clip(q[..., 0], -1.0, 1.0)
    xyz = q[..., 1:4]
    s = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(s, w)
    scale = np.divide(angle, s, out=np.zeros_like(angle), where=s > 1e-12)
    return xyz * scale[..., None]


def continuous_rotvec_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    rv = quat_wxyz_to_rotvec(q)
    if len(rv) <= 1:
        return rv
    out = rv.copy()
    for i in range(1, len(out)):
        cur = out[i]
        n = np.linalg.norm(cur)
        candidates = [cur]
        if n > 1e-12:
            axis = cur / n
            candidates.extend([cur + 2.0 * np.pi * axis, cur - 2.0 * np.pi * axis, -(2.0 * np.pi - n) * axis])
        out[i] = min(candidates, key=lambda x: np.linalg.norm(x - out[i - 1]))
    return out


def percentile(values: np.ndarray, p: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.nanpercentile(values, p))


def max_value(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.nanmax(values))


def longest_true_run(mask: np.ndarray) -> int:
    best = cur = 0
    for x in mask.astype(bool):
        if x:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def diff_norm(x: np.ndarray, order: int = 1) -> np.ndarray:
    if len(x) <= order:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(np.diff(x, n=order, axis=0), axis=-1)


def analyze_demo(path: Path, demo: str, args) -> DemoSmoothness:
    with h5py.File(path, "r") as f:
        g = f["data"][demo]
        eef = np.asarray(g["obs/eef_pose"], dtype=np.float64)
        pos = eef[:, :3]
        # obs/eef_pose is [xyz, qw, qx, qy, qz] in this dataset.
        rot = continuous_rotvec_from_quat_wxyz(eef[:, 3:7])
        if "obs/gripper_width" in g:
            gripper = np.asarray(g["obs/gripper_width"], dtype=np.float64).reshape(-1)
        elif "obs/gripper_pos" in g:
            gp = np.asarray(g["obs/gripper_pos"], dtype=np.float64)
            gripper = np.abs(gp).sum(axis=-1)
        else:
            gripper = eef[:, 6]

    pos_step = diff_norm(pos, 1)
    pos_acc = diff_norm(pos, 2)
    pos_jerk = diff_norm(pos, 3)
    rot_step = diff_norm(rot, 1)
    rot_acc = diff_norm(rot, 2)
    grip_step = np.abs(np.diff(gripper, n=1))
    grip_acc = np.abs(np.diff(gripper, n=2)) if len(gripper) > 2 else np.zeros((0,))

    stationary = (pos_step < args.stationary_pos_step) & (rot_step < args.stationary_rot_step)
    jump_mask = (pos_step > args.pos_jump_threshold) | (rot_step > args.rot_jump_threshold)
    grip_jump_mask = grip_step > args.gripper_jump_threshold

    pos_step_p95 = percentile(pos_step, 95)
    pos_step_max = max_value(pos_step)
    pos_acc_p95 = percentile(pos_acc, 95)
    pos_acc_max = max_value(pos_acc)
    pos_jerk_p95 = percentile(pos_jerk, 95)
    pos_jerk_max = max_value(pos_jerk)
    rot_step_p95 = percentile(rot_step, 95)
    rot_step_max = max_value(rot_step)
    rot_acc_p95 = percentile(rot_acc, 95)
    rot_acc_max = max_value(rot_acc)
    grip_step_p95 = percentile(grip_step, 95)
    grip_step_max = max_value(grip_step)
    grip_acc_p95 = percentile(grip_acc, 95)
    grip_acc_max = max_value(grip_acc)
    stationary_frac = float(np.mean(stationary)) if stationary.size else 0.0
    max_stationary_run = longest_true_run(stationary)
    jump_count = int(np.sum(jump_mask))
    gripper_jump_count = int(np.sum(grip_jump_mask))

    flags: list[str] = []
    if jump_count:
        flags.append("eef_jump")
    if gripper_jump_count:
        flags.append("gripper_jump")
    if max_stationary_run >= args.stationary_run_threshold:
        flags.append("long_stationary")
    if pos_acc_max > args.pos_acc_threshold:
        flags.append("pos_acc_spike")
    if grip_acc_max > args.gripper_acc_threshold:
        flags.append("gripper_acc_spike")

    score = (
        1000.0 * pos_step_max
        + 1500.0 * pos_acc_max
        + 100.0 * rot_step_max
        + 200.0 * rot_acc_max
        + 80.0 * grip_step_max
        + 100.0 * grip_acc_max
        + 0.5 * max_stationary_run
        + 5.0 * jump_count
        + 5.0 * gripper_jump_count
    )

    m = re.search(r"libero_spatial_task(\d+)", str(path))
    task = f"task{m.group(1)}" if m else "unknown"
    return DemoSmoothness(
        task=task,
        demo=demo,
        hdf5=str(path),
        frames=len(pos),
        duration_steps=max(len(pos) - 1, 0),
        pos_step_p95=pos_step_p95,
        pos_step_max=pos_step_max,
        pos_acc_p95=pos_acc_p95,
        pos_acc_max=pos_acc_max,
        pos_jerk_p95=pos_jerk_p95,
        pos_jerk_max=pos_jerk_max,
        rot_step_p95=rot_step_p95,
        rot_step_max=rot_step_max,
        rot_acc_p95=rot_acc_p95,
        rot_acc_max=rot_acc_max,
        gripper_step_p95=grip_step_p95,
        gripper_step_max=grip_step_max,
        gripper_acc_p95=grip_acc_p95,
        gripper_acc_max=grip_acc_max,
        stationary_frac=stationary_frac,
        max_stationary_run=max_stationary_run,
        jump_count=jump_count,
        gripper_jump_count=gripper_jump_count,
        score=float(score),
        flags=";".join(flags),
    )


def write_plots(rows: list[DemoSmoothness], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = sorted({r.task for r in rows}, key=natural_key)
    metrics = [
        ("pos_step_max", "EEF step max"),
        ("pos_acc_max", "EEF acceleration spike"),
        ("gripper_step_max", "Gripper step max"),
        ("max_stationary_run", "Longest stationary run"),
        ("score", "Smoothness risk score"),
    ]
    for field, title in metrics:
        data = [[getattr(r, field) for r in rows if r.task == t] for t in tasks]
        plt.figure(figsize=(10, 4))
        plt.boxplot(data, labels=tasks, showfliers=True)
        plt.title(title)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(output_dir / f"{field}.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pos-jump-threshold", type=float, default=0.025, help="m/frame")
    parser.add_argument("--rot-jump-threshold", type=float, default=0.25, help="rad/frame")
    parser.add_argument("--gripper-jump-threshold", type=float, default=0.012, help="m/frame")
    parser.add_argument("--pos-acc-threshold", type=float, default=0.020, help="m/frame^2")
    parser.add_argument("--gripper-acc-threshold", type=float, default=0.010, help="m/frame^2")
    parser.add_argument("--stationary-pos-step", type=float, default=0.0005, help="m/frame")
    parser.add_argument("--stationary-rot-step", type=float, default=0.005, help="rad/frame")
    parser.add_argument("--stationary-run-threshold", type=int, default=10)
    args = parser.parse_args()

    paths = sorted(args.root.glob("libero_spatial/libero_spatial_task*/replayed_demos/*.hdf5"))
    rows: list[DemoSmoothness] = []
    for path in paths:
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys(), key=natural_key)
        for demo in demos:
            rows.append(analyze_demo(path, demo, args))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "trajectory_smoothness_demos.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    by_task = {}
    for task in sorted({r.task for r in rows}, key=natural_key):
        rs = [r for r in rows if r.task == task]
        by_task[task] = {
            "demos": len(rs),
            "flagged": sum(bool(r.flags) for r in rs),
            "pos_step_max": max(r.pos_step_max for r in rs),
            "pos_acc_max": max(r.pos_acc_max for r in rs),
            "rot_step_max": max(r.rot_step_max for r in rs),
            "gripper_step_max": max(r.gripper_step_max for r in rs),
            "gripper_acc_max": max(r.gripper_acc_max for r in rs),
            "max_stationary_run": max(r.max_stationary_run for r in rs),
            "score_p95": float(np.percentile([r.score for r in rs], 95)),
            "worst": max(rs, key=lambda r: r.score).demo,
        }
    summary = {
        "root": str(args.root),
        "demos": len(rows),
        "flagged": sum(bool(r.flags) for r in rows),
        "thresholds": {
            "pos_jump_threshold": args.pos_jump_threshold,
            "rot_jump_threshold": args.rot_jump_threshold,
            "gripper_jump_threshold": args.gripper_jump_threshold,
            "pos_acc_threshold": args.pos_acc_threshold,
            "gripper_acc_threshold": args.gripper_acc_threshold,
            "stationary_run_threshold": args.stationary_run_threshold,
        },
        "by_task": by_task,
        "top20_worst": [asdict(r) for r in sorted(rows, key=lambda r: r.score, reverse=True)[:20]],
    }
    (args.output_dir / "trajectory_smoothness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_plots(rows, args.output_dir / "plots")

    print(json.dumps({k: summary[k] for k in ("demos", "flagged", "by_task")}, ensure_ascii=False, indent=2))
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
