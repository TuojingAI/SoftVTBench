#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


THRESHOLDS = {
    "soft_pastry001": (4.56, 9.12),
    "soft_pastry002": (1.21, 2.41),
    "soft_pastry003": (1.09, 2.17),
    "soft_pastry004": (0.80, 1.60),
    "soft_pastry005": (4.52, 9.04),
    "soft_pastry006": (8.19, 16.37),
    "soft_pastry010": (0.18, 0.35),
}


def read_episode(path: Path) -> dict | None:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        return None
    last = rows[-1]
    task = int(last.get("task_id", rows[0].get("task_id", -1)))
    exp_idx = int(last.get("exp_idx", rows[0].get("exp_idx", 0)))
    asset = str(last.get("fem_deformation_asset") or last.get("goal_ref_obj") or rows[0].get("fem_deformation_asset") or "")
    if not asset:
        for row in rows:
            asset = str(row.get("fem_deformation_asset") or row.get("goal_ref_obj") or "")
            if asset:
                break
    fem_vals = []
    actions = defaultdict(int)
    max_global = 0.0
    min_grip = None
    max_grip = None
    contact_seen = False
    for row in rows:
        value = row.get("fem_deformation_rms")
        if value is not None:
            try:
                fem_vals.append(float(value))
            except Exception:
                pass
        action = row.get("tactile_ctrl_action")
        if action:
            actions[str(action)] += 1
        try:
            max_global = max(max_global, float(row.get("tactile_ctrl_global_score") or 0.0))
        except Exception:
            pass
        grip = row.get("controller_gripper_finger", row.get("gripper_cmd"))
        try:
            grip = float(grip)
            min_grip = grip if min_grip is None else min(min_grip, grip)
            max_grip = grip if max_grip is None else max(max_grip, grip)
        except Exception:
            pass
        contact_seen = contact_seen or bool(row.get("tactile_ctrl_contact_seen"))
    fem_peak = max(fem_vals) if fem_vals else None
    success = any(bool(row.get("success_now")) for row in rows) or any(int(row.get("success_step_count") or 0) >= 8 for row in rows)
    strict_thr, unsafe_thr = THRESHOLDS.get(asset, (None, None))
    known = strict_thr is not None and fem_peak is not None
    strict_safe = bool(known and fem_peak <= strict_thr)
    not_unsafe = bool(known and fem_peak <= unsafe_thr)
    return {
        "task": task,
        "episode": path.parent.name,
        "asset": asset,
        "success": int(success),
        "fem_peak": fem_peak if fem_peak is not None else "",
        "threshold_known": bool(known),
        "safe_thr": strict_thr if strict_thr is not None else "",
        "unsafe_thr": unsafe_thr if unsafe_thr is not None else "",
        "strict_safe": int(strict_safe),
        "not_unsafe": int(not_unsafe),
        "strict_safe_success": int(success and strict_safe),
        "not_unsafe_success": int(success and not_unsafe),
        "contact_seen": int(contact_seen),
        "max_global_score": max_global,
        "min_gripper": min_grip if min_grip is not None else "",
        "max_gripper": max_grip if max_grip is not None else "",
        "actions": dict(actions),
        "path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.root / "summary_object_soft_forces")
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    for path in sorted(args.root.rglob("forces.jsonl")):
        item = read_episode(path)
        if item is not None:
            episodes.append(item)

    known = [e for e in episodes if e["threshold_known"]]
    overall = {
        "episodes": len(episodes),
        "goal_success": sum(e["success"] for e in episodes),
        "goal_total": len(episodes),
        "threshold_known_episodes": len(known),
        "strict_safe_success_known": sum(e["strict_safe_success"] for e in known),
        "not_unsafe_success_known": sum(e["not_unsafe_success"] for e in known),
        "known_goal_success": sum(e["success"] for e in known),
        "fem_peak_mean_all": sum(float(e["fem_peak"]) for e in episodes if e["fem_peak"] != "") / max(1, sum(1 for e in episodes if e["fem_peak"] != "")),
        "fem_peak_max_all": max([float(e["fem_peak"]) for e in episodes if e["fem_peak"] != ""] or [0.0]),
    }
    (out_dir / "overall.json").write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")

    fields = [
        "task", "episode", "asset", "success", "fem_peak", "threshold_known", "safe_thr", "unsafe_thr",
        "strict_safe", "not_unsafe", "strict_safe_success", "not_unsafe_success", "contact_seen",
        "max_global_score", "min_gripper", "max_gripper", "actions", "path",
    ]
    with (out_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in episodes:
            row = dict(e)
            row["actions"] = json.dumps(row["actions"], sort_keys=True)
            writer.writerow(row)

    by_task = defaultdict(list)
    for e in episodes:
        by_task[e["task"]].append(e)
    with (out_dir / "summary_by_task.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "total", "goal_success", "goal_rate", "threshold_known", "strict_safe_success", "not_unsafe_success", "fem_peak_mean", "fem_peak_max"])
        for task in sorted(by_task):
            rows = by_task[task]
            krows = [e for e in rows if e["threshold_known"]]
            fem = [float(e["fem_peak"]) for e in rows if e["fem_peak"] != ""]
            writer.writerow([
                task,
                len(rows),
                sum(e["success"] for e in rows),
                sum(e["success"] for e in rows) / max(1, len(rows)),
                len(krows),
                sum(e["strict_safe_success"] for e in krows),
                sum(e["not_unsafe_success"] for e in krows),
                sum(fem) / max(1, len(fem)),
                max(fem) if fem else "",
            ])
    print(json.dumps(overall, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
