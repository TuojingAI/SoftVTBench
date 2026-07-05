#!/usr/bin/env python3
"""Convert bench deep-press sweeps into SoftVTBench compression-sweep JSONL.

Reads the per-asset gripper-sweep CSVs produced by the calibration bench and
the bench yield report, and emits one JSONL record per valid compression trial
in the schema expected by calibrate_safety_thresholds.py:

    {"asset": "soft_<name>", "closure": <command_close_norm>,
     "stable": <bool>, "d_peak": <percent of reference bbox diagonal>,
     "level": <int>, "gripper_width_mm": <float>}

d_peak uses the sweep's rigid-aligned FEM-RMS over bbox-diagonal ratio
(`d_fem_ratio`), i.e. the same fem_rms_rigid_aligned_bbox_pct_v1 metric as
closed-loop evaluation. A trial is stable when the bench marked it valid (bilateral contact) and its
gripper width is strictly looser than the bench yield width; the yield row
itself is the yield event and counts as unstable.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sweep-dir", type=Path, required=True,
                    help="directory with <asset>_gripper_sweep.csv files")
parser.add_argument("--yield-json", type=Path, required=True,
                    help="bench yield report (yield.json)")
parser.add_argument("--assets", nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

yield_width_mm = {r["asset"]: float(r["width_yield_mm"])
                  for r in json.loads(args.yield_json.read_text())
                  if r.get("width_yield_mm") is not None}

rows_out = []
for asset in args.assets:
    csv_path = args.sweep_dir / f"{asset}_gripper_sweep.csv"
    y_width = yield_width_mm.get(asset)
    n_stable = 0
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # the bench's own validity flag: bilateral contact, no invalid_reason
            if row.get("valid") not in ("1", "1.0"):
                continue
            level = int(float(row["level"]))
            width_mm = float(row["gripper_width"]) * 1000.0
            # stable = strictly looser than the bench yield width; the yield
            # row itself is the yield event and counts as unstable
            stable = (y_width is None) or (width_mm > y_width)
            d_peak = float(row["d_fem_ratio"]) * 100.0
            if d_peak < 0:
                continue
            rows_out.append({
                "asset": f"soft_{asset}",
                "closure": round(float(row["command_close_norm"]), 6),
                "stable": stable,
                "d_peak": round(d_peak, 6),
                "level": level,
                "gripper_width_mm": round(float(row["gripper_width"]) * 1000.0, 3),
            })
            n_stable += int(stable)
    print(f"{asset}: yield_width={y_width} trials={sum(1 for r in rows_out if r['asset'] == 'soft_' + asset)} stable={n_stable}")

args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8") as f:
    for r in rows_out:
        f.write(json.dumps(r, sort_keys=True) + "\n")
print(f"wrote {args.output} ({len(rows_out)} records)")
