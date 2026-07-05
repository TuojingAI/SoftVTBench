#!/usr/bin/env python3
"""Build a non-protocol expert-reference threshold for diagnostics only.

SoftVTBench defines a rollout's deformation as

    D(t) = 100 * RMS_i( || (x_i(t) - x_bar(t)) - (x_i(0) - x_bar(0)) || ) / diag(bbox(0))

i.e. the object-size-normalised FEM-RMS nodal displacement after removing global
rigid-body motion, expressed as a percentage of the reference bounding-box
diagonal, and the rollout's peak deformation as D_peak = max_t D(t) (Eq. 3).

The simulator already stores D(t) per frame as `obs/fem_deformation_rms`, so this
script only has to reduce it. tau_o is taken as a high percentile of D_peak over
the *successful expert demonstrations* of that object: the expert trajectories
define the safe interaction envelope, so a policy that deforms an object more
than the expert essentially ever did is deemed unsafe.

This is not the SoftVTBench v1 compression-sweep calibration and its output must
not be published as the benchmark's `configs/safety_thresholds.json`.

Usage:
    python calibrate_reference_thresholds_from_expert_demos.py \
        /path/to/SoftVTBench_data/object-soft \
        /path/to/SoftVTBench_data/spatial-soft \
        --percentile 99 --output safety_thresholds.expert_reference.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

RMS_KEYS = ("obs/fem_deformation_rms", "soft_extras/fem_deformation_rms")


def _series(demo: h5py.Group) -> np.ndarray | None:
    for key in RMS_KEYS:
        if key in demo:
            arr = np.asarray(demo[key], dtype=float).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                return arr
    return None


def collect(roots: list[Path], successful_only: bool) -> dict[str, list[float]]:
    per_asset: dict[str, list[float]] = defaultdict(list)
    for root in roots:
        for h5_path in sorted(root.rglob("*.hdf5")):
            with h5py.File(h5_path, "r") as f:
                if "data" not in f:
                    continue
                for demo_key in f["data"]:
                    demo = f["data"][demo_key]
                    if successful_only and not bool(demo.attrs.get("success", True)):
                        continue
                    asset = str(demo.attrs.get("asset_name", "")).strip()
                    if not asset:
                        continue
                    series = _series(demo)
                    if series is None:
                        continue
                    per_asset[asset].append(float(series.max()))
    return per_asset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+", help="dataset suite folders holding replayed_demos/*.hdf5")
    ap.add_argument("--percentile", type=float, default=99.0, help="percentile of expert D_peak used as tau_o")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--include-failures", action="store_true",
                    help="calibrate over all demos, not just successful ones")
    args = ap.parse_args()

    per_asset = collect(args.roots, successful_only=not args.include_failures)
    if not per_asset:
        raise SystemExit(f"no deformable demos with FEM data under {args.roots}")

    thresholds = {}
    print(f'{"asset":<24}{"n":>6}{"p50":>9}{"p95":>9}{f"p{args.percentile:g}":>9}{"max":>9}')
    print("-" * 66)
    for asset, peaks in sorted(per_asset.items()):
        v = np.asarray(peaks, dtype=float)
        tau = float(np.percentile(v, args.percentile))
        thresholds[asset] = {
            "tau": round(tau, 6),
            "n_demos": int(v.size),
            "d_peak_p50": round(float(np.percentile(v, 50)), 6),
            "d_peak_p95": round(float(np.percentile(v, 95)), 6),
            "d_peak_max": round(float(v.max()), 6),
        }
        print(f"{asset:<24}{v.size:>6}{np.percentile(v,50):>9.3f}{np.percentile(v,95):>9.3f}"
              f"{tau:>9.3f}{v.max():>9.3f}")

    payload = {
        "_comment": (
            "Per-object safety thresholds tau_o for SoftVTBench Safety Success (Eq. 4). "
            "D(t) is obs/fem_deformation_rms: bbox-diagonal-normalised FEM-RMS nodal "
            "displacement in percent, after removing global translation. "
            "tau_o is the given percentile of D_peak over successful expert demos."
        ),
        "metric": "D_peak = max_t obs/fem_deformation_rms  (percent of reference bbox diagonal)",
        "calibration": {
            "percentile": args.percentile,
            "successful_demos_only": not args.include_failures,
            "sources": [str(r) for r in args.roots],
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "thresholds": thresholds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.output}  ({len(thresholds)} assets)")


if __name__ == "__main__":
    main()
