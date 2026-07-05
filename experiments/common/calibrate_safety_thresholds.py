#!/usr/bin/env python3
"""Build SoftVTBench per-object thresholds from compression-sweep records.

Each input is JSONL with one compression trial per line. Required fields:

    {"asset": "soft_pastry001", "stable": true, "d_peak": 12.4}

`d_peak` must use the same percent-of-reference-bbox-diagonal FEM-RMS metric as
closed-loop evaluation. For each object, D_ref is the largest stable d_peak and
the public v1 threshold is tau_o = kappa * D_ref (paper Appendix C.3).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .deformation import METRIC_ID
except ImportError:  # direct script execution
    from deformation import METRIC_ID


def _as_bool(value: Any, *, path: Path, line_no: int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{path}:{line_no}: stable must be true or false")


def collect(paths: list[Path]) -> dict[str, list[float]]:
    stable_peaks: dict[str, list[float]] = defaultdict(list)
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, 1):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                asset = str(row.get("asset", "")).strip()
                if not asset:
                    raise ValueError(f"{path}:{line_no}: missing asset")
                stable = _as_bool(row.get("stable"), path=path, line_no=line_no)
                try:
                    d_peak = float(row["d_peak"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_no}: d_peak must be numeric") from exc
                if not math.isfinite(d_peak) or d_peak < 0:
                    raise ValueError(f"{path}:{line_no}: invalid d_peak={d_peak}")
                if stable:
                    stable_peaks[asset].append(d_peak)
    return stable_peaks


def build_payload(paths: list[Path], kappa: float) -> dict[str, Any]:
    if not 0 < kappa <= 1:
        raise ValueError(f"kappa must be in (0, 1], got {kappa}")
    stable_peaks = collect(paths)
    if not stable_peaks:
        raise ValueError("no stable compression-sweep records found")

    thresholds: dict[str, dict[str, Any]] = {}
    for asset, peaks in sorted(stable_peaks.items()):
        d_ref = max(peaks)
        thresholds[asset] = {
            "tau": round(kappa * d_ref, 6),
            "d_ref": round(d_ref, 6),
            "kappa": kappa,
            "n_stable_trials": len(peaks),
        }

    return {
        "schema_version": 1,
        "protocol_id": "softvtbench-v1",
        "metric_id": METRIC_ID,
        "metric": "rigid-aligned D_peak in percent of reference bbox diagonal",
        "calibration": {
            "method": "compression_sweep",
            "definition": "tau_o = kappa * max_stable_d_peak",
            "kappa": kappa,
            "sources": [str(path) for path in paths],
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "thresholds": thresholds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="compression-sweep JSONL files")
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = build_payload(args.inputs, args.kappa)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(payload['thresholds'])} assets)")


if __name__ == "__main__":
    main()
