#!/usr/bin/env python3
"""DEPRECATED -- superseded by experiments/common/softvtbench_metrics.py.

This script predates the SoftVTBench paper protocol. It thresholds
fem_deformation_max and fem_bbox_delta against *global* thresholds, whereas
the paper defines Safety Success from D_peak = max_t fem_deformation_rms
against a calibrated per-object tau_o (Eq. 3-4). It is kept only for the
bbox-delta diagnostics it also reports; no evaluation script calls it.

Summarize success and soft-body safety metrics for spatial pastry evals.

The script accepts either a replay-format collection, an eval output folder, or
both. It treats missing FEM/deformation fields as missing evidence, not as safe.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


@dataclass
class EpisodeMetric:
    source: str
    task: str
    episode: str
    success: bool | None
    safety_known: bool
    safe: bool | None
    fem_max_peak: float | None
    fem_rms_peak: float | None
    bbox_delta_peak: float | None
    frames: int
    reason: str


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _read_success_attr(group: h5py.Group) -> bool | None:
    raw = group.attrs.get("success")
    if raw is None:
        return None
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    if text.lower() in {"true", "1", "yes"}:
        return True
    if text.lower() in {"false", "0", "no"}:
        return False
    return None


def _dataset(group: h5py.Group, names: Iterable[str]) -> np.ndarray | None:
    for name in names:
        if name in group:
            return np.asarray(group[name])
    return None


def _hdf5_episode_metrics(path: Path, max_thr: float, rms_thr: float, bbox_thr: float | None) -> list[EpisodeMetric]:
    out: list[EpisodeMetric] = []
    with h5py.File(path, "r") as f:
        if "data" in f:
            demos = [(name, f["data"][name]) for name in sorted(f["data"].keys(), key=_natural_key)]
        else:
            demos = [("demo_0", f)]
        for demo_name, group in demos:
            task_id = group.attrs.get("task_id", "")
            task = f"task{task_id}" if str(task_id) else _task_from_path(path)
            success = _read_success_attr(group)
            frames = int(group.attrs.get("num_frames", 0) or group.attrs.get("num_samples", 0) or 0)
            fem_max = _dataset(group, ("obs/fem_deformation_max", "soft_extras/fem_deformation_max"))
            fem_rms = _dataset(group, ("obs/fem_deformation_rms", "soft_extras/fem_deformation_rms"))
            bbox_dims = _dataset(group, ("obs/fem_bbox_dims", "soft_extras/fem_bbox_dims"))

            fem_max_peak = _as_float(np.nanmax(fem_max)) if fem_max is not None and fem_max.size else None
            fem_rms_peak = _as_float(np.nanmax(fem_rms)) if fem_rms is not None and fem_rms.size else None
            bbox_delta_peak = None
            if bbox_dims is not None and bbox_dims.size and bbox_dims.ndim >= 2:
                ref = bbox_dims[0].astype(np.float64)
                delta = np.linalg.norm(bbox_dims.astype(np.float64) - ref, axis=-1)
                bbox_delta_peak = _as_float(np.nanmax(delta))

            reasons: list[str] = []
            if fem_max_peak is None:
                reasons.append("missing_fem_max")
            elif fem_max_peak > max_thr:
                reasons.append(f"fem_max>{max_thr:g}")
            if fem_rms_peak is None:
                reasons.append("missing_fem_rms")
            elif fem_rms_peak > rms_thr:
                reasons.append(f"fem_rms>{rms_thr:g}")
            if bbox_thr is not None:
                if bbox_delta_peak is None:
                    reasons.append("missing_bbox_delta")
                elif bbox_delta_peak > bbox_thr:
                    reasons.append(f"bbox_delta>{bbox_thr:g}")

            safety_known = not any(r.startswith("missing_") for r in reasons)
            safe = None if not safety_known else not reasons
            out.append(
                EpisodeMetric(
                    source=str(path),
                    task=task,
                    episode=demo_name,
                    success=success,
                    safety_known=safety_known,
                    safe=safe,
                    fem_max_peak=fem_max_peak,
                    fem_rms_peak=fem_rms_peak,
                    bbox_delta_peak=bbox_delta_peak,
                    frames=frames,
                    reason=";".join(reasons),
                )
            )
    return out


def _jsonl_episode_metric(path: Path, max_thr: float, rms_thr: float, bbox_thr: float | None) -> EpisodeMetric | None:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except OSError:
        return None
    if not rows:
        return None
    # Collection manifests are also JSONL, but eval debug JSONL has per-frame
    # records with frame/exp_idx/task_id fields. Ignore manifests here.
    if not any("frame" in r and "exp_idx" in r and "task_id" in r for r in rows):
        return None
    if not any("fem_deformation_max" in r or "fem_deformation_rms" in r or "fem_bbox_delta" in r for r in rows):
        return None

    task_id = rows[-1].get("task_id", "")
    exp_idx = rows[-1].get("exp_idx", path.stem)
    success_values = [r.get("success_now") for r in rows if r.get("success_now") is not None]
    success = bool(success_values[-1]) if success_values else None

    fem_max_vals = [_as_float(r.get("fem_deformation_max")) for r in rows]
    fem_rms_vals = [_as_float(r.get("fem_deformation_rms")) for r in rows]
    bbox_vals = [_as_float(r.get("fem_bbox_delta")) for r in rows]
    fem_max_vals = [v for v in fem_max_vals if v is not None]
    fem_rms_vals = [v for v in fem_rms_vals if v is not None]
    bbox_vals = [v for v in bbox_vals if v is not None]

    fem_max_peak = max(fem_max_vals) if fem_max_vals else None
    fem_rms_peak = max(fem_rms_vals) if fem_rms_vals else None
    bbox_delta_peak = max(bbox_vals) if bbox_vals else None
    reasons: list[str] = []
    if fem_max_peak is None:
        reasons.append("missing_fem_max")
    elif fem_max_peak > max_thr:
        reasons.append(f"fem_max>{max_thr:g}")
    if fem_rms_peak is None:
        reasons.append("missing_fem_rms")
    elif fem_rms_peak > rms_thr:
        reasons.append(f"fem_rms>{rms_thr:g}")
    if bbox_thr is not None:
        if bbox_delta_peak is None:
            reasons.append("missing_bbox_delta")
        elif bbox_delta_peak > bbox_thr:
            reasons.append(f"bbox_delta>{bbox_thr:g}")
    safety_known = not any(r.startswith("missing_") for r in reasons)
    return EpisodeMetric(
        source=str(path),
        task=f"task{task_id}" if str(task_id) else _task_from_path(path),
        episode=f"exp_{exp_idx}",
        success=success,
        safety_known=safety_known,
        safe=None if not safety_known else not reasons,
        fem_max_peak=fem_max_peak,
        fem_rms_peak=fem_rms_peak,
        bbox_delta_peak=bbox_delta_peak,
        frames=len(rows),
        reason=";".join(reasons),
    )


def _task_from_path(path: Path) -> str:
    m = re.search(r"task[_-]?(\d+)", str(path))
    return f"task{m.group(1)}" if m else "unknown"


def _natural_key(text: str) -> list[Any]:
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", text)]


def collect_metrics(root: Path, max_thr: float, rms_thr: float, bbox_thr: float | None) -> list[EpisodeMetric]:
    metrics: list[EpisodeMetric] = []
    h5_paths = sorted(root.rglob("*.hdf5"))
    for path in h5_paths:
        try:
            metrics.extend(_hdf5_episode_metrics(path, max_thr, rms_thr, bbox_thr))
        except Exception as exc:
            metrics.append(
                EpisodeMetric(str(path), _task_from_path(path), path.stem, None, False, None, None, None, None, 0, f"read_error:{exc}")
            )
    for path in sorted(root.rglob("*.jsonl")):
        metric = _jsonl_episode_metric(path, max_thr, rms_thr, bbox_thr)
        if metric is not None:
            metrics.append(metric)
    return metrics


def summarize(metrics: list[EpisodeMetric]) -> dict[str, Any]:
    by_task: dict[str, list[EpisodeMetric]] = {}
    for m in metrics:
        by_task.setdefault(m.task, []).append(m)
    task_rows: dict[str, dict[str, Any]] = {}
    for task, rows in sorted(by_task.items(), key=lambda kv: _natural_key(kv[0])):
        total = len(rows)
        success_known = [r for r in rows if r.success is not None]
        safe_known = [r for r in rows if r.safety_known]
        safe_count = sum(1 for r in rows if r.safe is True)
        unsafe_count = sum(1 for r in rows if r.safe is False)
        safe_success_known = [r for r in rows if r.success is not None and r.safety_known]
        safe_success_count = sum(1 for r in safe_success_known if r.success is True and r.safe is True)
        task_rows[task] = {
            "episodes": total,
            "success_known": len(success_known),
            "success": sum(1 for r in success_known if r.success is True),
            "success_rate": _rate(sum(1 for r in success_known if r.success is True), len(success_known)),
            "safety_known": len(safe_known),
            "safe": safe_count,
            "unsafe": unsafe_count,
            "missing_safety": total - len(safe_known),
            "safe_rate_known_only": _rate(safe_count, len(safe_known)),
            "safe_rate_all": _rate(safe_count, total),
            "safe_success_known": len(safe_success_known),
            "safe_success": safe_success_count,
            "safe_success_rate_known_only": _rate(safe_success_count, len(safe_success_known)),
            "safe_success_rate_all": _rate(safe_success_count, total),
            "fem_max_peak": _max_or_none(r.fem_max_peak for r in rows),
            "fem_rms_peak": _max_or_none(r.fem_rms_peak for r in rows),
            "bbox_delta_peak": _max_or_none(r.bbox_delta_peak for r in rows),
        }
    all_known_success = [r for r in metrics if r.success is not None]
    all_known_safe = [r for r in metrics if r.safety_known]
    all_known_safe_success = [r for r in metrics if r.success is not None and r.safety_known]
    all_safe_success_count = sum(1 for r in all_known_safe_success if r.success is True and r.safe is True)
    return {
        "overall": {
            "episodes": len(metrics),
            "success_known": len(all_known_success),
            "success": sum(1 for r in all_known_success if r.success is True),
            "success_rate": _rate(sum(1 for r in all_known_success if r.success is True), len(all_known_success)),
            "safety_known": len(all_known_safe),
            "safe": sum(1 for r in metrics if r.safe is True),
            "unsafe": sum(1 for r in metrics if r.safe is False),
            "missing_safety": len(metrics) - len(all_known_safe),
            "safe_rate_known_only": _rate(sum(1 for r in metrics if r.safe is True), len(all_known_safe)),
            "safe_rate_all": _rate(sum(1 for r in metrics if r.safe is True), len(metrics)),
            "safe_success_known": len(all_known_safe_success),
            "safe_success": all_safe_success_count,
            "safe_success_rate_known_only": _rate(all_safe_success_count, len(all_known_safe_success)),
            "safe_success_rate_all": _rate(all_safe_success_count, len(metrics)),
            "fem_max_peak": _max_or_none(r.fem_max_peak for r in metrics),
            "fem_rms_peak": _max_or_none(r.fem_rms_peak for r in metrics),
            "bbox_delta_peak": _max_or_none(r.bbox_delta_peak for r in metrics),
        },
        "tasks": task_rows,
    }


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else 100.0 * num / den


def _max_or_none(values: Iterable[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    return max(valid) if valid else None


def write_outputs(metrics: list[EpisodeMetric], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(metrics)
    (output_dir / "spatial_soft_safety_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [asdict(m) for m in metrics]
    with (output_dir / "spatial_soft_safety_episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [f.name for f in EpisodeMetric.__dataclass_fields__.values()])
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "spatial_soft_safety_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "task",
            "episodes",
            "success_known",
            "success",
            "success_rate",
            "safety_known",
            "safe",
            "unsafe",
            "missing_safety",
            "safe_rate_known_only",
            "safe_rate_all",
            "safe_success_known",
            "safe_success",
            "safe_success_rate_known_only",
            "safe_success_rate_all",
            "fem_max_peak",
            "fem_rms_peak",
            "bbox_delta_peak",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for task, row in summary["tasks"].items():
            writer.writerow({"task": task, **row})
        writer.writerow({"task": "overall", **summary["overall"]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path, help="Replay-format collection or eval output directories.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fem-max-threshold", type=float, default=0.60, help="Peak max deformation threshold in the stored field units.")
    parser.add_argument("--fem-rms-threshold", type=float, default=0.55, help="Peak RMS deformation threshold in the stored field units.")
    parser.add_argument("--bbox-delta-threshold", type=float, default=None, help="Optional peak bbox-dimension delta threshold in meters.")
    args = parser.parse_args()

    metrics: list[EpisodeMetric] = []
    for root in args.roots:
        metrics.extend(collect_metrics(root, args.fem_max_threshold, args.fem_rms_threshold, args.bbox_delta_threshold))
    write_outputs(metrics, args.output_dir)
    summary = summarize(metrics)
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"Wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
