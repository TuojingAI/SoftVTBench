#!/usr/bin/env python3
"""Read-only preflight for SoftVTBench training and closed-loop evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


METRIC_ID = "fem_rms_rigid_aligned_bbox_pct_v1"
SOFT_SUITES = {"object-soft", "spatial-soft"}
ALL_SUITES = SOFT_SUITES | {"object-rigid", "spatial-rigid"}
ROOT = Path(__file__).resolve().parents[1]


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def pass_(self, message: str) -> None:
        print(f"PASS  {message}")

    def warn(self, message: str) -> None:
        print(f"WARN  {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"FAIL  {message}")

    def require(self, condition: bool, message: str) -> bool:
        if condition:
            self.pass_(message)
        else:
            self.fail(message)
        return condition


def check_python(report: Report, label: str, executable: str | None, imports: tuple[str, ...]) -> None:
    if not executable:
        report.fail(f"{label} interpreter was not supplied")
        return
    path = Path(executable).expanduser()
    if not report.require(path.is_file(), f"{label} interpreter exists: {path}"):
        return
    code = "; ".join(f"import {name}" for name in imports)
    result = subprocess.run(
        [str(path), "-c", code],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    if result.returncode == 0:
        report.pass_(f"{label} imports: {', '.join(imports)}")
    else:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else f"exit {result.returncode}"
        report.fail(f"{label} imports failed: {detail}")


def check_dataset(report: Report, root: Path, suite: str) -> None:
    if not report.require(root.is_dir(), f"dataset suite exists: {root}"):
        return
    if suite in SOFT_SUITES:
        task_suite = "libero_object" if suite == "object-soft" else "libero_spatial"
        task_root = root / task_suite
        task_dirs = sorted(path for path in task_root.glob(f"{task_suite}_task*") if path.is_dir())
        report.require(len(task_dirs) == 10, f"dataset has 10 nested task directories (found {len(task_dirs)})")
        hdf5 = sorted(task_root.glob(f"{task_suite}_task*/replayed_demos/*.hdf5"))
        video_dirs = sorted(task_root.glob(f"{task_suite}_task*/video_datasets/{task_suite}_task*"))
        report.require(len(video_dirs) == 10, f"dataset has 10 nested video directories (found {len(video_dirs)})")
    else:
        report.require((root / "replayed_demos").is_dir(), "dataset has replayed_demos/")
        report.require((root / "video_datasets").is_dir(), "dataset has video_datasets/")
        hdf5 = sorted((root / "replayed_demos").glob("*.hdf5"))
    report.require(len(hdf5) == 10, f"dataset has 10 task HDF5 files (found {len(hdf5)})")
    manifests = sorted(root.glob("manifest*.jsonl"))
    if manifests:
        report.pass_(f"dataset has {len(manifests)} manifest file(s)")
    else:
        report.warn("dataset has no manifest JSONL; this is expected for the current rigid release")
    if suite in SOFT_SUITES:
        report.require(bool(hdf5), "soft suite contains nested HDF5 demonstrations")


def check_checkpoint(report: Report, root: Path) -> None:
    if not report.require(root.is_dir(), f"checkpoint exists: {root}"):
        return
    report.require((root / "assets").is_dir(), "checkpoint has assets/")
    norm_stats = list((root / "assets").glob("**/norm_stats.json"))
    report.require(bool(norm_stats), f"checkpoint has norm_stats.json (found {len(norm_stats)})")
    weight_markers = [root / "params", root / "params.json", root / "metadata.json"]
    if any(path.exists() for path in weight_markers):
        report.pass_("checkpoint has a parameter/metadata marker")
    else:
        report.warn("checkpoint parameter layout was not recognized; OpenPI will validate it when loading")


def check_thresholds(report: Report, path: Path) -> None:
    if not report.require(path.is_file(), f"formal threshold file exists: {path}"):
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.fail(f"threshold JSON is invalid: {exc}")
        return
    report.require(payload.get("metric_id") == METRIC_ID, f"threshold metric_id is {METRIC_ID}")
    report.require(
        payload.get("calibration", {}).get("method") == "compression_sweep",
        "threshold calibration method is compression_sweep",
    )
    thresholds = payload.get("thresholds")
    report.require(isinstance(thresholds, dict) and bool(thresholds), "threshold map is non-empty")


def check_eval_assets(report: Report, root: Path) -> None:
    usd_root = root / "USD" if (root / "USD").is_dir() else root
    if not report.require(usd_root.is_dir(), f"evaluation USD directory exists: {usd_root}"):
        return
    usd_files = list(usd_root.rglob("*.usd")) + list(usd_root.rglob("*.usda")) + list(usd_root.rglob("*.usdc"))
    report.require(bool(usd_files), f"evaluation assets contain USD files (found {len(usd_files)})")


def check_runtime_assets(report: Report, root: Path) -> None:
    if not report.require(root.is_dir(), f"Franka/GelSight runtime asset directory exists: {root}"):
        return
    usd_files = list(root.rglob("*.usd")) + list(root.rglob("*.usda")) + list(root.rglob("*.usdc"))
    report.require(bool(usd_files), f"runtime assets contain robot/tactile USD files (found {len(usd_files)})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval", "all"), default="all")
    parser.add_argument("--suite", choices=sorted(ALL_SUITES), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--eval-assets", type=Path)
    parser.add_argument(
        "--runtime-assets",
        type=Path,
        default=ROOT / "SoftVTBench/source/tac_manip/tac_manip/assets/data",
        help="Franka/GelSight asset directory; defaults to the documented in-repository symlink",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--softvtbench-python")
    parser.add_argument("--openpi-python")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = Report()
    check_dataset(report, args.data_root.expanduser(), args.suite)
    check_python(report, "OpenPI", args.openpi_python, ("jax", "h5py", "pandas", "pyarrow"))
    if args.mode in {"eval", "all"}:
        check_python(report, "SoftVTBench", args.softvtbench_python, ("isaacsim", "h5py", "numpy", "scipy", "tyro"))
        check_runtime_assets(report, args.runtime_assets.expanduser())
        if args.checkpoint:
            check_checkpoint(report, args.checkpoint.expanduser())
        else:
            report.fail("--checkpoint is required for local-server evaluation")
        if args.suite in SOFT_SUITES:
            if args.eval_assets:
                check_eval_assets(report, args.eval_assets.expanduser())
            else:
                report.fail("--eval-assets is required for a soft suite")
            if args.thresholds:
                check_thresholds(report, args.thresholds.expanduser())
            else:
                report.fail("--thresholds is required for a soft suite")
    print(f"\nRESULT: {'FAIL' if report.failures else 'PASS'} ({report.failures} failure(s))")
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
