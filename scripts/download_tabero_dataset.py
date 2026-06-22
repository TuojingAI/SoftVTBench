#!/usr/bin/env python3
"""Download the SoftTacWorld-v0 dataset from ModelScope.

The public dataset is hosted at:
https://www.modelscope.cn/datasets/Arthur12137/SoftTacWorld-v0

This script intentionally keeps the downloaded payload outside Git-tracked paths
and only validates that the expected raw-data layout can be found.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


DEFAULT_DATASET_ID = "Arthur12137/SoftTacWorld-v0"
EXPECTED_SUITES = ("libero_object", "libero_spatial")


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _has_softtacworld_layout(path: Path) -> bool:
    for suite in EXPECTED_SUITES:
        suite_root = path / suite
        if not (suite_root / "replayed_demos").is_dir():
            return False
        if not (suite_root / "video_datasets").is_dir():
            return False
    return True


def _find_raw_root(local_dir: Path) -> Path | None:
    candidates = [local_dir]
    candidates.extend(path for path in local_dir.rglob("*") if path.is_dir())
    for candidate in candidates:
        if _has_softtacworld_layout(candidate):
            return candidate
    return None


def _extract_archives(local_dir: Path) -> None:
    suffixes = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
    archives = [path for path in local_dir.rglob("*") if path.is_file() and path.name.endswith(suffixes)]
    for archive in archives:
        marker = archive.with_name(f".{archive.name}.extracted")
        if marker.exists():
            continue
        print(f"Extracting archive: {archive}")
        shutil.unpack_archive(str(archive), extract_dir=str(archive.parent))
        marker.write_text("ok\n")


def _print_tree_hint(local_dir: Path) -> None:
    print(f"\nCould not find the expected SoftTacWorld raw-data layout under: {local_dir}", file=sys.stderr)
    print("Expected:", file=sys.stderr)
    print("  <RAW_ROOT>/libero_object/replayed_demos/*.hdf5", file=sys.stderr)
    print("  <RAW_ROOT>/libero_object/video_datasets/...", file=sys.stderr)
    print("  <RAW_ROOT>/libero_spatial/replayed_demos/*.hdf5", file=sys.stderr)
    print("  <RAW_ROOT>/libero_spatial/video_datasets/...", file=sys.stderr)
    print("\nTop-level downloaded files:", file=sys.stderr)
    for path in sorted(local_dir.glob("*"))[:80]:
        print(f"  {path.relative_to(local_dir)}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--local-dir", default="data_softtacworld/raw/SoftTacWorld-v0")
    parser.add_argument("--force", action="store_true", help="Remove local-dir before downloading.")
    parser.add_argument("--skip-download", action="store_true", help="Only validate an existing local-dir.")
    parser.add_argument("--no-extract", action="store_true", help="Do not auto-extract zip/tar archives.")
    parser.add_argument("--skip-validate", action="store_true", help="Download without checking the raw-data layout.")
    args = parser.parse_args()

    local_dir = Path(args.local_dir).expanduser().resolve()

    if args.force and local_dir.exists():
        shutil.rmtree(local_dir)

    if not args.skip_download:
        if shutil.which("modelscope") is None:
            print("Missing `modelscope` CLI. Install it first:", file=sys.stderr)
            print("  pip install modelscope", file=sys.stderr)
            return 127
        local_dir.mkdir(parents=True, exist_ok=True)
        _run(["modelscope", "download", "--dataset", args.dataset_id, "--local_dir", str(local_dir)])

    if args.skip_validate:
        print(f"Downloaded dataset directory: {local_dir}")
        return 0

    if not args.no_extract:
        _extract_archives(local_dir)

    raw_root = _find_raw_root(local_dir)
    if raw_root is None:
        _print_tree_hint(local_dir)
        return 2

    print("\nDataset layout looks valid.")
    print(f"Downloaded dataset directory: {local_dir}")
    print(f"Detected RAW_ROOT: {raw_root}")
    print("\nUse this for training setup:")
    print(f"export RAW_ROOT={raw_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
