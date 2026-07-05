#!/usr/bin/env python3
"""Read-only hygiene and portability audit for a SoftVTBench dataset bundle."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


SUITES = ("object-soft", "spatial-soft", "object-rigid", "spatial-rigid")
ABSOLUTE_WINDOWS = re.compile(r"^[A-Za-z]:[\\/]")


def strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, bytes):
        yield value.decode("utf-8", errors="replace")
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="root containing the six released folders")
    parser.add_argument("--asset-permission", type=Path, help="written redistribution evidence")
    args = parser.parse_args()
    root = args.root.resolve()
    failures: list[str] = []
    warnings: list[str] = []

    for name in (*SUITES, "eval-assets", "soft-assets"):
        if not (root / name).is_dir():
            failures.append(f"missing folder: {name}/")

    junk = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.name == ".ms_upload_cache" or path.name.startswith("._")
    ]
    if junk:
        failures.append(f"upload/OS junk files present ({len(junk)}): {junk[:10]}")

    absolute_examples: list[str] = []
    manifest_rows: dict[str, int] = {}
    for suite in SUITES:
        suite_root = root / suite
        total = 0
        for manifest in sorted(suite_root.glob("manifest*.jsonl")):
            rows = 0
            with manifest.open(encoding="utf-8", errors="replace") as file:
                for line_no, line in enumerate(file, 1):
                    if not line.strip():
                        continue
                    rows += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        failures.append(f"invalid JSONL {manifest.relative_to(root)}:{line_no}: {exc}")
                        continue
            manifest_rows[manifest.relative_to(root).as_posix()] = rows
            total += rows
        if suite.endswith("-soft") and total != 500:
            failures.append(f"{suite}: expected 500 soft manifest rows across fragments, found {total}")
        if suite.endswith("-rigid") and total == 0:
            warnings.append(f"{suite}: no manifest (current release layout)")

    # Scan every JSONL, including failure/duplicate provenance files, not only
    # the public manifest.
    for jsonl_path in sorted(root.rglob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8", errors="replace") as file:
            for line_no, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # already reported above for manifest files
                for value in strings(row):
                    if os.path.isabs(value) or ABSOLUTE_WINDOWS.match(value):
                        if len(absolute_examples) < 10:
                            absolute_examples.append(f"{jsonl_path.relative_to(root)}:{line_no}: {value}")
    if absolute_examples:
        failures.append("absolute paths in manifests:\n  " + "\n  ".join(absolute_examples))

    hdf5_absolute: list[str] = []
    try:
        import h5py
    except ImportError:
        failures.append("h5py is required to audit embedded HDF5 attributes")
    else:
        for suite in SUITES:
            for hdf5_path in (root / suite).rglob("*.hdf5"):
                with h5py.File(hdf5_path, "r") as file:
                    if "data" not in file:
                        continue
                    for demo_name, demo in file["data"].items():
                        for key, value in demo.attrs.items():
                            for text in strings(value.tolist() if hasattr(value, "tolist") else value):
                                if os.path.isabs(text) or ABSOLUTE_WINDOWS.match(text):
                                    if len(hdf5_absolute) < 10:
                                        relative = hdf5_path.relative_to(root)
                                        hdf5_absolute.append(f"{relative}:{demo_name}.attrs[{key!r}]: {text}")
    if hdf5_absolute:
        failures.append("absolute paths embedded in HDF5 attributes:\n  " + "\n  ".join(hdf5_absolute))

    permission_ok = bool(args.asset_permission and args.asset_permission.is_file())
    readme = root / "README.md"
    if readme.is_file() and "license: apache-2.0" in readme.read_text(encoding="utf-8", errors="ignore").lower():
        failures.append("dataset card claims blanket Apache-2.0 despite separately licensed assets")
    if not permission_ok:
        failures.append("written eval-assets/soft-assets redistribution evidence was not supplied")

    for item in warnings:
        print(f"WARN  {item}")
    if failures:
        for item in failures:
            print(f"FAIL  {item}")
        print(f"\nRESULT: FAIL ({len(failures)} failure(s), {len(warnings)} warning(s))")
        return 1
    print(f"PASS  six folders, manifests/HDF5 path portability, asset terms, and file hygiene ({manifest_rows})")
    print(f"\nRESULT: PASS (0 failure(s), {len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
