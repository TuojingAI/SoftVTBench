#!/usr/bin/env python3
"""Sanitize SoftVTBench dataset metadata for a portable public release.

The command is dry-run by default. ``--apply`` requires a new backup directory;
all replaced metadata and every removed junk file are copied there first. Large
HDF5 files are not duplicated, but the exact attributes changed in them are
recorded as JSONL for restoration.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import h5py

from build_release_manifest import build


SOFT_SUITES = ("object-soft", "spatial-soft")
ALL_SUITES = (*SOFT_SUITES, "object-rigid", "spatial-rigid")

ASSET_NOTICE = """# Asset terms

The SoftVTBench dataset card does **not** grant a blanket Apache-2.0 license to
the files in this directory. Scene, texture, LIBERO, SoftVTBench, tactile-simulation,
and deformable-object assets retain their original terms.

Publication of this notice does not establish redistribution permission. The
maintainers must attach the applicable license or written redistribution
permission for every file in this directory. Until that evidence is present,
do not download, mirror, or redistribute this directory; its status is
**redistribution pending**.

See the code release's root `README.md` and `THIRD_PARTY_NOTICES` for the
provenance policy.
"""

EVAL_ASSET_README = """# SoftVTBench evaluation assets (`eval-assets/`)

This directory contains the USD scene and deformable assets needed for
closed-loop evaluation in Isaac Sim. Training does not require it.

## Redistribution status

**Redistribution pending.** The SoftVTBench dataset and code licenses do not
grant a blanket Apache-2.0 license to this directory. Publication of this
README does not establish permission to download, mirror, or redistribute the
files here. See `ASSET_TERMS.md` and the code release's
root `README.md`. The maintainers must attach written redistribution evidence
before presenting this directory as publicly downloadable.

## Intended layout after clearance

```text
eval-assets/
└── USD/
    ├── <rigid scene object>/<asset>.usd
    └── <deformable object>/<asset>.usd
```

Once clearance is complete, point the evaluator at this directory:

```bash
export SOFTVT_EVAL_USD_DIR=/path/to/eval-assets/USD
```
"""


def _portable(value: Any) -> Any:
    if isinstance(value, str) and os.path.isabs(value):
        return Path(value).name
    if isinstance(value, list):
        return [_portable(item) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    return value


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _metadata_files(root: Path) -> list[Path]:
    paths = [root / "README.md"]
    paths.extend(root.rglob("*.jsonl"))
    for relative in ("eval-assets/README.md", "eval-assets/ASSET_TERMS.md", "soft-assets/README.md", "soft-assets/ASSET_TERMS.md"):
        paths.append(root / relative)
    return sorted({path for path in paths if path.is_file()})


def _junk_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and (path.name == ".ms_upload_cache" or path.name.startswith("._"))
    )


def _hdf5_absolute_attrs(root: Path) -> list[dict[str, str]]:
    changes: list[dict[str, str]] = []
    for suite in SOFT_SUITES:
        for hdf5_path in sorted((root / suite).rglob("*.hdf5")):
            with h5py.File(hdf5_path, "r") as file:
                for demo_name, demo in file.get("data", {}).items():
                    for key, raw in demo.attrs.items():
                        value = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
                        if os.path.isabs(value):
                            changes.append(
                                {
                                    "hdf5": hdf5_path.relative_to(root).as_posix(),
                                    "demo": demo_name,
                                    "key": key,
                                    "old_value": value,
                                    "new_value": Path(value).name,
                                }
                            )
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--dataset-card", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    card = args.dataset_card.resolve()
    backup = args.backup_dir.resolve()
    if not root.is_dir() or not card.is_file():
        raise SystemExit("dataset root or dataset card does not exist")
    for suite in ALL_SUITES:
        if not (root / suite).is_dir():
            raise SystemExit(f"missing suite directory: {root / suite}")

    manifests = {suite: build(root / suite, suite) for suite in SOFT_SUITES}
    for suite, records in manifests.items():
        if len(records) != 500:
            raise SystemExit(f"{suite}: expected 500 canonical records, got {len(records)}")
    junk = _junk_files(root)
    hdf5_changes = _hdf5_absolute_attrs(root)
    print(f"canonical manifests: {', '.join(f'{name}={len(rows)}' for name, rows in manifests.items())}")
    print(f"junk files to remove: {len(junk)}")
    print(f"absolute HDF5 attributes to make portable: {len(hdf5_changes)}")

    if not args.apply:
        print("DRY RUN: pass --apply to perform the backed-up rewrite")
        return 0
    if backup.exists():
        raise SystemExit(f"backup directory already exists: {backup}")
    backup.mkdir(parents=True)

    # Back up every small file that will be replaced or removed.
    for path in [*_metadata_files(root), *junk]:
        destination = backup / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    _write_jsonl(backup / "hdf5_attribute_changes.jsonl", hdf5_changes)

    # Canonical portable soft manifests replace all legacy fragments.
    for suite, records in manifests.items():
        suite_root = root / suite
        _write_jsonl(suite_root / "manifest.jsonl", records)
        for legacy in suite_root.glob("manifest*.jsonl"):
            if legacy.name != "manifest.jsonl":
                legacy.unlink()

    # Keep diagnostic JSONL files, but strip machine-specific paths.
    for jsonl_path in root.rglob("*.jsonl"):
        if jsonl_path.name == "manifest.jsonl":
            continue
        rows: list[dict[str, Any]] = []
        for line in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                rows.append(_portable(json.loads(line)))
        _write_jsonl(jsonl_path, rows)

    for change in hdf5_changes:
        with h5py.File(root / change["hdf5"], "r+") as file:
            file["data"][change["demo"]].attrs[change["key"]] = change["new_value"]

    for path in junk:
        path.unlink()

    shutil.copy2(card, root / "README.md")
    for folder in (root / "eval-assets", root / "soft-assets"):
        (folder / "ASSET_TERMS.md").write_text(ASSET_NOTICE, encoding="utf-8")
    (root / "eval-assets" / "README.md").write_text(EVAL_ASSET_README, encoding="utf-8")

    (backup / "SANITIZE_COMPLETE.json").write_text(
        json.dumps(
            {
                "dataset_root": str(root),
                "canonical_manifest_rows": {key: len(value) for key, value in manifests.items()},
                "junk_removed": [path.relative_to(root).as_posix() for path in junk],
                "hdf5_attributes_changed": len(hdf5_changes),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"APPLIED; backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
