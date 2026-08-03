#!/usr/bin/env python3
"""Regenerate deterministic SHA-256 manifests for both release repositories."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from softvtbench.quality import digest, files_under  # noqa: E402


def write_manifest(root: Path) -> None:
    manifest = root / "MANIFEST.sha256"
    lines = [
        f"{digest(path)}  {path.relative_to(root)}"
        for path in files_under(root)
        if path != manifest
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {manifest} ({len(lines)} files)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=ROOT.parent / "SoftVTBench-Models")
    args = parser.parse_args()
    write_manifest(ROOT)
    if args.models_root.is_dir():
        write_manifest(args.models_root.resolve())
    else:
        parser.error(f"models repository does not exist: {args.models_root}")


if __name__ == "__main__":
    main()

