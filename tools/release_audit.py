#!/usr/bin/env python3
"""Static release audit for the SoftVTBench public repository.

Default mode checks repository structure and source hygiene. ``--strict`` also
requires formal compression-sweep thresholds and verifies that evaluation
assets are either external or accompanied by redistribution evidence.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".pytest_cache", "__pycache__", ".venv", "results", "checkpoints"}
MAX_SOURCE_BYTES = 10 * 1024 * 1024
SECRET_PATTERNS = {
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "server-specific /vepfs path": re.compile(r"/vepfs(?:-[^/]+)?/"),
    "server-specific legacy workspace path": re.compile(
        r"/home/" + r"qiweiw/|/data/home/" + r"sim6g/|/data/Projects/" + r"Robotics/"
    ),
}


class Audit:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        print(f"PASS  {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"FAIL  {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"WARN  {message}")


def files(suffix: str | None = None):
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts):
            continue
        if suffix is None or path.suffix == suffix:
            yield path


def check_required(audit: Audit) -> None:
    required = [
        "README.md",
        "CITATION.cff",
        "LICENSE",
        "THIRD_PARTY_NOTICES",
        "environment.yml",
        "requirements.txt",
        "openpi/upstream/uv.lock",
        "configs/benchmark_protocol_v1.json",
        "configs/simulation_physics_v1.json",
        "openpi/scripts/train_softvtbench.sh",
        "openpi/scripts/evaluate_softvtbench.sh",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        audit.fail(f"missing required release files: {missing}")
    else:
        audit.ok("required metadata, environments, protocol, and public launchers exist")


def check_parsers(audit: Audit) -> None:
    errors: list[str] = []
    for path in files(".py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    for path in files(".json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    for path in files(".toml"):
        try:
            tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    shells = [str(path) for path in files(".sh")]
    for path in shells:
        result = subprocess.run(["bash", "-n", path], check=False, capture_output=True, text=True)
        if result.returncode:
            errors.append(f"{Path(path).relative_to(ROOT)}: {result.stderr.strip()}")
    if errors:
        audit.fail("syntax/parse failures:\n  " + "\n  ".join(errors))
    else:
        audit.ok("Python, JSON, TOML, and shell sources parse")


def check_hygiene(audit: Audit) -> None:
    findings: list[str] = []
    oversized: list[str] = []
    text_suffixes = {".py", ".sh", ".md", ".toml", ".json", ".jsonl", ".yaml", ".yml", ".cff", ".txt"}
    for path in files():
        if path.stat().st_size > MAX_SOURCE_BYTES:
            oversized.append(str(path.relative_to(ROOT)))
        if path.suffix not in text_suffixes:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                findings.append(f"{path.relative_to(ROOT)}: {label}")
    if findings:
        audit.fail("secret/path hygiene findings:\n  " + "\n  ".join(findings))
    else:
        audit.ok("no private keys, AWS access keys, or server-specific workspace paths")
    if oversized:
        audit.fail(f"source files larger than 10 MiB: {oversized}")
    else:
        audit.ok("no oversized source files")


def check_code_only_boundary(audit: Audit) -> None:
    markdown = sorted(str(path.relative_to(ROOT)) for path in files(".md"))
    if markdown == ["README.md"]:
        audit.ok("README.md is the only Markdown file")
    else:
        audit.fail(f"unexpected Markdown files: {markdown}")

    external_suffixes = {
        ".usd", ".usda", ".usdc", ".hdf5", ".h5", ".mp4", ".avi", ".mov",
        ".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz", ".pdf",
    }
    bundled = sorted(
        str(path.relative_to(ROOT)) for path in files() if path.suffix.lower() in external_suffixes
    )
    if bundled:
        audit.fail(f"data/model/paper artifacts are bundled in the code release: {bundled}")
    else:
        audit.ok("no datasets, runtime USDs, checkpoints, videos, arrays, or paper PDFs are bundled")

    removed_refs = []
    archived_requirement = "requirements-" + "openpi.txt"
    for path in files():
        if path.suffix.lower() not in {".py", ".sh", ".md", ".txt", ".toml", ".yml", ".yaml", ".cff"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if archived_requirement in content:
            removed_refs.append(str(path.relative_to(ROOT)))
    if removed_refs:
        audit.fail(f"references to archived {archived_requirement} remain: {removed_refs}")
    else:
        audit.ok("no references to archived public entry points remain")


def check_protocol(audit: Audit) -> None:
    protocol = json.loads((ROOT / "configs/benchmark_protocol_v1.json").read_text(encoding="utf-8"))
    models = protocol.get("release_scope", {}).get("models")
    metrics = set(protocol.get("metrics", {}))
    if models == ["pi05"] and metrics == {"goal_success", "safe_success", "d_peak"}:
        audit.ok("v1 protocol is π0.5-only and exposes Goal/Safe success")
    else:
        audit.fail(f"unexpected public protocol: models={models}, metrics={sorted(metrics)}")
    for launcher in ("openpi/scripts/train_softvtbench.sh", "openpi/scripts/evaluate_softvtbench.sh"):
        content = (ROOT / launcher).read_text(encoding="utf-8")
        if 'MODEL}" != "pi05' not in content:
            audit.fail(f"{launcher} does not enforce MODEL=pi05")


def check_external_gates(audit: Audit, strict: bool) -> None:
    threshold = ROOT / "configs/safety_thresholds.json"
    permission = ROOT / "legal/eval_assets_redistribution_permission.txt"
    missing = []
    threshold_valid = False
    if threshold.is_file():
        try:
            payload = json.loads(threshold.read_text(encoding="utf-8"))
            threshold_valid = (
                payload.get("metric_id") == "fem_rms_rigid_aligned_bbox_pct_v1"
                and payload.get("calibration", {}).get("method") == "compression_sweep"
                and isinstance(payload.get("thresholds"), dict)
                and bool(payload["thresholds"])
            )
        except (OSError, json.JSONDecodeError):
            threshold_valid = False
    if not threshold_valid:
        missing.append("configs/safety_thresholds.json (formal compression sweep)")
    bundled_eval_assets = [
        path.relative_to(ROOT)
        for path in files()
        if path.suffix.lower() in {".usd", ".usda", ".usdc"}
    ]
    notices = ROOT / "THIRD_PARTY_NOTICES"
    external_boundary_declared = notices.is_file() and (
        "Datasets and runtime assets are not included in this repository"
        in notices.read_text(encoding="utf-8")
    )
    if bundled_eval_assets and not permission.is_file():
        missing.append(
            "legal/eval_assets_redistribution_permission.txt "
            f"(required for {len(bundled_eval_assets)} bundled USD asset file(s))"
        )
    elif not bundled_eval_assets and not external_boundary_declared:
        missing.append("THIRD_PARTY_NOTICES external dataset/asset boundary")
    if not missing:
        if bundled_eval_assets:
            audit.ok("thresholds and bundled eval-asset redistribution evidence are present")
        else:
            audit.ok("thresholds are present and datasets/runtime assets remain external")
    elif strict:
        audit.fail("external release gates are open: " + "; ".join(missing))
    else:
        audit.warn("external release gates are open: " + "; ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="fail on external release gates")
    args = parser.parse_args()
    audit = Audit()
    check_required(audit)
    check_parsers(audit)
    check_hygiene(audit)
    check_code_only_boundary(audit)
    check_protocol(audit)
    check_external_gates(audit, args.strict)
    print(f"\nRESULT: {'FAIL' if audit.failures else 'PASS'} "
          f"({len(audit.failures)} failure(s), {len(audit.warnings)} warning(s))")
    return 1 if audit.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
