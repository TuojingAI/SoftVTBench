#!/usr/bin/env python3
"""Audit SoftVTBench and, when supplied, its companion model repository."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from softvtbench.quality import (  # noqa: E402
    cross_repository_duplicates,
    digest,
    duplicate_groups,
    files_under,
    generated_artifacts,
    private_path_hits,
    syntax_errors,
)


FORMAL_SUFFIXES = {
    "pi05_full_vo_c",
    "pi05_full_vt_c",
    "pi05_vo_c",
    "pi05_vt_c",
    "dp_vo_c",
    "dp_vt_c",
    "fastwam_vo_c",
    "fastwam_vt_c",
}


class Audit:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def require(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.failures.append(message)


def load_yaml(path: Path) -> Any:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def verify_manifest(root: Path) -> list[str]:
    manifest_path = root / "MANIFEST.sha256"
    if not manifest_path.is_file():
        return [f"missing checksum manifest: {manifest_path}"]
    declared: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        checksum, relative = line.split("  ", 1)
        declared[relative] = checksum
    expected = {
        str(path.relative_to(root)): digest(path)
        for path in files_under(root)
        if path != manifest_path
    }
    errors = []
    if declared.keys() != expected.keys():
        errors.append(
            f"manifest path set differs: missing={sorted(expected.keys() - declared.keys())}, "
            f"extra={sorted(declared.keys() - expected.keys())}"
        )
    for relative in declared.keys() & expected.keys():
        if declared[relative] != expected[relative]:
            errors.append(f"checksum mismatch: {relative}")
    return errors


def git_tracking_errors(root: Path) -> list[str]:
    """Reject files that exist locally but would disappear from a fresh clone."""
    inside = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        text=True,
        capture_output=True,
        check=False,
    )
    if inside.returncode:
        return []
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        check=True,
    )
    tracked = {
        item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    }
    present = {str(path.relative_to(root)) for path in files_under(root)}
    errors = []
    if untracked := sorted(present - tracked):
        errors.append(f"files are present but not Git-tracked: {untracked}")
    if missing := sorted(tracked - present):
        errors.append(f"Git-tracked files are missing from the worktree: {missing}")
    return errors


def audit_benchmark(audit: Audit) -> None:
    required = [
        "LICENSE",
        "README.md",
        "MANIFEST.sha256",
        "release.json",
        "config/policy_protocols.yaml",
        "config/ood/formal_n50/conditions_9.txt",
        "scripts/eval_stage.sh",
        "src/softvtbench/evaluation/runner.py",
    ]
    for relative in required:
        audit.require((ROOT / relative).is_file(), f"missing benchmark file: {relative}")

    manifest_errors = verify_manifest(ROOT)
    audit.require(not manifest_errors, "benchmark manifest errors:\n" + "\n".join(manifest_errors))
    tracking_errors = git_tracking_errors(ROOT)
    audit.require(not tracking_errors, "benchmark Git tracking errors:\n" + "\n".join(tracking_errors))

    duplicates = duplicate_groups(ROOT)
    audit.require(not duplicates, f"benchmark has exact duplicates: {duplicates}")
    artifacts = generated_artifacts(ROOT)
    audit.require(not artifacts, f"benchmark has generated/temporary artifacts: {artifacts}")
    errors = syntax_errors(ROOT)
    audit.require(not errors, "benchmark syntax/config errors:\n" + "\n".join(errors))

    scan_roots = [ROOT / name for name in ("src", "config", "scripts", "source/tac_manip")]
    hits = private_path_hits(path for base in scan_roots for path in files_under(base))
    audit.require(not hits, "benchmark contains private absolute paths:\n" + "\n".join(hits))

    oversized = [path for path in files_under(ROOT) if path.stat().st_size >= 100 * 1024 * 1024]
    audit.require(not oversized, f"files exceed GitHub's 100 MiB limit: {oversized}")

    referenced_objects: set[str] = set()
    referenced_scene_params: set[Path] = set()
    for suite_name in ("object_rigid", "spatial_rigid", "object_soft", "spatial_soft"):
        suite = load_yaml(ROOT / "config" / "suites" / f"{suite_name}.yaml")
        task_ids = [task["id"] for task in suite["tasks"]]
        audit.require(suite["name"] == suite_name, f"suite name mismatch: {suite_name}")
        audit.require(task_ids == list(range(10)), f"{suite_name} must define ordered task IDs 0..9")
        referenced_objects.update(task["object"] for task in suite["tasks"])
        libero_config = (
            ROOT
            / "config/libero_configs"
            / suite_name
            / f"{suite['libero_suite']}.json"
        )
        audit.require(libero_config.is_file(), f"missing LIBERO config: {libero_config}")
        if suite.get("scene_params_file"):
            referenced_scene_params.add(
                ROOT / "config/libero_configs" / suite_name / "scene_params.json"
            )

    object_cards = {path.stem for path in (ROOT / "config/objects").glob("*.yaml")}
    audit.require(
        object_cards == referenced_objects,
        "object cards and suite references differ: "
        f"unreferenced={sorted(object_cards - referenced_objects)}, "
        f"missing={sorted(referenced_objects - object_cards)}",
    )
    scene_params = set((ROOT / "config/libero_configs").glob("*/scene_params.json"))
    audit.require(
        scene_params == referenced_scene_params,
        "scene parameter files and suite references differ: "
        f"unreferenced={sorted(scene_params - referenced_scene_params)}, "
        f"missing={sorted(referenced_scene_params - scene_params)}",
    )
    audit.require(
        not (ROOT / "config/controllers.yaml").exists(),
        "legacy N=20 debug controller config must not be restored",
    )

    condition_lines = [
        line.split()
        for line in (ROOT / "config/ood/formal_n50/conditions_9.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    labels = [parts[0] for parts in condition_lines]
    audit.require(len(labels) == 9 and len(labels) == len(set(labels)), "formal OOD list must have 9 unique labels")
    for parts in condition_lines:
        target = parts[1].replace("${SOFTVTBENCH_ROOT}", str(ROOT))
        audit.require(Path(target).is_file(), f"OOD condition target is missing: {target}")

    release = json.loads((ROOT / "release.json").read_text())
    audit.require(
        release.get("formal_evaluation_source_commit") == "ed472c477df92ccc456cb6426621f3974221d6ac",
        "formal source commit changed or is missing",
    )
    audit.require(not (ROOT / "backends").exists(), "model backends must not live in SoftVTBench")


def audit_models(audit: Audit, models: Path) -> None:
    required = [
        "LICENSE",
        "README.md",
        "MANIFEST.sha256",
        "release.json",
        "configs/policies.yaml",
        "src/softvtbench_models/worker.py",
        "backends/openpi/LICENSE",
        "backends/act/LICENSE",
        "backends/diffusion_policy/LICENSE",
        "backends/fastwam/LICENSE",
        "backends/fastwam/converters/softvtbench_to_lerobot.py",
    ]
    for relative in required:
        audit.require((models / relative).is_file(), f"missing models file: {relative}")

    manifest_errors = verify_manifest(models)
    audit.require(not manifest_errors, "models manifest errors:\n" + "\n".join(manifest_errors))
    tracking_errors = git_tracking_errors(models)
    audit.require(not tracking_errors, "models Git tracking errors:\n" + "\n".join(tracking_errors))

    duplicates = duplicate_groups(models)
    audit.require(not duplicates, f"models repository has exact duplicates: {duplicates}")
    artifacts = generated_artifacts(models)
    audit.require(not artifacts, f"models repository has generated/temporary artifacts: {artifacts}")
    errors = syntax_errors(models)
    audit.require(not errors, "models syntax/config errors:\n" + "\n".join(errors))

    scan_roots = [models / name for name in ("src", "configs", "scripts", "backends")]
    hits = private_path_hits(path for base in scan_roots for path in files_under(base))
    audit.require(not hits, "models repository contains private absolute paths:\n" + "\n".join(hits))

    manifest = load_yaml(models / "configs/policies.yaml")
    policies = manifest.get("policies", [])
    ids = [policy.get("id") for policy in policies]
    expected = {f"{suite}/{suffix}" for suite in ("object_soft", "spatial_soft") for suffix in FORMAL_SUFFIXES}
    audit.require(len(policies) == 16 and set(ids) == expected, "model registry is not the formal 16-policy matrix")
    audit.require(len(ids) == len(set(ids)), "model registry contains duplicate IDs")

    profiles = load_yaml(ROOT / "config/policy_protocols.yaml")["profiles"]
    backend_fields = {
        "openpi": {"checkpoint", "openpi_config"},
        "diffusion": {"ckpt_path", "execution"},
        "fastwam": {"ckpt_path", "stats_path", "text_cache_dir"},
    }
    for policy in policies:
        backend = policy.get("backend")
        audit.require(backend in backend_fields, f"unknown formal backend: {backend}")
        if backend in backend_fields:
            audit.require(
                backend_fields[backend] <= policy.keys(),
                f"{policy.get('id')} lacks required {backend} loader fields",
            )
        audit.require(policy.get("execution_profile") in profiles, f"unknown execution profile in {policy.get('id')}")
        path_values = [
            value
            for key, value in policy.items()
            if key.endswith("path") or key.endswith("_dir") or key == "checkpoint"
        ]
        audit.require(
            all(str(value).startswith("${SOFTVT_CHECKPOINT_ROOT}") for value in path_values),
            f"checkpoint path is not rooted by SOFTVT_CHECKPOINT_ROOT in {policy.get('id')}",
        )

    fastwam_data = models / "backends/fastwam/configs/data"
    expected_fastwam = {
        f"softvt_{suite}_contgrip_{mode}_{'2cam' if mode == 'vision' else '3mot_marker3'}.yaml"
        for suite in ("object_rigid", "spatial_rigid", "object_soft", "spatial_soft")
        for mode in ("vision", "tactile")
    }
    audit.require(
        expected_fastwam <= {path.name for path in fastwam_data.glob("*.yaml")},
        "one or more launcher-derived FastWAM data configs are missing",
    )
    fastwam_wrapper = (
        models / "backends/fastwam/scripts/preprocess_softvt_contgrip_fastwam.py"
    ).read_text()
    audit.require(
        fastwam_wrapper.count("converters/softvtbench_to_lerobot.py") == 1,
        "FastWAM preprocessing wrapper must reference the canonical converter exactly once",
    )

    act_registry = json.loads((models / "backends/act/SIM_TASK_CONFIGS.json").read_text())
    audit.require(len(act_registry) == 12, "ACT registry must contain only the 12 released SoftVTBench datasets")
    openpi_config = (models / "backends/openpi/src/openpi/training/config.py").read_text()
    for name in (
        "pi05_full_vision_softvtbench",
        "pi05_full_tacall_softvtbench",
        "pi05_lora_vision_softvtbench",
        "pi05_lora_tacall_softvtbench",
    ):
        audit.require(f'name="{name}"' in openpi_config, f"missing OpenPI config: {name}")

    for forbidden in ("metrics.py", "rollout.py", "ood", "suites"):
        matches = [path for path in files_under(models) if path.name == forbidden]
        audit.require(not matches, f"benchmark-owned content found in models repository: {matches}")

    cross = cross_repository_duplicates(ROOT, models)
    audit.require(not cross, f"non-legal files are duplicated across repositories: {cross}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, help="path to the SoftVTBench-Models checkout")
    args = parser.parse_args()
    audit = Audit()
    audit_benchmark(audit)
    models = args.models_root or (ROOT.parent / "SoftVTBench-Models")
    if models.exists():
        audit_models(audit, models.resolve())
    elif args.models_root:
        audit.failures.append(f"models repository does not exist: {models}")

    if audit.failures:
        print(f"FAILED: {len(audit.failures)} issue(s) across {audit.checks} checks", file=sys.stderr)
        for failure in audit.failures:
            print(f"\n- {failure}", file=sys.stderr)
        raise SystemExit(1)
    scope = "benchmark + models" if models.exists() else "benchmark"
    print(
        f"OK: {scope}; {audit.checks} checks; "
        "no non-empty non-license duplicate files"
    )


if __name__ == "__main__":
    main()
