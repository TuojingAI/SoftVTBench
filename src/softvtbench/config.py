"""Configuration loading shared by CLI entry points and tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def repository_root() -> Path:
    """Return the checkout root; an explicit override supports packaged installs."""
    configured = os.environ.get("SOFTVTBENCH_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    inferred = Path(__file__).resolve().parents[2]
    if not (inferred / "config").is_dir():
        raise RuntimeError(
            "SoftVTBench runtime configs are not beside the installed package; "
            "use an editable checkout or set SOFTVTBENCH_ROOT"
        )
    return inferred


def config_root() -> Path:
    return Path(
        os.environ.get("SOFTVT_CONFIG_DIR", repository_root() / "config")
    ).expanduser().resolve()


def policy_manifest_path() -> Path:
    configured = os.environ.get("SOFTVT_POLICY_CONFIG")
    if configured:
        return Path(configured).expanduser().resolve()
    models_root = os.environ.get("SOFTVT_MODELS_ROOT")
    if not models_root:
        raise RuntimeError(
            "set SOFTVT_POLICY_CONFIG or SOFTVT_MODELS_ROOT; policy/checkpoint "
            "definitions are owned by SoftVTBench-Models"
        )
    return Path(models_root).expanduser().resolve() / "configs" / "policies.yaml"


def load_yaml(path: Path) -> Any:
    """Read YAML after expanding environment variables used by public configs."""
    import yaml

    return yaml.safe_load(os.path.expandvars(path.read_text(encoding="utf-8")))


def load_benchmark_config(relative_path: str) -> Any:
    return load_yaml(config_root() / relative_path)


def load_policy_manifest() -> dict[str, Any]:
    """Merge model-owned loader specs with benchmark-owned execution profiles.

    The split prevents checkpoint/loading concerns from leaking into the benchmark
    while keeping the resolved dictionary identical to what the evaluator consumes.
    """
    manifest = load_yaml(policy_manifest_path())
    profiles = load_benchmark_config("policy_protocols.yaml")["profiles"]
    resolved = []
    for model in manifest["policies"]:
        name = model.get("execution_profile")
        if name not in profiles:
            raise ValueError(
                f"policy {model.get('id')!r} references unknown execution profile {name!r}"
            )
        overlap = set(model) & set(profiles[name])
        if overlap:
            raise ValueError(
                f"policy/profile fields overlap for {model.get('id')!r}: {sorted(overlap)}"
            )
        entry = {**model, **profiles[name]}
        entry.pop("execution_profile", None)
        resolved.append(entry)
    return {"schema": "softvtbench_resolved_policies_v1", "policies": resolved}
