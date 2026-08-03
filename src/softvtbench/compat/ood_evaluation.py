"""Deterministic, paired OOD injection for the existing SoftVTBench rollout.

The adapter is intentionally small: the rollout owns reset, observation,
execution, and metrics; this module only resolves one versioned OOD config,
applies declared USD changes after the ID scene contract has been validated,
and corrupts external RGB observations when requested.
"""

from __future__ import annotations

import hashlib
import json
import os
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SUPPORTED_FACTORS = {
    "gaussian_blur",
    "gaussian_noise",
    "youngs_modulus_softer",
    "youngs_modulus_harder",
    "object_gripper_friction_low",
    "total_mass_heavy",
    "dome_light_intensity",
}

SCHEMA_LEVELS = {
    "softvtbench_ood_v1": 5,
    "softvtbench_ood_v2": 3,
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(f"PyYAML is required to read {path}") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"OOD config must contain a mapping: {path}")
    return value


def load_ood_config_from_env() -> tuple[dict[str, Any] | None, str | None]:
    raw = os.environ.get("SOFTVTBENCH_OOD_CONFIG", "").strip()
    if not raw:
        return None, None
    path = Path(raw).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SOFTVTBENCH_OOD_CONFIG does not exist: {path}")
    config = copy.deepcopy(_load_mapping(path))
    level_override = os.environ.get("SOFTVTBENCH_OOD_LEVEL", "").strip()
    if level_override:
        config.setdefault("ood", {})["level"] = int(level_override)
    validate_ood_config(config)
    return config, hashlib.sha256(_canonical_json(config).encode()).hexdigest()


def _factor_entries(ood: dict[str, Any]) -> list[dict[str, Any]]:
    if ood.get("category") == "compositional":
        factors = ood.get("factors")
        if not isinstance(factors, list) or len(factors) < 2:
            raise ValueError("compositional OOD requires at least two factors")
        return factors
    return [ood]


def validate_ood_config(config: dict[str, Any]) -> None:
    schema = config.get("schema")
    if schema not in SCHEMA_LEVELS:
        raise ValueError(f"OOD config schema must be one of {sorted(SCHEMA_LEVELS)}")
    level_count = SCHEMA_LEVELS[schema]
    if not str(config.get("calibration_version", "")).strip():
        raise ValueError("OOD config requires calibration_version")
    applicable_suites = config.get("applicable_suites")
    if applicable_suites is not None:
        if (
            not isinstance(applicable_suites, list)
            or not applicable_suites
            or any(not isinstance(x, str) or not x.strip() for x in applicable_suites)
        ):
            raise ValueError("applicable_suites must be a non-empty list of suite names")
        if len(applicable_suites) != len(set(applicable_suites)):
            raise ValueError("applicable_suites contains duplicates")
    ood = config.get("ood")
    if not isinstance(ood, dict):
        raise ValueError("OOD config requires an ood mapping")
    if not ood.get("enabled", False):
        if ood.get("split") != "id":
            raise ValueError("disabled OOD config must use split=id")
        return
    level = ood.get("level")
    if not isinstance(level, int) or not 1 <= level <= level_count:
        raise ValueError(
            f"OOD level must be an integer in [1, {level_count}], got {level!r}"
        )
    for entry in _factor_entries(ood):
        factor = entry.get("factor")
        if factor not in SUPPORTED_FACTORS:
            raise ValueError(f"unsupported OOD factor: {factor!r}")
        levels = entry.get("levels")
        expected_levels = {str(x) for x in range(1, level_count + 1)}
        if not isinstance(levels, dict) or set(levels) != expected_levels:
            raise ValueError(f"{factor} must define exactly Levels 1-{level_count}")
        values = []
        for idx in range(1, level_count + 1):
            spec = levels[str(idx)]
            if not isinstance(spec, dict):
                raise ValueError(f"{factor} Level {idx} must be a mapping")
            key = (
                "sigma"
                if factor == "gaussian_blur"
                else "std"
                if factor == "gaussian_noise"
                else "intensity"
                if factor == "dome_light_intensity"
                else "ratio"
            )
            value = spec.get(key)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{factor} Level {idx} requires positive finite {key}")
            values.append(float(value))
        if values != sorted(values) and values != sorted(values, reverse=True):
            raise ValueError(
                f"{factor} Levels 1-{level_count} must be strictly monotonic"
            )
        if len(set(values)) != level_count:
            raise ValueError(f"{factor} Levels 1-{level_count} must be distinct")


def _seed_for_episode(
    config: dict[str, Any],
    *,
    suite: str,
    task_id: int,
    episode_key: str,
) -> int:
    ood = config["ood"]
    payload = (
        f"{config['calibration_version']}|{ood.get('seed', 0)}|{suite}|"
        f"{task_id}|{episode_key}|{ood.get('category')}|{ood.get('level')}"
    )
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def _target_names_from_env() -> list[str]:
    names = []
    direct = os.environ.get("SOFTVTBENCH_EXTRA_ASSET_NAME", "").strip()
    if direct:
        names.append(direct)
    raw = os.environ.get("SOFTVTBENCH_EXTRA_ASSETS_JSON", "").strip()
    if raw:
        for item in json.loads(raw):
            name = str(item.get("name", "")).strip()
            if name and "distractor" not in name:
                names.append(name)
    return list(dict.fromkeys(names))


@dataclass
class OODEpisode:
    config: dict[str, Any]
    config_hash: str
    suite: str
    task_id: int
    episode_key: str
    seed: int = field(init=False)
    resolved_factors: list[dict[str, Any]] = field(init=False)
    actual_parameters: dict[str, Any] = field(default_factory=dict)
    clipped: bool = False

    def __post_init__(self) -> None:
        allowed = self.config.get("applicable_suites")
        if allowed is not None and self.suite not in allowed:
            raise ValueError(
                f"OOD config is not applicable to suite {self.suite!r}; allowed={allowed}"
            )
        self.seed = _seed_for_episode(
            self.config, suite=self.suite, task_id=self.task_id, episode_key=self.episode_key
        )
        ood = self.config["ood"]
        level = int(ood.get("level", 0))
        self.resolved_factors = []
        if ood.get("enabled", False):
            for entry in _factor_entries(ood):
                self.resolved_factors.append(
                    {
                        "category": entry.get("category", ood.get("category")),
                        "factor": entry["factor"],
                        "level": level,
                        "requested": dict(entry["levels"][str(level)]),
                    }
                )

    @property
    def enabled(self) -> bool:
        return bool(self.config["ood"].get("enabled", False))

    @property
    def requires_render(self) -> bool:
        return any(
            item["factor"] == "dome_light_intensity"
            for item in self.resolved_factors
        )

    def apply_physical(self) -> None:
        physical = [
            item
            for item in self.resolved_factors
            if item["factor"] not in {"gaussian_blur", "gaussian_noise"}
        ]
        if not physical:
            return
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        target = None
        for item in physical:
            factor = item["factor"]
            if factor == "dome_light_intensity":
                self.actual_parameters[factor] = self._set_dome_light_intensity(
                    stage, float(item["requested"]["intensity"])
                )
                continue
            if target is None:
                target_names = _target_names_from_env()
                if len(target_names) != 1:
                    raise RuntimeError(
                        "physical OOD requires exactly one non-distractor target asset; "
                        f"resolved {target_names!r}"
                    )
                target_path = f"/World/envs/env_0/{target_names[0]}"
                target = stage.GetPrimAtPath(target_path)
                if not target.IsValid():
                    raise RuntimeError(f"OOD target prim is absent: {target_path}")
            ratio = float(item["requested"]["ratio"])
            if factor.startswith("youngs_modulus_"):
                actual = self._scale_existing_attrs(
                    target, "physxDeformableBodyMaterial:youngsModulus", ratio
                )
            elif factor == "total_mass_heavy":
                actual = self._scale_mass_or_density(target, ratio)
            elif factor == "object_gripper_friction_low":
                actual = self._scale_or_bind_friction(stage, target, ratio, item["requested"])
            else:
                raise AssertionError(factor)
            self.actual_parameters[factor] = actual

    @staticmethod
    def _set_dome_light_intensity(stage, intensity: float) -> dict[str, Any]:
        path = "/World/textured_dim_dome"
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid() or not prim.IsActive():
            raise RuntimeError(f"collection DomeLight is absent or inactive: {path}")
        attr = prim.GetAttribute("inputs:intensity")
        if not attr or not attr.IsValid() or attr.Get() is None:
            raise RuntimeError(f"collection DomeLight intensity is unreadable: {path}")
        baseline = float(attr.Get())
        attr.Set(float(intensity))
        actual = float(attr.Get())
        if abs(actual - float(intensity)) > 1e-4:
            raise RuntimeError(
                f"DomeLight intensity write failed: requested={intensity}, actual={actual}"
            )
        return {
            "prim": path,
            "attribute": "inputs:intensity",
            "baseline": baseline,
            "requested": float(intensity),
            "actual": actual,
        }

    @staticmethod
    def _scale_existing_attrs(target, attr_name: str, ratio: float) -> list[dict[str, Any]]:
        from pxr import Usd

        changed = []
        for prim in Usd.PrimRange(target):
            attr = prim.GetAttribute(attr_name)
            value = attr.Get() if attr else None
            if value is None:
                continue
            baseline = float(value)
            actual = baseline * ratio
            attr.Set(actual)
            readback = float(attr.Get())
            changed.append(
                {
                    "prim": str(prim.GetPath()),
                    "attribute": attr_name,
                    "baseline": baseline,
                    "ratio": ratio,
                    "actual": readback,
                }
            )
        if not changed:
            raise RuntimeError(f"target {target.GetPath()} has no {attr_name}")
        return changed

    @classmethod
    def _scale_mass_or_density(cls, target, ratio: float) -> list[dict[str, Any]]:
        from pxr import Usd

        mass_attrs = []
        for prim in Usd.PrimRange(target):
            mass = prim.GetAttribute("physics:mass")
            density = prim.GetAttribute("physics:density")
            if mass and mass.Get() is not None and float(mass.Get()) > 0:
                mass_attrs.append((prim, "physics:mass"))
            elif density and density.Get() is not None and float(density.Get()) > 0:
                mass_attrs.append((prim, "physics:density"))
            deform_density = prim.GetAttribute("physxDeformableBodyMaterial:density")
            if deform_density and deform_density.Get() is not None and float(deform_density.Get()) > 0:
                mass_attrs.append((prim, "physxDeformableBodyMaterial:density"))
        changed = []
        seen = set()
        for prim, name in mass_attrs:
            key = (str(prim.GetPath()), name)
            if key in seen:
                continue
            seen.add(key)
            attr = prim.GetAttribute(name)
            baseline = float(attr.Get())
            attr.Set(baseline * ratio)
            changed.append(
                {
                    "prim": key[0],
                    "attribute": name,
                    "baseline": baseline,
                    "ratio": ratio,
                    "actual": float(attr.Get()),
                }
            )
        if not changed:
            raise RuntimeError(f"target {target.GetPath()} has no authoritative mass or density")
        return changed

    @staticmethod
    def _scale_or_bind_friction(stage, target, ratio: float, spec: dict[str, Any]) -> list[dict[str, Any]]:
        from pxr import Usd

        deformable = []
        for prim in Usd.PrimRange(target):
            attr = prim.GetAttribute("physxDeformableBodyMaterial:dynamicFriction")
            if attr and attr.Get() is not None:
                baseline = float(attr.Get())
                attr.Set(baseline * ratio)
                deformable.append(
                    {
                        "prim": str(prim.GetPath()),
                        "attribute": attr.GetName(),
                        "baseline": baseline,
                        "ratio": ratio,
                        "actual": float(attr.Get()),
                        "static_friction_supported": False,
                    }
                )
        if deformable:
            return deformable

        from pxr import UsdPhysics, UsdShade

        baseline_static = float(spec.get("reference_static", 0.5))
        baseline_dynamic = float(spec.get("reference_dynamic", 0.5))
        material_path = f"{target.GetPath()}/OODPhysicsMaterial"
        material = UsdShade.Material.Define(stage, material_path)
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr().Set(baseline_static * ratio)
        api.CreateDynamicFrictionAttr().Set(baseline_dynamic * ratio)
        binding = UsdShade.MaterialBindingAPI.Apply(target)
        binding.Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        return [
            {
                "prim": str(target.GetPath()),
                "material": material_path,
                "baseline_static": baseline_static,
                "baseline_dynamic": baseline_dynamic,
                "ratio": ratio,
                "actual_static": float(api.GetStaticFrictionAttr().Get()),
                "actual_dynamic": float(api.GetDynamicFrictionAttr().Get()),
                "binding_relation": str(binding.GetDirectBindingRel("physics").GetTargets()),
            }
        ]

    def corrupt_external_rgb(
        self, image: np.ndarray, wrist_image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        result = []
        for camera_name, source in (("agentview", image), ("wrist", wrist_image)):
            value = np.asarray(source, dtype=np.uint8)
            for item in self.resolved_factors:
                factor = item["factor"]
                if factor == "gaussian_blur":
                    sigma = float(item["requested"]["sigma"])
                    value = cv2.GaussianBlur(value, (0, 0), sigmaX=sigma, sigmaY=sigma)
                    self.actual_parameters[factor] = {
                        "sigma": sigma,
                        "channels": ["agentview", "wrist"],
                        "tactile_corrupted": False,
                    }
                elif factor == "gaussian_noise":
                    std = float(item["requested"]["std"])
                    camera_seed = self.seed ^ int.from_bytes(
                        hashlib.sha256(camera_name.encode()).digest()[:8], "big"
                    )
                    rng = np.random.default_rng(camera_seed)
                    noise = rng.normal(0.0, std * 255.0, size=value.shape)
                    value = np.clip(value.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                    self.actual_parameters[factor] = {
                        "std_0_1": std,
                        "channels": ["agentview", "wrist"],
                        "tactile_corrupted": False,
                    }
            result.append(value)
        return result[0], result[1]

    def receipt(self) -> dict[str, Any]:
        ood = self.config["ood"]
        return {
            "schema": "softvtbench_ood_episode_receipt_v1",
            "enabled": self.enabled,
            "calibration_version": self.config["calibration_version"],
            "config_sha256": self.config_hash,
            "category": ood.get("category"),
            "split": ood.get("split"),
            "level": int(ood.get("level", 0)),
            "severity_name": {
                0: "id",
                1: "mild",
                2: "moderate",
                3: "strong",
                4: "severe",
                5: "extreme",
            }[int(ood.get("level", 0))],
            "suite": self.suite,
            "task_id": self.task_id,
            "episode_key": self.episode_key,
            "seed": self.seed,
            "factors": self.resolved_factors,
            "actual_parameters": self.actual_parameters,
            "clipped": self.clipped,
        }

    def frame_metadata(self) -> dict[str, Any]:
        receipt = self.receipt()
        return {
            "ood_enabled": receipt["enabled"],
            "ood_category": receipt["category"],
            "ood_split": receipt["split"],
            "ood_level": receipt["level"],
            "ood_severity_name": receipt["severity_name"],
            "ood_seed": receipt["seed"],
            "ood_actual_parameters": receipt["actual_parameters"],
            "ood_calibration_version": receipt["calibration_version"],
            "ood_config_sha256": receipt["config_sha256"],
            "ood_clipped": receipt["clipped"],
        }


def make_ood_episode(
    config: dict[str, Any] | None,
    config_hash: str | None,
    *,
    suite: str,
    task_id: int,
    episode_key: str,
) -> OODEpisode | None:
    if config is None:
        return None
    return OODEpisode(
        config=config,
        config_hash=str(config_hash),
        suite=suite,
        task_id=task_id,
        episode_key=episode_key,
    )
