#!/usr/bin/env python3
"""Bake calibrated deformable materials into the eval-assets USD files.

Closed-loop evaluation reads material properties from the downloaded
eval-assets USDs (configs/simulation_physics_v1.json: source=authored_in_eval_usd),
so every soft asset's USD must carry the calibration-bench values. This tool
authors the typed physxDeformableBodyMaterial attributes on the asset's
PhysicsMaterial prim - the same structure the original working assets use
(the simulator applies the material API when it binds the deformable body).

Pastry assets get the full calibrated (E, nu, rho); procedural geometry
(stw_*) keeps its authored elasticity, which is the calibrated source for
those assets. Dynamic friction is the calibration-bench value for every asset.

Usage: bake_eval_asset_materials.py --usd-root /path/to/eval-assets/USD [--dry-run]
Requires any python with pxr (USD).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Sdf, Usd

CALIB_FRICTION = 1.5
CALIB_MATERIALS = {
    "pastry001": (60e3, 0.35, 240.0),
    "pastry002": (25e3, 0.30, 180.0),
    "pastry003": (75e3, 0.36, 280.0),
    "pastry004": (35e3, 0.32, 200.0),
    "pastry005": (110e3, 0.38, 320.0),
    "pastry006": (45e3, 0.34, 220.0),
    "pastry007": (160e3, 0.36, 260.0),
    "pastry008": (130e3, 0.38, 340.0),
    "pastry009": (90e3, 0.37, 300.0),
    "pastry010": (220e3, 0.36, 380.0),
    "pastry011": (180e3, 0.37, 350.0),
}
GEOMETRY_ONLY_FRICTION = ("stw_cube_hq", "stw_cylinder_hq", "stw_sphere_hq")


def material_prims(stage):
    for prim in stage.Traverse():
        has_attr = any("physxDeformableBodyMaterial" in a.GetName()
                       for a in prim.GetAttributes())
        if has_attr or prim.GetName() == "PhysicsMaterial":
            yield prim


API_NAME = "PhysxDeformableBodyMaterialAPI"
ATTR_PREFIX = "physxDeformableBodyMaterial"


def _set(prim, suffix: str, value: float) -> None:
    attr = prim.GetAttribute(f"{ATTR_PREFIX}:{suffix}")
    if not attr or not attr.IsValid():
        attr = prim.CreateAttribute(f"{ATTR_PREFIX}:{suffix}", Sdf.ValueTypeNames.Float)
    attr.Set(float(value))


def bake(usd_path: Path, elasticity, friction: float, dry_run: bool) -> list[str]:
    stage = Usd.Stage.Open(str(usd_path))
    touched = []
    for prim in material_prims(stage):
        if API_NAME not in prim.GetAppliedSchemas():
            prim.AddAppliedSchema(API_NAME)
        if elasticity is not None:
            youngs, poisson, density = elasticity
            _set(prim, "youngsModulus", youngs)
            _set(prim, "poissonsRatio", poisson)
            _set(prim, "density", density)
        _set(prim, "dynamicFriction", friction)
        touched.append(str(prim.GetPath()))
    if touched and not dry_run:
        stage.GetRootLayer().Save()
    return touched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for asset, elasticity in sorted(CALIB_MATERIALS.items()):
        usd = args.usd_root / asset / f"{asset}.usd"
        if not usd.exists():
            print(f"[skip] {asset}: no {usd.name}")
            continue
        touched = bake(usd, elasticity, CALIB_FRICTION, args.dry_run)
        print(f"[bake] {asset}: E={elasticity[0]:.0f} nu={elasticity[1]} "
              f"rho={elasticity[2]:.0f} mu={CALIB_FRICTION} prims={touched}")
    for asset in GEOMETRY_ONLY_FRICTION:
        usd = args.usd_root / asset / f"model_{asset}.usd"
        if not usd.exists():
            usd = args.usd_root / asset / f"{asset}.usd"
        if not usd.exists():
            print(f"[skip] {asset}: no usd found")
            continue
        touched = bake(usd, None, CALIB_FRICTION, args.dry_run)
        print(f"[bake] {asset}: authored elasticity kept, mu={CALIB_FRICTION} "
              f"prims={touched}")


if __name__ == "__main__":
    main()
