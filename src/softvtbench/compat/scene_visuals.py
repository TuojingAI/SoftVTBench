"""Re-apply the collection-time scene visuals inside the evaluation client.

Every July-2026 dataset (object/spatial x soft/rigid) was collected with two
unconditional scene edits made by the collection expert right after the first
``env.reset()`` in the collection-time scene setup:

1. ``add_textured_floor_overlay`` — a wood-tile textured quad laid over the
   pale physics floor (optionally hiding the physics floor's own visual), and
2. ``apply_reference_lighting`` — all stock lights deactivated and replaced by
   a single dim DomeLight (intensity 135, colour 0.78/0.78/0.76).

Without them the policy sees a bright, white-floored scene it was never
trained on.  This module is loaded by ``openpi_inference_client.py`` when
``SOFTVTBENCH_APPLY_COLLECTION_SCENE_VISUALS=1`` and
``SOFTVTBENCH_SCENE_VISUALS_MODULE`` points here; the code below is a faithful
copy of the collection-side functions with only the env-var prefix renamed.

Environment knobs (defaults match the collection wrappers):
- ``SOFTVTBENCH_FLOOR_OVERLAY_Z``            overlay height (object suites: -0.016)
- ``SOFTVTBENCH_HIDE_PHYSICS_FLOOR_VISUAL``  "1" to hide the pale floor visual
- ``SOFTVTBENCH_FLOOR_TEXTURE``              texture override (default: repo copy)
"""

from __future__ import annotations

import os
from pathlib import Path

import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

_DEFAULT_TEXTURE = (
    Path(__file__).resolve().parents[2]
    / "assets/libero/floor/textures/tile_grigia_caldera_porcelain_floor.png"
)


def _floor_texture() -> Path:
    override = os.environ.get("SOFTVTBENCH_FLOOR_TEXTURE", "").strip()
    return Path(override) if override else _DEFAULT_TEXTURE


def _set_light_attr(light, attr_name: str, value) -> None:
    """Attribute writer compatible with different USD light schemas."""
    attr = light.GetPrim().GetAttribute(attr_name)
    if not attr:
        if isinstance(value, bool):
            attr = light.GetPrim().CreateAttribute(attr_name, Sdf.ValueTypeNames.Bool)
        elif isinstance(value, float):
            attr = light.GetPrim().CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
        else:
            return
    attr.Set(value)


def add_textured_floor_overlay() -> None:
    """Add a realistic floor-texture preview plane, keeping the texture while avoiding the original floor's light/dark tiling."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[scene-visuals] warning: USD stage not ready, skip textured floor overlay", flush=True)
        return

    floor_texture = _floor_texture()
    if not floor_texture.exists():
        print(f"[scene-visuals] warning: floor texture missing: {floor_texture}", flush=True)
        return

    material = UsdShade.Material.Define(stage, "/World/Looks/textured_floor_overlay")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/textured_floor_overlay/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.78)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    uv_reader = UsdShade.Shader.Define(stage, "/World/Looks/textured_floor_overlay/PrimvarReader_st")
    uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
    uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    texture = UsdShade.Shader.Define(stage, "/World/Looks/textured_floor_overlay/DiffuseTexture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(floor_texture)))
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_reader.ConnectableAPI(), "result")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    overlay_path = "/World/envs/env_0/textured_floor_visual_overlay"
    overlay = UsdGeom.Mesh.Define(stage, overlay_path)
    z = float(os.environ.get("SOFTVTBENCH_FLOOR_OVERLAY_Z", "0.002"))
    overlay.CreatePointsAttr([
        Gf.Vec3f(-1.8, -1.8, z),
        Gf.Vec3f(1.8, -1.8, z),
        Gf.Vec3f(1.8, 1.8, z),
        Gf.Vec3f(-1.8, 1.8, z),
    ])
    overlay.CreateFaceVertexCountsAttr([4])
    overlay.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    overlay.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)])
    overlay.SetNormalsInterpolation("constant")
    primvars = UsdGeom.PrimvarsAPI(overlay)
    st = primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set([Gf.Vec2f(0.0, 0.0), Gf.Vec2f(5.0, 0.0), Gf.Vec2f(5.0, 5.0), Gf.Vec2f(0.0, 5.0)])
    UsdShade.MaterialBindingAPI.Apply(overlay.GetPrim()).Bind(material)
    print(f"[scene-visuals] added textured floor overlay: {overlay_path} z={z}", flush=True)
    # Hide the physics floor's own (pale, un-textured) visual so the wood-tile
    # overlay is the only visible floor. Collision on the Floor prim is untouched.
    if os.environ.get("SOFTVTBENCH_HIDE_PHYSICS_FLOOR_VISUAL", "0") == "1":
        for fp in ("/World/envs/env_0/Floor", "/World/envs/env_0/Floor/geometry", "/World/envs/env_0/Floor/visuals"):
            prim = stage.GetPrimAtPath(fp)
            if prim and prim.IsValid():
                try:
                    UsdGeom.Imageable(prim).MakeInvisible()
                    print(f"[scene-visuals] hid physics floor visual: {fp}", flush=True)
                except Exception as e:
                    print(f"[scene-visuals] could not hide {fp}: {e}", flush=True)


def apply_reference_lighting() -> None:
    """Replace the default lighting with uniform, slightly dim studio lighting."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[scene-visuals] warning: USD stage not ready, skip lighting", flush=True)
        return

    light_types = {"DomeLight", "DistantLight", "RectLight", "SphereLight", "DiskLight", "CylinderLight"}
    for prim in stage.Traverse():
        if prim.GetTypeName() in light_types:
            prim.SetActive(False)

    dome = UsdLux.DomeLight.Define(stage, "/World/textured_dim_dome")
    dome.CreateIntensityAttr(135.0)
    dome.CreateColorAttr(Gf.Vec3f(0.78, 0.78, 0.76))
    _set_light_attr(dome, "inputs:diffuse", 1.0)
    _set_light_attr(dome, "inputs:specular", 0.08)
    print("[scene-visuals] applied textured_dim_dome lighting: dome only", flush=True)


def apply_collection_scene_visuals() -> None:
    add_textured_floor_overlay()
    apply_reference_lighting()
