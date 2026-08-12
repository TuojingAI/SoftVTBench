"""Runtime scene receipt and fail-closed collection-contract checks.

The receipt records facts read back from the live USD stage after the selected
episode's ``initial_state`` has been restored.  A receipt is not merely a log:
``assert_contract`` must reject missing as well as mismatching evidence before
policy inference starts.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from softvtbench.evaluation.gripper_execution import (
    FIXED_POSITION_GRIPPER_MODES,
    RELATIVE_GRIPPER_MODES,
)

_PHYS_KEYS = (
    "youngsModulus", "poissonsRatio", "density", "mass",
    "staticFriction", "dynamicFriction", "dampingScale",
    "elasticityDamping", "simulationHexahedralResolution",
    "solverPositionIterationCount", "vertexVelocityDamping",
    "contactOffset", "restOffset", "maxDepenetrationVelocity",
    "collisionEnabled", "approximation", "enableCCD",
)
_SCHEMA = "softvtbench_scene_receipt_v3_fail_closed"
_EPISODE_SCHEMA = "softvtbench_episode_receipt_v1_fail_closed"


def obj_is_deformable(task: dict, suite: dict) -> bool:
    """Whether the configured target is a FEM body.

    Object cards are not passed to scene_receipt; suite identity is the
    generated source of truth here.  Rigid tasks now also carry asset_name for
    auditing, so asset_name presence alone must never trigger FEM access.
    """
    del task
    return bool(suite.get("name", "").endswith("_soft"))


def file_digest(path: str | Path, limit_mb: int = 64) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    out = {"path": str(p), "exists": True, "size": st.st_size, "mtime": int(st.st_mtime)}
    if p.is_file() and st.st_size <= limit_mb * 1 << 20:
        out["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()[:32]
    return out


def _read_physics(stage, prim_path: str) -> dict:
    out: dict[str, object] = {}
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        return {"error": f"prim not found: {prim_path}"}
    stack = [root]
    while stack:
        prim = stack.pop()
        for attr in prim.GetAttributes():
            key = attr.GetName().split(":")[-1]
            if key in _PHYS_KEYS:
                value = attr.Get()
                if value is not None:
                    if isinstance(value, bool):
                        parsed = value
                    elif isinstance(value, str):
                        parsed = value
                    else:
                        try:
                            parsed = float(value)
                        except (TypeError, ValueError):
                            parsed = str(value)
                    out.setdefault(key, parsed)
        stack.extend(prim.GetChildren())
    return out


def _prim_sources(stage, prim_path: str) -> list[str]:
    """Composition-layer identities contributing the target subtree."""
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        return []
    sources = set()
    stack = [root]
    while stack:
        prim = stack.pop()
        for spec in prim.GetPrimStack():
            ident = str(spec.layer.identifier)
            if ident and not ident.startswith("anon:"):
                sources.add(str(Path(ident).resolve()))
        stack.extend(prim.GetChildren())
    return sorted(sources)


def _as_vec(value) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.replace(",", " ").split()
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return None


def _prim_fact(stage, prim_path: str) -> dict:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return {"present": False}
    out = {"present": True, "active": bool(prim.IsActive())}
    for key, attr_name in (
        ("scale", "xformOp:scale"),
        ("position", "xformOp:translate"),
        ("orientation", "xformOp:orient"),
    ):
        attr = prim.GetAttribute(attr_name)
        value = attr.Get() if attr and attr.IsValid() else None
        if value is None:
            out[key] = None
        elif key == "orientation" and hasattr(value, "GetReal"):
            imag = value.GetImaginary()
            out[key] = [float(value.GetReal()), *(float(x) for x in imag)]
        else:
            out[key] = _as_vec(value)
    try:
        from pxr import UsdGeom
        out["visibility"] = str(UsdGeom.Imageable(prim).ComputeVisibility())
    except Exception:
        out["visibility"] = None
    return out


def _tiny_prims(stage, env_root: str) -> dict:
    out: dict[str, list[float]] = {}
    root = stage.GetPrimAtPath(env_root)
    if not root or not root.IsValid():
        return out
    for prim in root.GetChildren():
        value = prim.GetAttribute("xformOp:scale").Get()
        scale = _as_vec(value)
        if scale is not None and scale and max(abs(x) for x in scale) < 1e-3:
            out[prim.GetName()] = scale
    return out


def _attr_float(prim, *names: str) -> float | None:
    for name in names:
        attr = prim.GetAttribute(name)
        value = attr.Get() if attr and attr.IsValid() else None
        if value is not None:
            return float(value)
    return None


def _dome_light(stage) -> dict:
    prim = stage.GetPrimAtPath("/World/textured_dim_dome")
    if not prim or not prim.IsValid():
        return {"present": False}
    color_attr = prim.GetAttribute("inputs:color")
    color = color_attr.Get() if color_attr and color_attr.IsValid() else None
    return {
        "present": True,
        "active": bool(prim.IsActive()),
        "intensity": _attr_float(prim, "inputs:intensity", "intensity"),
        "diffuse": _attr_float(prim, "inputs:diffuse", "diffuse"),
        "specular": _attr_float(prim, "inputs:specular", "specular"),
        "color": _as_vec(color),
    }


def _active_lights(stage) -> list[dict]:
    out = []
    for prim in stage.Traverse():
        if not prim.GetTypeName().endswith("Light"):
            continue
        out.append({
            "path": str(prim.GetPath()),
            "type": prim.GetTypeName(),
            "intensity": _attr_float(prim, "inputs:intensity", "intensity"),
        })
    return sorted(out, key=lambda x: x["path"])


def _floor_fact(stage, suite: dict) -> dict:
    overlay = _prim_fact(stage, "/World/envs/env_0/textured_floor_visual_overlay")
    prim = stage.GetPrimAtPath("/World/envs/env_0/textured_floor_visual_overlay")
    points = prim.GetAttribute("points").Get() if prim and prim.IsValid() else None
    zs = sorted({round(float(p[2]), 8) for p in points}) if points else []
    overlay["mesh_z_values"] = zs
    floor = _prim_fact(stage, "/World/envs/env_0/Floor")
    return {
        "overlay": overlay,
        "physics_floor": floor,
        "configured_overlay_z": float(suite.get("floor_overlay_z", 0.002)),
        "hide_physics_floor": bool(suite.get("hide_physics_floor", False)),
    }


def _cameras_actual(env) -> dict:
    out: dict[str, dict] = {}
    try:
        for name, sensor in env.unwrapped.scene.sensors.items():
            cfg = getattr(sensor, "cfg", None)
            if cfg is not None and hasattr(cfg, "width"):
                offset = getattr(cfg, "offset", None)
                out[name] = {
                    "width": int(cfg.width),
                    "height": int(cfg.height),
                    "offset_pos": _as_vec(getattr(offset, "pos", None)),
                    "offset_rot": _as_vec(getattr(offset, "rot", None)),
                    "offset_convention": getattr(offset, "convention", None),
                }
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _camera_frame_stats(env) -> dict:
    """Post-warmup image evidence; no image bytes are embedded in receipts."""
    out = {}
    try:
        import numpy as np
        for name, sensor in env.unwrapped.scene.sensors.items():
            rgb = getattr(sensor.data, "output", {}).get("rgb")
            if rgb is None:
                continue
            value = rgb[0].detach().cpu().numpy() if hasattr(rgb[0], "detach") else np.asarray(rgb[0])
            value = np.asarray(value, dtype=np.float32)
            if value.max() <= 1.0 + 1e-6:
                value *= 255.0
            out[name] = {
                "mean_rgb": [round(float(x), 4) for x in value[..., :3].mean(axis=(0, 1))],
                "saturated_fraction": round(float((value[..., :3] >= 254.5).mean()), 8),
                "black_fraction": round(float((value[..., :3] <= 0.5).mean()), 8),
            }
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _simulator_fact(env) -> dict:
    import importlib.metadata
    import sys

    def installed_version(distribution: str) -> str | None:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    cfg = env.unwrapped.cfg
    sim = cfg.sim
    physx = getattr(sim, "physx", None)
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "isaac_sim": installed_version("isaacsim"),
        "isaac_lab": installed_version("isaaclab"),
        "dt": float(sim.dt),
        "decimation": int(cfg.decimation),
        "physics_hz": round(1.0 / float(sim.dt), 8),
        "control_hz": round(1.0 / (float(sim.dt) * int(cfg.decimation)), 8),
        "enable_ccd": bool(getattr(physx, "enable_ccd", False)),
        "first_render_had_collection_visuals": bool(
            getattr(env, "_softvt_first_render_had_visuals", False)
        ),
    }


def _gripper_joint_limits(env) -> dict:
    try:
        robot = env.unwrapped.scene["robot"]
        ids, names = robot.find_joints("panda_finger_joint.*")
        limits = robot.data.joint_pos_limits[0, ids].detach().cpu().numpy()
        return {
            "names": list(names),
            "lower": [float(x) for x in limits[:, 0]],
            "upper": [float(x) for x in limits[:, 1]],
        }
    except Exception as exc:
        return {"error": str(exc)}


def _initial_state_readback(env, h5file, episode_name: str, removed_sources: list[str]) -> dict:
    """Compare reset_to live state with the exact HDF5 reset payload."""
    import numpy as np
    from softvtbench.evaluation import metrics

    root = h5file[f"data/{episode_name}/initial_state"]
    scene = env.unwrapped.scene
    removed = set(removed_sources)
    checks: dict[str, dict] = {}
    missing = []

    def arr(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    def compare(name: str, actual, expected):
        a = np.asarray(arr(actual), dtype=np.float64).reshape(-1)
        e = np.asarray(expected, dtype=np.float64).reshape(-1)
        if a.shape != e.shape:
            checks[name] = {"shape_actual": list(a.shape), "shape_expected": list(e.shape)}
            return
        checks[name] = {
            "shape": list(a.shape),
            "max_abs_error": float(np.max(np.abs(a - e))) if a.size else 0.0,
        }

    art = root.get("articulation")
    if art is not None and "robot" in art:
        robot_h5 = art["robot"]
        robot = scene["robot"]
        compare("articulation/robot/joint_position",
                robot.data.joint_pos[0], robot_h5["joint_position"][0])
        compare("articulation/robot/joint_velocity",
                robot.data.joint_vel[0], robot_h5["joint_velocity"][0])

    rigid = root.get("rigid_object")
    if rigid is not None:
        for name, group in rigid.items():
            if name in removed:
                continue
            try:
                state = scene[name].data.root_state_w[0]
                compare(f"rigid_object/{name}/root_pose", state[:7], group["root_pose"][0])
                compare(f"rigid_object/{name}/root_velocity", state[7:13], group["root_velocity"][0])
            except Exception as exc:
                missing.append(f"rigid_object/{name}: {exc}")

    deform = root.get("deformable_object")
    if deform is not None:
        for name, group in deform.items():
            if "nodal_pos_w" not in group:
                continue
            actual = metrics.get_nodal_pos(env, name)
            if actual is None:
                missing.append(f"deformable_object/{name}: live nodes missing")
            else:
                compare(f"deformable_object/{name}/nodal_pos_w", actual, group["nodal_pos_w"][0])
    return {"checks": checks, "missing": missing}


def _fem_nodal_count(env, asset_name: str) -> int | None:
    from softvtbench.evaluation import metrics
    nodes = metrics.get_nodal_pos(env, asset_name)
    return int(nodes.shape[0]) if nodes is not None else None


def _scene_task(suite: dict, task_id: int) -> dict:
    path = Path(suite["libero_config_dir"]) / f"{suite['libero_suite']}.json"
    data = json.loads(path.read_text())
    tasks = data["tasks"] if isinstance(data, dict) else data
    matches = [x for x in tasks if int(x.get("task_id", -1)) == int(task_id)]
    if len(matches) != 1:
        raise RuntimeError(f"scene config {path}: expected exactly one task{task_id}, got {len(matches)}")
    return matches[0]


def _extra_assets(task: dict) -> list[dict]:
    if task.get("extra_assets_inline"):
        value = json.loads(task["extra_assets_inline"])
    elif task.get("extra_assets_file"):
        value = json.loads(Path(task["extra_assets_file"]).read_text())
    else:
        value = []
    if not isinstance(value, list):
        raise RuntimeError("extra assets must be a JSON list")
    return value


def _removed_names(task: dict) -> list[str]:
    value = task.get("removed_source_assets")
    if not value:
        return []
    if isinstance(value, str):
        return [x for x in value.replace(",", " ").split() if x]
    return [str(x) for x in value]


def _expected_scene(suite: dict, task: dict) -> dict:
    cfg = _scene_task(suite, int(task["id"]))
    objects = cfg.get("objects") or {}
    fixtures = cfg.get("fixtures") or {}
    extras = _extra_assets(task)
    removed = _removed_names(task)

    target_name = task.get("asset_name")
    if not target_name:
        physical = [
            x for x in extras
            if (x.get("rigid") or x.get("deformable"))
            and "distractor" not in str(x.get("name", "")).lower()
        ]
        if len(physical) != 1:
            raise RuntimeError(
                f"task{task['id']}: cannot resolve one physical target from extra assets: "
                f"{[x.get('name') for x in physical]}"
            )
        target_name = physical[0]["name"]

    fixture_prims = []
    for name, spec in fixtures.items():
        typ = str(spec.get("type", name)).lower()
        fixture_prims.append("Floor" if typ == "floor" else "Table")

    expected = set(objects) | set(fixture_prims)
    expected |= {"Robot", "agentview_camera"}
    if suite.get("apply_scene_visuals", False):
        expected.add("textured_floor_visual_overlay")
    expected |= {str(x["name"]) for x in extras}
    expected.add(str(target_name))
    expected -= set(removed)

    hidden = {
        name: _as_vec(spec.get("scale"))
        for name, spec in objects.items()
        if (_as_vec(spec.get("scale")) or [1.0]) and
        max(abs(x) for x in (_as_vec(spec.get("scale")) or [1.0])) < 1e-3
    }
    target_scale = _as_vec(task.get("asset_scale"))
    if target_scale is None:
        match = [x for x in extras if x.get("name") == target_name]
        target_scale = _as_vec(match[0].get("scale")) if match else None

    return {
        "scene_config_language": cfg.get("language_instruction"),
        "expected_prims": sorted(expected),
        "removed_sources": removed,
        "hidden_sources": hidden,
        "extra_assets": extras,
        "target_name": str(target_name),
        "target_scale": target_scale,
    }


def _close(got, want, *, rtol=1e-4, atol=1e-5) -> bool:
    if got is None or want is None or len(got) != len(want):
        return False
    return all(abs(float(a) - float(b)) <= max(atol, abs(float(b)) * rtol)
               for a, b in zip(got, want))


def _quat_close(got, want, *, atol=1e-4) -> bool:
    if got is None or want is None or len(got) != 4 or len(want) != 4:
        return False
    ng = sum(float(x) ** 2 for x in got) ** 0.5
    nw = sum(float(x) ** 2 for x in want) ** 0.5
    if ng == 0 or nw == 0:
        return False
    dot = abs(sum(float(a) * float(b) for a, b in zip(got, want)) / (ng * nw))
    return abs(1.0 - dot) <= atol


def _json_scalar(value):
    """Convert HDF5/numpy scalar metadata to stable JSON values."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _initial_state_fact(group) -> dict:
    """Hash the exact per-demo reset payload and record its schema."""
    digest = hashlib.sha256()
    datasets: dict[str, dict] = {}

    def visit(name, value):
        if not hasattr(value, "shape"):
            return
        import numpy as np
        array = np.asarray(value[()])
        dtype = str(array.dtype)
        shape = [int(x) for x in array.shape]
        raw = array.tobytes(order="C")
        digest.update(name.encode())
        digest.update(dtype.encode())
        digest.update(json.dumps(shape, separators=(",", ":")).encode())
        digest.update(raw)
        datasets[name] = {
            "dtype": dtype,
            "shape": shape,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    group.visititems(visit)
    return {
        "present": True,
        "datasets": datasets,
        "dataset_count": len(datasets),
        "sha256": digest.hexdigest(),
    }


def _hdf5_episode_fact(h5file, episode_name: str) -> dict:
    key = f"data/{episode_name}"
    if key not in h5file:
        return {"present": False, "episode": episode_name}
    group = h5file[key]
    attrs = {str(k): _json_scalar(v) for k, v in group.attrs.items()}
    state = group.get("initial_state")
    return {
        "present": True,
        "episode": episode_name,
        "attrs": attrs,
        "initial_state": (
            _initial_state_fact(state)
            if state is not None else {"present": False, "datasets": {}, "dataset_count": 0}
        ),
    }


def episode_receipt(
    env,
    *,
    suite: dict,
    task: dict,
    policy_entry: dict,
    h5file,
    episode_name: str,
    condition: str,
    episode_seed: int,
    static_receipt: dict,
    ood=None,
    grip_width: float | None = None,
    gripper_constraint_mode: str,
    scene_params_entry: dict | None = None,
) -> dict:
    """Read back one exact demo/condition after reset and before inference."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    env_root = "/World/envs/env_0"
    expected = _expected_scene(suite, task)
    target_name = expected["target_name"]
    target = _prim_fact(stage, f"{env_root}/{target_name}")
    target.update({
        "name": target_name,
        "configured_scale": expected["target_scale"],
        "physics": _read_physics(stage, f"{env_root}/{target_name}"),
        "composition_sources": _prim_sources(stage, f"{env_root}/{target_name}"),
    })
    extras = {}
    for asset in expected["extra_assets"]:
        name = str(asset["name"])
        fact = _prim_fact(stage, f"{env_root}/{name}")
        fact.update({
            "configured_position": _as_vec(asset.get("pos")),
            "configured_orientation": _as_vec(asset.get("rot")),
            "configured_scale": _as_vec(asset.get("scale")),
            "rigid": bool(asset.get("rigid")),
            "deformable": bool(asset.get("deformable")),
        })
        extras[name] = fact

    hdf = _hdf5_episode_fact(h5file, episode_name)
    source = file_digest(h5file.filename, limit_mb=0)
    source["episode_group"] = hdf
    ood_fact = ood.receipt() if ood is not None else None
    scene_params = dict(scene_params_entry or {})
    if "extra_assets_json" in scene_params:
        scene_params["extra_assets_json_sha256"] = hashlib.sha256(
            scene_params["extra_assets_json"].encode()
        ).hexdigest()
        scene_params.pop("extra_assets_json")

    return {
        "schema": _EPISODE_SCHEMA,
        "suite": suite["name"],
        "task_id": int(task["id"]),
        "episode": episode_name,
        "condition": condition,
        "condition_kind": (
            "ood" if ood is not None and ood.enabled else "id"
        ),
        "episode_seed": int(episode_seed),
        "language": task.get("language"),
        "grip_width_m": grip_width,
        "gripper_constraint_mode": gripper_constraint_mode,
        "release": {
            "commit": os.environ.get("SOFTVT_RELEASE_COMMIT", ""),
            "dirty": os.environ.get("SOFTVT_RELEASE_DIRTY", ""),
        },
        "static_contract_passed": bool((static_receipt.get("contract") or {}).get("passed")),
        "hdf5": source,
        "scene_params": scene_params,
        "target_asset": target,
        "extra_assets": extras,
        "dome_light": _dome_light(stage),
        "simulator": _simulator_fact(env),
        "camera_frame_stats": _camera_frame_stats(env),
        "gripper_joint_limits": _gripper_joint_limits(env),
        "initial_state_readback": _initial_state_readback(
            env, h5file, episode_name, expected["removed_sources"]
        ),
        "ood": ood_fact,
        "policy": {
            "id": policy_entry.get("id"),
            "backend": policy_entry.get("backend"),
            "modality": policy_entry.get("modality"),
            "evaluation_protocol": policy_entry.get("evaluation_protocol"),
            "gripper_execution": policy_entry.get("gripper_execution"),
        },
    }


def assert_episode_contract(rec: dict, *, suite: dict) -> None:
    """Fail closed on demo selection, reset evidence, per-demo assets and OOD."""
    errs: list[str] = []

    def require(ok: bool, message: str) -> None:
        if not ok:
            errs.append(message)

    require(rec.get("schema") == _EPISODE_SCHEMA,
            f"episode receipt schema is not {_EPISODE_SCHEMA}")
    require(rec.get("static_contract_passed") is True, "task-static contract did not pass")
    require(bool((rec.get("release") or {}).get("commit")), "release commit is missing")
    hdf = rec.get("hdf5") or {}
    ep = hdf.get("episode_group") or {}
    attrs = ep.get("attrs") or {}
    state = ep.get("initial_state") or {}
    require(hdf.get("exists") is True, "HDF5 file is missing")
    require(ep.get("present") is True, f"HDF5 episode {rec.get('episode')} is missing")
    require(ep.get("episode") == rec.get("episode"), "HDF5 episode selection mismatch")
    require(state.get("present") is True, "initial_state group is missing")
    require(int(state.get("dataset_count", 0)) > 0, "initial_state contains no datasets")
    require(bool(state.get("sha256")), "initial_state hash is missing")
    if "task_id" in attrs:
        require(int(attrs["task_id"]) == int(rec["task_id"]),
                f"HDF5 task_id {attrs['task_id']} != {rec['task_id']}")
    require(attrs.get("task_suite") == suite.get("libero_suite"),
            f"HDF5 task_suite {attrs.get('task_suite')!r} != {suite.get('libero_suite')!r}")
    require(attrs.get("language") == rec.get("language"),
            f"HDF5 language {attrs.get('language')!r} != {rec.get('language')!r}")
    original_demo_id = attrs.get("original_demo_id", attrs.get("demo_id"))
    require(original_demo_id is not None, "HDF5 original_demo_id/demo_id is missing")
    scene_params = rec.get("scene_params") or {}
    if scene_params:
        require(scene_params.get("asset_name") in (None, (rec.get("target_asset") or {}).get("name")),
                "scene_params target asset does not match live target")
        if scene_params.get("collection_demo_id") is not None and original_demo_id is not None:
            require(int(scene_params["collection_demo_id"]) == int(original_demo_id),
                    f"scene_params collection_demo_id={scene_params['collection_demo_id']} "
                    f"!= HDF5 original_demo_id={original_demo_id}")

    target = rec.get("target_asset") or {}
    require(target.get("present") is True, f"episode target {target.get('name')} is absent")
    require(attrs.get("asset_name") == target.get("name"),
            f"HDF5 asset_name {attrs.get('asset_name')!r} != live target {target.get('name')!r}")
    require(_close(target.get("scale"), target.get("configured_scale")),
            f"episode target scale {target.get('scale')} != {target.get('configured_scale')}")
    extras = rec.get("extra_assets")
    require(isinstance(extras, dict), "episode extra asset evidence missing")
    for name, fact in (extras or {}).items():
        require(fact.get("present") is True, f"episode extra asset {name} absent")
        require(_close(fact.get("scale"), fact.get("configured_scale")),
                f"episode extra asset {name} scale mismatch")
        if not fact.get("rigid") and not fact.get("deformable"):
            require(_close(fact.get("position"), fact.get("configured_position")),
                    f"episode distractor {name} position mismatch")
            require(_quat_close(fact.get("orientation"), fact.get("configured_orientation")),
                    f"episode distractor {name} orientation mismatch")

    readback = rec.get("initial_state_readback") or {}
    require(not readback.get("missing"), f"initial_state live assets missing: {readback.get('missing')}")
    checks = readback.get("checks")
    require(isinstance(checks, dict) and bool(checks), "initial_state live readback is empty")
    for name, fact in (checks or {}).items():
        require("max_abs_error" in fact, f"initial_state {name} shape mismatch: {fact}")
        if "max_abs_error" in fact:
            tolerance = 2e-4 if (
                name.startswith("rigid_object/") or name.startswith("deformable_object/")
            ) else 1e-5
            require(float(fact["max_abs_error"]) <= tolerance,
                    f"initial_state {name} max error {fact['max_abs_error']} > {tolerance}")

    simulator = rec.get("simulator") or {}
    require(rec.get("gripper_constraint_mode") == "lower_limit_only",
            "formal gripper constraint mode must be lower_limit_only")
    require(simulator.get("first_render_had_collection_visuals") is True,
            "episode first render did not have collection visuals")
    policy = rec.get("policy") or {}
    gripper_execution = policy.get("gripper_execution") or {}
    gripper_mode = gripper_execution.get("mode")
    width = rec.get("grip_width_m")
    needs_calibration = (
        suite.get("name", "").endswith("_soft")
        or gripper_mode in RELATIVE_GRIPPER_MODES
    )
    if needs_calibration:
        require(
            width is not None,
            f"{gripper_mode or 'soft episode'} has no recorded gripper width",
        )
        limits = rec.get("gripper_joint_limits") or {}
        require("error" not in limits and len(limits.get("lower") or []) == 2,
                f"gripper joint limit readback failed: {limits}")
        if width is not None:
            tighten = float(gripper_execution.get("total_width_tighten_m", 0.0))
            require(0.0 <= tighten < float(width),
                    f"invalid total_width_tighten_m {tighten} for width {width}")
            if tighten:
                require(
                    policy.get("modality") == "vo"
                    and gripper_execution.get("mode") in FIXED_POSITION_GRIPPER_MODES,
                    "total_width_tighten_m is restricted to soft VO "
                    "fixed-position gripper execution",
                )
            expected_lower = (float(width) - tighten) / 2.0
            for lower in limits.get("lower") or []:
                require(abs(float(lower) - expected_lower) <= 1e-6,
                        f"gripper lower limit {lower} != expected {expected_lower} "
                        f"(recorded width={width}, total tighten={tighten})")

    condition_kind = rec.get("condition_kind")
    require(condition_kind in {"id", "ood"}, "condition_kind must be id or ood")
    ood = rec.get("ood")
    dome = rec.get("dome_light") or {}
    require(dome.get("present") is True and dome.get("active") is True,
            "episode collection DomeLight missing/inactive")
    if condition_kind == "id":
        require(ood is None or ood.get("enabled") is False,
                "ID episode unexpectedly enables OOD")
        require(dome.get("intensity") is not None, "ID DomeLight intensity missing")
        if dome.get("intensity") is not None:
            require(abs(float(dome["intensity"]) - 135.0) <= 1e-4,
                    f"ID DomeLight intensity {dome['intensity']} != 135")
    elif condition_kind == "ood":
        require(isinstance(ood, dict) and ood.get("enabled") is True,
                f"OOD condition {rec.get('condition')} has no enabled OOD receipt")
        if isinstance(ood, dict):
            require(ood.get("suite") == rec.get("suite"), "OOD suite mismatch")
            require(int(ood.get("task_id", -1)) == int(rec.get("task_id", -2)),
                    "OOD task_id mismatch")
            require(bool(ood.get("config_sha256")), "OOD config hash missing")
            factors = {x.get("factor") for x in (ood.get("factors") or [])}
            actual = ood.get("actual_parameters") or {}
            physical_factors = factors - {"gaussian_blur", "gaussian_noise"}
            require(physical_factors <= set(actual),
                    f"OOD applied parameters missing: factors={sorted(physical_factors)}, "
                    f"actual={sorted(actual)}")
            if "dome_light_intensity" in factors:
                fact = actual.get("dome_light_intensity") or {}
                require(fact.get("requested") is not None and fact.get("actual") is not None,
                        f"lighting OOD readback incomplete: {fact}")
                if (
                    fact.get("requested") is not None
                    and fact.get("actual") is not None
                    and dome.get("intensity") is not None
                ):
                    require(abs(float(fact["requested"]) - float(fact["actual"])) <= 1e-4,
                            f"lighting requested {fact['requested']} != actual {fact['actual']}")
                    require(abs(float(dome["intensity"]) - float(fact["actual"])) <= 1e-4,
                            f"live DomeLight {dome['intensity']} != OOD readback {fact['actual']}")

    rec["contract"] = {"passed": not errs, "errors": errs}
    if errs:
        raise RuntimeError("episode contract violation:\n- " + "\n- ".join(errs))


def assert_contract(rec: dict, *, suite: dict, obj: dict | None) -> None:
    """Fail closed: missing evidence is a violation, never a reason to skip a check."""
    errs: list[str] = []

    def require(ok: bool, message: str) -> None:
        if not ok:
            errs.append(message)

    require(rec.get("schema") == _SCHEMA, f"receipt schema is not {_SCHEMA}")
    require(not rec.get("missing_scene_prims"),
            f"missing scene prims: {rec.get('missing_scene_prims')}")
    require(not rec.get("unexpected_scene_prims"),
            f"unexpected scene prims: {rec.get('unexpected_scene_prims')}")
    require(not rec.get("removed_sources_present"),
            f"removed source still loaded: {rec.get('removed_sources_present')}")

    tiny = rec.get("tiny_prims")
    require(isinstance(tiny, dict), "tiny_prims evidence missing")
    for name, scale in (rec.get("expected_hidden_sources") or {}).items():
        require(name in (tiny or {}), f"hidden source {name} missing or not tiny")
        if name in (tiny or {}):
            require(_close(tiny[name], scale), f"hidden source {name} scale {tiny[name]} != {scale}")

    target = rec.get("target_asset")
    require(isinstance(target, dict), "target_asset evidence missing")
    require(bool((target or {}).get("name")), "target name unresolved")
    require((target or {}).get("present") is True, f"target absent: {(target or {}).get('name')}")
    want_scale = (target or {}).get("configured_scale")
    require(want_scale is not None, "target configured scale missing")
    require(_close((target or {}).get("scale"), want_scale),
            f"target scale {(target or {}).get('scale')} != {want_scale}")

    extras = rec.get("extra_assets")
    require(isinstance(extras, dict), "extra asset evidence missing")
    for name, fact in (extras or {}).items():
        require(fact.get("present") is True, f"extra asset {name} absent")
        require(_close(fact.get("scale"), fact.get("configured_scale")),
                f"extra asset {name} scale {fact.get('scale')} != {fact.get('configured_scale')}")
        if not fact.get("rigid") and not fact.get("deformable"):
            require(_close(fact.get("position"), fact.get("configured_position")),
                    f"visual distractor {name} position {fact.get('position')} "
                    f"!= {fact.get('configured_position')}")
            require(_quat_close(fact.get("orientation"), fact.get("configured_orientation")),
                    f"visual distractor {name} orientation {fact.get('orientation')} "
                    f"!= {fact.get('configured_orientation')}")

    cameras = rec.get("cameras_actual")
    require(isinstance(cameras, dict) and "error" not in cameras, f"camera readback failed: {cameras}")
    want_w, want_h = int(suite["camera"]["width"]), int(suite["camera"]["height"])
    for view in suite["camera"]["views"]:
        name = f"{view}_cam"
        require(name in (cameras or {}), f"required camera {name} missing")
        if name in (cameras or {}):
            got = cameras[name]
            require((got.get("width"), got.get("height")) == (want_w, want_h),
                    f"camera {name}: {got} != {want_w}x{want_h}")

    # The cleaned scene JSON carries BDDL metadata, but it is not the prompt fed
    # to the policy.  HDF5 is the formal collection truth for prompt text.
    # Keep scene_config_language in the receipt for audit without treating a
    # harmless BDDL paraphrase as an execution-contract failure.
    hdf5_language = rec.get("hdf5_language")
    require(bool(hdf5_language), "HDF5 language evidence missing")
    require(rec.get("language") == hdf5_language,
            f"language mismatch: suite={rec.get('language')!r}, hdf5={hdf5_language!r}")

    if suite.get("apply_scene_visuals", False):
        dome = rec.get("dome_light") or {}
        require(dome.get("present") is True and dome.get("active") is True, "collection dome missing/inactive")
        require(dome.get("intensity") is not None, "collection dome intensity missing")
        if dome.get("intensity") is not None:
            require(abs(float(dome["intensity"]) - 135.0) <= 1e-4,
                    f"collection dome intensity {dome['intensity']} != 135")
        lights = rec.get("active_lights")
        require(isinstance(lights, list), "active light inventory missing")
        require([x.get("path") for x in (lights or [])] == ["/World/textured_dim_dome"],
                f"unexpected active lights: {lights}")
        floor = rec.get("floor") or {}
        overlay = floor.get("overlay") or {}
        require(overlay.get("present") is True and overlay.get("active") is True,
                "textured floor overlay missing/inactive")
        z = floor.get("configured_overlay_z")
        require(z is not None and any(abs(float(v) - float(z)) <= 1e-5
                                      for v in overlay.get("mesh_z_values", [])),
                f"floor overlay z {overlay.get('mesh_z_values')} does not contain configured {z}")
        if floor.get("hide_physics_floor"):
            require((floor.get("physics_floor") or {}).get("visibility") == "invisible",
                    "physics Floor must be invisible for this suite")
        simulator = rec.get("simulator") or {}
        require(simulator.get("first_render_had_collection_visuals") is True,
                "first render occurred before collection visuals were applied")
        frames = rec.get("camera_frame_stats")
        require(isinstance(frames, dict) and "error" not in frames,
                f"camera frame evidence missing: {frames}")

    simulator = rec.get("simulator") or {}
    physics_config = rec.get("physics_config") or {}
    expected_simulator = physics_config.get("simulator") or {}
    for key in ("python", "isaac_sim", "isaac_lab"):
        expected = expected_simulator.get(key)
        actual = simulator.get(key)
        require(expected is not None, f"physics config simulator.{key} missing")
        require(actual is not None, f"runtime simulator.{key} missing")
        if expected is not None and actual is not None:
            require(str(actual) == str(expected),
                    f"runtime simulator.{key} {actual} != configured {expected}")
    for key in ("physics_hz", "control_hz"):
        expected = expected_simulator.get(key)
        actual = simulator.get(key)
        require(expected is not None, f"physics config simulator.{key} missing")
        require(actual is not None, f"runtime simulator.{key} missing")
        if expected is not None and actual is not None:
            require(abs(float(actual) - float(expected)) <= 1e-6,
                    f"runtime simulator.{key} {actual} != configured {expected}")
    expected_ccd = (physics_config.get("deformable_body") or {}).get("enable_ccd")
    require(expected_ccd is not None, "physics config deformable_body.enable_ccd missing")
    if expected_ccd is not None:
        require(simulator.get("enable_ccd") is bool(expected_ccd),
                f"runtime enable_ccd {simulator.get('enable_ccd')} != configured {expected_ccd}")

    exp = (obj or {}).get("expected") or {}
    if obj:
        usd_file = exp.get("usd_file")
        usd_hash = exp.get("usd_sha256")
        require(bool(usd_file), "object card expected usd_file missing")
        require(bool(usd_hash), "object card expected usd_sha256 missing")
        sources = (target or {}).get("composition_sources")
        require(isinstance(sources, list) and bool(sources), "target composition sources missing")
        if usd_file:
            expected_path = str(Path(usd_file).resolve())
            require(expected_path in (sources or []),
                    f"target does not compose expected USD {expected_path}; sources={sources}")
            fact = file_digest(expected_path)
            require(fact.get("sha256") == str(usd_hash)[:32],
                    f"target USD digest {fact.get('sha256')} != card {str(usd_hash)[:32]}")

    if obj and obj.get("deformable"):
        exp_nodes = exp.get("fem_nodes") or obj.get("fem_nodes")
        require(exp_nodes is not None, "object card expected fem_nodes missing")
        got_nodes = rec.get("fem_nodal_count")
        require(got_nodes is not None, "runtime FEM node count missing")
        if exp_nodes is not None and got_nodes is not None:
            require(int(exp_nodes) == int(got_nodes),
                    f"FEM nodes: actual {got_nodes} != expected {exp_nodes}")
        actual_phys = (target or {}).get("physics")
        require(isinstance(actual_phys, dict) and "error" not in actual_phys,
                f"target physics readback failed: {actual_phys}")
        for key in ("youngsModulus", "poissonsRatio", "density", "dynamicFriction"):
            want = exp.get(key)
            got = (actual_phys or {}).get(key)
            require(want is not None, f"object card expected {key} missing")
            require(got is not None, f"runtime target physics {key} missing")
            if want is not None and got is not None:
                require(abs(float(want) - float(got)) <= max(1e-4, abs(float(want)) * 1e-4),
                        f"{key}: actual {got} != expected {want}")
        body = (rec.get("physics_config") or {}).get("deformable_body") or {}
        mapping = {
            "simulation_hexahedral_resolution": "simulationHexahedralResolution",
            "solver_position_iteration_count": "solverPositionIterationCount",
            "vertex_velocity_damping": "vertexVelocityDamping",
            "contact_offset": "contactOffset",
            "rest_offset": "restOffset",
            "max_depenetration_velocity": "maxDepenetrationVelocity",
        }
        for cfg_key, runtime_key in mapping.items():
            want = body.get(cfg_key)
            got = (actual_phys or {}).get(runtime_key)
            require(want is not None, f"physics config {cfg_key} missing")
            require(got is not None, f"runtime deformable {runtime_key} missing")
            if want is not None and got is not None:
                require(abs(float(want) - float(got)) <= max(1e-5, abs(float(want)) * 1e-4),
                        f"{runtime_key}: actual {got} != configured {want}")
    elif obj and obj.get("rigid"):
        actual_phys = (target or {}).get("physics") or {}
        mapping = {
            "density": "density",
            "mass": "mass",
            "approximation": "approximation",
            "collisionEnabled": "collisionEnabled",
            "contactOffset": "contactOffset",
            "restOffset": "restOffset",
            "solver_iters": "solverPositionIterationCount",
        }
        for card_key, runtime_key in mapping.items():
            want = exp.get(card_key)
            got = actual_phys.get(runtime_key)
            require(want is not None, f"rigid card expected {card_key} missing")
            require(got is not None, f"runtime rigid {runtime_key} missing")
            if want is not None and got is not None:
                if isinstance(want, (int, float)) and not isinstance(want, bool):
                    require(abs(float(want) - float(got)) <= max(1e-5, abs(float(want)) * 1e-4),
                            f"{runtime_key}: actual {got} != expected {want}")
                else:
                    require(got == want, f"{runtime_key}: actual {got!r} != expected {want!r}")

    require(not rec.get("robot_friction_configured"),
            "robot_friction override set although collection did not apply it")
    rec["contract"] = {"passed": not errs, "errors": errs}
    if errs:
        raise RuntimeError("collection contract violation:\n- " + "\n- ".join(errs))


def scene_receipt(
    env,
    *,
    suite: dict,
    task: dict,
    paths: dict,
    policy_entry: dict,
    physics_config: dict | None = None,
) -> dict:
    """Read the live scene after reset_to and before the first policy inference."""
    del paths
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    env_root = "/World/envs/env_0"
    root = stage.GetPrimAtPath(env_root)
    prims = sorted(p.GetName() for p in root.GetChildren()) if root and root.IsValid() else []
    expected = _expected_scene(suite, task)

    target_name = expected["target_name"]
    target_path = f"{env_root}/{target_name}"
    target = _prim_fact(stage, target_path)
    target.update({
        "name": target_name,
        "configured_scale": expected["target_scale"],
        "physics": _read_physics(stage, target_path),
        "composition_sources": _prim_sources(stage, target_path),
    })

    extra_facts = {}
    for asset in expected["extra_assets"]:
        name = str(asset["name"])
        fact = _prim_fact(stage, f"{env_root}/{name}")
        fact["configured_position"] = _as_vec(asset.get("pos"))
        fact["configured_orientation"] = _as_vec(asset.get("rot"))
        fact["configured_scale"] = _as_vec(asset.get("scale"))
        fact["rigid"] = bool(asset.get("rigid"))
        fact["deformable"] = bool(asset.get("deformable"))
        extra_facts[name] = fact

    ckpt = policy_entry.get("checkpoint") or policy_entry.get("ckpt_dir") \
        or policy_entry.get("ckpt_path", "")
    norm = ""
    if policy_entry.get("checkpoint"):
        assets = Path(policy_entry["checkpoint"]) / "assets"
        cands = sorted(assets.glob("*/norm_stats.json")) if assets.exists() else []
        norm = str(cands[0]) if cands else ""

    removed_present = sorted(set(expected["removed_sources"]) & set(prims))
    return {
        "schema": _SCHEMA,
        "suite": suite["name"],
        "task_id": task["id"],
        "language": task.get("language"),
        "scene_config_language": expected["scene_config_language"],
        "hdf5_language": task.get("_receipt_hdf5_language"),
        "scene_prims": prims,
        "expected_scene_prims": expected["expected_prims"],
        "missing_scene_prims": sorted(set(expected["expected_prims"]) - set(prims)),
        "unexpected_scene_prims": sorted(set(prims) - set(expected["expected_prims"])),
        "removed_source_assets": expected["removed_sources"],
        "removed_sources_present": removed_present,
        "expected_hidden_sources": expected["hidden_sources"],
        "tiny_prims": _tiny_prims(stage, env_root),
        "extra_assets": extra_facts,
        "dome_light": _dome_light(stage),
        "active_lights": _active_lights(stage),
        "floor": _floor_fact(stage, suite),
        "cameras_actual": _cameras_actual(env),
        "camera_frame_stats": _camera_frame_stats(env),
        "camera_configured": suite.get("camera"),
        "simulator": _simulator_fact(env),
        "physics_config": physics_config,
        "scene_render_warmup_frames": suite.get("scene_render_warmup_frames"),
        "fem_nodal_count": (
            _fem_nodal_count(env, target_name)
            if obj_is_deformable(task, suite) else None
        ),
        "target_asset": target,
        "robot_finger_physics": _read_physics(stage, f"{env_root}/Robot/panda_leftfinger"),
        "robot_friction_configured": task.get("robot_friction") or suite.get("robot_friction"),
        "checkpoint": file_digest(ckpt) if ckpt else None,
        "norm_stats": file_digest(norm) if norm else None,
        "code_fingerprint": os.environ.get("SOFTVT_CODE_FINGERPRINT", ""),
        "release": {
            "commit": os.environ.get("SOFTVT_RELEASE_COMMIT", ""),
            "dirty": os.environ.get("SOFTVT_RELEASE_DIRTY", ""),
        },
    }


def write(path: str | Path, data: dict) -> str:
    """Atomically write a receipt and return its full SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n").encode()
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)
    return hashlib.sha256(payload).hexdigest()
