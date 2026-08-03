"""Isaac environment construction -- App boots once, one env is built/destroyed per task.

Semantics transcribed from the benchmark repo's
benchmarks/common/closedloop_policy_inference.create_sim_environment (env-cfg surgery)
and the formal evaluation runner (scene/physics env vars). tac_manip's env cfg selects
the task via environment variables (TASK_SUITE/TASK_ID/LIBERO_CONFIG_DIR/
SOFTVTBENCH_EXTRA_*), so "per-task rebuild" = change env vars + gym.make. All parameter
values come from ../config/*.yaml; nothing is hard-coded.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_NAME = "Isaac-Libero-Franka-IK-Camera-Tactile-v0"   # unified formal evaluation env (binary control family)

_APP = None


def boot_app(headless: bool = True):
    """Boot the Isaac SimulationApp, once per process. Must be called before importing tac_manip."""
    global _APP
    if _APP is not None:
        return _APP
    os.environ.setdefault("USE_RELATIVE_MODE", "False")   # DiffIK absolute-pose control
    from isaaclab.app import AppLauncher
    experience = os.environ.get("SOFTVTBENCH_APP_EXPERIENCE", "").strip()
    kwargs = dict(headless=headless, enable_cameras=True, num_envs=1)
    launcher = AppLauncher(**kwargs, experience=experience) if experience else AppLauncher(**kwargs)
    _APP = launcher.app
    return _APP


def _export(env: dict[str, str]) -> None:
    for k, v in env.items():
        if v is None or v == "":
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)


def task_env_vars(paths: dict, suite: dict, task: dict, obj: dict | None, physics: dict) -> dict[str, str]:
    """Translate config values into the environment variables tac_manip reads, per the formal runner's convention."""
    e: dict[str, str] = {
        "TASK_SUITE": suite["libero_suite"],
        "TASK_ID": str(task["id"]),
        "LIBERO_CONFIG_DIR": suite["libero_config_dir"],
        "LIBERO_ASSETS_DATA_DIR": paths["libero_assets_data_dir"],
        # camera resolution aligned to collection (create_sim_environment's gated override)
        "SOFTVTBENCH_APPLY_COLLECTION_SCENE_VISUALS": "1" if suite.get("apply_scene_visuals", True) else "0",
        "SOFTVTBENCH_SCENE_VISUALS_MODULE": paths["scene_visuals_module"],
        "SOFTVT_CAMERA_WIDTH": str(suite["camera"]["width"]),
        "SOFTVT_CAMERA_HEIGHT": str(suite["camera"]["height"]),
        "SOFTVT_CAMERA_VIEWS": " ".join(suite["camera"]["views"]),
    }
    fric = task.get("robot_friction") or suite.get("robot_friction")
    if fric:  # collection used high-friction fingers throughout; pastry010 has an even higher special case (task-level override)
        e.update({
            "SOFTVTBENCH_ROBOT_HIGH_FRICTION": "1",
            "SOFTVTBENCH_ROBOT_STATIC_FRICTION": str(fric["static"]),
            "SOFTVTBENCH_ROBOT_DYNAMIC_FRICTION": str(fric["dynamic"]),
            "SOFTVTBENCH_ROBOT_FRICTION_COMBINE_MODE": str(fric["combine_mode"]),
        })
    # object-* collection covered the physics floor with a woodgrain floor (transcribed from the rigid runner)
    e["SOFTVTBENCH_FLOOR_OVERLAY_Z"] = str(suite["floor_overlay_z"]) if "floor_overlay_z" in suite else ""
    e["SOFTVTBENCH_HIDE_PHYSICS_FLOOR_VISUAL"] = "1" if suite.get("hide_physics_floor") else ""
    # extra-asset injection: rigid = staging per-task/ep json file; spatial_soft = scene_params inline distractors
    ea_file = task.get("extra_assets_file")
    if task.get("extra_assets_inline"):
        e["SOFTVTBENCH_EXTRA_ASSETS_JSON"] = task["extra_assets_inline"]
    elif ea_file:
        e["SOFTVTBENCH_EXTRA_ASSETS_JSON"] = Path(ea_file).read_text()
    else:
        e["SOFTVTBENCH_EXTRA_ASSETS_JSON"] = ""
    if obj is not None and obj.get("deformable"):
        # Soft suite only: inject the deformable target.  Rigid object cards
        # describe and audit the target already present in extra_assets; using
        # mere object-card presence here would create a duplicate soft target.
        usd_dir = Path(paths["eval_usd_dir"]) / task.get("usd_asset_override", obj["usd_asset"])
        usd_candidates = sorted(usd_dir.glob("*.usd")) + sorted(usd_dir.glob("*.usda"))
        if not usd_candidates:
            raise FileNotFoundError(f"no usd under {usd_dir}")
        body = physics["deformable_body"]
        e.update({
            "SOFTVTBENCH_EXTRA_ASSET_USD": str(usd_candidates[0]),
            "SOFTVTBENCH_EXTRA_ASSET_NAME": task["asset_name"],
            "SOFTVTBENCH_EXTRA_ASSET_SCALE": task.get("asset_scale", "1.0 1.0 1.0"),
            "SOFTVTBENCH_EXTRA_ASSET_DEFORMABLE": "1",
            "SOFTVTBENCH_EXTRA_ASSET_POS": task["asset_spawn_pos"],
            "SOFTVTBENCH_EXTRA_ASSET_ROT": task.get("asset_spawn_rot", "1 0 0 0"),
            "SOFTVTBENCH_EXTRA_HEX_RES": str(body["simulation_hexahedral_resolution"]),
            "SOFTVTBENCH_EXTRA_SOLVER_ITERS": str(body["solver_position_iteration_count"]),
            "SOFTVTBENCH_EXTRA_VERTEX_DAMPING": str(body["vertex_velocity_damping"]),
            "SOFTVTBENCH_EXTRA_CONTACT_OFFSET": str(body["contact_offset"]),
            "SOFTVTBENCH_EXTRA_REST_OFFSET": str(body["rest_offset"]),
            "SOFTVTBENCH_EXTRA_MAX_DEPENETRATION_VELOCITY": str(body["max_depenetration_velocity"]),
            "SOFTVTBENCH_SKIP_DEFORMABLE_CONTACT_SENSORS": "1",
            "SOFTVTBENCH_REMOVED_SOURCE_ASSETS": task.get("removed_source_assets") or "",
        })
    else:
        for k in ("SOFTVTBENCH_EXTRA_ASSET_USD", "SOFTVTBENCH_EXTRA_ASSET_NAME",
                  "SOFTVTBENCH_EXTRA_ASSET_DEFORMABLE", "SOFTVTBENCH_REMOVED_SOURCE_ASSETS"):
            e[k] = ""
        e["SOFTVTBENCH_REMOVED_SOURCE_ASSETS"] = task.get("removed_source_assets") or ""
    return e


def build_task_env(paths: dict, suite: dict, task: dict, obj: dict | None, physics: dict,
                   gripper_action_mode: str, seed: int = 11, device: str = "cuda",
                   num_envs: int = 1):
    """Build one task's environment. Returns (env, success_term). boot_app() must already have been called."""
    import random
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    _export(task_env_vars(paths, suite, task, obj, physics))
    # continuous execution (9D absolute finger positions) needs the env's gripper action switched to abs
    _export({"SOFTVTBENCH_GRIPPER_ACTION_MODE": "abs" if gripper_action_mode == "abs" else ""})

    env_cfg = parse_env_cfg(ENV_NAME, device=device, num_envs=num_envs)
    env_cfg.env_name = ENV_NAME
    env_cfg.seed = seed
    env_cfg.recorders = {}
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    env_cfg.terminations.time_out = None
    env_cfg.sim.physx.enable_ccd = True
    for attr in dir(env_cfg.terminations):  # prevent unexpected early termination
        if attr.endswith("_dropped"):
            setattr(env_cfg.terminations, attr, None)
    # camera resolution aligned to collection
    for view in suite["camera"]["views"]:
        sensor = getattr(env_cfg.scene, f"{view}_cam", None)
        if sensor is not None:
            sensor.width = int(suite["camera"]["width"])
            sensor.height = int(suite["camera"]["height"])

    env = gym.make(ENV_NAME, cfg=env_cfg).unwrapped
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env.seed(seed)
    return env, success_term


# OOD apply_physical() is "read current value x ratio" and records no original baseline
# (vendored implementation), while reset_to restores physics state but not USD materials
# -> compounding per-episode drift (measured 25000 -> 20000 -> 16000 -> 12800).
# Snapshot/restore guarantees each episode starts from authored values, and is also the
# precondition for running multiple OOD conditions on the same env.
_PHYS_SNAP_KEYS = ("youngsModulus", "poissonsRatio", "density", "mass",
                   "elasticityDamping", "dampingScale", "staticFriction", "dynamicFriction")


def settle_physics(env, steps: int = 60) -> None:
    """After building the env, idle the physics for a few steps to warm up the FEM/contact solvers.

    Without this the first episode is not the same physics as later ones: measured, the
    same demo run 4x gives d_peak=8.97/94 steps first, then stable at 3.95-3.98/109 steps
    (physics never steps after gym.make, so in the first episode the deformable is still
    settling from its spawn pose; _render renders without stepping physics).
    """
    sim = env.unwrapped.sim
    for _ in range(steps):
        sim.step(render=False)


def snapshot_target_physics(env, asset_name: str, idx: int = 0) -> list:
    """Snapshot the physics material attributes on the target asset subtree; returns [(attr, value), ...]."""
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(f"/World/envs/env_{idx}/{asset_name}")
    snap: list = []
    if not root or not root.IsValid():
        return snap
    stack = [root]
    while stack:
        prim = stack.pop()
        for attr in prim.GetAttributes():
            if attr.GetName().split(":")[-1] in _PHYS_SNAP_KEYS:
                v = attr.Get()
                if v is not None:
                    snap.append((attr, v))
        stack.extend(prim.GetChildren())
    return snap


def restore_physics(snapshot: list) -> None:
    for attr, value in snapshot:
        attr.Set(value)


def snapshot_dome_light() -> list:
    """Snapshot the collection DomeLight attributes changed by lighting OOD.

    This is intentionally captured only after collection_scene_visuals has run:
    before the first prepared episode the authored dome does not exist.
    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/textured_dim_dome")
    if not prim or not prim.IsValid() or not prim.IsActive():
        raise RuntimeError("cannot snapshot missing/inactive collection DomeLight")
    attr = prim.GetAttribute("inputs:intensity")
    if not attr or not attr.IsValid() or attr.Get() is None:
        raise RuntimeError("cannot snapshot collection DomeLight intensity")
    return [(attr, attr.Get())]


def pin_fingers_to_width(env, width_m: float, idx: int = 0) -> None:
    """Set both finger joint limits to width/2, forcing the fingers to that position (transcribed from the collection-side _pin_fingers_to_width).

    Clamping only the lower limit is not enough: soft-body elastic resistance stops the
    fingers above the limit (measured stop at 0.0175 instead of 0.01227), deformation only
    reaches 3.9% while the collection recording was 6.19%. Collection hard-pins the
    fingers after the close action and keeps them pinned through lift/transport.
    """
    robot = env.unwrapped.scene["robot"]
    ids, _ = robot.find_joints("panda_finger_joint.*")
    limits = robot.data.joint_pos_limits.clone()
    per_finger = float(width_m) / 2.0
    limits[idx, ids, 0] = per_finger
    limits[idx, ids, 1] = per_finger
    robot.write_joint_limits_to_sim(limits)


def unpin_fingers(env, width_m: float, open_finger: float = 0.04, idx: int = 0) -> None:
    """Release the hard pin: upper limit back to the open value, lower limit kept at width/2 (collection-time safe-width constraint)."""
    robot = env.unwrapped.scene["robot"]
    ids, _ = robot.find_joints("panda_finger_joint.*")
    limits = robot.data.joint_pos_limits.clone()
    limits[idx, ids, 0] = float(width_m) / 2.0
    limits[idx, ids, 1] = float(open_finger)
    robot.write_joint_limits_to_sim(limits)


def calibrated_finger_lower_limit(
    width_m: float, total_width_tighten_m: float = 0.0
) -> float:
    """Return the per-finger physical lower limit for one episode.

    ``width_m`` and ``total_width_tighten_m`` are both total jaw widths.  The
    latter is split evenly across the two fingers, so 0.0006 m means each
    finger closes an additional 0.0003 m.
    """
    width = float(width_m)
    tighten = float(total_width_tighten_m)
    if width <= 0.0:
        raise ValueError(f"gripper width must be > 0, got {width}")
    if not 0.0 <= tighten < width:
        raise ValueError(
            "total gripper-width tightening must lie in [0, width), "
            f"got tighten={tighten} width={width}"
        )
    return (width - tighten) / 2.0


def clamp_finger_lower_limit(
    env, width_m: float, idx: int = 0, *, total_width_tighten_m: float = 0.0
) -> float:
    """Replicates the collection-time soft_collection_expert._clamp_fingers_to_calibrated_width:

    each soft demo was collected with a safe gripper width sampled per
    (asset, original_demo_id), setting the panda_finger_joint.* lower limit to width/2
    (a binary close to 0 gets physically clamped). Evaluation reads the measured value
    from HDF5 soft_extras/gripper_width per episode (more faithful than recomputing the
    formula: 78 demos of stw_cuboid_hq used an older width band the formula does not match).
    """
    per_finger = calibrated_finger_lower_limit(width_m, total_width_tighten_m)
    robot = env.unwrapped.scene["robot"]
    ids, _ = robot.find_joints("panda_finger_joint.*")
    limits = robot.data.joint_pos_limits.clone()
    limits[idx, ids, 0] = per_finger
    robot.write_joint_limits_to_sim(limits)
    got = robot.data.joint_pos_limits[idx, ids, 0]
    if abs(float(got.min()) - per_finger) > 1e-6:   # verify on write, hard error on failure
        raise RuntimeError(
            f"finger limit clamp not applied: want {per_finger} got {got.tolist()}"
        )
    return per_finger


def episode_grip_width(h5file, episode_name: str) -> float | None:
    """Total jaw width (m) in effect when the demo was collected = min of soft_extras/gripper_width; None if the field is absent."""
    grp = h5file[f"data/{episode_name}"]
    if "soft_extras" not in grp or "gripper_width" not in grp["soft_extras"]:
        return None
    import numpy as np
    return float(np.asarray(grp["soft_extras/gripper_width"][()]).min())


def reposition_visual_assets(env, assets: list[dict], idx: int = 0) -> None:
    """Reposition purely visual assets (distractors) per episode. Physics bodies are restored by reset_to and are excluded here.

    Avoids per-episode env rebuilds (Isaac close crashes + 60s wasted each): visual prims
    get their USD xform edited directly.
    """
    from pxr import Gf
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    for a in assets:
        if a.get("rigid") or a.get("deformable"):
            continue                                  # physics bodies are handled by reset_to
        prim = stage.GetPrimAtPath(f"/World/envs/env_{idx}/{a['name']}")
        if not prim or not prim.IsValid():
            raise RuntimeError(f"visual asset prim missing: {a['name']}")
        t = prim.GetAttribute("xformOp:translate")
        t.Set(Gf.Vec3d(*[float(x) for x in a["pos"]]))
        rot = a.get("rot")
        if rot is not None:
            o = prim.GetAttribute("xformOp:orient")
            w, x, y, z = [float(v) for v in rot]
            quat = (Gf.Quatf if o.GetTypeName() == "quatf" else Gf.Quatd)(w, x, y, z)
            o.Set(quat)


def load_episodes(data_root: str, suite: dict, task_id: int):
    """Open the task's replayed_demos hdf5; returns (h5py.File, [episode_name, ... ordered]).

    Uses h5py directly (the new_data files lack the env_args attr the IsaacLab handler depends on).
    """
    import h5py

    task_dir = Path(data_root) / suite["data_subdir"].format(task_id=task_id) / "replayed_demos"
    files = sorted(task_dir.glob("*.hdf5"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected exactly one hdf5 under {task_dir}, got {files}")
    f = h5py.File(str(files[0]), "r")
    names = sorted(f["data"].keys(), key=lambda n: int(n.rsplit("_", 1)[-1]))
    return f, names


def load_initial_state(h5file, episode_name: str, removed_assets: str, device) -> dict:
    """Demo initial_state -> nested dict[torch tensor]; also transcribes the
    _remove_explicit_source_states filter: drops source-asset entries removed at
    collection time (any other unknown key still makes reset_to hard-error, keeping it auditable)."""
    import torch

    removed = {n for n in (removed_assets or "").replace(",", " ").split() if n}   # empty yaml value = None
    grp = h5file[f"data/{episode_name}/initial_state"]

    def to_dict(g):
        out = {}
        for k, v in g.items():
            if hasattr(v, "items"):
                sub = to_dict(v)
                if sub:                      # drop empty groups (e.g. deformable_object in rigid demos)
                    out[k] = sub
            else:
                out[k] = torch.as_tensor(v[()], dtype=torch.float32, device=device)
        return out

    state = to_dict(grp)
    for group_name in ("rigid_object", "articulation", "deformable_object"):
        g = state.get(group_name)
        if isinstance(g, dict):
            for name in list(g):
                if name in removed:
                    g.pop(name)
    # soft body: hdf5 stores nodal_pos_w (+bbox/root metadata); reset_to contract = nodal_position/velocity
    for name, entry in list((state.get("deformable_object") or {}).items()):
        if "nodal_pos_w" in entry:
            pos = entry["nodal_pos_w"]
            state["deformable_object"][name] = {
                "nodal_position": pos, "nodal_velocity": torch.zeros_like(pos)}
    return state
