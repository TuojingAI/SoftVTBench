"""Vectorized rollout: num_envs=N, runs N episodes at once (same task, different demos).

Semantics aligned item-for-item with rollout.run_episode (reset sequence / step budget /
8-consecutive-step success criterion / OOD / FEM); the only differences:
  - reset_to injects N initial states; env.step takes (N, D) actions
  - one independent policy instance and GripperExecutor per env (temporal buffers/latches
    never cross-contaminate)
  - an env that finishes early is frozen (its last action is re-sent), no longer scored,
    and waits for the whole batch
Only backends without cross-request state are supported (openpi/replay); ACT/DP use
process-level parallelism.
"""
from __future__ import annotations

import os

import numpy as np

from softvtbench.evaluation import metrics
from softvtbench.evaluation.preprocessing.rotation import axis_angle_to_quat_wxyz
from softvtbench.evaluation.rollout import _apply_scene_visuals, _render, capture_env_obs


def _extend_visuals_to_all_envs(env, n: int) -> None:
    """The scene-visuals module only handles env_0 (collection used a single env); copy the floor overlay/hiding to the remaining envs.

    Without this fix, the white physical floor edge of env_1 shows up as a white stripe in
    every env's cameras (out of the training distribution).
    """
    if n <= 1:
        return
    import omni.usd
    from pxr import Sdf, UsdGeom
    stage = omni.usd.get_context().get_stage()
    src = "/World/envs/env_0/textured_floor_visual_overlay"
    for i in range(1, n):
        floor = stage.GetPrimAtPath(f"/World/envs/env_{i}/Floor")
        if floor and floor.IsValid():
            UsdGeom.Imageable(floor).MakeInvisible()
        if not stage.GetPrimAtPath(src).IsValid():
            continue                                  # suite has no overlay enabled (e.g. spatial_*)
        dst = f"/World/envs/env_{i}/textured_floor_visual_overlay"
        if not stage.GetPrimAtPath(dst).IsValid():
            Sdf.CopySpec(stage.GetRootLayer(), Sdf.Path(src), stage.GetRootLayer(), Sdf.Path(dst))
        if not stage.GetPrimAtPath(dst).IsValid():
            raise RuntimeError(f"floor overlay copy failed for env_{i}")


def _stack_states(states: list[dict]):
    import torch
    out = {}
    for k in states[0]:
        if isinstance(states[0][k], dict):
            out[k] = _stack_states([s[k] for s in states])
        else:
            out[k] = torch.cat([s[k] for s in states], dim=0)
    return out


def _states_to_world(stacked: dict, env_origins):
    """Convert batched relative scene state to world coordinates.

    Isaac Lab's ``InteractiveScene.reset_to(..., is_relative=True)`` handles
    articulation and rigid root poses correctly, but the deployed version adds
    ``env_origins`` to deformable nodes with incompatible ``[env,node,xyz]``
    broadcasting.  Convert every position explicitly and call reset_to with
    ``is_relative=False`` instead.
    """
    import torch

    origins = torch.as_tensor(env_origins)
    if origins.ndim != 2 or origins.shape[-1] != 3:
        raise ValueError(f"expected env origins [N,3], got {tuple(origins.shape)}")

    def clone(value):
        if isinstance(value, dict):
            return {key: clone(item) for key, item in value.items()}
        return value.clone() if isinstance(value, torch.Tensor) else value

    world = clone(stacked)
    for group in ("articulation", "rigid_object"):
        for asset in world.get(group, {}).values():
            asset["root_pose"][:, :3] += origins
    for asset in world.get("deformable_object", {}).values():
        asset["nodal_position"][..., :3] += origins[:, None, :]
    return world


def run_batch(env, success_term, policies: list, gripper_execs: list, initial_states: list,
              *, suite: dict, control: dict, oods: list | None = None, episode_seeds: list | None = None,
              fem_asset: str | None = None,
              finger_lower_limits: list[float | None] | None = None) -> list[dict]:
    """Run N episodes in parallel. Returns result dicts in input order."""
    import torch

    n = len(initial_states)
    assert len(policies) == len(gripper_execs) == n
    oods = oods or [None] * n
    finger_lower_limits = finger_lower_limits or [None] * n
    assert len(finger_lower_limits) == n
    env_ids = torch.arange(n, device=env.unwrapped.device)
    warmup = int(suite.get("scene_render_warmup_frames", 4))
    after_state = bool(suite.get("scene_visuals_after_recorded_state", False))
    post_restore = int(suite.get("scene_post_restore_warmup_frames", 0))

    stacked = _stack_states(initial_states)
    world_state = _states_to_world(stacked, env.unwrapped.scene.env_origins[env_ids])
    obs, _ = env.reset()
    if not after_state:
        _apply_scene_visuals(env)
        _extend_visuals_to_all_envs(env, n)
    obs, _ = env.reset_to(world_state, env_ids, is_relative=False)
    if after_state:
        _apply_scene_visuals(env)
        _extend_visuals_to_all_envs(env, n)
    _render(env, warmup)
    if post_restore > 0:
        obs, _ = env.reset_to(world_state, env_ids, is_relative=False)
        _render(env, post_restore)

    for i in range(n):
        policies[i].reset(episode_seed=(episode_seeds[i] if episode_seeds else None))
        gripper_execs[i].reset(finger_lower_limit=finger_lower_limits[i])
        if oods[i] is not None and oods[i].enabled:
            oods[i].apply_physical()

    fem_refs = [metrics.get_nodal_pos(env, fem_asset, idx=i) if fem_asset else None for i in range(n)]
    d_series: list[list[float]] = [[] for _ in range(n)]

    max_env_steps = int(control["max_inference_steps"]) * int(control["replan_steps"])
    num_success = int(control["num_success_steps"])
    with_tactile = policies[0].modality == "vt"
    tot = suite.get("tactile_output_type", "markers_rgb")

    def observe_all():
        for i in range(n):
            if done[i]:
                continue
            eo = capture_env_obs(env, obs, with_tactile=with_tactile,
                                 tactile_output_type=tot, idx=i)
            if oods[i] is not None and oods[i].enabled:
                eo["agentview_rgb"], eo["eye_in_hand_rgb"] = oods[i].corrupt_external_rgb(
                    eo["agentview_rgb"], eo["eye_in_hand_rgb"])
            policies[i].observe(eo)

    done = [False] * n
    success = [False] * n
    streak = [0] * n
    steps_at_done = [0] * n
    last_action = [None] * n
    steps = 0
    while steps < max_env_steps and not all(done):
        observe_all()
        chunks = [policies[i].predict() if not done[i] else None for i in range(n)]
        k = next(len(c) for c in chunks if c is not None)
        for j in range(k):
            if steps >= max_env_steps:
                break
            rows = []
            for i in range(n):
                if done[i] or chunks[i] is None:
                    rows.append(last_action[i])           # frozen: re-send last action
                    continue
                a = np.asarray(chunks[i][j], dtype=np.float32)
                grip = gripper_execs[i].step(a[6])
                row = np.concatenate([a[:3], axis_angle_to_quat_wxyz(a[3:6]), grip])
                rows.append(row)
                last_action[i] = row
            action = torch.from_numpy(np.stack(rows)).float().to(env.unwrapped.device)
            obs, _, term, trunc, _ = env.step(action)
            steps += 1
            flags = success_term.func(env, **success_term.params) if success_term is not None else None
            for i in range(n):
                if done[i]:
                    continue
                if fem_refs[i] is not None:
                    d = metrics.fem_deformation_pct(fem_refs[i], metrics.get_nodal_pos(env, fem_asset, idx=i))
                    if d is not None:
                        d_series[i].append(d)
                if bool(term[i]) or bool(trunc[i]):
                    done[i], steps_at_done[i] = True, steps
                    continue
                if flags is not None and bool(flags[i]):
                    streak[i] += 1
                    if streak[i] >= num_success:
                        success[i], done[i], steps_at_done[i] = True, True, steps
                else:
                    streak[i] = 0
            if all(done):
                break
            if j < k - 1:
                observe_all()

    results = []
    for i in range(n):
        results.append({"success": bool(success[i]),
                        "steps": steps_at_done[i] if done[i] else steps,
                        **metrics.deformation_stats(d_series[i])})
    return results
